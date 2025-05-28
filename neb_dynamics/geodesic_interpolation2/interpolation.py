"""
Performs path interpolation by iteratively adding or removing midpoint images.

This module provides functions to:
1. Find an optimal Cartesian midpoint between two given molecular geometries.
   This is achieved by optimizing the midpoint in a scaled internal coordinate
   space, aiming for it to be "geodesically" halfway between the endpoints.
   The core logic is encapsulated in the internal `_MidpointFinder` class.
2. Redistribute images along a reaction path to achieve a target number of
   images. This involves iteratively adding midpoints to the largest gaps or
   removing images from the densest regions of the path.
"""

import logging
import numpy as np
from scipy.optimize import least_squares
import scipy.sparse
from typing import List, Tuple, Callable, Union, Optional, Set

# Local imports from the same package
from .morsegeodesic import MorseGeodesic
from .coord_utils import get_bond_list, compute_wij, morse_scaler, align_geom, align_path
from .config import INTERPOLATION_DEFAULTS

logger = logging.getLogger(__name__)


class _MidpointFinder:
    """
    Internal helper class to find an optimal Cartesian midpoint between two geometries.

    The process involves:
    - Defining a set of internal coordinates based on the two endpoints and any
      intermediate trial midpoints.
    - Using a least-squares optimization to find a trial midpoint whose scaled
      internal coordinates are the average of the endpoints' scaled internals.
    - Optionally, evaluating this trial midpoint using a local 3-point geodesic
      calculation to refine the choice among multiple trials.
    - Iteratively refining the set of internal coordinates if new significant
      interactions are discovered by trial midpoints.
    """

    # --- Configuration Parameters from INTERPOLATION_DEFAULTS ---
    _MORSE_ALPHA_LS: float = INTERPOLATION_DEFAULTS["midpoint_morse_alpha"]
    _MORSE_BETA_LS: float = INTERPOLATION_DEFAULTS["midpoint_morse_beta"]
    _LS_FRICTION_FACTOR: float = INTERPOLATION_DEFAULTS["midpoint_ls_friction_factor"]
    _MORSE_ALPHA_REFINE: float = INTERPOLATION_DEFAULTS["midpoint_morse_alpha_refine"] # For local evaluation step
    _LOCAL_GEO_FRICTION: float = INTERPOLATION_DEFAULTS["midpoint_friction"]         # For local evaluation step
    _GBL_THRESHOLD_FACTOR_LS: float = INTERPOLATION_DEFAULTS["midpoint_gbl_threshold_factor"]
    _MIN_NEIGHBORS_LS: int = INTERPOLATION_DEFAULTS["midpoint_min_neighbors_ls_prep"]
    _MIN_NEIGHBORS_EXTRAS_CHECK: int = INTERPOLATION_DEFAULTS["midpoint_gbl_min_neighbors_extras"]
    _INITIAL_GUESS_COEFFS: List[float] = INTERPOLATION_DEFAULTS["midpoint_initial_guess_coeffs"]


    def __init__(self,
                 atoms: List[str],
                 geom1: np.ndarray,
                 geom2: np.ndarray,
                 tol: float,
                 nudge: float,
                 threshold: float,
                 initial_enforced_pairs: Optional[Set[Tuple[int, int]]] = None):
        """
        Initializes the _MidpointFinder.

        Args:
            atoms (List[str]): List of atom symbols.
            geom1 (np.ndarray): Coordinates of the first endpoint geometry (natoms, 3).
            geom2 (np.ndarray): Coordinates of the second endpoint geometry (natoms, 3).
            tol (float): Tolerance for the least-squares optimization.
            nudge (float): Magnitude of random nudge for initial guesses.
            threshold (float): Base distance threshold for `get_bond_list`.
            initial_enforced_pairs (Optional[Set[Tuple[int, int]]]): Atom pairs to always
                include in the internal coordinate list.
        """
        if geom1.shape != geom2.shape or geom1.ndim != 2 or geom1.shape[1] != 3:
             raise ValueError(
                f"Inconsistent endpoint geometry shapes. geom1: {geom1.shape}, geom2: {geom2.shape}."
            )

        self.atoms: List[str] = atoms
        self.geom1: np.ndarray = geom1.astype(float, copy=False) # Ensure float64
        self.geom2: np.ndarray = geom2.astype(float, copy=False) # Ensure float64
        self.num_atoms: int = self.geom1.shape[0]
        self.tol: float = tol
        self.nudge: float = nudge
        self.base_threshold: float = threshold # Base distance threshold for get_bond_list calls

        # Set of atom pairs that must be included in the internal coordinate list.
        self.enforced_pairs: Set[Tuple[int, int]] = \
            initial_enforced_pairs if initial_enforced_pairs is not None else set()

        # List of geometries used to define the internal coordinate list (rij_list)
        # for the main least-squares (LS) step. Initially contains the two endpoints.
        # Can be expanded if trial midpoints reveal new important interactions.
        self.geoms_for_ls_rij_definition: List[np.ndarray] = [self.geom1, self.geom2]

        # --- State variables for the least-squares optimization ---
        self.rij_list_for_ls_py: Optional[List[Tuple[int, int]]] = None # Python list of (i,j) tuples
        self.rij_list_for_ls_np: Optional[np.ndarray] = None           # NumPy array of (i,j) indices

        self.scaler_func_for_ls: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None
        self.w1_for_ls: Optional[np.ndarray] = None          # Scaled internals of geom1
        self.w2_for_ls: Optional[np.ndarray] = None          # Scaled internals of geom2
        self.avg_w_for_ls: Optional[np.ndarray] = None       # Target: (w1_for_ls + w2_for_ls) / 2.0

        self.last_ls_nfev: int = 0 # Number of function evaluations in the last LS optimization call

    def _prepare_for_least_squares_step(self) -> None:
        """
        Prepares for the main least-squares (LS) minimization step.

        This involves:
        1. Generating the list of internal coordinates (`rij_list`) using `get_bond_list`.
           The `rij_list` is based on the geometries currently in
           `self.geoms_for_ls_rij_definition`.
        2. Creating the `scaler_func` (e.g., Morse scaler) based on these internal
           coordinates and their estimated equilibrium distances.
        3. Calculating the scaled internal coordinates for `geom1` (`w1_for_ls`) and
           `geom2` (`w2_for_ls`), and their average (`avg_w_for_ls`), which serves
           as the target for the midpoint's scaled internals.
        """
        if not self.geoms_for_ls_rij_definition:
            raise ValueError("Geometry list for LS R_ij definition is empty. Cannot prepare for LS step.")

        # Use a slightly larger threshold for get_bond_list during LS preparation
        # to be more inclusive of potential interactions.
        threshold_for_ls_gbl = self.base_threshold + self._GBL_THRESHOLD_FACTOR_LS

        current_rij_list_py, re_values_for_ls = get_bond_list(
            geom=self.geoms_for_ls_rij_definition,
            atoms=None,  # atoms=None means re_values will be based on default radii or a flat default
            threshold=threshold_for_ls_gbl,
            min_neighbors=self._MIN_NEIGHBORS_LS,
            enforce=self.enforced_pairs
        )
        self.rij_list_for_ls_py = current_rij_list_py

        if self.rij_list_for_ls_py:
            self.rij_list_for_ls_np = np.array(self.rij_list_for_ls_py, dtype=np.int32)
        else:
            self.rij_list_for_ls_np = np.empty((0, 2), dtype=np.int32) # Empty array if no pairs

        # Create the scaler function for this specific LS step
        self.scaler_func_for_ls = morse_scaler(
            alpha=self._MORSE_ALPHA_LS,
            beta=self._MORSE_BETA_LS,
            eq_distances=re_values_for_ls
        )

        # Compute scaled internals for endpoints and their average
        if self.rij_list_for_ls_np is None or self.rij_list_for_ls_np.size == 0:
            # No internal coordinates defined for this LS step
            self.w1_for_ls = np.array([])
            self.w2_for_ls = np.array([])
            self.avg_w_for_ls = np.array([])
        else:
            self.w1_for_ls, _ = compute_wij(self.geom1, self.rij_list_for_ls_np, self.scaler_func_for_ls)
            self.w2_for_ls, _ = compute_wij(self.geom2, self.rij_list_for_ls_np, self.scaler_func_for_ls)
            self.avg_w_for_ls = (self.w1_for_ls + self.w2_for_ls) / 2.0

    def _least_squares_minimize(self, initial_guess_flat: np.ndarray) -> np.ndarray:
        """
        Performs a least-squares minimization to find a midpoint geometry.

        The objective is to find Cartesian coordinates `X_flat` (for the midpoint)
        such that its scaled internal coordinates `wx_curr` are close to
        `self.avg_w_for_ls`. A friction term is added to regularize the optimization
        and keep the solution near the `initial_guess_flat`.

        Args:
            initial_guess_flat (np.ndarray): A flattened 1D array of Cartesian
                coordinates for the initial guess of the midpoint.

        Returns:
            np.ndarray: The optimized Cartesian coordinates (flattened 1D array)
                for the midpoint.
        """
        if self.rij_list_for_ls_np is None or \
           self.scaler_func_for_ls is None or self.avg_w_for_ls is None:
             raise RuntimeError("LS-specific definitions (rij_list, scaler, avg_w) are not ready for LS optimization.")

        # Friction coefficient, scaled by the square root of the number of atoms
        friction_coeff = self._LS_FRICTION_FACTOR / np.sqrt(self.num_atoms if self.num_atoms > 0 else 1)
        initial_guess_flat_f64 = initial_guess_flat.astype(float, copy=False) # Ensure float64

        def _residuals(X_flat: np.ndarray) -> np.ndarray:
            """
            Residuals function for `scipy.optimize.least_squares`.
            The residuals are composed of two parts:
            1. `delta_w`: Difference between the current midpoint's scaled internals (`wx_curr`)
                          and the target average (`self.avg_w_for_ls`).
            2. `friction_res`: A term penalizing deviation from the `initial_guess_flat`.
            """
            current_geom = X_flat.reshape(self.num_atoms, 3)
            active_rij_list_np = self.rij_list_for_ls_np # Use the rij_list defined for this LS step

            if active_rij_list_np is None or active_rij_list_np.size == 0:
                 # If no internal coordinates, residuals are only from the friction term
                 return (X_flat - initial_guess_flat_f64) * friction_coeff

            # Calculate scaled internals for the current trial midpoint geometry
            wx_curr, _ = compute_wij(current_geom, active_rij_list_np, self.scaler_func_for_ls) # type: ignore
            delta_w = wx_curr - self.avg_w_for_ls # type: ignore # Difference from target
            friction_res = (X_flat - initial_guess_flat_f64) * friction_coeff # Friction term

            return np.concatenate([delta_w, friction_res])

        def _jacobian(X_flat: np.ndarray) -> scipy.sparse.csc_matrix:
            """
            Jacobian of the `_residuals` function with respect to `X_flat`.
            This is also composed of two parts:
            1. Jacobian of `delta_w` (which is `d(wx_curr)/dX_flat`, obtained from `compute_wij`).
            2. Jacobian of `friction_res` (which is `friction_coeff * Identity_matrix`).
            """
            current_geom = X_flat.reshape(self.num_atoms, 3)
            num_cart_coords = initial_guess_flat_f64.size
            active_rij_list_np = self.rij_list_for_ls_np

            if active_rij_list_np is None or active_rij_list_np.size == 0:
                # Jacobian of the friction term only
                return scipy.sparse.identity(num_cart_coords, dtype=float, format='csc') * friction_coeff \
                       if num_cart_coords > 0 else scipy.sparse.csc_matrix((0,0), dtype=float)

            # Get Jacobian d(wx_curr)/dX_flat from compute_wij
            _, dwdR_flat_sparse = compute_wij(current_geom, active_rij_list_np, self.scaler_func_for_ls) # type: ignore

            dwdR_flat_sparse_csc = dwdR_flat_sparse.tocsc() if \
                not isinstance(dwdR_flat_sparse, scipy.sparse.csc_matrix) else dwdR_flat_sparse

            # Jacobian of the friction term (a scaled identity matrix)
            sparse_friction_jac_part = scipy.sparse.csc_matrix((0,0), dtype=float)
            if num_cart_coords > 0:
                sparse_friction_jac_part = scipy.sparse.identity(
                    num_cart_coords, dtype=float, format='csc') * friction_coeff

            if dwdR_flat_sparse_csc.shape[1] != sparse_friction_jac_part.shape[1] and num_cart_coords > 0:
                 raise ValueError(
                     f"Jacobian column count mismatch: d(w)/dR has {dwdR_flat_sparse_csc.shape[1]} cols, "
                     f"friction Jacobian has {sparse_friction_jac_part.shape[1]} cols."
                 )

            # Stack the two parts of the Jacobian vertically
            if dwdR_flat_sparse_csc.shape[0] > 0 and sparse_friction_jac_part.shape[0] > 0:
                return scipy.sparse.vstack([dwdR_flat_sparse_csc, sparse_friction_jac_part], format='csc')
            elif dwdR_flat_sparse_csc.shape[0] > 0 :
                return dwdR_flat_sparse_csc
            elif sparse_friction_jac_part.shape[0] > 0:
                return sparse_friction_jac_part
            else: # Should not happen if num_cart_coords > 0
                return scipy.sparse.csc_matrix((0, num_cart_coords), dtype=float)

        logger.debug('Starting LS minimization for midpoint (alpha_LS: %s, re=default).', self._MORSE_ALPHA_LS)

        # Provide Jacobian sparsity pattern if available (can speed up optimization)
        jac_at_x0 = _jacobian(initial_guess_flat_f64)
        sparsity_pattern = jac_at_x0 if jac_at_x0.size > 0 and jac_at_x0.nnz > 0 else None

        result = least_squares(
            fun=_residuals,
            x0=initial_guess_flat_f64,
            jac=_jacobian,
            method='trf',  # Trust Region Reflective method
            loss='linear', # Standard least squares loss
            ftol=self.tol,
            gtol=self.tol,
            jac_sparsity=sparsity_pattern
        )
        self.last_ls_nfev = result.nfev if hasattr(result, 'nfev') else 0
        return result.x  # Optimized Cartesian coordinates (flattened)

    def _evaluate_midpoint_candidate_locally(self, trial_midpoint: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluates a `trial_midpoint` by constructing a temporary 3-point path:
        `[self.geom1, trial_midpoint, self.geom2]`.

        It then uses a `MorseGeodesic` object to calculate properties of this
        3-point path, such as its length in the scaled internal coordinate space.
        The `trial_midpoint` itself is NOT modified by this function; this function
        serves to score the quality of the `trial_midpoint`.

        Args:
            trial_midpoint (np.ndarray): The Cartesian coordinates (natoms, 3) of
                the midpoint candidate to be evaluated.

        Returns:
            Tuple[np.ndarray, float]:
                - `evaluated_midpoint_coords` (np.ndarray): The coordinates of the
                  `trial_midpoint` after being part of the 3-point path and potentially
                  realigned by the internal `MorseGeodesic` alignment. This is
                  effectively `smoother.path[1]`.
                - `combined_metric` (float): A metric combining the path length of
                  the 3-point path and the maximum RMSD from the evaluated midpoint
                  to the endpoints. This metric is used to select the best candidate.
        """
        path3 = np.array([self.geom1, trial_midpoint.astype(float, copy=False), self.geom2])
        logger.debug(
            f"Local geodesic evaluation of midpoint candidate (alpha_refine: {self._MORSE_ALPHA_REFINE}, re from atoms)"
        )

        # Initialize a MorseGeodesic object for the temporary 3-point path
        smoother = MorseGeodesic(
            atoms=self.atoms,
            path=path3,
            scaler=self._MORSE_ALPHA_REFINE, # Use specific Morse alpha for this evaluation
            threshold=self.base_threshold,
            log_level=logging.DEBUG, # Use DEBUG for this internal smoother instance
            friction=self._LOCAL_GEO_FRICTION # Use specific friction for this evaluation
        )

        # Call `_compute_disps` to ensure the smoother's internal state,
        # particularly its path length (`smoother.length`), is calculated for the 3-point path.
        # No actual smoothing optimization is performed here.
        smoother._compute_disps()

        # `smoother.path[1]` contains the coordinates of the trial_midpoint after it has been
        # aligned as part of the 3-point path by the MorseGeodesic constructor.
        evaluated_midpoint_coords = smoother.path[1]

        # Calculate RMSD from the (potentially realigned) midpoint to the original endpoints
        rmsd1 = np.sqrt(np.mean(np.sum((self.geom1 - evaluated_midpoint_coords)**2, axis=1)))
        rmsd2 = np.sqrt(np.mean(np.sum((self.geom2 - evaluated_midpoint_coords)**2, axis=1)))
        width = max(rmsd1, rmsd2) # Maximum RMSD from midpoint to endpoints

        # Combine width and path length into a single metric for scoring
        combined_metric = width + smoother.length

        return evaluated_midpoint_coords, combined_metric

    def find(self) -> np.ndarray:
        """
        Finds the optimal midpoint geometry.

        This is an iterative process:
        1. Prepares for a least-squares (LS) optimization step by defining internal
           coordinates and target values (`_prepare_for_least_squares_step`).
        2. Iterates through different initial guesses for the midpoint:
            a. Performs an LS optimization (`_least_squares_minimize`) to get an
               `unrefined_midpoint_geom`.
            b. Checks if this `unrefined_midpoint_geom`, when considered alongside
               the endpoints, reveals any new significant atom pairs (potential internal
               coordinates) that were not in the current `rij_list_for_ls_py`.
            c. If new pairs are found:
                - The `unrefined_midpoint_geom` is added to `geoms_for_ls_rij_definition`.
                - The new pairs are added to `enforced_pairs`.
                - The LS preparation (`_prepare_for_least_squares_step`) is re-run with
                  the updated internal coordinate definition.
                - The process restarts from step 2 (trying initial guesses again).
            d. If no new pairs are found:
                - The `unrefined_midpoint_geom` is evaluated locally using
                  `_evaluate_midpoint_candidate_locally`.
                - The candidate that yields the best (lowest) evaluation metric is tracked.
        3. Once the loop over initial guesses completes without finding new pairs,
           the best candidate from the local evaluations is returned as the final midpoint.

        Returns:
            np.ndarray: The Cartesian coordinates (natoms, 3) of the found optimal midpoint.
        """
        self._prepare_for_least_squares_step() # Initial preparation

        final_midpoint_candidate: Optional[np.ndarray] = None

        # This outer loop allows the entire process to restart if new significant
        # internal coordinates are discovered by a trial midpoint.
        while True:
            best_evaluated_midpoint_this_iteration: Optional[np.ndarray] = None
            min_metric_this_iteration: float = np.inf
            extras_found_and_restarted_outer_loop = False

            # Try different starting coefficients for the initial guess
            for start_coef in self._INITIAL_GUESS_COEFFS:
                # Generate initial guess as a linear combination of endpoints, plus a random nudge
                initial_guess_cartesian = (self.geom1 * start_coef + (1.0 - start_coef) * self.geom2)
                initial_guess_nudged_flat = (
                    initial_guess_cartesian +
                    self.nudge * np.random.random_sample(initial_guess_cartesian.shape)
                ).ravel()

                # Perform least-squares minimization to get an unrefined midpoint
                unrefined_midpoint_flat = self._least_squares_minimize(initial_guess_nudged_flat)
                unrefined_midpoint_geom = unrefined_midpoint_flat.reshape(self.num_atoms, 3)

                # --- Check for "Extra" Internal Coordinates ---
                # See if this unrefined midpoint suggests new important atom pairs
                # that should be included in the definition of internal coordinates.
                geoms_for_extras_check = self.geoms_for_ls_rij_definition + [unrefined_midpoint_geom]
                rij_list_from_extras_check_py, _ = get_bond_list(
                    geom=geoms_for_extras_check,
                    atoms=None, # Default re estimation
                    threshold=self.base_threshold, # Use base threshold for this check
                    min_neighbors=self._MIN_NEIGHBORS_EXTRAS_CHECK,
                    enforce=self.enforced_pairs
                )

                current_ls_rij_set = set(self.rij_list_for_ls_py if self.rij_list_for_ls_py is not None else [])
                newly_found_pairs = set(rij_list_from_extras_check_py) - current_ls_rij_set

                if newly_found_pairs:
                    logger.info(
                        f'_MidpointFinder: {len(newly_found_pairs)} new atom pairs found based on trial midpoint. '
                        f'Restarting LS preparation with updated internal coordinate list.'
                    )
                    # Add the geometry that revealed these new pairs to the list used for defining rij_list
                    self.geoms_for_ls_rij_definition.append(unrefined_midpoint_geom)
                    # Enforce these newly found pairs in subsequent get_bond_list calls
                    self.enforced_pairs.update(newly_found_pairs)
                    # Re-run the preparation for the LS step with the new internal coordinate definition
                    self._prepare_for_least_squares_step()
                    extras_found_and_restarted_outer_loop = True
                    break # Break from the `for start_coef` loop to restart the `while True` loop

                # --- Evaluate the Unrefined Midpoint (if no new pairs found) ---
                evaluated_candidate, metric = self._evaluate_midpoint_candidate_locally(unrefined_midpoint_geom)
                logger.debug(
                    f'_MidpointFinder trial (start_coef={start_coef:.2f}): Metric={metric:8.3f} '
                    f'(LS func_evals: {self.last_ls_nfev}).'
                )
                if metric < min_metric_this_iteration:
                    min_metric_this_iteration = metric
                    best_evaluated_midpoint_this_iteration = evaluated_candidate

            if not extras_found_and_restarted_outer_loop:
                # If the inner loop (over start_coef) completed without finding new pairs,
                # then the current `best_evaluated_midpoint_this_iteration` is our final candidate.
                final_midpoint_candidate = best_evaluated_midpoint_this_iteration
                break # Exit the `while True` loop

        # Final check for a valid candidate
        if final_midpoint_candidate is None:
            if best_evaluated_midpoint_this_iteration is not None: # Should be set if loop ran
                 logger.warning(
                     "_MidpointFinder: `final_midpoint_candidate` was None after loop, "
                     "but `best_evaluated_midpoint_this_iteration` is available. Using it as fallback."
                 )
                 final_midpoint_candidate = best_evaluated_midpoint_this_iteration
            else: # This case should ideally not be reached if the logic is sound
                 raise RuntimeError("Midpoint finding loop failed to produce any candidate geometry.")

        return final_midpoint_candidate


def mid_point(atoms: List[str],
              geom1: np.ndarray,
              geom2: np.ndarray,
              tol: float,
              nudge: float = INTERPOLATION_DEFAULTS["midpoint_nudge"],
              threshold: float = INTERPOLATION_DEFAULTS["midpoint_threshold"]) -> np.ndarray:
    """
    Finds an optimal Cartesian midpoint geometry between two given geometries (`geom1`, `geom2`).

    This function serves as a public interface to the `_MidpointFinder` class,
    which encapsulates the detailed logic for midpoint determination.

    Args:
        atoms (List[str]): List of atom symbols for the molecular system.
        geom1 (np.ndarray): Cartesian coordinates of the first endpoint geometry,
                            shape (natoms, 3).
        geom2 (np.ndarray): Cartesian coordinates of the second endpoint geometry,
                            shape (natoms, 3).
        tol (float): Tolerance criterion for the underlying optimization processes.
        nudge (float, optional): Magnitude of random displacement applied to initial
                                 guesses in the midpoint search. Defaults to a value
                                 from `INTERPOLATION_DEFAULTS`.
        threshold (float, optional): Base distance threshold (Angstroms) for
                                     `get_bond_list` calls within the midpoint search.
                                     Defaults to a value from `INTERPOLATION_DEFAULTS`.

    Returns:
        np.ndarray: The Cartesian coordinates (natoms, 3) of the found optimal midpoint.

    Raises:
        ValueError: If endpoint shapes are incompatible or if the atom count
                    mismatches the geometry dimensions.
    """
    if geom1.shape != geom2.shape or geom1.ndim != 2 or geom1.shape[1] != 3:
        raise ValueError(f"Incompatible endpoint shapes: geom1 {geom1.shape}, geom2 {geom2.shape}.")
    if len(atoms) != geom1.shape[0]:
        raise ValueError(f"Atom count mismatch: `atoms` list has {len(atoms)} elements, "
                         f"but geom1 has {geom1.shape[0]} atoms.")

    # Ensure input geometries are float64 for internal calculations
    geom1_f64 = geom1.astype(float, copy=False)
    geom2_f64 = geom2.astype(float, copy=False)

    # Create and use the _MidpointFinder
    finder = _MidpointFinder(atoms, geom1_f64, geom2_f64, tol, nudge, threshold)
    return finder.find()


def redistribute(atoms: List[str],
                 geoms: Union[List[np.ndarray], np.ndarray],
                 nimages: int,
                 tol: float) -> List[np.ndarray]:
    """
    Redistributes images (geometries) along a reaction path to achieve a
    target number of images (`nimages`).

    The process involves:
    - If `len(geoms) < nimages`: Iteratively finds the largest gap (by RMSD
      between adjacent images) and inserts a new midpoint (calculated by `mid_point`)
      into that gap. The path is realigned after each insertion.
    - If `len(geoms) > nimages`: Iteratively identifies an image whose removal
      results in the smallest RMSD between its former neighbors (i.e., the "smoothest"
      contraction) and removes it. The path is realigned after each removal.

    Args:
        atoms (List[str]): List of atom symbols for the molecular system.
        geoms (Union[List[np.ndarray], np.ndarray]): The initial path, provided as
            a list of 2D NumPy arrays (each natoms, 3) or a single 3D NumPy
            array (nframes, natoms, 3).
        nimages (int): The target number of images for the redistributed path.
                       Must be >= 2.
        tol (float): Tolerance passed to `mid_point` for calculating new midpoints
                     during image addition.

    Returns:
        List[np.ndarray]: A list of 2D NumPy arrays, where each array represents
                          a geometry in the redistributed path. The list will
                          contain `nimages` geometries.

    Raises:
        ValueError: If input `geoms` shape is invalid, atom count mismatches,
                    or `nimages` < 2.
    """
    current_geoms_np: np.ndarray = np.array(geoms, dtype=float) # Ensure NumPy array of floats

    # --- Input Validation ---
    if current_geoms_np.size == 0 : # Empty input path
        if nimages == 0:
            return []
        else:
            # Cannot create non-zero images from an empty path.
            # Using .throw on a generator expression is a way to raise an exception
            # that might be more informative in some contexts, but a direct raise is also fine.
            return (_ for _ in ()).throw(ValueError("Cannot redistribute an empty input path to a non-zero number of images."))

    if current_geoms_np.ndim == 2 : # Single geometry provided, treat as a path of length 1
        current_geoms_np = np.expand_dims(current_geoms_np, axis=0)

    if current_geoms_np.ndim != 3 or current_geoms_np.shape[2] != 3: # Must be (Nimg, Natom, 3)
         raise ValueError(f"Invalid input geoms shape: {current_geoms_np.shape}. Expected (Nimg, Natom, 3).")
    if len(atoms) != current_geoms_np.shape[1]: # Atom list must match geometry
         raise ValueError(f"Atom count mismatch: `atoms` list ({len(atoms)}) vs geoms ({current_geoms_np.shape[1]} atoms).")
    if nimages < 2: # Target path must have at least a start and end point
        raise ValueError(f"Target number of images (`nimages`) must be >= 2, got {nimages}.")

    # --- Initial Path Alignment ---
    _, aligned_geoms_np = align_path(current_geoms_np)
    # Work with a list of float64 geometries internally
    geoms_list: List[np.ndarray] = [g.astype(float, copy=False) for g in list(aligned_geoms_np)]

    # --- Add Images if Current Number is Less Than Target ---
    while len(geoms_list) < nimages:
        num_current = len(geoms_list)

        if num_current < 2: # Path is too short to define a gap for insertion
            if num_current == 1 and nimages > 1:
                # If only one image exists and more are needed, duplicate it to create a segment.
                geoms_list.append(geoms_list[0].copy().astype(float, copy=False))
            else: # Cannot add more if path has 0 or 1 image and target is also small.
                break
            if len(geoms_list) == nimages: # Target reached after duplication.
                break

        geoms_arr_add = np.array(geoms_list)
        # Calculate Cartesian differences between adjacent images
        diffs = geoms_arr_add[:-1] - geoms_arr_add[1:]
        # Calculate RMSD between adjacent images to find the largest gap
        rmsd_dists = np.sqrt(np.mean(diffs**2, axis=(1, 2)))

        if rmsd_dists.size == 0: # Should not happen if num_current >= 2
            logger.error("Redistribute (Add Images): No RMSD distances found between segments. Breaking loop.")
            break

        insert_idx = np.argmax(rmsd_dists) # Index of the first image in the pair with the largest RMSD
        logger.info(
            f"Inserting new image between original indices {insert_idx} and {insert_idx+1} "
            f"(segment RMSD: {rmsd_dists[insert_idx]:.3f}). Path length will be {len(geoms_list)+1}."
        )
        g1 = geoms_list[insert_idx]
        g2 = geoms_list[insert_idx+1]

        # Calculate the new midpoint to insert
        insertion_geom = mid_point(atoms, g1, g2, tol=tol)
        # Align the new midpoint with respect to its preceding image (g1)
        _, aligned_insertion = align_geom(g1, insertion_geom.astype(float, copy=False))
        geoms_list.insert(insert_idx + 1, aligned_insertion.astype(float, copy=False))

        # Re-align the entire path after insertion
        _, re_aligned_np = align_path(np.array(geoms_list))
        geoms_list = [g.astype(float, copy=False) for g in list(re_aligned_np)]

    # --- Remove Images if Current Number is Greater Than Target ---
    while len(geoms_list) > nimages:
        num_current = len(geoms_list)

        if num_current <= 2: # Path is too short to remove images (must keep endpoints)
            logger.warning(f"Redistribute (Remove Images): Path length ({num_current}) is too short to remove more images. Breaking loop.")
            break

        geoms_arr_rem = np.array(geoms_list)
        # Calculate RMSD if an image `i+1` were removed (i.e., RMSD between image `i` and image `i+2`)
        # This helps find the image whose removal causes the least disruption.
        diffs_merged = geoms_arr_rem[:-2] - geoms_arr_rem[2:]
        rmsd_merged = np.sqrt(np.mean(diffs_merged**2, axis=(1,2)))

        if rmsd_merged.size == 0: # Should not happen if num_current > 2
            logger.warning("Redistribute (Remove Images): No merged sections found to evaluate for removal. Breaking loop.")
            break

        min_merged_idx = np.argmin(rmsd_merged) # Index of image `i` in the triplet (i, i+1, i+2)
                                                # whose removal (of `i+1`) results in the smallest merged RMSD.
        remove_idx = min_merged_idx + 1         # Index of the image to actually remove (image `i+1`)

        logger.info(
            f"Removing image at index {remove_idx} "
            f"(RMSD of merged segment [{min_merged_idx}...{min_merged_idx+2} without {remove_idx}]: {rmsd_merged[min_merged_idx]:.3f}). "
            f"Path length will be {len(geoms_list)-1}."
        )
        del geoms_list[remove_idx]

        # Re-align the entire path after removal
        _, re_aligned_np = align_path(np.array(geoms_list))
        geoms_list = [g.astype(float, copy=False) for g in list(re_aligned_np)]

    return geoms_list

