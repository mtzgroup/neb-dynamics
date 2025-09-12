"""
morsegeodesic.py: Provides the MorseGeodesic class for smoothing molecular reaction paths.

This module is central to the Morse-Geodesic interpolation package. It implements
the optimization of a reaction path by minimizing its length in a metric space
defined by scaled redundant internal coordinates. The scaling typically uses a
Morse-like potential function. The optimization itself operates on the Cartesian
coordinates of the atoms in the path images. This approach aims to combine the
chemical relevance of internal coordinates with the robustness of Cartesian optimization.
"""

import logging
import numpy as np
from scipy.optimize import least_squares  # For path optimization
import scipy.sparse  # For efficient handling of Jacobian matrices
from typing import List, Tuple, Callable, Union, Optional, Any

# Local imports from the same package
from .coord_utils import align_path, get_bond_list, morse_scaler, compute_wij
from .config import MAIN_DEFAULTS, COORD_UTILS_DEFAULTS

logger = logging.getLogger(__name__)

# A small epsilon value to consider a COO matrix element non-zero during Jacobian construction.
# Helps in maintaining sparsity by filtering out numerically insignificant values.
COO_NON_ZERO_EPSILON = 1e-8


class MorseGeodesic(object):
    """
    Optimizes a molecular reaction path using the Morse-Geodesic smoothing method.

    The core idea is to define a "length" for the path in a space of scaled
    internal coordinates (typically interatomic distances scaled by a Morse-like
    function). The optimization then adjusts the Cartesian coordinates of the
    intermediate images (frames) in the path to minimize this path length.
    The endpoints of the path are typically kept fixed during optimization.

    Attributes:
        path (np.ndarray): The current set of Cartesian coordinates for the path images,
                           shape (nimages, natoms, 3). This is updated during optimization.
        nimages (int): Number of images (frames) in the path.
        natoms (int): Number of atoms in each image.
        atoms (List[str]): List of atom symbols.
        num_cart_coords (int): Total number of Cartesian coordinates per image (natoms * 3).
        rij_list_np (np.ndarray): NumPy array of atom pair indices defining the
                                  internal coordinates, shape (nrij, 2).
        eq_distances (Optional[np.ndarray]): Equilibrium distances for the pairs in `rij_list_np`.
        nrij (int): Number of internal coordinates (atom pairs).
        scaler_func (Callable): The function used to scale raw interatomic distances
                                to `wij` values and calculate `dw_drij`.
        base_friction (float): Default friction coefficient for regularization.
        length (float): The current calculated length of the path in scaled internal space.
        optimality (float): A measure of convergence (infinity norm of J^T * residuals).
                            Lower values indicate better convergence.
        neval (int): Counter for function/Jacobian evaluations during optimization.
        # Internal caches for w, dwdR, X_mid, w_mid, dwdR_mid, disps, current_grad, etc.
    """

    def __init__(
        self,
        atoms: List[str],
        path: Union[np.ndarray, List[np.ndarray]],
        scaler: Union[float, Callable[[np.ndarray],
                                      Tuple[np.ndarray, np.ndarray]]] = MAIN_DEFAULTS["morse_alpha"],
        threshold: float = MAIN_DEFAULTS["distance_cutoff"],
        min_neighbors: int = COORD_UTILS_DEFAULTS["get_bond_list_min_neighbors"],
        log_level: int = logging.INFO,
        friction: float = MAIN_DEFAULTS["friction"],
        rij_list_external: Optional[List[Tuple[int, int]]] = None,
        eq_distances_external: Optional[np.ndarray] = None,
        align: bool = True
    ):
        """
        Initializes the MorseGeodesic object.

        Args:
            atoms (List[str]): List of atom symbols.
            path (Union[np.ndarray, List[np.ndarray]]): The initial reaction path,
                as a 3D NumPy array (nimages, natoms, 3) or a list of 2D arrays.
            scaler (Union[float, Callable]): Defines the scaling for internal coordinates.
                If float: Morse 'alpha' parameter; a default Morse scaler is created.
                If Callable: A custom function `scaler(r_values) -> (wij, dw_drij)`.
            threshold (float): Distance cutoff for `get_bond_list` if `rij_list_external` is not provided.
            min_neighbors (int): Minimum neighbors for `get_bond_list`.
            log_level (int): Logging level for this instance.
            friction (float): Base friction coefficient for regularization during optimization.
            rij_list_external (Optional[List[Tuple[int, int]]]): Optionally, an externally
                defined list of atom pairs (internal coordinates).
            eq_distances_external (Optional[np.ndarray]): Optionally, corresponding equilibrium
                distances for `rij_list_external`. Must be provided if `rij_list_external` is.
        """
        self.align = align

        # Ensure path is a NumPy float array
        path_arr = np.array(path, dtype=float)

        # --- Input Validation ---
        if path_arr.ndim != 3:
            raise ValueError(
                f'Input path must be a 3D array (nimages, natoms, 3). Got shape {path_arr.shape}')
        if len(atoms) != path_arr.shape[1]:
            raise ValueError(
                f"Atom count mismatch: `atoms` list has {len(atoms)} elements, "
                f"but path geometries have {path_arr.shape[1]} atoms."
            )
        if path_arr.shape[0] < 1:  # Path must have at least one image
            raise ValueError(
                f"Input path must contain at least one image. Got {path_arr.shape[0]}.")

        # --- Initial Path Alignment and Basic Setup ---
        if self.align:
            rmsd0, self.path = align_path(path_arr)  # Align the input path
        else:
            # Initial RMSD
            rmsd0 = np.sqrt(np.mean(np.square(path_arr[-1] - path_arr[0])))
            self.path = path_arr
        self.nimages, self.natoms, _ = self.path.shape
        self.atoms: List[str] = atoms
        self.num_cart_coords: int = self.natoms * 3  # Cartesian DOFs per image

        logger.log(
            log_level, f"Initial path alignment max RMSD change: {rmsd0:10.2f} Angstroms")

        # --- Internal Coordinate Definition (rij_list) ---
        # `rij_list_py` is a temporary local variable during initialization.
        # `self.rij_list_np` is the instance variable used for computations.
        rij_list_py: List[Tuple[int, int]]
        self.rij_list_np: np.ndarray
        self.eq_distances: Optional[np.ndarray] = None

        if rij_list_external is not None and eq_distances_external is not None:
            # Use externally provided internal coordinate definition
            rij_list_py = rij_list_external
            self.eq_distances = eq_distances_external
            logger.debug(
                f"Using externally provided rij_list ({len(rij_list_py)} pairs) and eq_distances."
            )
        elif rij_list_external is not None or eq_distances_external is not None:
            # If one is provided, the other must be too.
            raise ValueError(
                "If providing `rij_list_external` or `eq_distances_external`, both must be provided."
            )
        else:
            # Generate internal coordinate list and equilibrium distances automatically
            rij_list_py, self.eq_distances = get_bond_list(
                self.path,
                atoms=self.atoms,
                threshold=threshold,
                min_neighbors=min_neighbors
            )
            logger.debug(
                f"Generated internal rij_list ({len(rij_list_py)} pairs) and eq_distances."
            )

        # Convert the Python list of pair tuples to a NumPy array for efficiency
        if rij_list_py:
            self.rij_list_np = np.array(rij_list_py, dtype=np.int32)
        else:  # No internal coordinates
            self.rij_list_np = np.empty((0, 2), dtype=np.int32)

        # Number of defined internal coordinates (Rijs)
        self.nrij: int = len(rij_list_py)

        # --- Scaler Function Setup ---
        # If `scaler` is a number, it's treated as Morse alpha
        if isinstance(scaler, (float, int)):
            morse_beta_val = MAIN_DEFAULTS["morse_beta"]  # Use default beta
            if self.nrij > 0 and (self.eq_distances is None or self.eq_distances.size != self.nrij):
                raise ValueError(
                    f"Cannot create Morse scaler: `eq_distances` is missing or has incorrect size "
                    f"({self.eq_distances.size if self.eq_distances is not None else 'None'}) "
                    f"for the number of internal coordinates ({self.nrij})."
                )
            self.scaler_func = morse_scaler(
                eq_distances=self.eq_distances if self.nrij > 0 else np.array(
                    []),  # Pass empty if no rijs
                alpha=float(scaler),
                beta=morse_beta_val
            )
            logger.debug(
                f"Using default Morse scaler: alpha={scaler}, beta={morse_beta_val}")
        elif callable(scaler):  # User has provided a custom scaler function
            self.scaler_func = scaler
            logger.debug("Using user-provided custom scaler function.")
        else:
            raise TypeError(
                "`scaler` must be a number (Morse alpha parameter) or a callable function "
                "of the form `scaler(r_values) -> (wij, dw_drij)`."
            )

        # --- Other Instance Variables ---
        # Base friction coefficient for regularization
        self.base_friction: float = friction
        self.log_level: int = log_level

        logger.log(log_level, "Initializing Morse-Geodesic object:")
        logger.log(log_level, f"  Number of images: {self.nimages:4d}")
        logger.log(log_level, f"  Number of atoms per image: {self.natoms:4d}")
        logger.log(
            log_level, f"  Number of internal coordinates (Rijs): {self.nrij:6d}")

        # --- Cache and State Variables for Optimization ---
        self.neval: int = 0  # Counter for function/Jacobian evaluations

        # Caches for scaled internal coordinates (w) and their Cartesian gradients (dwdR)
        # for each image on the main path. `None` indicates the value needs computation.
        self.w: List[Optional[np.ndarray]] = [None] * self.nimages
        self.dwdR: List[Optional[scipy.sparse.csr_matrix]] = [
            None] * self.nimages

        num_midpoints = self.nimages - 1 if self.nimages > 0 else 0
        # Caches for midpoint geometries (X_mid) and their corresponding w_mid, dwdR_mid.
        # X_mid[i] is the geometric midpoint between path[i] and path[i+1].
        self.X_mid: List[Optional[np.ndarray]] = [None] * num_midpoints
        self.w_mid: List[Optional[np.ndarray]] = [None] * num_midpoints
        self.dwdR_mid: List[Optional[scipy.sparse.csr_matrix]] = [
            None] * num_midpoints

        # Current displacement vector (residuals for LS)
        self.disps: Optional[np.ndarray] = None
        # Current Jacobian of `disps`
        self.current_grad: Optional[scipy.sparse.csc_matrix] = None
        # Current path length in scaled internal space
        self.length: float = 0.0
        # Current optimality metric (norm of J^T * disps)
        self.optimality: float = 0.0

        self.segment_lengths:  Optional[np.ndarray] = None

        # Stores the last set of Cartesian coordinates (of the optimized segment) for which
        # the state (disps, grad) was computed. Used to avoid recomputation if X hasn't changed.
        self.last_X_for_state: Optional[np.ndarray] = None

        # Pre-allocated NumPy array buffers for assembling `w` and `w_mid` values from lists
        # of arrays. This can improve performance in `_compute_disps` by avoiding repeated
        # array allocations and copies when using `np.concatenate` or `np.vstack`.
        self._w_arr_buffer = np.empty((self.nimages, self.nrij), dtype=float) \
            if self.nrij > 0 and self.nimages > 0 else np.empty((self.nimages, 0), dtype=float)

        self._w_mid_arr_buffer = np.empty((max(0, self.nimages - 1), self.nrij), dtype=float) \
            if self.nrij > 0 and self.nimages > 1 else np.empty((max(0, self.nimages - 1), 0), dtype=float)

    def _update_intc(self) -> None:
        """
        Computes and caches scaled internal coordinates (w) and their Cartesian
        gradients (dwdR) for all images on the current path (`self.path`) and
        for all geometric midpoints (`self.X_mid`) between adjacent path images.

        This method ensures that `self.w`, `self.dwdR`, `self.X_mid`, `self.w_mid`,
        and `self.dwdR_mid` are populated with up-to-date values. It uses the
        pre-computed `self.rij_list_np` (the list of atom pairs defining internals)
        and `self.scaler_func`.
        """

        # Update for main path images
        for i in range(self.nimages):
            if self.w[i] is None or self.dwdR[i] is None:  # If not cached or invalidated
                self.w[i], self.dwdR[i] = compute_wij(
                    self.path[i], self.rij_list_np, self.scaler_func
                )

        # Update for midpoints (X_mid[i] is between path[i] and path[i+1])
        for i in range(self.nimages - 1):
            if self.X_mid[i] is None:  # Calculate geometric midpoint if not cached
                self.X_mid[i] = (self.path[i] + self.path[i + 1]) / 2.0

            # If midpoint internals not cached
            if self.w_mid[i] is None or self.dwdR_mid[i] is None:
                if self.X_mid[i] is None:
                    # This state should not be reached if X_mid[i] is set above.
                    raise RuntimeError(
                        f"Midpoint X_mid[{i}] is None before compute_wij call in _update_intc.")
                self.w_mid[i], self.dwdR_mid[i] = compute_wij(
                    # type: ignore
                    self.X_mid[i], self.rij_list_np, self.scaler_func
                )

    def _update_geometry(self, X_segment_flat: np.ndarray, start_slice: int, end_slice: int) -> bool:
        """
        Updates a segment of the main path (`self.path`) with new Cartesian coordinates.

        If the geometry of the segment changes, this method invalidates the cached
        internal coordinate data (w, dwdR, X_mid, w_mid, dwdR_mid) for the affected
        images and their adjacent midpoints.

        Args:
            X_segment_flat (np.ndarray): A flattened 1D array of new Cartesian
                coordinates for the images in the segment `self.path[start_slice:end_slice)`.
            start_slice (int): The starting index (inclusive) of the segment in
                `self.path` to be updated.
            end_slice (int): The ending index (exclusive) of the segment in
                `self.path` to be updated.

        Returns:
            bool: True if the geometry was actually updated (i.e., new coordinates
                  were different from existing ones), False otherwise.
        """

        num_images_in_segment = end_slice - start_slice
        expected_size = num_images_in_segment * self.num_cart_coords

        if X_segment_flat.size != expected_size:
            raise ValueError(
                f"Shape mismatch in _update_geometry. `X_segment_flat` size {X_segment_flat.size} "
                f"does not match expected size {expected_size} for segment [{start_slice}:{end_slice})."
            )

        new_segment_coords = X_segment_flat.reshape(
            num_images_in_segment, self.natoms, 3)
        current_segment_view = self.path[start_slice:end_slice]

        # Check if the new coordinates are actually different from the current ones
        if current_segment_view.shape == new_segment_coords.shape and \
           np.array_equal(new_segment_coords, current_segment_view):
            return False  # No change in geometry

        # Update the path segment with the new coordinates
        self.path[start_slice:end_slice] = new_segment_coords

        # Invalidate caches for the modified images and their adjacent midpoints.
        # Any image `k` within the updated segment `[start_slice, end_slice)` needs its
        # `w[k]` and `dwdR[k]` caches cleared.
        # Midpoint `X_mid[k]` (between `k` and `k+1`) is affected if `k` is updated.
        # Midpoint `X_mid[k-1]` (between `k-1` and `k`) is affected if `k` is updated.
        for k_path_idx in range(start_slice, end_slice):
            self.w[k_path_idx] = None
            self.dwdR[k_path_idx] = None

            # If there's a midpoint to its right (X_mid[k_path_idx])
            if k_path_idx < self.nimages - 1:
                self.X_mid[k_path_idx] = None
                self.w_mid[k_path_idx] = None
                self.dwdR_mid[k_path_idx] = None

            # If there's a midpoint to its left (X_mid[k_path_idx - 1])
            if k_path_idx > 0:
                self.X_mid[k_path_idx - 1] = None
                self.w_mid[k_path_idx - 1] = None
                self.dwdR_mid[k_path_idx - 1] = None

        return True  # Geometry was updated

    def _compute_disps(self,
                       friction_coeff_override: Optional[float] = None,
                       dx_for_friction: Optional[np.ndarray] = None) -> None:
        """
        Computes the displacement vectors (`self.disps`). These form the objective
        function (residuals) for the least-squares optimization.

        The displacements are defined as the differences in scaled internal coordinates
        around each midpoint of the path:
          `vec_l[m] = w_mid[m] - w[m]`
          `vec_r[m] = w[m+1] - w_mid[m]`
        The goal of the optimization is to make these `vec_l` and `vec_r` vectors small.
        A friction term can also be appended to `self.disps` to regularize the optimization.

        Args:
            friction_coeff_override (Optional[float]): If provided, this friction
                coefficient is used instead of `self.base_friction`.
            dx_for_friction (Optional[np.ndarray]): If provided, this is the vector
                (current_coords - reference_coords) used to compute the friction
                contribution to the displacements.
        """

        self._update_intc()  # Ensure w, dwdR, X_mid, w_mid, dwdR_mid are up-to-date

        if self.nrij == 0:  # No internal coordinates defined
            self.length = 0.0
            self.disps = np.array([])  # Empty displacements
            self.segment_lengths = np.array()
            active_friction_coeff = friction_coeff_override if friction_coeff_override is not None else self.base_friction
            # If friction is active and dx is provided, disps will be only the friction term
            if dx_for_friction is not None and dx_for_friction.size > 0 and abs(active_friction_coeff) > COO_NON_ZERO_EPSILON:
                self.disps = active_friction_coeff * dx_for_friction
            return

        # Check if caches are properly filled before proceeding
        if any(w_val is None for w_val in self.w):
            raise ValueError(
                "Cache 'w' for path images is incomplete before _compute_disps.")
        # Midpoints exist only if nimages > 1
        if self.nimages > 1 and any(wm_val is None for wm_val in self.w_mid):
            raise ValueError(
                "Cache 'w_mid' for midpoints is incomplete before _compute_disps.")

        # Copy cached w and w_mid values into pre-allocated contiguous NumPy arrays
        # This is done for potential performance benefits in subsequent vectorized operations.
        for idx in range(self.nimages):
            if self.w[idx] is not None:
                self._w_arr_buffer[idx, :] = self.w[idx]  # type: ignore
            else:
                raise ValueError(f"self.w[{idx}] is None in _compute_disps")

        if self.nimages > 1:
            for idx in range(self.nimages - 1):
                if self.w_mid[idx] is not None:
                    self._w_mid_arr_buffer[idx,
                                           :] = self.w_mid[idx]  # type: ignore
                else:
                    raise ValueError(
                        f"self.w_mid[{idx}] is None in _compute_disps")

        w_arr = self._w_arr_buffer
        w_mid_arr = self._w_mid_arr_buffer

        # Calculate "left" and "right" difference vectors in scaled internal coordinate space
        # for each midpoint `m`.
        # vec_l[m] = w_mid[m] - w[m]  (difference between midpoint `m` and image `m`)
        # vec_r[m] = w[m+1] - w_mid[m] (difference between image `m+1` and midpoint `m`)
        vecs_l_arr = w_mid_arr - \
            w_arr[:-1] if self.nimages > 1 else np.empty((0, self.nrij))
        vecs_r_arr = w_arr[1:] - \
            w_mid_arr if self.nimages > 1 else np.empty((0, self.nrij))

        # The path length is the sum of the magnitudes (norms) of these difference vectors.
        self.length = (np.sum(np.linalg.norm(vecs_l_arr, axis=1)) if vecs_l_arr.size > 0 else 0.0) + \
                      (np.sum(np.linalg.norm(vecs_r_arr, axis=1))
                       if vecs_r_arr.size > 0 else 0.0)
        self.segment_lengths = np.linalg.norm(
            vecs_l_arr, axis=1) + np.linalg.norm(vecs_r_arr, axis=1)

        # Assemble the full displacement vector (residuals for least-squares)
        # by concatenating all vec_l and vec_r components.
        disps_components = []
        if vecs_l_arr.size > 0:
            disps_components.append(vecs_l_arr.ravel())
        if vecs_r_arr.size > 0:
            disps_components.append(vecs_r_arr.ravel())

        # Add friction term to displacements if applicable
        active_friction_coeff = friction_coeff_override if friction_coeff_override is not None else self.base_friction
        if dx_for_friction is not None and dx_for_friction.size > 0 and abs(active_friction_coeff) > COO_NON_ZERO_EPSILON:
            friction_term = active_friction_coeff * dx_for_friction
            disps_components.append(friction_term)

        self.disps = np.concatenate(
            disps_components) if disps_components else np.array([])

    def _add_sparse_block_to_coo(self,
                                 coo_data_list: List[np.ndarray],
                                 coo_rows_list: List[np.ndarray],
                                 coo_cols_list: List[np.ndarray],
                                 source_sparse_dwdR_matrix: Optional[scipy.sparse.csr_matrix],
                                 scale_factor: float,
                                 jac_row_offset: int,  # Offset for rows in the global Jacobian
                                 jac_col_offset: int  # Offset for columns in the global Jacobian
                                 ) -> None:
        """
        Helper function to add a scaled block of a sparse matrix (typically a dwdR matrix)
        to lists of COO (Coordinate format) components. These lists are later used to
        construct the full sparse Jacobian matrix.

        Args:
            coo_data_list: List to append non-zero data values.
            coo_rows_list: List to append corresponding row indices.
            coo_cols_list: List to append corresponding column indices.
            source_sparse_dwdR_matrix: The sparse matrix block to add (e.g., d(w)/dR).
            scale_factor: Factor to multiply the elements of `source_sparse_dwdR_matrix`.
            jac_row_offset: Offset to be added to the local row indices of the block
                            to place it correctly in the global Jacobian.
            jac_col_offset: Offset to be added to the local column indices.
        """

        if source_sparse_dwdR_matrix is None or source_sparse_dwdR_matrix.nnz == 0:
            return  # Nothing to add if the source matrix is empty or None

        # Scale the source matrix block
        scaled_block_sparse = source_sparse_dwdR_matrix * scale_factor
        # Convert to COO for easy access to row, col, data
        scaled_block_coo = scaled_block_sparse.tocoo()

        block_rows_local = scaled_block_coo.row
        block_cols_local = scaled_block_coo.col
        block_data = scaled_block_coo.data

        # Filter out numerically zero elements to maintain sparsity and efficiency
        significant_mask = np.abs(block_data) > COO_NON_ZERO_EPSILON
        if np.any(significant_mask):
            coo_data_list.append(block_data[significant_mask])
            # Adjust row and column indices with offsets for their position in the global Jacobian
            coo_rows_list.append(
                block_rows_local[significant_mask] + jac_row_offset)
            coo_cols_list.append(
                block_cols_local[significant_mask] + jac_col_offset)

    def _compute_disp_grad(self,
                           slice_start: int,  # Start index of the varied segment in `self.path`
                           # End index (exclusive) of the varied segment
                           slice_end: int,
                           friction_coeff_for_jac: Optional[float] = None,
                           # True if friction term was in `self.disps`
                           dx_was_present_for_disps: bool = False
                           ) -> scipy.sparse.csc_matrix:
        """
        Computes the Jacobian matrix (`self.current_grad`) of the displacement
        vector (`self.disps`) with respect to the Cartesian coordinates of the
        images in the specified segment `self.path[slice_start:slice_end)`.

        The Jacobian is constructed in sparse format (CSC). Its rows correspond to
        the elements of `self.disps` (components of `vec_l` and `vec_r` vectors,
        plus friction terms if present). Its columns correspond to the Cartesian
        coordinates of the *varied* images (those in the `slice_start:slice_end` segment).

        Args:
            slice_start: Starting index of the path segment whose coordinates are varied.
            slice_end: Ending index (exclusive) of this segment.
            friction_coeff_for_jac: Friction coefficient to use for the Jacobian of the
                                    friction term. If None, `self.base_friction` is used.
            dx_was_present_for_disps: Indicates if a `dx_for_friction` term was included
                                      when `self.disps` was last computed. This determines
                                      if the friction part of the Jacobian should be added.
        Returns:
            scipy.sparse.csc_matrix: The computed sparse Jacobian matrix.
        """

        num_varied_images = slice_end - slice_start
        num_cart_coords_varied_segment = num_varied_images * self.num_cart_coords

        # Handle cases with no varied images or no internal coordinates (nrij=0)
        if num_varied_images <= 0 or self.nrij == 0:
            num_total_midpoints_for_rows = max(0, self.nimages - 1)
            # Number of rows from internal coordinate displacements (vec_l, vec_r components)
            num_rows_internal_disps_for_shape = num_total_midpoints_for_rows * 2 * self.nrij

            active_fric_coeff_shape = friction_coeff_for_jac if friction_coeff_for_jac is not None else self.base_friction
            # Number of rows from the friction term in the Jacobian
            num_rows_friction_jac_shape = num_cart_coords_varied_segment \
                if dx_was_present_for_disps and abs(active_fric_coeff_shape) > COO_NON_ZERO_EPSILON else 0

            # Return an empty sparse matrix of the correct shape
            return scipy.sparse.csc_matrix(
                (num_rows_internal_disps_for_shape +
                 num_rows_friction_jac_shape, num_cart_coords_varied_segment),
                dtype=float
            )

        # Total number of midpoints in the full path
        num_total_midpoints = self.nimages - 1
        num_rows_internal_disps = num_total_midpoints * 2 * \
            self.nrij  # Rows for all vec_l and vec_r components

        active_fric_coeff = friction_coeff_for_jac if friction_coeff_for_jac is not None else self.base_friction
        num_rows_friction_jac = num_cart_coords_varied_segment \
            if dx_was_present_for_disps and abs(active_fric_coeff) > COO_NON_ZERO_EPSILON else 0
        total_rows_in_jacobian = num_rows_internal_disps + num_rows_friction_jac

        # Lists to store COO components for building the sparse Jacobian
        temp_coo_data_arrays: List[np.ndarray] = []
        temp_coo_rows_arrays: List[np.ndarray] = []
        temp_coo_cols_arrays: List[np.ndarray] = []

        # Iterate over each image `k` whose coordinates are being varied (i.e., in the segment `slice_start` to `slice_end`).
        # `k_varied_path_idx` is the absolute index of this image in `self.path`.
        # `i_offset_in_segment` is its relative index within the varied segment (0 to `num_varied_images` - 1).
        for i_offset_in_segment, k_varied_path_idx in enumerate(range(slice_start, slice_end)):
            # Jacobian d(w[k_varied_path_idx]) / d(R[k_varied_path_idx])
            dwdRk_varied_sparse = self.dwdR[k_varied_path_idx]
            if dwdRk_varied_sparse is None:
                raise ValueError(
                    f"Sparse dwdR cache miss for varied image {k_varied_path_idx}. Call _update_intc first.")

            # Column offset in the final Jacobian for derivatives w.r.t. R[k_varied_path_idx]
            jac_col_offset_for_this_k_varied = i_offset_in_segment * self.num_cart_coords

            # Now, iterate over all midpoints `m` in the path to build the contributions of R[k_varied_path_idx]
            # to d(vec_l[m])/dR and d(vec_r[m])/dR.
            for m_mid_path_idx in range(num_total_midpoints):
                # Row offsets in the global Jacobian for components of vec_l[m] and vec_r[m]
                jac_row_offset_for_vec_l_m = m_mid_path_idx * self.nrij
                jac_row_offset_for_vec_r_m = (
                    num_rows_internal_disps // 2) + (m_mid_path_idx * self.nrij)

                # --- Derivatives involving w_mid[m] ---
                # vec_l[m] = w_mid[m] - w[m]  =>  d(vec_l[m])/dR_k = d(w_mid[m])/dR_k - d(w[m])/dR_k
                # vec_r[m] = w[m+1] - w_mid[m] => d(vec_r[m])/dR_k = d(w[m+1])/dR_k - d(w_mid[m])/dR_k
                #
                # d(w_mid[m])/dR_k = (d(w_mid[m])/d(X_mid[m])) * (d(X_mid[m])/dR_k)
                # Since X_mid[m] = 0.5 * (R[m] + R[m+1]), then d(X_mid[m])/dR_k is:
                #   - 0.5 * Identity if k = m
                #   - 0.5 * Identity if k = m+1
                #   - 0 otherwise.
                # So, d(w_mid[m])/dR_k = 0.5 * dwdR_mid[m] if k=m or k=m+1.

                if k_varied_path_idx == m_mid_path_idx or k_varied_path_idx == m_mid_path_idx + 1:
                    # Jacobian d(w_mid[m]) / d(X_mid[m])
                    dwdR_mid_m_sparse = self.dwdR_mid[m_mid_path_idx]
                    if dwdR_mid_m_sparse is None:
                        raise ValueError(
                            f"Sparse dwdR_mid cache miss for midpoint {m_mid_path_idx}. Call _update_intc first.")

                    # Contribution to d(vec_l[m])/dR_k from d(w_mid[m])/dR_k (scale by +0.5)
                    self._add_sparse_block_to_coo(temp_coo_data_arrays, temp_coo_rows_arrays, temp_coo_cols_arrays,
                                                  dwdR_mid_m_sparse, 0.5,
                                                  jac_row_offset_for_vec_l_m, jac_col_offset_for_this_k_varied)
                    # Contribution to d(vec_r[m])/dR_k from -d(w_mid[m])/dR_k (scale by -0.5)
                    self._add_sparse_block_to_coo(temp_coo_data_arrays, temp_coo_rows_arrays, temp_coo_cols_arrays,
                                                  dwdR_mid_m_sparse, -0.5,
                                                  jac_row_offset_for_vec_r_m, jac_col_offset_for_this_k_varied)

                # --- Derivatives involving w[m] or w[m+1] ---
                # d(w[idx])/dR_k is non-zero only if idx == k.
                # It is dwdRk_varied_sparse (which is d(w[k_varied_path_idx])/d(R[k_varied_path_idx])).

                if k_varied_path_idx == m_mid_path_idx:
                    # Contribution to d(vec_l[m])/dR_k from -d(w[m])/dR_k (scale by -1.0)
                    # This occurs when k_varied_path_idx (k) is equal to m.
                    self._add_sparse_block_to_coo(temp_coo_data_arrays, temp_coo_rows_arrays, temp_coo_cols_arrays,
                                                  dwdRk_varied_sparse, -1.0,
                                                  jac_row_offset_for_vec_l_m, jac_col_offset_for_this_k_varied)

                if k_varied_path_idx == m_mid_path_idx + 1:
                    # Contribution to d(vec_r[m])/dR_k from +d(w[m+1])/dR_k (scale by +1.0)
                    # This occurs when k_varied_path_idx (k) is equal to m+1.
                    self._add_sparse_block_to_coo(temp_coo_data_arrays, temp_coo_rows_arrays, temp_coo_cols_arrays,
                                                  dwdRk_varied_sparse, 1.0,
                                                  jac_row_offset_for_vec_r_m, jac_col_offset_for_this_k_varied)

        # --- Add Friction Term to Jacobian ---
        if num_rows_friction_jac > 0:
            # The friction term in `disps` is `active_fric_coeff * (R_segment - R_segment_ref)`.
            # Its derivative w.r.t. `R_segment` is `active_fric_coeff * Identity_matrix`.
            diag_indices = np.arange(num_cart_coords_varied_segment)
            temp_coo_data_arrays.append(
                np.full(num_cart_coords_varied_segment, active_fric_coeff, dtype=float))
            # Rows for the friction term start after rows for internal displacements
            temp_coo_rows_arrays.append(num_rows_internal_disps + diag_indices)
            # Columns correspond to the Cartesian coordinates of the varied segment
            temp_coo_cols_arrays.append(diag_indices)

        # If Jacobian is entirely zero (e.g., nrij=0 and no friction)
        if not temp_coo_data_arrays:
            return scipy.sparse.csc_matrix(
                (total_rows_in_jacobian, num_cart_coords_varied_segment), dtype=float
            )

        # Concatenate all COO components and create the final sparse matrix in CSC format
        final_coo_data = np.concatenate(temp_coo_data_arrays)
        final_coo_rows = np.concatenate(temp_coo_rows_arrays)
        final_coo_cols = np.concatenate(temp_coo_cols_arrays)

        return scipy.sparse.csc_matrix(
            (final_coo_data, (final_coo_rows, final_coo_cols)),
            shape=(total_rows_in_jacobian, num_cart_coords_varied_segment)
        )

    def _ensure_state_updated(self,
                              X_flat_segment: Optional[np.ndarray],
                              slice_start: int,
                              slice_end: int,
                              x0_friction_ref: Optional[np.ndarray],
                              friction_coeff_for_update: Optional[float]):
        """
        Ensures that `self.disps` and related caches (`w`, `dwdR`, etc.) are
        consistent with the provided `X_flat_segment` (current Cartesian coordinates
        of the segment being optimized).

        If `X_flat_segment` has changed since the last computation, or if relevant
        caches are invalid (e.g., due to friction parameter changes), this method:
        1. Updates the geometry of the path segment (`self.path`).
        2. Recomputes `self.disps` (which involves calling `_update_intc`).
        3. Invalidates `self.current_grad` (the Jacobian) as it will need recomputation.

        This method is typically called at the beginning of `target_func` or `target_deriv`
        to prepare the object's state for the current optimization step.

        Args:
            X_flat_segment: Current Cartesian coordinates (flattened) of the path
                            segment being optimized. Can be None if just checking initial state.
            slice_start: Starting index of the segment in `self.path`.
            slice_end: Ending index (exclusive) of the segment.
            x0_friction_ref: Reference coordinates for the friction term. If provided,
                             and `friction_coeff_for_update` is non-zero, friction is active.
            friction_coeff_for_update: Friction coefficient to use for this state update.
                                       If None, `self.base_friction` is used.

        Returns:
            Optional[np.ndarray]: `dx_for_friction_calc` (i.e., `current_path_segment_flat - x0_friction_ref`)
                                  if friction was active during the `_compute_disps` call, otherwise None.
        """
        needs_disp_recomputation = False
        # (current_path_segment - reference_path_segment)
        dx_for_friction_calc: Optional[np.ndarray] = None

        if X_flat_segment is not None:
            # Check if coordinates (X_flat_segment) have changed since last state computation
            if self.last_X_for_state is None or \
               X_flat_segment.shape != self.last_X_for_state.shape or \
               not np.array_equal(X_flat_segment, self.last_X_for_state):

                # If geometry actually changed
                if self._update_geometry(X_flat_segment, slice_start, slice_end):
                    self.last_X_for_state = X_flat_segment.copy()  # Store the new state
                    needs_disp_recomputation = True
                    self.current_grad = None  # Jacobian needs recomputation due to geometry change
                elif self.disps is None:
                    # Geometry didn't change (e.g., X_flat_segment was identical to current path),
                    # but disps might not have been computed yet for this state.
                    needs_disp_recomputation = True
            # Recompute if disps not yet computed, or if friction term parameters might have changed
            # (signaled by x0_friction_ref and friction_coeff_for_update being non-None).
            elif self.disps is None or \
                    (x0_friction_ref is not None and friction_coeff_for_update is not None):
                needs_disp_recomputation = True
        elif self.disps is None:
            # No X_flat_segment provided (e.g., initial computation), and disps not computed yet.
            needs_disp_recomputation = True
            self.last_X_for_state = None  # No specific X to associate with this state yet

        if needs_disp_recomputation:
            active_friction_coeff = friction_coeff_for_update if friction_coeff_for_update is not None else self.base_friction

            if x0_friction_ref is not None and abs(active_friction_coeff) > COO_NON_ZERO_EPSILON:
                # If friction is active, calculate dx = (current_path_segment - reference)
                current_path_segment_flat = self.path[slice_start: slice_end].ravel(
                )
                if x0_friction_ref.shape == current_path_segment_flat.shape:
                    dx_for_friction_calc = current_path_segment_flat - x0_friction_ref
                else:
                    logger.warning(
                        "Shape mismatch for friction dx calculation. `x0_friction_ref` shape "
                        f"{x0_friction_ref.shape} vs current segment shape {current_path_segment_flat.shape}. "
                        "Friction may not be applied as expected."
                    )

            # Recompute displacements
            self._compute_disps(active_friction_coeff, dx_for_friction_calc)

            if self.disps is None:  # Should be populated by _compute_disps
                raise RuntimeError(
                    "Internal state `self.disps` is None after _compute_disps call.")

            # Jacobian needs recomputation as disps (or its basis) changed
            self.current_grad = None

        return dx_for_friction_calc

    def _compute_optimality(self, current_grad_jacobian: scipy.sparse.csc_matrix, slice_start: int, slice_end: int) -> None:
        """
        Computes the optimality metric for the current state.

        The optimality is defined as the infinity norm of `J^T * f`, where `J` is
        the Jacobian (`current_grad_jacobian`) and `f` is the residual vector (`self.disps`).
        This is a common convergence criterion for least-squares problems, representing
        the magnitude of the gradient of the sum-of-squares objective function.

        Args:
            current_grad_jacobian (scipy.sparse.csc_matrix): The Jacobian matrix
                d(disps)/d(R_segment).
            slice_start (int): Start index of the optimized segment.
            slice_end (int): End index (exclusive) of the optimized segment.
        """

        if self.disps is None or self.disps.size == 0:
            self.optimality = 0.0  # No displacements, system is optimal or undefined
            return

        num_cart_coords_in_segment = (
            slice_end - slice_start) * self.num_cart_coords
        if num_cart_coords_in_segment == 0:
            self.optimality = 0.0  # No optimizable degrees of freedom
            return

        # --- Sanity checks for Jacobian and displacement shapes ---
        if current_grad_jacobian.shape[0] != self.disps.size:
            logger.warning(
                f"Optimality calculation: Jacobian row count ({current_grad_jacobian.shape[0]}) "
                f"does not match displacement vector size ({self.disps.size}). "
                "Setting optimality to infinity."
            )
            self.optimality = np.inf
            return
        if current_grad_jacobian.shape[1] != num_cart_coords_in_segment:
            logger.warning(
                f"Optimality calculation: Jacobian column count ({current_grad_jacobian.shape[1]}) "
                f"does not match optimizable DOFs in segment ({num_cart_coords_in_segment}). "
                "Setting optimality to infinity."
            )
            self.optimality = np.inf
            return

        # Calculate J^T * f (gradient of 0.5 * sum(disps^2))
        jt_f_gradient = current_grad_jacobian.transpose().dot(self.disps)

        # Optimality is the infinity norm of this gradient
        self.optimality = np.linalg.norm(
            jt_f_gradient, ord=np.inf) if jt_f_gradient.size > 0 else 0.0

    def _compute_target_func_for_least_squares(
        self,
        X_flat_segment: np.ndarray,
        slice_start: int,
        slice_end: int,
        log_level_override: Optional[int] = None,
        x0_friction_ref: Optional[np.ndarray] = None,
        friction_override: Optional[float] = None
    ) -> np.ndarray:
        """
        Core function called by `scipy.optimize.least_squares` to get the residual
        vector (`self.disps`).

        This method ensures the internal state of the `MorseGeodesic` object
        (displacements `self.disps`, Jacobian `self.current_grad`, path length
        `self.length`, and optimality `self.optimality`) is updated and consistent
        with the provided `X_flat_segment` (current Cartesian coordinates of the
        path segment being optimized).

        Args:
            X_flat_segment (np.ndarray): Current Cartesian coordinates (flattened) of the
                                         path segment being optimized by `least_squares`.
            slice_start (int): Starting index of this segment in `self.path`.
            slice_end (int): Ending index (exclusive) of this segment.
            log_level_override (Optional[int]): Logging level to use for this specific call.
            x0_friction_ref (Optional[np.ndarray]): Reference coordinates for the friction term.
            friction_override (Optional[float]): Friction coefficient to use for this call.

        Returns:
            np.ndarray: The displacement vector `self.disps` (residuals).
        """
        current_log_lvl = log_level_override if log_level_override is not None else self.log_level

        # --- Validate input X_flat_segment shape against slice definition ---
        num_dofs_per_image = self.natoms * 3
        if num_dofs_per_image == 0 and X_flat_segment.size > 0:
            raise ValueError(
                "X_flat_segment is not empty for a system with 0 atoms per image.")
        if num_dofs_per_image > 0 and X_flat_segment.size % num_dofs_per_image != 0:
            raise ValueError(
                f"X_flat_segment size {X_flat_segment.size} is not a multiple of "
                f"DOFs per image ({num_dofs_per_image})."
            )

        num_images_in_X_flat_calc = X_flat_segment.size // num_dofs_per_image if num_dofs_per_image > 0 else 0
        if (slice_end - slice_start) != num_images_in_X_flat_calc:
            raise ValueError(
                f"Number of images implied by X_flat_segment ({num_images_in_X_flat_calc}) "
                f"does not match the slice definition (length {slice_end - slice_start})."
            )

        # --- Ensure internal state (disps, caches) is updated for X_flat_segment ---
        active_friction_coeff = friction_override if friction_override is not None else self.base_friction
        dx_used_for_friction = self._ensure_state_updated(
            X_flat_segment, slice_start, slice_end, x0_friction_ref, active_friction_coeff
        )

        if self.disps is None:  # Should be set by _ensure_state_updated
            raise RuntimeError(
                "`self.disps` is None after call to `_ensure_state_updated`.")

        # Determine if friction was active for the Jacobian calculation
        dx_was_present_for_grad_calc = (dx_used_for_friction is not None) and \
                                       (abs(active_friction_coeff)
                                        > COO_NON_ZERO_EPSILON)

        # --- Compute Jacobian if not already up-to-date ---
        if self.current_grad is None:
            self.current_grad = self._compute_disp_grad(
                slice_start, slice_end, active_friction_coeff, dx_was_present_for_grad_calc
            )

        if self.current_grad is None:  # Should be set by _compute_disp_grad
            raise RuntimeError(
                "Jacobian `self.current_grad` is None after call to `_compute_disp_grad`.")

        # --- Compute Optimality ---
        self._compute_optimality(self.current_grad, slice_start, slice_end)

        logger.log(current_log_lvl,
                   f"  Iter {self.neval:3d}: PathLength={self.length:10.3f}, Optimality(|dL|_inf)={self.optimality:7.3e}")
        self.neval += 1

        return self.disps

    def target_func(self, X_flat_segment: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Residuals function (f) for `scipy.optimize.least_squares`.

        This function is called by the optimizer at each iteration. It computes
        the displacement vector (`self.disps`) for the given `X_flat_segment`.
        It requires `slice_start` and `slice_end` to be passed in `**kwargs`
        to define which part of the path `X_flat_segment` corresponds to.

        Args:
            X_flat_segment (np.ndarray): Current Cartesian coordinates (flattened)
                of the path segment being optimized.
            **kwargs: Must include `slice_start` and `slice_end`. Can also include
                `log_level_override`, `x0_friction_ref`, `friction_override`.

        Returns:
            np.ndarray: The displacement vector `self.disps` (residuals).
        """
        if 'slice_start' not in kwargs or 'slice_end' not in kwargs:
            raise ValueError(
                "The `target_func` requires 'slice_start' and 'slice_end' to be "
                "provided in its keyword arguments."
            )
        return self._compute_target_func_for_least_squares(X_flat_segment, **kwargs)

    def target_deriv(self, X_flat_segment: np.ndarray, **kwargs: Any) -> scipy.sparse.csc_matrix:
        """
        Jacobian function (J) for `scipy.optimize.least_squares`.

        This function is called by the optimizer to get the Jacobian of the
        residuals (`self.disps`) with respect to `X_flat_segment`.
        It ensures the internal state (including `self.current_grad`) is updated
        if `X_flat_segment` has changed since the last call to `target_func`.
        Requires `slice_start` and `slice_end` in `**kwargs`.

        Args:
            X_flat_segment (np.ndarray): Current Cartesian coordinates (flattened)
                of the path segment.
            **kwargs: Must include `slice_start` and `slice_end`.

        Returns:
            scipy.sparse.csc_matrix: The Jacobian `self.current_grad`.
        """
        if 'slice_start' not in kwargs or 'slice_end' not in kwargs:
            raise ValueError(
                "The `target_deriv` requires 'slice_start' and 'slice_end' to be "
                "provided in its keyword arguments."
            )

        # If X_flat_segment has changed since the last state update, or if the gradient
        # hasn't been computed for the current X, then call _compute_target_func_for_least_squares.
        # This function will ensure all internal states, including self.current_grad, are updated.
        if self.last_X_for_state is None or \
           X_flat_segment.shape != self.last_X_for_state.shape or \
           not np.array_equal(X_flat_segment, self.last_X_for_state) or \
           self.current_grad is None:
            self._compute_target_func_for_least_squares(
                X_flat_segment, **kwargs)  # This updates self.current_grad

        if self.current_grad is None:  # Should be set by the call above
            raise RuntimeError(
                "`self.current_grad` is None in `target_deriv` after state update attempt.")

        # Ensure the returned Jacobian is in CSC format, as preferred by some SciPy solvers
        return self.current_grad if isinstance(self.current_grad, scipy.sparse.csc_matrix) \
            else scipy.sparse.csc_matrix(self.current_grad)

    def _smooth_scipy_least_squares(
        self,
        tol: Optional[float] = None,
        max_nfev: Optional[int] = None,
        start_slice_idx: Optional[int] = None,
        end_slice_idx: Optional[int] = None,
        log_level_override: Optional[int] = None,
        friction_coeff_for_opt: Optional[float] = None,
        xref_segment: Optional[np.ndarray] = None
    ) -> None:
        """
        Internal method to perform the path smoothing optimization for a specific
        segment of the path using `scipy.optimize.least_squares`.

        Args:
            tol (Optional[float]): Convergence tolerance for `gtol` in `least_squares`.
            max_nfev (Optional[int]): Maximum number of function evaluations for `least_squares`.
            start_slice_idx (Optional[int]): Starting index of the path segment to optimize.
                                             Defaults to 1 (fixing the first image).
            end_slice_idx (Optional[int]): Ending index (exclusive) of the segment.
                                           Defaults to `self.nimages - 1` (fixing the last image).
            log_level_override (Optional[int]): Logging level for this optimization run.
            friction_coeff_for_opt (Optional[float]): Friction coefficient to use for this run.
            xref_segment (Optional[np.ndarray]): Reference coordinates (flattened) for the
                                                 friction term, specific to this segment.
        """
        current_log_lvl = log_level_override if log_level_override is not None else self.log_level
        active_tol = tol if tol is not None else MAIN_DEFAULTS["tolerance"]
        active_max_nfev = max_nfev if max_nfev is not None else MAIN_DEFAULTS["max_iterations"]

        # Determine the segment of the path to optimize (typically interior images)
        s0 = start_slice_idx if start_slice_idx is not None else 1
        e0 = end_slice_idx if end_slice_idx is not None else self.nimages - 1

        effective_friction = friction_coeff_for_opt if friction_coeff_for_opt is not None else self.base_friction

        # --- Validate segment indices ---
        if not (0 <= s0 < self.nimages and s0 < e0 <= self.nimages):
            logger.log(current_log_lvl,
                       f"SciPy LS: Skipping optimization. Invalid segment [{s0}:{e0}) "
                       f"for a path with {self.nimages} images.")
            # Ensure disps and optimality are computed for the current state if skipping optimization
            if self.disps is None and self.nrij > 0:
                # Compute with current friction
                self._compute_disps(effective_friction)
            if self.optimality == 0.0 and self.disps is not None and self.disps.size > 0:
                # Attempt to compute global optimality if segment was invalid but path exists
                # Global interior segment
                gs0_global, ge0_global = (1, self.nimages - 1)
                if gs0_global < ge0_global:  # If there is an interior segment
                    if self.current_grad is None:
                        self.current_grad = self._compute_disp_grad(
                            gs0_global, ge0_global, effective_friction, False)
                    if self.current_grad is not None and self.current_grad.size > 0:
                        self._compute_optimality(
                            self.current_grad, gs0_global, ge0_global)
            return

        # --- Prepare for Optimization ---
        # Get the initial Cartesian coordinates for the segment to be optimized
        X_flat_initial_segment = self.path[s0:e0].ravel().copy()
        if X_flat_initial_segment.size == 0:
            logger.log(current_log_lvl,
                       f"SciPy LS: Skipping optimization. Segment [{s0}:{e0}) is empty (0 DOFs).")
            return

        # Determine the reference coordinates for the friction term
        x0_friction_ref = xref_segment.copy() \
            if xref_segment is not None and xref_segment.shape == X_flat_initial_segment.shape \
            else X_flat_initial_segment.copy()  # Default to the initial state of the segment

        if xref_segment is not None and xref_segment.shape != X_flat_initial_segment.shape:
            logger.warning(
                "SciPy LS: `xref_segment` shape mismatch with the optimizable segment. "
                "Using the initial state of the segment as the friction reference."
            )

        logger.log(current_log_lvl,
                   f"  SciPy LS: Optimizing images [{s0} to {e0 - 1}]. "
                   f"Number of optimizable DOFs: {len(X_flat_initial_segment)}.")

        # Arguments to be passed to target_func and target_deriv by least_squares
        kwargs_for_target_eval = dict(
            slice_start=s0,
            slice_end=e0,
            log_level_override=current_log_lvl,
            x0_friction_ref=x0_friction_ref,
            friction_override=effective_friction
        )

        # Initial evaluation to get disps, grad, and optimality before optimization
        self.target_func(X_flat_initial_segment, **kwargs_for_target_eval)

        # --- Perform Optimization if Not Already Converged ---
        if self.optimality > active_tol:
            J0 = self.current_grad  # Initial Jacobian for sparsity pattern
            if J0 is None:
                raise RuntimeError(
                    "Initial Jacobian (J0) is None before least_squares call.")

            # Provide Jacobian sparsity pattern to the optimizer if available and non-empty
            jac_sparsity_pattern = J0 if isinstance(
                J0, scipy.sparse.spmatrix) and J0.size > 0 and J0.nnz > 0 else None
            if jac_sparsity_pattern is not None and not isinstance(jac_sparsity_pattern, scipy.sparse.csc_matrix):
                # Ensure CSC format for 'trf' method
                jac_sparsity_pattern = jac_sparsity_pattern.tocsc()

            result = least_squares(
                fun=self.target_func,          # Residuals function
                x0=X_flat_initial_segment,     # Initial guess
                jac=self.target_deriv,         # Jacobian function
                method='trf',                  # Trust Region Reflective method
                # Robust loss function (less sensitive to outliers)
                loss='soft_l1',
                ftol=None,                     # Use gtol for convergence
                gtol=active_tol,               # Gradient tolerance for convergence
                xtol=None,                     # Step tolerance (not used here)
                max_nfev=active_max_nfev,      # Max function evaluations
                kwargs=kwargs_for_target_eval,  # Args for target_func/deriv
                x_scale='jac',                 # Scale variables based on Jacobian columns
                jac_sparsity=jac_sparsity_pattern,  # Provide sparsity pattern
                # Solver for trust-region subproblems (good for sparse)
                tr_solver='lsmr'
            )

            # Update geometry with optimized coordinates from result.x
            self._update_geometry(result.x, s0, e0)
            # Recompute final state (disps, optimality) for the optimized segment
            self._compute_target_func_for_least_squares(
                result.x, **kwargs_for_target_eval)
            logger.log(current_log_lvl,
                       f"SciPy LS: Status {result.status} after {result.nfev} evaluations. "
                       f"Final Optimality for segment: {self.optimality:.3e}")
        else:
            logger.log(current_log_lvl,
                       f"SciPy LS: Skipping optimization for segment [{s0}:{e0}). "
                       f"Already optimal (optimality {self.optimality:.3e} <= tolerance {active_tol:.3e}).")

    def smooth(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        log_level: Optional[int] = None,
        friction: Optional[float] = None,
        xref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Main public method to perform Morse-Geodesic path smoothing.

        This method optimizes the interior images of the path (endpoints are typically
        fixed by default) by minimizing the path length in the scaled internal
        coordinate space.

        Args:
            tol (Optional[float]): Convergence tolerance. Defaults to `MAIN_DEFAULTS["tolerance"]`.
            max_iter (Optional[int]): Maximum number of optimizer iterations (function evaluations).
                                      Defaults to `MAIN_DEFAULTS["max_iterations"]`.
            start (Optional[int]): Starting image index (inclusive) of the segment to optimize.
                                   Defaults to 1 (i.e., the second image, fixing the first).
            end (Optional[int]): Ending image index (exclusive) of the segment to optimize.
                                 Defaults to `self.nimages - 1` (fixing the last image).
            log_level (Optional[int]): Logging level for this specific smoothing operation.
                                       Defaults to `self.log_level`.
            friction (Optional[float]): Friction coefficient to use for this smoothing operation.
                                        Overrides `self.base_friction` if provided.
            xref (Optional[np.ndarray]): External reference path for the friction term.
                Can be a full path (nimages, natoms, 3) or a pre-flattened segment
                matching the optimizable part of the path. If None, friction is
                relative to the initial state of the segment being optimized.

        Returns:
            np.ndarray: A copy of the final smoothed path, shape (nimages, natoms, 3).
        """
        current_log_lvl = log_level if log_level is not None else self.log_level

        # --- Determine the actual segment of the path to optimize ---
        # Defaults to optimizing all interior images (endpoints fixed).
        s0_actual = start if start is not None else 1
        e0_actual = end if end is not None else self.nimages - 1

        # Ensure segment indices are valid
        if s0_actual < 0:
            s0_actual = 0
        if e0_actual > self.nimages:
            e0_actual = self.nimages  # `end` is exclusive

        # Validate that the segment is optimizable (i.e., has at least one image)
        if not (0 <= s0_actual < self.nimages and
                0 < e0_actual <= self.nimages and
                (e0_actual - s0_actual) > 0):  # Segment length must be > 0
            logger.log(current_log_lvl,
                       f"Main Smooth: Skipping optimization. Invalid or empty segment [{s0_actual}:{e0_actual}) "
                       "for optimization.")
            # If skipping, ensure disps and global optimality are computed for the current path state
            if self.disps is None and self.nrij > 0:
                self._clear_caches_and_reset_state(True)
                self._compute_disps(
                    friction if friction is not None else self.base_friction)

            # Compute global optimality for the full interior path if segment was invalid
            # Standard global interior segment
            gs0_global, ge0_global = (1, self.nimages - 1)
            if gs0_global < ge0_global and self.disps is not None and self.disps.size > 0:
                if self.current_grad is None:
                    self.current_grad = self._compute_disp_grad(
                        gs0_global, ge0_global,
                        friction if friction is not None else self.base_friction,
                        False  # Assume no dx_for_friction for this global check if not explicitly passed
                    )
                if self.current_grad is not None and self.current_grad.size > 0:
                    self._compute_optimality(
                        self.current_grad, gs0_global, ge0_global)
            else:  # Path too short for a global interior segment
                self.optimality = 0.0
            return self.path.copy()  # Return current path

        # --- Prepare reference segment for friction if `xref` is provided ---
        xref_segment_for_solver: Optional[np.ndarray] = None
        expected_xref_segment_size = (
            e0_actual - s0_actual) * self.num_cart_coords

        if xref is not None:
            if xref.ndim == 3 and \
               xref.shape[0] == self.nimages and \
               xref.shape[1:] == (self.natoms, 3):
                # Full path `xref` provided, slice the relevant segment for the optimizer
                sliced_xref_segment = xref[s0_actual:e0_actual].ravel()
                if sliced_xref_segment.size == expected_xref_segment_size:
                    xref_segment_for_solver = sliced_xref_segment
            elif xref.ndim == 1 and xref.size == expected_xref_segment_size:
                # `xref` is already a pre-flattened segment matching the optimizable part
                xref_segment_for_solver = xref.copy()

            if xref_segment_for_solver is None:
                logger.warning(
                    "Main Smooth: `xref` was provided, but its shape is incompatible with the "
                    "optimizable path segment. Friction will use the initial state of the "
                    "segment as its reference."
                )

        # --- Perform Optimization ---
        self.neval = 0  # Reset evaluation counter for this smoothing operation
        # Clear all caches before starting new optimization
        self._clear_caches_and_reset_state(True)

        logger.log(current_log_lvl,
                   f"Starting path smoothing using SciPy least_squares for path segment [{s0_actual}:{e0_actual}).")

        self._smooth_scipy_least_squares(
            tol=tol,
            max_nfev=max_iter,  # Note: max_iter from user becomes max_nfev for least_squares
            start_slice_idx=s0_actual,
            end_slice_idx=e0_actual,
            log_level_override=current_log_lvl,
            friction_coeff_for_opt=friction,  # Pass user-specified friction override
            xref_segment=xref_segment_for_solver
        )

        # --- Finalize Path ---
        # Re-align the entire path after optimization
        if self.align:
            rmsd_final_alignment, self.path = align_path(self.path)
        else:
            # RMSD between first and last images
            rmsd_final_alignment = np.sqrt(
                np.mean(np.square(self.path[-1] - self.path[0])))
        self._clear_caches_and_reset_state(
            True)  # Clear caches after alignment
        # Recompute disps for the final aligned path using the base friction
        self._compute_disps(self.base_friction)

        # Compute final global optimality for the interior part of the entire path
        # Standard global interior segment
        gs0_global, ge0_global = (1, self.nimages - 1)
        if gs0_global < ge0_global:  # If there's an interior segment
            final_global_grad = self._compute_disp_grad(
                gs0_global, ge0_global, self.base_friction, False)
            if final_global_grad is not None and final_global_grad.size > 0:
                self._compute_optimality(
                    final_global_grad, gs0_global, ge0_global)
            else:  # Jacobian might be empty if nrij = 0
                self.optimality = 0.0
        else:  # Path is too short (e.g., 2 images) to have an interior segment
            self.optimality = 0.0

        logger.log(current_log_lvl,
                   f"Main Smooth: Finished. Final Path Length: {self.length:12.5f}. "
                   f"Final Alignment Max RMSD: {rmsd_final_alignment:10.2e}. "
                   f"Global Path Optimality (|dL|_inf): {self.optimality:.3e}")

        return self.path.copy()  # Return a copy of the final smoothed path

    def _clear_caches_and_reset_state(self, clear_path_caches: bool = True) -> None:
        """
        Resets internal state variables related to the last optimization step
        (e.g., `last_X_for_state`, `current_grad`, `disps`).
        Optionally, also clears all cached `w`, `dwdR`, `X_mid`, `w_mid`, `dwdR_mid`
        values by setting them to `None`, forcing their recomputation on next access.

        Args:
            clear_path_caches (bool): If True, clears all `w`, `dwdR`, `X_mid`,
                                      `w_mid`, `dwdR_mid` caches.
        """
        self.last_X_for_state = None
        self.current_grad = None
        self.disps = None
        # self.length and self.optimality are typically recomputed after disps.

        if clear_path_caches:
            self.w = [None] * self.nimages
            self.dwdR = [None] * self.nimages
            num_midpoints = max(0, self.nimages - 1)
            self.X_mid = [None] * num_midpoints
            self.w_mid = [None] * num_midpoints
            self.dwdR_mid = [None] * num_midpoints


def run_geodesic_py(
    trajectory,
    tol=2e-3,
    nudge=0.1,
    ntries=1,
    scaling=1.7,
    dist_cutoff=3,
    friction=1e-2,
    sweep=None,
    maxiter=15,
    microiter=20,
    reconstruct=None,
    nimages=5,
    min_neighbors=4,
    align=True
):
    from .interpolation import redistribute

    symbols, X = trajectory.symbols, trajectory.coords
    if len(X) < 2:
        raise ValueError("Need at least two initial geometries.")

    # First redistribute number of images.  Perform interpolation if too few and subsampling if too many
    # images are given
    raw = redistribute(symbols, X, nimages=nimages,
                       tol=tol * 5, nudge=nudge, ntries=ntries, align=align)
    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    smoother = MorseGeodesic(symbols, raw, scaling, threshold=dist_cutoff,
                             friction=friction, min_neighbors=min_neighbors, align=align)
    try:
        smoother.smooth(tol=tol, max_iter=maxiter)
    finally:
        return smoother.path


def run_geodesic_get_smoother(
    input_object,
    tol=2e-3,
    nudge=0.1,
    ntries=1,
    scaling=1.7,
    dist_cutoff=3,
    friction=1e-2,
    sweep=None,
    maxiter=15,
    microiter=20,
    reconstruct=None,
    nimages=5,
    min_neighbors=4,
    align=True
):
    from neb_dynamics.geodesic_interpolation.interpolation import redistribute

    # Read the initial geometries.
    symbols, X = input_object

    if len(X) < 2:
        raise ValueError("Need at least two initial geometries.")

    # First redistribute number of images.  Perform interpolation if too few and subsampling if too many
    # images are given
    raw = redistribute(
        symbols, X, nimages=nimages, tol=tol * 5, nudge=nudge, ntries=ntries, align=align
    )
    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    smoother = MorseGeodesic(
        symbols,
        raw,
        scaling,
        threshold=dist_cutoff,
        friction=friction,
        min_neighbors=min_neighbors,
        align=align
    )
    # return smoother

    try:

        smoother.smooth(tol=tol, max_iter=maxiter)
    finally:
        # Save the smoothed path to output file.  try block is to ensure output is saved if one ^C the
        # process, or there is an error

        return smoother
        # write_xyz(output, symbols, smoother.path)
