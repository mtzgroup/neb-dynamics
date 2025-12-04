"""
Coordinate utilities for molecular geometry manipulation, alignment,
and internal coordinate generation.

This module provides a collection of functions essential for processing and
analyzing molecular coordinates within the MorseGeodesic interpolation package.
Key functionalities include:
- Aligning paths (sequences of geometries) or individual geometries to minimize RMSD.
- Generating lists of internal coordinates (atom pairs) based on distance criteria
  and connectivity.
- Calculating interatomic distances and their gradients with respect to Cartesian
  coordinates.
- Computing scaled internal coordinates and their Cartesian gradients using a
  provided scaler function (e.g., a Morse potential).
"""

import logging
import numpy as np
from scipy.spatial import KDTree # For efficient nearest-neighbor searches
import scipy.sparse # For handling sparse Jacobian matrices
from typing import List, Tuple, Dict, Union, Callable, Optional, Iterable, Set

# Local imports from the same package
from config import COORD_UTILS_DEFAULTS, ATOMIC_RADIUS

logger = logging.getLogger(__name__)


# --- Alignment Functions ---

def align_path(path: Union[np.ndarray, List[np.ndarray]]) -> Tuple[float, np.ndarray]:
    """
    Align a sequence of molecular geometries (a "path") to minimize RMSD
    movements between successive frames.

    The alignment process involves:
    1. Centering the first frame of the path at the origin.
    2. Iteratively aligning each subsequent frame to the previously aligned frame
       using the Kabsch algorithm.
    The geometric center of the entire path (based on the first frame's final
    alignment) is effectively moved to the origin.

    Args:
        path (Union[np.ndarray, List[np.ndarray]]): The input path, which can be
            a 3D NumPy array (nframes, natoms, 3) or a list of 2D NumPy arrays,
            where each 2D array is (natoms, 3).

    Returns:
        Tuple[float, np.ndarray]: A tuple containing:
            - max_rmsd (float): The maximum RMSD change observed between any
              two successive frames during the alignment process.
            - aligned_path (np.ndarray): A 3D NumPy array (nframes, natoms, 3)
              representing the aligned path.

    Raises:
        ValueError: If the input path is not a 3D array (or convertible to one),
                    is empty, or if geometries do not have 3 coordinates.
    """

    path_arr = np.array(path, dtype=float)  # Ensure path is a NumPy float array

    # --- Input Validation ---
    if path_arr.ndim != 3:
        raise ValueError(f"Input path must be a 3D array. Got shape {path_arr.shape}")
    if path_arr.shape[0] == 0: # Empty path
        return 0.0, path_arr
    if path_arr.shape[2] != 3: # Coordinates must be 3D (X, Y, Z)
        raise ValueError(f"Geometries must have 3 coordinates (X, Y, Z). Got {path_arr.shape[2]}.")

    # --- Alignment Process ---
    # Center the first frame of the path at the origin.
    # This sets the reference for the entire path's position.
    path_arr[0] -= np.mean(path_arr[0], axis=0)

    max_rmsd = 0.0  # To track the largest RMSD change during alignment

    # Iteratively align each frame to the one preceding it.
    for i in range(len(path_arr) - 1):
        ref_geom = path_arr[i]          # The (already aligned) reference frame
        target_geom = path_arr[i + 1]   # The frame to be aligned

        # Align target_geom to ref_geom
        rmsd, aligned_target_geom = align_geom(ref_geom, target_geom)

        path_arr[i + 1] = aligned_target_geom  # Update the path with the aligned frame
        if rmsd > max_rmsd:
            max_rmsd = rmsd

    return max_rmsd, path_arr


def align_geom(refgeom: np.ndarray, geom: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Aligns a given geometry (`geom`) to a reference geometry (`refgeom`)
    using the Kabsch algorithm.

    The Kabsch algorithm finds the optimal rotation matrix that minimizes the
    RMSD between two sets of paired points. This implementation also handles
    translation by centering the geometries before rotation.

    Args:
        refgeom (np.ndarray): The reference geometry, a NumPy array of shape
                              (number_of_atoms, 3).
        geom (np.ndarray): The geometry to be aligned, a NumPy array of the
                           same shape as `refgeom`.
                           Both inputs are assumed to be float64.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing:
            - rmsd (float): The Root Mean Square Deviation (RMSD) between the
              aligned `geom` and `refgeom`.
            - aligned_geom (np.ndarray): The `geom` after alignment, translated
              such that its centroid matches `refgeom`'s original centroid (if
              `refgeom` was centered at origin, `aligned_geom` will be too).

    Raises:
        ValueError: If `refgeom` and `geom` have mismatched shapes or are not
                    2D arrays with 3 columns.
    """

    # --- Input Validation ---
    if refgeom.shape != geom.shape:
        raise ValueError(f"Geometry shape mismatch: reference {refgeom.shape}, target {geom.shape}")
    if refgeom.ndim != 2 or refgeom.shape[1] != 3:
        raise ValueError(f"Geometries must be 2D arrays of shape (natoms, 3). Got: {refgeom.shape}")

    # --- Kabsch Algorithm Steps ---
    # 1. Translate both geometries so their centroids are at the origin.
    center_ref = np.mean(refgeom, axis=0)
    center_geom = np.mean(geom, axis=0)
    ref_centered = refgeom - center_ref
    geom_centered = geom - center_geom

    # 2. Calculate the covariance matrix H = geom_centered^T * ref_centered.
    #    Note: Some definitions use H = ref_centered^T * geom_centered.
    #    The choice affects the SVD components but leads to the same rotation.
    cov_matrix = np.dot(geom_centered.T, ref_centered)

    # 3. Perform Singular Value Decomposition (SVD) on the covariance matrix: H = U * S * Vt.
    try:
        U, S, Vt = np.linalg.svd(cov_matrix)
    except np.linalg.LinAlgError as e:
        logger.error(f"SVD failed in Kabsch algorithm: {e}. Returning unaligned geometry.")
        # If SVD fails, return the original geometry and its RMSD to the reference.
        unaligned_rmsd = np.sqrt(np.mean(np.sum((geom - refgeom) ** 2, axis=1)))
        return unaligned_rmsd, geom.copy()

    # 4. Calculate the optimal rotation matrix R = U * Vt.
    rotation_matrix = np.dot(U, Vt)

    # 5. Check for reflection. If det(R) < 0, it's a reflection, not a proper rotation.
    #    Correct by flipping the sign of the column in U corresponding to the smallest singular value.
    #    (Here, it's the last column of U as SVD typically sorts singular values).
    reflection_threshold = COORD_UTILS_DEFAULTS.get("kabsch_svd_reflection_threshold", -1e-9)
    if np.linalg.det(rotation_matrix) < reflection_threshold :
        U_corrected = U.copy()
        U_corrected[:, -1] *= -1.0  # Flip the sign of the last column
        rotation_matrix = np.dot(U_corrected, Vt) # Recalculate rotation matrix

    # 6. Apply the rotation to the centered `geom`.
    new_geom_centered = np.dot(geom_centered, rotation_matrix)

    # 7. Translate the rotated geometry back by adding the centroid of the reference geometry.
    #    This places the aligned geometry in the same frame of reference as the original `refgeom`.
    aligned_geom = new_geom_centered + center_ref

    # 8. Calculate the RMSD between the newly aligned geometry and the reference geometry.
    rmsd = np.sqrt(np.mean(np.sum((aligned_geom - refgeom) ** 2, axis=1)))

    return rmsd, aligned_geom


# --- Internal Coordinate List Generation ---

def get_bond_list(
    geom: Union[np.ndarray, List[np.ndarray]],
    atoms: Optional[List[str]] = None,
    threshold: float = COORD_UTILS_DEFAULTS["get_bond_list_threshold"],
    min_neighbors: int = COORD_UTILS_DEFAULTS["get_bond_list_min_neighbors"],
    snapshots: int = COORD_UTILS_DEFAULTS["get_bond_list_snapshots"],
    bond_threshold: float = COORD_UTILS_DEFAULTS["get_bond_list_bond_threshold"],
    enforce: Iterable[Tuple[int, int]] = ()
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Generates a list of "important" atom pairs (potential bonds or significant
    non-bonded interactions) and their estimated equilibrium distances.

    This list forms the basis for defining the redundant internal coordinates used
    in Morse-Geodesic calculations. The process involves:
    - Analyzing one or more geometry snapshots.
    - Identifying pairs within a `threshold` distance.
    - Adding pairs based on a "neighbor-of-neighbor" logic using a tighter `bond_threshold`.
    - Ensuring each atom has at least `min_neighbors` connections.
    - Estimating equilibrium distances for these pairs.

    Args:
        geom (Union[np.ndarray, List[np.ndarray]]): A single geometry (2D array)
            or a path of geometries (3D array or list of 2D arrays).
        atoms (Optional[List[str]]): List of atom symbols (e.g., ['C', 'H']).
            Used to estimate equilibrium distances from atomic radii. If None,
            a default equilibrium distance is used.
        threshold (float): Primary distance threshold (Angstroms) for identifying
            initial atom pairs.
        min_neighbors (int): Minimum number of connections (pairs) an atom should
            have in the final list. Additional pairs are added if an atom falls
            below this count.
        snapshots (int): Maximum number of geometry snapshots from the input `geom`
            (if it's a path) to sample for pair identification.
        bond_threshold (float): A tighter distance threshold (Angstroms) used to
            define "bonded" pairs for the neighbor-of-neighbor search logic.
        enforce (Iterable[Tuple[int, int]]): An iterable of atom index pairs
            (0-indexed) that should always be included in the returned list.

    Returns:
        Tuple[List[Tuple[int, int]], np.ndarray]:
            - rijlist (List[Tuple[int, int]]): A sorted list of unique atom pair
              indices, e.g., `[(0, 1), (0, 2), (1, 3), ...]`.
            - re_values (np.ndarray): A 1D NumPy array of estimated equilibrium
              distances (in Angstroms) corresponding to each pair in `rijlist`.
    """

    geom_arr_input = np.asarray(geom, dtype=float)

    # Ensure geom_arr is a 3D array (nframes, natoms, 3)
    if geom_arr_input.ndim == 2:  # Single geometry provided
        geom_arr = geom_arr_input.reshape(1, geom_arr_input.shape[0], geom_arr_input.shape[1])
    elif geom_arr_input.ndim == 3:  # Path (multiple geometries) provided
        geom_arr = geom_arr_input
    else:
        raise ValueError(f"Input geometry must be 2D (single frame) or 3D (path). Got shape {geom_arr_input.shape}")

    num_frames, num_atoms, _ = geom_arr.shape
    if num_atoms == 0:
        return [], np.array([]) # No atoms, no pairs

    # Ensure min_neighbors is not greater than the number of other atoms
    min_neighbors = min(min_neighbors, num_atoms - 1 if num_atoms > 0 else 0)

    # --- Determine Snapshots to Analyze ---
    actual_snapshots_to_consider = min(num_frames, snapshots)
    selected_indices_for_snapshots: Set[int] = set()

    if num_frames > 0:
        selected_indices_for_snapshots.add(0)  # Always include the first frame
    if num_frames > 1:
        selected_indices_for_snapshots.add(num_frames - 1)  # Always include the last frame

    # Sample interior frames if more snapshots are needed and available
    if actual_snapshots_to_consider > 2 and num_frames > 2:
        interior_indices_pool = list(range(1, num_frames - 1))
        num_additional_snapshots = actual_snapshots_to_consider - len(selected_indices_for_snapshots)
        if num_additional_snapshots > 0 and len(interior_indices_pool) > 0:
            num_to_sample = min(num_additional_snapshots, len(interior_indices_pool))
            chosen_interior_indices = np.random.choice(interior_indices_pool, num_to_sample, replace=False)
            selected_indices_for_snapshots.update(chosen_interior_indices)

    image_indices_to_process = sorted(list(selected_indices_for_snapshots))
    if not image_indices_to_process and num_frames > 0:
        image_indices_to_process = [0]  # Fallback for a single frame if selection logic fails

    # --- Identify Atom Pairs ---
    # Initialize set of atom pairs, ensuring (i,j) with i < j for uniqueness.
    rijset: Set[Tuple[int, int]] = set(tuple(sorted(p)) for p in enforce) if enforce else set()
    last_snapshot_tree: Optional[KDTree] = None  # KDTree of the last processed snapshot (for min_neighbors)

    for image_idx in image_indices_to_process:
        current_geom_snapshot = geom_arr[image_idx]
        tree = KDTree(current_geom_snapshot) # For efficient distance queries
        last_snapshot_tree = tree

        # 1. Add pairs within the primary distance `threshold`.
        for i, j in tree.query_pairs(r=threshold):
            rijset.add(tuple(sorted((i, j))))

        # 2. Add pairs based on neighbor-of-neighbor logic using a tighter `bond_threshold`.
        #    If A is bonded to B, and C is bonded to D, this logic might consider A-C, A-D, B-C, B-D
        #    or more commonly, if A-B and B-C are bonds, A-C (an angle) is considered.
        #    Here, it adds pairs (ni, nj) where ni is a "bonded" neighbor of i, and nj is of j.
        bonded_pairs_in_snapshot = list(tree.query_pairs(r=bond_threshold))
        snapshot_neighbors_dict: Dict[int, Set[int]] = {atom_idx: {atom_idx} for atom_idx in range(num_atoms)}
        for i, j in bonded_pairs_in_snapshot:
            snapshot_neighbors_dict[i].add(j)
            snapshot_neighbors_dict[j].add(i)

        for i, j in bonded_pairs_in_snapshot:  # For each "bonded" pair (i,j)
            for ni in snapshot_neighbors_dict.get(i, set()):  # For each neighbor of i (including i itself)
                for nj in snapshot_neighbors_dict.get(j, set()):  # For each neighbor of j (including j itself)
                    if ni != nj: # Ensure they are different atoms
                        rijset.add(tuple(sorted((ni, nj))))

    rijlist = sorted(list(rijset)) # Convert set to sorted list

    # --- Ensure Minimum Neighbors ---
    if min_neighbors > 0 and last_snapshot_tree is not None and num_frames > 0:
        # Count current neighbors for each atom from rijlist
        counts = np.zeros(num_atoms, dtype=int)
        for i, j in rijlist:
            counts[i] += 1
            counts[j] += 1

        geom_of_last_snapshot = geom_arr[num_frames - 1] # Use the last geometry for adding missing neighbors
        additional_pairs_to_satisfy_min_neighbors: Set[Tuple[int,int]] = set()

        for atom_idx in range(num_atoms):
            if counts[atom_idx] < min_neighbors:
                try:
                    # Query for enough neighbors to satisfy min_neighbors.
                    # `k` includes the atom itself, so query `min_neighbors + 1` (or `num_atoms` if smaller).
                    k_query = min(min_neighbors + 1, num_atoms)
                    _, neighbor_indices = last_snapshot_tree.query(geom_of_last_snapshot[atom_idx], k=k_query)

                    if not isinstance(neighbor_indices, (np.ndarray, list, tuple)):
                        neighbor_indices = [neighbor_indices] # Handle single neighbor case (scalar output)

                    for neighbor_j_idx_any_type in neighbor_indices:
                        neighbor_j_idx = int(neighbor_j_idx_any_type) # Ensure integer index
                        if atom_idx != neighbor_j_idx: # Don't add self-pairs
                            additional_pairs_to_satisfy_min_neighbors.add(tuple(sorted((atom_idx, neighbor_j_idx))))
                except Exception as e:
                    logger.warning(f"KDTree query failed for atom {atom_idx} during min_neighbors check: {e}")
        
        if additional_pairs_to_satisfy_min_neighbors:
            rijlist = sorted(list(rijset.union(additional_pairs_to_satisfy_min_neighbors)))

    # --- Estimate Equilibrium Distances (re_values) ---
    default_eq_dist = COORD_UTILS_DEFAULTS.get("default_eq_distance", 2.0)
    re_values: np.ndarray

    if atoms:  # If atom symbols are provided
        default_radius = COORD_UTILS_DEFAULTS.get("default_atomic_radius", 1.5)
        if len(atoms) == num_atoms:
            # Get radii for each atom, using default if symbol not found
            atom_radii = np.array([ATOMIC_RADIUS.get(s.capitalize(), default_radius) for s in atoms])
            # Estimate re as sum of radii for each pair in rijlist
            re_values = np.array([atom_radii[i] + atom_radii[j] for i, j in rijlist], dtype=float) if rijlist else np.array([], dtype=float)
        else:
            logger.warning(
                f"Atom list length ({len(atoms)}) does not match geometry atom count ({num_atoms}). "
                "Using default equilibrium distances for all pairs."
            )
            re_values = np.full(len(rijlist), default_eq_dist, dtype=float)
    else:  # If atom symbols are not provided, use a flat default equilibrium distance
        re_values = np.full(len(rijlist), default_eq_dist, dtype=float)

    logger.debug(f"Generated `get_bond_list` with {len(rijlist)} pairs.")
    return rijlist, re_values


# --- Distance and Gradient Calculations ---

def compute_rij(geom: np.ndarray, rij_list_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates raw interatomic distances (`rij_values`) and unit vectors
    (`grad_unit_vecs`) for specified atom pairs.

    The unit vectors `grad_unit_vecs` point from atom `j` to atom `i` for each
    pair `(i,j)` in `rij_list_np`. These are essentially `d(rij)/dR_i`.

    Args:
        geom (np.ndarray): Cartesian coordinates of the molecule, shape (natoms, 3).
                           Assumed to be a float64 NumPy array.
        rij_list_np (np.ndarray): A NumPy array of atom pair indices, shape (n_pairs, 2).
                                  Each row `[i, j]` defines a pair.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - rij_values (np.ndarray): 1D array of interatomic distances for each pair.
            - grad_unit_vecs (np.ndarray): 2D array (n_pairs, 3) of unit vectors
              (R_i - R_j) / ||R_i - R_j||.
    """

    if geom.ndim != 2 or geom.shape[1] != 3:
        raise ValueError(f"Input geometry must be a 2D array of shape (natoms, 3). Got {geom.shape}")
    if rij_list_np.size == 0: # No pairs to compute
        return np.array([]), np.empty((0, 3), dtype=float)
    if rij_list_np.ndim != 2 or rij_list_np.shape[1] != 2: # Each pair must have two indices
        raise ValueError(f"rij_list_np must be a 2D array of shape (n_pairs, 2). Got {rij_list_np.shape}")

    # Extract atom indices for all pairs
    atom_indices_i = rij_list_np[:, 0]  # Indices of the first atom in each pair
    atom_indices_j = rij_list_np[:, 1]  # Indices of the second atom in each pair

    # Calculate difference vectors (R_i - R_j) for all pairs
    diff_vectors = geom[atom_indices_i] - geom[atom_indices_j]

    # Calculate magnitudes of difference vectors (i.e., the distances rij)
    rij_values = np.linalg.norm(diff_vectors, axis=1)

    # Calculate unit vectors (gradients of distance w.r.t. Cartesian coordinates of atom i)
    # Handle potential division by zero for very close atoms using an epsilon.
    epsilon = COORD_UTILS_DEFAULTS.get("rij_norm_epsilon", 1e-9)
    grad_unit_vecs = np.zeros_like(diff_vectors) # Initialize as zeros

    # Create a mask for pairs where the distance is greater than epsilon
    valid_dist_mask = rij_values > epsilon
    if np.any(valid_dist_mask):
        # For valid distances, compute unit vector: (R_i - R_j) / ||R_i - R_j||
        grad_unit_vecs[valid_dist_mask] = diff_vectors[valid_dist_mask] / rij_values[valid_dist_mask, np.newaxis]

    return rij_values, grad_unit_vecs


def compute_wij(
    geom: np.ndarray,
    rij_list_np: np.ndarray,
    scaler_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, scipy.sparse.csr_matrix]:
    """
    Calculates scaled internal coordinates (`wij`) and their sparse Cartesian
    gradients (`dwdR`, i.e., the Jacobian d(wij)/d(Cartesians)).

    This function uses a provided `scaler_func` to transform raw interatomic
    distances (`rij`) into scaled internal coordinates (`wij`). It then computes
    the derivatives of these `wij` with respect to the Cartesian coordinates (`R`)
    of all atoms.

    Args:
        geom (np.ndarray): Cartesian coordinates of the molecule, shape (natoms, 3).
                           Assumed to be a float64 NumPy array.
        rij_list_np (np.ndarray): NumPy array of atom pair indices, shape (n_pairs, 2).
        scaler_func (Callable): A function that takes a 1D array of raw distances (`rij_values`)
                                and returns a tuple:
                                  - `wij_values` (1D array of scaled internal coordinates)
                                  - `dw_drij_values` (1D array of derivatives d(wij)/d(rij))

    Returns:
        Tuple[np.ndarray, scipy.sparse.csr_matrix]:
            - wij (np.ndarray): 1D array of scaled internal coordinates.
            - dwdR_flat_sparse (scipy.sparse.csr_matrix): Sparse Jacobian matrix
              of shape (n_pairs, num_cart_coords), where `num_cart_coords` is
              `natoms * 3`. Each row `k` contains the derivatives of `wij[k]`
              with respect to all Cartesian coordinates.
    """

    if geom.ndim != 2 or geom.shape[1] != 3:
        raise ValueError(f"Input geometry must be a 2D array of shape (natoms, 3). Got {geom.shape}")

    num_rij = rij_list_np.shape[0]  # Number of internal coordinates (pairs)
    num_atoms = geom.shape[0]
    num_cart_coords = num_atoms * 3 # Total number of Cartesian coordinates

    if num_rij == 0:  # No internal coordinates to process
        return np.array([]), scipy.sparse.csr_matrix((0, num_cart_coords), dtype=float)

    # 1. Calculate raw interatomic distances (rij) and their unit vector gradients (d(rij)/dR_i)
    rij_values, grad_unit_vecs = compute_rij(geom, rij_list_np)

    # 2. Apply the scaler function to get scaled internals (wij) and their derivatives w.r.t. rij (dw/drij)
    wij, dw_drij = scaler_func(rij_values)

    # --- Construct the Jacobian d(wij)/d(R) in COO sparse format ---
    # The Jacobian has `num_rij` rows and `num_cart_coords` columns.
    # Each internal coordinate `wij_k` (for pair (i,j)) depends on the Cartesian
    # coordinates of atom i and atom j. So, each row of the Jacobian will have
    # at most 6 non-zero elements (3 for atom i, 3 for atom j).
    max_nnz = num_rij * 6  # Maximum number of non-zero elements in the Jacobian

    # Row indices for COO format: each `wij_k` corresponds to `k`-th row, repeated 6 times for its 6 derivatives.
    coo_rows_np = np.repeat(np.arange(num_rij, dtype=np.int32), 6)

    # Atom indices for each pair
    atom_i_indices = rij_list_np[:, 0]  # First atom in each pair
    atom_j_indices = rij_list_np[:, 1]  # Second atom in each pair

    # Column indices for COO format: map to Cartesian coordinates (atom_idx * 3 + coord_idx)
    coo_cols_np = np.empty(max_nnz, dtype=np.int32)
    coo_cols_np[0::6] = atom_i_indices * 3       # Atom i, X-coordinate column index
    coo_cols_np[1::6] = atom_i_indices * 3 + 1   # Atom i, Y-coordinate column index
    coo_cols_np[2::6] = atom_i_indices * 3 + 2   # Atom i, Z-coordinate column index
    coo_cols_np[3::6] = atom_j_indices * 3       # Atom j, X-coordinate column index
    coo_cols_np[4::6] = atom_j_indices * 3 + 1   # Atom j, Y-coordinate column index
    coo_cols_np[5::6] = atom_j_indices * 3 + 2   # Atom j, Z-coordinate column index

    # Data for COO format: Jacobian elements d(wij_k)/d(R_alpha)
    # This is calculated using the chain rule:
    #   d(wij_k)/d(R_atom_p_coord_alpha) = (d(wij_k)/d(rij_k)) * (d(rij_k)/d(R_atom_p_coord_alpha))
    # Where d(rij_k)/d(R_i_alpha) = grad_unit_vecs[k, alpha]
    # And   d(rij_k)/d(R_j_alpha) = -grad_unit_vecs[k, alpha]
    coo_data_np = np.empty(max_nnz, dtype=float)
    # dw_drij is already a 1D array of shape (num_rij,)

    # Derivatives with respect to coordinates of atom i in each pair
    coo_data_np[0::6] = dw_drij * grad_unit_vecs[:, 0]  # d(wij)/d(Rx_i)
    coo_data_np[1::6] = dw_drij * grad_unit_vecs[:, 1]  # d(wij)/d(Ry_i)
    coo_data_np[2::6] = dw_drij * grad_unit_vecs[:, 2]  # d(wij)/d(Rz_i)

    # Derivatives with respect to coordinates of atom j in each pair
    coo_data_np[3::6] = dw_drij * (-grad_unit_vecs[:, 0]) # d(wij)/d(Rx_j)
    coo_data_np[4::6] = dw_drij * (-grad_unit_vecs[:, 1]) # d(wij)/d(Ry_j)
    coo_data_np[5::6] = dw_drij * (-grad_unit_vecs[:, 2]) # d(wij)/d(Rz_j)

    # Create sparse matrix in COO format and convert to CSR for efficient arithmetic operations.
    dwdR_flat_sparse = scipy.sparse.coo_matrix(
        (coo_data_np, (coo_rows_np, coo_cols_np)),  # (data, (row_ind, col_ind))
        shape=(num_rij, num_cart_coords),           # Shape of the Jacobian
        dtype=float
    ).tocsr()

    return wij, dwdR_flat_sparse


# --- SCALER FUNCTIONS ---

def morse_scaler(
    eq_distances: Union[np.ndarray, float],
    alpha: float,
    beta: float
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a higher-order function (a "scaler" function) that applies a
    Morse-like potential scaling to interatomic distances.

    The Morse-like form used is: `wij = exp(alpha * (1 - r/re)) + beta / (r/re)`,
    where `r` is the current distance and `re` is the equilibrium distance.

    Args:
        eq_distances (Union[np.ndarray, float]): Equilibrium distances (`re`).
            This can be a single float (if all pairs have the same `re`) or a
            1D NumPy array providing an `re` for each atom pair.
        alpha (float): The 'alpha' parameter of the Morse-like potential,
                       controlling the steepness/width of the exponential term.
        beta (float): The 'beta' parameter, controlling the strength of the
                      inverse term (1/ratio).

    Returns:
        Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            A `scaler` function. This returned function accepts a 1D NumPy array
            of current interatomic distances (`r_values`) and returns a tuple:
            - `wij` (np.ndarray): 1D array of scaled internal coordinates.
            - `dw_drij` (np.ndarray): 1D array of the derivatives d(wij)/d(rij).
    """

    _eq_dist_arr = np.asarray(eq_distances, dtype=float) # Ensure re is a NumPy array

    def scaler(r_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The actual scaling function returned by `morse_scaler`.
        Calculates `wij` and `d(wij)/d(rij)`.
        """
        if r_values.ndim != 1:
            raise ValueError("The `morse_scaler`'s returned function expects a 1D `r_values` input.")

        # Validate shape consistency between equilibrium distances and input distances
        # Allows _eq_dist_arr to be scalar or 1D array matching r_values, or a single-element array.
        if not (_eq_dist_arr.ndim == 0 or \
                (_eq_dist_arr.ndim == 1 and _eq_dist_arr.shape == r_values.shape) or \
                (_eq_dist_arr.size == 1 and _eq_dist_arr.ndim <=1 ) ):
             raise ValueError(
                 f"Shape mismatch between equilibrium distances (`eq_distances`, shape: "
                 f"{_eq_dist_arr.shape if _eq_dist_arr.ndim > 0 else 'scalar'}) and input "
                 f"distances (`r_values`, shape: {r_values.shape})."
             )

        # Ensure numerical stability by preventing division by zero or operations on very small numbers.
        epsilon = COORD_UTILS_DEFAULTS.get("scaler_epsilon", 1e-12)
        r_safe = np.maximum(r_values, epsilon)  # Current distances, floored by epsilon
        re_safe = np.maximum(_eq_dist_arr, epsilon) # Equilibrium distances, floored by epsilon

        ratio = r_safe / re_safe  # r/re

        # Calculate scaled coordinate (wij)
        exp_term_val = np.exp(alpha * (1.0 - ratio))
        inv_ratio_term_val = beta / ratio
        wij = exp_term_val + inv_ratio_term_val

        # Calculate derivative d(wij)/d(rij)
        # Derivative of exp(alpha * (1 - r/re)) w.r.t. r is exp(...) * (-alpha/re)
        d_exp_term = exp_term_val * (-alpha / re_safe)
        # Derivative of beta / (r/re) = beta * re / r w.r.t. r is -beta * re / r^2
        d_inv_ratio_term = -beta * re_safe / (r_safe**2)
        dw_drij = d_exp_term + d_inv_ratio_term

        return wij, dw_drij

    return scaler

