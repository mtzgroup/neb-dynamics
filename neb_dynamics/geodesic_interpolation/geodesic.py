"""Geodesic smoothing.   Minimize the path length using redundant internal coordinate
metric to find geodesics directly in Cartesian, to avoid feasibility problems associated
with redundant internals.
"""

import numpy as np
from scipy.optimize import least_squares

from .coord_utils import align_path, compute_wij, get_bond_list, morse_scaler


class Geodesic(object):
    """Optimizer to obtain geodesic in redundant internal coordinates.  Core part is the calculation
    of the path length in the internal metric."""

    def __init__(
        self,
        atoms,
        path,
        scaler=1.7,
        threshold=3,
        min_neighbors=4,
        friction=1e-3,
    ):
        """Initialize the interpolater
        Args:
            atoms:      Atom symbols, used to lookup radii
            path:       Initial geometries of the path, must be of dimension `nimage * natoms * 3`
            scaler:     Either the alpha parameter for morse potential, or an explicit scaling function.
                        It is easier to get smoother paths with small number of data points using small
                        scaling factors, as they have large range, but larger values usually give
                        better energetics because they better represent the (sharp) energy landscape.
            threshold:  Distance cut-off for constructing inter-nuclear distance coordinates.  Note that
                        any atoms linked by three or less bonds will also be added.
            min_neighbors:  Minimum number of neighbors an atom must have in the atom pair list.
            log_level:  Logging level to use.
            friction:   Friction term in the target function which regularizes the optimization step
                        size to prevent explosion.
        """
        rmsd0, self.path = align_path(path)
        self.atoms = atoms

        if self.path.ndim != 3:
            raise ValueError(
                "The path to be interpolated must have 3 dimensions")
        self.nimages, self.natoms, _ = self.path.shape
        # Construct coordinates
        self.threshold = threshold
        self.min_neighbors = min_neighbors
        self.scaler_input = scaler
        self.construct_coords()
        self.friction = friction
        # Initalize interal storages for mid points, internal coordinates and B matrices

        self.neval = 0
        self.conv_path = []

    def construct_coords(self, index=None):
        """Construct coordinate system (pair list)"""
        if index is None:
            self.rij_list, self.re = get_bond_list(
                self.path,
                self.atoms,
                threshold=self.threshold,
                min_neighbors=self.min_neighbors,
            )
        else:
            self.rij_list, self.re = get_bond_list(
                self.path[index - 1: index + 2],
                self.atoms,
                threshold=self.threshold,
                min_neighbors=self.min_neighbors,
            )
        if isinstance(self.scaler_input, float):
            self.scaler = morse_scaler(re=self.re, alpha=self.scaler_input)
        else:
            self.scaler = self.scaler_input

        nimages = len(self.path)
        self.nrij = len(self.rij_list)
        self.w = [None] * nimages
        self.dwdR = [None] * nimages
        self.X_mid = [None] * (nimages - 1)
        self.w_mid = [None] * (nimages - 1)
        self.dwdR_mid = [None] * (nimages - 1)
        self.disps = self.grad = self.segment = None

    def update_intc(self):
        """Adjust unknown locations of mid points and compute missing values of internal coordinates
        and their derivatives.  Any missing values will be marked with None values in internal storage,
        and this routine finds and calculates them.  This is to avoid redundant evaluation of value and
        gradients of internal coordinates."""
        for i, (X, w, dwdR) in enumerate(zip(self.path, self.w, self.dwdR)):
            if w is None:
                self.w[i], self.dwdR[i] = compute_wij(
                    X, self.rij_list, self.scaler)
        for i, (X0, X1, w) in enumerate(zip(self.path, self.path[1:], self.w_mid)):
            if w is None:
                self.X_mid[i] = Xm = (X0 + X1) / 2
                self.w_mid[i], self.dwdR_mid[i] = compute_wij(
                    Xm, self.rij_list, self.scaler
                )

    def update_geometry(self, X, start, end):
        """Update the geometry of a segment of the path, then set the corresponding internal
        coordinate, derivatives and midpoint locations to unknown"""
        X = X.reshape(self.path[start:end].shape)
        if np.array_equal(X, self.path[start:end]):
            return False
        self.path[start:end] = X
        for i in range(start, end):
            self.w_mid[i] = self.w[i] = None
        self.w_mid[start - 1] = None
        return True

    def compute_disps(self, start=1, end=-1, dx=None, friction=1e-3):
        """Compute displacement vectors and total length between two images.
        Only recalculate internal coordinates if they are unknown."""
        if end < 0:
            end += self.nimages
        self.update_intc()
        # Calculate displacement vectors in each segment, and the total length
        vecs_l = [
            wm - wl
            for wl, wm in zip(self.w[start - 1: end], self.w_mid[start - 1: end])
        ]
        vecs_r = [
            wr - wm
            for wr, wm in zip(self.w[start: end + 1], self.w_mid[start - 1: end])
        ]
        self.length = np.sum(np.linalg.norm(vecs_l, axis=1)) + np.sum(
            np.linalg.norm(vecs_r, axis=1)
        )
        if dx is None:
            trans = np.zeros(self.path[start:end].size)
        else:
            trans = friction * dx  # Translation from initial geometry.  friction term
        self.disps = np.concatenate(vecs_l + vecs_r + [trans])
        self.disps0 = self.disps[: len(vecs_l) * 2]

    def compute_disp_grad(self, start, end, friction=1e-3):
        """Compute derivatives of the displacement vectors with respect to the Cartesian coordinates"""
        # Calculate derivatives of displacement vectors with respect to image Cartesians
        length = end - start + 1
        self.grad = np.zeros(
            (
                length * 2 * self.nrij + 3 * (end - start) * self.natoms,
                (end - start) * 3 * self.natoms,
            )
        )
        self.grad0 = self.grad[: length * 2 * self.nrij]
        grad_shape = (length, self.nrij, end - start, 3 * self.natoms)
        grad_l = self.grad[: length * self.nrij].reshape(grad_shape)
        grad_r = self.grad[length * self.nrij: length * self.nrij * 2].reshape(
            grad_shape
        )
        for i, image in enumerate(range(start, end)):
            dmid1 = self.dwdR_mid[image - 1] / 2
            dmid2 = self.dwdR_mid[image] / 2
            grad_l[i + 1, :, i, :] = dmid2 - self.dwdR[image]
            grad_l[i, :, i, :] = dmid1
            grad_r[i + 1, :, i, :] = -dmid2
            grad_r[i, :, i, :] = self.dwdR[image] - dmid1
        for idx in range((end - start) * 3 * self.natoms):
            self.grad[length * self.nrij * 2 + idx, idx] = friction

    def compute_target_func(self, X=None, start=1, end=-1, x0=None, friction=1e-3):
        """Compute the vectorized target function, which is then used for least
        squares minimization."""
        if end < 0:
            end += self.nimages
        if (
            X is not None
            and not self.update_geometry(X, start, end)
            and self.segment == (start, end)
        ):
            return
        self.segment = start, end
        dx = (
            np.zeros(self.path[start:end].size)
            if x0 is None
            else self.path[start:end].ravel() - x0.ravel()
        )
        self.compute_disps(start, end, dx=dx, friction=friction)
        self.compute_disp_grad(start, end, friction=friction)
        self.optimality = np.linalg.norm(
            np.einsum("i,i...", self.disps, self.grad), ord=np.inf
        )

        self.conv_path.append(self.path[1].copy())
        self.neval += 1
        return self.disps, self.grad

    def target_func(self, X, **kwargs):
        """Wrapper around `compute_target_func` to prevent repeated evaluation at
        the same geometry"""
        self.compute_target_func(X, **kwargs)
        return self.disps

    def target_deriv(self, X, **kwargs):
        """Wrapper around `compute_target_func` to prevent repeated evaluation at
        the same geometry"""
        self.compute_target_func(X, **kwargs)
        return self.grad

    def smooth(
        self,
        tol=1e-3,
        max_iter=50,
        start=1,
        end=-1,
        friction=None,
        xref=None,
    ):
        """Minimize the path length as an overall function of the coordinates of all the images.
        This should in principle be very efficient, but may be quite costly for large systems with
        many images.

        Args:
            tol:        Convergence tolerance of the optimality. (.i.e uniform gradient of target func)
            max_iter:   Maximum number of iterations to run.
            start, end: Specify which section of the path to optimize.
            log_level:  Logging level during the optimization

        Returns:
            The optimized path.  This is also stored in self.path
        """
        X0 = np.array(self.path[start:end]).ravel()
        if xref is None:
            xref = X0
        self.disps = self.grad = self.segment = None

        if friction is None:
            friction = self.friction
        # Configure the keyword arguments that will be sent to the target function.
        kwargs = dict(start=start, end=end, x0=xref, friction=friction)
        self.compute_target_func(**kwargs)  # Compute length and optimality
        if self.optimality > tol:
            result = least_squares(
                self.target_func,
                X0,
                self.target_deriv,
                ftol=tol,
                gtol=tol,
                max_nfev=max_iter,
                kwargs=kwargs,
                loss="soft_l1",
            )
            self.update_geometry(result["x"], start, end)

        rmsd, self.path = align_path(self.path)

        return self.path

    def sweep(
        self, tol=1e-3, max_iter=50, micro_iter=20, start=1, end=-1, reconstruct=False
    ):
        """Minimize the path length by adjusting one image at a time and sweeping the optimization
        side across the chain.  This is not as efficient, but scales much more friendly with the
        size of the system given the slowness of scipy's optimizers.  Also allows more detailed
        control and easy way of skipping nearly optimal points than the overall case.

        Args:
            tol:        Convergence tolerance of the optimality. (.i.e uniform gradient of target func)
            max_iter:   Maximum number of sweeps through the path.
            micro_iter: Number of micro-iterations to be performed when optimizing each image.
            start, end: Specify which section of the path to optimize.
            log_level:  Logging level during the optimization
            reconstruct: Whether to reconstruct pair list before each sweep

        Returns:
            The optimized path.  This is also stored in self.path
        """
        if end < 0:
            end = self.nimages + end
        self.neval = 0
        images = range(start, end)

        # Microiteration convergence tolerances are adjusted on the fly based on level of convergence.
        curr_tol = tol * 10
        self.compute_disps()  # Compute and print the initial path length

        for iteration in range(max_iter):
            max_dL = 0
            X0 = self.path.copy()
            for i in images[:-1]:  # Use self.smooth() to optimize individual images

                if reconstruct:
                    self.construct_coords()
                xmid = (self.path[i - 1] + self.path[i + 1]) * 0.5
                self.smooth(
                    curr_tol,
                    max_iter=min(micro_iter, iteration + 6),
                    start=i,
                    end=i + 1,
                    friction=self.friction if iteration else 0.1,
                    xref=xmid,
                )
                max_dL = max(max_dL, self.optimality)
            if reconstruct:
                self.construct_coords()
            self.compute_disps()  # Compute final length after sweep

            if max_dL < tol:  # Check for convergence.

                break
            # Adjust micro-iteration threshold
            curr_tol = max(tol * 0.5, max_dL * 0.2)
            images = list(reversed(images))  # Alternate sweeping direction.

        rmsd, self.path = align_path(self.path)

        return self.path


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
):
    from neb_dynamics.geodesic_interpolation.interpolation import redistribute
    from neb_dynamics.geodesic_interpolation.geodesic import Geodesic

    # Read the initial geometries.
    symbols, X = input_object

    if len(X) < 2:
        raise ValueError("Need at least two initial geometries.")

    # First redistribute number of images.  Perform interpolation if too few and subsampling if too many
    # images are given
    raw = redistribute(
        symbols, X, nimages=nimages, tol=tol * 5, nudge=nudge, ntries=ntries
    )
    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    smoother = Geodesic(
        symbols,
        raw,
        scaling,
        threshold=dist_cutoff,
        friction=friction,
        min_neighbors=min_neighbors,
    )
    # return smoother

    if sweep is None:
        sweep = len(symbols) > 35
    try:
        if sweep:
            smoother.sweep(
                tol=tol, max_iter=maxiter, micro_iter=microiter, reconstruct=reconstruct
            )
        else:
            smoother.smooth(tol=tol, max_iter=maxiter)
    finally:
        # Save the smoothed path to output file.  try block is to ensure output is saved if one ^C the
        # process, or there is an error

        return smoother
        # write_xyz(output, symbols, smoother.path)


def run_geodesic_py(
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
):
    smoother = run_geodesic_get_smoother(
        input_object=input_object,
        tol=tol,
        nudge=nudge,
        ntries=ntries,
        scaling=scaling,
        dist_cutoff=dist_cutoff,
        friction=friction,
        sweep=sweep,
        maxiter=maxiter,
        microiter=microiter,
        reconstruct=reconstruct,
        nimages=nimages,
        min_neighbors=min_neighbors,
    )
    return smoother.path
