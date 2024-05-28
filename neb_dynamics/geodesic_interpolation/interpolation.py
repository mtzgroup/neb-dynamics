"""Simplified geodesic interpolations module, which uses geodesic lengths as criteria
to add bisection points until point count meet desired number.
Will need another following geodesic smoothing to get final path.
"""


import numpy as np
from scipy.optimize import least_squares, minimize

from .coord_utils import (
    align_geom,
    align_path,
    compute_wij,
    get_bond_list,
    morse_scaler,
)
from .geodesic import Geodesic


def mid_point(atoms, geom1, geom2, tol=1e-2, nudge=0.01, threshold=4, ntries=1):
    """Find the Cartesian geometry that has internal coordinate values closest to the average of
    two geometries.

    Simply perform a least-squares minimization on the difference between the current internal
    and the average of the two end points.  This is done twice, using either end point as the
    starting guess.  DON'T USE THE CARTESIAN AVERAGE AS GUESS, THINGS WILL BLOW UP.

    This is used to generate an initial guess path for the later smoothing routine.
    Genenrally, the added point may not be continuous with the both end points, but
    provides a good enough starting guess.

    Random nudges are added to the initial geometry, so running multiple times may not yield
    the same converged geometry. For larger systems, one will never get the same geometry
    twice.  So one may want to perform multiple runs and check which yields the best result.

    Args:
        geom1, geom2:   Cartesian geometry of the end points
        tol:    Convergence tolarnce for the least-squares minimization process
        nudge:  Random nudges added to the initial geometry, which helps to discover different
                solutions.  Also helps in cases where optimal paths break the symmetry.
        threshold:  Threshold for including an atom-pair in the coordinate system

    Returns:
        Optimized mid-point which bisects the two endpoints in internal coordinates
    """
    # Process the initial geometries, construct coordinate system and obtain average internals
    geom1, geom2 = np.array(geom1), np.array(geom2)
    # print(f"{geom1=}\n{geom2=}")
    # print(f"{geom1 - geom2}")
    add_pair = set()
    geom_list = [geom1, geom2]
    # This loop is for ensuring a sufficient large coordinate system.  The interpolated point may
    # have atom pairs in contact that are far away at both end-points, which may cause collision.
    # One can include all atom pairs, but this may blow up for large molecules.  Here the compromise
    # is to use a screened list of atom pairs first, then add more if additional atoms come into
    # contant, then rerun the minimization until the coordinate system is consistant with the
    # interpolated geometry
    while True:
        rijlist, re = get_bond_list(
            geom_list, threshold=threshold + 1, enforce=add_pair
        )
        scaler = morse_scaler(alpha=0.7, re=re)
        w1, _ = compute_wij(geom1, rijlist, scaler)
        w2, _ = compute_wij(geom2, rijlist, scaler)
        w = (w1 + w2) / 2
        # print(f"{w1=}\n{w2=}")
        d_min, x_min = np.inf, None
        friction = 0.1 / np.sqrt(geom1.shape[0])

        def target_func(X):
            """Squared difference with reference w0"""
            wx, dwdR = compute_wij(X, rijlist, scaler)
            delta_w = wx - w
            val, grad = 0.5 * np.dot(delta_w, delta_w), np.einsum(
                "i,ij->j", delta_w, dwdR
            )

            return val, grad

        # The inner loop performs minimization using either end-point as the starting guess.
        for coef in [0.01, 0.99] * ntries:
            # print(f"COEF:{coef}")
            x0 = (geom1 * coef + (1 - coef) * geom2).ravel()
            x0 += nudge * np.random.random_sample(x0.shape)

            d = {
                "w": w,
            }
            # psave(d,'d.p')
            result = least_squares(
                lambda x: np.concatenate(
                    [compute_wij(x, rijlist, scaler)[0] - w, (x - x0) * friction]
                ),
                x0,
                lambda x: np.vstack(
                    [compute_wij(x, rijlist, scaler)[1], np.identity(x.size) * friction]
                ),
                ftol=tol,
                gtol=tol,
            )

            x_mid = result["x"].reshape(-1, 3)
            # Take the interpolated geometry, construct new pair list and check for new contacts
            new_list = geom_list + [x_mid]
            new_rij, _ = get_bond_list(new_list, threshold=threshold, min_neighbors=0)
            extras = set(new_rij) - set(rijlist)
            if extras:

                # Update pair list then go back to the minimization loop if new contacts are found
                geom_list = new_list
                add_pair |= extras
                break
            # Perform local geodesic optimization for the new image.
            smoother = Geodesic(
                atoms, [geom1, x_mid, geom2], 0.7, threshold=threshold, friction=1
            )
            smoother.compute_disps()
            width = max(
                [np.sqrt(np.mean((g - smoother.path[1]) ** 2)) for g in [geom1, geom2]]
            )
            dist, x_mid = width + smoother.length, smoother.path[1]

            if dist < d_min:
                d_min, x_min = dist, x_mid
        else:  # Both starting guesses finished without new atom pairs.  Minimization successful
            break
    return x_min


def redistribute(atoms, geoms, nimages, tol=1e-2, nudge=0.1, ntries=1):
    """Add or remove images so that the path length matches the desired number.

    If the number is too few, new points are added by bisecting the largest RMSD. If too numerous,
    one image is removed at a time so that the new merged segment has the shortest RMSD.

    Args:
        geoms:      Geometry of the original path.
        nimages:    The desired number of images
        tol:        Convergence tolerance for bisection.

    Returns:
        An aligned and redistributed path with has the correct number of images.
    """
    _, geoms = align_path(geoms)
    geoms = list(geoms)
    # If there are too few images, add bisection points
    while len(geoms) < nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[1:], geoms)]
        max_i = np.argmax(dists)

        insertion = mid_point(
            atoms, geoms[max_i], geoms[max_i + 1], tol, nudge=nudge, ntries=ntries
        )
        _, insertion = align_geom(geoms[max_i], insertion)
        geoms.insert(max_i + 1, insertion)
        geoms = list(align_path(geoms)[1])
    # If there are too many images, remove points
    while len(geoms) > nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[2:], geoms)]
        min_i = np.argmin(dists)

        del geoms[min_i + 1]
        geoms = list(align_path(geoms)[1])
    return geoms
