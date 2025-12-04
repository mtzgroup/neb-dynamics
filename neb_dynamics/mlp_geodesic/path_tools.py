# mlp_geodesic/path_tools.py
"""
Utilities for path operations: alignment, analysis, and geodesic length/gradient calculation.

This module contains the core mathematical implementations for the geodesic path method,
including the functions to calculate the segment lengths (s_k) and their analytical
gradients with respect to atomic coordinates.
"""
import torch
import numpy as np
from typing import TYPE_CHECKING, Tuple, Dict, Any, List
import logging

from .utils import EPS_4THRT_DOUBLE, PathData
from .mlp_tools import evaluate_mlp_batch

if TYPE_CHECKING:
    from optimizer import GeodesicOptimizer

log = logging.getLogger("geodesic")

def _calculate_tangent_vector(
    R_prev: torch.Tensor, R_curr: torch.Tensor, R_next: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the normalized tangent vector(s) at interior node(s) R_curr.
    The tangent is defined as the normalized sum of the vectors from the
    previous and to the next nodes.
    """
    v_fwd = R_next - R_curr
    v_bwd = R_curr - R_prev

    v_fwd_norm = torch.linalg.norm(v_fwd, dim=(-2, -1), keepdim=True)
    v_bwd_norm = torch.linalg.norm(v_bwd, dim=(-2, -1), keepdim=True)

    u_fwd = v_fwd / (v_fwd_norm + EPS_4THRT_DOUBLE)
    u_bwd = v_bwd / (v_bwd_norm + EPS_4THRT_DOUBLE)

    tangent_sum = u_fwd + u_bwd
    tangent_sum_norm = torch.linalg.norm(tangent_sum, dim=(-2, -1), keepdim=True)

    u_k = tangent_sum / (tangent_sum_norm + EPS_4THRT_DOUBLE)
    return u_k

def align_geom(ref_coords: torch.Tensor, geom_coords: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Aligns geom_coords to ref_coords using Kabsch algorithm."""
    ref_center, geom_center = ref_coords.mean(dim=0), geom_coords.mean(dim=0)
    r_centered, g_centered = ref_coords - ref_center, geom_coords - geom_center
    cov_matrix = g_centered.T @ r_centered
    try:
        U, _, Vh = torch.linalg.svd(cov_matrix)
    except Exception as e:
        log.warning(f"SVD failed in align_geom (error: {e}). Using centered geometry without rotation.")
        aligned_coords = g_centered + ref_center
        rmsd_val = torch.sqrt(((aligned_coords - ref_coords)**2).sum(dim=1).mean()).item()
        return aligned_coords, rmsd_val
    R_rot = U @ Vh
    if torch.det(R_rot) < 0:
        Vh[-1, :] *= -1
        R_rot = U @ Vh
    aligned_coords = (g_centered @ R_rot) + ref_center
    rmsd_val = torch.sqrt(((aligned_coords - ref_coords)**2).sum(dim=1).mean()).item()
    return aligned_coords, rmsd_val

def align_path_with_product_preservation(
    path_nodes: torch.Tensor, product_initial_state: torch.Tensor
) -> torch.Tensor:
    """
    Performs a robust sequential alignment of the path.
    Product node always aligned from initial geometry to prevent drift accumulation.
    """
    if path_nodes.size(0) < 2:
        return path_nodes.clone()

    with torch.no_grad():
        aligned_path = path_nodes.clone()
        for i in range(path_nodes.size(0) - 2):
            aligned_node_i_plus_1, _ = align_geom(aligned_path[i], aligned_path[i+1])
            aligned_path[i+1] = aligned_node_i_plus_1
        last_aligned_node = aligned_path[-2]
        aligned_product_node, _ = align_geom(last_aligned_node, product_initial_state)
        aligned_path[-1] = aligned_product_node
        return aligned_path

def analyze_segment_parabola(
    Ek_abs: float, Emid_abs: float, Ekp1_abs: float,
    Rk_coords: torch.Tensor, Rkp1_coords: torch.Tensor
) -> Dict[str, Any]:
    """Fits a parabola to nodes and carterian midpoints. Also identifies potential extrema."""
    x_pts, y_pts = np.array([0.0, 0.5, 1.0]), np.array([Ek_abs, Emid_abs, Ekp1_abs])
    a_coeff, b_coeff, _ = np.polyfit(x_pts, y_pts, 2)
    analysis: Dict[str, Any] = {'is_valid_extremum_in_segment': False}
    if abs(a_coeff) < EPS_4THRT_DOUBLE: return analysis
    x_extremum = -b_coeff / (2 * a_coeff)
    is_valid = (EPS_4THRT_DOUBLE < x_extremum < 1.0 - EPS_4THRT_DOUBLE)
    if is_valid:
        analysis.update({
            'extremum_type': 'min' if a_coeff > 0 else 'max',
            'x_extremum': x_extremum,
            'is_valid_extremum_in_segment': True,
            'R_extremum_coords_est': (1.0 - x_extremum) * Rk_coords + x_extremum * Rkp1_coords
        })
    return analysis

def find_extremum_candidates(
    path_data: PathData, optimizer_instance: 'GeodesicOptimizer'
) -> List[Dict[str, Any]]:
    """Finds potential energy extrema along a path using quadratic fits, and computes their MLP energies."""
    if path_data.nodes.size(0) <= 1 or path_data.midpoint_energies is None: return []

    proposed_candidates_geom, proposal_metadata = [], []
    for k in range(path_data.nodes.size(0) - 1):
        quad_analysis = analyze_segment_parabola(
            path_data.energies[k].item(), path_data.midpoint_energies[k].item(), path_data.energies[k+1].item(),
            path_data.nodes[k], path_data.nodes[k+1]
        )

        if quad_analysis['is_valid_extremum_in_segment']:
            proposed_candidates_geom.append(quad_analysis['R_extremum_coords_est'])
            proposal_metadata.append({'original_segment_idx': k, 'extremum_type': quad_analysis['extremum_type']})

    if not proposed_candidates_geom: return []

    coords_to_eval = torch.stack(proposed_candidates_geom)
    actual_mlp_energies, _ = evaluate_mlp_batch(
        coords_to_eval, optimizer_instance.calc, optimizer_instance.raw_module,
        optimizer_instance._template_atoms, optimizer_instance.Z,
        optimizer_instance.device, optimizer_instance.dtype, optimizer_instance.backend
    )
    return [{'coords': coords_to_eval[i], 'energy': actual_mlp_energies[i].item(), **meta} for i, meta in enumerate(proposal_metadata)]

def calculate_geodesic_segments(E_main: torch.Tensor, E_mid: torch.Tensor) -> torch.Tensor:
    """Calculates the length of each path segment based on quadratic fits."""
    num_segments = E_main.size(0) - 1

    if num_segments <= 0: return torch.empty(0, device=E_main.device, dtype=E_main.dtype)

    tensor_opts = {'device': E_main.device, 'dtype': E_main.dtype}

    epsilon_tensor_sq = torch.tensor(EPS_4THRT_DOUBLE, **tensor_opts)

    Ek_all, Ekp1_all = E_main[:-1], E_main[1:]
    a_coeff = 2 * (Ek_all + Ekp1_all - 2 * E_mid)
    b_coeff = -3 * Ek_all - Ekp1_all + 4 * E_mid
    u0, u1 = b_coeff, 2 * a_coeff + b_coeff #Integral limits

    def G_func(u_val, eps_sq_tensor):
        """
        Analytic form of the integral
        """
        sqrt_term = torch.sqrt(u_val.pow(2) + eps_sq_tensor)
        log_arg = u_val + sqrt_term
        safe_log_arg = torch.where(log_arg < EPS_4THRT_DOUBLE, -eps_sq_tensor/(2*u_val), log_arg)
        return u_val * sqrt_term + eps_sq_tensor * torch.log(safe_log_arg)

    L_segments = torch.zeros(num_segments, **tensor_opts)

    mask_taylor = torch.abs(a_coeff) < EPS_4THRT_DOUBLE #for Taylor expansion about a_coeff
    if torch.any(mask_taylor): L_segments[mask_taylor] = torch.sqrt(b_coeff[mask_taylor].pow(2) + epsilon_tensor_sq) #Linear behavior

    mask_analytical = ~mask_taylor #a_coeff large enough for stable quadratic expansion
    if torch.any(mask_analytical):
        a_an = a_coeff[mask_analytical]
        L_segments[mask_analytical] = (G_func(u1[mask_analytical], epsilon_tensor_sq) - G_func(u0[mask_analytical], epsilon_tensor_sq)) / (4 * a_an)
    return L_segments

def calculate_gradient_from_segments(
    path_data: PathData, L_segments: torch.Tensor, beta: float,
    tangent_project: bool, climb: bool, alpha_climb: float
) -> torch.Tensor:
    """Calculates the gradient for the geodesic path optimization."""
    num_segments = path_data.nodes.size(0) - 1

    if num_segments <= 0 or path_data.midpoint_energies is None or path_data.midpoint_forces is None:
        return torch.zeros_like(path_data.nodes)

    tensor_opts = {'device': path_data.energies.device, 'dtype': path_data.energies.dtype}

    epsilon_tensor_sq = torch.tensor(EPS_4THRT_DOUBLE, **tensor_opts)

    Ek_all, Ekp1_all, Emid_all = path_data.energies[:-1], path_data.energies[1:], path_data.midpoint_energies

    a_g, b_g, Lk_g = 2 * (Ek_all + Ekp1_all - 2 * Emid_all), -3 * Ek_all - Ekp1_all + 4 * Emid_all, L_segments.detach()

    #Derivatives of segment length vs quad fit coefficients
    dsda, dsdb = torch.zeros_like(a_g), torch.zeros_like(a_g)

    mask_taylor = torch.abs(a_g) < EPS_4THRT_DOUBLE

    mask_analytical = ~mask_taylor

    #Analytical derivative when a is large enough
    if torch.any(mask_analytical):
        a_an, b_an, Lk_an = a_g[mask_analytical], b_g[mask_analytical], Lk_g[mask_analytical]
        u0_an, u1_an = b_an, 2 * a_an + b_an
        sqrt_u0_an, sqrt_u1_an = torch.sqrt(u0_an.pow(2) + epsilon_tensor_sq), torch.sqrt(u1_an.pow(2) + epsilon_tensor_sq)
        dsda[mask_analytical], dsdb[mask_analytical] = (sqrt_u1_an - Lk_an) / a_an, (sqrt_u1_an - sqrt_u0_an) / (2 * a_an)

    #Taylor series derivative otherwise
    if torch.any(mask_taylor):
        a_tay, b_tay = a_g[mask_taylor], b_g[mask_taylor]
        sqrt_b2_eps2 = torch.sqrt(b_tay.pow(2) + epsilon_tensor_sq)
        dsda[mask_taylor], dsdb[mask_taylor] = 0, b_tay / sqrt_b2_eps2

    #Derivatives of segment length vs node/midpoint energies
    dLdEk, dLdEmid, dLdEkp1 = dsda * 2 - dsdb * 3, -dsda * 4 + dsdb * 4, dsda * 2 - dsdb

    #Derivatives of energy vs node position
    Fk_all, Fkp1_all, Fmid_all = path_data.forces[:-1], path_data.forces[1:], path_data.midpoint_forces

    #Accumulate into derivatives of segment length vs node positions
    grad_s_Rk = dLdEk.view(-1, 1, 1) * (-Fk_all) + dLdEmid.view(-1, 1, 1) * (-0.5 * Fmid_all)
    grad_s_Rkp1 = dLdEkp1.view(-1, 1, 1) * (-Fkp1_all) + dLdEmid.view(-1, 1, 1) * (-0.5 * Fmid_all)

    #Penalty term
    meanL = L_segments.mean()
    dJvar_dL_factor = beta * (2.0 * (L_segments - (L_segments**2).sum() / L_segments.sum()) / (meanL**2)) if beta > 0 and L_segments.numel() > 1 else 0.0

    #Accumulate into full path length (grad_path_acc) and penalty (grad_path_var) derivatives
    grad_path_acc, grad_var_acc = torch.zeros_like(path_data.nodes), torch.zeros_like(path_data.nodes)
    indices_k, indices_kp1 = torch.arange(num_segments, device=path_data.nodes.device), torch.arange(1, num_segments + 1, device=path_data.nodes.device)
    grad_path_acc.index_add_(0, indices_k, grad_s_Rk)
    grad_path_acc.index_add_(0, indices_kp1, grad_s_Rkp1)

    if torch.is_tensor(dJvar_dL_factor):
        grad_var_acc.index_add_(0, indices_k, dJvar_dL_factor.view(-1, 1, 1) * grad_s_Rk)
        grad_var_acc.index_add_(0, indices_kp1, dJvar_dL_factor.view(-1, 1, 1) * grad_s_Rkp1)

    #Tangent projection out of total path length derivative
    if tangent_project and path_data.nodes.size(0) > 2:
        R_prev, R_curr, R_next = path_data.nodes[:-2], path_data.nodes[1:-1], path_data.nodes[2:]
        unit_tangents = _calculate_tangent_vector(R_prev, R_curr, R_next)
        grad_to_project = grad_path_acc[1:-1]
        dot_product = torch.sum(grad_to_project * unit_tangents, dim=(-2, -1), keepdim=True)
        projection_vector = dot_product * unit_tangents
        grad_path_acc[1:-1] -= projection_vector

    #Total loss gradient
    grad_nodes_acc = grad_path_acc + grad_var_acc

    #Climbing force
    if climb and path_data.nodes.size(0) > 2:
        k_peak = torch.argmax(path_data.energies).item()
        if 0 < k_peak < path_data.nodes.size(0) - 1:
            u_k = _calculate_tangent_vector(
                path_data.nodes[k_peak - 1].unsqueeze(0),
                path_data.nodes[k_peak].unsqueeze(0),
                path_data.nodes[k_peak + 1].unsqueeze(0)
            ).squeeze(0)
            if torch.linalg.norm(u_k) > EPS_4THRT_DOUBLE:
                grad_U = -path_data.forces[k_peak]
                grad_U_parallel_scalar = torch.sum(grad_U * u_k)
                grad_nodes_acc[k_peak] -= torch.sum(grad_nodes_acc[k_peak] * u_k) * u_k
                climb_grad_term = - (grad_U_parallel_scalar * u_k) * alpha_climb
                grad_nodes_acc[k_peak] += climb_grad_term

    return grad_nodes_acc

