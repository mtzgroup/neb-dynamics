# mlp_geodesic/path_refinement.py
"""
Handles the adaptive path refinement logic for GeodesicOptimizer.

This module contains the logic for finding potential energy extrema between nodes,
adaptively inserting nodes to better resolve the path, and handling periodic path alignment.
"""
import torch
import logging
from typing import TYPE_CHECKING, Tuple, Dict, Any, List

from .path_tools import align_path_with_product_preservation, find_extremum_candidates, calculate_geodesic_segments
from .utils import PathData

if TYPE_CHECKING:
    from optimizer import GeodesicOptimizer

log = logging.getLogger("geodesic")

def _get_insertion_decision(
    candidate: Dict[str, Any],
    E_boundary_max: float,
    E_boundary_min: float,
    dynamic_threshold: float
) -> Tuple[bool, str]:
    """
    Determines if a candidate node should be inserted based on a set of rules.
    """
    E_quad_geom_abs = candidate['energy']
    extremum_type = candidate['extremum_type']

    if extremum_type == 'max':
        if E_quad_geom_abs > E_boundary_max + dynamic_threshold:
            return True, "Significant maximum that should be sampled"
        elif E_quad_geom_abs < max(E_boundary_max - dynamic_threshold, E_boundary_min):
            return True, "Poorly-predicted maximum: more sampling needed"

    return False, ""


def attempt_path_refinement(
    optimizer_instance: 'GeodesicOptimizer',
    initial_path_data: PathData
) -> Tuple[PathData, bool, bool]:
    """
    Performs adaptive path refinement by identifying poorly fit segments and
    inserting new nodes to improve the path representation.

    Returns:
        A tuple containing the new PathData object, a boolean indicating if any
        action was taken, and a boolean indicating if the path structure was changed.
    """
    any_action_taken = False
    structure_changed = False

    log.info("--- Path Refinement Check ---")
    log.info("  [Path Alignment] Aligning path at start of refinement check.")
    aligned_path_before_refinement = align_path_with_product_preservation(
        initial_path_data.nodes, optimizer_instance.RN_initial_state
    )

    current_path_data = optimizer_instance._evaluate_path_energies_forces(
        aligned_path_before_refinement, evaluate_midpoints=True
    )
    E_start_abs = current_path_data.energies[0].item()

    extremum_candidates = find_extremum_candidates(current_path_data, optimizer_instance)

    insert_proposals: List[Dict[str, Any]] = []
    if extremum_candidates and current_path_data.midpoint_energies is not None:

        segment_lengths = calculate_geodesic_segments(current_path_data.energies, current_path_data.midpoint_energies)

        for cand in extremum_candidates:
            k = cand['original_segment_idx']
            Ek_abs, Ekp1_abs = current_path_data.energies[k].item(), current_path_data.energies[k+1].item()
            Emid_abs = current_path_data.midpoint_energies[k].item()
            E_boundary_max, E_boundary_min = max(Ek_abs, Ekp1_abs, Emid_abs), min(Ek_abs, Ekp1_abs, Emid_abs)

            dynamic_threshold = optimizer_instance.config.refinement_dynamic_threshold_fraction * segment_lengths[k].item()

            should_insert, reason = _get_insertion_decision(cand, E_boundary_max, E_boundary_min, dynamic_threshold)
            if should_insert:
                insert_proposals.append({
                    'insert_after_full_path_idx': k,
                    'new_geom_tensor': cand['coords'],
                    'log_info': (f" Seg {k}, pred_type={cand['extremum_type']}, reason='{reason}'. E_rel:"
                                 f" E_k={(Ek_abs-E_start_abs):.3f},"
                                 f" E_mid={(Emid_abs-E_start_abs):.3f},"
                                 f" E_kp1={(Ekp1_abs-E_start_abs):.3f}."
                                 f" E_quad_geom={(cand['energy']-E_start_abs):.3f}."
                                 f" s_k={segment_lengths[k].item():.3f}.")
                })

    if insert_proposals:
        any_action_taken = True
        structure_changed = True
        nodes_to_process = list(torch.unbind(current_path_data.nodes))
        insert_proposals.sort(key=lambda x: x['insert_after_full_path_idx'], reverse=True)

        log.info(f"  [Refinement Action] Proposing {len(insert_proposals)} insertions:")
        for insertion in insert_proposals:
            nodes_to_process.insert(insertion['insert_after_full_path_idx'] + 1, insertion['new_geom_tensor'])
            log.info(f"    - INSERT:{insertion['log_info']}")

        path_before_final_eval = torch.stack(nodes_to_process)
        log.info("  [Path Alignment]  Re-aligning path after insertions.")
        aligned_path = align_path_with_product_preservation(
            path_before_final_eval, optimizer_instance.RN_initial_state
        )
        final_path_data = optimizer_instance._evaluate_path_energies_forces(aligned_path, True)
    else:
        log.info("  [Refinement Action] No insertion points met criteria.")
        final_path_data = current_path_data

    log.info("--- End Refinement Check ---")
    return final_path_data, any_action_taken, structure_changed

