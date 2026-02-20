# mlp_geodesic/optimizer.py
"""
Core GeodesicOptimizer class for finding optimized paths.

This class orchestrates the entire optimization process, from loading the MLP model
to running the optimization stages and processing the final path.
"""
import numpy as np
import torch
import logging
from ase import Atoms
from typing import List, Tuple, Callable

from .mlp_tools import load_calculator, evaluate_mlp_batch
from .path_tools import (
    align_path_with_product_preservation,
    calculate_geodesic_segments,
    calculate_gradient_from_segments
)
from .optimization_stages import FIRE_Stage
from .path_refinement import attempt_path_refinement
from .utils import PathData, OptimizerConfig, EPS_4THRT_DOUBLE

log = logging.getLogger("geodesic")

class GeodesicOptimizer:
    """
    Optimizes a geodesic path on an MLP potential energy surface.
    """

    def __init__(
        self,
        frames: List[Atoms],
        backend: str,
        model_path: str,
        device: str,
        dtype: torch.dtype,
        config: OptimizerConfig,
        status_callback: Callable[[str], None] | None = None,
    ):
        """
        Initializes the GeodesicOptimizer.
        """
        torch.set_default_dtype(dtype)
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config
        self.backend = backend
        self.status_callback = status_callback

        self.Z = torch.as_tensor(frames[0].numbers, dtype=torch.long, device=self.device)
        self.num_atoms = len(self.Z)

        xyz = torch.stack([torch.as_tensor(f.positions, dtype=self.dtype, device=self.device) for f in frames])
        self.R0, self.RN = xyz[0].clone(), xyz[-1].clone()

        self.RN_initial_state = self.RN.clone().detach()

        self.Rint = torch.nn.Parameter(xyz[1:-1].clone())

        dtype_str = 'float32' if self.dtype == torch.float32 else 'float64'
        self.calc, self.raw_module = load_calculator(backend, model_path, device, dtype_str)
        log.info(f"Loaded {backend} calculator.")
        self._emit_status(f"Loaded {backend} model on {self.device.type}")

        if self.backend == 'fairchem' and self.dtype == torch.float64:
            log.warning("Fairchem backend with float64 is not recommended due to potential precision issues.")

        self._template_atoms = Atoms(
            numbers=self.Z.cpu().numpy(), positions=np.zeros((self.num_atoms, 3)),
            pbc=frames[0].get_pbc(), cell=frames[0].get_cell()
        )

    def _emit_status(self, message: str):
        if self.status_callback is not None:
            self.status_callback(message)

    def _get_current_path_nodes(self) -> torch.Tensor:
        """Assembles the full path from the fixed endpoints and optimizable interior nodes."""
        if self.Rint.numel() == 0:
            rint_data = torch.empty((0, self.num_atoms, 3), device=self.device, dtype=self.dtype)
        else:
            rint_data = self.Rint.data if isinstance(self.Rint, torch.nn.Parameter) else self.Rint

        return torch.cat([self.R0.unsqueeze(0), rint_data, self.RN.unsqueeze(0)], dim=0)

    def _update_path_from_tensor(self, path_tensor: torch.Tensor):
        """Updates R0, Rint, and RN state from a full path tensor."""
        with torch.no_grad():
            self.R0.copy_(path_tensor[0])
            self.RN.copy_(path_tensor[-1])

            new_rint_data = path_tensor[1:-1].detach().clone()

            if self.Rint.shape != new_rint_data.shape:
                self.Rint = torch.nn.Parameter(new_rint_data)
            elif self.Rint.numel() > 0:
                self.Rint.data.copy_(new_rint_data)

    def _evaluate_path_energies_forces(
        self, path_nodes_tensor: torch.Tensor, evaluate_midpoints: bool = False
    ) -> PathData:
        """Evaluates energies and forces for path nodes and optional midpoints."""
        E_main, F_main_flat = evaluate_mlp_batch(
            path_nodes_tensor, self.calc, self.raw_module, self._template_atoms, self.Z,
            self.device, self.dtype, self.backend
        )
        F_main_reshaped = F_main_flat.view_as(path_nodes_tensor)

        E_mid, F_mid_reshaped = None, None
        if evaluate_midpoints and path_nodes_tensor.size(0) > 1:
            geom_midpoints_coords = 0.5 * (path_nodes_tensor[:-1] + path_nodes_tensor[1:])
            E_mid_batch, F_mid_flat_batch = evaluate_mlp_batch(
                geom_midpoints_coords, self.calc, self.raw_module, self._template_atoms, self.Z,
                self.device, self.dtype, self.backend
            )
            E_mid = E_mid_batch
            F_mid_reshaped = F_mid_flat_batch.view_as(geom_midpoints_coords)

        return PathData(
            nodes=path_nodes_tensor, energies=E_main, forces=F_main_reshaped,
            midpoint_energies=E_mid, midpoint_forces=F_mid_reshaped
        )

    def _compute_loss(
        self, current_stage_step: int, beta_for_this_stage: float,
        enable_refinement_this_stage: bool, enable_climbing_this_stage: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Core function to compute the total loss and its gradient."""
        nodes_for_eval = self._get_current_path_nodes()
        path_data = self._evaluate_path_energies_forces(nodes_for_eval, evaluate_midpoints=True)
        structure_changed = False

        refinement_interval = self.config.refinement_step_interval
        attempt_refinement = enable_refinement_this_stage and \
                             ((refinement_interval == 0 and current_stage_step == 1) or \
                              (refinement_interval > 0 and current_stage_step % refinement_interval == 0))

        if attempt_refinement:
            path_data, _, structure_changed = attempt_path_refinement(self, path_data)
            if structure_changed:
                self._update_path_from_tensor(path_data.nodes)

        num_segments = path_data.nodes.size(0) - 1
        if num_segments <= 0 or path_data.midpoint_energies is None:
             return torch.tensor(0.0, device=self.device, dtype=self.dtype), \
                    torch.empty(0, device=self.device, dtype=self.dtype), \
                    path_data.energies.detach(), structure_changed

        L_segments = calculate_geodesic_segments(path_data.energies, path_data.midpoint_energies)

        meanL, J_path = L_segments.mean(), L_segments.sum()
        J_variance = beta_for_this_stage * ((L_segments / meanL - 1.0).pow(2)).sum()
        J_total_loss = J_path + J_variance

        is_climb= self.config.climb and enable_climbing_this_stage

        grad_nodes_acc = calculate_gradient_from_segments(
            path_data, L_segments, beta_for_this_stage, self.config.tangent_project,
            is_climb, self.config.alpha_climb
        )

        grad_nodes_acc[0].zero_()
        grad_nodes_acc[-1].zero_()

        if self.Rint.numel() > 0:
            g_Rint = grad_nodes_acc[1:-1].detach().clone()
            if self.Rint.grad is None or self.Rint.grad.shape != g_Rint.shape:
                 self.Rint.grad = torch.zeros_like(g_Rint)
            self.Rint.grad.copy_(g_Rint)

        return J_total_loss, L_segments.detach(), path_data.energies.detach(), structure_changed

    def report_step(self, stage: str, step: int, E_main: torch.Tensor, L_segments: torch.Tensor, J_penalty: float, gn: float):
        """Logs a formatted report of the current optimization step."""
        e0, eN = E_main[0].item(), E_main[-1].item()
        e_peak = E_main.max().item()
        barrier_fwd, barrier_back = e_peak - e0, e_peak - eN
        total_path_len = L_segments.sum().item()

        barrier_sum = barrier_fwd + barrier_back
        optimality_ratio = total_path_len / barrier_sum if barrier_sum > EPS_4THRT_DOUBLE else float('inf')

        min_s = L_segments.min().item() if L_segments.numel() > 0 else 0.0
        mean_s = L_segments.mean().item() if L_segments.numel() > 0 else 0.0
        max_s = L_segments.max().item() if L_segments.numel() > 0 else 0.0

        log_str = (
            f"{stage.ljust(20)} {step:4d} | E_prod = {eN - e0:.3f} E_peak = {barrier_fwd:.3f} Path_Len = {total_path_len:.3f} "
            f"Penalty = {J_penalty:.3f}  [s_min,s_mean,s_max] = [{min_s:.2f},{mean_s:.2f},{max_s:.2f}] "
            f"Optimality_Ratio = {optimality_ratio:.2f} |grad| = {gn:.2e}"
        )
        log.info(log_str)
        if step == 1 or step % 20 == 0:
            self._emit_status(
                f"MLPGI {stage} step {step} | E_peak={barrier_fwd:.3f} | grad={gn:.2e}"
            )

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Executes the full two-stage optimization process."""
        log.info("Starting two-stage FIRE optimization.")
        self._emit_status("Starting MLPGI two-stage FIRE optimization")
        if self.config.tangent_project:
            log.info("Tangent projection mode is ENABLED for both stages.")
        if self.config.climb:
            log.info(f"Climbing image configured with alpha={self.config.alpha_climb}. It will be activated in Stage 2.")
        else:
            log.info("Climbing image is disabled by user configuration.")

        log.info("[Path Alignment] Performing initial alignment before Stage 1.")
        self._emit_status("MLPGI aligning initial path")
        current_path_tensor = self._get_current_path_nodes()
        aligned_path = align_path_with_product_preservation(current_path_tensor, self.RN_initial_state)
        self._update_path_from_tensor(aligned_path)

        stage1 = FIRE_Stage(
            self, "FIRE Stage1 Relaxation", self.config.variance_penalty_weight,
            self.config.fire_stage1_iter, enable_refinement=False, enable_climbing=False,
            apply_convergence_checks=True
        )
        stage1.execute()

        log.info("[Path Alignment] Aligning path after Stage 1.")
        self._emit_status("MLPGI aligning path after stage 1")
        current_path_tensor = self._get_current_path_nodes()
        aligned_path = align_path_with_product_preservation(current_path_tensor, self.RN_initial_state)
        self._update_path_from_tensor(aligned_path)

        stage2_iters = (self.config.fire_stage2_iter // self.config.refinement_step_interval) * self.config.refinement_step_interval + self.config.refinement_step_interval // 2
        if stage2_iters != self.config.fire_stage2_iter:
            log.info(f"Number of Stage 2 Iterations changed to {stage2_iters} for compatibility with refining")

        stage2 = FIRE_Stage(
            self, "FIRE Stage2 Refine", self.config.variance_penalty_weight,
            stage2_iters, enable_refinement=True, enable_climbing=True,
            apply_convergence_checks=True
        )
        stage2.execute()

        final_path_data = self._evaluate_path_energies_forces(self._get_current_path_nodes().detach())
        log.info("Optimization finished.")
        self._emit_status("MLPGI optimization finished")
        return final_path_data.nodes.cpu().numpy(), final_path_data.energies.cpu().numpy()
