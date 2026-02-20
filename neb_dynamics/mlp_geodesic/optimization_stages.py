# mlp_geodesic/optimization_stages.py
"""
Encapsulates the logic for distinct optimization stages.

This module defines an abstract base class `OptimizationStage` and a concrete
implementation for the FIRE optimization stage.
"""
import torch
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Any, Dict
import numpy as np
from ase import Atoms
from ase.optimize import FIRE
from ase.calculators.calculator import Calculator, all_changes


if TYPE_CHECKING:
    from optimizer import GeodesicOptimizer

log = logging.getLogger("geodesic")

class _GeodesicCalculator(Calculator):
    """
    An ASE-compatible calculator that wraps the PyTorch-based loss function.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, optimizer_instance: 'GeodesicOptimizer', stage_instance: 'FIRE_Stage', initial_step_count: int):
        super().__init__()
        self.optimizer = optimizer_instance
        self.stage = stage_instance
        self.step = initial_step_count
        self.structure_changed_on_last_step = False
        self.observer_data: Dict[str, Any] = {}

    def calculate(self, atoms: Atoms, properties=['energy'], system_changes=all_changes):
        """
        Performs a single evaluation of the loss (energy) and its gradient (forces).
        """
        super().calculate(atoms, properties, system_changes)
        
        self.step += 1
        self.structure_changed_on_last_step = False
        
        with torch.no_grad():
            new_rint_flat = torch.as_tensor(self.atoms.get_positions(), device=self.optimizer.device, dtype=self.optimizer.dtype)
            new_rint = new_rint_flat.view(*self.optimizer.Rint.shape)
            self.optimizer.Rint.data.copy_(new_rint)

        loss, L_tensor, E_main, structure_changed = self.optimizer._compute_loss(
            self.step, self.stage.beta_for_loss, self.stage.enable_refinement, self.stage.enable_climbing
        )
        
        if structure_changed:
            log.info(f"Path structure changed during {self.stage.stage_name}. Interrupting FIRE run.")
            self.structure_changed_on_last_step = True
            raise RuntimeError("Path structure changed, optimizer state is now invalid.")

        forces_flat = -self.optimizer.Rint.grad.detach().view(-1, 3).cpu().numpy()
        
        self.results['energy'] = loss.item()
        self.results['forces'] = forces_flat

        self.observer_data['L_tensor'] = L_tensor
        self.observer_data['E_main'] = E_main
        self.observer_data['grad_norm'] = np.linalg.norm(forces_flat)
        self.observer_data['J_penalty'] = loss.item() - L_tensor.sum().item() if L_tensor.numel() > 0 else loss.item()


class OptimizationStage(ABC):
    """Abstract base class for an optimization stage."""
    def __init__(self, optimizer_instance: 'GeodesicOptimizer', stage_name: str, max_iters: int):
        self.optimizer = optimizer_instance
        self.stage_name = stage_name
        self.max_iters = max_iters

    @abstractmethod
    def execute(self) -> None:
        """Executes the optimization stage."""
        pass

class StageConvergence(Exception):
    """Custom exception used to signal convergence from within the ASE observer."""
    pass

class FIRE_Stage(OptimizationStage):
    """A concrete optimization stage using the ASE FIRE optimizer."""
    def __init__(
        self,
        optimizer_instance: 'GeodesicOptimizer',
        stage_name: str,
        beta_for_loss: float,
        max_iters: int,
        enable_refinement: bool,
        enable_climbing: bool,
        apply_convergence_checks: bool
    ):
        super().__init__(optimizer_instance, stage_name, max_iters)
        self.beta_for_loss = beta_for_loss
        self.enable_refinement = enable_refinement
        self.enable_climbing = enable_climbing
        self.apply_convergence_checks = apply_convergence_checks
        
        conv_config = self.optimizer.config
        self._geolen_history = deque(maxlen=conv_config.fire_conv_window)
        self._erelpeak_fwd_history = deque(maxlen=conv_config.fire_conv_window)
        self._erelpeak_back_history = deque(maxlen=conv_config.fire_conv_window)

    def _observer(self, calculator: _GeodesicCalculator):
        """Callback function executed by ASE at each optimization step."""
        obs_data = calculator.observer_data
        self.optimizer.report_step(
            self.stage_name, calculator.step,
            obs_data['E_main'], obs_data['L_tensor'],
            obs_data['J_penalty'], obs_data['grad_norm']
        )
        
        if not self.apply_convergence_checks:
            return

        E_main, L_tensor = obs_data['E_main'], obs_data['L_tensor']
        e0, eN = E_main[0].item(), E_main[-1].item()
        e_peak = E_main.max().item()
        
        self._geolen_history.append(L_tensor.sum().item() if L_tensor.numel() > 0 else 0.0)
        self._erelpeak_fwd_history.append(e_peak - e0)
        self._erelpeak_back_history.append(e_peak - eN)
        
        conv_config = self.optimizer.config
        if len(self._geolen_history) == conv_config.fire_conv_window:
            geolen_span = max(self._geolen_history) - min(self._geolen_history)
            fwd_span = max(self._erelpeak_fwd_history) - min(self._erelpeak_fwd_history)
            back_span = max(self._erelpeak_back_history) - min(self._erelpeak_back_history)

            if (geolen_span < conv_config.fire_conv_geolen_tol and
                fwd_span < conv_config.fire_conv_erelpeak_tol and
                back_span < conv_config.fire_conv_erelpeak_tol):
                log.info(f"{self.stage_name} converged: Path and barriers are stable.")
                raise StageConvergence("Stage convergence criteria met.")

    def execute(self) -> None:
        """
        Runs the optimization loop for this stage using ASE's FIRE optimizer.
        """
        log.info("--- Starting Stage: %s (Max iters: %d, Beta: %.2e, Climbing: %s) ---",
                 self.stage_name, self.max_iters, self.beta_for_loss, self.enable_climbing)
        self.optimizer._emit_status(
            f"{self.stage_name} started (max_iters={self.max_iters}, climbing={self.enable_climbing})"
        )

        total_iters_performed = 0
        
        while total_iters_performed < self.max_iters:
            if self.optimizer.Rint.numel() == 0:
                log.info(f"{self.stage_name}: No intermediate nodes to optimize. Skipping stage.")
                return

            num_intermediate_nodes = self.optimizer.Rint.shape[0]
            single_node_symbols = self.optimizer._template_atoms.get_chemical_symbols()
            super_molecule_symbols = single_node_symbols * num_intermediate_nodes
            initial_coords_flat = self.optimizer.Rint.data.view(-1, 3).cpu().numpy()
            
            atoms = Atoms(symbols=super_molecule_symbols, positions=initial_coords_flat)
            calculator = _GeodesicCalculator(self.optimizer, self, total_iters_performed)
            atoms.calc = calculator
            
            fire_optimizer = FIRE(atoms, logfile=None)
            fire_optimizer.attach(lambda: self._observer(calculator), interval=1)
            
            conv_config = self.optimizer.config
            log.info(f"Starting FIRE optimization with fmax = {conv_config.fire_grad_tol:.2e} and max_steps = {self.max_iters - total_iters_performed}")
            
            try:
                fire_optimizer.run(fmax=conv_config.fire_grad_tol, steps=self.max_iters - total_iters_performed)
                log.info("FIRE finished: fmax or max_steps reached.")
                self.optimizer._emit_status(f"{self.stage_name} reached FIRE stopping criteria")
                total_iters_performed = calculator.step
                break 
            except RuntimeError as e:
                if calculator.structure_changed_on_last_step:
                    log.info(f"Restarting FIRE stage due to path refinement. (Details: {e})")
                    self.optimizer._emit_status(f"{self.stage_name} restarting after path refinement")
                    total_iters_performed = calculator.step
                    continue 
                else:
                    log.error(f"An unexpected error occurred during FIRE: {e}")
                    raise 
            except StageConvergence as e:
                log.info(f"FIRE finished: {e}")
                self.optimizer._emit_status(f"{self.stage_name} converged")
                total_iters_performed = calculator.step
                break 
        
        final_grad_norm = calculator.observer_data.get('grad_norm', float('inf'))

        with torch.no_grad():
            final_coords_tensor = torch.as_tensor(atoms.get_positions(), device=self.optimizer.device, dtype=self.optimizer.dtype)
            if final_coords_tensor.view(*self.optimizer.Rint.data.shape).shape == self.optimizer.Rint.data.shape:
                 self.optimizer.Rint.data.copy_(final_coords_tensor.view(*self.optimizer.Rint.data.shape))
            else:
                 log.warning("Could not update Rint tensor post-FIRE; final shape mismatch.")

        log.info("--- Stage: %s finished after %d total iterations. Final Grad Norm: %.2e ---",
                 self.stage_name, total_iters_performed, final_grad_norm)
        self.optimizer._emit_status(
            f"{self.stage_name} finished after {total_iters_performed} iterations"
        )
