# mlp_geodesic/utils.py
"""
General utility helpers, including data classes, XYZ I/O, a MACE patch,
and numerical constants.
"""
import numpy as np
import torch
from ase import Atoms
from typing import List, Sequence, Any, Optional, Dict
import logging
from dataclasses import dataclass, fields

log = logging.getLogger("geodesic")

# --- Numerical Constants ---
EPS_MACHINE_DOUBLE = np.finfo(np.float64).eps
EPS_4THRT_DOUBLE = EPS_MACHINE_DOUBLE**0.25 #For numerical stability control

# --- Data Classes ---
@dataclass
class OptimizerConfig:
    """
    A dataclass to hold all hyperparameters for the GeodesicOptimizer.
    The field names are now consistent with the cleaned-up CLI arguments.
    """
    fire_stage1_iter: int
    fire_stage2_iter: int
    fire_grad_tol: float
    variance_penalty_weight: float
    fire_conv_window: int
    fire_conv_geolen_tol: float
    fire_conv_erelpeak_tol: float
    refinement_step_interval: int
    refinement_dynamic_threshold_fraction: float
    tangent_project: bool
    climb: bool
    alpha_climb: float

    @classmethod
    def from_cli_kwargs(cls, kwargs: Dict[str, Any]) -> 'OptimizerConfig':
        """
        Creates an OptimizerConfig instance from a dictionary of keyword arguments,
        typically from a CLI parser like click. It filters the dictionary to
        only include keys that are fields of this dataclass. This version is
        now simple and correct.
        """
        config_keys = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in config_keys}
        return cls(**filtered_kwargs)

@dataclass
class PathData:
    """A container for all data related to a reaction path at a given step."""
    nodes: torch.Tensor
    energies: torch.Tensor
    forces: torch.Tensor
    midpoint_energies: Optional[torch.Tensor] = None
    midpoint_forces: Optional[torch.Tensor] = None

# --- I/O & Structure Functions ---
def write_xyz_with_energies(frames: List[Atoms], energies: Sequence[float], out_path: str) -> None:
    """Writes a list of ASE Atoms objects to an XYZ file with energies."""
    if len(frames) != len(energies):
        raise ValueError(f"Frames count ({len(frames)}) must match energies count ({len(energies)}).")
    try:
        with open(out_path, "w") as f:
            for i, frame in enumerate(frames):
                f.write(f"{len(frame)}\n")
                f.write(f"Energy: {energies[i]:.6f} eV\n")
                for sym, pos in zip(frame.get_chemical_symbols(), frame.positions):
                    f.write(f"{sym:2s} {pos[0]: 15.8f} {pos[1]: 15.8f} {pos[2]: 15.8f}\n")
    except IOError as e:
        log.error(f"Failed to write XYZ file to {out_path}")
        raise IOError(f"Failed to write XYZ file to {out_path}: {e}") from e

def create_frames_from_coords(coords: np.ndarray, template_atoms: Atoms) -> List[Atoms]:
    """
    Creates a list of ASE Atoms objects from a numpy array of coordinates.
    """
    template_atoms.calc = None
    frames = []
    for i in range(coords.shape[0]):
        frame = template_atoms.copy()
        frame.positions = coords[i]
        frames.append(frame)
    return frames

# --- MACE Patch ---
def patched_compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = False, **kwargs
) -> torch.Tensor:
    """Overrides MACE's default force computation using a more robust autograd call."""
    if not positions.requires_grad:
        positions.requires_grad_(True)
    grad_outputs = torch.ones_like(energy)
    grad = torch.autograd.grad(
        outputs=energy, inputs=positions, grad_outputs=grad_outputs,
        retain_graph=training, create_graph=training, allow_unused=False
    )[0]
    if grad is None:
        raise RuntimeError("Autograd returned None for position gradients.")
    return -grad

