# mlp_geodesic/mlp_tools.py
"""
Tools for loading MLP models and evaluating energies/forces.

This module provides a unified interface for interacting with different
machine-learned potential backends like MACE and FAIRChem.
"""
import logging
log = logging.getLogger("geodesic")

from pathlib import Path
from typing import Tuple, Any

import torch
import numpy as np
from ase import Atoms

try:
   from mace.calculators import mace_off
   from mace import data as mace_data
   from mace.tools import torch_geometric
except ModuleNotFoundError:
   log.info("MACE not found: Can't use MACE MLPs.")

try:
   from fairchem.core import FAIRChemCalculator
   from fairchem.core.units.mlip_unit import load_predict_unit
   from fairchem.core.datasets import data_list_collater
except ModuleNotFoundError:
   log.info("FAIRChem not found: Can't use FAIRChem MLPs.")

from .utils import patched_compute_forces, create_frames_from_coords

log = logging.getLogger("geodesic")

def load_calculator(
    backend: str, model_path: str, device: str, dtype_str: str
) -> Tuple[Any, Any]:
    """
    Load a MACE, Egret, or FAIRChem model.
    """
    log.info(f"Attempting to load {backend} model from: {model_path}")
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not model_file.is_file():
        raise ValueError(f"Model path is not a file: {model_file}")

    try:
        if backend == "fairchem":
            predictor = load_predict_unit(str(model_file), device=device)
            calc = FAIRChemCalculator(predictor, task_name="omol")
            return calc, None
        elif backend in ["mace", "egret"]:
            kwargs = {"model": str(model_file), "device": device, "default_dtype": dtype_str}
            calc = mace_off(**kwargs)
            if not hasattr(calc, 'models') or not calc.models:
                raise RuntimeError(f"No model loaded into MACE/Egret calculator: {model_file}.")
            raw_module = calc.models[0].eval()

            if backend == "mace":
                try:
                    import mace.modules.utils as mace_internal_utils
                    if hasattr(mace_internal_utils, 'compute_forces') and \
                       mace_internal_utils.compute_forces is not patched_compute_forces:
                        log.warning("Patching mace.modules.utils.compute_forces.")
                        mace_internal_utils.compute_forces = patched_compute_forces
                    elif not hasattr(mace_internal_utils, 'compute_forces'):
                         log.error("Cannot patch MACE: mace.modules.utils.compute_forces not found.")
                except ImportError:
                    log.error("Cannot patch MACE: Failed to import mace.modules.utils.")
            return calc, raw_module
        else:
            raise ValueError(f"Unknown backend: {backend}.")
    except Exception as e:
        log.exception(f"Failed to load model '{model_file}' with backend '{backend}'.")
        raise RuntimeError(f"Failed to load model {model_file} for {backend}: {e}") from e

def evaluate_mlp_batch(
    nodes_for_eval: torch.Tensor, calc: Any, raw_module: Any,
    template_atoms: Atoms, Z_tensor: torch.Tensor,
    device: torch.device, dtype: torch.dtype, backend_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates energies and forces for a batch of geometries.
    """
    bsz = nodes_for_eval.size(0)
    if bsz == 0:
        return torch.empty(0, device=device, dtype=dtype), \
               torch.empty((0, Z_tensor.numel() * 3), device=device, dtype=dtype)

    eval_coords = nodes_for_eval.detach().to(device=device, dtype=dtype)

    if backend_name == 'fairchem':
        atoms_list = create_frames_from_coords(eval_coords.cpu().numpy(), template_atoms)

        for atm in atoms_list:
            if hasattr(calc, '_validate_charge_and_spin'):
                calc._validate_charge_and_spin(atm)

        data_list = [calc.a2g(atoms) for atoms in atoms_list]
        batch = data_list_collater(data_list, otf_graph=getattr(calc, 'otf_graph', False)).to(device)
        pred = calc.predictor.predict(batch)

        E_batch = pred['energy'].detach().to(device=device, dtype=dtype).view(-1)
        F_batch_flat = pred['forces'].detach().to(device=device, dtype=dtype).view(bsz, -1)
        return E_batch, F_batch_flat

    elif backend_name in ['mace', 'egret']:
        if raw_module is None:
            log.warning(f"Raw module not available for {backend_name}, using slower ASE calculator fallback.")
            energies_list, forces_list = [], []
            atoms_list = create_frames_from_coords(eval_coords.cpu().numpy(), template_atoms)
            for at_i in atoms_list:
                at_i.calc = calc
                energies_list.append(at_i.get_potential_energy())
                forces_list.append(at_i.get_forces().flatten())
            return torch.tensor(energies_list, device=device, dtype=dtype), \
                   torch.tensor(np.array(forces_list), device=device, dtype=dtype)

        atoms_list = create_frames_from_coords(eval_coords.cpu().numpy(), template_atoms)
        datas = []
        for at in atoms_list:
            keyspec_arrays_keys: dict = {}
            if hasattr(calc, 'charges_key') and calc.charges_key is not None:
                keyspec_arrays_keys["charges"] = calc.charges_key
            key_spec = mace_data.KeySpecification(info_keys={}, arrays_keys=keyspec_arrays_keys)
            config = mace_data.config_from_atoms(at, key_specification=key_spec, head_name=getattr(calc, 'head', None))
            atomic_data = mace_data.AtomicData.from_config(
                config, z_table=calc.z_table, cutoff=calc.r_max, heads=getattr(calc, 'available_heads', None)
            )
            datas.append(atomic_data)

        loader = torch_geometric.dataloader.DataLoader(dataset=datas, batch_size=bsz, shuffle=False, drop_last=False)
        batch_data = next(iter(loader)).to(device)
        output = raw_module(batch_data.to_dict(), training=True, compute_stress=False, compute_force=True)

        return output['energy'].to(dtype=dtype, device=device).view(bsz), \
               output['forces'].to(dtype=dtype, device=device).view(bsz, -1)
    else:
        raise ValueError(f"Unsupported backend for batch evaluation: {backend_name}")

