# mlp_geodesic/mlp_tools.py
"""
Tools for loading MLP models and evaluating energies/forces.

This module provides a unified interface for interacting with different
machine-learned potential backends like MACE and FAIRChem.
"""
import logging
log = logging.getLogger("geodesic")

from pathlib import Path
from typing import Tuple, Any, Callable
import os

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
   FAIRChemCalculator = None
   load_predict_unit = None
   data_list_collater = None
   log.info("FAIRChem not found: Can't use FAIRChem MLPs.")

try:
   from fairchem.core.pretrained_mlip import get_predict_unit, available_models
except ModuleNotFoundError:
   get_predict_unit = None
   available_models = ()

from .utils import patched_compute_forces, create_frames_from_coords

log = logging.getLogger("geodesic")


def resolve_fairchem_model_path(
    model_path: str,
    auto_download: bool = False,
    model_repo: str = "facebook/OMol25",
    cache_dir: str | None = None,
    hf_token: str | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> str:
    """
    Resolve fairchem checkpoint path.

    If `model_path` exists locally, use it. Otherwise optionally download
    `filename = basename(model_path)` from a Hugging Face repo.
    """
    p = Path(model_path).expanduser()
    if p.exists():
        if not p.is_file():
            raise ValueError(f"Model path is not a file: {p}")
        if status_callback is not None:
            status_callback(f"Using local fairchem checkpoint: {p}")
        return str(p)

    if not auto_download:
        raise FileNotFoundError(
            f"Model file not found: {p}. "
            "Set `path_min_inputs.auto_download_model = true` to fetch it automatically, "
            "or provide an existing local `path_min_inputs.model_path`."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for auto-downloading fairchem checkpoints."
        ) from exc

    filename = p.name

    candidates = [filename]
    if "_" in filename:
        candidates.append(filename.replace("_", "-"))
    if "esen" in filename.lower() and "omol" not in filename.lower():
        stem = Path(filename).stem
        suffix = Path(filename).suffix or ".pt"
        candidates.extend(
            [
                f"{stem}_omol{suffix}",
                f"{stem}-omol{suffix}",
                f"{stem.replace('_', '-')}-omol{suffix}",
                f"{stem.replace('_', '-')}_omol{suffix}",
            ]
        )
    prefixed = [f"checkpoints/{c}" for c in candidates]
    candidates.extend(prefixed)
    # Preserve order while removing duplicates.
    candidates = list(dict.fromkeys(candidates))

    common_kwargs = {"repo_id": model_repo}
    if cache_dir:
        common_kwargs["cache_dir"] = cache_dir
    if hf_token:
        common_kwargs["token"] = hf_token
    elif os.environ.get("HF_TOKEN"):
        common_kwargs["token"] = os.environ["HF_TOKEN"]

    last_exc = None
    for cand in candidates:
        try:
            if status_callback is not None:
                status_callback(f"Trying checkpoint candidate: {cand}")
            downloaded = hf_hub_download(filename=cand, **common_kwargs)
            log.info(f"Downloaded fairchem checkpoint: {downloaded}")
            if status_callback is not None:
                status_callback(f"Downloaded fairchem checkpoint: {downloaded}")
            return downloaded
        except Exception as exc:
            last_exc = exc
            # Keep trying other plausible names for not-found style errors.
            if "Entry Not Found" in str(exc) or "EntryNotFound" in type(exc).__name__:
                continue
            # Network/auth errors should be surfaced immediately.
            raise

    raise FileNotFoundError(
        "Could not find checkpoint in Hugging Face repo "
        f"'{model_repo}'. Tried: {', '.join(candidates)}"
    ) from last_exc

def load_calculator(
    backend: str, model_path: str, device: str, dtype_str: str
) -> Tuple[Any, Any]:
    """
    Load a MACE, Egret, or FAIRChem model.
    """
    log.info(f"Attempting to load {backend} model from: {model_path}")
    try:
        if backend == "fairchem":
            if FAIRChemCalculator is None or load_predict_unit is None:
                raise ModuleNotFoundError(
                    "FAIRChem backend is not available in this environment."
                )
            model_file = Path(model_path)
            if model_file.exists():
                if not model_file.is_file():
                    raise ValueError(f"Model path is not a file: {model_file}")
                predictor = load_predict_unit(str(model_file), device=device)
            else:
                if get_predict_unit is None:
                    raise FileNotFoundError(
                        f"Model file not found: {model_file}. "
                        "This fairchem build does not expose pretrained model keys; "
                        "provide a local checkpoint path."
                    )
                if model_path not in available_models:
                    raise FileNotFoundError(
                        f"FAIRChem model path not found and model key '{model_path}' is unknown."
                    )
                log.info(f"Loading pretrained FAIRChem model key: {model_path}")
                predictor = get_predict_unit(model_path, device=device)
            calc = FAIRChemCalculator(predictor, task_name="omol")
            return calc, None
        elif backend in ["mace", "egret"]:
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            if not model_file.is_file():
                raise ValueError(f"Model path is not a file: {model_file}")
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
        log.exception(f"Failed to load model '{model_path}' with backend '{backend}'.")
        raise RuntimeError(f"Failed to load model {model_path} for {backend}: {e}") from e

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
    if status_callback is not None:
        status_callback(
            f"Resolving fairchem checkpoint from Hugging Face repo '{model_repo}'"
        )
