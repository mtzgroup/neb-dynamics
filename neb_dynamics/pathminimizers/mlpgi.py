from neb_dynamics.mlp_geodesic.utils import OptimizerConfig
from neb_dynamics.mlp_geodesic.optimizer import GeodesicOptimizer
from neb_dynamics.mlp_geodesic.mlp_tools import resolve_fairchem_model_path
from neb_dynamics.qcio_structure_helpers import structure_to_ase_atoms
import logging
import os
import sys
from neb_dynamics.pathminimizers.pathminimizer import PathMinimizer
from neb_dynamics.engines.engine import Engine
from neb_dynamics.elementarystep import ElemStepResults, check_if_elem_step
import numpy as np
import warnings
from neb_dynamics.scripts.progress import update_status, print_persistent
warnings.filterwarnings('ignore')

from dataclasses import dataclass, field
from neb_dynamics import Chain, RunInputs
from neb_dynamics.constants import ANGSTROM_TO_BOHR

import torch

@dataclass
class MLPGI(PathMinimizer):
    initial_chain: Chain
    engine: Engine
    parameters: object = None

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)
    geom_grad_calls_made: int = 0
    grad_calls_made: int = 0
    _verbose: bool = False

    def __post_init__(self):
        geodesic_optimizer_defaults = {
        # Backend and Device Settings
        #'backend': 'fairchem',
        #'dtype': 'float32',
        #'device': 'cuda',

        # Optimization Iterations and Tolerances
        'fire_stage1_iter': 200,
        'fire_stage2_iter': 500,
        'fire_grad_tol': 1e-2,

        # Path/Energy Convergence Parameters
        'variance_penalty_weight': 0.0433641,
        'fire_conv_window': 20,
        'fire_conv_geolen_tol': 0.0108410,
        'fire_conv_erelpeak_tol': 0.0108410,

        # Refinement Settings
        'refinement_step_interval': 10,
        'refinement_dynamic_threshold_fraction': 0.1,

        # Force/Climbing Settings
        'tangent_project': True,
        'climb': True,
        'alpha_climb': 0.5,
        }
        # 1. Define Configuration
        self.config = OptimizerConfig(**geodesic_optimizer_defaults)

        logger = logging.getLogger('geodesic')
        logger.propagate = False
        logger.setLevel(logging.WARNING)
        self._verbose = bool(getattr(self.parameters, "v", False))

    def _status(self, message: str, persistent: bool = False):
        if self._verbose:
            print(message)
            sys.stdout.flush()
            return
        if persistent:
            print_persistent(message=message)
        else:
            update_status(message)


    def optimize_chain(self) -> ElemStepResults:
        chain = self.initial_chain.copy()
        self._status("MLPGI: computing initial chain energies")
        self.engine.compute_energies(chain)
        self.grad_calls_made += len(chain)  # assuming one grad call per node
        self.chain_trajectory.append(chain)
        # convert structure
        initial_frames = [structure_to_ase_atoms(node.structure) for node in chain]

        # Allow overrides from environment for portability without changing API.
        backend = getattr(self.parameters, "backend", None) or os.environ.get(
            "NEB_MLPGI_BACKEND", "fairchem"
        )
        model_path = getattr(self.parameters, "model_path", None) or os.environ.get(
            "NEB_MLPGI_MODEL",
            "esen_sm_conserving_all.pt",
        )
        auto_download_model = getattr(
            self.parameters, "auto_download_model", None
        )
        if auto_download_model is None:
            auto_download_model = os.environ.get(
                "NEB_MLPGI_AUTO_DOWNLOAD", ""
            ).lower() in {"1", "true", "yes"}
        model_repo = getattr(self.parameters, "model_repo", None) or os.environ.get(
            "NEB_MLPGI_MODEL_REPO",
            "facebook/OMol25",
        )
        model_cache_dir = getattr(
            self.parameters, "model_cache_dir", None
        ) or os.environ.get("NEB_MLPGI_MODEL_CACHE_DIR")
        hf_token = getattr(self.parameters, "hf_token", None) or os.environ.get(
            "HF_TOKEN"
        )
        dtype_name = (
            getattr(self.parameters, "dtype", None)
            or os.environ.get("NEB_MLPGI_DTYPE", "float32")
        ).lower()
        dtype = torch.float64 if dtype_name == "float64" else torch.float32
        device = getattr(self.parameters, "device", None) or os.environ.get(
            "NEB_MLPGI_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        if device.startswith("cuda") and not torch.cuda.is_available():
            self._status("MLPGI: CUDA unavailable, using CPU")
            device = "cpu"

        if backend == "fairchem":
            self._status("MLPGI: resolving fairchem checkpoint")
            model_path = resolve_fairchem_model_path(
                model_path=model_path,
                auto_download=bool(auto_download_model),
                model_repo=model_repo,
                cache_dir=model_cache_dir,
                hf_token=hf_token,
                status_callback=self._status,
            )
        self._status(
            f"MLPGI setup: backend={backend} device={device} dtype={dtype_name}",
            persistent=True,
        )

        # 3. Initialize Optimizer
        opt = GeodesicOptimizer(
            frames=initial_frames,
            backend=backend,
            model_path=model_path,
            device=device,
            dtype=dtype,
            config=self.config,
            status_callback=self._status,
        )

        self._status("MLPGI: starting optimizer stages")
        main_coords, main_E = opt.optimize()

        new_chain = chain.copy()
        new_nodes = [chain[0].update_coords(c*ANGSTROM_TO_BOHR) for c in main_coords]
        new_chain.nodes = new_nodes
        self._status("MLPGI: evaluating optimized chain on target engine")
        self.engine.compute_energies(new_chain)
        self.grad_calls_made += len(new_chain)  # assuming one grad call per node

        self.chain_trajectory.append(new_chain)
        self.optimized = new_chain

        self._status("MLPGI: running elementary-step checks")
        elem_step_results = check_if_elem_step(
                inp_chain=new_chain, engine=self.engine)
        self.geom_grad_calls_made += elem_step_results.number_grad_calls

        self.optimized = new_chain
        self._status("MLPGI: complete", persistent=True)
        return elem_step_results


