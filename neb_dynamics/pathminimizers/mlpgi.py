from neb_dynamics.mlp_geodesic.utils import OptimizerConfig
from neb_dynamics.mlp_geodesic.optimizer import GeodesicOptimizer

from neb_dynamics.qcio_structure_helpers import structure_to_ase_atoms

import logging
from neb_dynamics.pathminimizers.pathminimizer import PathMinimizer
from neb_dynamics.engines.engine import Engine
from neb_dynamics.elementarystep import ElemStepResults, check_if_elem_step

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass, field
from neb_dynamics import Chain, RunInputs
from neb_dynamics.constants import ANGSTROM_TO_BOHR

import torch

@dataclass
class MLPGI(PathMinimizer):
    initial_chain: Chain
    engine: Engine

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)
    geom_grad_calls_made: int = 0
    grad_calls_made: int = 0

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
# now if you use logger it will not log to console.


    def optimize_chain(self) -> ElemStepResults:
        chain = self.initial_chain.copy()
        self.engine.compute_energies(chain)
        self.grad_calls_made += len(chain)  # assuming one grad call per node
        self.chain_trajectory.append(chain)
        # convert structure
        initial_frames = [structure_to_ase_atoms(node.structure) for node in chain]

        # 3. Initialize Optimizer
        opt = GeodesicOptimizer(
            frames=initial_frames,
            backend='fairchem',
            model_path="/home/diptarka/fairchem/esen_sm_conserving_all.pt",
            device='cuda',  # Or 'cuda'
            dtype=torch.float32,
            config=self.config
        )

        main_coords, main_E = opt.optimize()

        new_chain = chain.copy()
        new_nodes = [chain[0].update_coords(c*ANGSTROM_TO_BOHR) for c in main_coords]
        new_chain.nodes = new_nodes
        self.engine.compute_energies(new_chain)
        self.grad_calls_made += len(new_chain)  # assuming one grad call per node

        self.chain_trajectory.append(new_chain)
        self.optimized = new_chain

        elem_step_results = check_if_elem_step(
                inp_chain=new_chain, engine=self.engine)
        self.geom_grad_calls_made += elem_step_results.number_grad_calls

        self.optimized = new_chain
        return elem_step_results




