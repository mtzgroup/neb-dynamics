from dataclasses import dataclass, field
from neb_dynamics.Node import Node
from neb_dynamics.Node3D import Node3D
from neb_dynamics.ChainBiaser import ChainBiaser


@dataclass
class NEBInputs:
    # neb params
    tol: float = 0.01
    climb: bool = False
    en_thre: float = None
    rms_grad_thre: float = None
    grad_thre: float = None
    max_steps: float = 1000
    early_stop_force_thre: float = 0.0
    early_stop_chain_rms_thre: float = 0.0
    early_stop_corr_thre: float = 10.
    early_stop_still_steps_thre: int = 20

    vv_force_thre: float = 0.0
    v: bool = False

    def __post_init__(self):
        if self.en_thre is None:
            self.en_thre = self.tol / 450

        if self.rms_grad_thre is None:
            self.rms_grad_thre = self.tol * (2 / 3)

        if self.grad_thre is None:
            self.grad_thre = self.tol
            
            


@dataclass
class ChainInputs:
    k: float = 0.1
    delta_k: float = 0.0
    step_size: float = 1.0
    min_step_size: float = 0.33
    node_class: Node = Node3D
    do_local_xtb: bool = True
    do_parallel: bool = True
    use_geodesic_interpolation: bool = True
    friction_optimal_gi: bool = False
    als_max_steps: int = 3
    do_chain_biasing: bool = False
    cb: ChainBiaser = None
    
    node_freezing: bool = False
    
    def __post_init__(self):
        if self.do_chain_biasing and self.cb is None:
            raise ValueError("No chain biaser was inputted. Fix this or set 'do_chain_biasing' to False.")
    
    


@dataclass
class GIInputs:
    nimages: int = 15
    friction: float = 0.01
    nudge: float = 0.001
    extra_kwds: dict = field(default_factory=dict)
    
