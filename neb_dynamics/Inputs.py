from dataclasses import dataclass, field
from neb_dynamics.Node import Node
from neb_dynamics.nodes.Node3D import Node3D
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
    do_bfgs: bool = True

    vv_force_thre: float = 0.0
    v: bool = False
    
    bfgs_flush_steps: int = None
    bfgs_flush_thre: float = 0.98

    def __post_init__(self):
        if self.en_thre is None:
            self.en_thre = self.tol / 450

        if self.rms_grad_thre is None:
            self.rms_grad_thre = self.tol * (2 / 3)

        if self.grad_thre is None:
            self.grad_thre = self.tol
            
        if self.bfgs_flush_steps is None:
            self.bfgs_flush_steps = self.max_steps
            
            


@dataclass
class ChainInputs:
    k: float = 0.1
    delta_k: float = 0.0
    
    node_class: Node = Node3D
    do_local_xtb: bool = True
    do_parallel: bool = True
    use_geodesic_interpolation: bool = True
    friction_optimal_gi: bool = False
    
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
    
