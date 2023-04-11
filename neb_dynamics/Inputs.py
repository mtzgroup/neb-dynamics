from dataclasses import dataclass, field
from neb_dynamics.Node import Node
from neb_dynamics.Node3D import Node3D


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
    node_freezing: bool = False

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
    node_class: Node = Node3D
    do_local_xtb: bool = True
    do_parallel: bool = True
    use_geodesic_interpolation: bool = True


@dataclass
class GIInputs:
    nimages: int = 15
    friction: float = 0.01
    nudge: float = 0.001
    extra_kwds: dict = field(default_factory=dict)
    
