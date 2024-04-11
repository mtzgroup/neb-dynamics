from dataclasses import dataclass, field
from neb_dynamics.Node import Node
from neb_dynamics.nodes.Node3D import Node3D
from neb_dynamics.ChainBiaser import ChainBiaser
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

@dataclass
class NEBInputs:
    # neb params
    tol: float = 0.001*BOHR_TO_ANGSTROMS
    climb: bool = False
    en_thre: float = None
    rms_grad_thre: float = None
    max_rms_grad_thre: float = None
    
    grad_thre: float = None
    ts_grad_thre: float = None
    barrier_thre: float = 0.1 # kcal/mol
    
    early_stop_force_thre: float = 0.0
    early_stop_chain_rms_thre: float = 0.0
    early_stop_corr_thre: float = 10.
    early_stop_still_steps_thre: int = 100
    
    negative_steps_thre: int = 10
    
    max_steps: float = 1000
    
    v: bool = False
    
    preopt_with_xtb: bool = False
    
    # im sorry god
    _use_dlf_conv: bool = False

    def __post_init__(self):
        if self.en_thre is None:
            if self._use_dlf_conv:
                self.en_thre = self.tol
            else:
                self.en_thre = self.tol / 450
                
        if self.rms_grad_thre is None:
            self.rms_grad_thre = self.tol * 2

        if self.grad_thre is None:
            self.grad_thre = self.tol
            
        if self.ts_grad_thre is None:
            self.ts_grad_thre = self.tol*5
            
        if self.max_rms_grad_thre is None:
            self.max_rms_grad_thre = self.tol*5
            
            
            
    
    def copy(self):
        return NEBInputs(**self.__dict__)
            
            


@dataclass
class ChainInputs:
    k: float = 0.1
    delta_k: float = 0.0
    
    node_class: Node = Node3D
    do_local_xtb: bool = True
    do_parallel: bool = True
    use_geodesic_interpolation: bool = True
    use_maxima_recyling: bool = False
    friction_optimal_gi: bool = True
    
    do_chain_biasing: bool = False
    cb: ChainBiaser = None
    
    node_freezing: bool = False
    node_conf_barrier_thre: float = 5 # kcal/mol
    node_conf_en_thre: float = 0.5 # kcal/mol
    
    def __post_init__(self):
        if self.do_chain_biasing and self.cb is None:
            raise ValueError("No chain biaser was inputted. Fix this or set 'do_chain_biasing' to False.")
    
    def copy(self):
        return ChainInputs(**self.__dict__)
    


@dataclass
class GIInputs:
    nimages: int = 15
    friction: float = 0.01
    nudge: float = 0.001
    extra_kwds: dict = field(default_factory=dict)
    
