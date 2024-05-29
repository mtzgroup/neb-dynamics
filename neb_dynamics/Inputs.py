from dataclasses import dataclass, field
from neb_dynamics.Node import Node
from neb_dynamics.nodes.Node3D import Node3D
from neb_dynamics.ChainBiaser import ChainBiaser
from neb_dynamics.constants import BOHR_TO_ANGSTROMS


@dataclass
class NEBInputs:
    # neb params
    tol: float = 0.001 * BOHR_TO_ANGSTROMS
    climb: bool = False
    en_thre: float = None
    rms_grad_thre: float = None
    max_rms_grad_thre: float = None

    grad_thre: float = None
    ts_grad_thre: float = None
    ts_spring_thre: float = None
    barrier_thre: float = 0.1  # kcal/mol

    early_stop_force_thre: float = 0.0
    early_stop_chain_rms_thre: float = 0.0
    early_stop_corr_thre: float = 10.0
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
            self.ts_grad_thre = self.tol * 5

        if self.ts_spring_thre is None:
            self.ts_spring_thre = self.tol * 5

        if self.max_rms_grad_thre is None:
            self.max_rms_grad_thre = self.tol * 5

    def copy(self):
        return NEBInputs(**self.__dict__)


@dataclass
class ChainInputs:
    """
    Object containing parameters relevant to chain.
        k: maximum spring constant.
        delta_k: parameter to use for calculating energy weighted spring constants
                see: https://pubs.acs.org/doi/full/10.1021/acs.jctc.1c00462

        node_class: type of node to use
        do_parallel: whether to compute gradients and energies in parallel
        use_geodesic_interpolation: whether to use GI in interpolations
        use_maxima_recyling: whether to use maxima recyling in early stop checks
        friction_optimal_gi: whether to optimize 'friction' parameter when running GI

        skip_identical_graphs: whether to skip chains with identical graph endpoints when
                            running NEB (***TODO: SHOULD PROBABLY BE A NEBInput***)

        do_chain_biasing: whether to use chain biasing (Under Development, not ready for use)
        cb: Chain biaser object (Under Development, not ready for use)

        node_freezing: whether to freeze nodes in NEB convergence
        node_conf_barrier_thre: threshold for pseudobarrier calculation (kcal/mol)
                                for identifying identical conformers
        node_conf_en_thre: float = threshold for energy difference (kcal/mol) of geometries
                                for indentigying identical conformers

        tc_model_method: 'method' parameter for electronic structure calculations
        tc_model_basis: 'method' parameter for electronic structure calculations
        tc_kwds: keyword arguments for electronic structure calculations
    """
    k: float = 0.1
    delta_k: float = 0.0

    node_class: Node = Node3D
    do_parallel: bool = True
    use_geodesic_interpolation: bool = True
    use_maxima_recyling: bool = False
    friction_optimal_gi: bool = True

    skip_identical_graphs: bool = True

    do_chain_biasing: bool = False
    cb: ChainBiaser = None

    node_freezing: bool = True
    node_conf_barrier_thre: float = 5  # kcal/mol
    node_conf_en_thre: float = 0.5  # kcal/mol

    tc_model_method: str = "b3lyp"
    tc_model_basis: str = "6-31g"
    tc_kwds: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.do_chain_biasing and self.cb is None:
            raise ValueError(
                "No chain biaser was inputted. Fix this or set 'do_chain_biasing' to False."
            )

    def copy(self):
        return ChainInputs(**self.__dict__)


@dataclass
class GIInputs:
    nimages: int = 15
    friction: float = 0.01
    nudge: float = 0.001
    extra_kwds: dict = field(default_factory=dict)
