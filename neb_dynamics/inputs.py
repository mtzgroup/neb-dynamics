from __future__ import annotations
from dataclasses import dataclass, field
from neb_dynamics.nodes.node import Node
from neb_dynamics.constants import BOHR_TO_ANGSTROMS


@dataclass
class NEBInputs:
    """
    Object containing inputs relating to NEB convergence.
    `tol`: tolerace for optimizations (Hartrees)

    `climb`: whether to use climbing image NEB

    `en_thre`: energy difference threshold. (default: tol/450)

    `rms_grad_thre`: RMS of perpendicular gradient threhsold (default: tol)

    `max_rms_grad_thre`: maximum(RMS) of perpedicular gradients threshold (default: tol*2.5)

    `ts_grad_thre`= infinity norm of TS node threshold (default: tol*2.5)

    `ts_spring_thre`= infinity norm of spring forces of triplet around TS node (default: tol * 1.5),

    `skip_identical_graphs`: whether to skip minimizations where endpoints have identical graphs

    `early_stop_force_thre`: infinity norm of TS node early stop check threshold \
        (default: 0.0 | i.e. no early stop check)

    `negative_steps_thre`: number of steps chain can oscillate until the step size is halved (default: 10)

    `max_steps`: maximum number of NEB steps allowed (default: 1000)

    `v`: whether to be verbose (default: True)

    `preopt_with_xtb`: whether to preconverge a chain using XTB (default: False)
    """

    tol: float = 0.002
    climb: bool = False
    en_thre: float = None
    rms_grad_thre: float = None
    max_rms_grad_thre: float = None
    skip_identical_graphs: bool = True

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
    pygsm_kwds: dict = field(default_factory=dict)
    fneb_kwds: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.en_thre is None:
            self.en_thre = self.tol / 450

        if self.rms_grad_thre is None:
            self.rms_grad_thre = self.tol

        if self.grad_thre is None:
            self.grad_thre = self.tol / 2

        if self.ts_grad_thre is None:
            self.ts_grad_thre = self.tol * 5 / 2

        if self.ts_spring_thre is None:
            self.ts_spring_thre = self.tol * 5 / 2

        if self.max_rms_grad_thre is None:
            self.max_rms_grad_thre = self.tol * 5 / 2

    def copy(self) -> NEBInputs:
        return NEBInputs(**self.__dict__)


@dataclass
class ChainInputs:
    """
    Object containing parameters relevant to chain.
    `k`: maximum spring constant.
    `delta_k`: parameter to use for calculating energy weighted spring constants
            see: https://pubs.acs.org/doi/full/10.1021/acs.jctc.1c00462

    `node_class`: type of node to use
    `do_parallel`: whether to compute gradients and energies in parallel
    `use_geodesic_interpolation`: whether to use GI in interpolations
    `friction_optimal_gi`: whether to optimize 'friction' parameter when running GI

    `do_chain_biasing`: whether to use chain biasing (Under Development, not ready for use)
    `cb`: Chain biaser object (Under Development, not ready for use)

    `node_freezing`: whether to freeze nodes in NEB convergence
    `node_conf_en_thre`: float = threshold for energy difference (kcal/mol) of geometries
                            for identifying identical conformers

    `tc_model_method`: 'method' parameter for electronic structure calculations
    `tc_model_basis`: 'method' parameter for electronic structure calculations
    `tc_kwds`: keyword arguments for electronic structure calculations
    """

    k: float = 0.1
    delta_k: float = 0.0

    node_class: Node = Node
    do_parallel: bool = True
    use_geodesic_interpolation: bool = True
    friction_optimal_gi: bool = True

    node_freezing: bool = True

    node_rms_thre: float = 1.0  # Bohr
    node_ene_thre: float = 1.0  # kcal/mol

    def copy(self) -> ChainInputs:
        return ChainInputs(**self.__dict__)


@dataclass
class GIInputs:
    """
    Inputs for geodesic interpolation. See \
        [geodesic interpolation](https://pubs.aip.org/aip/jcp/article/150/16/164103/198363/Geodesic-interpolation-for-reaction-pathways) \
            for details.

    `nimages`: number of images to use (default: 15)

    `friction`: value for friction parameter. influences the penalty for \
        pairwise distances becoming too large. (default: 0.01)

    `nudge`: value for nudge parameter. (default: 0.001)

    `extra_kwds`: dictionary containing other keywords geodesic interpolation might use.
    """

    nimages: int = 15
    friction: float = 0.01
    nudge: float = 0.001
    extra_kwds: dict = field(default_factory=dict)

    def copy(self) -> GIInputs:
        return GIInputs(**self.__dict__)


@dataclass
class NetworkInputs:
    n_max_conformers: int = 10  # maximum number of conformers to keep of each endpoint
    subsample_confs: bool = True

    conf_rmsd_cutoff: float = 0.5
    # minimum distance to be considered new conformer
    # given that the graphs are identical

    network_nodes_are_conformers: bool = False
    # whether each conformer should be a separate node in the network

    maximum_barrier_height: float = 1000  # kcal/mol
    # will only populate edges with a barrier lower than this input

    use_slurm: bool = False
    # whether to submit minimization jobs to slurm queue

    verbose: bool = True

    tolerate_kinks: bool = True
    # whether to include chains with a minimum apparently present in the
    # network construction

    CREST_temp: float = 298.15  # Kelvin
    CREST_ewin: float = 6.0  # kcal/mol
    # crest inputs for conformer generation. Incomplete list.


@dataclass
class PathMinInputs:
    keywords: dict = field(default_factory=dict)

    NEB_DEFAULTS = NEBInputs().__dict__

    def __post_init__(self):
        assert (
            "method" in self.keywords.keys()
        ), "Need to specify path minimization method in keywords"

        if self.keywords["method"].upper() == "NEB":
            neb_inputs = self.NEB_DEFAULTS.copy()
            for k, v in self.keywords.items():
                if k == "method":
                    continue
                else:
                    neb_inputs[k] = v

            for k, v in neb_inputs.items():
                setattr(self, k, v)
