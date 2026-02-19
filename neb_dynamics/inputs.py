from __future__ import annotations
import shutil
from qcio import ProgramArgs
from types import SimpleNamespace
from dataclasses import dataclass, field

from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics.optimizers.cg import ConjugateGradient
from neb_dynamics.optimizers.lbfgs import LBFGS
from neb_dynamics.optimizers.adam import AdamOptimizer
import tomli
import tomli_w
from pathlib import Path


@dataclass
class PathMinInputs:
    keywords: dict = field(default_factory=dict)


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

    climb: bool = False
    en_thre: float = None
    rms_grad_thre: float = None
    max_rms_grad_thre: float = None
    skip_identical_graphs: bool = True

    ts_grad_thre: float = None
    ts_spring_thre: float = None
    barrier_thre: float = .1  # kcal/mol

    early_stop_force_thre: float = 0.0

    negative_steps_thre: int = 5
    positive_steps_thre: int = 10
    use_geodesic_tangent: bool = False
    do_elem_step_checks: bool = False

    max_steps: float = 500

    v: bool = False

    def __post_init__(self):

        if self.en_thre is None:
            self.en_thre = 1e-4

        if self.rms_grad_thre is None:
            self.rms_grad_thre = 0.02

        if self.ts_grad_thre is None:
            self.ts_grad_thre = 0.05

        if self.ts_spring_thre is None:
            self.ts_spring_thre = 0.02

        if self.max_rms_grad_thre is None:
            self.max_rms_grad_thre = 0.05

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
    delta_k: float = 0.09

    do_parallel: bool = True
    use_geodesic_interpolation: bool = True
    friction_optimal_gi: bool = True

    node_freezing: bool = True

    node_rms_thre: float = 5.0  # Bohr
    node_ene_thre: float = 5.0  # kcal/mol
    frozen_atom_indices: str = ""

    def _post_init__(self):
        if len(self.frozen_atom_indices) > 0:
            self.frozen_atom_indices = [
                int(x) for x in self.frozen_atom_indices.split(" ")]

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

    `nudge`: value for nudge parameter. (default: 0.1)

    `extra_kwds`: dictionary containing other keywords geodesic interpolation might use.

    !Protip: run multiple geodesic interpolations with high nudge values and select the path
    with the shortest length.
    """

    nimages: int = 10
    friction: float = 0.001
    nudge: float = 0.1
    extra_kwds: dict = field(default_factory=dict)
    align: bool = True

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
class RunInputs:
    engine_name: str = "chemcloud"
    program: str = "xtb"

    path_min_method: str = 'NEB'
    path_min_inputs: dict = None

    chain_inputs: dict = None
    gi_inputs: dict = None

    program_kwds: ProgramArgs = None
    optimizer_kwds: dict = None

    def __post_init__(self):

        if self.path_min_method.upper() == "NEB":
            default_kwds = NEBInputs().__dict__

        elif self.path_min_method.upper() == "FNEB":
            default_kwds = {
                "max_min_iter": 100,
                "max_grow_iter": 20,
                "verbosity": 1,
                "skip_identical_graphs": True,
                "do_elem_step_checks": True,
                "grad_tol": 0.05,  # Hartree/Bohr,
                "barrier_thre": 5,  # kcal/mol,
                "tangent": 'geodesic',
                "tangent_alpha": 1.0,  # mixing coefficient for tangents,
                "use_xtb_grow": True,
                "distance_metric": "GEODESIC",
                "min_images": 10,
                "todd_way": True,
                "dist_err": 0.1,

            }
        #     default_kwds = FSMInputs()
        # elif self.path_min_method.upper() == "PYGSM":
        #     default_kwds = PYGSMInputs()

        if self.path_min_method.upper() == 'MLPGI':
            default_kwds = {}

        if self.path_min_inputs is None:
            self.path_min_inputs = SimpleNamespace(**default_kwds)

        else:
            for key, val in self.path_min_inputs.items():
                default_kwds[key] = val

            self.path_min_inputs = SimpleNamespace(**default_kwds)

        if self.gi_inputs is None:
            self.gi_inputs = GIInputs()
        else:
            self.gi_inputs = GIInputs(**self.gi_inputs)

        if self.program_kwds is None:
            if self.program == "xtb":
                if shutil.which("crest") is not None:
                    self.program = 'crest'
                    program_args = ProgramArgs(
                        model={"method": "gfn2",
                               "basis": "gfn2"},
                        keywords={"threads": 1})
                else:
                    program_args = ProgramArgs(
                        model={"method": "GFN2xTB", "basis": "GFN2xTB"},
                        keywords={})

            elif "terachem" in self.program:
                program_args = ProgramArgs(
                    model={"method": "ub3lyp", "basis": "3-21g"},
                    keywords={})
            else:
                raise ValueError("Need to specify program arguments")

            if self.engine_name in ['qcop', 'chemcloud']:
                self.program_kwds = program_args
        elif self.program_kwds is not None and self.engine_name in ['qcop', 'chemcloud']:
            program_args = ProgramArgs(**self.program_kwds)
            self.program_kwds = program_args

        if self.chain_inputs is None:
            self.chain_inputs = ChainInputs()

        else:
            self.chain_inputs = ChainInputs(**self.chain_inputs)

        if self.optimizer_kwds is None:
            self.optimizer_kwds = {"name": "cg", "timestep": 0.5}
        elif "name" not in self.optimizer_kwds:
            self.optimizer_kwds["name"] = "cg"

        if self.engine_name == 'qcop' or self.engine_name == 'chemcloud':
            from neb_dynamics.engines.qcop import QCOPEngine
            eng = QCOPEngine(program_args=self.program_kwds,
                             program=self.program,
                             compute_program=self.engine_name
                             )
        elif self.engine_name == 'ase':
            from neb_dynamics.engines.ase import ASEEngine
            ase_progs = ['omol25']
            assert self.program in ase_progs, f"{self.program} not yet supported with ASEEngine. Use one of {ase_progs} instead."
            if self.program == 'omol25':
                from fairchem.core import pretrained_mlip, FAIRChemCalculator
                predictor = pretrained_mlip.load_predict_unit(
                    "/home/diptarka/fairchem/esen_sm_conserving_all.pt", device="cuda")
                calc = FAIRChemCalculator(predictor, task_name="omol")
            else:
                raise ValueError(f"Unsupported program: {self.program}")
            eng = ASEEngine(calculator=calc)
        else:
            raise ValueError(f"Unsupported engine: {self.engine_name}")

        self.engine = eng
        optimizer_kwds = dict(self.optimizer_kwds)
        optimizer_name = optimizer_kwds.pop("name").lower()
        optimizer_map = {
            "cg": ConjugateGradient,
            "conjugate_gradient": ConjugateGradient,
            "vpo": VelocityProjectedOptimizer,
            "velocity_projected": VelocityProjectedOptimizer,
            "lbfgs": LBFGS,
            "adam": AdamOptimizer,
        }
        if optimizer_name not in optimizer_map:
            available = ", ".join(sorted(set(optimizer_map.keys())))
            raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported values: {available}")
        self.optimizer = optimizer_map[optimizer_name](**optimizer_kwds)

    @classmethod
    def open(cls, fp):

        with open(fp, 'rb') as f:
            data = tomli.load(f)
        # data_dict = json.loads(data)
        obj = cls(**data)
        if hasattr(obj.program_kwds, 'files') and obj.program_kwds.files is not None:
            file_keys = obj.program_kwds.files.keys()
            if "ca0" in file_keys and "cb0" in file_keys:
                obj.program_kwds.files['ca0'] = Path(
                    obj.program_kwds.files['ca0']).read_bytes()
                obj.program_kwds.files['cb0'] = Path(
                    obj.program_kwds.files['cb0']).read_bytes()

        return obj

    def save(self, fp):
        json_dict = self.__dict__.copy()
        del json_dict['engine']
        del json_dict['optimizer']
        for key, val in json_dict.items():
            if 'input' in key:
                json_dict[key] = val.__dict__
            elif 'program_kwds' in key:
                d = val.json()

                if d != None:
                    d = d.replace("null", "None")
                    json_dict[key] = eval(d)
                else:
                    d = ""
                    json_dict[key] = d

        with open(fp, "w+") as f:
            # json.dump(json_dict, f)
            f.write(tomli_w.dumps(json_dict))
