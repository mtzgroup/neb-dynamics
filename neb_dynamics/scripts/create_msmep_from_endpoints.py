import sys
from argparse import ArgumentParser
from pathlib import Path

from neb_dynamics.nodes.node import StructureNode
from qcio import Structure, ProgramInput

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.Janitor import Janitor
from neb_dynamics.msmep import MSMEP
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics.engines import QCOPEngine, ASEEngine
from neb_dynamics.helper_functions import _load_info_from_tcin
import time


def read_single_arguments():
    """
    Command line reader
    """
    description_string = "will take path to an xyz file of geodesic \
        trajectory and relax it using XTB neb"
    parser = ArgumentParser(description=description_string)

    parser.add_argument(
        "-st",
        "--start",
        dest="st",
        required=True,
        type=str,
        help="path to the first xyz structure",
    )

    parser.add_argument(
        "-en",
        "--end",
        dest="en",
        required=True,
        type=str,
        help="path to the final xyz structure",
    )

    parser.add_argument(
        "-met",
        "--method",
        dest="method",
        required=False,
        type=str,
        help="what type of NEB to use. options: [neb, fneb, pygsm, asneb, asfneb, aspygsm]",
        default="asneb",
    )

    parser.add_argument(
        "-eng",
        "--engine",
        dest="eng",
        required=True,
        type=str,
        help="What electornic structure engine to use. E.g. 'qcop', 'ase', or 'chemcloud'",
    )

    parser.add_argument(
        "-prog",
        "--program",
        dest="program",
        required=True,
        type=str,
        help="what electronic structure program to use",
        default="xtb",
    )

    parser.add_argument(
        "-es_ft",
        "--early_stop_ft",
        dest="es_ft",
        required=False,
        type=float,
        help="force threshold for early stopping",
        default=0.03,
    )

    parser.add_argument(
        "-c", "--charge", dest="c", type=int, default=0, help="total charge of system"
    )

    parser.add_argument(
        "-nimg",
        "--nimages",
        dest="nimg",
        type=int,
        default=8,
        help="number of images in the chain",
    )

    parser.add_argument(
        "-s",
        "--spinmult",
        dest="s",
        type=int,
        default=1,
        help="total spinmultiplicity of system",
    )

    parser.add_argument(
        "-dc",
        "--do_cleanup",
        dest="dc",
        type=int,
        default=0,
        help="whether to do conformer-conformer NEBs at the end",
    )

    parser.add_argument(
        "-nc",
        "--node_class",
        dest="nc",
        type=str,
        default="node3d",
        help="what node type to use. options are: node3d, \
            node3d_tc, node3d_tc_local, node3d_tcpb",
    )

    parser.add_argument(
        "-tol",
        "--tolerance",
        dest="tol",
        required=False,
        type=float,
        help="convergence threshold in H/Bohr^2",
        default=0.002,
    )

    parser.add_argument(
        "-sig",
        "--skip_identical_graphs",
        dest="sig",
        required=False,
        type=int,
        help="whether to skip optimizations for identical graph endpoints",
        default=0,
    )

    parser.add_argument(
        "-min_ends",
        "--minimize_endpoints",
        dest="min_ends",
        required=False,
        type=int,
        help="whether to minimize the endpoints before starting",
        default=0,
    )

    parser.add_argument(
        "-name",
        "--file_name",
        dest="name",
        required=False,
        type=str,
        help="name of folder to output to || defaults to react_msmep",
        default=None,
    )

    parser.add_argument(
        "-preopt",
        "--preopt_with_xtb",
        dest="preopt",
        required=False,
        type=int,
        help="whether to preconverge chains using xtb",
        default=0,
    )

    parser.add_argument(
        "-delk",
        "--delta_k",
        dest="delk",
        required=False,
        type=float,
        help="parameter for the delta k",
        default=0.09,
    )

    parser.add_argument(
        "-pif",
        "--program_input_file",
        dest="pif",
        required=False,
        type=str,
        help="file path to a ProgramInput file from qcio.",
        default=None,
    )

    parser.add_argument(
        "-tcin",
        "--terachem_input",
        dest="tcin",
        required=False,
        type=str,
        help="file path to a Terachem input file.",
        default=None,
    )

    parser.add_argument(
        "-geom_opt",
        "--geometry_optimization",
        dest="geom_opt",
        required=False,
        type=str,
        help="Which optimizer to use. Default is 'geometric'. When using ASE as an engine, can also use 'LBFGS','LBFGSLineSearch','BFGS','FIRE' and more.",
        default="geometric",
    )

    parser.add_argument(
        "-maxsteps",
        "--maxsteps",
        dest="maxsteps",
        required=False,
        type=int,
        help="Maximum number of optimization steps per path minimization run",
        default=500,
    )

    parser.add_argument(
        "-step",
        "--stepsize",
        dest="stepsize",
        required=False,
        type=float,
        help="default step size for optimizer",
        default=1.0,
    )

    parser.add_argument(
        "-node_rms_thre",
        "--node_rms_threshold",
        dest="node_rms_thre",
        required=False,
        type=float,
        help="default distance for nodes being identical (Bohr) if they\
            have identical graphs",
        default=1.0,
    )

    parser.add_argument(
        "-node_ene_thre",
        "--node_ene_threshold",
        dest="node_ene_thre",
        required=False,
        type=float,
        help="default energy difference for nodes being identical (kcal/mol)\
            if they have identical graphs",
        default=1.0,
    )

    parser.add_argument(
        "-fog",
        "--friction_optimal_gi",
        dest="fog",
        required=False,
        type=int,
        help="whether to use XTB to optimize friction parameter in geodesic interpolation",
        default=1,
    )

    parser.add_argument(
        "-smi",
        "--input_smiles",
        dest="smi",
        required=False,
        type=int,
        help="whether -st and -en inputs are smiles. If so, will use RXNMapper to get an atomic mapping and \
            create the structures.",
        default=False,
    )

    return parser.parse_args()


def main():
    start_time = time.time()
    args = read_single_arguments()
    nodes = {"node3d": StructureNode}
    nc = nodes[args.nc]

    if args.smi:
        from neb_dynamics.nodes.nodehelpers import create_pairs_from_smiles

        print(
            "WARNING: Using RXNMapper to create atomic mapping. Carefully check output to see how labels\
                 affected reaction path."
        )
        start, end = create_pairs_from_smiles(smi1=args.st, smi2=args.en)
    else:
        start = Structure.open(args.st)
        end = Structure.open(args.en)
    s_dict = start.model_dump()
    s_dict["charge"], s_dict["multiplicity"] = args.c, args.s
    start = Structure(**s_dict)

    e_dict = end.model_dump()
    e_dict["charge"], e_dict["multiplicity"] = args.c, args.s
    end = Structure(**e_dict)

    tol = args.tol

    fog = args.fog
    print(nodes.keys(), fog, start, end)

    cni = ChainInputs(
        k=0.1,
        delta_k=args.delk,
        node_class=nc,
        friction_optimal_gi=fog,
        node_freezing=True,
        node_rms_thre=float(args.node_rms_thre),
        node_ene_thre=float(args.node_ene_thre),
    )

    optimizer = VelocityProjectedOptimizer(timestep=args.stepsize, activation_tol=0.1)

    nbi = NEBInputs(
        tol=tol,
        barrier_thre=0.1,  # kcalmol,
        climb=0,  # not supporting this right now.
        rms_grad_thre=tol,
        max_rms_grad_thre=tol * 2.5,
        ts_grad_thre=tol * 2.5,
        ts_spring_thre=tol * 1.5,
        v=1,
        max_steps=int(args.maxsteps),
        early_stop_force_thre=args.es_ft,
        skip_identical_graphs=bool(args.sig),
        preopt_with_xtb=bool(int(args.preopt)),
        fneb_kwds={
            "stepsize": args.stepsize,
            "ngradcalls": 3,
            "max_cycles": args.maxsteps,
            "path_resolution": 1 / 20,  # BOHR,
            "max_atom_displacement": 0.1,
            "early_stop_scaling": args.es_ft,
            "dist_err": 1 / 10,
            "distance_metric": "geodesic",
            "use_geodesic_tangent": True,
            "verbosity": 1,
        },
    )

    if args.tcin:
        print(
            "Warning! Directly inserting TC input files may be deleted\
        in future versions. using --program_input_file is preferred."
        )
        method, basis, charge, spinmult, inp_kwds = _load_info_from_tcin(args.tcin)
        program_input = ProgramInput(
            structure=start,
            calctype="gradient",
            model={"method": method, "basis": basis},
            keywords=inp_kwds,
        )
    elif args.pif:
        program_input = ProgramInput.open(args.pif)
    else:
        if args.program == "xtb":
            program_input = ProgramInput(
                structure=start,
                calctype="gradient",
                model={"method": "GFN2xTB"},
                keywords={},
            )
        elif "terachem" in args.program:
            program_input = ProgramInput(
                structure=start,
                calctype="gradient",
                model={"method": "ub3lyp", "basis": "3-21g"},
                keywords={},
            )
        else:
            raise TypeError(
                f"Need to specify a program input file in -pif flag. No defaults \
                            exist for program {args.program}"
            )

    if args.eng == "qcop":
        print("Using QCOPEngine")
        eng = QCOPEngine(
            program_input=program_input,
            program=args.program,
            geometry_optimizer=args.geom_opt,
            compute_program="qcop",
        )
    elif args.eng == "chemcloud":
        print("Using QCOPEngine")
        eng = QCOPEngine(
            program_input=program_input,
            program=args.program,
            geometry_optimizer=args.geom_opt,
            compute_program="chemcloud",
        )

    elif args.eng == "ase":
        print("Using ASEEngine")
        assert (
            args.program == "xtb"
        ), f"Invalid 'progam' {args.program}. It is not yet supported. This will likely change."
        from xtb.ase.calculator import XTB

        # from xtb.utils import get_solvent, Solvent

        calc = XTB(method="GFN2-xTB", solvent="water")
        eng = ASEEngine(calculator=calc, ase_opt_str=args.geom_opt)

    else:
        raise ValueError(f"Invdalid engine: {args.eng}")

    if "fneb" in args.method:
        pmm = "fneb"

    elif "neb" in args.method:
        pmm = "neb"

    elif "pygsm" in args.method:
        print(
            "WARNING: PYGSM is *very* experimental for now. Engines other than ASEEngine not yet supported. Must manually change calculator if you want something that is not XTB. This  *will* change."
        )
        pmm = "pygsm"
        assert args.eng == "ase", "PYGMS currently only works with ASE calculator."

    if args.min_ends:
        print("Minimizing input endpoints...")
        start_tr = eng.compute_geometry_optimization(StructureNode(structure=start))
        start_node = start_tr[-1]
        end_tr = eng.compute_geometry_optimization(StructureNode(structure=end))
        end_node = end_tr[-1]
        print("Done!")
    else:
        start_node = StructureNode(structure=start)
        end_node = StructureNode(structure=end)

    print(
        f"NEBinputs: {nbi}\nChainInputs: {cni}\nOptimizer: {optimizer}\nEngine: {eng}"
    )
    print(f"{args.method=}")
    sys.stdout.flush()
    gii = GIInputs(nimages=args.nimg, extra_kwds={"sweep": False})
    chain = Chain(
        nodes=[start_node, end_node],
        parameters=cni,
    )
    m = MSMEP(
        neb_inputs=nbi,
        chain_inputs=cni,
        gi_inputs=gii,
        optimizer=optimizer,
        engine=eng,
        path_min_method=pmm,
    )

    from openbabel import openbabel

    ob_log_handler = openbabel.OBMessageHandler()
    ob_log_handler.SetOutputLevel(0)

    if "as" in args.method:
        print("RUNNING AUTOSPLITTING")

        history = m.run_recursive_minimize(chain)

        leaves_nebs = [obj for obj in history.get_optimization_history() if obj]
        fp = Path(args.st)
        data_dir = fp.parent

        if args.name:
            foldername = data_dir / args.name
            filename = data_dir / (args.name + ".xyz")

        else:
            foldername = data_dir / f"{fp.stem}_msmep"
            filename = data_dir / f"{fp.stem}_msmep.xyz"
        end_time = time.time()
        history.output_chain.write_to_disk(filename)
        history.write_to_disk(foldername)

        tot_grad_calls = sum([obj.grad_calls_made for obj in leaves_nebs])
        geom_grad_calls = sum([obj.geom_grad_calls_made for obj in leaves_nebs])
        print(f">>> Made {tot_grad_calls} gradient calls total.")
        print(
            f"<<< Made {geom_grad_calls} gradient for geometry\
               optimizations."
        )

        if args.dc:
            op = data_dir / f"{filename.stem}_cleanups"
            cni.skip_identical_graphs = False
            m = MSMEP(
                neb_inputs=nbi,
                chain_inputs=cni,
                gi_inputs=gii,
                optimizer=optimizer,
            )

            j = Janitor(history_object=history, out_path=op, msmep_object=m)

            clean_msmep = j.create_clean_msmep()

            if clean_msmep:
                clean_msmep.write_to_disk(
                    filename.parent / (str(filename.stem) + "_clean.xyz")
                )
                j.write_to_disk(op)

            tot_grad_calls = j.get_n_grad_calls()
            print(f">>> Made {tot_grad_calls} gradient calls in cleanup.")

    elif args.method in ["fneb", "neb", "pygsm"]:
        print(f"RUNNING REGULAR {args.method.upper()}")

        n, elem_step_results = m.run_minimize_chain(input_chain=chain)
        fp = Path(args.st)
        data_dir = fp.parent
        if args.name:
            filename = data_dir / (args.name + "_neb.xyz")

        else:
            filename = data_dir / f"{fp.stem}_neb.xyz"
        end_time = time.time()
        n.write_to_disk(filename)
        tot_grad_calls = n.grad_calls_made
        print(f">>> Made {tot_grad_calls} gradient calls total.")

    else:
        raise ValueError("Incorrect input method. Use 'asneb' or 'neb'")

    print(f"***Walltime: {end_time - start_time} s")


if __name__ == "__main__":
    main()
