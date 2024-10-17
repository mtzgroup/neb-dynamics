import sys
from argparse import ArgumentParser
from pathlib import Path

from neb_dynamics.nodes.node import StructureNode
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import RunInputs
from neb_dynamics.Janitor import Janitor
from neb_dynamics.msmep import MSMEP
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
        "-c",
        "--charge",
        dest="c",
        required=False,
        type=int,
        help="charge of molecular system",
        default=0
    )

    parser.add_argument(
        "-s",
        "--spinmult",
        dest="s",
        required=False,
        type=int,
        help="spin multiplicity of molecular system",
        default=1
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
        "-pif",
        "--program_input_file",
        dest="pif",
        required=False,
        type=str,
        help="file path to a ProgramInput file from qcio.",
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

    parser.add_argument(
        "-rec",
        "--recursive",
        dest="rec",
        required=False,
        type=int,
        help="whether to run recursive path minimization. 1: yes. 0: no",
        default=0,
    )

    return parser.parse_args()


def main():
    start_time = time.time()
    args = read_single_arguments()

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

    if args.pif:
        program_input = RunInputs.open(args.pif)
    else:
        program_input = RunInputs(program='xtb', engine='qcop')

    if args.min_ends:
        print("Minimizing input endpoints...")
        start_tr = program_input.engine.compute_geometry_optimization(
            StructureNode(structure=start))
        start_node = start_tr[-1]
        end_tr = program_input.engine.compute_geometry_optimization(
            StructureNode(structure=end))
        end_node = end_tr[-1]
        print("Done!")
    else:
        start_node = StructureNode(structure=start)
        end_node = StructureNode(structure=end)

    print(program_input)
    sys.stdout.flush()
    chain = Chain(
        nodes=[start_node, end_node],
        parameters=program_input.chain_inputs,
    )
    m = MSMEP(inputs=program_input)

    from openbabel import openbabel

    ob_log_handler = openbabel.OBMessageHandler()
    ob_log_handler.SetOutputLevel(0)

    if bool(args.rec):
        print(f"RUNNING AUTOSPLITTING {program_input.path_min_method}")
        history = m.run_recursive_minimize(chain)

        leaves_nebs = [
            obj for obj in history.get_optimization_history() if obj]
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
        geom_grad_calls = sum(
            [obj.geom_grad_calls_made for obj in leaves_nebs])
        print(f">>> Made {tot_grad_calls} gradient calls total.")
        print(
            f"<<< Made {geom_grad_calls} gradient for geometry\
               optimizations."
        )

    else:
        print(f"RUNNING REGULAR {program_input.path_min_method}")

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

    print(f"***Walltime: {end_time - start_time} s")


if __name__ == "__main__":
    main()
