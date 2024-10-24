import typer
from typing_extensions import Annotated

from openbabel import openbabel
from qcio import Structure
import sys
from pathlib import Path
import time

from neb_dynamics.inputs import RunInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.chain import Chain
from neb_dynamics.msmep import MSMEP


ob_log_handler = openbabel.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)

app = typer.Typer()


@app.command()
def run(
        start: Annotated[str, typer.Argument(help='path to start file, or a reactant smiles')],
        end: Annotated[str, typer.Argument(help='path to end file, or a product smiles')],
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to RunInputs toml file')] = None,
        use_smiles: bool = False,
        minimize_ends: bool = False,
        recursive: bool = False,
        name: str = None,
        charge: int = 0,
        multiplicity: int = 1

):
    start_time = time.time()
    # load the structures
    if use_smiles:
        from neb_dynamics.nodes.nodehelpers import create_pairs_from_smiles
        from neb_dynamics.arbalign import align_structures

        print(
            "WARNING: Using RXNMapper to create atomic mapping. Carefully check output to see how labels\
                 affected reaction path."
        )
        start_structure, end_structure = create_pairs_from_smiles(
            smi1=start, smi2=end)

        print("Using arbalign to optimize index labelling for endpoints")
        end_structure = align_structures(
            start_structure, end_structure, distance_metric='xtb')
    else:
        start_structure = Structure.open(start)
        end_structure = Structure.open(end)

    s_dict = start_structure.model_dump()
    s_dict["charge"], s_dict["multiplicity"] = charge, multiplicity
    start_structure = Structure(**s_dict)

    e_dict = end_structure.model_dump()
    e_dict["charge"], e_dict["multiplicity"] = charge, multiplicity
    end_structure = Structure(**e_dict)

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')
    print(program_input)
    sys.stdout.flush()

    # minimize endpoints if requested
    if minimize_ends:
        print("Minimizing input endpoints...")
        start_tr = program_input.engine.compute_geometry_optimization(
            StructureNode(structure=start_structure))
        start_node = start_tr[-1]
        end_tr = program_input.engine.compute_geometry_optimization(
            StructureNode(structure=end_structure))
        end_node = end_tr[-1]
        print("Done!")
    else:
        start_node = StructureNode(structure=start_structure)
        end_node = StructureNode(structure=end_structure)

    # create Chain
    chain = Chain(
        nodes=[start_node, end_node],
        parameters=program_input.chain_inputs,
    )

    # create MSMEP object
    m = MSMEP(inputs=program_input)

    # Run the optimization
    if recursive:
        print(f"RUNNING AUTOSPLITTING {program_input.path_min_method}")
        history = m.run_recursive_minimize(chain)

        leaves_nebs = [
            obj for obj in history.get_optimization_history() if obj]
        fp = Path(start)
        data_dir = fp.parent

        if name is not None:
            foldername = data_dir / name
            filename = data_dir / (name + ".xyz")

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
        fp = Path(start)
        data_dir = fp.parent
        if name is not None:
            filename = data_dir / (name + "_neb.xyz")

        else:
            filename = data_dir / f"{fp.stem}_neb.xyz"

        end_time = time.time()
        n.write_to_disk(filename)
        tot_grad_calls = n.grad_calls_made
        print(f">>> Made {tot_grad_calls} gradient calls total.")

    print(f"***Walltime: {end_time - start_time} s")


if __name__ == "__main__":
    app()
