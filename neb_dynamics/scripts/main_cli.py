import typer
from typing_extensions import Annotated

import os
from openbabel import openbabel
from qcio import Structure, ProgramOutput
import sys
from pathlib import Path
import time

from neb_dynamics.inputs import RunInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.chain import Chain
from neb_dynamics.msmep import MSMEP
from neb_dynamics.nodes.nodehelpers import displace_by_dr
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file

from itertools import product


ob_log_handler = openbabel.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)

app = typer.Typer()


@app.command()
def run(
        start: Annotated[str, typer.Option(
            help='path to start file, or a reactant smiles')] = None,
        end: Annotated[str, typer.Option(
            help='path to end file, or a product smiles')] = None,
        geometries:  Annotated[str, typer.Argument(help='file containing an approximate path between two endpoints. \
            Use this if you have precompted a path you want to use. Do not use this with smiles.')] = None,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to RunInputs toml file')] = None,
        use_smiles: bool = False,
        use_tsopt: Annotated[bool, typer.Option(
            help='whether to run a transition state optimization on each TS guess')] = False,
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
            start_structure, end_structure, distance_metric='RMSD')

        all_structs = [start_structure, end_structure]
    else:

        if geometries is not None:
            try:
                all_structs = read_multiple_structure_from_file(
                    geometries, charge=charge, spinmult=multiplicity)
            except ValueError:  # qcio does not allow an input for charge if file has a charge in it
                all_structs = read_multiple_structure_from_file(
                    geometries, charge=None, spinmult=None)
        elif start is not None and end is not None:
            try:
                start_structure = Structure.open(
                    start, charge=charge, multiplicity=multiplicity)
                end_structure = Structure.open(
                    end, charge=charge, multiplicity=multiplicity)
                all_structs = [start_structure, end_structure]
            except ValueError:
                start_structure = Structure.open(
                    start, charge=None, multiplicity=None)
                end_structure = Structure.open(
                    end, charge=None, multiplicity=None)
                all_structs = [start_structure, end_structure]
        else:
            raise ValueError(
                "Either 'geometries' or 'start' and 'end' flags must be populated!")

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')
    print(program_input)
    sys.stdout.flush()

    # minimize endpoints if requested
    all_nodes = [StructureNode(structure=s) for s in all_structs]
    if minimize_ends:
        print("Minimizing input endpoints...")
        start_tr = program_input.engine.compute_geometry_optimization(
            StructureNode(structure=all_structs[0]))

        all_nodes[0] = start_tr[-1]
        end_tr = program_input.engine.compute_geometry_optimization(
            StructureNode(structure=all_structs[-1]))
        all_nodes[-1] = end_tr[-1]
        print("Done!")

    # create Chain
    chain = Chain(
        nodes=all_nodes,
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
        fp = Path("mep_output")

        if name is not None:
            data_dir = Path(name).resolve().parent
            foldername = data_dir / name
            filename = data_dir / (name + ".xyz")

        else:
            data_dir = Path(os.getcwd())
            foldername = data_dir / f"{fp.stem}_msmep"
            filename = data_dir / f"{fp.stem}_msmep.xyz"

        end_time = time.time()
        history.output_chain.write_to_disk(filename)
        history.write_to_disk(foldername)

        if use_tsopt:
            for i, leaf in enumerate(history.ordered_leaves):
                if not leaf.data:
                    continue
                print(f"Running TS opt on leaf {i}")
                try:
                    tsres = program_input.engine._compute_ts_result(
                        leaf.data.chain_trajectory[-1].get_ts_node())
                except Exception as e:
                    tsres = e.program_output
                tsres.save(data_dir / (filename.stem+f"_tsres_{i}.qcio"))
                if tsres.success:
                    tsres.return_result.save(
                        data_dir / (filename.stem+f"_ts_{i}.xyz"))

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
        fp = Path("mep_output")
        data_dir = Path(os.getcwd())
        if name is not None:
            filename = data_dir / (name + ".xyz")

        else:
            filename = data_dir / f"{fp.stem}_neb.xyz"

        end_time = time.time()
        n.write_to_disk(filename)

        if use_tsopt:
            print("Running TS opt")
            try:
                tsres = program_input.engine._compute_ts_result(
                    n.chain_trajectory[-1].get_ts_node())
            except Exception as e:
                tsres = e.program_output
            tsres.save(data_dir / (filename.stem+"_tsres.qcio"))
            if tsres.success:
                tsres.return_result.save(
                    data_dir / (filename.stem+"_ts.xyz"))

        tot_grad_calls = n.grad_calls_made
        print(f">>> Made {tot_grad_calls} gradient calls total.")

    print(f"***Walltime: {end_time - start_time} s")


@ app.command()
def ts(
    geometry: Annotated[str, typer.Argument(help='path to geometry file tpo optimize')],
    inputs: Annotated[str, typer.Option("--inputs", "-i",
                                        help='path to RunInputs toml file')] = None,
    name: str = None,
    charge: int = 0,
    multiplicity: int = 1,
    bigchem: bool = False
):
    # create output names
    fp = Path(geometry)
    data_dir = Path(os.getcwd())

    if name is not None:
        results_name = data_dir / (name + ".qcio")
        filename = data_dir / (name + ".xyz")
    else:
        results_name = fp.stem + ".qcio"
        filename = fp.stem + ".xyz"

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')

    print(f"Optimizing: {geometry}...")
    sys.stdout.flush()
    try:
        struct = Structure.open(geometry)
        s_dict = struct.model_dump()
        s_dict["charge"], s_dict["multiplicity"] = charge, multiplicity
        struct = Structure(**s_dict)

        node = StructureNode(structure=struct)
        output = program_input.engine._compute_ts_result(
            node=node, use_bigchem=bigchem)

    except Exception as e:
        output = e.program_output

    output.save(results_name)
    output.results.final_structure.save(filename)
    print("Done!")


@ app.command()
def pseuirc(geometry: Annotated[str, typer.Argument(help='path to geometry file tpo optimize')],
            inputs: Annotated[str, typer.Option("--inputs", "-i",
                                                help='path to RunInputs toml file')] = None,
            name: str = None,
            charge: int = 0,
            multiplicity: int = 1,
            dr: float = 1.0):
    # create output names
    fp = Path(geometry)
    data_dir = Path(os.getcwd())

    if name is not None:
        results_name = data_dir / (name + ".qcio")
    else:
        results_name = Path(fp.stem + ".qcio")

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')

    print(f"Computing hessian: {geometry}...")

    sys.stdout.flush()
    try:
        struct = Structure.open(geometry)
        s_dict = struct.model_dump()
        s_dict["charge"], s_dict["multiplicity"] = charge, multiplicity
        struct = Structure(**s_dict)

        node = StructureNode(structure=struct)
        hessres = program_input.engine._compute_hessian_result(node)

    except Exception as e:
        hessres = e.program_output

    hessres.save(results_name.parent / (results_name.stem+"_hessian.qcio"))

    print(f"Minimizing TS(-): {geometry}...")
    sys.stdout.flush()
    tsminus_raw = displace_by_dr(
        node=node, displacement=hessres.results.normal_modes_cartesian[0], dr=-dr)
    tsminus_res = program_input.engine._compute_geom_opt_result(
        tsminus_raw)
    tsminus_res.save(results_name.parent / (results_name.stem+"_minus.qcio"))

    print(f"Minimizing TS(+): {geometry}...")
    sys.stdout.flush()
    tsplus_raw = displace_by_dr(
        node=node, displacement=hessres.results.normal_modes_cartesian[0], dr=dr)
    tsplus_res = program_input.engine._compute_geom_opt_result(
        tsplus_raw)

    tsplus_res.save(results_name.parent / (results_name.stem+"_plus.qcio"))
    print("Done!")


@ app.command()
def make_default_inputs(
        name: Annotated[str, typer.Option(
            "--name", help='path to output toml file')] = None,
        path_min_method: Annotated[str, typer.Option("--path-min-method", "-pmm",
                                                     help='name of path minimization. Options are: [neb, fneb]')] = "neb"):
    if name is None:
        name = Path(Path(os.getcwd()) / 'default_inputs')
    ri = RunInputs(path_min_method=path_min_method)
    out = Path(name)
    ri.save(out.parent / (out.stem+".toml"))


@app.command()
def run_netgen(
        start: Annotated[str, typer.Option(
            help='path to start conformers data')] = None,
        end: Annotated[str, typer.Option(
            help='path to end conformers data')] = None,

        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to RunInputs toml file')] = None,

        name: str = None,
        charge: int = 0,
        multiplicity: int = 1

):
    start_time = time.time()

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')
    print(program_input)

    # load the structures
    print(Path(start).suffix, Path(end).suffix)
    if Path(start).suffix == ".qcio" and Path(end).suffix == ".qcio":
        start_structures = ProgramOutput.open(start).results.conformers
        end_structures = ProgramOutput.open(end).results.conformers

        start_nodes = [StructureNode(structure=s) for s in start_structures]
        end_nodes = [StructureNode(structure=s) for s in end_structures]

    elif Path(start).suffix == ".xyz" and Path(end).suffix == ".xyz":
        start_structures = read_multiple_structure_from_file(
            start, charge=charge, spinmult=multiplicity)
        end_structures = read_multiple_structure_from_file(
            end, charge=charge, spinmult=multiplicity)
        if len(start_structures) == 1 and len(end_structures) == 1:
            print("Only one structure in each file. Sampling using CREST!")
            print("Sampling reactant...")
            start_conf_result = program_input.engine._compute_conf_result(
                StructureNode(structure=start_structures[0]))
            start_conf_result.save(Path(start).resolve(
            ).parent / (Path(start).stem + "_conformers.qcio"))

            start_nodes = [StructureNode(structure=s)
                           for s in start_conf_result.results.conformers]
            print("Done!")

            print("Sampling product...")
            end_conf_result = program_input.engine._compute_conf_result(
                StructureNode(structure=end_structures[0]))
            end_conf_result.save(Path(end).resolve().parent /
                                 (Path(end).stem + "_conformers.qcio"))
            end_nodes = [StructureNode(structure=s)
                         for s in end_conf_result.results.conformers]
            print("Done!")
    else:
        raise ValueError(
            f"Either both start and end must be .qcio files, or both must be .xyz files. Inputs: {start}, {end}")

    sys.stdout.flush()

    pairs = list(product(start_nodes, end_nodes))
    for i, (start, end) in enumerate(pairs):
        # create Chain
        chain = Chain(
            nodes=[start, end],
            parameters=program_input.chain_inputs,
        )

        # define output names
        fp = Path("mep_output")
        if name is not None:
            data_dir = Path(name).resolve().parent
            foldername = data_dir / (name + f"_pair{i}")
            filename = data_dir / (name + f"_pair{i}.xyz")

        else:
            data_dir = Path(os.getcwd())
            foldername = data_dir / f"{fp.stem}_msmep_pair{i}"
            filename = data_dir / f"{fp.stem}_msmep_pair{i}.xyz"

        # create MSMEP object
        m = MSMEP(inputs=program_input)

        # Run the optimization

        print(
            f"RUNNING AUTOSPLITTING ON PAIR {i}/{len(pairs)} {program_input.path_min_method}")
        if filename.exists():
            print("already done. Skipping...")
            continue
        history = m.run_recursive_minimize(chain)

        end_time = time.time()
        history.output_chain.write_to_disk(filename)
        history.write_to_disk(foldername)

    print(f"***Walltime: {end_time - start_time} s")


if __name__ == "__main__":
    app()
