from __future__ import annotations

import typer
from typing_extensions import Annotated

import os
from openbabel import openbabel
from qcio import Structure, ProgramOutput
from qcop.exceptions import ExternalProgramError
import sys
from pathlib import Path
import time
import traceback

from neb_dynamics.inputs import RunInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.chain import Chain
from neb_dynamics.msmep import MSMEP
from neb_dynamics.nodes.nodehelpers import displace_by_dr
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file
from neb_dynamics.NetworkBuilder import NetworkBuilder
from neb_dynamics.inputs import NetworkInputs, ChainInputs
from neb_dynamics.helper_functions import compute_irc_chain
from neb_dynamics.pot import Pot
from neb_dynamics.pot import plot_results_from_pot_obj
from itertools import product
from typing import List
import networkx as nx
import logging


class _SuppressWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING


logging.getLogger().addFilter(_SuppressWarningFilter())


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
        multiplicity: int = 1,
        create_irc: Annotated[bool, typer.Option(
            help='whether to run and output an IRC chain. Need to set --use_tsopt also, otherwise\
                will attempt use the guess structure.')] = False,
        use_bigchem: Annotated[bool, typer.Option(
            help='whether to use chemcloud to compute hessian for ts opt and irc jobs')] = False):

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
            # try:
            print("CHARGE", charge)
            print("MULTIPLICITY", multiplicity)

            start_ref = Structure.open(start)
            end_ref = Structure.open(end)

            if start_ref.charge != charge or start_ref.multiplicity != multiplicity:
                print(
                    f"WARNING: {start} has charge {start_ref.charge} and multiplicity {start_ref.multiplicity}.\
                        Using {charge} and {multiplicity} instead."
                )
                start_ref = Structure(geometry=start_ref.geometry,
                                      charge=charge,
                                      multiplicity=multiplicity,
                                      symbols=start_ref.symbols)
            if end_ref.charge != charge or end_ref.multiplicity != multiplicity:
                print(
                    f"WARNING: {end} has charge {end_ref.charge} and multiplicity {end_ref.multiplicity}.\
                        Using {charge} and {multiplicity} instead."
                )
                end_ref = Structure(geometry=end_ref.geometry,
                                    charge=charge,
                                    multiplicity=multiplicity,
                                    symbols=end_ref.symbols)

            all_structs = [start_ref, end_ref]

            print(type(start_ref.charge), end_ref.charge)
            # except ValueError:
            #     start_structure = Structure.open(
            #         start, charge=None, multiplicity=None)
            #     end_structure = Structure.open(
            #         end, charge=None, multiplicity=None)
            #     all_structs = [start_structure, end_structure]
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
            all_nodes[0], keywords={'coordsys': 'cart'})

        all_nodes[0] = start_tr[-1]
        end_tr = program_input.engine.compute_geometry_optimization(
            all_nodes[-1], keywords={'coordsys': 'cart'})
        all_nodes[-1] = end_tr[-1]
        print("Done!")

    # create Chain
    print('nodes!', [node.structure.charge for node in all_nodes])
    chain = Chain.model_validate({
        "nodes": all_nodes,
        "parameters": program_input.chain_inputs}
    )

    # create MSMEP object
    m = MSMEP(inputs=program_input)

    # Run the optimization

    if recursive:
        program_input.path_min_inputs.do_elem_step_checks = True
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
                    if create_irc:
                        try:
                            irc = compute_irc_chain(ts_node=StructureNode(
                                structure=tsres.return_result), engine=program_input.engine)
                            irc.write_to_disk(
                                filename.stem+f"_tsres_{i}_IRC.xyz")

                        except Exception:
                            print(traceback.format_exc())
                            print("IRC failed. Continuing...")
                else:
                    print("TS optimization did not converge...")

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
        [print(node.structure.charge) for node in chain]
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

                if create_irc:
                    try:
                        irc = compute_irc_chain(ts_node=StructureNode(
                            structure=tsres.return_result), engine=program_input.engine)
                        irc.write_to_disk(
                            filename.stem+"_tsres_IRC.xyz")

                    except Exception:
                        print(traceback.format_exc())
                        print("IRC failed. Continuing...")

            else:
                print("TS optimization failed.")

        tot_grad_calls = n.grad_calls_made
        print(f">>> Made {tot_grad_calls} gradient calls total.")

    print(f"***Walltime: {end_time - start_time} s")


@app.command()
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


@app.command()
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


@app.command()
def make_netgen_path(
    name: Annotated[Path, typer.Option("--name", help='path to json file containing network objet')],
    inds: Annotated[List[int], typer.Option("--inds", help="sequence of node indices on network \
                                                               to create path for.")]):
    name = Path(name)
    assert name.exists(), f"{name.resolve()} does not exist."
    pot = Pot.read_from_disk(name)
    if len(inds) > 2:
        p = inds
    else:
        print("Computing shortest path weighed by barrier heights")
        p = nx.shortest_path(pot.graph, weight='barrier',
                             source=inds[0], target=inds[1])
    print(f"Path: {p}")
    chain = pot.path_to_chain(path=p)
    chain.write_to_disk(
        name.parent / f"path_{'-'.join([str(a) for a in inds])}.xyz")


@app.command()
def make_default_inputs(
        name: Annotated[str, typer.Option(
            "--name", help='path to output toml file')] = None,
        path_min_method: Annotated[str, typer.Option("--path-min-method", "-pmm",
                                                     help='name of path minimization.\
                                                          Options are: [neb, fneb]')] = "neb"):
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
        multiplicity: int = 1,
        max_pairs: int = 500,
        minimize_ends: bool = False


):
    start_time = time.time()

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')
    print(program_input)

    valid_suff = ['.qcio', '.xyz']
    assert (Path(start).suffix in valid_suff and Path(
        end).suffix in valid_suff), "Invalid file type. Make sure they are .qcio or .xyz files"

    # load the structures
    print(Path(start).suffix, Path(end).suffix)
    if Path(start).suffix == ".qcio":
        start_structures = ProgramOutput.open(start).results.conformers
        start_nodes = [StructureNode(structure=s) for s in start_structures]
    elif Path(start).suffix == ".xyz":
        start_structures = read_multiple_structure_from_file(
            start, charge=charge, spinmult=multiplicity)
        if len(start_structures) != 1:
            start_nodes = [StructureNode(structure=s)
                           for s in start_structures]
        else:
            print("Only one structure in reactant file. Sampling using CREST!")
            if minimize_ends:
                print("Minimizing endpoints...")
                sys.stdout.flush()
                start_structure = program_input.engine.compute_geometry_optimization(
                    StructureNode(structure=start_structures[0]))[-1].structure
            else:
                start_structure = start_structures[0]

            print("Sampling reactant...")
            sys.stdout.flush()
            try:
                start_conf_result = program_input.engine._compute_conf_result(
                    StructureNode(structure=start_structure))
                start_conf_result.save(Path(start).resolve(
                ).parent / (Path(start).stem + "_conformers.qcio"))

                start_nodes = [StructureNode(structure=s)
                               for s in start_conf_result.results.conformers]
                print("Done!")

            except ExternalProgramError as e:
                print(e.stdout)

    if Path(end).suffix == ".qcio":
        end_structures = ProgramOutput.open(end).results.conformers
        end_nodes = [StructureNode(structure=s) for s in end_structures]
    elif Path(end).suffix == ".xyz":
        end_structures = read_multiple_structure_from_file(
            end, charge=charge, spinmult=multiplicity)

        if len(end_structures) != 1:
            end_nodes = [StructureNode(structure=s)
                         for s in end_structures]
        else:
            print("Only one structure in product file. Sampling using CREST!")
            if minimize_ends:
                print("Minimizing endpoints...")
                sys.stdout.flush()

                end_structure = program_input.engine.compute_geometry_optimization(
                    StructureNode(structure=end_structures[0]))[-1].structure
            else:
                end_structure = end_structures[0]

            print("Sampling product...")
            sys.stdout.flush()
            end_conf_result = program_input.engine._compute_conf_result(
                StructureNode(structure=end_structure))
            end_conf_result.save(Path(end).resolve().parent /
                                 (Path(end).stem + "_conformers.qcio"))
            end_nodes = [StructureNode(structure=s)
                         for s in end_conf_result.results.conformers]
            print("Done!")
            sys.stdout.flush()

    sys.stdout.flush()

    pairs = list(product(start_nodes, end_nodes))[:max_pairs]
    for i, (start, end) in enumerate(pairs):
        # create Chain
        chain = Chain.model_validate({
            "nodes": [start, end],
            "parameters": program_input.chain_inputs,
        })

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

        try:
            history = m.run_recursive_minimize(chain)

            history.output_chain.write_to_disk(filename)
            history.write_to_disk(foldername)
        except Exception:
            print(f"FAILED ON PAIR {i}")
            continue

        end_time = time.time()
    print(f"***Walltime: {end_time - start_time} s")


@app.command()
def make_netgen_summary(
        directory: Annotated[str, typer.Option(
            "--directory", help='path to data directory')] = None,
        verbose: bool = False,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to RunInputs toml file')] = None,
        name: Annotated[str, typer.Option(
            "--name", help='name of pot and summary file')] = 'netgen'

):
    if directory is None:
        directory = Path(os.getcwd())
    else:
        directory = Path(directory).resolve()
    if inputs is not None:
        ri = RunInputs.open(inputs)
        chain_inputs = ri.chain_inputs
    else:
        chain_inputs = ChainInputs()

    nb = NetworkBuilder(data_dir=directory,
                        start=None, end=None,
                        network_inputs=NetworkInputs(verbose=verbose),
                        chain_inputs=chain_inputs)

    nb.msmep_data_dir = directory
    pot_fp = Path(directory / (name+".json"))
    if not pot_fp.exists():
        pot = nb.create_rxn_network(file_pattern="*")
        pot.write_to_disk(pot_fp)
    else:
        print(f"{pot_fp} already exists. Loading")
        pot = Pot.read_from_disk(pot_fp)

    plot_results_from_pot_obj(
        fp_out=(directory / f"{Path(name).stem+'.png'}"), pot=pot, include_pngs=True)
    plot_results_from_pot_obj(
        fp_out=(directory / f"{Path(name).stem+'.png'}"), pot=pot, include_pngs=False)

    # write nodes to xyz file
    nodes = [pot.graph.nodes[x]["td"] for x in pot.graph.nodes]
    chain = Chain.model_validate({"nodes": nodes})
    chain.write_to_disk(directory / 'nodes.xyz')


if __name__ == "__main__":
    app()
