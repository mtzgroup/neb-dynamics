from __future__ import annotations
import logging
import networkx as nx
from typing import List
from itertools import product
from neb_dynamics.pot import plot_results_from_pot_obj
from neb_dynamics.pot import Pot
from neb_dynamics.helper_functions import compute_irc_chain
from neb_dynamics.inputs import NetworkInputs, ChainInputs
from neb_dynamics.NetworkBuilder import NetworkBuilder
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file
from neb_dynamics.nodes.nodehelpers import displace_by_dr
from neb_dynamics.msmep import MSMEP
from neb_dynamics.chain import Chain
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.inputs import RunInputs

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
from datetime import datetime
from qcinf import structure_to_smiles

from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.status import Status
from rich.table import Table
from rich.box import Box
from rich.text import Text
from rich.syntax import Syntax
from rich import box

# Custom theme for Claude Code-like styling
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "header": "bold magenta",
    "dim": "dim",
})


class _SuppressWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING


logging.getLogger().addFilter(_SuppressWarningFilter())


ob_log_handler = openbabel.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)

app = typer.Typer(
    rich_markup_mode="rich"
)

# CLI Banner
BANNER = """
[bold magenta]╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║                        [bold cyan]NEB[/bold cyan] [bold white]-[/bold white] [bold cyan]Dynamics[/bold cyan]                        ║
║        [dim]Reaction Path Optimization & Network Generation[/dim]        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝[/bold magenta]
"""


def print_banner():
    """Print the CLI banner."""
    console.print(BANNER)


# Global console instance for consistent styling
console = Console(theme=custom_theme)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """Show banner before running any command."""
    if ctx.invoked_subcommand is None:
        print_banner()


def create_progress():
    """Create a rich progress bar for long-running tasks."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _truncate_label(label: str, max_len: int) -> str:
    if len(label) <= max_len:
        return label
    if max_len <= 3:
        return label[:max_len]
    return label[: max_len - 3] + "..."


def _build_ascii_energy_profile(energies, labels, width: int = 60, height: int = 12):
    if len(energies) == 0:
        return "No energies to plot."
    if len(energies) != len(labels):
        raise ValueError("labels must be same length as energies")

    min_e = float(min(energies))
    max_e = float(max(energies))
    if max_e == min_e:
        max_e = min_e + 1e-6

    def _xpos(i):
        if len(energies) == 1:
            return 0
        return int(round(i * (width - 1) / (len(energies) - 1)))

    def _ypos(e):
        return int(round((e - min_e) * (height - 1) / (max_e - min_e)))

    grid = [[" " for _ in range(width)] for _ in range(height)]

    xs = [_xpos(i) for i in range(len(energies))]
    ys = [_ypos(e) for e in energies]

    for i in range(len(energies) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        if x0 == x1:
            continue
        step = 1 if x1 > x0 else -1
        for x in range(x0, x1 + step, step):
            t = (x - x0) / (x1 - x0)
            y = int(round(y0 + (y1 - y0) * t))
            row = height - 1 - y
            if 0 <= row < height and 0 <= x < width and grid[row][x] == " ":
                grid[row][x] = "-"

    for x, y in zip(xs, ys):
        row = height - 1 - y
        if 0 <= row < height and 0 <= x < width:
            grid[row][x] = "*"

    prefix_len = len(f"{max_e:7.2f} |")
    lines = []
    for r in range(height):
        y_val = max_e - (max_e - min_e) * (r / (height - 1))
        prefix = f"{y_val:7.2f} |"
        lines.append(prefix + "".join(grid[r]))

    lines.append(" " * prefix_len + "-" * width)

    label_line = [" " for _ in range(width)]
    max_label_len = 12
    truncated = False
    for x, label in zip(xs, labels):
        tlabel = _truncate_label(label, max_label_len)
        if tlabel != label:
            truncated = True
        start = max(0, min(width - len(tlabel), x - len(tlabel) // 2))
        for i, ch in enumerate(tlabel):
            label_line[start + i] = ch
    lines.append(" " * prefix_len + "".join(label_line))

    if truncated:
        lines.append("")
        lines.append("Full SMILES labels:")
        for i, label in enumerate(labels):
            lines.append(f"{i}: {label}")

    return "\n".join(lines)


def _ascii_profile_for_chain(chain: Chain):
    try:
        energies = chain.energies_kcalmol
    except Exception as exc:
        console.print(
            f"[yellow]⚠ Could not compute energy profile: {exc}[/yellow]")
        return

    labels = []
    for i, node in enumerate(chain.nodes):
        try:
            smi = structure_to_smiles(node.structure)
        except Exception:
            smi = f"node_{i}"
        labels.append(smi)

    plot = _build_ascii_energy_profile(energies, labels)
    console.print("\nASCII Reaction Profile (Energy vs Node)")
    console.print(plot, markup=False)


@app.command("run")
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

    # Print header
    console.print(BANNER)

    table = Table(box=None, show_header=False)
    table.add_column(style="dim")
    table.add_row("[bold cyan]Command:[/bold cyan]", "[white]run[/white]")
    table.add_row("[bold cyan]Method:[/bold cyan]",
                  f"[yellow]{'recursive' if recursive else 'regular'}[/yellow]")
    table.add_row("[bold cyan]SMILES mode:[/bold cyan]",
                  f"[yellow]{use_smiles}[/yellow]")
    console.print(table)
    console.print()

    start_time = time.time()
    # load the structures
    if use_smiles:
        from neb_dynamics.nodes.nodehelpers import create_pairs_from_smiles
        from neb_dynamics.arbalign import align_structures

        console.print(
            "[yellow]⚠ WARNING:[/yellow] Using RXNMapper to create atomic mapping. Carefully check output to see how labels affected reaction path.")
        with console.status("[bold cyan]Creating structures from SMILES...[/bold cyan]") as status:
            start_structure, end_structure = create_pairs_from_smiles(
                smi1=start, smi2=end)

            console.print(
                "[cyan]Using arbalign to optimize index labelling for endpoints[/cyan]")
            end_structure = align_structures(
                start_structure, end_structure, distance_metric='RMSD')

        all_structs = [start_structure, end_structure]
    else:

        if geometries is not None:
            with console.status(f"[bold cyan]Loading geometries from {geometries}...[/bold cyan]"):
                try:
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=charge, spinmult=multiplicity)
                except ValueError:  # qcio does not allow an input for charge if file has a charge in it
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=None, spinmult=None)
        elif start is not None and end is not None:
            with console.status(f"[bold cyan]Loading structures...[/bold cyan]"):
                console.print(
                    f"[dim]Charge: {charge}, Multiplicity: {multiplicity}[/dim]")

                start_ref = Structure.open(start)
                end_ref = Structure.open(end)

                if start_ref.charge != charge or start_ref.multiplicity != multiplicity:
                    console.print(
                        f"[yellow]⚠ WARNING:[/yellow] {start} has charge {start_ref.charge} and multiplicity {start_ref.multiplicity}. Using {charge} and {multiplicity} instead."
                    )
                    start_ref = Structure(geometry=start_ref.geometry,
                                          charge=charge,
                                          multiplicity=multiplicity,
                                          symbols=start_ref.symbols)
                if end_ref.charge != charge or end_ref.multiplicity != multiplicity:
                    console.print(
                        f"[yellow]⚠ WARNING:[/yellow] {end} has charge {end_ref.charge} and multiplicity {end_ref.multiplicity}. Using {charge} and {multiplicity} instead."
                    )
                    end_ref = Structure(geometry=end_ref.geometry,
                                        charge=charge,
                                        multiplicity=multiplicity,
                                        symbols=end_ref.symbols)

                all_structs = [start_ref, end_ref]
        else:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Either 'geometries' or 'start' and 'end' flags must be populated!")
            raise typer.Exit(1)

    # load the RunInputs
    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        if inputs is not None:
            program_input = RunInputs.open(inputs)
        else:
            program_input = RunInputs(program='xtb', engine_name='qcop')

    console.print(Panel(str(program_input),
                  title="[bold cyan]Input Parameters[/bold cyan]", border_style="cyan", box=box.ROUNDED))
    sys.stdout.flush()

    # minimize endpoints if requested
    all_nodes = [StructureNode(structure=s) for s in all_structs]
    if minimize_ends:
        console.print("[bold cyan]⟳ Minimizing input endpoints...[/bold cyan]")
        start_tr = program_input.engine.compute_geometry_optimization(
            all_nodes[0], keywords={'coordsys': 'cart', 'maxiter': 500})

        all_nodes[0] = start_tr[-1]
        end_tr = program_input.engine.compute_geometry_optimization(
            all_nodes[-1], keywords={'coordsys': 'cart', 'maxiter': 500})
        all_nodes[-1] = end_tr[-1]
        console.print("[bold green]✓ Done![/bold green]")

    # create Chain
    console.print(f"[dim]Loading {len(all_nodes)} nodes...[/dim]")
    chain = Chain.model_validate({
        "nodes": all_nodes,
        "parameters": program_input.chain_inputs}
    )

    # create MSMEP object
    m = MSMEP(inputs=program_input)

    # Run the optimization
    chain_for_profile = None

    if recursive:
        if not program_input.path_min_inputs.do_elem_step_checks:
            console.print(
                "[yellow]⚠ WARNING: do_elem_step_checks is set to False. This may cause issues with recursive splitting.[/yellow]")
            console.print(
                "[yellow]Making it True to ensure proper functioning of recursive splitting.[/yellow]")
            program_input.path_min_inputs.do_elem_step_checks = True
        console.print(
            f"[bold magenta]▶ RUNNING AUTOSPLITTING {program_input.path_min_method}[/bold magenta]")
        history = m.run_recursive_minimize(chain)

        if not history.data:
            console.print("[bold red]✗ ERROR:[/bold red] Program did not run. Likely because your endpoints are conformers of the same molecular graph. Tighten the node_rms_thre and/or node_ene_thre parameters in chain_inputs and try again.")
            raise typer.Exit(1)

        leaves_nebs = [
            obj for obj in history.get_optimization_history() if obj]
        fp = Path("mep_output")
        if name is not None:
            name = Path(name)
            data_dir = Path(name).resolve().parent
            foldername = data_dir / name.stem
            filename = data_dir / (name.stem + ".xyz")

        else:
            data_dir = Path(os.getcwd())
            foldername = data_dir / f"{fp.stem}_msmep"
            filename = data_dir / f"{fp.stem}_msmep.xyz"

        end_time = time.time()
        history.output_chain.write_to_disk(filename)
        history.write_to_disk(foldername)
        chain_for_profile = history.output_chain

        if use_tsopt:
            for i, leaf in enumerate(history.ordered_leaves):
                if not leaf.data:
                    continue
                console.print(
                    f"[bold cyan]⟳ Running TS opt on leaf {i}...[/bold cyan]")
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
                            console.print(
                                f"[yellow]⚠ IRC failed: {traceback.format_exc()}[/yellow]")
                            console.print(
                                "[yellow]IRC failed. Continuing...[/yellow]")
                else:
                    console.print(
                        f"[yellow]⚠ TS optimization did not converge on leaf {i}...[/yellow]")

        tot_grad_calls = sum([obj.grad_calls_made for obj in leaves_nebs])
        geom_grad_calls = sum(
            [obj.geom_grad_calls_made for obj in leaves_nebs])
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {tot_grad_calls} gradient calls total.[/cyan]")
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {geom_grad_calls} gradient for geometry optimizations.[/cyan]")

    else:
        console.print(
            f"[bold magenta]▶ RUNNING REGULAR {program_input.path_min_method}[/bold magenta]")
        n, elem_step_results = m.run_minimize_chain(input_chain=chain)
        fp = Path("mep_output")
        data_dir = Path(os.getcwd())
        if name is not None:
            filename = data_dir / (name + ".xyz")

        else:
            filename = data_dir / f"{fp.stem}_neb.xyz"

        end_time = time.time()
        n.write_to_disk(filename)
        if n.chain_trajectory:
            chain_for_profile = n.chain_trajectory[-1]
        else:
            chain_for_profile = n.optimized

        if use_tsopt:
            console.print("[bold cyan]⟳ Running TS opt...[/bold cyan]")
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
                        console.print(
                            f"[yellow]⚠ IRC failed: {traceback.format_exc()}[/yellow]")
                        console.print(
                            "[yellow]IRC failed. Continuing...[/yellow]")

            else:
                console.print("[yellow]⚠ TS optimization failed.[/yellow]")

        tot_grad_calls = n.grad_calls_made
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {tot_grad_calls} gradient calls total.[/cyan]")

    end_time = time.time()
    elapsed = end_time - start_time

    # Print summary panel
    summary = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    if elapsed > 60:
        summary.add_row(
            "⏱ Walltime:", f"[yellow]{elapsed/60:.1f} min[/yellow]")
    else:
        summary.add_row("⏱ Walltime:", f"[yellow]{elapsed:.1f} s[/yellow]")
    summary.add_row("📁 Output:", f"[cyan]{filename}[/cyan]")
    console.print(Panel(
        summary, title="[bold green]✓ Complete![/bold green]", border_style="green"))

    if chain_for_profile is not None:
        _ascii_profile_for_chain(chain_for_profile)


@app.command("ts")
def ts(
    geometry: Annotated[str, typer.Argument(help='path to geometry file to optimize')],
    inputs: Annotated[str, typer.Option("--inputs", "-i",
                                        help='path to RunInputs toml file')] = None,
    name: str = None,
    charge: int = 0,
    multiplicity: int = 1,
    bigchem: bool = False
):
    console.print(BANNER)

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

    with console.status(f"[bold cyan]Optimizing transition state: {geometry}...[/bold cyan]") as status:
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
    console.print(f"[bold green]✓ TS optimization complete![/bold green]")
    console.print(f"[dim]Results: {results_name}[/dim]")
    console.print(f"[dim]Geometry: {filename}[/dim]")


@app.command("pseuirc")
def pseuirc(geometry: Annotated[str, typer.Argument(help='path to geometry file to optimize')],
            inputs: Annotated[str, typer.Option("--inputs", "-i",
                                                help='path to RunInputs toml file')] = None,
            name: str = None,
            charge: int = 0,
            multiplicity: int = 1,
            dr: float = 1.0):
    console.print(BANNER)

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

    with console.status(f"[bold cyan]Computing hessian...[/bold cyan]"):
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

    console.print("[bold cyan]⟳ Minimizing TS(-)...[/bold cyan]")
    sys.stdout.flush()
    tsminus_raw = displace_by_dr(
        node=node, displacement=hessres.results.normal_modes_cartesian[0], dr=-dr)
    tsminus_res = program_input.engine._compute_geom_opt_result(
        tsminus_raw)
    tsminus_res.save(results_name.parent / (results_name.stem+"_minus.qcio"))

    console.print("[bold cyan]⟳ Minimizing TS(+)...[/bold cyan]")
    sys.stdout.flush()
    tsplus_raw = displace_by_dr(
        node=node, displacement=hessres.results.normal_modes_cartesian[0], dr=dr)
    tsplus_res = program_input.engine._compute_geom_opt_result(
        tsplus_raw)

    tsplus_res.save(results_name.parent / (results_name.stem+"_plus.qcio"))
    console.print(f"[bold green]✓ Pseudo-IRC complete![/bold green]")


@app.command("make-netgen-path")
def make_netgen_path(
    name: Annotated[Path, typer.Option("--name", help='path to json file containing network object')],
    inds: Annotated[List[int], typer.Option("--inds", help="sequence of node indices on network \
                                                               to create path for.")]):
    console.print(BANNER)
    name = Path(name)
    assert name.exists(), f"{name.resolve()} does not exist."
    pot = Pot.read_from_disk(name)
    if len(inds) > 2:
        p = inds
    else:
        console.print(
            "[cyan]Computing shortest path weighed by barrier heights[/cyan]")
        p = nx.shortest_path(pot.graph, weight='barrier',
                             source=inds[0], target=inds[1])
    console.print(f"[green]✓ Path:[/green] {p}")
    chain = pot.path_to_chain(path=p)
    chain.write_to_disk(
        name.parent / f"path_{'-'.join([str(a) for a in inds])}.xyz")
    console.print(f"[bold green]✓ Path written to disk![/bold green]")


@app.command("make-default-inputs")
def make_default_inputs(
        name: Annotated[str, typer.Option(
            "--name", help='path to output toml file')] = None,
        path_min_method: Annotated[str, typer.Option("--path-min-method", "-pmm",
                                                     help='name of path minimization.\
                                                          Options are: [neb, fneb]')] = "neb"):
    console.print(BANNER)
    if name is None:
        name = Path(Path(os.getcwd()) / 'default_inputs')
    ri = RunInputs(path_min_method=path_min_method)
    out = Path(name)
    ri.save(out.parent / (out.stem+".toml"))
    console.print(
        f"[bold green]✓ Default inputs saved to:[/bold green] {out.parent / (out.stem+'.toml')}")


@app.command("run-netgen")
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
    console.print(BANNER)

    start_time = time.time()

    # load the RunInputs
    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        if inputs is not None:
            program_input = RunInputs.open(inputs)
        else:
            program_input = RunInputs(program='xtb', engine_name='qcop')

    console.print(Panel(str(program_input),
                  title="[bold cyan]Input Parameters[/bold cyan]", border_style="cyan", box=box.ROUNDED))

    valid_suff = ['.qcio', '.xyz']
    assert (Path(start).suffix in valid_suff and Path(
        end).suffix in valid_suff), "Invalid file type. Make sure they are .qcio or .xyz files"

    # load the structures
    console.print(
        f"[dim]Loading structures: {Path(start).suffix} → {Path(end).suffix}[/dim]")
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
            console.print(
                "[cyan]Only one structure in reactant file. Sampling using CREST![/cyan]")
            if minimize_ends:
                console.print(
                    "[bold cyan]⟳ Minimizing endpoints...[/bold cyan]")
                sys.stdout.flush()
                start_structure = program_input.engine.compute_geometry_optimization(
                    StructureNode(structure=start_structures[0]))[-1].structure
            else:
                start_structure = start_structures[0]

            console.print("[bold cyan]⟳ Sampling reactant...[/bold cyan]")
            sys.stdout.flush()
            try:
                start_conf_result = program_input.engine._compute_conf_result(
                    StructureNode(structure=start_structure))
                start_conf_result.save(Path(start).resolve(
                ).parent / (Path(start).stem + "_conformers.qcio"))

                start_nodes = [StructureNode(structure=s)
                               for s in start_conf_result.results.conformers]
                console.print("[bold green]✓ Done![/bold green]")

            except ExternalProgramError as e:
                console.print(f"[bold red]Error:[/bold red] {e.stdout}")

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
            console.print(
                "[cyan]Only one structure in product file. Sampling using CREST![/cyan]")
            if minimize_ends:
                console.print(
                    "[bold cyan]⟳ Minimizing endpoints...[/bold cyan]")
                sys.stdout.flush()

                end_structure = program_input.engine.compute_geometry_optimization(
                    StructureNode(structure=end_structures[0]))[-1].structure
            else:
                end_structure = end_structures[0]

            console.print("[bold cyan]⟳ Sampling product...[/bold cyan]")
            sys.stdout.flush()
            end_conf_result = program_input.engine._compute_conf_result(
                StructureNode(structure=end_structure))
            end_conf_result.save(Path(end).resolve().parent /
                                 (Path(end).stem + "_conformers.qcio"))
            end_nodes = [StructureNode(structure=s)
                         for s in end_conf_result.results.conformers]
            console.print("[bold green]✓ Done![/bold green]")
            sys.stdout.flush()

    sys.stdout.flush()

    pairs = list(product(start_nodes, end_nodes))[:max_pairs]
    for i, (start_node, end_node) in enumerate(pairs):
        # create Chain
        chain = Chain.model_validate({
            "nodes": [start_node, end_node],
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

        console.print(
            f"[bold magenta]▶ Running autosplitting on pair {i+1}/{len(pairs)} ({program_input.path_min_method})[/bold magenta]")
        if filename.exists():
            console.print("[yellow]⚠ Already done. Skipping...[/yellow]")
            continue

        try:
            history = m.run_recursive_minimize(chain)

            history.output_chain.write_to_disk(filename)
            history.write_to_disk(foldername)
        except Exception:
            console.print(f"[bold red]✗ Failed on pair {i}[/bold red]")
            continue

    end_time = time.time()
    elapsed = end_time - start_time
    if elapsed > 60:
        time_str = f"{elapsed/60:.1f} min"
    else:
        time_str = f"{elapsed:.1f} s"
    console.print(
        f"[bold green]✓ Netgen complete![/bold green] [dim]Walltime: {time_str}[/dim]")


@app.command("make-netgen-summary")
def make_netgen_summary(
        directory: Annotated[str, typer.Option(
            "--directory", help='path to data directory')] = None,
        verbose: bool = False,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to RunInputs toml file')] = None,
        name: Annotated[str, typer.Option(
            "--name", help='name of pot and summary file')] = 'netgen'

):
    console.print(BANNER)

    if directory is None:
        directory = Path(os.getcwd())
    else:
        directory = Path(directory).resolve()
    if inputs is not None:
        ri = RunInputs.open(inputs)
        chain_inputs = ri.chain_inputs
    else:
        chain_inputs = ChainInputs()

    with console.status("[bold cyan]Building reaction network...[/bold cyan]"):
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
            console.print(
                f"[yellow]⚠ {pot_fp} already exists. Loading...[/yellow]")
            pot = Pot.read_from_disk(pot_fp)

    with console.status("[bold cyan]Generating plots...[/bold cyan]"):
        plot_results_from_pot_obj(
            fp_out=(directory / f"{Path(name).stem+'.png'}"), pot=pot, include_pngs=True)
        plot_results_from_pot_obj(
            fp_out=(directory / f"{Path(name).stem+'.png'}"), pot=pot, include_pngs=False)

    # write nodes to xyz file
    nodes = [pot.graph.nodes[x]["td"] for x in pot.graph.nodes]
    chain = Chain.model_validate({"nodes": nodes})
    chain.write_to_disk(directory / 'nodes.xyz')

    console.print(f"[bold green]✓ Network summary complete![/bold green]")
    console.print(f"[dim]Network: {pot_fp}[/dim]")
    console.print(f"[dim]Nodes: {directory / 'nodes.xyz'}[/dim]")


if __name__ == "__main__":
    app()
