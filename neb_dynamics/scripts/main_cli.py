from __future__ import annotations
import base64
import contextlib
import io
import json
import logging
from dataclasses import dataclass
import networkx as nx
import numpy as np
from typing import List
from itertools import product
from neb_dynamics.pot import plot_results_from_pot_obj
from neb_dynamics.pot import Pot
from neb_dynamics.helper_functions import (
    compute_irc_chain,
    parse_nma_freq_data,
    parse_terachem_input_file,
    parse_symbols_from_prmtop,
    rst7_to_coords_and_indices,
)
from neb_dynamics.inputs import NetworkInputs, ChainInputs
from neb_dynamics.NetworkBuilder import NetworkBuilder
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file
from neb_dynamics.nodes.nodehelpers import displace_by_dr
from neb_dynamics.msmep import MSMEP
from neb_dynamics.chain import Chain
from neb_dynamics.engines.engine import build_hessian_result_from_matrix
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.neb import NEB
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.nodes.nodehelpers import (
    _is_connectivity_identical,
    is_identical,
)
from neb_dynamics.inputs import RunInputs
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.geodesic_interpolation2.fileio import write_xyz

import typer
from typing_extensions import Annotated

import os
from openbabel import openbabel
from qcio import Structure, ProgramOutput
from qcio.view import generate_structure_viewer_html
from qcop.exceptions import ExternalProgramError
import sys
from pathlib import Path
import time
import traceback
from datetime import datetime
import webbrowser
import shutil
import tomli_w

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
from neb_dynamics.chainhelpers import generate_neb_plot
from neb_dynamics.mepd_drive import launch_mepd_drive
from neb_dynamics.retropaths_workflow import (
    RetropathsWorkspace,
    create_workspace,
    ensure_retropaths_available,
    prepare_neb_workspace,
    refine_drive_workspace_network,
    run_netgen_smiles_workflow,
    summarize_queue,
    write_status_html,
)
from neb_dynamics.scripts._cli_results import (
    _create_recursive_request_record,
    _load_status_snapshot,
    _recursive_split_manifest_path,
    _request_record_summary,
    _resolve_status_artifact,
    _run_status_path,
    _summarize_network_file,
    _upsert_request_record,
    _write_chain_history_with_nan_fallback,
    _write_chain_with_nan_fallback,
    _write_json_atomic,
    _write_neb_results_with_history as _write_neb_results_with_history_impl,
    _write_recursive_split_manifest,
    _write_run_status,
)
from neb_dynamics.scripts._cli_runtime import (
    BANNER,
    _configure_cli_logging,
    console,
    create_progress,
    print_banner,
)
from neb_dynamics.scripts import _cli_visualize
from neb_dynamics.scripts.progress import stop_status

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


def _format_drive_access_panel(
    *,
    actual_host: str,
    actual_port: int,
    ssh_login: str | None = None,
    local_port: int | None = None,
    access_url: str | None = None,
) -> Panel:
    local_port = int(local_port or actual_port)
    ui_url = str(access_url or f"http://{actual_host}:{actual_port}/")
    lines = [
        "[bold cyan]MEPD Drive[/bold cyan]",
        f"[white]{ui_url}[/white]",
        "[dim]Press Ctrl+C to stop the server.[/dim]",
    ]
    if ssh_login:
        lines.extend(
            [
                "",
                "[bold]SSH Tunnel[/bold]",
                f"[white]ssh -N -L {local_port}:127.0.0.1:{actual_port} {ssh_login}[/white]",
                f"[dim]Then open {ui_url.replace(f'http://{actual_host}:{actual_port}/', f'http://127.0.0.1:{local_port}/')} on your laptop.[/dim]",
            ]
        )
    return Panel.fit("\n".join(lines), border_style="cyan")


openbabel.obErrorLog.SetOutputLevel(0)


@contextlib.contextmanager
def _suppress_rdkit_valence_warnings():
    try:
        from rdkit import RDLogger  # type: ignore
    except Exception:
        yield
        return
    with contextlib.suppress(Exception):
        RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            RDLogger.EnableLog("rdApp.*")


def _parse_kmc_initial_condition_overrides(
    overrides: list[str] | None,
) -> dict[int, float] | None:
    if not overrides:
        return None

    parsed: dict[int, float] = {}
    for override in overrides:
        if "=" not in override:
            raise typer.BadParameter(
                f"Invalid --initial-condition '{override}'. Use NODE=VALUE, for example 0=1.0."
            )
        node_text, value_text = override.split("=", 1)
        try:
            node_index = int(node_text.strip())
            value = float(value_text.strip())
        except ValueError as exc:
            raise typer.BadParameter(
                f"Invalid --initial-condition '{override}'. Use NODE=VALUE, for example 0=1.0."
            ) from exc
        parsed[node_index] = value
    return parsed


app = typer.Typer(
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """Show banner before running any command."""
    _configure_cli_logging()
    if ctx.invoked_subcommand is None:
        print_banner()


_VisualizationData = _cli_visualize.VisualizationData
_truncate_label = _cli_visualize._truncate_label
_build_ascii_energy_profile = _cli_visualize._build_ascii_energy_profile


def _visualization_deps() -> _cli_visualize.VisualizationDeps:
    return _cli_visualize.VisualizationDeps(
        Chain=Chain,
        ChainInputs=ChainInputs,
        NEB=NEB,
        Path=Path,
        Pot=Pot,
        Structure=Structure,
        StructureNode=StructureNode,
        TreeNode=TreeNode,
        is_connectivity_identical=_is_connectivity_identical,
        nx=nx,
        np=np,
        read_multiple_structure_from_file=read_multiple_structure_from_file,
        reverse_chain=_reverse_chain,
        concat_chains=_concat_chains,
        collect_tree_layers_for_visualization=_collect_tree_layers_for_visualization,
        match_network_endpoint_indices_by_connectivity=_match_network_endpoint_indices_by_connectivity,
        find_pot_root_node_index=_find_pot_root_node_index,
        find_pot_target_node_index=_find_pot_target_node_index,
        best_path_by_apparent_barrier=_best_path_by_apparent_barrier,
        path_chain_from_pot=_path_chain_from_pot,
        best_chain_for_directed_edge=_best_chain_for_directed_edge,
        load_network_endpoint_hints=_load_network_endpoint_hints,
        load_network_endpoint_structures=_load_network_endpoint_structures,
    )


def _ascii_profile_for_chain(chain: Chain):
    return _cli_visualize._ascii_profile_for_chain(chain, console)


def _write_neb_results_with_history(
    neb_result, fp: Path, write_qcio: bool = False
) -> bool:
    return _write_neb_results_with_history_impl(
        neb_result, fp, console=console, write_qcio=write_qcio
    )


def _collect_tree_layers_for_visualization(tree: TreeNode) -> list[dict]:
    return _cli_visualize._collect_tree_layers_for_visualization(tree)


def _load_visualization_data(
    result_path: Path,
    charge: int = 0,
    multiplicity: int = 1,
) -> _VisualizationData:
    return _cli_visualize._load_visualization_data(
        result_path=result_path,
        deps=_visualization_deps(),
        charge=charge,
        multiplicity=multiplicity,
    )


def _load_chain_for_visualization(
    result_path: Path,
    charge: int = 0,
    multiplicity: int = 1,
) -> Chain:
    return _cli_visualize._load_chain_for_visualization(
        result_path=result_path,
        deps=_visualization_deps(),
        charge=charge,
        multiplicity=multiplicity,
    )


def _generate_opt_history_plot_b64(
    chain_trajectory: list[Chain],
    selected_index: int,
) -> str:
    return ""


def _best_chain_for_directed_edge(pot: Pot, source: int, target: int) -> Chain:
    return _cli_visualize._best_chain_for_directed_edge(
        pot,
        source,
        target,
        _visualization_deps(),
    )


def _best_path_by_apparent_barrier(
    pot: Pot,
    root_idx: int,
    target_idx: int,
) -> tuple[list[int], float] | tuple[None, None]:
    return _cli_visualize._best_path_by_apparent_barrier(
        pot,
        root_idx,
        target_idx,
        _visualization_deps(),
    )


def _find_pot_root_node_index(pot: Pot) -> int | None:
    return _cli_visualize._find_pot_root_node_index(pot)


def _load_network_endpoint_hints(network_json_fp: Path) -> dict | None:
    return _cli_visualize._load_network_endpoint_hints(network_json_fp)


def _load_network_endpoint_structures(
    network_json_fp: Path,
) -> tuple[StructureNode | None, StructureNode | None]:
    return _cli_visualize._load_network_endpoint_structures(
        network_json_fp,
        _visualization_deps(),
    )


def _match_network_endpoint_indices_by_connectivity(
    pot: Pot,
    start_node: StructureNode | None,
    end_node: StructureNode | None,
) -> dict | None:
    return _cli_visualize._match_network_endpoint_indices_by_connectivity(
        pot,
        start_node,
        end_node,
        _visualization_deps(),
    )


def _find_pot_target_node_index(
    pot: Pot,
    target_idx_hint: int | None = None,
) -> int | None:
    return _cli_visualize._find_pot_target_node_index(
        pot,
        _visualization_deps(),
        target_idx_hint,
    )


def _path_chain_from_pot(pot: Pot, path: list[int]) -> Chain | None:
    return _cli_visualize._path_chain_from_pot(pot, path, _visualization_deps())


def _build_network_visualization_payload(
    pot: Pot,
    atom_indices: list[int] | None = None,
    endpoint_hints: dict | None = None,
) -> dict:
    return _cli_visualize._build_network_visualization_payload(
        pot,
        _visualization_deps(),
        atom_indices=atom_indices,
        endpoint_hints=endpoint_hints,
    )


def _build_chain_visualizer_html(
    chain: Chain,
    chain_trajectory: list[Chain] | None = None,
    tree_layers: list[dict] | None = None,
    network_payload: dict | None = None,
) -> str:
    return _cli_visualize._build_chain_visualizer_html(
        chain,
        chain_trajectory=chain_trajectory,
        tree_layers=tree_layers,
        network_payload=network_payload,
    )


def _parse_visualize_atom_indices(
    qminds_fp: str | None = None,
    atom_indices: str | None = None,
) -> list[int] | None:
    return _cli_visualize._parse_visualize_atom_indices(
        qminds_fp=qminds_fp,
        atom_indices=atom_indices,
    )


def _subset_chain_for_visualization(
    chain: Chain,
    atom_indices: list[int],
) -> Chain:
    return _cli_visualize._subset_chain_for_visualization(
        chain,
        atom_indices,
        _visualization_deps(),
    )


def _subset_chain_trajectory_for_visualization(
    chain_trajectory: list[Chain],
    atom_indices: list[int],
) -> list[Chain]:
    return _cli_visualize._subset_chain_trajectory_for_visualization(
        chain_trajectory,
        atom_indices,
        _visualization_deps(),
    )


def _subset_tree_layers_for_visualization(
    tree_layers: list[dict],
    atom_indices: list[int],
) -> list[dict]:
    return _cli_visualize._subset_tree_layers_for_visualization(
        tree_layers,
        atom_indices,
        _visualization_deps(),
    )


def _compute_ts_node(engine, ts_guess: StructureNode, bigchem: bool = False):
    """Run TS optimization through the engine and normalize to (StructureNode|None, ProgramOutput|None)."""
    try:
        if bigchem and hasattr(engine, "_compute_ts_result"):
            raw_out = engine._compute_ts_result(
                node=ts_guess, use_bigchem=True)
            if getattr(raw_out, "success", False):
                return StructureNode(structure=raw_out.return_result), raw_out
            return None, raw_out

        if hasattr(engine, "compute_transition_state"):
            raw_out = engine.compute_transition_state(node=ts_guess)
            if isinstance(raw_out, StructureNode):
                return raw_out, None
            if isinstance(raw_out, ProgramOutput):
                if getattr(raw_out, "success", False):
                    return StructureNode(structure=raw_out.return_result), raw_out
                return None, raw_out
            if getattr(raw_out, "success", False) and getattr(raw_out, "return_result", None) is not None:
                return StructureNode(structure=raw_out.return_result), raw_out
            return None, raw_out

        if hasattr(engine, "_compute_ts_result"):
            raw_out = engine._compute_ts_result(
                node=ts_guess, use_bigchem=bigchem)
            if getattr(raw_out, "success", False):
                return StructureNode(structure=raw_out.return_result), raw_out
            return None, raw_out

        raise AttributeError(
            "Engine does not implement transition-state optimization.")
    except Exception as exc:
        program_output = getattr(exc, "program_output", None)
        if program_output is not None:
            return None, program_output
        raise


def _extract_normal_modes_from_hessian_result(
    hessres,
) -> tuple[list[np.ndarray], list[float]]:
    results = getattr(hessres, "results", None)
    if results is None:
        raise ValueError("Hessian result is missing `results`.")

    modes = getattr(results, "normal_modes_cartesian", None)
    freqs = getattr(results, "freqs_wavenumber", None)
    if modes is not None and len(modes) > 0:
        normal_modes = [np.array(mode) for mode in modes]
        frequencies = [float(freq) for freq in (freqs or [])]
        return normal_modes, frequencies

    normal_modes, frequencies = parse_nma_freq_data(hessres)
    if len(normal_modes) == 0:
        raise ValueError("No normal modes found in Hessian result.")
    return [np.array(mode) for mode in normal_modes], [float(freq) for freq in frequencies]


def _compute_hessian_result_for_sampling(engine, node: StructureNode):
    if hasattr(engine, "_compute_hessian_result"):
        return engine._compute_hessian_result(node)
    hessian = np.asarray(engine.compute_hessian(node), dtype=float)
    return build_hessian_result_from_matrix(node=node, hessian=hessian)


def _resolve_command_base_path(geometry: str, name: str | None) -> Path:
    if name is None:
        return Path.cwd() / Path(geometry).stem
    raw = Path(name)
    if raw.suffix:
        raw = raw.with_suffix("")
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    return raw


def _extract_minima_nodes(history: TreeNode) -> list[StructureNode]:
    """Extract minima candidates from recursive cheap-history leaves."""
    minima: list[StructureNode] = []
    for leaf in history.ordered_leaves:
        if not leaf.data or not leaf.data.chain_trajectory:
            continue
        final_chain = leaf.data.chain_trajectory[-1]
        if len(final_chain.nodes) == 0:
            continue
        minima.append(final_chain[0].copy())
        minima.append(final_chain[-1].copy())
    return minima


def _extract_minima_nodes_from_chain(chain: Chain) -> list[StructureNode]:
    """Extract endpoints + strict local minima from a single optimized chain."""
    if len(chain.nodes) == 0:
        return []
    if len(chain.nodes) <= 2:
        return [node.copy() for node in chain.nodes]
    energies = chain.energies
    minima_inds = {0, len(chain.nodes) - 1}
    for i in range(1, len(chain.nodes) - 1):
        if energies[i] < energies[i - 1] and energies[i] < energies[i + 1]:
            minima_inds.add(i)
    return [chain.nodes[i].copy() for i in sorted(minima_inds)]


def _dedupe_minima_nodes(nodes: list[StructureNode], chain_inputs: ChainInputs) -> list[StructureNode]:
    """Drop duplicate minima by geometry/graph identity."""
    unique_nodes: list[StructureNode] = []
    for node in nodes:
        duplicate = False
        for existing in unique_nodes:
            if is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False,
            ):
                duplicate = True
                break
        if not duplicate:
            unique_nodes.append(node.copy())
    return unique_nodes


def _load_best_path_chain_from_network_splits(
    *,
    network_fp: Path | None,
    output_dir: Path,
    base_name: str,
) -> Chain | None:
    if network_fp is None or not Path(network_fp).exists():
        return None
    best_path_fp = output_dir / f"{base_name}_best_path.json"
    if not best_path_fp.exists():
        return None
    try:
        payload = json.loads(best_path_fp.read_text(encoding="utf-8"))
        path = [int(v) for v in payload.get("path") or []]
        if len(path) < 2:
            return None
        pot = Pot.read_from_disk(network_fp)
        return _path_chain_from_pot(pot, path)
    except Exception:
        return None


def _dedupe_minima_and_sources(
    minima: list[StructureNode],
    sources: list[StructureNode],
    chain_inputs: ChainInputs,
) -> tuple[list[StructureNode], list[StructureNode]]:
    """Deduplicate minima while preserving same-index source mapping."""
    if len(minima) != len(sources):
        raise ValueError("minima and sources must have equal lengths.")

    unique_minima: list[StructureNode] = []
    unique_sources: list[StructureNode] = []
    for node, source in zip(minima, sources):
        duplicate = False
        for existing in unique_minima:
            if is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False,
            ):
                duplicate = True
                break
        if not duplicate:
            unique_minima.append(node.copy())
            unique_sources.append(source.copy())
    return unique_minima, unique_sources


def _clear_node_cached_properties(node: StructureNode) -> StructureNode:
    clean = node.copy()
    clean._cached_result = None
    clean._cached_energy = None
    clean._cached_gradient = None
    return clean


def _clear_chain_cached_properties(chain: Chain, parameters: ChainInputs) -> Chain:
    """Return chain copy with cached energies/gradients removed from every node."""
    clean_nodes = [_clear_node_cached_properties(node) for node in chain.nodes]
    return Chain.model_validate({"nodes": clean_nodes, "parameters": parameters})


def _find_matching_node_index(
    target: StructureNode,
    chain: Chain,
    chain_inputs: ChainInputs,
) -> int | None:
    try:
        dists = [np.linalg.norm(node.coords - target.coords)
                 for node in chain.nodes]
        if len(dists) == 0:
            return None
        closest_idx = int(np.argmin(dists))
        if dists[closest_idx] < 1e-8:
            return closest_idx
    except Exception:
        dists = []

    candidates: list[tuple[int, float]] = []
    for i, node in enumerate(chain.nodes):
        if is_identical(
            target,
            node,
            fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
            kcal_mol_cutoff=chain_inputs.node_ene_thre,
            verbose=False,
        ):
            dist = np.linalg.norm(node.coords - target.coords)
            candidates.append((i, dist))

    if candidates:
        return min(candidates, key=lambda t: t[1])[0]

    if len(dists) > 0:
        return int(np.argmin(dists))
    return None


def _build_recycled_pair_chain(
    cheap_output_chain: Chain,
    cheap_start_ref: StructureNode,
    cheap_end_ref: StructureNode,
    expensive_start: StructureNode,
    expensive_end: StructureNode,
    cheap_chain_inputs: ChainInputs,
    expensive_chain_inputs: ChainInputs,
    expected_nimages: int,
) -> Chain | None:
    start_idx = _find_matching_node_index(
        cheap_start_ref, cheap_output_chain, cheap_chain_inputs
    )
    end_idx = _find_matching_node_index(
        cheap_end_ref, cheap_output_chain, cheap_chain_inputs
    )
    if start_idx is None or end_idx is None or start_idx == end_idx:
        return None

    if start_idx < end_idx:
        segment_nodes = [node.copy()
                         for node in cheap_output_chain.nodes[start_idx:end_idx + 1]]
    else:
        segment_nodes = [
            node.copy() for node in cheap_output_chain.nodes[end_idx:start_idx + 1]][::-1]

    if len(segment_nodes) != expected_nimages:
        return None

    segment_nodes[0] = expensive_start.copy()
    segment_nodes[-1] = expensive_end.copy()
    recycled = Chain.model_validate(
        {"nodes": segment_nodes, "parameters": expensive_chain_inputs}
    )
    return _clear_chain_cached_properties(recycled, expensive_chain_inputs)


def _reverse_chain(chain: Chain) -> Chain:
    return Chain.model_validate(
        {"nodes": [node.copy() for node in chain.nodes[::-1]],
         "parameters": chain.parameters}
    )


def _concat_chains(chains: list[Chain], parameters: ChainInputs) -> Chain:
    if len(chains) == 0:
        raise ValueError("Cannot concatenate an empty list of chains.")
    nodes = []
    for i, chain in enumerate(chains):
        chain_nodes = chain.nodes if i == 0 else chain.nodes[1:]
        nodes.extend([node.copy() for node in chain_nodes])
    return Chain.model_validate({"nodes": nodes, "parameters": parameters})


@dataclass
class _QueuedRecursivePairRequest:
    request_id: int
    start_node: StructureNode
    end_node: StructureNode
    start_index: int
    end_index: int
    parent_request_id: int | None = None


def _find_registered_node_index(
    node: StructureNode,
    registry: list[StructureNode],
    chain_inputs: ChainInputs,
) -> int | None:
    for i, existing in enumerate(registry):
        try:
            if is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False,
            ):
                return i
        except Exception:
            if (
                list(node.structure.symbols) == list(
                    existing.structure.symbols)
                and np.allclose(node.coords, existing.coords)
            ):
                return i
    return None


def _register_recursive_split_node(
    node: StructureNode,
    registry: list[StructureNode],
    chain_inputs: ChainInputs,
) -> int:
    existing_index = _find_registered_node_index(
        node=node, registry=registry, chain_inputs=chain_inputs
    )
    if existing_index is not None:
        return existing_index
    registry.append(node.copy())
    return len(registry) - 1


def _ordered_leaf_path_nodes(
    history: TreeNode, chain_inputs: ChainInputs
) -> list[StructureNode]:
    path_nodes: list[StructureNode] = []
    for leaf in history.ordered_leaves:
        if not leaf.data or not leaf.data.chain_trajectory:
            continue
        final_chain = leaf.data.chain_trajectory[-1]
        if len(final_chain.nodes) == 0:
            continue
        start_node = final_chain[0].copy()
        end_node = final_chain[-1].copy()
        if not path_nodes:
            path_nodes.append(start_node)
        elif not is_identical(
            path_nodes[-1],
            start_node,
            fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
            kcal_mol_cutoff=chain_inputs.node_ene_thre,
            verbose=False,
        ):
            path_nodes.append(start_node)
        path_nodes.append(end_node)
    return path_nodes


def _queue_follow_on_recursive_requests(
    path_nodes: list[StructureNode],
    target_node: StructureNode,
    parent_request_id: int,
    next_request_id: int,
    chain_inputs: ChainInputs,
    node_registry: list[StructureNode],
    attempted_pairs: set[tuple[int, int]],
) -> tuple[list[_QueuedRecursivePairRequest], int]:
    queued: list[_QueuedRecursivePairRequest] = []
    if len(path_nodes) < 4:
        return queued, next_request_id

    target_index = _register_recursive_split_node(
        node=target_node, registry=node_registry, chain_inputs=chain_inputs
    )
    for intermediate in path_nodes[1:-2]:
        start_index = _register_recursive_split_node(
            node=intermediate, registry=node_registry, chain_inputs=chain_inputs
        )
        pair_key = (start_index, target_index)
        if pair_key in attempted_pairs:
            continue
        attempted_pairs.add(pair_key)
        queued.append(
            _QueuedRecursivePairRequest(
                request_id=next_request_id,
                start_node=intermediate.copy(),
                end_node=target_node.copy(),
                start_index=start_index,
                end_index=target_index,
                parent_request_id=parent_request_id,
            )
        )
        next_request_id += 1
    return queued, next_request_id


def _mark_path_pairs_attempted(
    path_nodes: list[StructureNode],
    *,
    chain_inputs: ChainInputs,
    node_registry: list[StructureNode],
    attempted_pairs: set[tuple[int, int]],
) -> None:
    if len(path_nodes) < 2:
        return
    path_indices = [
        _register_recursive_split_node(
            node=node, registry=node_registry, chain_inputs=chain_inputs
        )
        for node in path_nodes
    ]
    for start_index, end_index in zip(path_indices[:-1], path_indices[1:]):
        attempted_pairs.add((start_index, end_index))


def _write_recursive_split_request_artifacts(
    output_dir: Path,
    request_id: int,
    history: TreeNode,
    write_qcio: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_chain: Chain | None = None
    leaf_chains: list[Chain] = []
    for leaf in history.ordered_leaves:
        if not leaf.data:
            continue
        if getattr(leaf.data, "chain_trajectory", None):
            leaf_chains.append(leaf.data.chain_trajectory[-1])
        elif getattr(leaf.data, "optimized", None) is not None:
            leaf_chains.append(leaf.data.optimized)
    if len(leaf_chains) == 1:
        output_chain = leaf_chains[0]
    elif len(leaf_chains) > 1:
        output_chain = _concat_chains(leaf_chains, leaf_chains[0].parameters)
    elif history.data is not None:
        if getattr(history.data, "chain_trajectory", None):
            output_chain = history.data.chain_trajectory[-1]
        elif getattr(history.data, "optimized", None) is not None:
            output_chain = history.data.optimized

    if output_chain is None:
        raise ValueError("Recursive split request history produced no output chain.")

    output_chain.write_to_disk(
        output_dir / f"request_{request_id}.xyz", write_qcio=write_qcio)
    history.write_to_disk(
        output_dir / f"request_{request_id}_msmep", write_qcio=write_qcio)


def _load_recursive_split_request_history(
    output_dir: Path,
    request_id: int,
    *,
    chain_inputs: ChainInputs,
    engine,
    charge: int,
    multiplicity: int,
) -> TreeNode | None:
    request_dir = Path(output_dir) / f"request_{request_id}_msmep"
    if not request_dir.is_dir():
        return None
    try:
        return TreeNode.read_from_disk(
            request_dir,
            chain_parameters=chain_inputs,
            engine=engine,
            charge=charge,
            multiplicity=multiplicity,
        )
    except Exception:
        return None


def _maybe_resume_recursive_history(
    status_fp: Path,
    *,
    chain_inputs: ChainInputs,
    engine,
    charge: int,
    multiplicity: int,
) -> TreeNode | None:
    if not Path(status_fp).exists():
        return None
    try:
        snapshot = _load_status_snapshot(str(status_fp))
        run_status = snapshot.get("run_status") or {}
        if str(run_status.get("phase") or "") not in {"network_splits", "complete"}:
            return None
        tree_path = run_status.get("tree_path")
        if not tree_path:
            return None
        tree_dir = Path(tree_path)
        if not tree_dir.exists():
            return None
        return TreeNode.read_from_disk(
            tree_dir,
            chain_parameters=chain_inputs,
            engine=engine,
            charge=charge,
            multiplicity=multiplicity,
        )
    except Exception:
        return None


def _build_recursive_split_network_summary(
    output_dir: Path,
    base_name: str,
    chain_inputs: ChainInputs,
    root_index: int | None = None,
    target_index: int | None = None,
    root_node: StructureNode | None = None,
    target_node: StructureNode | None = None,
    verbose: bool = False,
) -> Path | None:
    nb = NetworkBuilder(
        data_dir=output_dir,
        start=None,
        end=None,
        network_inputs=NetworkInputs(verbose=verbose),
        chain_inputs=chain_inputs,
    )
    nb.msmep_data_dir = output_dir

    msmep_paths = [p for p in output_dir.glob("*_msmep") if p.is_dir()]
    if not msmep_paths:
        return None

    pot = nb.create_rxn_network(file_pattern="*_msmep")
    matched_indices = _match_network_endpoint_indices_by_connectivity(
        pot,
        start_node=root_node,
        end_node=target_node,
    )
    resolved_root_index = (
        matched_indices.get("root_index")
        if matched_indices and matched_indices.get("root_index") is not None
        else root_index
    )
    resolved_target_index = (
        matched_indices.get("target_index")
        if matched_indices and matched_indices.get("target_index") is not None
        else target_index
    )
    for node_idx in pot.graph.nodes:
        pot.graph.nodes[node_idx]["root"] = int(node_idx) == int(
            resolved_root_index) if resolved_root_index is not None else bool(pot.graph.nodes[node_idx].get("root"))
        pot.graph.nodes[node_idx]["requested_target"] = int(node_idx) == int(
            resolved_target_index) if resolved_target_index is not None else bool(pot.graph.nodes[node_idx].get("requested_target"))
    pot_fp = output_dir / f"{base_name}_network.json"
    pot.write_to_disk(pot_fp)

    try:
        plot_results_from_pot_obj(
            fp_out=(output_dir / f"{base_name}_network.png"),
            pot=pot,
            include_pngs=True,
        )
        plot_results_from_pot_obj(
            fp_out=(output_dir / f"{base_name}_network.png"),
            pot=pot,
            include_pngs=False,
        )
    except Exception:
        console.print(
            "[yellow]⚠ Failed to generate network plots. Continuing with JSON only.[/yellow]"
        )

    try:
        nodes = [pot.graph.nodes[x]["td"] for x in pot.graph.nodes]
        chain = Chain.model_validate({"nodes": nodes})
        chain.write_to_disk(output_dir / f"{base_name}_network_nodes.xyz")
    except Exception:
        console.print(
            "[yellow]⚠ Failed to export network node geometries. Continuing.[/yellow]"
        )

    try:
        best_path_nodes, _ = _best_path_by_apparent_barrier(
            pot,
            root_idx=resolved_root_index,
            target_idx=resolved_target_index,
        ) if resolved_root_index is not None and resolved_target_index is not None else (None, None)
        if best_path_nodes:
            best_path_chain = _path_chain_from_pot(pot, best_path_nodes)
            if best_path_chain is not None:
                _write_chain_with_nan_fallback(
                    best_path_chain,
                    output_dir / f"{base_name}_best_path.xyz",
                )
                _write_json_atomic(
                    output_dir / f"{base_name}_best_path.json",
                    {
                        "root_index": int(resolved_root_index),
                        "target_index": int(resolved_target_index),
                        "path": [int(v) for v in best_path_nodes],
                    },
                )
    except Exception:
        console.print(
            "[yellow]⚠ Failed to export best network path chain. Continuing.[/yellow]"
        )

    return pot_fp


def _run_recursive_network_splits(
    history: TreeNode,
    program_input: RunInputs,
    initial_start: StructureNode,
    initial_end: StructureNode,
    output_dir: Path,
    base_name: str,
    status_fp: Path | None = None,
    parallel_recursive: bool = False,
    parallel_workers: int | None = None,
) -> tuple[list[dict], Path | None, Path]:
    manifest_fp = _recursive_split_manifest_path(
        output_dir=output_dir, base_name=base_name)
    resume_mode = output_dir.exists() and (
        manifest_fp.exists() or (output_dir / "request_0_msmep").is_dir()
    )
    if output_dir.exists() and not resume_mode:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_mode:
        console.print(
            f"[bold cyan]Resuming network splits from {output_dir}[/bold cyan]"
        )

    attempted_pairs: set[tuple[int, int]] = set()
    node_registry: list[StructureNode] = []
    request_records: list[dict] = []
    network_fp: Path | None = None
    write_qcio = bool(getattr(program_input, "write_qcio", False))
    charge = int(initial_start.structure.charge)
    multiplicity = int(initial_start.structure.multiplicity)

    start_index = _register_recursive_split_node(
        node=initial_start, registry=node_registry, chain_inputs=program_input.chain_inputs
    )
    end_index = _register_recursive_split_node(
        node=initial_end, registry=node_registry, chain_inputs=program_input.chain_inputs
    )
    attempted_pairs.add((start_index, end_index))

    initial_history = (
        _load_recursive_split_request_history(
            output_dir,
            0,
            chain_inputs=program_input.chain_inputs,
            engine=program_input.engine,
            charge=charge,
            multiplicity=multiplicity,
        )
        if resume_mode
        else None
    ) or history
    if not (output_dir / "request_0_msmep").is_dir():
        _write_recursive_split_request_artifacts(
            output_dir=output_dir,
            request_id=0,
            history=initial_history,
            write_qcio=write_qcio,
        )
    initial_path_nodes = _ordered_leaf_path_nodes(
        history=initial_history, chain_inputs=program_input.chain_inputs
    )
    _mark_path_pairs_attempted(
        initial_path_nodes,
        chain_inputs=program_input.chain_inputs,
        node_registry=node_registry,
        attempted_pairs=attempted_pairs,
    )
    _upsert_request_record(
        request_records,
        _create_recursive_request_record(
            request_id=0,
            parent_request_id=None,
            start_index=start_index,
            end_index=end_index,
            status="completed",
            n_path_nodes=len(initial_path_nodes),
            completed_at=datetime.now().isoformat(),
        ),
    )
    network_fp = _build_recursive_split_network_summary(
        output_dir=output_dir,
        base_name=base_name,
        chain_inputs=program_input.chain_inputs,
        root_index=start_index,
        target_index=end_index,
        root_node=initial_start,
        target_node=initial_end,
    )
    manifest_fp = _write_recursive_split_manifest(
        output_dir=output_dir,
        base_name=base_name,
        request_records=request_records,
        run_state="running",
        current_request_id=None,
        network_fp=network_fp,
    )
    if status_fp is not None:
        _write_run_status(
            status_fp,
            base_name=base_name,
            run_state="running",
            phase="network_splits",
            network_splits_dir=output_dir,
            manifest_fp=manifest_fp,
            network_fp=network_fp,
        )

    queue, next_request_id = _queue_follow_on_recursive_requests(
        path_nodes=initial_path_nodes,
        target_node=initial_end,
        parent_request_id=0,
        next_request_id=1,
        chain_inputs=program_input.chain_inputs,
        node_registry=node_registry,
        attempted_pairs=attempted_pairs,
    )
    for request in queue:
        _upsert_request_record(
            request_records,
            _create_recursive_request_record(
                request_id=request.request_id,
                parent_request_id=request.parent_request_id,
                start_index=request.start_index,
                end_index=request.end_index,
                status="queued",
                queued_at=datetime.now().isoformat(),
            ),
        )
    manifest_fp = _write_recursive_split_manifest(
        output_dir=output_dir,
        base_name=base_name,
        request_records=request_records,
        run_state="running",
        current_request_id=None,
        network_fp=network_fp,
    )
    if status_fp is not None:
        _write_run_status(
            status_fp,
            base_name=base_name,
            run_state="running",
            phase="network_splits",
            network_splits_dir=output_dir,
            manifest_fp=manifest_fp,
            network_fp=network_fp,
        )

    msmep_runner = MSMEP(inputs=program_input)
    parallel_recursive = bool(parallel_recursive)
    resolved_parallel_workers = (
        None if parallel_workers is None else max(1, int(parallel_workers))
    )
    while queue:
        request = queue.pop(0)
        existing_history = _load_recursive_split_request_history(
            output_dir,
            request.request_id,
            chain_inputs=program_input.chain_inputs,
            engine=program_input.engine,
            charge=charge,
            multiplicity=multiplicity,
        )
        if existing_history is None:
            _upsert_request_record(
                request_records,
                _create_recursive_request_record(
                    request_id=request.request_id,
                    parent_request_id=request.parent_request_id,
                    start_index=request.start_index,
                    end_index=request.end_index,
                    status="running",
                    started_at=datetime.now().isoformat(),
                ),
            )
            manifest_fp = _write_recursive_split_manifest(
                output_dir=output_dir,
                base_name=base_name,
                request_records=request_records,
                run_state="running",
                current_request_id=request.request_id,
                network_fp=network_fp,
            )
            if status_fp is not None:
                _write_run_status(
                    status_fp,
                    base_name=base_name,
                    run_state="running",
                    phase="network_splits",
                    network_splits_dir=output_dir,
                    manifest_fp=manifest_fp,
                    network_fp=network_fp,
                )
        request_chain = Chain.model_validate(
            {
                "nodes": [request.start_node.copy(), request.end_node.copy()],
                "parameters": program_input.chain_inputs,
            }
        )
        if existing_history is None:
            try:
                if parallel_recursive:
                    request_history = msmep_runner.run_parallel_recursive_minimize(
                        request_chain,
                        max_workers=resolved_parallel_workers,
                    )
                else:
                    request_history = msmep_runner.run_recursive_minimize(
                        request_chain)
            except Exception:
                _upsert_request_record(
                    request_records,
                    _create_recursive_request_record(
                        request_id=request.request_id,
                        parent_request_id=request.parent_request_id,
                        start_index=request.start_index,
                        end_index=request.end_index,
                        status="failed",
                        completed_at=datetime.now().isoformat(),
                        error=traceback.format_exc().strip(),
                    ),
                )
                manifest_fp = _write_recursive_split_manifest(
                    output_dir=output_dir,
                    base_name=base_name,
                    request_records=request_records,
                    run_state="running",
                    current_request_id=None,
                    network_fp=network_fp,
                )
                continue
        else:
            request_history = existing_history

        if not request_history.data:
            _upsert_request_record(
                request_records,
                _create_recursive_request_record(
                    request_id=request.request_id,
                    parent_request_id=request.parent_request_id,
                    start_index=request.start_index,
                    end_index=request.end_index,
                    status="empty",
                    completed_at=datetime.now().isoformat(),
                ),
            )
            manifest_fp = _write_recursive_split_manifest(
                output_dir=output_dir,
                base_name=base_name,
                request_records=request_records,
                run_state="running",
                current_request_id=None,
                network_fp=network_fp,
            )
            continue

        if existing_history is None:
            _write_recursive_split_request_artifacts(
                output_dir=output_dir,
                request_id=request.request_id,
                history=request_history,
                write_qcio=write_qcio,
            )
        request_path_nodes = _ordered_leaf_path_nodes(
            history=request_history, chain_inputs=program_input.chain_inputs
        )
        _mark_path_pairs_attempted(
            request_path_nodes,
            chain_inputs=program_input.chain_inputs,
            node_registry=node_registry,
            attempted_pairs=attempted_pairs,
        )
        _upsert_request_record(
            request_records,
            _create_recursive_request_record(
                request_id=request.request_id,
                parent_request_id=request.parent_request_id,
                start_index=request.start_index,
                end_index=request.end_index,
                status="completed",
                n_path_nodes=len(request_path_nodes),
                completed_at=datetime.now().isoformat(),
            ),
        )
        new_requests, next_request_id = _queue_follow_on_recursive_requests(
            path_nodes=request_path_nodes,
            target_node=request.end_node,
            parent_request_id=request.request_id,
            next_request_id=next_request_id,
            chain_inputs=program_input.chain_inputs,
            node_registry=node_registry,
            attempted_pairs=attempted_pairs,
        )
        for new_request in new_requests:
            _upsert_request_record(
                request_records,
                _create_recursive_request_record(
                    request_id=new_request.request_id,
                    parent_request_id=new_request.parent_request_id,
                    start_index=new_request.start_index,
                    end_index=new_request.end_index,
                    status="queued",
                    queued_at=datetime.now().isoformat(),
                ),
            )
        queue.extend(new_requests)
        network_fp = _build_recursive_split_network_summary(
            output_dir=output_dir,
            base_name=base_name,
            chain_inputs=program_input.chain_inputs,
            root_index=start_index,
            target_index=end_index,
            root_node=initial_start,
            target_node=initial_end,
        )
        manifest_fp = _write_recursive_split_manifest(
            output_dir=output_dir,
            base_name=base_name,
            request_records=request_records,
            run_state="running",
            current_request_id=None,
            network_fp=network_fp,
        )
        if status_fp is not None:
            _write_run_status(
                status_fp,
                base_name=base_name,
                run_state="running",
                phase="network_splits",
                network_splits_dir=output_dir,
                manifest_fp=manifest_fp,
                network_fp=network_fp,
            )

    network_fp = _build_recursive_split_network_summary(
        output_dir=output_dir,
        base_name=base_name,
        chain_inputs=program_input.chain_inputs,
        root_index=start_index,
        target_index=end_index,
        root_node=initial_start,
        target_node=initial_end,
    )
    manifest_fp = _write_recursive_split_manifest(
        output_dir=output_dir,
        base_name=base_name,
        request_records=request_records,
        run_state="completed",
        current_request_id=None,
        network_fp=network_fp,
    )
    if status_fp is not None:
        _write_run_status(
            status_fp,
            base_name=base_name,
            run_state="running",
            phase="network_splits",
            network_splits_dir=output_dir,
            manifest_fp=manifest_fp,
            network_fp=network_fp,
        )
    return request_records, network_fp, manifest_fp


def _section_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"value": obj}


def _flatten_params(data, prefix=""):
    if isinstance(data, dict):
        rows = []
        for key in sorted(data.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_params(data[key], next_prefix))
        return rows
    return [(prefix if prefix else "value", _format_param_value(data))]


def _format_param_value(value, max_str: int = 160):
    if isinstance(value, list):
        if len(value) > 20:
            head = ", ".join(str(v) for v in value[:8])
            tail = ", ".join(str(v) for v in value[-3:])
            return f"[{head}, ..., {tail}] (n={len(value)})"
        return str(value)
    s = str(value)
    if len(s) > max_str:
        return s[: max_str - 3] + "..."
    return s


def _render_runinputs(program_input: RunInputs):
    table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
    table.add_column("Section", style="bold cyan", no_wrap=True)
    table.add_column("Key", style="magenta")
    table.add_column("Value", style="white")

    sections = [
        ("General", {
            "engine_name": program_input.engine_name,
            "program": program_input.program,
            "path_min_method": program_input.path_min_method,
        }),
        ("QMMM", _section_dict(program_input.qmmm_inputs)
         if getattr(program_input, "qmmm_inputs", None) else {}),
        ("Path Minimizer", _section_dict(program_input.path_min_inputs)),
        ("Chain", _section_dict(program_input.chain_inputs)),
        ("GI", _section_dict(program_input.gi_inputs)),
        ("Program Args", _section_dict(program_input.program_kwds)),
        ("Optimizer", _section_dict(program_input.optimizer_kwds)),
    ]

    for section_name, section_data in sections:
        flat_rows = _flatten_params(section_data)
        for i, (key, value) in enumerate(flat_rows):
            table.add_row(section_name if i == 0 else "", key, value)

    console.print(
        Panel(
            table,
            title="[bold cyan]Input Parameters[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def _load_endpoint_structure(
    path: str,
    charge: int,
    multiplicity: int,
    rst7_prmtop_text: str | None = None,
) -> Structure:
    fp = Path(path)
    is_rst7 = fp.suffix.lower() == ".rst7"
    if is_rst7:
        if rst7_prmtop_text is None:
            raise ValueError(
                "RST7 endpoint conversion requires --rst7-prmtop."
            )
        coords, _ = rst7_to_coords_and_indices(fp.read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            symbols = parse_symbols_from_prmtop(rst7_prmtop_text)
        return Structure(
            geometry=coords * ANGSTROM_TO_BOHR,
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
        )
    return Structure.open(path)


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
        parallel: Annotated[bool, typer.Option(
            "--parallel",
            help="Run recursive autosplitting in parallel with bounded worker concurrency.",
        )] = False,
        parallel_workers: Annotated[int | None, typer.Option(
            "--parallel-workers",
            help="Maximum number of concurrent recursive split workers used by --parallel. Defaults to min(4, CPU count).",
        )] = None,
        name: str = None,
        charge: int = 0,
        multiplicity: int = 1,
        rst7_prmtop: Annotated[str, typer.Option(
            "--rst7-prmtop",
            help="Path to AMBER prmtop used to map atomic symbols when converting rst7 endpoints.",
        )] = None,
        network_splits: Annotated[bool, typer.Option(
            "--network-splits",
            help="After a recursive MSMEP run, enqueue intermediate-to-target follow-on runs and build a reaction network from all attempted pairs.",
        )] = False,
        create_irc: Annotated[bool, typer.Option(
            help='whether to run and output an IRC chain. Need to set --use_tsopt also, otherwise\
                will attempt use the guess structure.')] = False,
        use_bigchem: Annotated[bool, typer.Option(
            help='whether to use chemcloud to compute hessian for ts opt and irc jobs')] = False):

    # Print header
    console.print(BANNER)

    if parallel and recursive:
        raise typer.BadParameter(
            "--parallel cannot be combined with --recursive. Use one mode."
        )

    cpu_cap = max(1, int(os.cpu_count() or 1))
    default_parallel_workers = min(4, cpu_cap)
    if parallel_workers is None:
        parallel_workers = default_parallel_workers
    if parallel_workers < 1:
        raise typer.BadParameter("--parallel-workers must be at least 1.")
    if parallel_workers > cpu_cap:
        console.print(
            f"[yellow]⚠ Requested {parallel_workers} parallel workers exceeds detected CPU count ({cpu_cap}). Continuing with requested value; tune based on your host capacity.[/yellow]"
        )

    if network_splits and not recursive and not parallel:
        console.print(
            Panel(
                "[bold yellow]--network-splits requires recursive MSMEP.[/bold yellow]\n"
                "[bold cyan]Automatically enabling --recursive for this run.[/bold cyan]",
                title="[bold yellow]Recursive Mode Forced[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        recursive = True

    table = Table(box=None, show_header=False)
    table.add_column(style="dim")
    mode_label = "parallel" if parallel else ("recursive" if recursive else "regular")
    table.add_row("[bold cyan]Command:[/bold cyan]", "[white]run[/white]")
    table.add_row("[bold cyan]Method:[/bold cyan]",
                  f"[yellow]{mode_label}[/yellow]")
    table.add_row("[bold cyan]SMILES mode:[/bold cyan]",
                  f"[yellow]{use_smiles}[/yellow]")
    table.add_row("[bold cyan]Parallel:[/bold cyan]",
                  f"[yellow]{parallel}[/yellow]")
    if parallel:
        table.add_row("[bold cyan]Parallel workers:[/bold cyan]",
                      f"[yellow]{parallel_workers}[/yellow]")
    table.add_row("[bold cyan]Network splits:[/bold cyan]",
                  f"[yellow]{network_splits}[/yellow]")
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
                endpoint_paths = [Path(start), Path(end)]
                needs_rst7_prmtop = any(
                    fp.suffix.lower() == ".rst7" for fp in endpoint_paths
                )
                if needs_rst7_prmtop and rst7_prmtop is None:
                    console.print(
                        "[bold red]✗ ERROR:[/bold red] .rst7 endpoints require --rst7-prmtop."
                    )
                    raise typer.Exit(1)
                prmtop_text = Path(rst7_prmtop).read_text(
                ) if needs_rst7_prmtop else None
                start_ref = _load_endpoint_structure(
                    start,
                    charge=charge,
                    multiplicity=multiplicity,
                    rst7_prmtop_text=prmtop_text,
                )
                end_ref = _load_endpoint_structure(
                    end,
                    charge=charge,
                    multiplicity=multiplicity,
                    rst7_prmtop_text=prmtop_text,
                )

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

    _render_runinputs(program_input)
    sys.stdout.flush()
    write_qcio = bool(getattr(program_input, "write_qcio", False))

    # minimize endpoints if requested
    if program_input.engine.__class__.__name__ == "QMMMEngine":
        all_nodes = [program_input.engine._make_structure_node(s) for s in all_structs]
    else:
        all_nodes = [StructureNode(structure=s) for s in all_structs]
    if minimize_ends:
        console.print("[bold cyan]⟳ Minimizing input endpoints...[/bold cyan]")
        batch_optimizer = getattr(
            program_input.engine, "compute_geometry_optimizations", None)
        if callable(batch_optimizer):
            console.print(
                "[dim]Submitting batched endpoint geometry optimizations...[/dim]")
            try:
                trajectories = batch_optimizer(
                    [all_nodes[0], all_nodes[-1]],
                    keywords={'coordsys': 'cart', 'maxiter': 500},
                )
                all_nodes[0] = trajectories[0][-1]
                all_nodes[-1] = trajectories[-1][-1]
            except TypeError:
                trajectories = batch_optimizer([all_nodes[0], all_nodes[-1]])
                all_nodes[0] = trajectories[0][-1]
                all_nodes[-1] = trajectories[-1][-1]
        else:
            console.print("[dim]Minimizing start endpoint...[/dim]")
            start_tr = program_input.engine.compute_geometry_optimization(
                all_nodes[0], keywords={'coordsys': 'cart', 'maxiter': 500})
            all_nodes[0] = start_tr[-1]
            console.print("[dim]Minimizing end endpoint...[/dim]")
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
    fp = Path("mep_output")

    if parallel:
        if name is not None:
            name = Path(name)
            data_dir = Path(name).resolve().parent
            foldername = data_dir / name.stem
            filename = data_dir / (name.stem + ".xyz")
        else:
            data_dir = Path(os.getcwd())
            foldername = data_dir / f"{fp.stem}_parallel_msmep"
            filename = data_dir / f"{fp.stem}_parallel_msmep.xyz"
        status_fp = _run_status_path(data_dir, filename.stem)
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="running",
            phase="parallel_recursive_request",
            recursive=False,
            parallel=True,
            network_splits=network_splits,
            path_min_method=str(program_input.path_min_method),
        )

        if not program_input.path_min_inputs.do_elem_step_checks:
            console.print(
                "[yellow]⚠ WARNING: do_elem_step_checks is set to False. This may cause issues with recursive splitting.[/yellow]")
            console.print(
                "[yellow]Making it True to ensure proper functioning of recursive splitting.[/yellow]")
            program_input.path_min_inputs.do_elem_step_checks = True

        console.print(
            f"[bold magenta]▶ RUNNING PARALLEL AUTOSPLITTING {program_input.path_min_method} (max_workers={parallel_workers})[/bold magenta]"
        )
        try:
            history = m.run_parallel_recursive_minimize(
                chain,
                max_workers=parallel_workers,
            )
        except KeyboardInterrupt:
            stop_status()
            raise
        except BaseException:
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="failed",
                phase="parallel_recursive_request",
                recursive=False,
                parallel=True,
                network_splits=network_splits,
                path_min_method=str(program_input.path_min_method),
                error=traceback.format_exc().strip(),
            )
            stop_status()
            raise

        if not history.data:
            leaf_status = str(getattr(history, "leaf_status", "") or "unknown")
            if leaf_status == "identical_endpoints":
                empty_history_msg = (
                    "Program did not run because endpoints were classified as identical "
                    "under current thresholds (node_rms_thre/node_ene_thre)."
                )
            elif leaf_status == "electronic_structure_error":
                empty_history_msg = (
                    "Program did not run because electronic structure evaluation failed "
                    "during recursive minimization."
                )
            else:
                empty_history_msg = (
                    "Program did not run. Likely because your endpoints are conformers "
                    "of the same molecular graph."
                )
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="failed",
                phase="parallel_recursive_request",
                recursive=False,
                parallel=True,
                network_splits=network_splits,
                path_min_method=str(program_input.path_min_method),
                error=f"{empty_history_msg} leaf_status={leaf_status}",
            )
            stop_status()
            console.print(
                f"[bold red]✗ ERROR:[/bold red] {empty_history_msg} "
                "Tighten node_rms_thre/node_ene_thre in chain_inputs and try again."
            )
            raise typer.Exit(1)

        successful_leaf_chains = []
        for leaf in history.ordered_leaves:
            if not leaf.data:
                continue
            leaf_chain = None
            if getattr(leaf.data, "chain_trajectory", None):
                leaf_chain = leaf.data.chain_trajectory[-1]
            elif getattr(leaf.data, "optimized", None) is not None:
                leaf_chain = leaf.data.optimized
            if leaf_chain is not None:
                successful_leaf_chains.append(leaf_chain)

        parallel_failures = list(getattr(history, "parallel_failures", []) or [])

        if not successful_leaf_chains:
            root_chain = None
            if history.data is not None:
                if getattr(history.data, "chain_trajectory", None):
                    root_chain = history.data.chain_trajectory[-1]
                elif getattr(history.data, "optimized", None) is not None:
                    root_chain = history.data.optimized

            if root_chain is not None:
                console.print(
                    "[yellow]⚠ Parallel autosplitting produced no successful child leaves; "
                    "falling back to the root optimized chain.[/yellow]"
                )
                successful_leaf_chains = [root_chain]
            else:
                _write_run_status(
                    status_fp,
                    base_name=filename.stem,
                    run_state="failed",
                    phase="parallel_recursive_request",
                    recursive=False,
                    parallel=True,
                    network_splits=network_splits,
                    path_min_method=str(program_input.path_min_method),
                    error="Parallel autosplitting produced no successful leaf chains.",
                )
                console.print(
                    "[bold red]✗ ERROR:[/bold red] Parallel autosplitting did not yield any successful leaf chains."
                )
                if parallel_failures:
                    console.print(
                        f"[yellow]Captured {len(parallel_failures)} branch failure(s). "
                        "Showing the first one below.[/yellow]"
                    )
                    console.print(f"[dim]{parallel_failures[0]}[/dim]")
                raise typer.Exit(1)

        if len(successful_leaf_chains) == 1:
            merged_chain = successful_leaf_chains[0]
        else:
            merged_chain = _concat_chains(
                successful_leaf_chains, program_input.chain_inputs
            )

        if parallel_failures:
            console.print(
                f"[yellow]⚠ {len(parallel_failures)} branch worker failure(s) occurred during "
                "parallel autosplitting; recovered branches were retained where possible.[/yellow]"
            )
            max_shown = min(3, len(parallel_failures))
            console.print(
                f"[yellow]Showing {max_shown} branch failure detail(s):[/yellow]"
            )
            for i, failure_text in enumerate(parallel_failures[:max_shown], start=1):
                console.print(f"[dim][parallel-failure {i}] {failure_text}[/dim]")
        identical_skipped_leaves = sum(
            1
            for leaf in history.depth_first_ordered_nodes
            if leaf.is_leaf
            and not bool(leaf.data)
            and getattr(leaf, "leaf_status", "") == "identical_endpoints"
        )
        failed_leaves = sum(
            1
            for leaf in history.depth_first_ordered_nodes
            if leaf.is_leaf
            and not bool(leaf.data)
            and getattr(leaf, "leaf_status", "") != "identical_endpoints"
        )
        if identical_skipped_leaves > 0:
            console.print(
                f"[yellow]⚠ {identical_skipped_leaves} parallel branch(es) were skipped because endpoints were identical.[/yellow]"
            )
        if failed_leaves > 0:
            console.print(
                f"[yellow]⚠ {failed_leaves} parallel branch(es) failed. Writing a partial merged chain from successful leaves.[/yellow]"
            )

        leaves_nebs = [obj for obj in history.get_optimization_history() if obj]
        end_time = time.time()
        merged_chain.write_to_disk(filename, write_qcio=write_qcio)
        history.write_to_disk(foldername, write_qcio=write_qcio)
        chain_for_profile = merged_chain
        if network_splits:
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="running",
                phase="network_splits",
                recursive=False,
                parallel=True,
                network_splits=True,
                path_min_method=str(program_input.path_min_method),
                output_chain_path=filename,
                tree_path=foldername,
            )
            network_dir = data_dir / f"{filename.stem}_network_splits"
            console.print(
                "[bold magenta]▶ RUNNING FOLLOW-ON NETWORK SPLIT REQUESTS[/bold magenta]"
            )
            request_records, network_fp, manifest_fp = _run_recursive_network_splits(
                history=history,
                program_input=program_input,
                initial_start=chain[0],
                initial_end=chain[-1],
                output_dir=network_dir,
                base_name=filename.stem,
                status_fp=status_fp,
                parallel_recursive=True,
                parallel_workers=parallel_workers,
            )
            console.print(
                f"[cyan]Completed {len(request_records)} total recursive pair requests.[/cyan]"
            )
            if network_fp is not None:
                console.print(
                    f"[cyan]Network summary written to {network_fp}[/cyan]"
                )
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="completed",
                phase="complete",
                recursive=False,
                parallel=True,
                network_splits=True,
                path_min_method=str(program_input.path_min_method),
                output_chain_path=filename,
                tree_path=foldername,
                network_splits_dir=network_dir,
                manifest_fp=manifest_fp,
                network_fp=network_fp,
            )
        else:
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="completed",
                phase="complete",
                recursive=False,
                parallel=True,
                network_splits=False,
                path_min_method=str(program_input.path_min_method),
                output_chain_path=filename,
                tree_path=foldername,
            )

        if use_tsopt:
            for i, leaf in enumerate(history.ordered_leaves):
                if not leaf.data:
                    continue
                if not leaf.data.chain_trajectory:
                    console.print(
                        f"[yellow]⚠ Skipping TS optimization on leaf {i}: no chain trajectory.[/yellow]"
                    )
                    continue
                console.print(
                    f"[bold cyan]⟳ Running TS opt on leaf {i}...[/bold cyan]")
                try:
                    ts_node, tsres = _compute_ts_node(
                        engine=program_input.engine,
                        ts_guess=leaf.data.chain_trajectory[-1].get_ts_node(),
                    )
                except Exception:
                    console.print(
                        f"[yellow]⚠ TS optimization crashed on leaf {i}: {traceback.format_exc()}[/yellow]"
                    )
                    continue

                if tsres is not None and hasattr(tsres, "save"):
                    tsres.save(data_dir / (filename.stem+f"_tsres_{i}.qcio"))
                if ts_node is not None:
                    ts_node.structure.save(
                        data_dir / (filename.stem+f"_ts_{i}.xyz"))
                    if create_irc:
                        try:
                            irc = compute_irc_chain(
                                ts_node=ts_node,
                                engine=program_input.engine,
                            )
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

        tot_grad_calls = sum(getattr(obj, "grad_calls_made", 0)
                             for obj in leaves_nebs)
        geom_grad_calls = sum(
            getattr(obj, "geom_grad_calls_made", 0) for obj in leaves_nebs
        )
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {tot_grad_calls} gradient calls total.[/cyan]")
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {geom_grad_calls} gradient for geometry optimizations.[/cyan]")

    elif recursive:
        if name is not None:
            name = Path(name)
            data_dir = Path(name).resolve().parent
            foldername = data_dir / name.stem
            filename = data_dir / (name.stem + ".xyz")
        else:
            data_dir = Path(os.getcwd())
            foldername = data_dir / f"{fp.stem}_msmep"
            filename = data_dir / f"{fp.stem}_msmep.xyz"
        status_fp = _run_status_path(data_dir, filename.stem)
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="running",
            phase="initial_recursive_request",
            recursive=True,
            parallel=False,
            network_splits=network_splits,
            path_min_method=str(program_input.path_min_method),
        )

        if not program_input.path_min_inputs.do_elem_step_checks:
            console.print(
                "[yellow]⚠ WARNING: do_elem_step_checks is set to False. This may cause issues with recursive splitting.[/yellow]")
            console.print(
                "[yellow]Making it True to ensure proper functioning of recursive splitting.[/yellow]")
            program_input.path_min_inputs.do_elem_step_checks = True
        console.print(
            f"[bold magenta]▶ RUNNING AUTOSPLITTING {program_input.path_min_method}[/bold magenta]")
        history = (
            _maybe_resume_recursive_history(
                status_fp,
                chain_inputs=program_input.chain_inputs,
                engine=program_input.engine,
                charge=int(chain[0].structure.charge),
                multiplicity=int(chain[0].structure.multiplicity),
            )
            if network_splits
            else None
        )
        if history is not None:
            console.print(
                f"[bold cyan]Resuming saved recursive history from {foldername}[/bold cyan]"
            )
        else:
            try:
                history = m.run_recursive_minimize(chain)
            except KeyboardInterrupt:
                stop_status()
                raise
            except BaseException:
                _write_run_status(
                    status_fp,
                    base_name=filename.stem,
                    run_state="failed",
                    phase="initial_recursive_request",
                    recursive=True,
                    parallel=False,
                    network_splits=network_splits,
                    path_min_method=str(program_input.path_min_method),
                    error=traceback.format_exc().strip(),
                )
                stop_status()
                raise

        if not history.data:
            leaf_status = str(getattr(history, "leaf_status", "") or "unknown")
            if leaf_status == "identical_endpoints":
                empty_history_msg = (
                    "Program did not run because endpoints were classified as identical "
                    "under current thresholds (node_rms_thre/node_ene_thre)."
                )
            elif leaf_status == "electronic_structure_error":
                empty_history_msg = (
                    "Program did not run because electronic structure evaluation failed "
                    "during recursive minimization."
                )
            else:
                empty_history_msg = (
                    "Program did not run. Likely because your endpoints are conformers "
                    "of the same molecular graph."
                )
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="failed",
                phase="initial_recursive_request",
                recursive=True,
                parallel=False,
                network_splits=network_splits,
                path_min_method=str(program_input.path_min_method),
                error=f"{empty_history_msg} leaf_status={leaf_status}",
            )
            stop_status()
            console.print(
                f"[bold red]✗ ERROR:[/bold red] {empty_history_msg} "
                "Tighten node_rms_thre/node_ene_thre in chain_inputs and try again."
            )
            raise typer.Exit(1)

        leaves_nebs = [
            obj for obj in history.get_optimization_history() if obj]
        end_time = time.time()
        history.output_chain.write_to_disk(filename, write_qcio=write_qcio)
        history.write_to_disk(foldername, write_qcio=write_qcio)
        chain_for_profile = history.output_chain
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="running" if network_splits else "completed",
            phase="network_splits" if network_splits else "complete",
            recursive=True,
            parallel=False,
            network_splits=network_splits,
            path_min_method=str(program_input.path_min_method),
            output_chain_path=filename,
            tree_path=foldername,
        )

        if network_splits:
            network_dir = data_dir / f"{filename.stem}_network_splits"
            console.print(
                "[bold magenta]▶ RUNNING FOLLOW-ON NETWORK SPLIT REQUESTS[/bold magenta]"
            )
            request_records, network_fp, manifest_fp = _run_recursive_network_splits(
                history=history,
                program_input=program_input,
                initial_start=chain[0],
                initial_end=chain[-1],
                output_dir=network_dir,
                base_name=filename.stem,
                status_fp=status_fp,
            )
            console.print(
                f"[cyan]Completed {len(request_records)} total recursive pair requests.[/cyan]"
            )
            if network_fp is not None:
                console.print(
                    f"[cyan]Network summary written to {network_fp}[/cyan]"
                )
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="completed",
                phase="complete",
                recursive=True,
                parallel=False,
                network_splits=network_splits,
                path_min_method=str(program_input.path_min_method),
                output_chain_path=filename,
                tree_path=foldername,
                network_splits_dir=network_dir,
                manifest_fp=manifest_fp,
                network_fp=network_fp,
            )

        if use_tsopt:
            for i, leaf in enumerate(history.ordered_leaves):
                if not leaf.data:
                    continue
                if not leaf.data.chain_trajectory:
                    console.print(
                        f"[yellow]⚠ Skipping TS optimization on leaf {i}: no chain trajectory.[/yellow]"
                    )
                    continue
                console.print(
                    f"[bold cyan]⟳ Running TS opt on leaf {i}...[/bold cyan]")
                try:
                    ts_node, tsres = _compute_ts_node(
                        engine=program_input.engine,
                        ts_guess=leaf.data.chain_trajectory[-1].get_ts_node(),
                    )
                except Exception:
                    console.print(
                        f"[yellow]⚠ TS optimization crashed on leaf {i}: {traceback.format_exc()}[/yellow]"
                    )
                    continue

                if tsres is not None and hasattr(tsres, "save"):
                    tsres.save(data_dir / (filename.stem+f"_tsres_{i}.qcio"))
                if ts_node is not None:
                    ts_node.structure.save(
                        data_dir / (filename.stem+f"_ts_{i}.xyz"))
                    if create_irc:
                        try:
                            irc = compute_irc_chain(
                                ts_node=ts_node,
                                engine=program_input.engine,
                            )
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

        tot_grad_calls = sum(getattr(obj, "grad_calls_made", 0)
                             for obj in leaves_nebs)
        geom_grad_calls = sum(
            getattr(obj, "geom_grad_calls_made", 0) for obj in leaves_nebs
        )
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {tot_grad_calls} gradient calls total.[/cyan]")
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {geom_grad_calls} gradient for geometry optimizations.[/cyan]")

    else:
        data_dir = Path(os.getcwd())
        if name is not None:
            filename = data_dir / (name + ".xyz")
        else:
            filename = data_dir / f"{fp.stem}_neb.xyz"
        status_fp = _run_status_path(data_dir, filename.stem)
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="running",
            phase="path_minimization",
            recursive=False,
            parallel=False,
            network_splits=False,
            path_min_method=str(program_input.path_min_method),
        )
        console.print(
            f"[bold magenta]▶ RUNNING REGULAR {program_input.path_min_method}[/bold magenta]")
        try:
            n, elem_step_results = m.run_minimize_chain(input_chain=chain)
        except Exception:
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="failed",
                phase="path_minimization",
                recursive=False,
                parallel=False,
                network_splits=False,
                path_min_method=str(program_input.path_min_method),
                error=traceback.format_exc().strip(),
            )
            raise

        end_time = time.time()
        try:
            wrote_outputs = _write_neb_results_with_history(
                n, filename, write_qcio=write_qcio
            )
        except TypeError:
            wrote_outputs = _write_neb_results_with_history(n, filename)
        if n.chain_trajectory:
            chain_for_profile = n.chain_trajectory[-1]
        elif n.optimized is not None:
            chain_for_profile = n.optimized

        if not wrote_outputs:
            console.print(
                "[yellow]⚠ Skipping output write/profile because path minimization did not produce an optimized chain.[/yellow]"
            )
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="completed",
            phase="complete",
            recursive=False,
            parallel=False,
            network_splits=False,
            path_min_method=str(program_input.path_min_method),
            output_chain_path=filename,
        )

        if use_tsopt and n.optimized is not None:
            console.print("[bold cyan]⟳ Running TS opt...[/bold cyan]")
            try:
                source_chain = n.chain_trajectory[-1] if n.chain_trajectory else n.optimized
                ts_node, tsres = _compute_ts_node(
                    engine=program_input.engine,
                    ts_guess=source_chain.get_ts_node(),
                )
            except Exception:
                console.print(
                    f"[yellow]⚠ TS optimization crashed: {traceback.format_exc()}[/yellow]"
                )
                ts_node, tsres = None, None
            if tsres is not None and hasattr(tsres, "save"):
                tsres.save(data_dir / (filename.stem+"_tsres.qcio"))
            if ts_node is not None:
                ts_node.structure.save(
                    data_dir / (filename.stem+"_ts.xyz"))

                if create_irc:
                    try:
                        irc = compute_irc_chain(
                            ts_node=ts_node, engine=program_input.engine
                        )
                        irc.write_to_disk(
                            filename.stem+"_tsres_IRC.xyz")

                    except Exception:
                        console.print(
                            f"[yellow]⚠ IRC failed: {traceback.format_exc()}[/yellow]")
                        console.print(
                            "[yellow]IRC failed. Continuing...[/yellow]")

            else:
                console.print("[yellow]⚠ TS optimization failed.[/yellow]")
        elif use_tsopt:
            console.print(
                "[yellow]⚠ Skipping TS optimization because path minimization did not converge.[/yellow]"
            )

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


@app.command("run-refine")
def run_refine(
        start: Annotated[str, typer.Option(
            help='path to start file, or a reactant smiles')] = None,
        end: Annotated[str, typer.Option(
            help='path to end file, or a product smiles')] = None,
        geometries:  Annotated[str, typer.Argument(help='file containing an approximate path between two endpoints. \
            Use this if you have precompted a path you want to use. Do not use this with smiles.')] = None,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to expensive RunInputs toml file')] = None,
        cheap_inputs: Annotated[str, typer.Option("--cheap-inputs", "-ci",
                                                  help='optional path to cheaper RunInputs toml file for initial discovery')] = None,
        recycle_nodes: Annotated[bool, typer.Option(
            "--recycle-nodes",
            help="Reuse cheap-stage path nodes as initial guess for expensive pair refinement.",
        )] = False,
        network_splits: Annotated[bool, typer.Option(
            "--network-splits",
            help="For recursive cheap discovery, run follow-on split requests and refine only the best path through the resulting network.",
        )] = False,
        use_smiles: bool = False,
        recursive: bool = False,
        minimize_ends: bool = False,
        name: str = None,
        charge: int = 0,
        multiplicity: int = 1,
        rst7_prmtop: Annotated[str, typer.Option(
            "--rst7-prmtop",
            help="Path to AMBER prmtop used to map atomic symbols when converting rst7 endpoints.",
        )] = None):
    """Two-stage refinement: cheap discovery -> expensive minima/path refinement."""
    console.print(BANNER)

    if inputs is None:
        console.print(
            "[bold red]✗ ERROR:[/bold red] --inputs/-i is required for run-refine."
        )
        raise typer.Exit(1)

    if network_splits and not recursive:
        recursive = True

    start_time = time.time()
    table = Table(box=None, show_header=False)
    table.add_column(style="dim")
    table.add_row("[bold cyan]Command:[/bold cyan]",
                  "[white]run-refine[/white]")
    table.add_row("[bold cyan]SMILES mode:[/bold cyan]",
                  f"[yellow]{use_smiles}[/yellow]")
    table.add_row("[bold cyan]Method:[/bold cyan]",
                  f"[yellow]{'recursive' if recursive else 'regular'}[/yellow]")
    table.add_row("[bold cyan]Network splits:[/bold cyan]",
                  f"[yellow]{network_splits}[/yellow]")
    table.add_row("[bold cyan]Cheap Inputs:[/bold cyan]",
                  f"[yellow]{cheap_inputs if cheap_inputs else inputs}[/yellow]")
    table.add_row("[bold cyan]Expensive Inputs:[/bold cyan]",
                  f"[yellow]{inputs}[/yellow]")
    table.add_row("[bold cyan]Recycle Nodes:[/bold cyan]",
                  f"[yellow]{recycle_nodes}[/yellow]")
    console.print(table)
    console.print()

    # load structures
    if use_smiles:
        from neb_dynamics.nodes.nodehelpers import create_pairs_from_smiles
        from neb_dynamics.arbalign import align_structures

        console.print(
            "[yellow]⚠ WARNING:[/yellow] Using RXNMapper to create atomic mapping. Carefully check output to see how labels affected reaction path.")
        with console.status("[bold cyan]Creating structures from SMILES...[/bold cyan]"):
            start_structure, end_structure = create_pairs_from_smiles(
                smi1=start, smi2=end)
            end_structure = align_structures(
                start_structure, end_structure, distance_metric='RMSD')
        all_structs = [start_structure, end_structure]
    else:
        if geometries is not None:
            with console.status(f"[bold cyan]Loading geometries from {geometries}...[/bold cyan]"):
                try:
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=charge, spinmult=multiplicity)
                except ValueError:
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=None, spinmult=None)
        elif start is not None and end is not None:
            with console.status(f"[bold cyan]Loading structures...[/bold cyan]"):
                endpoint_paths = [Path(start), Path(end)]
                needs_rst7_prmtop = any(
                    fp.suffix.lower() == ".rst7" for fp in endpoint_paths
                )
                if needs_rst7_prmtop and rst7_prmtop is None:
                    console.print(
                        "[bold red]✗ ERROR:[/bold red] .rst7 endpoints require --rst7-prmtop."
                    )
                    raise typer.Exit(1)
                prmtop_text = Path(rst7_prmtop).read_text(
                ) if needs_rst7_prmtop else None
                start_ref = _load_endpoint_structure(
                    start,
                    charge=charge,
                    multiplicity=multiplicity,
                    rst7_prmtop_text=prmtop_text,
                )
                end_ref = _load_endpoint_structure(
                    end,
                    charge=charge,
                    multiplicity=multiplicity,
                    rst7_prmtop_text=prmtop_text,
                )
                if start_ref.charge != charge or start_ref.multiplicity != multiplicity:
                    start_ref = Structure(
                        geometry=start_ref.geometry,
                        charge=charge,
                        multiplicity=multiplicity,
                        symbols=start_ref.symbols,
                    )
                if end_ref.charge != charge or end_ref.multiplicity != multiplicity:
                    end_ref = Structure(
                        geometry=end_ref.geometry,
                        charge=charge,
                        multiplicity=multiplicity,
                        symbols=end_ref.symbols,
                    )
                all_structs = [start_ref, end_ref]
        else:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Either 'geometries' or 'start' and 'end' flags must be populated!")
            raise typer.Exit(1)

    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        expensive_input = RunInputs.open(inputs)
        cheap_input = RunInputs.open(
            cheap_inputs) if cheap_inputs else RunInputs.open(inputs)

    console.print("[bold cyan]Cheap-level Inputs[/bold cyan]")
    _render_runinputs(cheap_input)
    console.print("[bold cyan]Expensive-level Inputs[/bold cyan]")
    _render_runinputs(expensive_input)

    if cheap_input.engine.__class__.__name__ == "QMMMEngine":
        all_nodes = [cheap_input.engine._make_structure_node(s) for s in all_structs]
    else:
        all_nodes = [StructureNode(structure=s) for s in all_structs]
    if minimize_ends:
        console.print(
            "[bold cyan]⟳ Minimizing input endpoints at cheap level...[/bold cyan]")
        start_tr = cheap_input.engine.compute_geometry_optimization(
            all_nodes[0], keywords={'coordsys': 'cart', 'maxiter': 500})
        all_nodes[0] = start_tr[-1]
        end_tr = cheap_input.engine.compute_geometry_optimization(
            all_nodes[-1], keywords={'coordsys': 'cart', 'maxiter': 500})
        all_nodes[-1] = end_tr[-1]

    cheap_chain = Chain.model_validate(
        {"nodes": all_nodes, "parameters": cheap_input.chain_inputs}
    )
    cheap_msmep = MSMEP(inputs=cheap_input)

    console.print(
        f"[bold magenta]▶ CHEAP DISCOVERY RUN ({cheap_input.path_min_method})[/bold magenta]"
    )
    if recursive:
        if not cheap_input.path_min_inputs.do_elem_step_checks:
            console.print(
                "[yellow]⚠ WARNING: do_elem_step_checks is False with --recursive. Setting it to True.[/yellow]"
            )
            cheap_input.path_min_inputs.do_elem_step_checks = True
        cheap_history = cheap_msmep.run_recursive_minimize(cheap_chain)
        if not cheap_history.data:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Cheap run returned no valid history."
            )
            raise typer.Exit(1)
        cheap_output_chain = cheap_history.output_chain
        cheap_minima = _extract_minima_nodes(cheap_history)
    else:
        cheap_neb, _ = cheap_msmep.run_minimize_chain(cheap_chain)
        cheap_output_chain = cheap_neb.chain_trajectory[-1] if cheap_neb.chain_trajectory else cheap_neb.optimized
        if cheap_output_chain is None:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Cheap run produced no optimized chain."
            )
            raise typer.Exit(1)
        cheap_history = None
        cheap_minima = _extract_minima_nodes_from_chain(cheap_output_chain)

    base_name = name if name is not None else "mep_output"
    data_dir = Path(os.getcwd())
    cheap_chain_fp = data_dir / f"{base_name}_cheap.xyz"
    cheap_tree_dir = data_dir / f"{base_name}_cheap_msmep"
    cheap_output_chain.write_to_disk(cheap_chain_fp)
    if cheap_history is not None:
        cheap_history.write_to_disk(cheap_tree_dir)

    if recursive and network_splits and cheap_history is not None:
        network_dir = data_dir / f"{base_name}_cheap_network_splits"
        console.print(
            "[bold magenta]▶ CHEAP DISCOVERY NETWORK SPLITS[/bold magenta]"
        )
        _request_records, network_fp, _manifest_fp = _run_recursive_network_splits(
            history=cheap_history,
            program_input=cheap_input,
            initial_start=cheap_chain[0],
            initial_end=cheap_chain[-1],
            output_dir=network_dir,
            base_name=f"{base_name}_cheap",
            status_fp=None,
        )
        best_path_chain = _load_best_path_chain_from_network_splits(
            network_fp=network_fp,
            output_dir=network_dir,
            base_name=f"{base_name}_cheap",
        )
        if best_path_chain is None:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Network splits did not produce a best path for refinement."
            )
            raise typer.Exit(1)
        cheap_output_chain = best_path_chain
        cheap_minima = [node.copy() for node in best_path_chain.nodes]
        console.print(
            f"[cyan]Using best network path with {len(cheap_minima)} nodes for expensive refinement.[/cyan]"
        )

    cheap_minima = _dedupe_minima_nodes(cheap_minima, cheap_input.chain_inputs)
    console.print(
        f"[cyan]Discovered {len(cheap_minima)} unique cheap minima (including endpoints).[/cyan]"
    )

    console.print(
        "[bold magenta]▶ REOPTIMIZING MINIMA AT EXPENSIVE LEVEL[/bold magenta]")
    refined_minima: list[StructureNode] = []
    refined_source_minima: list[StructureNode] = []
    dropped_count = 0
    kept_unoptimized_count = 0
    for i, node in enumerate(cheap_minima):
        optimized_successfully = False
        try:
            try:
                traj = expensive_input.engine.compute_geometry_optimization(
                    node, keywords={'coordsys': 'cart', 'maxiter': 500}
                )
            except TypeError:
                traj = expensive_input.engine.compute_geometry_optimization(
                    node)
            opt_node = traj[-1]
            optimized_successfully = True
        except Exception:
            console.print(
                f"[yellow]⚠ Failed to optimize minimum {i}; keeping cheap-level geometry for refinement.[/yellow]"
            )
            console.print(
                f"[yellow]Reason:[/yellow] {traceback.format_exc().strip()}"
            )
            opt_node = _clear_node_cached_properties(node)
            kept_unoptimized_count += 1

        if optimized_successfully and node.has_molecular_graph and opt_node.has_molecular_graph:
            same_connectivity = _is_connectivity_identical(
                node, opt_node, verbose=False
            )
            if not same_connectivity:
                dropped_count += 1
                continue
        refined_minima.append(opt_node)
        refined_source_minima.append(node.copy())

    refined_minima, refined_source_minima = _dedupe_minima_and_sources(
        refined_minima, refined_source_minima, expensive_input.chain_inputs
    )
    if len(refined_minima) < 2:
        console.print(
            "[bold red]✗ ERROR:[/bold red] Fewer than 2 minima remain after expensive-level refinement."
        )
        raise typer.Exit(1)

    refined_minima_chain = Chain.model_validate(
        {"nodes": refined_minima, "parameters": expensive_input.chain_inputs}
    )
    refined_minima_fp = data_dir / f"{base_name}_refined_minima.xyz"
    refined_minima_chain.write_to_disk(refined_minima_fp)
    console.print(
        f"[cyan]Retained {len(refined_minima)} minima, dropped {dropped_count} due to connectivity changes, kept {kept_unoptimized_count} without expensive optimization.[/cyan]"
    )

    console.print(
        "[bold magenta]▶ EXPENSIVE PAIRWISE PATH MINIMIZATION[/bold magenta]")
    pair_dir = data_dir / f"{base_name}_refined_pairs"
    pair_dir.mkdir(exist_ok=True)
    expensive_msmep = MSMEP(inputs=expensive_input)
    if recursive and not expensive_input.path_min_inputs.do_elem_step_checks:
        console.print(
            "[yellow]⚠ WARNING: expensive do_elem_step_checks is False with --recursive. Setting it to True.[/yellow]"
        )
        expensive_input.path_min_inputs.do_elem_step_checks = True

    pair_chains: list[Chain] = []
    pair_inds = list(zip(range(len(refined_minima) - 1),
                     range(1, len(refined_minima))))
    for i, j in pair_inds:
        endpoint_pair = Chain.model_validate(
            {"nodes": [refined_minima[i], refined_minima[j]],
                "parameters": expensive_input.chain_inputs}
        )
        pair = endpoint_pair
        if recycle_nodes:
            recycled_pair = _build_recycled_pair_chain(
                cheap_output_chain=cheap_output_chain,
                cheap_start_ref=refined_source_minima[i],
                cheap_end_ref=refined_source_minima[j],
                expensive_start=refined_minima[i],
                expensive_end=refined_minima[j],
                cheap_chain_inputs=cheap_input.chain_inputs,
                expensive_chain_inputs=expensive_input.chain_inputs,
                expected_nimages=expensive_input.gi_inputs.nimages,
            )
            if recycled_pair is not None:
                pair = recycled_pair
            else:
                console.print(
                    f"[yellow]⚠ Could not recycle cheap nodes for pair ({i}, {j}); using fresh interpolation.[/yellow]"
                )
        try:
            if recursive:
                pair_history = expensive_msmep.run_recursive_minimize(pair)
                if not pair_history.data:
                    console.print(
                        f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
                    )
                    continue
                out_chain = pair_history.output_chain
                pair_history.write_to_disk(pair_dir / f"pair_{i}_{j}_msmep")
            else:
                neb_obj, _ = expensive_msmep.run_minimize_chain(pair)
                out_chain = neb_obj.chain_trajectory[-1] if neb_obj.chain_trajectory else neb_obj.optimized
            if out_chain is None:
                console.print(
                    f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
                )
                continue
            pair_chains.append(out_chain)
            out_chain.write_to_disk(pair_dir / f"pair_{i}_{j}.xyz")
        except Exception:
            console.print(
                f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
            )
            continue

    if len(pair_chains) == 0:
        console.print(
            "[bold red]✗ ERROR:[/bold red] No expensive sequential pair paths converged."
        )
        raise typer.Exit(1)

    refined_chain = _concat_chains(
        pair_chains, expensive_input.chain_inputs)
    refined_chain_fp = data_dir / f"{base_name}_refined.xyz"
    refined_chain.write_to_disk(refined_chain_fp)

    elapsed = time.time() - start_time
    summary = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row(
        "⏱ Walltime:", f"[yellow]{elapsed/60:.1f} min[/yellow]" if elapsed > 60 else f"[yellow]{elapsed:.1f} s[/yellow]")
    summary.add_row("📁 Cheap chain:", f"[cyan]{cheap_chain_fp}[/cyan]")
    summary.add_row("📁 Refined minima:", f"[cyan]{refined_minima_fp}[/cyan]")
    summary.add_row("📁 Refined chain:", f"[cyan]{refined_chain_fp}[/cyan]")
    summary.add_row("🔗 Pair sequence:",
                    f"[white]{pair_inds}[/white]")
    console.print(Panel(
        summary, title="[bold green]✓ run-refine Complete![/bold green]", border_style="green"))

    _ascii_profile_for_chain(refined_chain)


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
            ts_node, output = _compute_ts_node(
                engine=program_input.engine,
                ts_guess=node,
                bigchem=bigchem,
            )

        except Exception:
            console.print(
                f"[bold red]✗ TS optimization failed:[/bold red] {traceback.format_exc()}"
            )
            raise typer.Exit(1)

    if output is not None and hasattr(output, "save"):
        output.save(results_name)
        console.print(f"[dim]Results: {results_name}[/dim]")
    if ts_node is None:
        console.print(
            "[bold red]✗ TS optimization did not converge.[/bold red]")
        raise typer.Exit(1)
    ts_node.structure.save(filename)
    console.print(f"[bold green]✓ TS optimization complete![/bold green]")
    console.print(f"[dim]Geometry: {filename}[/dim]")


@app.command("hessian-sample")
def hessian_sample(
    geometry: Annotated[str, typer.Argument(help="Path to input geometry file.")],
    inputs: Annotated[str, typer.Option(
        "--inputs", "-i", help="Path to RunInputs TOML file.")] = None,
    name: Annotated[str, typer.Option(
        "--name", help="Optional output basename (without extension).")] = None,
    charge: Annotated[int, typer.Option(
        "--charge", help="Total charge for input geometry.")] = 0,
    multiplicity: Annotated[int, typer.Option(
        "--multiplicity", help="Spin multiplicity for input geometry.")] = 1,
    dr: Annotated[float, typer.Option(
        "--dr", help="Displacement magnitude along each normal mode.")] = 0.1,
    max_candidates: Annotated[int, typer.Option(
        "--max-candidates", help="Maximum number of displaced structures to optimize.")] = 100,
    maxiter: Annotated[int, typer.Option(
        "--maxiter", help="Maximum geometry-optimization steps for each displaced structure.")] = 500,
):
    console.print(BANNER)

    if max_candidates < 1:
        raise typer.BadParameter("--max-candidates must be at least 1.")
    if dr <= 0:
        raise typer.BadParameter("--dr must be positive.")
    if maxiter < 1:
        raise typer.BadParameter("--maxiter must be at least 1.")

    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        if inputs is not None:
            program_input = RunInputs.open(inputs)
        else:
            program_input = RunInputs(program="xtb", engine_name="qcop")

    _render_runinputs(program_input)
    write_qcio = bool(getattr(program_input, "write_qcio", False))

    struct = Structure.open(geometry)
    s_dict = struct.model_dump()
    s_dict["charge"], s_dict["multiplicity"] = charge, multiplicity
    struct = Structure(**s_dict)
    node = StructureNode(structure=struct)

    engine = program_input.engine
    if not (hasattr(engine, "_compute_hessian_result") or hasattr(engine, "compute_hessian")):
        console.print(
            "[bold red]✗ ERROR:[/bold red] This engine does not expose Hessian computation "
            "(`_compute_hessian_result` or `compute_hessian`), which is required for normal-mode sampling."
        )
        raise typer.Exit(1)

    with console.status("[bold cyan]Computing Hessian...[/bold cyan]"):
        try:
            hessres = _compute_hessian_result_for_sampling(engine, node)
        except Exception as exc:
            hessres = getattr(exc, "program_output", None)
            if hessres is None:
                console.print(
                    f"[bold red]✗ Hessian computation failed:[/bold red] {traceback.format_exc()}")
                raise typer.Exit(1)

    try:
        normal_modes, frequencies = _extract_normal_modes_from_hessian_result(
            hessres)
    except Exception:
        console.print(
            f"[bold red]✗ Failed to extract normal modes:[/bold red] {traceback.format_exc()}")
        raise typer.Exit(1)

    if len(normal_modes) == 0:
        console.print(
            "[bold red]✗ ERROR:[/bold red] No normal modes were returned from Hessian computation.")
        raise typer.Exit(1)

    base = _resolve_command_base_path(geometry=geometry, name=name)
    base.parent.mkdir(parents=True, exist_ok=True)
    hessian_fp = base.parent / f"{base.stem}_hessian.qcio"
    displaced_fp = base.parent / f"{base.stem}_hessian_sample_displaced.xyz"
    optimized_fp = base.parent / f"{base.stem}_hessian_sample_optimized.xyz"
    unique_fp = base.parent / f"{base.stem}_hessian_sample_unique.xyz"
    summary_fp = base.parent / f"{base.stem}_hessian_sample_summary.json"

    if hasattr(hessres, "save"):
        hessres.save(hessian_fp)

    displaced_nodes: list[StructureNode] = []
    displaced_metadata: list[dict] = []
    clipped = False
    for mode_index, mode in enumerate(normal_modes):
        freq = float(frequencies[mode_index]) if mode_index < len(
            frequencies) else None
        for direction, signed_dr in (("+", dr), ("-", -dr)):
            displaced = displace_by_dr(
                node=node, displacement=np.array(mode), dr=signed_dr)
            displaced_nodes.append(displaced)
            displaced_metadata.append(
                {
                    "mode_index": int(mode_index),
                    "direction": direction,
                    "frequency_wavenumber": freq,
                    "dr": float(abs(signed_dr)),
                }
            )
            if len(displaced_nodes) >= max_candidates:
                clipped = True
                break
        if clipped:
            break

    if len(displaced_nodes) == 0:
        console.print(
            "[bold red]✗ ERROR:[/bold red] No displaced candidates were generated.")
        raise typer.Exit(1)

    displaced_chain = Chain.model_validate(
        {"nodes": [cand.copy() for cand in displaced_nodes],
         "parameters": program_input.chain_inputs}
    )
    displaced_chain.write_to_disk(displaced_fp, write_qcio=False)

    console.print(
        f"[cyan]Generated {len(displaced_nodes)} displaced candidates from {len(normal_modes)} normal modes"
        f"{' (clipped by --max-candidates).' if clipped else '.'}[/cyan]"
    )

    optimized_nodes: list[StructureNode] = []
    optimized_metadata: list[dict] = []
    failed_candidates: list[dict] = []

    engine_name = str(getattr(program_input, "engine_name", "")).lower()
    compute_program = str(getattr(engine, "compute_program", "")).lower()
    batch_optimizer = getattr(engine, "compute_geometry_optimizations", None)
    use_chemcloud_batch = engine_name == "chemcloud" or compute_program == "chemcloud"
    optimization_submission_mode = "serial"

    if use_chemcloud_batch:
        if engine_name == "chemcloud" and compute_program not in {"", "chemcloud"}:
            console.print(
                "[bold red]✗ ERROR:[/bold red] RunInputs requested ChemCloud (`engine_name=chemcloud`) "
                f"but engine reports `compute_program={compute_program}`. Refusing to run serial fallback."
            )
            raise typer.Exit(1)
        if not callable(batch_optimizer):
            console.print(
                "[bold red]✗ ERROR:[/bold red] ChemCloud engine selected, but batch geometry optimization is unavailable (`compute_geometry_optimizations`)."
            )
            raise typer.Exit(1)

        console.print(
            f"[cyan]Submitting {len(displaced_nodes)} geometry optimizations to ChemCloud in parallel.[/cyan]"
        )
        optimization_submission_mode = "chemcloud_batch"
        try:
            try:
                trajectories = batch_optimizer(
                    displaced_nodes, keywords={
                        "coordsys": "cart", "maxiter": maxiter}
                )
            except TypeError:
                trajectories = batch_optimizer(displaced_nodes)
        except Exception:
            console.print(
                f"[bold red]✗ ChemCloud batch geometry optimization failed:[/bold red] {traceback.format_exc()}"
            )
            raise typer.Exit(1)

        if len(trajectories) != len(displaced_nodes):
            console.print(
                "[bold red]✗ ERROR:[/bold red] ChemCloud batch returned a trajectory count different from the submitted candidate count."
            )
            raise typer.Exit(1)

        for meta, trajectory in zip(displaced_metadata, trajectories):
            if trajectory and len(trajectory) > 0:
                optimized_nodes.append(trajectory[-1])
                optimized_metadata.append(meta)
            else:
                failed_candidates.append(
                    {
                        **meta,
                        "error": "Batch optimization returned an empty trajectory.",
                    }
                )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Optimizing displaced candidates", total=len(displaced_nodes))
            for candidate, meta in zip(displaced_nodes, displaced_metadata):
                try:
                    try:
                        trajectory = engine.compute_geometry_optimization(
                            candidate, keywords={
                                "coordsys": "cart", "maxiter": maxiter}
                        )
                    except TypeError:
                        trajectory = engine.compute_geometry_optimization(
                            candidate)
                    optimized_nodes.append(trajectory[-1])
                    optimized_metadata.append(meta)
                except Exception:
                    failed_candidates.append(
                        {
                            **meta,
                            "error": traceback.format_exc().strip(),
                        }
                    )
                progress.update(task, advance=1)

    if len(optimized_nodes) == 0:
        console.print(
            "[bold red]✗ ERROR:[/bold red] All displaced-candidate optimizations failed.")
        raise typer.Exit(1)

    optimized_chain = Chain.model_validate(
        {"nodes": [cand.copy() for cand in optimized_nodes],
         "parameters": program_input.chain_inputs}
    )
    optimized_chain.write_to_disk(optimized_fp, write_qcio=write_qcio)

    unique_nodes = _dedupe_minima_nodes(
        optimized_nodes, program_input.chain_inputs)
    unique_chain = Chain.model_validate(
        {"nodes": [cand.copy() for cand in unique_nodes],
         "parameters": program_input.chain_inputs}
    )
    unique_chain.write_to_disk(unique_fp, write_qcio=write_qcio)

    summary_payload = {
        "geometry": str(Path(geometry).resolve()),
        "inputs": str(Path(inputs).resolve()) if inputs is not None else None,
        "dr": float(dr),
        "max_candidates": int(max_candidates),
        "maxiter": int(maxiter),
        "normal_modes_total": int(len(normal_modes)),
        "frequencies_wavenumber": [float(freq) for freq in frequencies],
        "displaced_candidates": int(len(displaced_nodes)),
        "optimized_candidates": int(len(optimized_nodes)),
        "failed_candidates": int(len(failed_candidates)),
        "unique_minima": int(len(unique_nodes)),
        "optimization_submission_mode": optimization_submission_mode,
        "engine_name": engine_name,
        "engine_compute_program": compute_program,
        "chain_inputs_thresholds": {
            "node_rms_thre": float(program_input.chain_inputs.node_rms_thre),
            "node_ene_thre": float(program_input.chain_inputs.node_ene_thre),
        },
        "displaced_metadata": displaced_metadata,
        "optimized_metadata": optimized_metadata,
        "failed_candidate_details": failed_candidates,
        "output_files": {
            "hessian_qcio": str(hessian_fp),
            "displaced_xyz": str(displaced_fp),
            "optimized_xyz": str(optimized_fp),
            "unique_xyz": str(unique_fp),
        },
    }
    _write_json_atomic(summary_fp, summary_payload)

    table = Table(box=box.ROUNDED, border_style="green", show_header=False)
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("Normal modes", str(len(normal_modes)))
    table.add_row("Displaced candidates", str(len(displaced_nodes)))
    table.add_row("Optimized candidates", str(len(optimized_nodes)))
    table.add_row("Failed candidates", str(len(failed_candidates)))
    table.add_row("Unique minima", str(len(unique_nodes)))
    table.add_row("Optimization mode", optimization_submission_mode)
    table.add_row("Hessian", str(hessian_fp))
    table.add_row("Displaced", str(displaced_fp))
    table.add_row("Optimized", str(optimized_fp))
    table.add_row("Unique", str(unique_fp))
    table.add_row("Summary", str(summary_fp))
    console.print(
        Panel(
            table,
            title="[bold green]✓ Hessian Sample Complete[/bold green]",
            border_style="green",
        )
    )


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
            hessres = _compute_hessian_result_for_sampling(
                program_input.engine, node)

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


@app.command("status")
def status(
    path: Annotated[str, typer.Argument(help="Path to a run artifact, .xyz output, status JSON, or *_network_splits directory")],
):
    console.print(BANNER)
    snapshot = _load_status_snapshot(path)
    run_status = snapshot.get("run_status") or {}
    manifest = snapshot.get("manifest") or {}
    root_info = run_status or manifest

    summary = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Artifact", snapshot["artifact_path"])
    summary.add_row("Base name", str(root_info.get("base_name", "unknown")))
    summary.add_row("Run state", str(root_info.get("run_state", "unknown")))
    if run_status:
        summary.add_row("Phase", str(run_status.get("phase", "unknown")))
        if "recursive" in run_status:
            summary.add_row("Recursive", str(run_status.get("recursive")))
        if "parallel" in run_status:
            summary.add_row("Parallel", str(run_status.get("parallel")))
        if "network_splits" in run_status:
            summary.add_row("Network splits", str(
                run_status.get("network_splits")))
        if "path_min_method" in run_status:
            summary.add_row("Path method", str(
                run_status.get("path_min_method")))
    console.print(
        Panel(summary, title="[bold cyan]MSMEP Status[/bold cyan]", border_style="cyan"))

    if manifest:
        counts = manifest.get("counts", {})
        counts_line = ", ".join(
            f"{key}={counts[key]}" for key in sorted(counts)) if counts else "none"
        console.print(
            f"[cyan]Requests:[/cyan] total={manifest.get('total_requests', 0)} [{counts_line}]")
        current_request_id = manifest.get("current_request_id")
        if current_request_id is not None:
            console.print(
                f"[yellow]Currently running request:[/yellow] {current_request_id}")

        request_table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
        request_table.add_column("ID", style="bold cyan", justify="right")
        request_table.add_column("Parent", style="dim", justify="right")
        request_table.add_column("Pair", style="magenta")
        request_table.add_column("Status", style="white")
        request_table.add_column("Path Nodes", style="white", justify="right")
        for record in manifest.get("requests", []):
            request_table.add_row(
                str(record.get("request_id", "")),
                "" if record.get("parent_request_id") is None else str(
                    record.get("parent_request_id")),
                f"{record.get('start_index', '?')} -> {record.get('end_index', '?')}",
                str(record.get("status", "")),
                "" if record.get("n_path_nodes") is None else str(
                    record.get("n_path_nodes")),
            )
        console.print(request_table)

        network_summary = manifest.get(
            "network_summary") or run_status.get("network_summary")
        if network_summary:
            network_table = Table(
                box=box.SIMPLE, show_header=True, pad_edge=False)
            network_table.add_column("Nodes", style="bold cyan")
            network_table.add_column("Edges", style="bold cyan")
            network_table.add_row(
                str(network_summary.get("node_count", 0)),
                str(network_summary.get("edge_count", 0)),
            )
            console.print(Panel(
                network_table, title="[bold cyan]Current Network[/bold cyan]", border_style="cyan"))
            edges = network_summary.get("edges") or []
            if edges:
                edge_text = ", ".join(f"{a}->{b}" for a, b in edges[:20])
                if len(edges) > 20:
                    edge_text += ", ..."
                console.print(f"[dim]{edge_text}[/dim]")


@app.command("visualize")
def visualize(
    result_path: Annotated[str, typer.Argument(help="Path to a NEB result .xyz or TreeNode result folder")],
    output_html: Annotated[str, typer.Option(
        "--output", "-o", help="Output HTML file path")] = None,
    qminds_fp: Annotated[str, typer.Option(
        "--qminds-fp", help="Path to qmindices.dat for atom-subset visualization")] = None,
    atom_indices: Annotated[str, typer.Option(
        "--atom-indices", help="Comma/space-separated atom indices (e.g. '1,2,3' or '1 2 3')")] = None,
    charge: Annotated[int, typer.Option(
        help="Charge used when reading serialized geometries")] = 0,
    multiplicity: Annotated[int, typer.Option(
        help="Spin multiplicity used when reading serialized geometries")] = 1,
    no_open: Annotated[bool, typer.Option(
        "--no-open", help="Do not auto-open browser window")] = False,
):
    console.print(BANNER)
    src = Path(result_path).resolve()
    with console.status("[bold cyan]Loading result object...[/bold cyan]"):
        viz_data = _load_visualization_data(
            result_path=src,
            charge=charge,
            multiplicity=multiplicity,
        )
        selected = _parse_visualize_atom_indices(
            qminds_fp=qminds_fp, atom_indices=atom_indices
        )
        if selected is not None:
            viz_data.chain = _subset_chain_for_visualization(
                viz_data.chain, selected)
            if viz_data.chain_trajectory:
                viz_data.chain_trajectory = _subset_chain_trajectory_for_visualization(
                    viz_data.chain_trajectory, selected
                )
            if viz_data.tree_layers:
                viz_data.tree_layers = _subset_tree_layers_for_visualization(
                    viz_data.tree_layers, selected
                )
            console.print(
                f"[dim]Visualizing atom subset with {len(selected)} atoms.[/dim]"
            )

    with console.status("[bold cyan]Building interactive HTML...[/bold cyan]"):
        network_payload = None
        if viz_data.network_pot is not None:
            network_payload = _build_network_visualization_payload(
                viz_data.network_pot,
                atom_indices=selected,
                endpoint_hints=viz_data.network_endpoint_hints,
            )
        html = _build_chain_visualizer_html(
            chain=viz_data.chain,
            chain_trajectory=viz_data.chain_trajectory,
            tree_layers=viz_data.tree_layers,
            network_payload=network_payload,
        )

    if output_html is None:
        suffix = src.stem if src.is_file() else src.name
        out_fp = Path.cwd() / f"{suffix}_visualize.html"
    else:
        out_fp = Path(output_html).resolve()
    out_fp.write_text(html, encoding="utf-8")
    console.print(
        f"[bold green]✓ Visualization written:[/bold green] {out_fp}")

    if not no_open:
        webbrowser.open(out_fp.resolve().as_uri())
        console.print("[dim]Opened in default browser.[/dim]")


@app.command("extract-best-path")
def extract_best_path(
    network_json: Annotated[str, typer.Argument(help="Path to a network .json file")],
    output_xyz: Annotated[str, typer.Option(
        "--output", "-o", help="Output XYZ file path for the joined best path")] = None,
    start_node: Annotated[int, typer.Option(
        "--start-node", help="Explicit network node index to use as the path start")] = None,
    end_node: Annotated[int, typer.Option(
        "--end-node", help="Explicit network node index to use as the path end")] = None,
    charge: Annotated[int, typer.Option(
        help="Charge used when reading serialized geometries")] = 0,
    multiplicity: Annotated[int, typer.Option(
        help="Spin multiplicity used when reading serialized geometries")] = 1,
):
    console.print(BANNER)
    src = Path(network_json).resolve()
    with console.status("[bold cyan]Loading network...[/bold cyan]"):
        viz_data = _load_visualization_data(
            result_path=src,
            charge=charge,
            multiplicity=multiplicity,
        )
    if viz_data.network_pot is None:
        raise typer.BadParameter(
            "extract-best-path requires a network .json input."
        )
    endpoint_hints = dict(viz_data.network_endpoint_hints or {})
    if start_node is not None:
        endpoint_hints["root_index"] = int(start_node)
    if end_node is not None:
        endpoint_hints["target_index"] = int(end_node)
    if not endpoint_hints:
        endpoint_hints = None

    with console.status("[bold cyan]Finding best path...[/bold cyan]"):
        payload = _build_network_visualization_payload(
            viz_data.network_pot,
            endpoint_hints=endpoint_hints,
        )
        path_nodes = payload.get("highlighted_path") or []
        if not path_nodes:
            raise typer.BadParameter(
                "No best path could be inferred from this network.")
        chain = _path_chain_from_pot(viz_data.network_pot, path_nodes)
        if chain is None:
            raise typer.BadParameter(
                "Could not construct a chain for the inferred best path.")

    if output_xyz is None:
        base_name = src.stem.replace("_network", "")
        out_fp = src.with_name(f"{base_name}_best_path.xyz")
    else:
        out_fp = Path(output_xyz).resolve()

    _write_chain_with_nan_fallback(chain, out_fp)
    _write_json_atomic(
        out_fp.with_suffix(".json"),
        {
            "network_path": str(src),
            "root_index": payload.get("root_index"),
            "target_index": payload.get("target_index"),
            "path": path_nodes,
        },
    )
    console.print(f"[bold green]✓ Best path written:[/bold green] {out_fp}")


@app.command("make-default-inputs")
def make_default_inputs(
        name: Annotated[str, typer.Option(
            "--name", help='path to output toml file')] = None,
        path_min_method: Annotated[str, typer.Option("--path-min-method", "-pmm",
                                                     help='name of path minimization.\
                                                          Options are: [neb, fneb, mlpgi, neb-dlf]')] = "neb"):
    console.print(BANNER)
    if name is None:
        name = Path(Path(os.getcwd()) / 'default_inputs')
    ri = RunInputs(path_min_method=path_min_method)
    out = Path(name)
    ri.save(out.parent / (out.stem+".toml"))
    console.print(
        f"[bold green]✓ Default inputs saved to:[/bold green] {out.parent / (out.stem+'.toml')}")


@app.command("netgen-smiles")
def netgen_smiles(
    smiles: Annotated[str, typer.Option(
        "--smiles", "-s", help="Root reactant SMILES")] = None,
    inputs: Annotated[str, typer.Option(
        "--inputs", "-i", help="Path minimization RunInputs TOML")] = None,
    reactions_fp: Annotated[str, typer.Option(
        "--reactions-fp", help="Path to retropaths reactions.p file")] = None,
    environment: Annotated[str, typer.Option(
        "--environment", "-e", help="Environment SMILES")] = "",
    name: Annotated[str, typer.Option(
        "--name", help="Run name / default workspace folder name")] = None,
    directory: Annotated[str, typer.Option(
        "--directory", "-d", help="Workspace directory")] = None,
    timeout_seconds: Annotated[int, typer.Option(
        "--timeout-seconds", help="Retropaths growth timeout in seconds")] = 30,
    max_nodes: Annotated[int, typer.Option(
        "--max-nodes", help="Retropaths maximum number of nodes")] = 40,
    max_depth: Annotated[int, typer.Option(
        "--max-depth", help="Retropaths maximum search depth")] = 4,
    max_parallel_nebs: Annotated[int, typer.Option(
        "--max-parallel-nebs", help="Number of recursive NEBs to run concurrently")] = 1,
    no_open: Annotated[bool, typer.Option(
        "--no-open", help="Do not auto-open the generated status HTML")] = False,
):
    console.print(BANNER)
    if smiles is None:
        raise typer.BadParameter("--smiles is required.")
    if inputs is None:
        raise typer.BadParameter("--inputs/-i is required.")
    try:
        ensure_retropaths_available(feature="`netgen-smiles`")
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc

    requested_dir = Path(directory).resolve() if directory else None
    if requested_dir is not None and (requested_dir / "workspace.json").exists():
        workspace = RetropathsWorkspace.read(requested_dir)
        workspace.max_parallel_nebs = max_parallel_nebs
        if reactions_fp is not None:
            workspace.reactions_fp = str(Path(reactions_fp).resolve())
        workspace.write()
    else:
        workspace = create_workspace(
            root_smiles=smiles,
            environment_smiles=environment,
            inputs_fp=inputs,
            reactions_fp=reactions_fp,
            name=name,
            directory=directory,
            timeout_seconds=timeout_seconds,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_parallel_nebs=max_parallel_nebs,
        )

    console.print(
        f"[bold cyan]Workspace:[/bold cyan] [white]{workspace.workdir}[/white]")
    console.print(
        f"[bold cyan]Root SMILES:[/bold cyan] [white]{workspace.root_smiles}[/white]")
    console.print(
        f"[bold cyan]Environment:[/bold cyan] [white]{workspace.environment_smiles or '(none)'}[/white]")
    console.print(
        f"[bold cyan]Reactions File:[/bold cyan] [white]{workspace.reactions_path}[/white]")

    with console.status("[bold cyan]Preparing retropaths cache, converted pot, and queue...[/bold cyan]"):
        prepare_neb_workspace(workspace)
        queue, _, status_fp = write_status_html(workspace)
    console.print(f"[dim]Initial status HTML: {status_fp}[/dim]")

    try:
        queue, _pot = run_netgen_smiles_workflow(
            workspace,
            progress=lambda msg: console.print(f"[cyan]{msg}[/cyan]"),
        )
    finally:
        queue, _, status_fp = write_status_html(workspace)

    counts = summarize_queue(queue)
    summary = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Workspace", workspace.workdir)
    summary.add_row("Queue Items", str(counts["items"]))
    summary.add_row("Completed", str(counts.get("completed", 0)))
    summary.add_row("Running", str(counts.get("running", 0)))
    summary.add_row("Pending", str(counts.get("pending", 0)))
    summary.add_row("Failed", str(counts.get("failed", 0)))
    summary.add_row("Incompatible", str(counts.get("incompatible", 0)))
    summary.add_row("Optimized Endpoints", str(sum(
        bool(_pot.graph.nodes[n].get("endpoint_optimized")) for n in _pot.graph.nodes)))
    summary.add_row("Status HTML", str(status_fp))
    console.print(Panel(
        summary, title="[bold green]✓ netgen-smiles Finished[/bold green]", border_style="green"))

    if not no_open:
        webbrowser.open(status_fp.resolve().as_uri())


@app.command("status")
def status_cmd(
    directory: Annotated[str, typer.Option(
        "--directory", "-d", help="Workspace directory containing workspace.json")] = ".",
    output_html: Annotated[str, typer.Option(
        "--output", "-o", help="Optional override path for generated status HTML")] = None,
    temperature: Annotated[float, typer.Option(
        "--temperature", help="KMC temperature in kelvin for the status page")] = 298.15,
    initial_conditions: Annotated[List[str], typer.Option(
        "--initial-condition", help="Override KMC initial conditions as NODE=VALUE. Repeatable.")] = None,
    no_open: Annotated[bool, typer.Option(
        "--no-open", help="Do not auto-open browser window")] = False,
):
    console.print(BANNER)
    workspace_dir = Path(directory).resolve()
    workspace_fp = workspace_dir / "workspace.json"
    if not workspace_fp.exists():
        console.print(
            f"[bold red]✗ ERROR:[/bold red] No workspace.json found in {workspace_dir}")
        raise typer.Exit(1)

    workspace = RetropathsWorkspace.read(workspace_dir)
    kmc_initial_conditions = _parse_kmc_initial_condition_overrides(
        initial_conditions)
    queue, pot, status_fp = write_status_html(
        workspace,
        kmc_temperature_kelvin=temperature,
        kmc_initial_conditions=kmc_initial_conditions,
    )
    counts = summarize_queue(queue)

    if output_html is not None:
        out_fp = Path(output_html).resolve()
        out_fp.write_text(status_fp.read_text(
            encoding="utf-8"), encoding="utf-8")
        status_fp = out_fp

    summary = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Workspace", workspace.workdir)
    summary.add_row("Root", workspace.root_smiles)
    summary.add_row("Environment", workspace.environment_smiles or "(none)")
    summary.add_row("Queue Items", str(counts["items"]))
    summary.add_row("Completed", str(counts.get("completed", 0)))
    summary.add_row("Running", str(counts.get("running", 0)))
    summary.add_row("Pending", str(counts.get("pending", 0)))
    summary.add_row("Failed", str(counts.get("failed", 0)))
    summary.add_row("Incompatible", str(counts.get("incompatible", 0)))
    summary.add_row("Network Nodes", str(pot.graph.number_of_nodes()))
    summary.add_row("Network Edges", str(pot.graph.number_of_edges()))
    summary.add_row("Optimized Endpoints", str(
        sum(bool(pot.graph.nodes[n].get("endpoint_optimized")) for n in pot.graph.nodes)))
    summary.add_row("KMC Temperature (K)", f"{temperature:.2f}")
    summary.add_row("Status HTML", str(status_fp))
    console.print(Panel(
        summary, title="[bold cyan]Network Status[/bold cyan]", border_style="cyan"))

    if not no_open:
        webbrowser.open(status_fp.resolve().as_uri())


@app.command("drive-refine")
def drive_refine(
    workspace: Annotated[str, typer.Option("--workspace", "-w", help="Path to a drive workspace directory or workspace.json")] = None,
    inputs: Annotated[str, typer.Option("--inputs", "-i", help="RunInputs TOML for TS/IRC refinement")] = None,
    output_directory: Annotated[str, typer.Option("--output-directory", "-o", help="Directory for the refined workspace")] = None,
    name: Annotated[str, typer.Option("--name", help="Run name for the refined workspace")] = None,
    use_bigchem: Annotated[bool, typer.Option("--use-bigchem/--no-use-bigchem", help="Use BigChem for Hessian-backed TS/IRC steps when supported")] = False,
    write_status_html_output: Annotated[bool, typer.Option("--write-status-html/--no-write-status-html", help="Generate full status.html for refined workspace (slower)")] = False,
):
    console.print(BANNER)
    if workspace is None:
        raise typer.BadParameter("--workspace/-w is required.")
    if inputs is None:
        raise typer.BadParameter("--inputs/-i is required.")

    workspace_path = Path(workspace).expanduser().resolve()
    workspace_dir = workspace_path.parent if workspace_path.name == "workspace.json" else workspace_path
    workspace_json = workspace_dir / "workspace.json"
    if not workspace_json.exists():
        raise typer.BadParameter(f"No workspace.json found at: {workspace_dir}")

    source_workspace = RetropathsWorkspace.read(workspace_dir)
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Preparing refinement...", total=1)

        def _on_progress(event: dict[str, object]) -> None:
            message = str(event.get("message") or "Running refinement...")
            total = int(event.get("total") or 1)
            current = int(event.get("current") or 0)
            total = max(1, total)
            current = max(0, min(total, current))
            progress.update(
                task,
                description=message,
                total=total,
                completed=current,
            )

        with _suppress_rdkit_valence_warnings():
            summary = refine_drive_workspace_network(
                source_workspace,
                refinement_inputs_fp=Path(inputs).expanduser().resolve(),
                refined_workspace_dir=(Path(output_directory).expanduser().resolve() if output_directory else None),
                refined_run_name=name,
                use_bigchem=use_bigchem,
                write_status_html_output=write_status_html_output,
                progress=_on_progress,
            )
        progress.update(task, description="Refinement complete.", total=1, completed=1)

    table = Table(box=box.ROUNDED, border_style="green", show_header=False)
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("Source Workspace", str(summary["source_workspace"]))
    table.add_row("Refined Workspace", str(summary["workspace"]))
    table.add_row("Inputs", str(summary["inputs_fp"]))
    table.add_row("Edges Scanned", str(summary["edges_scanned"]))
    table.add_row("Edges With Chains", str(summary["edges_with_chains"]))
    table.add_row("TS Attempted", str(summary["ts_guesses_attempted"]))
    table.add_row("TS Submitted", str(summary.get("ts_jobs_submitted", summary["ts_guesses_attempted"])))
    table.add_row("TS Converged", str(summary["ts_converged"]))
    table.add_row("TS Failed", str(summary.get("ts_failed", 0)))
    if summary.get("ts_top_errors"):
        table.add_row("TS Top Error", str(summary.get("ts_top_errors", [""])[0]))
    if summary.get("ts_failure_log"):
        table.add_row("TS Failure Log", str(summary["ts_failure_log"]))
    table.add_row("IRC Submitted", str(summary.get("irc_jobs_submitted", summary["ts_converged"])))
    table.add_row("IRC Converged", str(summary["irc_converged"]))
    table.add_row("IRC Failed", str(summary.get("irc_failed", 0)))
    if summary.get("irc_top_errors"):
        table.add_row("IRC Top Error", str(summary.get("irc_top_errors", [""])[0]))
    if summary.get("irc_failure_log"):
        table.add_row("IRC Failure Log", str(summary["irc_failure_log"]))
    table.add_row("ChemCloud Parallel", str(bool(summary.get("chemcloud_parallel", False))))
    table.add_row("Use BigChem", str(bool(summary.get("use_bigchem", False))))
    table.add_row("Added Nodes", str(summary["added_nodes"]))
    table.add_row("Added Edges", str(summary["added_edges"]))
    table.add_row("Final Nodes", str(summary["final_nodes"]))
    table.add_row("Final Edges", str(summary["final_edges"]))
    table.add_row("Queue Items", str(summary["queue_items"]))
    table.add_row("Artifacts", str(summary["artifacts_dir"]))
    console.print(Panel(table, title="[bold green]✓ drive-refine Complete[/bold green]", border_style="green"))


@app.command("drive")
def drive(
    inputs: Annotated[str, typer.Option("--inputs", "-i", help="Path minimization RunInputs TOML")] = None,
    smiles: Annotated[str, typer.Option("--smiles", "-s", help="Root reactant SMILES to bootstrap a drive workspace before opening the UI")] = None,
    product_smiles: Annotated[str, typer.Option("--product-smiles", "--end", help="Optional product / end SMILES to bootstrap a target node and queued NEB edge")] = None,
    start_xyz_fp: Annotated[str, typer.Option("--start-xyz-fp", "--start-xyz", "--reactant-xyz-fp", help="Path to reactant/start XYZ file used to seed endpoint geometry")] = None,
    end_xyz_fp: Annotated[str, typer.Option("--end-xyz-fp", "--end-xyz", "--product-xyz-fp", help="Path to product/end XYZ file used to seed endpoint geometry")] = None,
    environment: Annotated[str, typer.Option("--environment", "-e", help="Environment SMILES for SMILES-based drive initialization")] = "",
    charge: Annotated[int, typer.Option("--charge", help="Total charge for SMILES-bootstrapped drive endpoint structures")] = 0,
    multiplicity: Annotated[int, typer.Option("--multiplicity", help="Spin multiplicity for SMILES-bootstrapped drive endpoint structures")] = 1,
    name: Annotated[str, typer.Option("--name", help="Run name / workspace name for SMILES-based drive initialization")] = None,
    workspace: Annotated[str, typer.Option("--workspace", help="Existing workspace directory or workspace.json to load on startup")] = None,
    reactions_fp: Annotated[str, typer.Option("--reactions-fp", help="Path to retropaths reactions.p file")] = None,
    directory: Annotated[str, typer.Option("--directory", "-d", help="Directory where MEPD Drive workspaces should be created")] = None,
    host: Annotated[str, typer.Option("--host", help="Host interface for the local drive server")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port for the local drive server (0 selects a free port)")] = 0,
    ssh_login: Annotated[str, typer.Option("--ssh-login", help="SSH target to print a ready-made tunnel command, e.g. user@cluster")] = None,
    local_port: Annotated[int, typer.Option("--local-port", help="Laptop-side port for the printed SSH tunnel command")] = None,
    timeout_seconds: Annotated[int, typer.Option("--timeout-seconds", help="Retropaths growth timeout in seconds")] = 30,
    max_nodes: Annotated[int, typer.Option("--max-nodes", help="Retropaths maximum number of nodes")] = 40,
    max_depth: Annotated[int, typer.Option("--max-depth", help="Retropaths maximum search depth")] = 4,
    max_parallel_nebs: Annotated[int, typer.Option("--max-parallel-nebs", help="Number of autosplitting NEBs to run concurrently")] = 1,
    parallel_autosplit_nebs: Annotated[bool, typer.Option("--parallel-autosplit-nebs/--no-parallel-autosplit-nebs", help="Run each autosplitting NEB tree with parallel recursive branch fan-out.")] = False,
    parallel_autosplit_workers: Annotated[int, typer.Option("--parallel-autosplit-workers", help="Max workers per autosplitting NEB tree when --parallel-autosplit-nebs is enabled.")] = 4,
    network_splits: Annotated[bool, typer.Option("--network-splits/--no-network-splits", help="Use recursive autosplit results to build the displayed network overlay")] = True,
    hawaii: Annotated[bool, typer.Option("--hawaii/--no-hawaii", help="Run autonomous connect/NEB/Hessian exploration loop on startup")] = False,
    hawaii_discovery_tools: Annotated[str, typer.Option("--hawaii-discovery-tools", help="Comma-separated Hawaii discovery tools: hessian-sample, retropaths, nanoreactor. Empty string disables discovery.")] = None,
    no_open: Annotated[bool, typer.Option("--no-open", help="Do not auto-open the browser")] = False,
):
    console.print(BANNER)
    workspace_path = str(workspace or "").strip() or None
    requested_dir = Path(directory).resolve() if directory else None
    has_xyz_bootstrap = bool(str(start_xyz_fp or "").strip())
    if workspace_path is None and requested_dir is not None and (requested_dir / "workspace.json").exists():
        workspace_path = str(requested_dir)

    if (smiles or has_xyz_bootstrap) and workspace_path:
        raise typer.BadParameter(
            "Choose either bootstrap inputs (--smiles/--start-xyz-fp/--inputs) or --workspace/--directory to load an existing run, not both."
        )
    if parallel_autosplit_workers < 1:
        raise typer.BadParameter("--parallel-autosplit-workers must be at least 1.")
    if (smiles or has_xyz_bootstrap) and inputs is None:
        raise typer.BadParameter("--inputs/-i is required when using --smiles or --start-xyz-fp.")
    if end_xyz_fp and not has_xyz_bootstrap and not smiles:
        raise typer.BadParameter("--end-xyz-fp requires --start-xyz-fp or --smiles.")

    launch_kwargs = {
        "directory": directory,
        "inputs_fp": inputs,
        "workspace_path": workspace_path,
        "smiles": smiles,
        "product_smiles": product_smiles,
        "start_xyz_fp": start_xyz_fp,
        "end_xyz_fp": end_xyz_fp,
        "environment_smiles": environment,
        "charge": charge,
        "multiplicity": multiplicity,
        "run_name": name,
        "reactions_fp": reactions_fp,
        "host": host,
        "port": port,
        "timeout_seconds": timeout_seconds,
        "max_nodes": max_nodes,
        "max_depth": max_depth,
        "max_parallel_nebs": max_parallel_nebs,
        "parallel_autosplit_nebs": parallel_autosplit_nebs,
        "parallel_autosplit_workers": parallel_autosplit_workers,
        "network_splits": network_splits,
        "hawaii": hawaii,
        "hawaii_discovery_tools": hawaii_discovery_tools,
        "open_browser": not no_open,
    }

    if workspace_path:
        startup_total = 6 if network_splits else 4
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=36),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Resolving workspace path...", total=startup_total)
            progress_state = {"total": float(startup_total), "completed": 0.0}

            def _on_startup_progress(event: dict):
                message = str(event.get("message") or "Loading workspace...")
                total_steps = float(event.get("total_steps") or startup_total)
                completed_steps = float(event.get("completed_steps") or 0.0)
                progress_state["total"] = max(progress_state["total"], total_steps)
                progress_state["completed"] = max(
                    0.0, min(progress_state["total"], completed_steps)
                )
                progress.update(
                    task,
                    description=message,
                    total=max(1.0, progress_state["total"]),
                    completed=progress_state["completed"],
                )

            launch_kwargs["startup_progress"] = _on_startup_progress
            server = launch_mepd_drive(**launch_kwargs)
            final_total = max(float(startup_total), float(progress_state["total"]))
            progress.update(
                task,
                description="Workspace loaded.",
                total=final_total,
                completed=final_total,
            )
    else:
        server = launch_mepd_drive(**launch_kwargs)
    actual_host, actual_port = server.server_address[:2]
    access_url = f"http://{actual_host}:{actual_port}/"
    ui_token = str(getattr(server, "ui_token", "") or "").strip()
    if ui_token:
        access_url = f"{access_url}?token={ui_token}"
    console.print(
        _format_drive_access_panel(
            actual_host=actual_host,
            actual_port=actual_port,
            ssh_login=ssh_login,
            local_port=local_port,
            access_url=access_url,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("[dim]Stopping MEPD Drive.[/dim]")
    finally:
        server.shutdown()
        server.server_close()


def _path_str_for_toml(path_text: str | None, source_dir: Path, output_dir: Path) -> str | None:
    if path_text is None:
        return None
    src = Path(path_text)
    if not src.is_absolute():
        src = (source_dir / src).resolve()
    try:
        return str(src.relative_to(output_dir))
    except ValueError:
        return str(src)


def _resolve_tcin_reference(path_text: str | None, source_dir: Path) -> str | None:
    if path_text is None:
        return None
    p = Path(path_text)
    if p.is_absolute():
        return str(p) if p.exists() else None

    direct = (source_dir / p)
    if direct.exists():
        return str(p)

    stem, suffix = p.stem, p.suffix
    candidates = sorted(source_dir.glob(f"{stem}*{suffix}"))
    if len(candidates) == 1:
        return candidates[0].name

    if "react" in stem.lower():
        react_candidates = sorted(source_dir.glob(f"*react*{suffix}"))
        if len(react_candidates) == 1:
            return react_candidates[0].name

    return None


@app.command("toml-from-tcin")
def toml_from_tcin(
    tcin: Annotated[str, typer.Argument(help="Path to TeraChem input file (.in/.tc.in)")],
    output: Annotated[str, typer.Option(
        "--output", "-o", help="Output TOML file path")] = "qmmm_inputs_from_tc.toml",
    compute_program: Annotated[str, typer.Option(
        "--compute-program", help="QMMM backend: chemcloud or qcop")] = "chemcloud",
    queue: Annotated[str, typer.Option(
        "--queue", help="Optional ChemCloud queue to store in TOML")] = None,
):
    console.print(BANNER)
    tcin_fp = Path(tcin).resolve()
    if not tcin_fp.exists():
        raise typer.BadParameter(f"TeraChem input not found: {tcin_fp}")
    out_fp = Path(output).resolve()
    parsed = parse_terachem_input_file(tcin_fp)
    resolved_qminds = _resolve_tcin_reference(
        parsed["qmindices"], tcin_fp.parent)
    resolved_prmtop = _resolve_tcin_reference(parsed["prmtop"], tcin_fp.parent)
    resolved_coords = _resolve_tcin_reference(
        parsed["coordinates"], tcin_fp.parent)

    missing = []
    if resolved_qminds is None:
        missing.append(f"qmindices ({parsed['qmindices']})")
    if resolved_prmtop is None:
        missing.append(f"prmtop ({parsed['prmtop']})")
    if resolved_coords is None:
        missing.append(f"coordinates ({parsed['coordinates']})")
    if missing:
        raise typer.BadParameter(
            "Could not resolve required file references from tc.in: " +
            ", ".join(missing)
        )

    qmmm_inputs = {
        "qminds_fp": _path_str_for_toml(resolved_qminds, tcin_fp.parent, out_fp.parent),
        "prmtop_fp": _path_str_for_toml(resolved_prmtop, tcin_fp.parent, out_fp.parent),
        "rst7_fp_react": _path_str_for_toml(resolved_coords, tcin_fp.parent, out_fp.parent),
        "compute_program": compute_program,
        "charge": parsed["charge"],
        "spinmult": parsed["spinmult"],
        # NEB requires ene+grad evaluations; tc.in references are often minimization inputs.
        "run_type": "gradient",
        "min_coordinates": parsed["min_coordinates"],
    }
    qmmm_inputs = {k: v for k, v in qmmm_inputs.items() if v is not None}

    out = {
        "engine_name": "qmmm",
        "program": "terachem",
        "path_min_method": "neb",
        "program_kwds": {
            "model": {
                "method": parsed["method"],
                "basis": parsed["basis"],
            },
            "keywords": parsed["keywords"],
        },
        "qmmm_inputs": qmmm_inputs,
        "chain_inputs": {
            "frozen_atom_indices": parsed["frozen_atom_indices"],
            "use_geodesic_interpolation": False,
        },
    }
    if queue:
        out["chemcloud_queue"] = queue

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(tomli_w.dumps(out))

    n_frozen = len(parsed["frozen_atom_indices"])
    console.print(
        f"[bold green]✓ QMMM inputs TOML written:[/bold green] {out_fp}")
    console.print(
        f"[dim]Parsed {n_frozen} frozen atoms from $constraints.[/dim]")


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

    _render_runinputs(program_input)

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

            history.output_chain.write_to_disk(filename, write_qcio=bool(
                getattr(program_input, "write_qcio", False)))
            history.write_to_disk(foldername, write_qcio=bool(
                getattr(program_input, "write_qcio", False)))
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
