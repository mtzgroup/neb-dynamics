from __future__ import annotations

import base64
import contextlib
import hmac
import json
import multiprocessing
import os
import re
import secrets
import shutil
import threading
import time
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from qcio import ProgramArgs, Structure

from neb_dynamics.chain import Chain
from neb_dynamics.constants import ANGSTROM_TO_BOHR
from neb_dynamics.inputs import ChainInputs, RunInputs
from neb_dynamics.kmc import build_kmc_payload, simulate_kmc
from neb_dynamics.molecule import Molecule
from neb_dynamics.pot import Pot
from neb_dynamics.qcio_structure_helpers import molecule_to_structure, structure_to_molecule
from neb_dynamics.rdkit_draw import moldrawsvg
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.retropaths_queue import (
    _identical_endpoint_skip_reason,
    _ensure_pair_endpoints_optimized,
    _make_pair_chain,
    _run_single_item_worker,
    NEBQueueItem,
    RetropathsNEBQueue,
    build_retropaths_neb_queue,
    ensure_queue_item_for_edge,
)
from neb_dynamics.retropaths_compat import structure_node_from_graph_like_molecule
from neb_dynamics.retropaths_workflow import (
    _build_network_explorer_payload,
    _find_matching_node_by_molecule,
    _history_leaf_chains,
    _json_safe,
    _load_template_payloads,
    _node_label_for_explorer,
    _persist_endpoint_optimization_result,
    _quiet_force_smiles,
    _strip_cached_result,
    _write_edge_visualizations,
    add_manual_edge,
    add_manual_node,
    apply_reactions_to_node,
    ensure_retropaths_available,
    initialize_workspace_with_progress,
    RetropathsWorkspace,
    create_workspace,
    load_partial_annotated_pot,
    load_retropaths_pot,
    materialize_drive_graph,
    prepare_neb_workspace,
    run_hessian_sample_for_edge,
    run_hessian_sample_for_node,
    run_nanoreactor_for_node,
    summarize_queue,
)

_MERGED_DRIVE_POT_CACHE: dict[tuple[str, bool], Pot] = {}
_RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HAWAII_DISCOVERY_TOOL_ALLOWED = ("hessian-sample", "retropaths", "nanoreactor")
_HAWAII_DISCOVERY_TOOL_DEFAULT = ("hessian-sample",)
_HAWAII_DISCOVERY_TOOL_ALIASES = {
    "hessian": "hessian-sample",
    "hessian-sample": "hessian-sample",
    "hessian_sample": "hessian-sample",
    "retropaths": "retropaths",
    "nanoreactor": "nanoreactor",
}


def _is_within_directory(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_run_name(raw_run_name: str) -> str:
    run_name = str(raw_run_name or "").strip()
    if not run_name:
        raise ValueError("Run name cannot be empty.")
    if not _RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "Run name may contain only letters, digits, '.', '-', '_' and must not include path separators."
        )
    return run_name


def _normalize_hawaii_discovery_tools(
    value: Any,
    *,
    default: tuple[str, ...] | None = _HAWAII_DISCOVERY_TOOL_DEFAULT,
) -> list[str]:
    if value is None:
        return list(default or [])

    source_items: list[Any] = []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed: Any = None
        if text.startswith("["):
            with contextlib.suppress(Exception):
                parsed = json.loads(text)
        if isinstance(parsed, (list, tuple, set)):
            source_items = list(parsed)
        else:
            source_items = [token for token in re.split(r"[\s,]+", text) if token]
    elif isinstance(value, (list, tuple, set)):
        source_items = list(value)
    else:
        source_items = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    invalid: list[str] = []
    for raw in source_items:
        text = str(raw or "").strip()
        if not text:
            continue
        canonical = _HAWAII_DISCOVERY_TOOL_ALIASES.get(text.lower(), text.lower())
        if canonical not in _HAWAII_DISCOVERY_TOOL_ALLOWED:
            invalid.append(text)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)

    if invalid:
        allowed = ", ".join(_HAWAII_DISCOVERY_TOOL_ALLOWED)
        invalid_list = ", ".join(invalid)
        raise ValueError(
            f"Unsupported Hawaii discovery tool(s): {invalid_list}. Allowed tools: {allowed}."
        )
    return normalized


def _validate_workspace_path_for_init(
    workspace_path: Path,
    *,
    allowed_base_dir: Path | None,
) -> None:
    resolved_workspace = workspace_path.resolve()
    if resolved_workspace == resolved_workspace.parent:
        raise ValueError("Refusing to initialize workspace at filesystem root.")
    if allowed_base_dir is not None:
        resolved_base = allowed_base_dir.resolve()
        if not _is_within_directory(resolved_workspace, resolved_base):
            raise ValueError(
                f"Workspace path `{resolved_workspace}` escapes base directory `{resolved_base}`."
            )
        if resolved_workspace == resolved_base:
            raise ValueError("Workspace path must be a subdirectory of the configured drive base directory.")


def _is_loopback_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1"}


def _drive_merge_version(workspace: RetropathsWorkspace) -> str:
    fingerprints: list[str] = []
    for fp in filter(
        None,
        (
            getattr(workspace, "neb_pot_fp", None),
            getattr(workspace, "queue_fp", None),
            getattr(workspace, "retropaths_pot_fp", None),
            getattr(workspace, "annotated_neb_pot_fp", None),
        ),
    ):
        if fp.exists():
            stat = fp.stat()
            fingerprints.append(f"{fp.name}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            fingerprints.append(f"{fp.name}:missing")
    return "|".join(fingerprints)


def _merge_drive_pot_compat(workspace: RetropathsWorkspace, *, network_splits: bool) -> Pot:
    cache_key = (_drive_merge_version(workspace), bool(network_splits))
    cached = _MERGED_DRIVE_POT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        pot = _merge_drive_pot(workspace, network_splits=network_splits)
    except TypeError:
        pot = _merge_drive_pot(workspace)
    _MERGED_DRIVE_POT_CACHE.clear()
    _MERGED_DRIVE_POT_CACHE[cache_key] = pot
    return pot


def _load_existing_workspace_job_compat(
    workspace_path: str,
    *,
    network_splits: bool,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    try:
        return _load_existing_workspace_job(
            workspace_path,
            network_splits=network_splits,
            progress=progress,
        )
    except TypeError:
        return _load_existing_workspace_job(workspace_path)


def _load_partial_annotated_pot_compat(
    workspace: RetropathsWorkspace,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Pot:
    if progress is None:
        return load_partial_annotated_pot(workspace)
    try:
        return load_partial_annotated_pot(workspace, progress=progress)
    except TypeError:
        return load_partial_annotated_pot(workspace)


def _call_drive_payload_builder(
    builder: Callable[..., dict[str, Any]],
    workspace: RetropathsWorkspace,
    *,
    product_smiles: str,
    active_job_label: str,
    active_action: dict[str, Any] | None,
    network_splits: bool,
) -> dict[str, Any]:
    try:
        return builder(
            workspace,
            product_smiles=product_smiles,
            active_job_label=active_job_label,
            active_action=active_action,
            network_splits=network_splits,
        )
    except TypeError:
        return builder(
            workspace,
            product_smiles=product_smiles,
            active_job_label=active_job_label,
            active_action=active_action,
        )


def _energy_value(node: Any) -> float | None:
    with contextlib.suppress(Exception):
        return float(node.energy)
    cached = getattr(node, "_cached_energy", None)
    with contextlib.suppress(Exception):
        if cached is not None:
            return float(cached)
    return None


def _trajectory_plot_payload(
    trajectory: list[Any] | None,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> dict[str, Any] | None:
    if not trajectory:
        return None
    y_vals: list[float] = []
    for node in trajectory:
        energy = _energy_value(node)
        if energy is None:
            return None
        y_vals.append(energy)
    return {
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "x": list(range(len(y_vals))),
        "y": y_vals,
    }


def _run_geometry_optimization_with_trajectory(node: Any, run_inputs: RunInputs) -> tuple[Any, str | None, list[Any] | None]:
    try:
        try:
            traj = run_inputs.engine.compute_geometry_optimization(
                node, keywords={"coordsys": "cart", "maxiter": 500}
            )
        except TypeError:
            traj = run_inputs.engine.compute_geometry_optimization(node)
        optimized = traj[-1]
        if getattr(node, "has_molecular_graph", False):
            optimized.graph = structure_to_molecule(optimized.structure)
            optimized.has_molecular_graph = True
        else:
            optimized.graph = node.graph
            optimized.has_molecular_graph = node.has_molecular_graph
        return optimized, None, traj
    except Exception as exc:
        fallback = node.copy()
        fallback._cached_energy = None
        fallback._cached_gradient = None
        fallback._cached_result = None
        return fallback, f"{type(exc).__name__}: {exc}", None


def _run_geometry_optimization_batch_with_trajectories(
    nodes: list[Any],
    run_inputs: RunInputs,
) -> list[tuple[Any, str | None, list[Any] | None]]:
    if not nodes:
        return []

    def _failed_node(node: Any) -> Any:
        fallback = node.copy() if hasattr(node, "copy") else node
        with contextlib.suppress(Exception):
            fallback._cached_energy = None
        with contextlib.suppress(Exception):
            fallback._cached_gradient = None
        with contextlib.suppress(Exception):
            fallback._cached_result = None
        return fallback

    batch_optimizer = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    compute_program = str(getattr(run_inputs.engine, "compute_program", "") or "").lower()
    engine_name = str(getattr(run_inputs, "engine_name", "") or "").lower()
    use_batch = callable(batch_optimizer) and (compute_program == "chemcloud" or engine_name == "chemcloud")
    if not use_batch:
        return [
            _run_geometry_optimization_with_trajectory(node=node, run_inputs=run_inputs)
            for node in nodes
        ]

    try:
        try:
            trajectories = batch_optimizer(
                nodes,
                keywords={"coordsys": "cart", "maxiter": 500},
            )
        except TypeError:
            trajectories = batch_optimizer(nodes)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return [(_failed_node(node), error, None) for node in nodes]

    try:
        trajectory_list = list(trajectories)
    except Exception:
        trajectory_list = [trajectories]

    if len(trajectory_list) < len(nodes):
        trajectory_list = trajectory_list + [[] for _ in range(len(nodes) - len(trajectory_list))]
    elif len(trajectory_list) > len(nodes):
        trajectory_list = trajectory_list[: len(nodes)]

    optimized_nodes: list[tuple[Any, str | None, list[Any] | None]] = []
    for original, traj in zip(nodes, trajectory_list):
        try:
            traj_list = list(traj) if traj is not None else []
        except Exception:
            traj_list = [traj]

        if not traj_list:
            optimized_nodes.append(
                (
                    _failed_node(original),
                    "ChemCloud batch optimization returned an empty trajectory.",
                    [],
                )
            )
            continue

        optimized = traj_list[-1]
        if getattr(original, "has_molecular_graph", False):
            optimized.graph = structure_to_molecule(optimized.structure)
            optimized.has_molecular_graph = True
        else:
            optimized.graph = original.graph
            optimized.has_molecular_graph = original.has_molecular_graph
        optimized_nodes.append((optimized, None, traj_list))
    return optimized_nodes


def _resolve_minimize_target_indices(
    workspace: RetropathsWorkspace,
    node_indices: list[int] | None,
) -> list[int]:
    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    selected = set(int(node_index) for node_index in (node_indices or []))
    resolved: list[int] = []
    for node_index in pot.graph.nodes:
        if selected and int(node_index) not in selected:
            continue
        attrs = pot.graph.nodes[node_index]
        minimizable, _note = _node_minimize_status(int(node_index), attrs)
        if not minimizable:
            continue
        resolved.append(int(node_index))
    return resolved


def _node_minimize_status(node_index: int, node_attrs: dict[str, Any]) -> tuple[bool, str]:
    td = node_attrs.get("td")
    if td is None:
        return False, f"Node {node_index} has no geometry attached, so it cannot be minimized."
    structure = getattr(td, "structure", None)
    if structure is None:
        return False, f"Node {node_index} has no 3D structure attached, so it cannot be minimized."
    if node_attrs.get("endpoint_optimization_error"):
        return True, str(node_attrs.get("endpoint_optimization_error"))
    if node_attrs.get("endpoint_optimized"):
        return False, f"Node {node_index} is already geometry-optimized."
    return True, ""


def _node_apply_reaction_status(node_index: int, node_attrs: dict[str, Any]) -> tuple[bool, str]:
    molecule = node_attrs.get("molecule")
    if molecule is None:
        td = node_attrs.get("td")
        molecule = getattr(td, "graph", None)
    if molecule is None:
        return False, f"Node {node_index} has no molecular graph, so reaction templates cannot be applied to it."
    return True, ""


def _node_nanoreactor_status(
    node_index: int,
    node_attrs: dict[str, Any],
    inputs_summary: dict[str, Any],
) -> tuple[bool, str]:
    td = node_attrs.get("td")
    structure = getattr(td, "structure", None)
    if structure is None:
        return False, f"Node {node_index} has no 3D structure attached, so nanoreactor sampling cannot be started."
    if str(inputs_summary.get("error") or "").strip():
        return False, "The inputs file could not be loaded, so nanoreactor availability is unknown."
    engine_name = str(inputs_summary.get("engine_name") or "").strip().lower()
    program = str(inputs_summary.get("program") or "").strip().lower()
    if engine_name not in {"chemcloud", "qcop"}:
        return False, "Nanoreactor sampling currently requires a QCOP or ChemCloud-backed inputs file."
    if "terachem" in program:
        return True, "Run a TeraChem MD nanoreactor trajectory and merge distinct minimized products."
    if "crest" in program:
        return True, "Run a CREST MSREACT nanoreactor search and merge distinct minimized products."
    return False, "Nanoreactor sampling currently requires a CREST or TeraChem-backed inputs file."


def _node_hessian_sample_status(
    node_index: int,
    node_attrs: dict[str, Any],
    inputs_summary: dict[str, Any],
) -> tuple[bool, str]:
    td = node_attrs.get("td")
    structure = getattr(td, "structure", None)
    if structure is None:
        return False, f"Node {node_index} has no 3D structure attached, so Hessian sampling cannot be started."
    if str(inputs_summary.get("error") or "").strip():
        return False, "The inputs file could not be loaded, so Hessian-sample availability is unknown."
    if not bool(inputs_summary.get("can_hessian_sample", False)):
        return False, str(inputs_summary.get("hessian_sample_note") or "This inputs configuration does not support Hessian sampling.")
    return True, "Compute Hessian normal modes, displace by ±dr, optimize each displacement, and merge unique minima."


def _edge_hessian_sample_status(edge: dict[str, Any]) -> tuple[bool, str]:
    has_completed_chain = bool(edge.get("neb_backed")) or bool(edge.get("result_from_completed_queue")) or str(
        edge.get("queue_status") or ""
    ).strip() == "completed"
    if has_completed_chain:
        return True, "Use the highest-energy geometry from the completed NEB chain as the Hessian-sample seed."
    return False, "Select an edge with a completed NEB chain before running Hessian sampling from the edge peak."


def _edge_neb_status(edge: dict[str, Any]) -> tuple[bool, str]:
    status = str(edge.get("queue_status") or "").strip()
    queue_error = str(edge.get("queue_error") or "").strip()
    if not status or status == "pending":
        return True, ""
    if status == "running":
        return False, "This edge already has an NEB running."
    if status == "completed":
        return False, "This edge already has a completed NEB result."
    if status == "skipped_identical":
        return False, queue_error or "This edge was skipped because its optimized endpoints are identical."
    if status == "skipped_attempted":
        return False, queue_error or "This edge was already attempted with the same endpoints."
    if status in {"incompatible", "missing_td", "failed"}:
        return False, queue_error or f"This edge cannot be queued from its current state ({status})."
    return False, queue_error or f"This edge cannot be queued from its current state ({status})."


def _build_minimize_live_payload(active_action: dict[str, Any] | None) -> dict[str, Any] | None:
    if active_action is None or active_action.get("type") != "minimize":
        return None
    jobs = [dict(job) for job in active_action.get("jobs", [])]
    running_jobs = [job for job in jobs if job.get("status") == "running"]
    completed_jobs = [job for job in jobs if job.get("status") == "completed"]
    failed_jobs = [job for job in jobs if job.get("status") == "failed"]
    plot = None
    note = (
        "This backend does not stream intermediate geometry-optimization steps while a job is in flight. "
        "The list below is live, and completed trajectories appear as soon as each node returns."
    )
    if running_jobs:
        plot = running_jobs[-1].get("plot")
    if plot is None and completed_jobs:
        plot = completed_jobs[-1].get("plot")
    if plot is None and failed_jobs:
        note = "No optimization trajectory was returned for the current geometry job."
    return {
        "type": "minimize",
        "title": active_action.get("label") or "Running geometry minimizations",
        "jobs": jobs,
        "plot": plot,
        "note": note,
    }


def _build_neb_live_payload(
    active_action: dict[str, Any] | None,
    workspace: RetropathsWorkspace | None = None,
) -> dict[str, Any] | None:
    if active_action is None or active_action.get("type") != "neb":
        return None
    live_chain = _read_chain_payload(active_action.get("chain_fp"))
    reactant_structure = None
    product_structure = None
    if live_chain:
        reactant_smiles = str((live_chain.get("plot") or {}).get("reactant_smiles") or "").strip()
        product_smiles = str((live_chain.get("plot") or {}).get("product_smiles") or "").strip()
        if reactant_smiles:
            reactant_structure = _molecule_visual_payload(reactant_smiles)
        if product_smiles:
            product_structure = _molecule_visual_payload(product_smiles)
    if workspace is not None:
        with contextlib.suppress(Exception):
            pot = Pot.read_from_disk(workspace.neb_pot_fp)
            source_node = int(active_action.get("source_node", -1))
            target_node = int(active_action.get("target_node", -1))
            if reactant_structure is None and source_node in pot.graph.nodes:
                reactant_payload = _node_structure_payload(pot.graph.nodes[source_node])
                reactant_structure = (reactant_payload or {}).get("molecule_viz")
            if product_structure is None and target_node in pot.graph.nodes:
                product_payload = _node_structure_payload(pot.graph.nodes[target_node])
                product_structure = (product_payload or {}).get("molecule_viz")
    history = list((live_chain or {}).get("history") or [])
    if len(history) > 24:
        history = history[-24:]
    monitors = dict((live_chain or {}).get("monitors") or {})
    return {
        "type": "neb",
        "title": active_action.get("label") or "Running autosplitting NEB",
        "plot": live_chain.get("plot") if live_chain else None,
        "history": history,
        "ascii_plot": live_chain.get("ascii_plot") if live_chain else "",
        "monitors": monitors,
        "console_text": _read_log_tail(active_action.get("progress_fp"), max_chars=6000),
        "reactant_structure": reactant_structure,
        "product_structure": product_structure,
        "note": "Live optimization-history view. Faded curves are earlier optimization steps; the highlighted curve is the latest chain.",
    }


def _build_growth_live_payload(active_action: dict[str, Any] | None) -> dict[str, Any] | None:
    if active_action is None or active_action.get("type") not in {"initialize", "apply-reactions", "nanoreactor", "hessian-sample", "hawaii"}:
        return None
    progress = _read_growth_progress(active_action.get("progress_fp")) or {}
    return {
        "type": "growth",
        "title": str(progress.get("title") or active_action.get("label") or "Growing Retropaths network"),
        "note": str(progress.get("note") or ""),
        "phase": str(progress.get("phase") or "growing"),
        "network": dict(progress.get("network") or {"nodes": [], "edges": []}),
    }


def _build_hawaii_live_payload(
    active_action: dict[str, Any] | None,
    workspace: RetropathsWorkspace | None = None,
) -> dict[str, Any] | None:
    if active_action is None or active_action.get("type") != "hawaii":
        return None
    progress = _read_growth_progress(active_action.get("progress_fp")) or {}
    stage = str(progress.get("stage") or "").strip().lower()
    if stage == "neb":
        neb = dict(progress.get("neb") or {})
        source_node = neb.get("source_node")
        target_node = neb.get("target_node")
        synthetic_action = {
            "type": "neb",
            "status": "running",
            "label": str(progress.get("note") or active_action.get("label") or "Hawaii autosplitting NEB"),
            "source_node": (int(source_node) if source_node is not None else -1),
            "target_node": (int(target_node) if target_node is not None else -1),
            "progress_fp": str(neb.get("progress_fp") or ""),
            "chain_fp": str(neb.get("chain_fp") or ""),
        }
        neb_payload = _build_neb_live_payload(synthetic_action, workspace)
        if neb_payload is not None:
            return neb_payload
    return _build_growth_live_payload(active_action)


def _read_log_tail(log_fp: Any, max_chars: int = 12000) -> str:
    if not log_fp:
        return ""
    try:
        text = Path(str(log_fp)).read_text(errors="replace")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _read_chain_payload(chain_fp: Any) -> dict[str, Any] | None:
    if not chain_fp:
        return None
    try:
        return json.loads(Path(str(chain_fp)).read_text())
    except Exception:
        return None


def _read_growth_progress(progress_fp: Any) -> dict[str, Any] | None:
    if not progress_fp:
        return None
    try:
        return json.loads(Path(str(progress_fp)).read_text())
    except Exception:
        return None


def _drive_network_version(workspace: RetropathsWorkspace) -> str:
    fingerprints: list[str] = []
    for fp in filter(
        None,
        (
            getattr(workspace, "neb_pot_fp", None),
            getattr(workspace, "queue_fp", None),
            getattr(workspace, "retropaths_pot_fp", None),
        ),
    ):
        if fp.exists():
            stat = fp.stat()
            fingerprints.append(f"{fp.name}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            fingerprints.append(f"{fp.name}:missing")
    return "|".join(fingerprints)


def _format_user_facing_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    raw = str(exc)
    if "Exec format error" in raw or "No such file or directory" in raw:
        program_name = ""
        if "'" in raw:
            with contextlib.suppress(Exception):
                program_name = raw.split("'")[-2]
        if program_name:
            return (
                f"Could not execute `{program_name}`. "
                "Check that the configured external program is installed on this machine, "
                "is executable, and matches the current platform."
            )
        return (
            "Could not execute the configured external program. "
            "Check that it is installed on this machine and is executable."
        )
    return message


def _parse_xyz_text_to_structure(
    xyz_text: str,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> Structure:
    lines = [line.rstrip() for line in xyz_text.strip().splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("XYZ text must contain an atom count, comment line, and atom rows.")

    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("First XYZ line must be an integer atom count.") from exc

    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"XYZ text declared {atom_count} atoms but only {len(atom_lines)} rows were provided.")

    symbols: list[str] = []
    geometry_angstrom: list[list[float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom row: {line}")
        symbols.append(parts[0])
        geometry_angstrom.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return Structure(
        symbols=symbols,
        geometry=[[value * ANGSTROM_TO_BOHR for value in row] for row in geometry_angstrom],
        charge=charge,
        multiplicity=multiplicity,
    )


def _read_xyz_file_text(path_value: str | None, *, label: str) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    fp = Path(raw).expanduser().resolve()
    if not fp.exists():
        raise FileNotFoundError(f"{label} XYZ file was not found: {fp}")
    if not fp.is_file():
        raise ValueError(f"{label} XYZ path is not a file: {fp}")
    try:
        return fp.read_text(encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Could not read {label} XYZ file `{fp}`: {type(exc).__name__}: {exc}") from exc


def _molecule_visual_payload(molecule_like: Any) -> dict[str, Any] | None:
    if molecule_like is None:
        return None
    smiles = ""
    try:
        if isinstance(molecule_like, str):
            smiles = Molecule.from_smiles(molecule_like).force_smiles()
        elif hasattr(molecule_like, "force_smiles"):
            smiles = _quiet_force_smiles(molecule_like)
        else:
            smiles = _quiet_force_smiles(structure_to_molecule(molecule_like))
    except Exception:
        smiles = ""
    if not smiles:
        return None
    try:
        svg = moldrawsvg(
            smiles,
            {},
            molSize=(340, 220),
            fixed_bond_length=32,
            lineWidth=2,
        )
        if "<svg" not in svg:
            raise ValueError("RDKit render did not return SVG")
        return {
            "smiles": smiles,
            "svg": svg,
            "render_error": "",
        }
    except Exception as exc:
        return {
            "smiles": smiles,
            "svg": "",
            "render_error": f"{type(exc).__name__}: {exc}",
        }


def _species_payload(
    *,
    smiles: str,
    structure: Structure | None,
) -> dict[str, Any]:
    xyz_b64 = ""
    if structure is not None:
        xyz_b64 = base64.b64encode(structure.to_xyz().encode("utf-8")).decode("ascii")
    return {
        "smiles": smiles,
        "xyz_b64": xyz_b64,
        "symbols": list(structure.symbols) if structure is not None else [],
        "charge": int(getattr(structure, "charge", 0) if structure is not None else 0),
        "multiplicity": int(getattr(structure, "multiplicity", 1) if structure is not None else 1),
    }


def _resolve_species_input(
    *,
    smiles: str,
    xyz_text: str,
    charge: int = 0,
    multiplicity: int = 1,
) -> dict[str, Any]:
    smiles = smiles.strip()
    xyz_text = xyz_text.strip()
    if not smiles and not xyz_text:
        return {}

    structure = None
    if xyz_text:
        structure = _parse_xyz_text_to_structure(
            xyz_text,
            charge=charge,
            multiplicity=multiplicity,
        )

    if not smiles:
        if structure is None:
            raise ValueError("A species needs either SMILES or XYZ text.")
        smiles = _quiet_force_smiles(structure_to_molecule(structure))
    else:
        smiles = Molecule.from_smiles(smiles).force_smiles()

    if structure is None:
        structure = molecule_to_structure(
            Molecule.from_smiles(smiles),
            charge=charge,
            spinmult=multiplicity,
        )

    return _species_payload(smiles=smiles, structure=structure)


def _species_structure_from_payload(species: dict[str, Any] | None) -> Structure | None:
    if not species:
        return None
    xyz_b64 = str(species.get("xyz_b64") or "").strip()
    if not xyz_b64:
        return None
    charge = int(species.get("charge", 0) or 0)
    multiplicity = int(species.get("multiplicity", 1) or 1)
    try:
        xyz_text = base64.b64decode(xyz_b64.encode("ascii")).decode("utf-8")
    except Exception:
        return None
    with contextlib.suppress(Exception):
        return _parse_xyz_text_to_structure(
            xyz_text,
            charge=charge,
            multiplicity=multiplicity,
        )
    return None


def _species_graph_from_payload(species: dict[str, Any] | None) -> Molecule | None:
    if not species:
        return None
    structure = _species_structure_from_payload(species)
    if structure is not None:
        with contextlib.suppress(Exception):
            return structure_to_molecule(structure)
    smiles = str(species.get("smiles") or "").strip()
    if smiles:
        with contextlib.suppress(Exception):
            return Molecule.from_smiles(smiles)
    return None


def _structure_node_from_species_payload(
    species: dict[str, Any] | None,
    *,
    fallback_molecule: Molecule | None = None,
) -> Any | None:
    graph = _species_graph_from_payload(species)
    if graph is None and fallback_molecule is None:
        return None
    graph = graph if graph is not None else fallback_molecule.copy()
    charge = int((species or {}).get("charge", 0) or 0)
    multiplicity = int((species or {}).get("multiplicity", 1) or 1)
    td = structure_node_from_graph_like_molecule(
        graph,
        charge=charge,
        spinmult=multiplicity,
    )
    structure = _species_structure_from_payload(species)
    if structure is not None:
        td.structure = structure
    return td


def _try_map_product_to_reactant_indices(
    reactant_graph: Molecule | None,
    product_graph: Molecule,
) -> Molecule:
    if reactant_graph is None:
        return product_graph
    try:
        reactant_no_h = reactant_graph.remove_Hs()
        product_no_h = product_graph.remove_Hs()
        if len(reactant_no_h.nodes) != len(product_no_h.nodes):
            return product_graph
        isomorphisms = reactant_no_h.get_subgraph_isomorphisms_of(product_no_h)
        if not isomorphisms:
            return product_graph
        mapping = dict(isomorphisms[0].reverse_mapping)
        if len(mapping) != len(product_no_h.nodes):
            return product_graph
        mapped_product = product_no_h.renumber_indexes(mapping)
        mapped_product = mapped_product.add_Hs()
        mapped_product.set_neighbors()
        return mapped_product
    except Exception:
        return product_graph


def _bootstrap_product_endpoint(
    workspace: RetropathsWorkspace,
    product: dict[str, Any] | None,
) -> None:
    if not product:
        return
    product_structure = _species_structure_from_payload(product)
    product_smiles = str(product.get("smiles") or "").strip()
    if (not product_smiles and product_structure is None) or not workspace.neb_pot_fp.exists():
        return
    product_charge = int(product.get("charge", 0) or 0)
    product_multiplicity = int(product.get("multiplicity", 1) or 1)

    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    if 0 not in pot.graph.nodes:
        return
    root_attrs = pot.graph.nodes[0]
    root_td = root_attrs.get("td")
    if root_td is None or getattr(root_td, "structure", None) is None:
        return

    product_graph = _species_graph_from_payload(product)
    if product_graph is None and product_smiles:
        product_graph = Molecule.from_smiles(product_smiles)
    if product_graph is None:
        return
    if product_structure is None:
        product_graph = _try_map_product_to_reactant_indices(
            root_attrs.get("molecule") or getattr(root_td, "graph", None),
            product_graph,
        )
    target_index = _find_matching_node_by_molecule(pot, product_graph)
    if target_index is None:
        target_index = (max(pot.graph.nodes) + 1) if pot.graph.nodes else 0
        product_td = _structure_node_from_species_payload(
            product,
            fallback_molecule=product_graph,
        )
        if product_td is None:
            product_td = structure_node_from_graph_like_molecule(
                product_graph,
                charge=product_charge,
                spinmult=product_multiplicity,
            )
        pot.graph.add_node(
            target_index,
            molecule=product_graph.copy(),
            converged=False,
            td=product_td,
            endpoint_optimized=False,
            generated_by="drive_product_smiles",
        )

    if int(target_index) == 0:
        pot.write_to_disk(workspace.neb_pot_fp)
        build_retropaths_neb_queue(pot=pot, queue_fp=workspace.queue_fp, overwrite=False)
        return

    if not pot.graph.has_edge(0, target_index):
        pot.graph.add_edge(
            0,
            target_index,
            reaction="Product target",
            list_of_nebs=[],
            generated_by="drive_product_smiles",
        )
    pot.write_to_disk(workspace.neb_pot_fp)
    ensure_queue_item_for_edge(
        pot=pot,
        source_node=0,
        target_node=int(target_index),
        queue_fp=workspace.queue_fp,
        overwrite=False,
    )
    build_retropaths_neb_queue(pot=pot, queue_fp=workspace.queue_fp, overwrite=False)


def _apply_bootstrap_species_overrides(
    workspace: RetropathsWorkspace,
    *,
    reactant: dict[str, Any],
    product: dict[str, Any] | None,
) -> None:
    if not workspace.neb_pot_fp.exists():
        return
    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    if 0 in pot.graph.nodes:
        reactant_graph = _species_graph_from_payload(reactant)
        if reactant_graph is None:
            reactant_graph = pot.graph.nodes[0].get("molecule") or Molecule.from_smiles(str(reactant["smiles"]))
        reactant_td = _structure_node_from_species_payload(
            reactant,
            fallback_molecule=reactant_graph,
        )
        if reactant_td is not None:
            pot.graph.nodes[0]["molecule"] = reactant_graph.copy()
            pot.graph.nodes[0]["td"] = reactant_td
        pot.graph.nodes[0]["endpoint_optimized"] = False
    pot.write_to_disk(workspace.neb_pot_fp)
    _bootstrap_product_endpoint(workspace, product)


def _bootstrap_minimal_drive_workspace(
    workspace: RetropathsWorkspace,
    *,
    reactant: dict[str, Any],
    product: dict[str, Any] | None,
) -> None:
    reactant_graph = _species_graph_from_payload(reactant)
    if reactant_graph is None:
        reactant_graph = Molecule.from_smiles(str(reactant["smiles"]))
    reactant_td = _structure_node_from_species_payload(
        reactant,
        fallback_molecule=reactant_graph,
    )
    if reactant_td is None:
        reactant_td = structure_node_from_graph_like_molecule(
            reactant_graph,
            charge=int(reactant.get("charge", 0) or 0),
            spinmult=int(reactant.get("multiplicity", 1) or 1),
        )

    pot = Pot(root=reactant_graph.copy(), target=Molecule())
    pot.graph.nodes[0].update(
        molecule=reactant_graph.copy(),
        converged=False,
        td=reactant_td,
        endpoint_optimized=False,
        generated_by="drive_bootstrap",
        root=True,
    )

    if product:
        product_graph = _species_graph_from_payload(product)
        if product_graph is None:
            product_smiles = str(product.get("smiles") or "").strip()
            if product_smiles:
                product_graph = Molecule.from_smiles(product_smiles)
        if product_graph is not None:
            if _species_structure_from_payload(product) is None:
                product_graph = _try_map_product_to_reactant_indices(
                    reactant_graph,
                    product_graph,
                )
            target_index = _find_matching_node_by_molecule(pot, product_graph)
            if target_index is None:
                target_index = (max(pot.graph.nodes) + 1) if pot.graph.nodes else 0
                product_td = _structure_node_from_species_payload(
                    product,
                    fallback_molecule=product_graph,
                )
                if product_td is None:
                    product_td = structure_node_from_graph_like_molecule(
                        product_graph,
                        charge=int(product.get("charge", 0) or 0),
                        spinmult=int(product.get("multiplicity", 1) or 1),
                    )
                pot.graph.add_node(
                    target_index,
                    molecule=product_graph.copy(),
                    converged=False,
                    td=product_td,
                    endpoint_optimized=False,
                    generated_by="drive_product_smiles",
                )
            if int(target_index) != 0 and not pot.graph.has_edge(0, int(target_index)):
                pot.graph.add_edge(
                    0,
                    int(target_index),
                    reaction="Product target",
                    list_of_nebs=[],
                    generated_by="drive_product_smiles",
                )

    workspace.directory.mkdir(parents=True, exist_ok=True)
    pot.write_to_disk(workspace.neb_pot_fp)
    build_retropaths_neb_queue(
        pot=pot,
        queue_fp=workspace.queue_fp,
        overwrite=True,
    )
    if not workspace.retropaths_pot_fp.exists():
        workspace.retropaths_pot_fp.write_text("{}", encoding="utf-8")


def _node_structure_payload(node_attrs: dict[str, Any]) -> dict[str, Any] | None:
    td = node_attrs.get("td")
    structure = getattr(td, "structure", None)
    if structure is None:
        molecule = node_attrs.get("molecule")
        if molecule is not None:
            with contextlib.suppress(Exception):
                structure = molecule_to_structure(molecule)
    if structure is None:
        return None
    return {
        "xyz_b64": base64.b64encode(structure.to_xyz().encode("utf-8")).decode("ascii"),
        "symbols": list(structure.symbols),
        "molecule_viz": _molecule_visual_payload(getattr(td, "graph", None) or node_attrs.get("molecule")),
    }


def _node_structure_payload_fast(node_attrs: dict[str, Any]) -> dict[str, Any] | None:
    td = node_attrs.get("td")
    structure = getattr(td, "structure", None)
    if structure is None:
        return None
    return {
        "xyz_b64": base64.b64encode(structure.to_xyz().encode("utf-8")).decode("ascii"),
        "symbols": list(structure.symbols),
        "molecule_viz": None,
    }


def _structure_payload_from_structure(
    structure: Any,
    *,
    molecule_like: Any = None,
) -> dict[str, Any] | None:
    if structure is None:
        return None
    return {
        "xyz_b64": base64.b64encode(structure.to_xyz().encode("utf-8")).decode("ascii"),
        "symbols": list(structure.symbols),
        "molecule_viz": _molecule_visual_payload(molecule_like if molecule_like is not None else structure),
    }


def _structure_payload_from_structure_fast(structure: Any) -> dict[str, Any] | None:
    if structure is None:
        return None
    return {
        "xyz_b64": base64.b64encode(structure.to_xyz().encode("utf-8")).decode("ascii"),
        "symbols": list(structure.symbols),
        "molecule_viz": None,
    }


def _load_completed_queue_chain(item: Any) -> Chain | Any | None:
    chain = None
    output_chain_xyz = str(getattr(item, "output_chain_xyz", "") or "").strip()
    if output_chain_xyz:
        with contextlib.suppress(Exception):
            chain = Chain.from_xyz(
                output_chain_xyz,
                ChainInputs(),
            )
    if chain is not None:
        return chain
    try:
        history = TreeNode.read_from_disk(
            folder_name=item.result_dir,
            charge=0,
            multiplicity=1,
        )
    except Exception:
        return None
    return getattr(history, "output_chain", None)


@dataclass
class _CompletedQueueEntry:
    source_node: int
    target_node: int
    result_dir: str
    output_chain_xyz: str = ""
    finished_at: str = ""


def _completed_queue_entries(queue: Any) -> list[_CompletedQueueEntry]:
    by_edge: dict[tuple[int, int], _CompletedQueueEntry] = {}

    def _entry_score(entry: _CompletedQueueEntry) -> tuple[str, int, int]:
        return (
            str(entry.finished_at or ""),
            1 if str(entry.result_dir or "").strip() else 0,
            1 if str(entry.output_chain_xyz or "").strip() else 0,
        )

    def _upsert(entry: _CompletedQueueEntry) -> None:
        edge_key = (int(entry.source_node), int(entry.target_node))
        existing = by_edge.get(edge_key)
        if existing is None or _entry_score(entry) >= _entry_score(existing):
            by_edge[edge_key] = entry

    for item in list(getattr(queue, "items", []) or []):
        if str(getattr(item, "status", "") or "").strip().lower() != "completed":
            continue
        result_dir = str(getattr(item, "result_dir", "") or "").strip()
        output_chain_xyz = str(getattr(item, "output_chain_xyz", "") or "").strip()
        if not result_dir and not output_chain_xyz:
            continue
        with contextlib.suppress(Exception):
            _upsert(
                _CompletedQueueEntry(
                    source_node=int(getattr(item, "source_node")),
                    target_node=int(getattr(item, "target_node")),
                    result_dir=result_dir,
                    output_chain_xyz=output_chain_xyz,
                    finished_at=str(getattr(item, "finished_at", "") or ""),
                )
            )

    attempted_pairs = dict(getattr(queue, "attempted_pairs", {}) or {})
    for attempt in attempted_pairs.values():
        status = str((attempt or {}).get("status") or "").strip().lower()
        if status != "completed":
            continue
        result_dir = str((attempt or {}).get("result_dir") or "").strip()
        output_chain_xyz = str((attempt or {}).get("output_chain_xyz") or "").strip()
        if not result_dir and not output_chain_xyz:
            continue
        with contextlib.suppress(Exception):
            _upsert(
                _CompletedQueueEntry(
                    source_node=int((attempt or {}).get("source_node")),
                    target_node=int((attempt or {}).get("target_node")),
                    result_dir=result_dir,
                    output_chain_xyz=output_chain_xyz,
                    finished_at=str((attempt or {}).get("finished_at") or ""),
                )
            )

    return [by_edge[key] for key in sorted(by_edge)]


def _completed_queue_result_payload(item: Any) -> dict[str, Any] | None:
    leaf_count = 0
    result_dir = str(getattr(item, "result_dir", "") or "").strip()
    if result_dir:
        try:
            history = TreeNode.read_from_disk(
                folder_name=result_dir,
                charge=0,
                multiplicity=1,
            )
        except Exception:
            history = None
        if history is not None:
            with contextlib.suppress(Exception):
                leaf_count = len(_history_leaf_chains(history))
    if leaf_count > 1:
        return {
            "is_elementary_result": False,
            "leaf_count": int(leaf_count),
        }
    chain = _load_completed_queue_chain(item)
    if chain is None or len(chain) < 2:
        return None
    source_node = chain[0]
    target_node = chain[-1]
    return {
        "source_node": int(item.source_node),
        "target_node": int(item.target_node),
        "source_structure": _structure_payload_from_structure(
            getattr(source_node, "structure", None),
            molecule_like=getattr(source_node, "graph", None),
        ),
        "target_structure": _structure_payload_from_structure(
            getattr(target_node, "structure", None),
            molecule_like=getattr(target_node, "graph", None),
        ),
        "barrier": float(chain.get_eA_chain()) if hasattr(chain, "get_eA_chain") else None,
        "chain": chain,
        "is_elementary_result": leaf_count <= 1,
        "leaf_count": int(leaf_count),
    }


def _merge_drive_pot(workspace: RetropathsWorkspace, *, network_splits: bool = True) -> Pot:
    base_pot = Pot.read_from_disk(workspace.neb_pot_fp)
    if not network_splits:
        return base_pot
    try:
        annotated = load_partial_annotated_pot(workspace)
    except Exception:
        return base_pot

    merged = Pot.read_from_disk(workspace.neb_pot_fp)
    for node_index in annotated.graph.nodes:
        attrs = dict(annotated.graph.nodes[node_index])
        if node_index in merged.graph.nodes:
            merged.graph.nodes[node_index].update(attrs)
        else:
            merged.graph.add_node(node_index, **attrs)
    for source, target in annotated.graph.edges:
        attrs = dict(annotated.graph.edges[(source, target)])
        if merged.graph.has_edge(source, target):
            merged.graph.edges[(source, target)].update(attrs)
        else:
            merged.graph.add_edge(source, target, **attrs)
    return merged


def _namespace_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return _json_safe(value)
    with contextlib.suppress(Exception):
        return _json_safe(vars(value))
    return {}


def _inputs_summary_payload(workspace: RetropathsWorkspace) -> dict[str, Any]:
    summary = {
        "path": str(workspace.inputs_fp),
        "engine_name": "",
        "program": "",
        "path_min_method": "",
        "path_min_inputs": {},
        "chain_inputs": {},
        "network_inputs": {},
        "gi_inputs": {},
        "optimizer_kwds": {},
        "program_kwds": {},
        "can_hessian_sample": False,
        "hessian_sample_note": "",
        "error": "",
    }
    try:
        run_inputs = RunInputs.open(workspace.inputs_fp)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary

    summary["engine_name"] = str(getattr(run_inputs, "engine_name", "") or "")
    summary["program"] = str(getattr(run_inputs, "program", "") or "")
    summary["path_min_method"] = str(getattr(run_inputs, "path_min_method", "") or "")
    summary["path_min_inputs"] = _namespace_payload(getattr(run_inputs, "path_min_inputs", None))
    summary["chain_inputs"] = _namespace_payload(getattr(run_inputs, "chain_inputs", None))
    summary["network_inputs"] = _namespace_payload(getattr(run_inputs, "network_inputs", None))
    summary["gi_inputs"] = _namespace_payload(getattr(run_inputs, "gi_inputs", None))
    summary["optimizer_kwds"] = _namespace_payload(getattr(run_inputs, "optimizer_kwds", None))
    program_kwds = getattr(run_inputs, "program_kwds", None)
    if isinstance(program_kwds, dict):
        summary["program_kwds"] = _json_safe(program_kwds)
    elif hasattr(program_kwds, "model_dump"):
        with contextlib.suppress(Exception):
            summary["program_kwds"] = _json_safe(program_kwds.model_dump())
    engine = getattr(run_inputs, "engine", None)
    if engine is None:
        summary["can_hessian_sample"] = False
        summary["hessian_sample_note"] = "The inputs file did not construct an engine, so Hessian sampling cannot be started."
    elif not (hasattr(engine, "_compute_hessian_result") or hasattr(engine, "compute_hessian")):
        summary["can_hessian_sample"] = False
        summary["hessian_sample_note"] = (
            "The configured engine does not expose Hessian computation "
            "(`_compute_hessian_result` or `compute_hessian`)."
        )
    elif not (hasattr(engine, "compute_geometry_optimization") or hasattr(engine, "compute_geometry_optimizations")):
        summary["can_hessian_sample"] = False
        summary["hessian_sample_note"] = "The configured engine does not expose geometry-optimization methods."
    else:
        summary["can_hessian_sample"] = True
        summary["hessian_sample_note"] = ""
    return summary


def _deployment_program_defaults(program: str) -> tuple[str, str]:
    program_name = str(program or "").strip().lower()
    if program_name == "crest":
        return "gfn2", "gfn2"
    if program_name == "terachem":
        return "ub3lyp", "sto-3g"
    raise ValueError("Program must be either 'crest' or 'terachem'.")


def _normalize_deployment_engine_name(engine_name: str | None) -> str:
    normalized = str(engine_name or "").strip().lower()
    if normalized in {"chemcloud", "qcop", "ase", "qmmm"}:
        return normalized
    return "chemcloud"


def _extract_program_model_fields(program_kwds: Any) -> tuple[str, str]:
    payload: dict[str, Any] = {}
    if isinstance(program_kwds, dict):
        payload = dict(program_kwds)
    elif hasattr(program_kwds, "model_dump"):
        with contextlib.suppress(Exception):
            payload = dict(program_kwds.model_dump())
    model = dict(payload.get("model") or {})
    method = str(model.get("method") or "").strip()
    basis = str(model.get("basis") or "").strip()
    return method, basis


def _drive_defaults_payload(inputs_fp: Path | None, reactions_fp: Path | None) -> dict[str, Any]:
    defaults = {
        "inputs_path": str(inputs_fp.resolve()) if inputs_fp else "",
        "reactions_fp": str(reactions_fp.resolve()) if reactions_fp else "",
        "engine_name": "chemcloud",
        "allowed_programs": ["crest", "terachem"],
        "program": "terachem",
        "method": "ub3lyp",
        "basis": "sto-3g",
    }
    if inputs_fp is None or not inputs_fp.exists():
        return defaults
    try:
        run_inputs = RunInputs.open(inputs_fp)
    except Exception:
        return defaults
    defaults["engine_name"] = _normalize_deployment_engine_name(getattr(run_inputs, "engine_name", ""))
    program = str(getattr(run_inputs, "program", "") or "").strip().lower()
    if program:
        defaults["program"] = program
    configured_method, configured_basis = _extract_program_model_fields(getattr(run_inputs, "program_kwds", None))
    if defaults["program"] in {"crest", "terachem"}:
        program_default_method, program_default_basis = _deployment_program_defaults(defaults["program"])
        defaults["method"] = configured_method or program_default_method
        defaults["basis"] = configured_basis or program_default_basis
    else:
        defaults["method"] = configured_method
        defaults["basis"] = configured_basis
    allowed_programs = ["crest", "terachem"]
    if defaults["program"] and defaults["program"] not in allowed_programs:
        allowed_programs.append(defaults["program"])
    defaults["allowed_programs"] = allowed_programs
    return defaults


def _materialize_deployment_inputs(
    *,
    template_fp: Path,
    output_dir: Path,
    run_name: str,
    theory_program: str | None,
    theory_method: str | None,
    theory_basis: str | None,
) -> Path:
    run_name = _validate_run_name(run_name)
    run_inputs = RunInputs.open(template_fp)
    selected_engine = str(getattr(run_inputs, "engine_name", "") or "").strip().lower()
    if not selected_engine:
        selected_engine = _normalize_deployment_engine_name(getattr(run_inputs, "engine_name", ""))
    selected_program = str(theory_program or getattr(run_inputs, "program", "") or "").strip().lower()
    if not selected_program:
        selected_program = "terachem"
    if selected_program in {"crest", "terachem"}:
        default_method, default_basis = _deployment_program_defaults(selected_program)
    else:
        default_method, default_basis = "", ""

    existing_payload: dict[str, Any] = {}
    existing_program_kwds = getattr(run_inputs, "program_kwds", None)
    if isinstance(existing_program_kwds, dict):
        existing_payload = dict(existing_program_kwds)
    elif hasattr(existing_program_kwds, "model_dump"):
        with contextlib.suppress(Exception):
            existing_payload = dict(existing_program_kwds.model_dump())

    model_payload = dict(existing_payload.get("model") or {})
    existing_method = str(model_payload.get("method") or "").strip()
    existing_basis = str(model_payload.get("basis") or "").strip()
    method = str(theory_method or existing_method or default_method).strip()
    basis = str(theory_basis or existing_basis or default_basis).strip()
    if selected_program in {"crest", "terachem"}:
        method = method or default_method
        basis = basis or default_basis
    model_payload["method"] = method
    model_payload["basis"] = basis

    run_inputs.engine_name = selected_engine
    run_inputs.program = selected_program
    run_inputs.program_kwds = ProgramArgs(
        model=model_payload,
        keywords=dict(existing_payload.get("keywords") or {}),
        extras=dict(existing_payload.get("extras") or {}),
        files=dict(existing_payload.get("files") or {}),
        cmdline_args=list(existing_payload.get("cmdline_args") or []),
    )

    inputs_dir = output_dir / "_drive_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    target_fp = inputs_dir / f"{run_name}.toml"
    run_inputs.save(target_fp)
    return target_fp.resolve()


def _neb_backed_nodes(graph) -> set[int]:
    backed: set[int] = set()
    for source, target in graph.edges:
        attrs = graph.edges[(source, target)]
        if attrs.get("list_of_nebs"):
            backed.add(int(source))
            backed.add(int(target))
    return backed


def _resolve_display_edge_attrs(graph, source: int, target: int) -> tuple[dict[str, Any], bool]:
    attrs = dict(graph.edges[(source, target)])
    if attrs.get("list_of_nebs") or attrs.get("barrier") is not None:
        return attrs, False
    if graph.has_edge(target, source):
        reverse_attrs = dict(graph.edges[(target, source)])
        if reverse_attrs.get("list_of_nebs") or reverse_attrs.get("barrier") is not None:
            return reverse_attrs, True
    return attrs, False


def _lookup_edge_payload_with_reverse(
    payload_by_edge: dict[str, Any],
    *,
    source: int,
    target: int,
    prefer_reverse: bool = False,
) -> Any | None:
    forward_key = f"{int(source)} -> {int(target)}"
    reverse_key = f"{int(target)} -> {int(source)}"
    first_key = reverse_key if prefer_reverse else forward_key
    second_key = forward_key if prefer_reverse else reverse_key
    return payload_by_edge.get(first_key) or payload_by_edge.get(second_key)


def _parse_kmc_initial_conditions(raw: str | None) -> dict[int, float] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Kinetics initial conditions must be a JSON object mapping node ids to populations.")
    normalized: dict[int, float] = {}
    for key, value in payload.items():
        normalized[int(key)] = float(value)
    return normalized


def _kmc_defaults_payload(pot: Pot) -> dict[str, Any]:
    payload = build_kmc_payload(pot, include_labels=False)
    return {
        "available": True,
        "temperature_kelvin": float(payload["temperature_kelvin"]),
        "default_end_time": float(payload["default_end_time"]),
        "default_max_steps": 200,
        "node_count": len(payload["nodes"]),
        "edge_count": len(payload["edges"]),
        "suppressed_edge_count": len(payload.get("suppressed_edges", [])),
        "nodes": [
            {
                "id": int(node["id"]),
                "label": str(node.get("label") or node["id"]),
                "initial": float(node.get("initial", 0.0)),
            }
            for node in payload["nodes"]
        ],
    }


def _run_kmc_payload(
    workspace: RetropathsWorkspace,
    *,
    temperature_kelvin: float,
    final_time: float | None,
    max_steps: int,
    initial_conditions: dict[int, float] | None,
    network_splits: bool = True,
) -> dict[str, Any]:
    pot: Pot | None = None
    if network_splits:
        annotated_fp = getattr(workspace, "annotated_neb_pot_fp", None)
        if annotated_fp is not None and annotated_fp.exists():
            with contextlib.suppress(Exception):
                pot = Pot.read_from_disk(annotated_fp)
    if pot is None:
        pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
    payload = build_kmc_payload(
        pot,
        temperature_kelvin=float(temperature_kelvin),
        initial_conditions=initial_conditions,
        include_labels=False,
    )
    result = simulate_kmc(
        pot,
        temperature_kelvin=float(temperature_kelvin),
        initial_conditions=initial_conditions,
        max_steps=int(max_steps),
        final_time=float(final_time) if final_time is not None else None,
        payload=payload,
    )
    labels = {
        int(node["id"]): str(node.get("label") or node["id"])
        for node in payload["nodes"]
    }
    history = list(result.get("history") or [])
    time_points = [float(step.get("time", 0.0)) for step in history]
    final_populations = {
        int(node_index): float(value)
        for node_index, value in dict(result.get("final_populations") or {}).items()
    }
    ranked_node_ids = [
        node_id
        for node_id, _value in sorted(
            final_populations.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    top_node_ids = ranked_node_ids[: min(6, len(ranked_node_ids))]
    series = [
        {
            "node_id": int(node_id),
            "label": labels.get(int(node_id), str(node_id)),
            "y": [
                float(dict(step.get("populations") or {}).get(int(node_id), 0.0))
                for step in history
            ],
        }
        for node_id in top_node_ids
    ]
    summary = [
        {
            "node_id": int(node_id),
            "label": labels.get(int(node_id), str(node_id)),
            "population": float(population),
        }
        for node_id, population in sorted(
            final_populations.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    dominant_node = summary[0] if summary else None
    return {
        "temperature_kelvin": float(result["temperature_kelvin"]),
        "final_time": float(result["final_time"]),
        "max_steps": int(result["max_steps"]),
        "event_count": max(len(history) - 1, 0),
        "dominant_node": dominant_node,
        "plot": {
            "title": "Population vs time",
            "x_label": "Time (a.u.)",
            "y_label": "Population",
            "x": time_points,
            "series": series,
        },
        "summary": summary,
        "initial_conditions": {
            str(node_index): float(value)
            for node_index, value in dict(payload.get("initial_conditions") or {}).items()
        },
        "suppressed_edges": list(payload.get("suppressed_edges") or []),
    }


def _write_completed_queue_visualizations(
    workspace: RetropathsWorkspace,
    queue: RetropathsNEBQueue,
) -> list[dict[str, Any]]:
    from neb_dynamics.scripts.main_cli import _build_chain_visualizer_html

    rows: list[dict[str, Any]] = []
    out_dir = workspace.edge_visualizations_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in _completed_queue_entries(queue):
        filename = f"queue_edge_{int(item.source_node)}_{int(item.target_node)}.html"
        out_fp = out_dir / filename
        meta_fp = out_dir / f"queue_edge_{int(item.source_node)}_{int(item.target_node)}.meta.json"
        cache_key = {
            "result_dir": str(item.result_dir),
            "output_chain_xyz": str(getattr(item, "output_chain_xyz", "") or ""),
            "finished_at": str(item.finished_at or ""),
        }
        if out_fp.exists() and meta_fp.exists():
            with contextlib.suppress(Exception):
                meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                if (
                    meta.get("result_dir") == cache_key["result_dir"]
                    and str(meta.get("output_chain_xyz") or "") == cache_key["output_chain_xyz"]
                    and meta.get("finished_at") == cache_key["finished_at"]
                    and meta.get("is_elementary_result") is True
                    and "source_structure" in meta
                    and "target_structure" in meta
                ):
                    rows.append(
                        {
                            "edge": f"{int(item.source_node)} -> {int(item.target_node)}",
                            "barrier": meta.get("barrier"),
                            "href": filename,
                            "source_node": int(meta.get("source_node", item.source_node)),
                            "target_node": int(meta.get("target_node", item.target_node)),
                            "source_structure": meta.get("source_structure"),
                            "target_structure": meta.get("target_structure"),
                        }
                    )
                    continue
        queue_payload = _completed_queue_result_payload(item)
        if queue_payload is None:
            continue
        if not queue_payload.get("is_elementary_result", True):
            continue
        chain = queue_payload["chain"]

        title_html = (
            f"<div style=\"font-family: -apple-system, BlinkMacSystemFont, sans-serif; "
            f"margin: 0 0 12px 0; padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa;\">"
            f"<div><strong>Attempted Edge:</strong> {int(item.source_node)} -&gt; {int(item.target_node)}</div>"
            f"<div><strong>Queue Status:</strong> completed</div>"
            f"</div>"
        )
        html = _build_chain_visualizer_html(chain=chain, chain_trajectory=None)
        html = html.replace("<body>", f"<body>\n  {title_html}", 1)
        out_fp.write_text(html, encoding="utf-8")

        barrier = queue_payload["barrier"]
        with contextlib.suppress(Exception):
            meta_fp.write_text(
                json.dumps(
                    {
                        **cache_key,
                        "source_node": queue_payload["source_node"],
                        "target_node": queue_payload["target_node"],
                        "source_structure": queue_payload["source_structure"],
                        "target_structure": queue_payload["target_structure"],
                        "barrier": barrier,
                        "is_elementary_result": True,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        rows.append(
            {
                "edge": f"{int(item.source_node)} -> {int(item.target_node)}",
                "barrier": barrier,
                "href": filename,
                "source_node": queue_payload["source_node"],
                "target_node": queue_payload["target_node"],
                "source_structure": queue_payload["source_structure"],
                "target_structure": queue_payload["target_structure"],
            }
        )

    return rows


def _read_edge_visualization_metadata(
    workspace: RetropathsWorkspace,
    pot: Pot,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    out_dir = workspace.edge_visualizations_dir
    if not out_dir.exists():
        return rows
    for source_node, target_node in sorted(pot.graph.edges):
        filename = f"edge_{source_node}_{target_node}.html"
        meta_fp = out_dir / f"edge_{source_node}_{target_node}.meta.json"
        if not meta_fp.exists():
            continue
        with contextlib.suppress(Exception):
            meta = json.loads(meta_fp.read_text(encoding="utf-8"))
            rows.append(
                {
                    "edge": str(meta.get("edge") or f"{source_node} -> {target_node}"),
                    "start": str(meta.get("start") or source_node),
                    "end": str(meta.get("end") or target_node),
                    "reaction": str(meta.get("reaction") or ""),
                    "barrier": str(meta.get("barrier") or ""),
                    "chains": str(meta.get("chains") or ""),
                    "href": filename,
                    "source_structure": meta.get("source_structure"),
                    "target_structure": meta.get("target_structure"),
                }
            )
    return rows


def _read_completed_queue_visualization_metadata(
    workspace: RetropathsWorkspace,
    queue: RetropathsNEBQueue,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    out_dir = workspace.edge_visualizations_dir
    if not out_dir.exists():
        return rows
    for item in _completed_queue_entries(queue):
        filename = f"queue_edge_{int(item.source_node)}_{int(item.target_node)}.html"
        meta_fp = out_dir / f"queue_edge_{int(item.source_node)}_{int(item.target_node)}.meta.json"
        if not meta_fp.exists():
            continue
        with contextlib.suppress(Exception):
            meta = json.loads(meta_fp.read_text(encoding="utf-8"))
            rows.append(
                {
                    "edge": f"{int(item.source_node)} -> {int(item.target_node)}",
                    "barrier": meta.get("barrier"),
                    "href": filename,
                    "source_node": int(meta.get("source_node", item.source_node)),
                    "target_node": int(meta.get("target_node", item.target_node)),
                    "source_structure": meta.get("source_structure"),
                    "target_structure": meta.get("target_structure"),
                }
            )
    return rows


def _build_drive_payload(
    workspace: RetropathsWorkspace,
    *,
    product_smiles: str = "",
    active_job_label: str = "",
    active_action: dict[str, Any] | None = None,
    network_splits: bool = True,
) -> dict[str, Any]:
    retropaths_nodes = 0
    retropaths_edges = 0
    with contextlib.suppress(Exception):
        retropaths_pot = load_retropaths_pot(workspace)
        retropaths_nodes = int(retropaths_pot.graph.number_of_nodes())
        retropaths_edges = int(retropaths_pot.graph.number_of_edges())
    queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)
    pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
    inputs_summary = _inputs_summary_payload(workspace)
    display_run_inputs = None
    with contextlib.suppress(Exception):
        display_run_inputs = RunInputs.open(workspace.inputs_fp)
    edge_visualizations = _write_edge_visualizations(workspace=workspace, pot=pot)
    completed_queue_visualizations = _write_completed_queue_visualizations(workspace=workspace, queue=queue)
    viewer_by_edge = {
        str(item.get("edge") or ""): f"edge_visualizations/{item['href']}"
        for item in edge_visualizations
        if item.get("edge") and item.get("href")
    }
    viewer_row_by_edge = {
        str(item.get("edge") or ""): item
        for item in edge_visualizations
        if item.get("edge")
    }
    queue_result_by_edge = {
        str(item.get("edge") or ""): item
        for item in completed_queue_visualizations
        if item.get("edge")
    }
    template_payloads = _load_template_payloads(workspace)
    explorer = _build_network_explorer_payload(
        pot.graph,
        template_payloads=template_payloads,
        edge_visualizations=edge_visualizations,
    )
    optimized_endpoints = sum(
        bool(pot.graph.nodes[node_index].get("endpoint_optimized"))
        for node_index in pot.graph.nodes
    )
    neb_backed_edges = sum(
        bool(pot.graph.edges[(source, target)].get("list_of_nebs"))
        for source, target in pot.graph.edges
    )

    queue_by_edge = {
        (int(item.source_node), int(item.target_node)): item
        for item in queue.items
    }
    backed_nodes = _neb_backed_nodes(pot.graph)
    normalized_product = product_smiles.strip()

    for node in explorer["nodes"]:
        node_index = int(node["id"])
        attrs = pot.graph.nodes[node_index]
        node["structure"] = _node_structure_payload(attrs)
        node["endpoint_optimized"] = bool(attrs.get("endpoint_optimized"))
        node["endpoint_optimization_error"] = str(attrs.get("endpoint_optimization_error") or "")
        node["minimizable"], node["minimize_note"] = _node_minimize_status(node_index, attrs)
        node["can_apply_reactions"], node["apply_reactions_note"] = _node_apply_reaction_status(node_index, attrs)
        node["can_nanoreactor"], node["nanoreactor_note"] = _node_nanoreactor_status(node_index, attrs, inputs_summary)
        node["can_hessian_sample"], node["hessian_sample_note"] = _node_hessian_sample_status(node_index, attrs, inputs_summary)
        node["neb_backed"] = node_index in backed_nodes
        node["is_target"] = bool(normalized_product and str(node["label"]) == normalized_product)

    for edge in explorer["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        attrs = pot.graph.edges[(source, target)]
        display_attrs, used_reverse_result = _resolve_display_edge_attrs(pot.graph, source, target)
        queue_item = queue_by_edge.get((source, target))
        queue_result = _lookup_edge_payload_with_reverse(
            queue_result_by_edge,
            source=source,
            target=target,
            prefer_reverse=used_reverse_result,
        )
        viewer_edge_key = f"{target} -> {source}" if used_reverse_result else f"{source} -> {target}"
        viewer_row = viewer_row_by_edge.get(viewer_edge_key)
        edge["neb_backed"] = bool(display_attrs.get("list_of_nebs"))
        edge["barrier"] = float(display_attrs["barrier"]) if display_attrs.get("barrier") is not None else None
        edge["chains"] = len(display_attrs.get("list_of_nebs") or [])
        edge["viewer_href"] = viewer_by_edge.get(viewer_edge_key)
        edge["data"] = _json_safe(
            {
                key: value
                for key, value in dict(display_attrs).items()
                if key != "list_of_nebs"
            }
        )
        edge["result_from_reverse_edge"] = bool(used_reverse_result)
        edge["result_from_completed_queue"] = False
        if queue_result is not None:
            if edge["barrier"] is None and queue_result.get("barrier") is not None:
                edge["barrier"] = float(queue_result["barrier"])
            if queue_result.get("href"):
                edge["viewer_href"] = f"edge_visualizations/{queue_result['href']}"
            if edge["chains"] == 0:
                edge["chains"] = 1
            edge["neb_backed"] = edge["neb_backed"] or bool(edge["viewer_href"]) or edge["barrier"] is not None
            edge["result_from_completed_queue"] = True
        edge["queue_status"] = queue_item.status if queue_item is not None else ""
        edge["queue_error"] = queue_item.error if queue_item is not None else ""
        edge["can_queue_neb"], edge["queue_note"] = _edge_neb_status(edge)
        edge["can_hessian_sample"], edge["hessian_sample_note"] = _edge_hessian_sample_status(edge)
        edge["source_structure"] = _node_structure_payload(pot.graph.nodes[source])
        edge["target_structure"] = _node_structure_payload(pot.graph.nodes[target])
        viewer_source_structure = viewer_row.get("source_structure") if isinstance(viewer_row, dict) else None
        viewer_target_structure = viewer_row.get("target_structure") if isinstance(viewer_row, dict) else None
        if viewer_source_structure is not None:
            edge["source_structure"] = viewer_source_structure
        if viewer_target_structure is not None:
            edge["target_structure"] = viewer_target_structure
        if queue_result is not None:
            queue_source = int(queue_result.get("source_node", source))
            queue_target = int(queue_result.get("target_node", target))
            if queue_source == source and queue_target == target:
                edge["source_structure"] = queue_result.get("source_structure") or edge["source_structure"]
                edge["target_structure"] = queue_result.get("target_structure") or edge["target_structure"]
            elif queue_source == target and queue_target == source:
                edge["source_structure"] = queue_result.get("target_structure") or edge["source_structure"]
                edge["target_structure"] = queue_result.get("source_structure") or edge["target_structure"]
        if (
            (edge["source_structure"] is None or edge["target_structure"] is None)
            and display_run_inputs is not None
        ):
            with contextlib.suppress(Exception):
                pair_item = queue_item or NEBQueueItem(
                    job_id=f"{source}->{target}",
                    source_node=source,
                    target_node=target,
                    reaction=str(attrs.get("reaction") or ""),
                    attempt_key="preview",
                )
                display_pair = _make_pair_chain(pot=pot, item=pair_item, run_inputs=display_run_inputs)
                edge["source_structure"] = _structure_payload_from_structure(
                    display_pair.nodes[0].structure,
                    molecule_like=getattr(display_pair.nodes[0], "graph", None),
                )
                edge["target_structure"] = _structure_payload_from_structure(
                    display_pair.nodes[-1].structure,
                    molecule_like=getattr(display_pair.nodes[-1], "graph", None),
                )

    user_started_items = [
        item for item in queue.items
        if item.started_at is not None
        or item.finished_at is not None
        or item.status in {"running", "completed", "failed", "skipped_attempted"}
    ]
    drive_queue_counts = {
        "items": len(user_started_items),
        "pending": 0,
        "running": sum(item.status == "running" for item in user_started_items),
        "completed": sum(item.status == "completed" for item in user_started_items),
        "failed": sum(item.status == "failed" for item in user_started_items),
        "incompatible": sum(item.status == "incompatible" for item in user_started_items),
    }
    if active_action is not None and active_action.get("status") == "running":
        drive_queue_counts["running"] += 1

    return {
        "workspace": {
            "run_name": workspace.run_name,
            "directory": workspace.workdir,
            "root_smiles": workspace.root_smiles,
            "environment_smiles": workspace.environment_smiles,
            "reactions_fp": str(workspace.reactions_path),
            "retropaths_nodes": int(retropaths_nodes),
            "retropaths_edges": int(retropaths_edges),
            "network_nodes": int(pot.graph.number_of_nodes()),
            "network_edges": int(pot.graph.number_of_edges()),
            "optimized_endpoints": int(optimized_endpoints),
            "neb_backed_edges": int(neb_backed_edges),
        },
        "inputs": inputs_summary,
        "kmc": _kmc_defaults_payload(pot),
        "queue": drive_queue_counts,
        "version": _drive_network_version(workspace),
        "network": explorer,
    }


def _build_drive_payload_fast(
    workspace: RetropathsWorkspace,
    *,
    product_smiles: str = "",
    active_job_label: str = "",
    active_action: dict[str, Any] | None = None,
    network_splits: bool = True,
) -> dict[str, Any]:
    queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)
    pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
    inputs_summary = _inputs_summary_payload(workspace)
    normalized_product = product_smiles.strip()
    explorer = _build_network_explorer_payload(pot.graph)
    edge_visualizations = _read_edge_visualization_metadata(workspace=workspace, pot=pot)
    # Fast snapshots should still surface newly completed queue results while an action
    # (notably Hawaii) is running, otherwise barriers can appear without a viewer link
    # until an eventual full rebuild occurs.
    completed_queue_visualizations = _write_completed_queue_visualizations(workspace=workspace, queue=queue)
    viewer_by_edge = {
        str(item.get("edge") or ""): f"edge_visualizations/{item['href']}"
        for item in edge_visualizations
        if item.get("edge") and item.get("href")
    }
    queue_result_by_edge = {
        str(item.get("edge") or ""): item
        for item in completed_queue_visualizations
        if item.get("edge")
    }

    queue_by_edge = {
        (int(item.source_node), int(item.target_node)): item
        for item in queue.items
    }
    backed_nodes = _neb_backed_nodes(pot.graph)
    for node in explorer["nodes"]:
        node_index = int(node["id"])
        attrs = pot.graph.nodes[node_index]
        node["structure"] = _node_structure_payload_fast(attrs)
        node["endpoint_optimized"] = bool(attrs.get("endpoint_optimized"))
        node["endpoint_optimization_error"] = str(attrs.get("endpoint_optimization_error") or "")
        node["minimizable"], node["minimize_note"] = _node_minimize_status(node_index, attrs)
        node["can_apply_reactions"], node["apply_reactions_note"] = _node_apply_reaction_status(node_index, attrs)
        node["can_nanoreactor"], node["nanoreactor_note"] = _node_nanoreactor_status(node_index, attrs, inputs_summary)
        node["can_hessian_sample"], node["hessian_sample_note"] = _node_hessian_sample_status(node_index, attrs, inputs_summary)
        node["neb_backed"] = node_index in backed_nodes
        node["is_target"] = bool(normalized_product and str(node["label"]) == normalized_product)

    for edge in explorer["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        attrs = pot.graph.edges[(source, target)]
        display_attrs, used_reverse_result = _resolve_display_edge_attrs(pot.graph, source, target)
        queue_item = queue_by_edge.get((source, target))
        queue_result = _lookup_edge_payload_with_reverse(
            queue_result_by_edge,
            source=source,
            target=target,
            prefer_reverse=used_reverse_result,
        )
        viewer_edge_key = f"{target} -> {source}" if used_reverse_result else f"{source} -> {target}"
        edge["neb_backed"] = bool(display_attrs.get("list_of_nebs"))
        edge["barrier"] = float(display_attrs["barrier"]) if display_attrs.get("barrier") is not None else None
        edge["chains"] = len(display_attrs.get("list_of_nebs") or [])
        edge["viewer_href"] = viewer_by_edge.get(viewer_edge_key)
        edge["result_from_reverse_edge"] = bool(used_reverse_result)
        edge["result_from_completed_queue"] = False
        if queue_result is not None:
            if edge["barrier"] is None and queue_result.get("barrier") is not None:
                edge["barrier"] = float(queue_result["barrier"])
            if queue_result.get("href"):
                edge["viewer_href"] = f"edge_visualizations/{queue_result['href']}"
            if edge["chains"] == 0:
                edge["chains"] = 1
            edge["neb_backed"] = edge["neb_backed"] or bool(edge["viewer_href"]) or edge["barrier"] is not None
            edge["result_from_completed_queue"] = True
        edge["queue_status"] = queue_item.status if queue_item is not None else ""
        edge["queue_error"] = queue_item.error if queue_item is not None else ""
        edge["can_queue_neb"], edge["queue_note"] = _edge_neb_status(edge)
        edge["can_hessian_sample"], edge["hessian_sample_note"] = _edge_hessian_sample_status(edge)
        edge["source_structure"] = _node_structure_payload_fast(pot.graph.nodes[source])
        edge["target_structure"] = _node_structure_payload_fast(pot.graph.nodes[target])
        if queue_result is not None:
            queue_source = int(queue_result.get("source_node", source))
            queue_target = int(queue_result.get("target_node", target))
            if queue_source == source and queue_target == target:
                edge["source_structure"] = queue_result.get("source_structure") or edge["source_structure"]
                edge["target_structure"] = queue_result.get("target_structure") or edge["target_structure"]
            elif queue_source == target and queue_target == source:
                edge["source_structure"] = queue_result.get("target_structure") or edge["source_structure"]
                edge["target_structure"] = queue_result.get("source_structure") or edge["target_structure"]

    user_started_items = [
        item for item in queue.items
        if item.started_at is not None
        or item.finished_at is not None
        or item.status in {"running", "completed", "failed", "skipped_attempted"}
    ]
    drive_queue_counts = {
        "items": len(user_started_items),
        "pending": 0,
        "running": sum(item.status == "running" for item in user_started_items),
        "completed": sum(item.status == "completed" for item in user_started_items),
        "failed": sum(item.status == "failed" for item in user_started_items),
        "incompatible": sum(item.status == "incompatible" for item in user_started_items),
    }
    if active_action is not None and active_action.get("status") == "running":
        drive_queue_counts["running"] += 1

    optimized_endpoints = sum(
        bool(pot.graph.nodes[node_index].get("endpoint_optimized"))
        for node_index in pot.graph.nodes
    )

    return {
        "workspace": {
            "run_name": workspace.run_name,
            "directory": workspace.workdir,
            "root_smiles": workspace.root_smiles,
            "environment_smiles": workspace.environment_smiles,
            "reactions_fp": str(workspace.reactions_path),
            "retropaths_nodes": 0,
            "retropaths_edges": 0,
            "network_nodes": int(pot.graph.number_of_nodes()),
            "network_edges": int(pot.graph.number_of_edges()),
            "optimized_endpoints": int(optimized_endpoints),
            "neb_backed_edges": sum(
                bool(pot.graph.edges[(source, target)].get("list_of_nebs"))
                for source, target in pot.graph.edges
            ),
        },
        "inputs": inputs_summary,
        "kmc": {
            "available": False,
            "node_count": int(pot.graph.number_of_nodes()),
            "edge_count": 0,
            "suppressed_edge_count": 0,
            "nodes": [],
        },
        "queue": drive_queue_counts,
        "version": _drive_network_version(workspace),
        "network": explorer,
    }


def _build_drive_payload_fast_neb(
    workspace: RetropathsWorkspace,
    *,
    product_smiles: str = "",
    active_job_label: str = "",
    active_action: dict[str, Any] | None = None,
    network_splits: bool = True,
) -> dict[str, Any]:
    queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)
    pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
    inputs_summary = _inputs_summary_payload(workspace)
    normalized_product = product_smiles.strip()
    explorer = _build_network_explorer_payload(pot.graph)
    edge_visualizations = _write_edge_visualizations(workspace=workspace, pot=pot)
    completed_queue_visualizations = _write_completed_queue_visualizations(workspace=workspace, queue=queue)
    viewer_by_edge = {
        str(item.get("edge") or ""): f"edge_visualizations/{item['href']}"
        for item in edge_visualizations
        if item.get("edge") and item.get("href")
    }
    queue_result_by_edge = {
        str(item.get("edge") or ""): item
        for item in completed_queue_visualizations
        if item.get("edge")
    }

    queue_by_edge = {
        (int(item.source_node), int(item.target_node)): item
        for item in queue.items
    }
    active_edge = None
    if active_action is not None and active_action.get("type") == "neb":
        active_edge = (
            int(active_action.get("source_node", -1)),
            int(active_action.get("target_node", -1)),
        )

    for node in explorer["nodes"]:
        node_index = int(node["id"])
        attrs = pot.graph.nodes[node_index]
        node["structure"] = _node_structure_payload_fast(attrs)
        node["endpoint_optimized"] = bool(attrs.get("endpoint_optimized"))
        node["endpoint_optimization_error"] = str(attrs.get("endpoint_optimization_error") or "")
        node["minimizable"], node["minimize_note"] = _node_minimize_status(node_index, attrs)
        node["can_apply_reactions"], node["apply_reactions_note"] = _node_apply_reaction_status(node_index, attrs)
        node["can_nanoreactor"], node["nanoreactor_note"] = _node_nanoreactor_status(node_index, attrs, inputs_summary)
        node["can_hessian_sample"], node["hessian_sample_note"] = _node_hessian_sample_status(node_index, attrs, inputs_summary)
        node["neb_backed"] = node_index in _neb_backed_nodes(pot.graph)
        node["is_target"] = bool(normalized_product and str(node["label"]) == normalized_product)

    for edge in explorer["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        display_attrs, used_reverse_result = _resolve_display_edge_attrs(pot.graph, source, target)
        queue_item = queue_by_edge.get((source, target))
        queue_result = _lookup_edge_payload_with_reverse(
            queue_result_by_edge,
            source=source,
            target=target,
            prefer_reverse=used_reverse_result,
        )
        viewer_edge_key = f"{target} -> {source}" if used_reverse_result else f"{source} -> {target}"
        edge["neb_backed"] = bool((source, target) == active_edge) or bool(display_attrs.get("list_of_nebs"))
        edge["barrier"] = float(display_attrs["barrier"]) if display_attrs.get("barrier") is not None else None
        edge["chains"] = len(display_attrs.get("list_of_nebs") or [])
        edge["viewer_href"] = viewer_by_edge.get(viewer_edge_key)
        edge["result_from_reverse_edge"] = bool(used_reverse_result)
        edge["result_from_completed_queue"] = False
        if queue_result is not None:
            if edge["barrier"] is None and queue_result.get("barrier") is not None:
                edge["barrier"] = float(queue_result["barrier"])
            if queue_result.get("href"):
                edge["viewer_href"] = f"edge_visualizations/{queue_result['href']}"
            if edge["chains"] == 0:
                edge["chains"] = 1
            edge["neb_backed"] = edge["neb_backed"] or bool(edge["viewer_href"]) or edge["barrier"] is not None
            edge["result_from_completed_queue"] = True
        edge["queue_status"] = queue_item.status if queue_item is not None else ""
        edge["queue_error"] = queue_item.error if queue_item is not None else ""
        edge["can_queue_neb"], edge["queue_note"] = _edge_neb_status(edge)
        edge["can_hessian_sample"], edge["hessian_sample_note"] = _edge_hessian_sample_status(edge)
        edge["source_structure"] = _node_structure_payload_fast(pot.graph.nodes[source])
        edge["target_structure"] = _node_structure_payload_fast(pot.graph.nodes[target])
        if queue_result is not None:
            queue_source = int(queue_result.get("source_node", source))
            queue_target = int(queue_result.get("target_node", target))
            if queue_source == source and queue_target == target:
                edge["source_structure"] = queue_result.get("source_structure") or edge["source_structure"]
                edge["target_structure"] = queue_result.get("target_structure") or edge["target_structure"]
            elif queue_source == target and queue_target == source:
                edge["source_structure"] = queue_result.get("target_structure") or edge["source_structure"]
                edge["target_structure"] = queue_result.get("source_structure") or edge["target_structure"]

    user_started_items = [
        item for item in queue.items
        if item.started_at is not None
        or item.finished_at is not None
        or item.status in {"running", "completed", "failed", "skipped_attempted"}
    ]
    drive_queue_counts = {
        "items": len(user_started_items),
        "pending": 0,
        "running": sum(item.status == "running" for item in user_started_items),
        "completed": sum(item.status == "completed" for item in user_started_items),
        "failed": sum(item.status == "failed" for item in user_started_items),
        "incompatible": sum(item.status == "incompatible" for item in user_started_items),
    }
    if active_action is not None and active_action.get("status") == "running" and drive_queue_counts["running"] == 0:
        drive_queue_counts["running"] = 1

    optimized_endpoints = sum(
        bool(pot.graph.nodes[node_index].get("endpoint_optimized"))
        for node_index in pot.graph.nodes
    )

    return {
        "workspace": {
            "run_name": workspace.run_name,
            "directory": workspace.workdir,
            "root_smiles": workspace.root_smiles,
            "environment_smiles": workspace.environment_smiles,
            "reactions_fp": str(workspace.reactions_path),
            "retropaths_nodes": 0,
            "retropaths_edges": 0,
            "network_nodes": int(pot.graph.number_of_nodes()),
            "network_edges": int(pot.graph.number_of_edges()),
            "optimized_endpoints": int(optimized_endpoints),
            "neb_backed_edges": sum(
                bool(pot.graph.edges[(source, target)].get("list_of_nebs"))
                for source, target in pot.graph.edges
            ),
        },
        "inputs": inputs_summary,
        "kmc": {
            "available": False,
            "node_count": int(pot.graph.number_of_nodes()),
            "edge_count": 0,
            "suppressed_edge_count": 0,
            "nodes": [],
        },
        "queue": drive_queue_counts,
        "version": _drive_network_version(workspace),
        "network": explorer,
    }


def _initialize_workspace_job(
    *,
    reactant: dict[str, Any],
    product: dict[str, Any] | None,
    run_name: str,
    workspace_dir: str,
    inputs_fp: str,
    reactions_fp: str | None,
    environment_smiles: str = "",
    timeout_seconds: int,
    max_nodes: int,
    max_depth: int,
    max_parallel_nebs: int,
    parallel_autosplit_nebs: bool = False,
    parallel_autosplit_workers: int = 4,
    seed_only: bool = False,
    network_splits: bool = True,
    progress_fp: str | None = None,
    allowed_base_dir: str | None = None,
) -> dict[str, Any]:
    workspace_path = Path(workspace_dir).resolve()
    allowed_base = Path(allowed_base_dir).resolve() if allowed_base_dir else None
    _validate_workspace_path_for_init(
        workspace_path,
        allowed_base_dir=allowed_base,
    )
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace = create_workspace(
        root_smiles=reactant["smiles"],
        environment_smiles=str(environment_smiles or ""),
        inputs_fp=inputs_fp,
        reactions_fp=reactions_fp,
        name=run_name,
        directory=workspace_path,
        timeout_seconds=timeout_seconds,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_parallel_nebs=max_parallel_nebs,
        parallel_autosplit_nebs=bool(parallel_autosplit_nebs),
        parallel_autosplit_workers=max(1, int(parallel_autosplit_workers)),
    )
    init_error = ""
    if seed_only:
        _bootstrap_minimal_drive_workspace(
            workspace,
            reactant=reactant,
            product=product,
        )
        if not workspace.retropaths_pot_fp.exists():
            workspace.retropaths_pot_fp.write_text("{}", encoding="utf-8")
        with contextlib.suppress(Exception):
            _apply_bootstrap_species_overrides(workspace, reactant=reactant, product=product)
    else:
        try:
            if progress_fp:
                initialize_workspace_with_progress(workspace, progress_fp=progress_fp)
            else:
                prepare_neb_workspace(workspace)
            _apply_bootstrap_species_overrides(workspace, reactant=reactant, product=product)
        except Exception as exc:
            init_error = f"{type(exc).__name__}: {exc}"
            required_core = [
                workspace.workspace_fp,
                workspace.neb_pot_fp,
                workspace.queue_fp,
            ]
            if not all(fp.exists() for fp in required_core):
                with contextlib.suppress(Exception):
                    _bootstrap_minimal_drive_workspace(
                        workspace,
                        reactant=reactant,
                        product=product,
                    )
            if not workspace.retropaths_pot_fp.exists():
                with contextlib.suppress(Exception):
                    workspace.retropaths_pot_fp.write_text("{}", encoding="utf-8")
            with contextlib.suppress(Exception):
                _apply_bootstrap_species_overrides(workspace, reactant=reactant, product=product)
            required_partial = [*required_core, workspace.retropaths_pot_fp]
            if not all(fp.exists() for fp in required_partial):
                raise
    try:
        _force_minimize_initial_input_structures(workspace)
    except Exception as exc:
        minimization_error = (
            "Initial input-structure minimization failed: "
            f"{type(exc).__name__}: {exc}"
        )
        if init_error:
            init_error = f"{init_error}; {minimization_error}"
        else:
            init_error = minimization_error
    payload = _workspace_snapshot_payload(
        workspace,
        reactant=reactant,
        product=product,
        message=(
            f"Initialized partial workspace {workspace.run_name}."
            if init_error
            else f"Initialized workspace {workspace.run_name}."
        ),
    )
    payload["network_splits"] = bool(network_splits)
    if init_error:
        payload["error"] = init_error
    return payload


def _workspace_snapshot_payload(
    workspace: RetropathsWorkspace,
    *,
    reactant: dict[str, Any] | None = None,
    product: dict[str, Any] | None = None,
    message: str = "",
) -> dict[str, Any]:
    return {
        "workspace": {
            "workdir": workspace.workdir,
            "run_name": workspace.run_name,
            "root_smiles": workspace.root_smiles,
            "environment_smiles": workspace.environment_smiles,
            "inputs_fp": workspace.inputs_fp,
            "reactions_fp": workspace.reactions_fp,
            "timeout_seconds": workspace.timeout_seconds,
            "max_nodes": workspace.max_nodes,
            "max_depth": workspace.max_depth,
            "max_parallel_nebs": workspace.max_parallel_nebs,
            "parallel_autosplit_nebs": bool(getattr(workspace, "parallel_autosplit_nebs", False)),
            "parallel_autosplit_workers": int(getattr(workspace, "parallel_autosplit_workers", 4) or 4),
        },
        "reactant": reactant if reactant is not None else {"smiles": workspace.root_smiles},
        "product": product,
        "message": message or f"Loaded workspace {workspace.run_name}.",
    }


def _workspace_charge_multiplicity(workspace: RetropathsWorkspace) -> tuple[int, int]:
    with contextlib.suppress(Exception):
        pot = Pot.read_from_disk(workspace.neb_pot_fp)
        for node_index in pot.graph.nodes:
            td = pot.graph.nodes[node_index].get("td")
            structure = getattr(td, "structure", None)
            if structure is None:
                continue
            return int(structure.charge), int(structure.multiplicity)
    return 0, 1


def _optimize_selected_nodes(
    workspace: RetropathsWorkspace,
    node_indices: list[int] | None,
    progress: Callable[[str], None] | None = None,
    on_node_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if progress is None:
        progress = lambda _message: None
    if on_node_update is None:
        on_node_update = lambda _payload: None

    run_inputs = RunInputs.open(workspace.inputs_fp)
    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    selected = set(int(node_index) for node_index in (node_indices or []))

    pending_indices: list[int] = []
    pending_nodes = []
    for node_index in pot.graph.nodes:
        if selected and int(node_index) not in selected:
            continue
        attrs = pot.graph.nodes[node_index]
        if attrs.get("td") is None:
            continue
        pending_indices.append(int(node_index))
        pending_nodes.append(attrs["td"])

    if not pending_nodes:
        return {"message": "No geometries matched the minimization request."}

    total = len(pending_indices)
    batch_optimizer = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    compute_program = str(getattr(run_inputs.engine, "compute_program", "") or "").lower()
    engine_name = str(getattr(run_inputs, "engine_name", "") or "").lower()
    use_chemcloud_batch = callable(batch_optimizer) and (compute_program == "chemcloud" or engine_name == "chemcloud") and total > 1

    if use_chemcloud_batch:
        progress(f"Submitting {total} geometry optimizations to ChemCloud in parallel.")
        for offset, node_index in enumerate(pending_indices, start=1):
            on_node_update(
                {
                    "node_id": int(node_index),
                    "status": "running",
                    "index": offset,
                    "total": total,
                }
            )
        results = _run_geometry_optimization_batch_with_trajectories(
            nodes=pending_nodes,
            run_inputs=run_inputs,
        )
        for offset, (node_index, (optimized_td, error, trajectory)) in enumerate(zip(pending_indices, results), start=1):
            progress(f"Collecting ChemCloud geometry {offset}/{total}: node {node_index}")
            attrs = pot.graph.nodes[node_index]
            result_fp = None
            if error is None:
                result_fp = _persist_endpoint_optimization_result(
                    workspace=workspace,
                    node_index=node_index,
                    optimized_td=optimized_td,
                )
                if result_fp is not None:
                    optimized_td = _strip_cached_result(optimized_td)
            attrs["td"] = optimized_td
            attrs["endpoint_optimized"] = error is None
            if error is None:
                attrs.pop("endpoint_optimization_error", None)
                if result_fp is not None:
                    attrs["endpoint_optimization_result_fp"] = result_fp
            else:
                attrs["endpoint_optimization_error"] = error
                attrs.pop("endpoint_optimization_result_fp", None)
            pot.write_to_disk(workspace.neb_pot_fp)
            if error is None:
                progress(f"Finished geometry {offset}/{total}: node {node_index}")
                on_node_update(
                    {
                        "node_id": int(node_index),
                        "status": "completed",
                        "index": offset,
                        "total": total,
                        "plot": _trajectory_plot_payload(
                            trajectory,
                            title=f"Node {node_index} optimization trajectory",
                            x_label="Trajectory step",
                            y_label="Energy (Hartree)",
                        ),
                        "structure": _node_structure_payload(attrs),
                    }
                )
            else:
                progress(f"Geometry {offset}/{total} failed for node {node_index}: {error}")
                on_node_update(
                    {
                        "node_id": int(node_index),
                        "status": "failed",
                        "index": offset,
                        "total": total,
                        "error": error,
                    }
                )
    else:
        for offset, (node_index, node) in enumerate(zip(pending_indices, pending_nodes), start=1):
            progress(f"Optimizing geometry {offset}/{total}: node {node_index}")
            on_node_update(
                {
                    "node_id": int(node_index),
                    "status": "running",
                    "index": offset,
                    "total": total,
                }
            )
            optimized_td, error, trajectory = _run_geometry_optimization_with_trajectory(node=node, run_inputs=run_inputs)
            attrs = pot.graph.nodes[node_index]
            result_fp = None
            if error is None:
                result_fp = _persist_endpoint_optimization_result(
                    workspace=workspace,
                    node_index=node_index,
                    optimized_td=optimized_td,
                )
                if result_fp is not None:
                    optimized_td = _strip_cached_result(optimized_td)
            attrs["td"] = optimized_td
            attrs["endpoint_optimized"] = error is None
            if error is None:
                attrs.pop("endpoint_optimization_error", None)
                if result_fp is not None:
                    attrs["endpoint_optimization_result_fp"] = result_fp
            else:
                attrs["endpoint_optimization_error"] = error
                attrs.pop("endpoint_optimization_result_fp", None)
            pot.write_to_disk(workspace.neb_pot_fp)
            if error is None:
                progress(f"Finished geometry {offset}/{total}: node {node_index}")
                on_node_update(
                    {
                        "node_id": int(node_index),
                        "status": "completed",
                        "index": offset,
                        "total": total,
                        "plot": _trajectory_plot_payload(
                            trajectory,
                            title=f"Node {node_index} optimization trajectory",
                            x_label="Trajectory step",
                            y_label="Energy (Hartree)",
                        ),
                        "structure": _node_structure_payload(attrs),
                    }
                )
            else:
                progress(f"Geometry {offset}/{total} failed for node {node_index}: {error}")
                on_node_update(
                    {
                        "node_id": int(node_index),
                        "status": "failed",
                        "index": offset,
                        "total": total,
                        "error": error,
                    }
                )

    return {
        "message": f"Queued minimization finished for {len(pending_indices)} geometries.",
        "node_indices": pending_indices,
    }


def _initial_input_minimization_targets(workspace: RetropathsWorkspace) -> list[int]:
    if not workspace.neb_pot_fp.exists():
        return []
    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    targets: set[int] = set()
    if 0 in pot.graph.nodes and pot.graph.nodes[0].get("td") is not None:
        targets.add(0)
    for node_index in pot.graph.nodes:
        attrs = pot.graph.nodes[node_index]
        if attrs.get("td") is None:
            continue
        if str(attrs.get("generated_by") or "") == "drive_product_smiles":
            targets.add(int(node_index))
    return sorted(int(node_index) for node_index in targets)


def _force_minimize_initial_input_structures(workspace: RetropathsWorkspace) -> dict[str, Any] | None:
    inputs_fp_value = getattr(workspace, "inputs_fp", None)
    if not inputs_fp_value:
        return None
    inputs_fp = Path(str(inputs_fp_value)).expanduser()
    if not inputs_fp.exists():
        return None
    node_indices = _initial_input_minimization_targets(workspace)
    if not node_indices:
        return None
    return _optimize_selected_nodes(
        workspace,
        node_indices=node_indices,
        progress=lambda _message: None,
    )


def _run_selected_edge_neb(
    workspace: RetropathsWorkspace,
    *,
    source_node: int,
    target_node: int,
    network_splits: bool = True,
) -> dict[str, Any]:
    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    run_inputs = RunInputs.open(workspace.inputs_fp)
    queue = build_retropaths_neb_queue(
        pot=pot,
        queue_fp=workspace.queue_fp,
        overwrite=False,
    )
    item = queue.find_item(source_node, target_node)
    if item is None:
        raise ValueError(f"Edge {source_node} -> {target_node} is not present in the NEB queue.")
    if item.status == "completed":
        return {"message": f"Edge {source_node} -> {target_node} already has a completed NEB result."}
    if item.status == "running":
        return {"message": f"Edge {source_node} -> {target_node} is already running."}
    if item.status in {"incompatible", "missing_td", "skipped_attempted"}:
        raise ValueError(item.error or f"Edge {source_node} -> {target_node} cannot be queued from its current state ({item.status}).")

    item.status = "running"
    item.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    queue.attempted_pairs[item.attempt_key] = {
        "job_id": item.job_id,
        "source_node": item.source_node,
        "target_node": item.target_node,
        "status": "running",
        "started_at": item.started_at,
    }
    queue.write_to_disk(workspace.queue_fp)

    _ensure_pair_endpoints_optimized(pot=pot, item=item, run_inputs=run_inputs)
    pot.write_to_disk(workspace.neb_pot_fp)

    pair = _make_pair_chain(pot=pot, item=item, run_inputs=run_inputs)
    skip_reason = _identical_endpoint_skip_reason(pair, run_inputs)
    if skip_reason:
        item.status = "skipped_identical"
        item.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        item.error = skip_reason
        queue.attempted_pairs[item.attempt_key].update(
            {
                "status": "skipped_identical",
                "finished_at": item.finished_at,
                "error": item.error,
            }
        )
        queue.write_to_disk(workspace.queue_fp)
        return {"message": f"Skipped autosplitting NEB for edge {source_node} -> {target_node}: identical endpoints."}
    result_dir = workspace.queue_output_dir / f"pair_{item.source_node}_{item.target_node}_msmep"
    output_chain_fp = workspace.queue_output_dir / f"pair_{item.source_node}_{item.target_node}.xyz"
    parallel_autosplit = bool(getattr(workspace, "parallel_autosplit_nebs", False))
    parallel_autosplit_workers = max(
        1, int(getattr(workspace, "parallel_autosplit_workers", 4) or 4)
    )

    try:
        result = _run_single_item_worker(
            pair,
            run_inputs,
            str(result_dir),
            str(output_chain_fp),
            parallel_recursive=parallel_autosplit,
            parallel_workers=parallel_autosplit_workers,
        )
    except Exception as exc:
        item.status = "failed"
        item.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        item.error = f"{type(exc).__name__}: {exc}"
        queue.attempted_pairs[item.attempt_key].update(
            {
                "status": "failed",
                "finished_at": item.finished_at,
                "error": item.error,
            }
        )
        queue.write_to_disk(workspace.queue_fp)
        raise

    item.status = "completed"
    item.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    item.result_dir = result["result_dir"]
    item.output_chain_xyz = result["output_chain_xyz"]
    queue.attempted_pairs[item.attempt_key].update(
        {
            "status": "completed",
            "finished_at": item.finished_at,
            "result_dir": item.result_dir,
            "output_chain_xyz": item.output_chain_xyz,
        }
    )
    queue.write_to_disk(workspace.queue_fp)
    if network_splits:
        load_partial_annotated_pot(workspace)
    return {"message": f"Autosplitting NEB completed for edge {source_node} -> {target_node}."}


def _normalize_hawaii_control_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"go", "yellow", "red"}:
        return mode
    return "go"


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(value)


def _hawaii_control_fp_for_workspace(workspace: RetropathsWorkspace | None) -> Path | None:
    if workspace is None:
        return None
    directory = getattr(workspace, "directory", None)
    if directory is None:
        workdir = getattr(workspace, "workdir", None)
        if not workdir:
            return None
        directory = Path(str(workdir))
    return (Path(directory) / "drive_hawaii.control.json").resolve()


def _read_hawaii_control_payload(control_fp: Any) -> dict[str, Any]:
    if not control_fp:
        return {"mode": "go"}
    try:
        payload = json.loads(Path(str(control_fp)).read_text(encoding="utf-8"))
    except Exception:
        return {"mode": "go"}
    payload["mode"] = _normalize_hawaii_control_mode(payload.get("mode"))
    if "discovery_tools" in payload:
        payload["discovery_tools"] = _normalize_hawaii_discovery_tools(
            payload.get("discovery_tools"),
            default=None,
        )
    return payload


def _read_hawaii_control_mode(control_fp: Any) -> str:
    payload = _read_hawaii_control_payload(control_fp)
    return _normalize_hawaii_control_mode(payload.get("mode"))


def _write_hawaii_control_payload(
    control_fp: Any,
    *,
    mode: str,
    source: str,
    note: str = "",
    discovery_tools: Any = None,
) -> dict[str, Any]:
    existing_payload: dict[str, Any] = {}
    if control_fp:
        existing_payload = _read_hawaii_control_payload(control_fp)
    payload = {
        "mode": _normalize_hawaii_control_mode(mode),
        "source": str(source or ""),
        "note": str(note or ""),
        "updated_at": int(time.time()),
    }
    tools_value = discovery_tools if discovery_tools is not None else existing_payload.get("discovery_tools")
    if tools_value is not None:
        payload["discovery_tools"] = _normalize_hawaii_discovery_tools(
            tools_value,
            default=None,
        )
    if not control_fp:
        return payload
    fp = Path(str(control_fp))
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


class _HawaiiImmediateStop(RuntimeError):
    pass


def _raise_if_hawaii_stop_now(control_fp: Any) -> None:
    if _read_hawaii_control_mode(control_fp) == "red":
        raise _HawaiiImmediateStop("Hawaii stoplight set to red.")


def _hawaii_edge_key(source_node: int, target_node: int) -> str:
    source = int(source_node)
    target = int(target_node)
    if source <= target:
        return f"{source}-{target}"
    return f"{target}-{source}"


def _hawaii_singleton_node_key(node_index: int) -> str:
    return f"singleton-node-{int(node_index)}"


def _write_hawaii_progress(
    workspace: RetropathsWorkspace,
    *,
    network_splits: bool,
    pot: Pot | None = None,
    progress_fp: str | None,
    title: str,
    note: str,
    phase: str,
    stage: str = "",
    dr: float | None = None,
    control_mode: str = "go",
    growing_nodes: list[int] | None = None,
    neb_source_node: int | None = None,
    neb_target_node: int | None = None,
    neb_progress_fp: str | None = None,
    neb_chain_fp: str | None = None,
) -> None:
    if not progress_fp:
        return
    with contextlib.suppress(Exception):
        progress_pot = pot if pot is not None else _merge_drive_pot_compat(workspace, network_splits=network_splits)
        growing = {int(node_id) for node_id in (growing_nodes or [])}
        payload = {
            "title": str(title),
            "note": str(note),
            "phase": str(phase),
            "stage": str(stage or ""),
            "dr": (float(dr) if dr is not None else None),
            "control_mode": _normalize_hawaii_control_mode(control_mode),
            "neb": (
                {
                    "source_node": (int(neb_source_node) if neb_source_node is not None else None),
                    "target_node": (int(neb_target_node) if neb_target_node is not None else None),
                    "progress_fp": str(neb_progress_fp or ""),
                    "chain_fp": str(neb_chain_fp or ""),
                }
                if (
                    neb_source_node is not None
                    or neb_target_node is not None
                    or neb_progress_fp
                    or neb_chain_fp
                )
                else {}
            ),
            "network": {
                "nodes": [
                    {
                        "id": int(node_index),
                        "label": _node_label_for_explorer(
                            int(node_index),
                            progress_pot.graph.nodes[int(node_index)],
                        ),
                        "growing": int(node_index) in growing,
                    }
                    for node_index in sorted(int(n) for n in progress_pot.graph.nodes)
                ],
                "edges": [
                    {
                        "source": int(source_node),
                        "target": int(target_node),
                    }
                    for source_node, target_node in sorted(
                        (int(source), int(target))
                        for source, target in progress_pot.graph.edges
                    )
                ],
            },
        }
        progress_path = Path(progress_fp)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _hawaii_node_count(workspace: RetropathsWorkspace, *, network_splits: bool) -> int:
    pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
    return int(pot.graph.number_of_nodes())


def _connect_all_unconnected_nodes(
    workspace: RetropathsWorkspace,
    *,
    network_splits: bool,
    progress_fp: str | None,
    control_fp: str | None = None,
    dr: float | None = None,
) -> int:
    _raise_if_hawaii_stop_now(control_fp)
    pot = materialize_drive_graph(workspace)
    node_ids = sorted(int(node_index) for node_index in pot.graph.nodes)
    missing_edges: list[tuple[int, int]] = []
    for offset, source_node in enumerate(node_ids):
        for target_node in node_ids[offset + 1:]:
            if pot.graph.has_edge(source_node, target_node) or pot.graph.has_edge(target_node, source_node):
                continue
            missing_edges.append((int(source_node), int(target_node)))

    if not missing_edges:
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note="Step 1/3: all current nodes are already connected by at least one edge.",
            phase="growing",
            stage="connect",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[],
        )
        return 0

    for index, (source_node, target_node) in enumerate(missing_edges, start=1):
        _raise_if_hawaii_stop_now(control_fp)
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note=(
                f"Step 1/3: connecting missing edge {source_node} -> {target_node} "
                f"({index}/{len(missing_edges)})."
            ),
            phase="growing",
            stage="connect",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[source_node, target_node],
        )
        add_manual_edge(
            workspace,
            source_node=int(source_node),
            target_node=int(target_node),
            reaction_label=f"Hawaii auto edge {source_node}->{target_node}",
        )
        pot.graph.add_edge(int(source_node), int(target_node))
    return len(missing_edges)


def _run_unattempted_nebs(
    workspace: RetropathsWorkspace,
    *,
    network_splits: bool,
    progress_fp: str | None,
    control_fp: str | None = None,
    dr: float | None = None,
) -> tuple[int, int]:
    _raise_if_hawaii_stop_now(control_fp)
    pot = materialize_drive_graph(workspace)
    queue = build_retropaths_neb_queue(
        pot=pot,
        queue_fp=workspace.queue_fp,
        overwrite=False,
    )
    attempted_keys = set(str(key) for key in queue.attempted_pairs.keys())
    pending_items: list[NEBQueueItem] = []
    seen_pairs: set[tuple[int, int]] = set()
    for item in sorted(queue.items, key=lambda candidate: (int(candidate.source_node), int(candidate.target_node))):
        if str(item.status) != "pending":
            continue
        attempt_key = str(item.attempt_key or "").strip()
        if not attempt_key or attempt_key in attempted_keys:
            continue
        pair_key = tuple(sorted((int(item.source_node), int(item.target_node))))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        pending_items.append(item)

    attempts = 0
    failures = 0
    if not pending_items:
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note="Step 2/3: no unattempted NEB queue items remain.",
            phase="growing",
            stage="neb",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[],
        )
        return attempts, failures

    for index, item in enumerate(pending_items, start=1):
        _raise_if_hawaii_stop_now(control_fp)
        attempts += 1
        source_node = int(item.source_node)
        target_node = int(item.target_node)
        neb_log_fp = str(
            (
                workspace.directory
                / f"drive_hawaii_neb_{source_node}_{target_node}.log"
            ).resolve()
        )
        neb_progress_fp = str(
            (
                workspace.directory
                / f"drive_hawaii_neb_{source_node}_{target_node}.progress.json"
            ).resolve()
        )
        neb_chain_fp = str(
            (
                workspace.directory
                / f"drive_hawaii_neb_{source_node}_{target_node}.chain.json"
            ).resolve()
        )
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note=(
                f"Step 2/3: running autosplitting NEB for edge {source_node} -> {target_node} "
                f"({index}/{len(pending_items)})."
            ),
            phase="growing",
            stage="neb",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[source_node, target_node],
            neb_source_node=source_node,
            neb_target_node=target_node,
            neb_progress_fp=neb_progress_fp,
            neb_chain_fp=neb_chain_fp,
        )
        try:
            _run_selected_edge_neb_logged(
                dict(workspace.__dict__),
                source_node=source_node,
                target_node=target_node,
                network_splits=network_splits,
                log_fp=neb_log_fp,
                progress_fp=neb_progress_fp,
                chain_fp=neb_chain_fp,
            )
        except Exception as exc:
            failures += 1
            _write_hawaii_progress(
                workspace,
                network_splits=network_splits,
                pot=pot,
                progress_fp=progress_fp,
                title="Hawaii autonomous exploration",
                note=(
                    f"Step 2/3: autosplitting NEB for edge {source_node} -> {target_node} "
                    f"failed ({index}/{len(pending_items)}): {type(exc).__name__}: {exc}"
                ),
                phase="growing",
                stage="neb",
                dr=dr,
                control_mode=_read_hawaii_control_mode(control_fp),
                growing_nodes=[source_node, target_node],
            )
            continue
        pot = materialize_drive_graph(workspace)
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note=(
                f"Step 2/3: completed autosplitting NEB for edge {source_node} -> {target_node} "
                f"({index}/{len(pending_items)})."
            ),
            phase="growing",
            stage="neb",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[],
        )
    return attempts, failures


def _has_completed_neb_data_for_pair(
    pot: Pot,
    queue: RetropathsNEBQueue,
    *,
    source_node: int,
    target_node: int,
) -> bool:
    if pot.graph.has_edge(source_node, target_node):
        attrs = dict(pot.graph.edges[(source_node, target_node)])
        if bool(attrs.get("list_of_nebs")):
            return True
    if pot.graph.has_edge(target_node, source_node):
        attrs = dict(pot.graph.edges[(target_node, source_node)])
        if bool(attrs.get("list_of_nebs")):
            return True
    forward_item = queue.find_item(source_node, target_node)
    reverse_item = queue.find_item(target_node, source_node)
    if forward_item is not None and str(forward_item.status) == "completed":
        return True
    if reverse_item is not None and str(reverse_item.status) == "completed":
        return True
    return False


def _run_hessian_on_completed_edges(
    workspace: RetropathsWorkspace,
    *,
    network_splits: bool,
    progress_fp: str | None,
    dr: float,
    max_candidates: int,
    attempted_edge_keys: set[str],
    control_fp: str | None = None,
) -> tuple[int, int, int]:
    _raise_if_hawaii_stop_now(control_fp)
    pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
    queue = build_retropaths_neb_queue(
        pot=pot,
        queue_fp=workspace.queue_fp,
        overwrite=False,
    )

    candidates: list[tuple[int, int, str]] = []
    seen_keys: set[str] = set()
    for source_node, target_node in sorted(
        (int(source), int(target))
        for source, target in pot.graph.edges
    ):
        edge_key = _hawaii_edge_key(source_node, target_node)
        if edge_key in seen_keys:
            continue
        seen_keys.add(edge_key)
        if edge_key in attempted_edge_keys:
            continue
        if not _has_completed_neb_data_for_pair(
            pot,
            queue,
            source_node=source_node,
            target_node=target_node,
        ):
            continue
        candidates.append((source_node, target_node, edge_key))

    attempts = 0
    failures = 0
    added_nodes = 0
    if not candidates:
        singleton_node_index: int | None = None
        if int(pot.graph.number_of_nodes()) == 1:
            with contextlib.suppress(Exception):
                singleton_node_index = int(next(iter(pot.graph.nodes)))
        if singleton_node_index is not None:
            singleton_key = _hawaii_singleton_node_key(singleton_node_index)
            if singleton_key not in attempted_edge_keys:
                _raise_if_hawaii_stop_now(control_fp)
                attempts += 1
                _write_hawaii_progress(
                    workspace,
                    network_splits=network_splits,
                    pot=pot,
                    progress_fp=progress_fp,
                    title="Hawaii autonomous exploration",
                    note=(
                        f"Step 3/3 (dr={float(dr):.1f}): no completed NEB edges are available; "
                        f"running Hessian sample from seed node {int(singleton_node_index)}."
                    ),
                    phase="growing",
                    stage="hessian",
                    dr=dr,
                    control_mode=_read_hawaii_control_mode(control_fp),
                    growing_nodes=[int(singleton_node_index)],
                )
                try:
                    result = run_hessian_sample_for_node(
                        workspace,
                        int(singleton_node_index),
                        dr=float(dr),
                        max_candidates=int(max_candidates),
                        progress_fp=progress_fp,
                    )
                    added_nodes += int(result.get("added_nodes") or 0)
                except Exception as exc:
                    failures += 1
                    _write_hawaii_progress(
                        workspace,
                        network_splits=network_splits,
                        pot=pot,
                        progress_fp=progress_fp,
                        title="Hawaii autonomous exploration",
                        note=(
                            f"Step 3/3 (dr={float(dr):.1f}): Hessian sample for seed node "
                            f"{int(singleton_node_index)} failed: {type(exc).__name__}: {exc}"
                        ),
                        phase="growing",
                        stage="hessian",
                        dr=dr,
                        control_mode=_read_hawaii_control_mode(control_fp),
                        growing_nodes=[int(singleton_node_index)],
                    )
                finally:
                    attempted_edge_keys.add(singleton_key)
                return attempts, failures, added_nodes
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note=(
                f"Step 3/3 (dr={float(dr):.1f}): no untried edges with completed NEB data are available."
            ),
            phase="growing",
            stage="hessian",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[],
        )
        return attempts, failures, added_nodes

    for index, (source_node, target_node, edge_key) in enumerate(candidates, start=1):
        _raise_if_hawaii_stop_now(control_fp)
        attempts += 1
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note=(
                f"Step 3/3 (dr={float(dr):.1f}): Hessian sample for edge "
                f"{source_node} -> {target_node} ({index}/{len(candidates)})."
            ),
            phase="growing",
            stage="hessian",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[source_node, target_node],
        )
        try:
            result = run_hessian_sample_for_edge(
                workspace,
                source_node=source_node,
                target_node=target_node,
                dr=float(dr),
                max_candidates=int(max_candidates),
                progress_fp=progress_fp,
            )
            added_nodes += int(result.get("added_nodes") or 0)
            pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
        except Exception as exc:
            failures += 1
            _write_hawaii_progress(
                workspace,
                network_splits=network_splits,
                pot=pot,
                progress_fp=progress_fp,
                title="Hawaii autonomous exploration",
                note=(
                    f"Step 3/3 (dr={float(dr):.1f}): Hessian sample for edge "
                    f"{source_node} -> {target_node} failed ({index}/{len(candidates)}): "
                    f"{type(exc).__name__}: {exc}"
                ),
                phase="growing",
                stage="hessian",
                dr=dr,
                control_mode=_read_hawaii_control_mode(control_fp),
                growing_nodes=[source_node, target_node],
            )
        finally:
            attempted_edge_keys.add(edge_key)
    return attempts, failures, added_nodes


def _hawaii_node_key(node_index: int) -> str:
    return f"node-{int(node_index)}"


def _run_node_discovery_tool(
    workspace: RetropathsWorkspace,
    *,
    tool_label: str,
    run_for_node: Callable[[int], dict[str, Any]],
    attempted_node_keys: set[str],
    network_splits: bool,
    progress_fp: str | None,
    dr: float | None,
    control_fp: str | None = None,
) -> tuple[int, int, int]:
    _raise_if_hawaii_stop_now(control_fp)
    pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
    candidates: list[tuple[int, str]] = []
    for node_index in sorted(int(node_id) for node_id in pot.graph.nodes):
        node_key = _hawaii_node_key(node_index)
        if node_key in attempted_node_keys:
            continue
        candidates.append((node_index, node_key))

    attempts = 0
    failures = 0
    added_nodes = 0
    if not candidates:
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note=f"Step 3/3: {tool_label} has no untried node seeds remaining.",
            phase="growing",
            stage="discovery",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[],
        )
        return attempts, failures, added_nodes

    for index, (node_index, node_key) in enumerate(candidates, start=1):
        _raise_if_hawaii_stop_now(control_fp)
        attempts += 1
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note=(
                f"Step 3/3: running {tool_label} from node {int(node_index)} "
                f"({index}/{len(candidates)})."
            ),
            phase="growing",
            stage="discovery",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[int(node_index)],
        )
        try:
            result = run_for_node(int(node_index))
            added_nodes += int(result.get("added_nodes") or 0)
            pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
        except Exception as exc:
            failures += 1
            _write_hawaii_progress(
                workspace,
                network_splits=network_splits,
                pot=pot,
                progress_fp=progress_fp,
                title="Hawaii autonomous exploration",
                note=(
                    f"Step 3/3: {tool_label} failed from node {int(node_index)} "
                    f"({index}/{len(candidates)}): {type(exc).__name__}: {exc}"
                ),
                phase="growing",
                stage="discovery",
                dr=dr,
                control_mode=_read_hawaii_control_mode(control_fp),
                growing_nodes=[int(node_index)],
            )
        finally:
            attempted_node_keys.add(node_key)
    return attempts, failures, added_nodes


def _run_discovery_cycle(
    workspace: RetropathsWorkspace,
    *,
    discovery_tools: list[str],
    network_splits: bool,
    progress_fp: str | None,
    dr: float,
    max_hessian_candidates: int,
    attempted_hessian_edge_keys: set[str],
    attempted_node_keys_by_tool: dict[str, set[str]],
    control_fp: str | None = None,
) -> dict[str, int]:
    stats = {
        "attempts": 0,
        "failures": 0,
        "added_nodes": 0,
        "hessian_attempts": 0,
        "hessian_failures": 0,
        "retropaths_attempts": 0,
        "retropaths_failures": 0,
        "nanoreactor_attempts": 0,
        "nanoreactor_failures": 0,
    }
    if not discovery_tools:
        pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            pot=pot,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note="Step 3/3: discovery stage skipped (no discovery tools configured).",
            phase="growing",
            stage="discovery",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[],
        )
        return stats

    for tool_index, tool_name in enumerate(discovery_tools, start=1):
        _raise_if_hawaii_stop_now(control_fp)
        tool_note = (
            f"Step 3/3: discovery tool {tool_index}/{len(discovery_tools)} "
            f"({tool_name}) is running."
        )
        _write_hawaii_progress(
            workspace,
            network_splits=network_splits,
            progress_fp=progress_fp,
            title="Hawaii autonomous exploration",
            note=tool_note,
            phase="growing",
            stage="discovery",
            dr=dr,
            control_mode=_read_hawaii_control_mode(control_fp),
            growing_nodes=[],
        )

        if tool_name == "hessian-sample":
            attempts, failures, added_nodes = _run_hessian_on_completed_edges(
                workspace,
                network_splits=network_splits,
                progress_fp=progress_fp,
                dr=float(dr),
                max_candidates=int(max_hessian_candidates),
                attempted_edge_keys=attempted_hessian_edge_keys,
                control_fp=control_fp,
            )
            stats["hessian_attempts"] += int(attempts)
            stats["hessian_failures"] += int(failures)
        elif tool_name == "retropaths":
            try:
                ensure_retropaths_available(feature="Hawaii Retropaths discovery")
            except Exception as exc:
                attempts, failures, added_nodes = 0, 1, 0
                _write_hawaii_progress(
                    workspace,
                    network_splits=network_splits,
                    progress_fp=progress_fp,
                    title="Hawaii autonomous exploration",
                    note=(
                        "Step 3/3: Retropaths discovery is unavailable and was skipped: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    phase="growing",
                    stage="discovery",
                    dr=dr,
                    control_mode=_read_hawaii_control_mode(control_fp),
                    growing_nodes=[],
                )
            else:
                attempts, failures, added_nodes = _run_node_discovery_tool(
                    workspace,
                    tool_label="Retropaths template growth",
                    run_for_node=lambda node_id: apply_reactions_to_node(
                        workspace,
                        int(node_id),
                        progress_fp=progress_fp,
                    ),
                    attempted_node_keys=attempted_node_keys_by_tool.setdefault("retropaths", set()),
                    network_splits=network_splits,
                    progress_fp=progress_fp,
                    dr=dr,
                    control_fp=control_fp,
                )
            stats["retropaths_attempts"] += int(attempts)
            stats["retropaths_failures"] += int(failures)
        elif tool_name == "nanoreactor":
            attempts, failures, added_nodes = _run_node_discovery_tool(
                workspace,
                tool_label="Nanoreactor sampling",
                run_for_node=lambda node_id: run_nanoreactor_for_node(
                    workspace,
                    int(node_id),
                    progress_fp=progress_fp,
                ),
                attempted_node_keys=attempted_node_keys_by_tool.setdefault("nanoreactor", set()),
                network_splits=network_splits,
                progress_fp=progress_fp,
                dr=dr,
                control_fp=control_fp,
            )
            stats["nanoreactor_attempts"] += int(attempts)
            stats["nanoreactor_failures"] += int(failures)
        else:
            # This should be unreachable because discovery tools are validated.
            attempts, failures, added_nodes = 0, 1, 0
            _write_hawaii_progress(
                workspace,
                network_splits=network_splits,
                progress_fp=progress_fp,
                title="Hawaii autonomous exploration",
                note=f"Step 3/3: unsupported discovery tool `{tool_name}` was skipped.",
                phase="growing",
                stage="discovery",
                dr=dr,
                control_mode=_read_hawaii_control_mode(control_fp),
                growing_nodes=[],
            )

        stats["attempts"] += int(attempts)
        stats["failures"] += int(failures)
        stats["added_nodes"] += int(added_nodes)
    return stats


def _run_hawaii_autonomy(
    workspace_data: dict[str, Any],
    *,
    network_splits: bool = True,
    progress_fp: str | None = None,
    control_fp: str | None = None,
    max_hessian_candidates: int = 100,
    discovery_tools: list[str] | None = None,
) -> dict[str, Any]:
    workspace = RetropathsWorkspace(**workspace_data)
    configured_tools = _normalize_hawaii_discovery_tools(
        discovery_tools if discovery_tools is not None else _read_hawaii_control_payload(control_fp).get("discovery_tools"),
        default=_HAWAII_DISCOVERY_TOOL_DEFAULT,
    )
    current_pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
    hessian_enabled = "hessian-sample" in configured_tools
    dr_schedule = (1.0, 2.0, 3.0) if hessian_enabled else (1.0,)
    attempted_hessian_by_dr: dict[str, set[str]] = {
        f"{float(dr):.1f}": set()
        for dr in dr_schedule
    }
    attempted_node_keys_by_tool: dict[str, set[str]] = {
        "retropaths": set(),
        "nanoreactor": set(),
    }
    cycle_index = 0
    completion_reason = ""
    total_connected_edges = 0
    total_neb_attempts = 0
    total_neb_failures = 0
    total_discovery_attempts = 0
    total_discovery_failures = 0
    total_hessian_attempts = 0
    total_hessian_failures = 0
    total_retropaths_attempts = 0
    total_retropaths_failures = 0
    total_nanoreactor_attempts = 0
    total_nanoreactor_failures = 0
    total_new_minima = 0
    discovery_label = ", ".join(configured_tools) if configured_tools else "none"
    current_dr = float(dr_schedule[0]) if dr_schedule else None

    _write_hawaii_progress(
        workspace,
        network_splits=network_splits,
        pot=current_pot,
        progress_fp=progress_fp,
        title="Hawaii autonomous exploration",
        note=(
            "Starting autonomous connect/refinement/discovery exploration "
            f"(discovery tools: {discovery_label})."
        ),
        phase="growing",
        stage="connect",
        dr=current_dr,
        control_mode=_read_hawaii_control_mode(control_fp),
        growing_nodes=[],
    )

    dr_index = 0
    stopped_immediately = False
    try:
        while dr_index < len(dr_schedule):
            _raise_if_hawaii_stop_now(control_fp)
            cycle_index += 1
            dr = float(dr_schedule[dr_index])
            current_dr = float(dr)
            current_pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
            before_nodes = int(current_pot.graph.number_of_nodes())
            _write_hawaii_progress(
                workspace,
                network_splits=network_splits,
                pot=current_pot,
                progress_fp=progress_fp,
                title="Hawaii autonomous exploration",
                note=(
                    f"Cycle {cycle_index}: running autonomous connect/refinement/discovery cycle "
                    f"(tools: {discovery_label}, dr={float(dr):.1f})."
                ),
                phase="growing",
                stage="connect",
                dr=dr,
                control_mode=_read_hawaii_control_mode(control_fp),
                growing_nodes=[],
            )

            connected_edges = _connect_all_unconnected_nodes(
                workspace,
                network_splits=network_splits,
                progress_fp=progress_fp,
                control_fp=control_fp,
                dr=dr,
            )
            total_connected_edges += int(connected_edges)
            if _read_hawaii_control_mode(control_fp) == "yellow":
                break

            neb_attempts, neb_failures = _run_unattempted_nebs(
                workspace,
                network_splits=network_splits,
                progress_fp=progress_fp,
                control_fp=control_fp,
                dr=dr,
            )
            total_neb_attempts += int(neb_attempts)
            total_neb_failures += int(neb_failures)
            if _read_hawaii_control_mode(control_fp) == "yellow":
                break

            discovery_stats = _run_discovery_cycle(
                workspace,
                discovery_tools=list(configured_tools),
                network_splits=network_splits,
                progress_fp=progress_fp,
                dr=float(dr),
                max_hessian_candidates=int(max_hessian_candidates),
                attempted_hessian_edge_keys=attempted_hessian_by_dr[f"{float(dr):.1f}"],
                attempted_node_keys_by_tool=attempted_node_keys_by_tool,
                control_fp=control_fp,
            )
            total_discovery_attempts += int(discovery_stats["attempts"])
            total_discovery_failures += int(discovery_stats["failures"])
            total_hessian_attempts += int(discovery_stats["hessian_attempts"])
            total_hessian_failures += int(discovery_stats["hessian_failures"])
            total_retropaths_attempts += int(discovery_stats["retropaths_attempts"])
            total_retropaths_failures += int(discovery_stats["retropaths_failures"])
            total_nanoreactor_attempts += int(discovery_stats["nanoreactor_attempts"])
            total_nanoreactor_failures += int(discovery_stats["nanoreactor_failures"])
            if _read_hawaii_control_mode(control_fp) == "yellow":
                break

            current_pot = _merge_drive_pot_compat(workspace, network_splits=network_splits)
            after_nodes = int(current_pot.graph.number_of_nodes())
            discovered_minima = max(0, int(after_nodes - before_nodes))
            if discovered_minima <= 0:
                discovered_minima = int(discovery_stats["added_nodes"])

            if discovered_minima > 0:
                total_new_minima += int(discovered_minima)
                _write_hawaii_progress(
                    workspace,
                    network_splits=network_splits,
                    pot=current_pot,
                    progress_fp=progress_fp,
                    title="Hawaii autonomous exploration",
                    note=(
                        f"Cycle {cycle_index}, dr={float(dr):.1f}: discovered {int(discovered_minima)} "
                        "new minima. Restarting from step 1 (connect stage)."
                    ),
                    phase="growing",
                    stage="connect",
                    dr=dr,
                    control_mode=_read_hawaii_control_mode(control_fp),
                    growing_nodes=[],
                )
                continue

            if not configured_tools:
                if connected_edges <= 0 and neb_attempts <= 0:
                    completion_reason = (
                        "Autonomous loop finished after connect/refinement reached a steady state "
                        "with discovery tools disabled."
                    )
                    _write_hawaii_progress(
                        workspace,
                        network_splits=network_splits,
                        pot=current_pot,
                        progress_fp=progress_fp,
                        title="Hawaii autonomous exploration",
                        note=completion_reason,
                        phase="growing",
                        stage="connect",
                        dr=dr,
                        control_mode=_read_hawaii_control_mode(control_fp),
                        growing_nodes=[],
                    )
                    break
                _write_hawaii_progress(
                    workspace,
                    network_splits=network_splits,
                    pot=current_pot,
                    progress_fp=progress_fp,
                    title="Hawaii autonomous exploration",
                    note=(
                        f"Cycle {cycle_index}: no discovery tools configured; "
                        "continuing connect/refinement stages until stable."
                    ),
                    phase="growing",
                    stage="connect",
                    dr=dr,
                    control_mode=_read_hawaii_control_mode(control_fp),
                    growing_nodes=[],
                )
                continue

            if not hessian_enabled:
                completion_reason = (
                    "Autonomous loop finished because discovery tools without Hessian sampling "
                    "found no new minima this cycle."
                )
                _write_hawaii_progress(
                    workspace,
                    network_splits=network_splits,
                    pot=current_pot,
                    progress_fp=progress_fp,
                    title="Hawaii autonomous exploration",
                    note=completion_reason,
                    phase="growing",
                    stage="connect",
                    dr=dr,
                    control_mode=_read_hawaii_control_mode(control_fp),
                    growing_nodes=[],
                )
                break

            _write_hawaii_progress(
                workspace,
                network_splits=network_splits,
                pot=current_pot,
                progress_fp=progress_fp,
                title="Hawaii autonomous exploration",
                note=(
                    f"Cycle {cycle_index}, dr={float(dr):.1f}: no new minima found; "
                    "escalating displacement if available."
                ),
                phase="growing",
                stage="connect",
                dr=dr,
                control_mode=_read_hawaii_control_mode(control_fp),
                growing_nodes=[],
            )
            dr_index += 1
    except _HawaiiImmediateStop:
        stopped_immediately = True

    final_mode = _read_hawaii_control_mode(control_fp)
    if stopped_immediately or final_mode == "red":
        final_note = "Hawaii stopped immediately (red stoplight)."
        phase = "finished"
    elif final_mode == "yellow":
        final_note = "Hawaii paused after finishing the current stage (yellow stoplight)."
        phase = "finished"
    elif completion_reason:
        final_note = completion_reason
        phase = "finished"
    elif dr_index >= len(dr_schedule):
        final_note = "Autonomous loop finished because no new minima were found at dr=1.0, 2.0, or 3.0."
        phase = "finished"
    else:
        final_note = "Hawaii autonomous exploration stopped."
        phase = "finished"

    _write_hawaii_progress(
        workspace,
        network_splits=network_splits,
        pot=_merge_drive_pot_compat(workspace, network_splits=network_splits),
        progress_fp=progress_fp,
        title="Hawaii autonomous exploration",
        note=final_note,
        phase=phase,
        stage="finished",
        dr=current_dr,
        control_mode=final_mode,
        growing_nodes=[],
    )
    return {
        "message": (
            f"{final_note} "
            f"Connected edges: {total_connected_edges}. "
            f"NEB attempts: {total_neb_attempts} (failures: {total_neb_failures}). "
            f"Discovery attempts: {total_discovery_attempts} (failures: {total_discovery_failures}). "
            f"Hessian attempts: {total_hessian_attempts} (failures: {total_hessian_failures}). "
            f"Retropaths attempts: {total_retropaths_attempts} (failures: {total_retropaths_failures}). "
            f"Nanoreactor attempts: {total_nanoreactor_attempts} (failures: {total_nanoreactor_failures}). "
            f"New minima discovered: {total_new_minima}."
        ),
        "cycles": int(cycle_index),
        "discovery_tools": list(configured_tools),
        "connected_edges": int(total_connected_edges),
        "neb_attempts": int(total_neb_attempts),
        "neb_failures": int(total_neb_failures),
        "discovery_attempts": int(total_discovery_attempts),
        "discovery_failures": int(total_discovery_failures),
        "hessian_attempts": int(total_hessian_attempts),
        "hessian_failures": int(total_hessian_failures),
        "retropaths_attempts": int(total_retropaths_attempts),
        "retropaths_failures": int(total_retropaths_failures),
        "nanoreactor_attempts": int(total_nanoreactor_attempts),
        "nanoreactor_failures": int(total_nanoreactor_failures),
        "new_minima": int(total_new_minima),
    }


def _run_selected_edge_neb_logged(
    workspace_data: dict[str, Any],
    *,
    source_node: int,
    target_node: int,
    network_splits: bool = True,
    log_fp: str,
    progress_fp: str,
    chain_fp: str,
) -> dict[str, Any]:
    workspace = RetropathsWorkspace(**workspace_data)
    log_path = Path(log_fp)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chain_path = Path(chain_fp)
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    old_log = os.environ.get("MEPD_DRIVE_PROGRESS_LOG")
    old_chain = os.environ.get("MEPD_DRIVE_CHAIN_JSON")
    os.environ["MEPD_DRIVE_PROGRESS_LOG"] = str(progress_fp)
    os.environ["MEPD_DRIVE_CHAIN_JSON"] = str(chain_path)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        try:
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                return _run_selected_edge_neb(
                    workspace,
                    source_node=source_node,
                    target_node=target_node,
                    network_splits=network_splits,
                )
        finally:
            if old_log is None:
                os.environ.pop("MEPD_DRIVE_PROGRESS_LOG", None)
            else:
                os.environ["MEPD_DRIVE_PROGRESS_LOG"] = old_log
            if old_chain is None:
                os.environ.pop("MEPD_DRIVE_CHAIN_JSON", None)
            else:
                os.environ["MEPD_DRIVE_CHAIN_JSON"] = old_chain


def _extract_network_splits_run_name(network_fp: Path) -> str:
    suffix = "_network.json"
    if network_fp.name.endswith(suffix):
        return network_fp.name[: -len(suffix)] or network_fp.stem
    return network_fp.stem


def _resolve_network_splits_pot_fp(path: Path) -> Path | None:
    candidate_path = Path(path).expanduser().resolve()
    if candidate_path.is_file():
        if candidate_path.name.endswith("_network.json"):
            return candidate_path
        if candidate_path.name.endswith("_request_manifest.json"):
            base_name = candidate_path.name[: -len("_request_manifest.json")]
            paired = candidate_path.parent / f"{base_name}_network.json"
            if paired.exists():
                return paired
        return None
    if not candidate_path.is_dir():
        return None

    candidates = sorted(
        fp for fp in candidate_path.glob("*_network.json") if fp.is_file()
    )
    if not candidates:
        return None

    suffix = "_network_splits"
    if candidate_path.name.endswith(suffix):
        base_name = candidate_path.name[: -len(suffix)]
        preferred = candidate_path / f"{base_name}_network.json"
        if preferred.exists():
            return preferred

    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda fp: fp.stat().st_mtime_ns)


def _guess_inputs_fp_for_network_splits(workspace_dir: Path, run_name: str) -> Path:
    candidates = [
        workspace_dir / f"{run_name}.toml",
        workspace_dir.parent / f"{run_name}.toml",
        workspace_dir / "inputs.toml",
        workspace_dir.parent / "inputs.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _materialize_workspace_from_network_splits(path: Path) -> RetropathsWorkspace | None:
    network_fp = _resolve_network_splits_pot_fp(path)
    if network_fp is None:
        return None

    workspace_dir = network_fp.parent
    workspace_json_fp = workspace_dir / "workspace.json"
    if workspace_json_fp.exists():
        return RetropathsWorkspace.read(workspace_dir)

    pot = Pot.read_from_disk(network_fp)
    run_name = _extract_network_splits_run_name(network_fp)
    root_smiles = ""
    with contextlib.suppress(Exception):
        root_smiles = _quiet_force_smiles(getattr(pot, "root", None))
    if not root_smiles and 0 in pot.graph.nodes:
        molecule = pot.graph.nodes[0].get("molecule")
        if isinstance(molecule, str):
            root_smiles = molecule
        elif molecule is not None:
            with contextlib.suppress(Exception):
                root_smiles = _quiet_force_smiles(molecule)
    if not root_smiles:
        root_smiles = run_name

    workspace = RetropathsWorkspace(
        workdir=str(workspace_dir),
        run_name=run_name,
        root_smiles=root_smiles,
        environment_smiles="",
        inputs_fp=str(_guess_inputs_fp_for_network_splits(workspace_dir, run_name)),
        reactions_fp="",
        timeout_seconds=30,
        max_nodes=40,
        max_depth=4,
        max_parallel_nebs=1,
        parallel_autosplit_nebs=False,
        parallel_autosplit_workers=4,
    )
    workspace.write()

    if workspace.neb_pot_fp.resolve() != network_fp.resolve():
        shutil.copy2(network_fp, workspace.neb_pot_fp)
        pot = Pot.read_from_disk(workspace.neb_pot_fp)

    build_retropaths_neb_queue(
        pot=pot,
        queue_fp=workspace.queue_fp,
        overwrite=False,
    )

    if not workspace.retropaths_pot_fp.exists():
        workspace.retropaths_pot_fp.write_text("{}", encoding="utf-8")

    return workspace


def _load_existing_workspace_job(
    workspace_path: str,
    *,
    network_splits: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    total_steps = 6 if network_splits else 4

    def _emit_progress(message: str, completed_steps: float) -> None:
        if progress is None:
            return
        with contextlib.suppress(Exception):
            progress(
                {
                    "message": message,
                    "completed_steps": float(max(0.0, min(float(total_steps), completed_steps))),
                    "total_steps": float(total_steps),
                }
            )

    _emit_progress("Resolving workspace path...", 0)
    path = Path(workspace_path).expanduser().resolve()
    workspace_dir = path.parent if path.is_file() else path
    resolved_workspace_dir = workspace_dir.resolve()
    workspace_json_fp = workspace_dir / "workspace.json"
    if workspace_json_fp.exists():
        workspace = RetropathsWorkspace.read(workspace_dir)
        configured_workdir = Path(str(workspace.workdir)).expanduser().resolve()
        if configured_workdir != resolved_workspace_dir:
            raise ValueError(
                "workspace.json `workdir` does not match the selected workspace directory."
            )
    else:
        workspace = _materialize_workspace_from_network_splits(path)
        if workspace is None:
            raise ValueError(
                "Workspace path must point to a drive workspace directory, a workspace.json file, "
                "a *_network_splits directory, or a *_network.json artifact."
            )
    _emit_progress("Workspace metadata loaded.", 1)
    required = [
        workspace.workspace_fp,
        workspace.neb_pot_fp,
        workspace.queue_fp,
        workspace.retropaths_pot_fp,
    ]
    missing = [str(fp) for fp in required if not fp.exists()]
    if missing:
        raise ValueError(f"Workspace is missing required files: {', '.join(missing)}")
    _emit_progress("Verified required workspace files.", 2)
    with contextlib.suppress(Exception):
        charge, multiplicity = _workspace_charge_multiplicity(workspace)
        queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)
        if queue.recover_stale_running_items(
            output_dir=workspace.queue_output_dir,
            charge=charge,
            multiplicity=multiplicity,
        ):
            queue.write_to_disk(workspace.queue_fp)
    _emit_progress("Queue state checked.", 3)
    # Rebuild the annotated overlay immediately so completed NEB results
    # are visible as soon as the workspace is opened in drive.
    if network_splits:
        _emit_progress("Rebuilding annotated network overlay...", 4)

        def _overlay_progress(payload: dict[str, Any]) -> None:
            if progress is None:
                return
            total_items = int(payload.get("total_items", 0) or 0)
            current_item = int(payload.get("current_item", 0) or 0)
            message = str(payload.get("message") or "Rebuilding annotated network overlay...")
            if total_items > 0:
                fraction = max(0.0, min(1.0, float(current_item) / float(total_items)))
                completed = 4.0 + fraction
            else:
                completed = 4.0
            _emit_progress(message, completed)

        _load_partial_annotated_pot_compat(workspace, progress=_overlay_progress)
        _emit_progress("Annotated network overlay rebuilt.", 5)
    _emit_progress("Finalizing workspace snapshot...", total_steps - 1)
    result = _workspace_snapshot_payload(
        workspace,
        message=f"Loaded existing workspace {workspace.run_name}.",
    )
    return result


def _drive_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MEPD Drive</title>
  <style>
    :root {
      --bg: #08111f;
      --bg-elevated: #0d1a2c;
      --panel: rgba(13, 24, 41, 0.9);
      --panel-strong: rgba(16, 31, 53, 0.96);
      --panel-soft: rgba(12, 23, 39, 0.76);
      --line: rgba(114, 146, 197, 0.2);
      --line-strong: rgba(120, 170, 233, 0.38);
      --ink: #eef4ff;
      --ink-soft: #c9d8f0;
      --muted: #8ea3c2;
      --accent: #63d5ff;
      --accent-strong: #1ca4dd;
      --accent-2: #7ef0c7;
      --backed: #59d8b6;
      --target: #ff8eb0;
      --warn: #ff7a8f;
      --shadow: 0 18px 44px rgba(2, 8, 18, 0.45);
      --radius-lg: 22px;
      --radius-md: 16px;
      --radius-sm: 12px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 500 14px/1.5 "IBM Plex Sans", "Aptos", "Segoe UI Variable", "Avenir Next", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(93, 213, 255, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(126, 240, 199, 0.08), transparent 24%),
        linear-gradient(180deg, #0a1425 0%, #08111f 100%);
      color: var(--ink);
    }
    .shell { max-width: 1680px; margin: 0 auto; padding: 22px; display: grid; gap: 18px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .page-title { padding: 24px; background: linear-gradient(180deg, rgba(17, 32, 54, 0.94), rgba(11, 22, 38, 0.9)); }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
      padding: 6px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(99, 213, 255, 0.08);
      color: var(--ink-soft);
      font-size: 11px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .page-title h1 { margin: 0 0 8px; font-size: 30px; line-height: 1.1; letter-spacing: -0.03em; }
    .section-head { display: flex; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
    .section-head h2 { margin: 0; font-size: 18px; letter-spacing: -0.02em; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 10px; }
    .stat {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: 12px 13px;
      background: linear-gradient(180deg, rgba(17, 30, 49, 0.92), rgba(10, 21, 36, 0.84));
    }
    .muted { color: var(--muted); }
    textarea, input, select, button { font: inherit; }
    label {
      display: block;
      margin-bottom: 7px;
      color: var(--ink-soft);
      font-size: 12px;
      letter-spacing: 0.02em;
    }
    textarea, input[type="text"], input[type="number"], select {
      width: 100%;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: rgba(8, 16, 28, 0.74);
      color: var(--ink);
      transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    textarea::placeholder, input::placeholder { color: rgba(142, 163, 194, 0.7); }
    textarea:focus, input:focus, select:focus, button:focus {
      outline: none;
      border-color: var(--line-strong);
      box-shadow: 0 0 0 3px rgba(99, 213, 255, 0.12);
    }
    textarea { min-height: 110px; resize: vertical; }
    button {
      min-height: 40px;
      padding: 10px 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--ink);
      cursor: pointer;
      transition: transform 0.16s ease, background 0.16s ease, border-color 0.16s ease, color 0.16s ease;
    }
    button:hover:not(:disabled) { transform: translateY(-1px); border-color: var(--line-strong); background: rgba(99, 213, 255, 0.08); }
    button.primary {
      background: linear-gradient(180deg, rgba(99, 213, 255, 0.22), rgba(28, 164, 221, 0.16));
      color: var(--ink);
      border-color: rgba(99, 213, 255, 0.38);
    }
    button.secondary { background: rgba(126, 240, 199, 0.06); border-color: rgba(126, 240, 199, 0.22); }
    button:disabled { opacity: 0.55; cursor: default; }
    .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .summary-strip { display: grid; grid-template-columns: minmax(300px, 0.78fr) minmax(0, 1.92fr); gap: 16px; align-items: start; margin-bottom: 14px; }
    .tool-tabs, .detail-tabs { display: flex; gap: 8px; margin: 10px 0 14px; flex-wrap: wrap; }
    .tool-tab, .detail-tab {
      min-height: 36px;
      padding: 7px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--muted);
      cursor: pointer;
    }
    .tool-tab.active, .detail-tab.active {
      background: rgba(99, 213, 255, 0.12);
      color: var(--ink);
      border-color: rgba(99, 213, 255, 0.34);
    }
    .tool-panel, .detail-panel { display: none; }
    .tool-panel.active, .detail-panel.active { display: block; }
    .inputs-grid { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.86fr); gap: 16px; }
    .network-workspace-grid { display: grid; grid-template-columns: minmax(0, 1.9fr) minmax(320px, 0.95fr); gap: 16px; align-items: start; }
    .network-left-stack { display: grid; gap: 16px; }
    .network-canvas-shell { position: relative; }
    .path-browser {
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, rgba(16, 30, 50, 0.92), rgba(10, 20, 34, 0.84));
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .exploration-shell, .kinetics-shell {
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, rgba(16, 30, 50, 0.92), rgba(10, 20, 34, 0.84));
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .kinetics-shell { margin-top: 16px; }
    .json-textarea {
      min-height: 160px;
      font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre;
      tab-size: 2;
    }
    .json-input-controls {
      display: flex;
      gap: 8px;
      justify-content: flex-end;
      margin-top: 8px;
    }
    .json-input-controls button { min-height: 32px; padding: 6px 10px; }
    .path-browser-head { display: flex; align-items: start; justify-content: space-between; gap: 10px; margin-bottom: 12px; }
    .path-browser-head h3 { margin: 0; font-size: 17px; }
    .path-browser-controls { display: grid; gap: 10px; margin-bottom: 12px; }
    .product-list { display: grid; gap: 8px; max-height: 540px; overflow: auto; padding-right: 2px; }
    .product-row {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: rgba(255, 255, 255, 0.03);
      padding: 10px 12px;
      text-align: left;
    }
    .product-row.active {
      border-color: rgba(99, 213, 255, 0.34);
      background: rgba(99, 213, 255, 0.1);
    }
    .product-row.unreachable { opacity: 0.6; }
    .product-row-title { display: block; font-weight: 600; color: var(--ink); word-break: break-word; }
    .product-row-meta { display: block; margin-top: 4px; color: var(--muted); font-size: 12px; }
    .network-toolbar {
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 3;
      min-width: 272px;
      max-width: min(300px, calc(100% - 32px));
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: rgba(7, 15, 27, 0.82);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .network-toolbar-title { font-weight: 600; margin-bottom: 6px; }
    .network-toolbar-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .network-tool-button {
      min-width: 44px;
      min-height: 44px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.04);
      font-size: 20px;
      line-height: 1;
    }
    .network-tool-button.active { background: rgba(126, 240, 199, 0.12); color: var(--ink); border-color: rgba(126, 240, 199, 0.34); }
    .explorer-svg {
      width: 100%;
      min-height: 700px;
      border: 1px solid var(--line);
      border-radius: calc(var(--radius-lg) - 4px);
      cursor: grab;
      touch-action: none;
      background:
        radial-gradient(circle at 20% 18%, rgba(99, 213, 255, 0.11), transparent 20%),
        radial-gradient(circle at 82% 12%, rgba(126, 240, 199, 0.08), transparent 18%),
        linear-gradient(180deg, #0d1728 0%, #08111f 100%);
    }
    .explorer-svg.is-panning { cursor: grabbing; }
    .network-canvas-hint {
      margin-top: 8px;
      color: rgba(178, 198, 226, 0.82);
      font-size: 12px;
    }
    .network-context-menu {
      position: absolute;
      display: none;
      flex-direction: column;
      gap: 6px;
      min-width: 220px;
      max-width: min(260px, calc(100% - 24px));
      padding: 9px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(7, 15, 27, 0.97);
      box-shadow: 0 12px 28px rgba(2, 7, 14, 0.58);
      backdrop-filter: blur(12px);
      z-index: 6;
    }
    .network-context-menu.visible { display: flex; }
    .network-context-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--ink);
      margin-bottom: 2px;
    }
    .network-context-menu button {
      border: 1px solid var(--line);
      border-radius: 9px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--ink-soft);
      padding: 7px 9px;
      font-size: 12px;
      text-align: left;
      cursor: pointer;
    }
    .network-context-menu button:hover {
      background: rgba(99, 213, 255, 0.14);
      color: var(--ink);
      border-color: rgba(99, 213, 255, 0.34);
    }
    .network-context-menu button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .network-edge-line {
      stroke: rgba(128, 154, 194, 0.42);
      stroke-width: 2.1;
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      pointer-events: none;
    }
    .network-edge-line.neb-backed { stroke: var(--backed); stroke-width: 3.4; }
    .network-edge-line.path-highlight { stroke: #ffd166; stroke-width: 4.6; }
    .network-edge-line.selected { stroke: var(--accent); stroke-width: 4.4; }
    .network-edge-line.pending-add {
      stroke: #ffd166;
      stroke-width: 3.2;
      stroke-dasharray: 7 7;
      stroke-linecap: round;
      opacity: 0.95;
      animation: pending-edge-dash 0.95s linear infinite;
    }
    .network-edge-hitbox {
      stroke: transparent;
      stroke-width: 14;
      stroke-linecap: round;
      stroke-linejoin: round;
      fill: none;
      cursor: pointer;
      pointer-events: stroke;
    }
    .network-node {
      fill: #7d94bb;
      stroke: rgba(238, 244, 255, 0.85);
      stroke-width: 1.8;
      cursor: pointer;
      filter: drop-shadow(0 8px 14px rgba(4, 9, 18, 0.3));
    }
    .network-node.root { fill: var(--accent); }
    .network-node.neb-backed { fill: var(--backed); }
    .network-node.target { fill: var(--target); }
    .network-node.path-highlight { stroke: #fff7de; stroke-width: 3.2; }
    .network-node.path-source { fill: var(--accent-2); }
    .network-node.path-target { fill: #ffd166; stroke: #fff7de; stroke-width: 3.2; }
    .network-node.selected { fill: #ffd166; stroke: #fff7de; stroke-width: 3; }
    .network-node.connect-source { fill: var(--accent-2); stroke: rgba(238, 244, 255, 0.92); stroke-width: 3.2; }
    .network-label { font-size: 11px; fill: rgba(233, 241, 255, 0.94); pointer-events: none; }
    @keyframes pending-edge-dash {
      from { stroke-dashoffset: 28; }
      to { stroke-dashoffset: 0; }
    }
    @keyframes flask-turn {
      0% { transform: rotate(0deg); }
      20% { transform: rotate(-18deg); }
      52% { transform: rotate(12deg); }
      82% { transform: rotate(-8deg); }
      100% { transform: rotate(0deg); }
    }
    @keyframes flask-bubble {
      0%, 100% { opacity: 0.26; transform: translateY(0); }
      50% { opacity: 1; transform: translateY(-2px); }
    }
    .viewer-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    iframe.structure {
      width: 100%;
      height: 320px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #101b2b 0%, #0a1321 100%);
    }
    .mol-card {
      width: 100%;
      min-height: 320px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, rgba(16, 30, 50, 0.98), rgba(10, 20, 33, 0.94));
      display: grid;
      align-content: start;
      overflow: hidden;
    }
    .mol-card svg { width: 100%; height: auto; display: block; }
    .mol-empty { padding: 18px 14px; color: var(--muted); }
    .mol-meta { padding: 10px 12px 12px; border-top: 1px solid var(--line); font-size: 12px; color: var(--muted); word-break: break-word; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(4, 10, 18, 0.72);
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: 12px;
      color: var(--ink-soft);
    }
    .badge {
      display: inline-block;
      padding: 4px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(99, 213, 255, 0.08);
      margin-right: 6px;
      margin-bottom: 6px;
    }
    .message {
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, rgba(14, 26, 44, 0.95), rgba(10, 20, 34, 0.9));
    }
    .activity-indicator {
      display: none;
      align-items: center;
      gap: 12px;
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px dashed rgba(126, 240, 199, 0.4);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, rgba(15, 30, 50, 0.95), rgba(10, 20, 34, 0.9));
    }
    .activity-indicator.visible { display: flex; }
    .activity-flask {
      width: 34px;
      height: 34px;
      border-radius: 999px;
      border: 1px solid rgba(126, 240, 199, 0.42);
      background: rgba(126, 240, 199, 0.1);
      display: grid;
      place-items: center;
      flex: 0 0 auto;
    }
    .activity-flask svg {
      width: 20px;
      height: 20px;
      transform-origin: 50% 78%;
      animation: flask-turn 1.2s ease-in-out infinite;
    }
    .activity-flask .bubble-a { animation: flask-bubble 1.1s ease-in-out infinite; }
    .activity-flask .bubble-b { animation: flask-bubble 1.1s ease-in-out 0.36s infinite; }
    .activity-title { font-weight: 600; color: var(--ink); }
    .activity-detail { font-size: 12px; color: var(--muted); }
    .activity-elapsed {
      margin-top: 2px;
      font-size: 11px;
      color: rgba(201, 216, 240, 0.8);
      letter-spacing: 0.02em;
    }
    .stoplight-shell {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, rgba(14, 26, 44, 0.95), rgba(10, 20, 34, 0.9));
      padding: 12px;
    }
    .stoplight-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin-bottom: 8px; }
    .stoplight-row { display: flex; gap: 8px; flex-wrap: wrap; }
    .stoplight-button {
      min-width: 110px;
      border-radius: 999px;
      font-weight: 600;
    }
    .stoplight-button.go { border-color: rgba(126, 240, 199, 0.42); background: rgba(126, 240, 199, 0.1); }
    .stoplight-button.yellow { border-color: rgba(255, 209, 102, 0.46); background: rgba(255, 209, 102, 0.1); }
    .stoplight-button.red { border-color: rgba(255, 122, 143, 0.5); background: rgba(255, 122, 143, 0.12); }
    .stoplight-button.active { box-shadow: inset 0 0 0 1px rgba(238, 244, 255, 0.4); }
    .stoplight-options {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }
    .stoplight-options-title {
      font-size: 12px;
      letter-spacing: 0.02em;
      color: var(--ink-soft);
      margin-bottom: 7px;
    }
    .stoplight-options-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    .stoplight-option {
      margin: 0;
      display: inline-flex;
      gap: 7px;
      align-items: center;
      font-size: 12px;
      color: var(--ink-soft);
      letter-spacing: 0.01em;
      cursor: pointer;
    }
    .stoplight-option input[type="checkbox"] {
      width: auto;
      margin: 0;
      padding: 0;
      cursor: pointer;
      accent-color: #63d5ff;
    }
    .stoplight-summary {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .kv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
    .kv-card {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, rgba(16, 30, 50, 0.92), rgba(10, 20, 34, 0.84));
      padding: 10px;
    }
    .code-block {
      background: rgba(4, 10, 18, 0.72);
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: 12px;
      font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
      font-size: 12px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--ink-soft);
    }
    .live-activity {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, rgba(16, 30, 50, 0.92), rgba(10, 20, 34, 0.84));
      padding: 12px;
    }
    .live-activity svg { width: 100%; height: auto; border: 1px solid var(--line); border-radius: var(--radius-md); background: linear-gradient(180deg, #101b2b 0%, #0a1321 100%); }
    .live-activity pre { margin-top: 10px; font-size: 12px; max-height: 220px; overflow: auto; }
    .live-activity-inline {
      position: absolute;
      top: 16px;
      right: 16px;
      z-index: 2;
      width: min(440px, calc(100% - 32px));
      max-height: calc(100% - 32px);
      margin-top: 0;
      overflow: auto;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .live-neb-layout { display: grid; grid-template-columns: 190px minmax(0, 1fr) 190px; gap: 12px; align-items: start; }
    .live-neb-layout .mol-card { min-height: 210px; }
    .job-list { display: grid; gap: 8px; margin-top: 12px; }
    .job-row { border: 1px solid var(--line); border-radius: 14px; background: rgba(255, 255, 255, 0.03); padding: 8px 10px; }
    .job-row.running { border-color: rgba(126, 240, 199, 0.38); }
    .job-row.completed { border-color: rgba(89, 216, 182, 0.3); }
    .job-row.failed { border-color: rgba(255, 122, 143, 0.42); }
    .log-grid { display: grid; gap: 12px; }
    @media (max-width: 1080px) {
      .summary-strip, .inputs-grid, .form-grid, .viewer-grid, .network-workspace-grid { grid-template-columns: 1fr; }
      .explorer-svg { min-height: 520px; }
      .live-neb-layout { grid-template-columns: 1fr; }
      .network-toolbar, .live-activity-inline {
        position: static;
        width: auto;
        max-height: none;
        margin-bottom: 12px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel page-title">
      <div class="eyebrow">Reaction Discovery Workspace</div>
      <h1>MEPD Drive</h1>
      <div class="muted" style="margin-bottom:12px;">Set inputs, build a seed network, then manually grow and explore the graph while inspecting logs from a single workspace.</div>
      <div id="job-banner" class="message">Idle.</div>
      <div id="job-subtext" class="muted" style="margin-top:8px;">No action submitted yet.</div>
      <div id="activity-indicator" class="activity-indicator" role="status" aria-live="polite">
        <div class="activity-flask" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M9 2h6v2h-1v4.6l4.6 7.2c1.2 2-0.3 4.2-2.6 4.2H8c-2.3 0-3.8-2.2-2.6-4.2L10 8.6V4H9V2z" stroke="#7ef0c7" stroke-width="1.6" stroke-linejoin="round"/>
            <path d="M8.1 14.4h7.8" stroke="#63d5ff" stroke-width="1.5" stroke-linecap="round"/>
            <circle class="bubble-a" cx="11.1" cy="12.1" r="1.05" fill="#7ef0c7"/>
            <circle class="bubble-b" cx="13.6" cy="10.9" r="0.85" fill="#63d5ff"/>
          </svg>
        </div>
        <div>
          <div id="activity-title" class="activity-title">Working...</div>
          <div id="activity-detail" class="activity-detail">Preparing computation...</div>
          <div id="activity-elapsed" class="activity-elapsed">Elapsed 0s</div>
        </div>
      </div>
      <div id="hawaii-stoplight" class="stoplight-shell">
        <div class="stoplight-head">
          <div><strong>Hawaii Stoplight</strong></div>
          <div id="hawaii-mode" class="muted">Mode: GO</div>
        </div>
        <div class="stoplight-row">
          <button id="hawaii-go" class="stoplight-button go">Green: GO</button>
          <button id="hawaii-yellow" class="stoplight-button yellow">Yellow: Stop After Stage</button>
          <button id="hawaii-red" class="stoplight-button red">Red: Stop Now</button>
        </div>
        <div class="stoplight-options">
          <div class="stoplight-options-title">Discovery Tools (Step 3)</div>
          <div class="stoplight-options-row">
            <label class="stoplight-option"><input id="hawaii-discovery-hessian" type="checkbox" checked /> hessian-sample</label>
            <label class="stoplight-option"><input id="hawaii-discovery-retropaths" type="checkbox" /> retropaths</label>
            <label class="stoplight-option"><input id="hawaii-discovery-nanoreactor" type="checkbox" /> nanoreactor</label>
          </div>
          <div id="hawaii-discovery-summary" class="stoplight-summary">Selected: hessian-sample</div>
        </div>
        <div id="hawaii-note" class="muted" style="margin-top:8px;">Green starts autonomous mode with the selected discovery tools. Yellow lets the current stage finish. Red stops immediately.</div>
      </div>
    </div>

    <div class="panel">
      <div class="section-head">
        <h2>Inputs</h2>
        <div class="muted">Starting geometries, workspace controls, and current path-minimization settings.</div>
      </div>
      <div class="tool-tabs">
        <button class="tool-tab active" data-tool-tab="inputs" data-tool-target="setup">Workspace Setup</button>
        <button class="tool-tab" data-tool-tab="inputs" data-tool-target="pathmin">Path Minimization Inputs</button>
      </div>
      <div id="tool-panel-inputs-setup" class="tool-panel active">
        <div class="inputs-grid">
          <div>
            <div class="form-grid">
              <div>
                <label>Mode</label>
                <select id="mode">
                  <option value="reactant">Reactant only</option>
                  <option value="reactant-product">Reactant and product</option>
                </select>
              </div>
              <div>
                <label>Run name</label>
                <input id="run-name" type="text" placeholder="Optional workspace name" />
              </div>
              <div>
                <label>Existing workspace path</label>
                <input id="workspace-path" type="text" placeholder="Path to an existing mepd-drive workspace or workspace.json" />
              </div>
              <div>
                <label>Inputs TOML</label>
                <input id="inputs-path" type="text" placeholder="Path to RunInputs TOML (optional if drive was launched with --inputs)" />
              </div>
              <div>
                <label>Engine</label>
                <input id="theory-engine" type="text" value="chemcloud" readonly />
              </div>
              <div>
                <label>Program</label>
                <select id="theory-program">
                  <option value="crest">crest</option>
                  <option value="terachem">terachem</option>
                </select>
              </div>
              <div>
                <label>Method</label>
                <input id="theory-method" type="text" placeholder="gfn2 or ub3lyp" />
              </div>
              <div>
                <label>Basis</label>
                <input id="theory-basis" type="text" placeholder="gfn2 or sto-3g" />
              </div>
              <div>
                <label>Reactions File</label>
                <input id="reactions-path" type="text" placeholder="Optional path to retropaths reactions.p" />
              </div>
              <div>
                <label>Environment SMILES</label>
                <input id="environment-smiles" type="text" placeholder="Optional environment SMILES" />
              </div>
              <div>
                <label>Reactant SMILES</label>
                <input id="reactant-smiles" type="text" placeholder="SMILES or leave blank if you paste XYZ below" />
              </div>
              <div>
                <label>Product SMILES</label>
                <input id="product-smiles" type="text" placeholder="Optional product SMILES" />
              </div>
              <div>
                <label>Reactant XYZ</label>
                <textarea id="reactant-xyz" placeholder="Paste an XYZ block here if you want to initialize from a 3D structure"></textarea>
              </div>
              <div>
                <label>Product XYZ</label>
                <textarea id="product-xyz" placeholder="Optional product XYZ block"></textarea>
              </div>
            </div>
            <div style="margin-top:12px;">
              <button id="initialize-seed" class="primary">Build Seed Network</button>
              <button id="initialize-grow" class="secondary">Build + Grow Retropaths</button>
              <button id="load-workspace" class="secondary">Load Existing Workspace</button>
              <button id="minimize-all" class="secondary">Queue Minimization For All Geometries</button>
            </div>
            <div class="muted" style="margin-top:10px;">Drive uses the engine declared in the selected inputs TOML (`qcop` or `chemcloud`). You can choose the QC program and model used for this deployment.</div>
          </div>
          <div>
            <div id="workspace-summary" class="muted">No workspace initialized yet.</div>
            <div id="inputs-summary" style="margin-top:12px;"></div>
          </div>
        </div>
      </div>
      <div id="tool-panel-inputs-pathmin" class="tool-panel">
        <div id="pathmin-config-panel" class="muted">Load or initialize a workspace to inspect the current path minimization inputs.</div>
      </div>
    </div>

    <div class="panel">
      <div class="section-head">
        <h2>Reaction Network</h2>
        <div class="muted">The populated reaction network is the main workspace. Click nodes and edges directly on the canvas.</div>
      </div>
      <div class="summary-strip">
        <div id="network-summary" class="muted">No workspace initialized yet.</div>
        <div id="stats" class="stats"></div>
      </div>
      <div class="network-workspace-grid">
        <div class="network-left-stack">
          <div class="network-canvas-shell">
            <div id="network-toolbar" class="network-toolbar"></div>
            <svg id="network-svg" class="explorer-svg" viewBox="0 0 1180 680" role="img" aria-label="MEPD Drive network graph"></svg>
            <div id="network-context-menu" class="network-context-menu"></div>
            <div id="live-activity-inline" class="live-activity live-activity-inline" style="display:none;"></div>
          </div>
          <div class="network-canvas-hint">Right-click inside the network for tools. Scroll to zoom and drag empty space to pan.</div>
          <div class="path-browser">
            <div class="path-browser-head">
              <div>
                <h3>Products & Paths</h3>
                <div class="muted">Select structure A, then highlight the shortest route(s) to a created product.</div>
              </div>
            </div>
            <div class="path-browser-controls">
              <div>
                <label for="path-source-node">Structure A</label>
                <select id="path-source-node"></select>
              </div>
              <div style="display:flex; gap:8px; flex-wrap:wrap;">
                <button id="clear-product-path" class="secondary" type="button">Clear Highlight</button>
              </div>
            </div>
            <div id="product-path-summary" class="muted">Initialize or load a workspace to browse created products.</div>
            <div id="product-path-list" class="product-list" style="margin-top:12px;"></div>
          </div>
        </div>
        <div class="exploration-shell">
          <div class="section-head">
            <h2>Exploration</h2>
            <div class="muted">Queue NEBs, minimizations, reaction-template application, nanoreactor sampling, Hessian minima exploration, and inspect the selected graph item.</div>
          </div>
          <div id="detail-title" style="font-size:22px; margin-bottom:6px;">Select a node or edge</div>
          <div id="detail-summary" class="muted">Click a node to inspect its geometry or click an edge to inspect the targeted reaction, template data, and queue NEB work.</div>
          <div class="detail-tabs">
            <button class="detail-tab active" data-tab="targeted">Queue & Actions</button>
            <button class="detail-tab" data-tab="template-data">Template Data</button>
            <button class="detail-tab" data-tab="structures">Structures</button>
            <button class="detail-tab" data-tab="manual-edge">Manual Edge</button>
          </div>
          <div id="panel-targeted" class="detail-panel active"></div>
          <div id="panel-template-data" class="detail-panel"></div>
          <div id="panel-structures" class="detail-panel"></div>
          <div id="panel-manual-edge" class="detail-panel">
            <div class="form-grid">
              <div>
                <label>Manual edge source node</label>
                <input id="manual-edge-source" type="number" min="0" placeholder="Source node id" />
              </div>
              <div>
                <label>Manual edge target node</label>
                <input id="manual-edge-target" type="number" min="0" placeholder="Target node id" />
              </div>
              <div>
                <label>Manual edge label</label>
                <input id="manual-edge-label" type="text" placeholder="Optional reaction label" />
              </div>
              <div style="display:flex; align-items:end;">
                <button id="add-manual-edge" class="secondary" style="width:100%;">Attempt To Add Manual Edge</button>
              </div>
            </div>
            <div class="form-grid" style="margin-top:12px;">
              <div style="grid-column:1 / -1;">
                <label>Manual node XYZ</label>
                <textarea id="manual-node-xyz" placeholder="Paste XYZ block for the node you want to insert"></textarea>
              </div>
              <div>
                <label>Manual node charge</label>
                <input id="manual-node-charge" type="number" step="1" value="0" />
              </div>
              <div>
                <label>Manual node multiplicity</label>
                <input id="manual-node-multiplicity" type="number" step="1" min="1" value="1" />
              </div>
              <div style="display:flex; align-items:end;">
                <button id="insert-manual-node-mode" data-drive-action="insert-node-mode" class="secondary" style="width:100%;">Insert Node On Next Network Click</button>
              </div>
            </div>
            <div class="muted" style="margin-top:10px;">
              Paste an XYZ block, click "Insert Node On Next Network Click", then click empty space in the network canvas where the node should appear.
            </div>
          </div>
        </div>
      </div>
      <div id="kinetics-shell" class="kinetics-shell">
        <div class="section-head">
          <h2>Kinetics</h2>
          <div class="muted">Run a kinetic Monte Carlo model from the current network.</div>
        </div>
        <div class="form-grid">
          <div>
            <label>Kinetics temperature (K)</label>
            <input id="kmc-temperature" type="number" step="0.1" min="0" value="298.15" />
          </div>
          <div>
            <label>Kinetics final time</label>
            <input id="kmc-final-time" type="number" step="any" min="0" value="" />
          </div>
          <div>
            <label>Kinetics max steps</label>
            <input id="kmc-max-steps" type="number" step="1" min="1" value="200" />
          </div>
          <div>
            <label>Initial conditions JSON</label>
            <textarea
              id="kmc-initial-conditions"
              class="json-textarea"
              spellcheck="false"
              placeholder='{\n  "0": 1.0,\n  "4": 0.25\n}'
            ></textarea>
            <div class="json-input-controls">
              <button id="format-kmc-json" class="secondary" type="button">Format JSON</button>
            </div>
          </div>
        </div>
        <div style="margin-top:12px;">
          <button id="run-kmc" class="secondary">Run Kinetics Model</button>
        </div>
        <div id="kmc-panel" style="margin-top:12px;"></div>
      </div>
    </div>

    <div class="panel">
      <div class="section-head">
        <h2>Logging</h2>
        <div class="muted">Live monitors, debug state, and detailed backend output.</div>
      </div>
      <div class="tool-tabs">
        <button class="tool-tab active" data-tool-tab="logging" data-tool-target="activity">Activity</button>
        <button class="tool-tab" data-tool-tab="logging" data-tool-target="console">Console</button>
        <button class="tool-tab" data-tool-tab="logging" data-tool-target="state">State</button>
      </div>
      <div id="tool-panel-logging-activity" class="tool-panel active">
        <div id="live-activity-panel" class="live-activity" style="display:none;"></div>
      </div>
      <div id="tool-panel-logging-console" class="tool-panel">
        <div id="log-console" class="code-block">No log output yet.</div>
      </div>
      <div id="tool-panel-logging-state" class="tool-panel">
        <div class="log-grid">
          <div id="log-status" class="message">Idle.</div>
          <div id="log-state" class="code-block">No state captured yet.</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const ns = "http://www.w3.org/2000/svg";
    const DRIVE_URL_TOKEN = (() => {
      try {
        return String(new URLSearchParams(window.location.search).get("token") || "");
      } catch (_error) {
        return "";
      }
    })();
    if (DRIVE_URL_TOKEN) {
      try {
        const cleanUrl = new URL(window.location.href);
        cleanUrl.searchParams.delete("token");
        window.history.replaceState({}, document.title, cleanUrl.toString());
      } catch (_error) {}
    }
    function authHeaders() {
      return DRIVE_URL_TOKEN ? { "X-MEPD-Token": DRIVE_URL_TOKEN } : {};
    }
    const HAWAII_DISCOVERY_TOOL_ORDER = ["hessian-sample", "retropaths", "nanoreactor"];
    const HAWAII_DISCOVERY_TOOL_INPUTS = {
      "hessian-sample": "hawaii-discovery-hessian",
      "retropaths": "hawaii-discovery-retropaths",
      "nanoreactor": "hawaii-discovery-nanoreactor",
    };
    const state = {
      snapshot: null,
      selected: null,
      networkVersion: "",
      networkView: { x: 0, y: 0, scale: 1, minScale: 0.35, maxScale: 4.5, suppressClickUntil: 0 },
      networkCanvas: null,
      inputsDefaultsKey: "",
      connectSourceNodeId: null,
      manualEdgeRequestInFlight: false,
      manualNodeInsertMode: false,
      manualNodeRequestInFlight: false,
      contextMenuGraphPoint: null,
      pathSourceNodeId: 0,
      selectedProductKey: "",
      pathHighlight: null,
      pendingLiveActivity: null,
      pendingEdgeAddition: null,
      networkLayoutVersion: "",
      networkNodePositions: {},
      refreshTimer: null,
      kmcResult: null,
      hessianSampleDr: 1.0,
      hessianSampleMaxCandidates: 100,
      hessianSampleUseBigchem: false,
      hawaiiControlInFlight: false,
      hawaiiDiscoveryDirty: false,
      localActivities: new Map(),
      nextActivityToken: 1,
      activityTicker: null,
      backendBusySince: 0,
    };

    function normalizedHawaiiDiscoveryTools(rawTools) {
      const include = new Set();
      if (Array.isArray(rawTools)) {
        rawTools.forEach((value) => {
          const key = String(value || "").trim().toLowerCase();
          if (HAWAII_DISCOVERY_TOOL_ORDER.includes(key)) include.add(key);
        });
      }
      return HAWAII_DISCOVERY_TOOL_ORDER.filter((tool) => include.has(tool));
    }

    function getSelectedHawaiiDiscoveryTools() {
      return HAWAII_DISCOVERY_TOOL_ORDER.filter((tool) => {
        const input = document.getElementById(HAWAII_DISCOVERY_TOOL_INPUTS[tool]);
        return Boolean(input?.checked);
      });
    }

    function setSelectedHawaiiDiscoveryTools(tools, { force = false } = {}) {
      if (state.hawaiiDiscoveryDirty && !force) return;
      const normalized = normalizedHawaiiDiscoveryTools(tools);
      const selected = new Set(normalized);
      HAWAII_DISCOVERY_TOOL_ORDER.forEach((tool) => {
        const input = document.getElementById(HAWAII_DISCOVERY_TOOL_INPUTS[tool]);
        if (!input) return;
        input.checked = selected.has(tool);
      });
      state.hawaiiDiscoveryDirty = false;
    }

    function setManualEdgeEndpoint(which, nodeId) {
      const input = document.getElementById(which === "source" ? "manual-edge-source" : "manual-edge-target");
      if (input) input.value = String(Number(nodeId));
    }

    function setPendingEdgeAddition(sourceNodeId, targetNodeId) {
      state.pendingEdgeAddition = {
        source: Number(sourceNodeId),
        target: Number(targetNodeId),
      };
      if (state.snapshot?.drive?.network) renderNetwork(state.snapshot);
    }

    function clearPendingEdgeAddition() {
      if (!state.pendingEdgeAddition) return;
      state.pendingEdgeAddition = null;
      if (state.snapshot?.drive?.network) renderNetwork(state.snapshot);
    }

    function setManualEdgeRequestInFlight(inFlight) {
      state.manualEdgeRequestInFlight = Boolean(inFlight);
      const addButton = document.getElementById("add-manual-edge");
      if (addButton) addButton.disabled = state.manualEdgeRequestInFlight;
      const sourceInput = document.getElementById("manual-edge-source");
      if (sourceInput) sourceInput.disabled = state.manualEdgeRequestInFlight;
      const targetInput = document.getElementById("manual-edge-target");
      if (targetInput) targetInput.disabled = state.manualEdgeRequestInFlight;
      const labelInput = document.getElementById("manual-edge-label");
      if (labelInput) labelInput.disabled = state.manualEdgeRequestInFlight;
      renderNetworkToolbar();
    }

    function setManualNodeInsertMode(enabled) {
      const nextMode = Boolean(enabled) && !state.manualNodeRequestInFlight;
      state.manualNodeInsertMode = nextMode;
      const insertButton = document.getElementById("insert-manual-node-mode");
      if (insertButton) {
        insertButton.classList.toggle("active", nextMode);
        insertButton.textContent = nextMode
          ? "Click Network To Insert Node (Esc To Cancel)"
          : "Insert Node On Next Network Click";
        insertButton.disabled = state.manualNodeRequestInFlight || Boolean(state.snapshot?.busy);
      }
      renderNetworkToolbar();
    }

    function setManualNodeRequestInFlight(inFlight) {
      state.manualNodeRequestInFlight = Boolean(inFlight);
      if (state.manualNodeRequestInFlight) {
        state.manualNodeInsertMode = false;
      }
      const insertButton = document.getElementById("insert-manual-node-mode");
      if (insertButton) insertButton.disabled = state.manualNodeRequestInFlight;
      const xyzInput = document.getElementById("manual-node-xyz");
      if (xyzInput) xyzInput.disabled = state.manualNodeRequestInFlight;
      const chargeInput = document.getElementById("manual-node-charge");
      if (chargeInput) chargeInput.disabled = state.manualNodeRequestInFlight;
      const multiplicityInput = document.getElementById("manual-node-multiplicity");
      if (multiplicityInput) multiplicityInput.disabled = state.manualNodeRequestInFlight;
      setManualNodeInsertMode(state.manualNodeInsertMode);
    }

    function getManualNodeCharge() {
      const input = document.getElementById("manual-node-charge");
      const parsed = Number(input ? input.value : 0);
      const value = Number.isFinite(parsed) ? Math.trunc(parsed) : 0;
      if (input) input.value = String(value);
      return value;
    }

    function getManualNodeMultiplicity() {
      const input = document.getElementById("manual-node-multiplicity");
      const parsed = Number(input ? input.value : 1);
      const value = Number.isFinite(parsed) && parsed > 0 ? Math.trunc(parsed) : 1;
      if (input) input.value = String(value);
      return value;
    }

    function canvasCoordsToGraphCoords(canvasX, canvasY) {
      const view = state.networkView || {};
      const scaleRaw = Number(view.scale || 1);
      const scale = Number.isFinite(scaleRaw) && Math.abs(scaleRaw) > 1e-8 ? scaleRaw : 1;
      return {
        x: (Number(canvasX) - Number(view.x || 0)) / scale,
        y: (Number(canvasY) - Number(view.y || 0)) / scale,
      };
    }

    async function insertManualNodeAtGraphCoords(graphX, graphY) {
      if (state.manualNodeRequestInFlight) {
        setBanner("A manual node insertion request is already in progress.", true);
        return;
      }
      const xyzInput = document.getElementById("manual-node-xyz");
      const xyzText = String(xyzInput ? xyzInput.value : "").trim();
      if (!xyzText) {
        setBanner("Manual node insertion requires an XYZ block.", true);
        setSubtext("Paste XYZ text in the Manual Edge tab before inserting a node.");
        return;
      }
      const charge = getManualNodeCharge();
      const multiplicity = getManualNodeMultiplicity();
      try {
        setManualNodeRequestInFlight(true);
        setBanner("Adding manual node from XYZ...");
        setSubtext("Inserting a new node at the selected network location.");
        const result = await postJson("/api/add-node", {
          xyz_text: xyzText,
          charge: Number(charge),
          multiplicity: Number(multiplicity),
        });
        const nodeId = Number(result.node_id);
        if (Number.isFinite(nodeId) && nodeId >= 0) {
          state.networkNodePositions[String(nodeId)] = {
            x: Number(graphX),
            y: Number(graphY),
          };
        }
        setBanner(result.message || `Manual node ${Number.isFinite(nodeId) ? nodeId : ""} inserted.`);
        await refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The manual node could not be inserted.");
      } finally {
        setManualNodeRequestInFlight(false);
      }
    }

    function toggleManualNodeInsertMode() {
      if (state.manualNodeRequestInFlight) {
        setBanner("A manual node insertion request is already in progress.", true);
        setSubtext("Wait for the current manual node insertion request to finish.");
        return;
      }
      if (state.manualNodeInsertMode) {
        setManualNodeInsertMode(false);
        setSubtext("Node insert mode cleared.");
        return;
      }
      if (state.snapshot?.busy) {
        setBanner("Wait for the current background action to finish before inserting nodes.", true);
        return;
      }
      const xyzInput = document.getElementById("manual-node-xyz");
      const xyzText = String(xyzInput ? xyzInput.value : "").trim();
      if (!xyzText) {
        setBanner("Manual node insertion requires an XYZ block.", true);
        setSubtext("Paste XYZ text in the Manual Edge tab before enabling click-to-insert mode.");
        return;
      }
      setManualNodeInsertMode(true);
      setSubtext("Node insert mode active. Click empty space in the network canvas to place the node.");
    }

    function normalizeHessianSampleDr(value) {
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed <= 0) return 1.0;
      return parsed;
    }

    function normalizeHessianSampleMaxCandidates(value) {
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed <= 0) return 100;
      return Math.max(1, Math.floor(parsed));
    }

    function getHessianSampleDr() {
      const input = document.getElementById("hessian-sample-dr");
      const value = normalizeHessianSampleDr(input ? input.value : state.hessianSampleDr);
      state.hessianSampleDr = value;
      if (input) {
        input.value = String(value);
      }
      return value;
    }

    function getHessianSampleMaxCandidates() {
      const input = document.getElementById("hessian-sample-max-candidates");
      const value = normalizeHessianSampleMaxCandidates(
        input ? input.value : state.hessianSampleMaxCandidates
      );
      state.hessianSampleMaxCandidates = value;
      if (input) {
        input.value = String(value);
      }
      return value;
    }

    function getHessianSampleUseBigchem() {
      const input = document.getElementById("hessian-sample-use-bigchem");
      const value = Boolean(input ? input.checked : state.hessianSampleUseBigchem);
      state.hessianSampleUseBigchem = value;
      if (input) {
        input.checked = value;
      }
      return value;
    }

    function bindHessianSampleDrInput() {
      const drInput = document.getElementById("hessian-sample-dr");
      if (drInput) {
        drInput.value = String(normalizeHessianSampleDr(state.hessianSampleDr));
        if (drInput.dataset.bound !== "true") {
          drInput.dataset.bound = "true";
          drInput.addEventListener("change", () => {
            getHessianSampleDr();
          });
          drInput.addEventListener("blur", () => {
            getHessianSampleDr();
          });
        }
      }
      const maxInput = document.getElementById("hessian-sample-max-candidates");
      if (maxInput) {
        maxInput.value = String(
          normalizeHessianSampleMaxCandidates(state.hessianSampleMaxCandidates)
        );
        if (maxInput.dataset.bound !== "true") {
          maxInput.dataset.bound = "true";
          maxInput.addEventListener("change", () => {
            getHessianSampleMaxCandidates();
          });
          maxInput.addEventListener("blur", () => {
            getHessianSampleMaxCandidates();
          });
        }
      }
      const useBigchemInput = document.getElementById("hessian-sample-use-bigchem");
      if (useBigchemInput) {
        useBigchemInput.checked = Boolean(state.hessianSampleUseBigchem);
        if (useBigchemInput.dataset.bound !== "true") {
          useBigchemInput.dataset.bound = "true";
          useBigchemInput.addEventListener("change", () => {
            getHessianSampleUseBigchem();
          });
        }
      }
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function renderMoleculeCard(payload) {
      if (!payload) {
        return `<div class="mol-card"><div class="mol-empty">No structure available.</div></div>`;
      }
      if (payload.svg) {
        return `
          <div class="mol-card">
            ${payload.svg}
            <div class="mol-meta"><strong>SMILES:</strong> ${escapeHtml(payload.smiles || "")}</div>
          </div>
        `;
      }
      return `
        <div class="mol-card">
          <div class="mol-empty">RDKit render unavailable.</div>
          <div class="mol-meta"><strong>SMILES:</strong> ${escapeHtml(payload.smiles || "")}</div>
        </div>
      `;
    }

    function makeStructureSrcdoc(xyzB64) {
      if (!xyzB64) {
        return "<html><body style='margin:0;padding:16px;background:#0b1423;color:#8ea3c2;font:500 14px/1.5 IBM Plex Sans,Aptos,Segoe UI Variable,sans-serif;'>No 3D structure available.</body></html>";
      }
      return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>
    html, body, #viewer { margin: 0; width: 100%; height: 100%; background: linear-gradient(180deg, #101b2b 0%, #0a1321 100%); overflow: hidden; }
    #status { padding: 12px; color: #8ea3c2; font: 500 14px/1.5 IBM Plex Sans,Aptos,Segoe UI Variable,sans-serif; }
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div id="status">Loading 3D viewer...</div>
  <script>
    const xyz = decodeURIComponent(escape(atob("__XYZ_B64__")));
    function boot() {
      const host = document.getElementById("viewer");
      const status = document.getElementById("status");
      status.remove();
      const viewer = $3Dmol.createViewer(host, { backgroundColor: "#0d1728" });
      viewer.addModel(xyz, "xyz");
      viewer.setStyle({}, { stick: { radius: 0.18 }, sphere: { scale: 0.28 } });
      viewer.zoomTo();
      viewer.render();
    }
    if (window.$3Dmol) {
      boot();
    } else {
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/3dmol@2.5.3/build/3Dmol-min.js";
      script.onload = boot;
      script.onerror = () => {
        document.getElementById("status").textContent = "Failed to load 3Dmol.js";
      };
      document.head.appendChild(script);
    }
  <\\/script>
</body>
</html>`.replace("__XYZ_B64__", xyzB64);
    }

    function setBanner(text, isError = false) {
      const banner = document.getElementById("job-banner");
      banner.textContent = text;
      banner.style.borderColor = isError ? "var(--warn)" : "var(--line)";
      banner.style.color = isError ? "var(--warn)" : "var(--ink)";
    }

    function setSubtext(text) {
      document.getElementById("job-subtext").textContent = text;
    }

    function formatElapsedDuration(milliseconds) {
      const totalSeconds = Math.max(0, Math.floor(Number(milliseconds || 0) / 1000));
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      return minutes > 0 ? `${minutes}m ${String(seconds).padStart(2, "0")}s` : `${seconds}s`;
    }

    function visibleLocalActivityEntries() {
      return Array.from(state.localActivities.entries())
        .filter(([, entry]) => Boolean(entry && entry.visible))
        .sort((left, right) => Number(left[0]) - Number(right[0]));
    }

    function latestVisibleLocalActivity() {
      const visibleEntries = visibleLocalActivityEntries();
      if (!visibleEntries.length) return null;
      return visibleEntries[visibleEntries.length - 1][1];
    }

    function ensureActivityTicker() {
      if (state.activityTicker != null) return;
      state.activityTicker = window.setInterval(() => {
        renderActivityIndicator(state.snapshot);
      }, 1000);
    }

    function maybeStopActivityTicker(snapshot = state.snapshot) {
      const hasVisibleLocal = visibleLocalActivityEntries().length > 0;
      const backendBusy = Boolean(snapshot?.busy);
      if (!hasVisibleLocal && !backendBusy && state.activityTicker != null) {
        clearInterval(state.activityTicker);
        state.activityTicker = null;
      }
    }

    function renderActivityIndicator(snapshot = state.snapshot) {
      const indicator = document.getElementById("activity-indicator");
      const titleEl = document.getElementById("activity-title");
      const detailEl = document.getElementById("activity-detail");
      const elapsedEl = document.getElementById("activity-elapsed");
      if (!indicator || !titleEl || !detailEl || !elapsedEl) return;

      const localActivity = latestVisibleLocalActivity();
      const backendBusy = Boolean(snapshot?.busy);
      if (backendBusy && !state.backendBusySince) {
        state.backendBusySince = Date.now();
      } else if (!backendBusy) {
        state.backendBusySince = 0;
      }

      if (!localActivity && !backendBusy) {
        indicator.classList.remove("visible");
        maybeStopActivityTicker(snapshot);
        return;
      }

      let title = "Running background computation...";
      let detail = "Processing the current request.";
      let startedAt = Date.now();
      if (localActivity) {
        title = String(localActivity.label || title);
        detail = String(localActivity.detail || detail);
        startedAt = Number(localActivity.startedAt || Date.now());
      } else {
        const action = snapshot?.active_action || null;
        const actionType = String(action?.type || "").toLowerCase();
        title = String(action?.label || snapshot?.busy_label || title);
        if (actionType === "load-workspace") {
          detail = "Reading workspace files and rebuilding the network snapshot.";
        } else if (actionType === "initialize") {
          detail = "Seeding and expanding the network graph.";
        } else if (actionType === "neb") {
          detail = "Autosplitting NEB is running in the backend.";
        } else {
          detail = "Waiting for backend updates.";
        }
        startedAt = Number(state.backendBusySince || Date.now());
      }

      titleEl.textContent = title;
      detailEl.textContent = detail;
      elapsedEl.textContent = `Elapsed ${formatElapsedDuration(Date.now() - startedAt)}`;
      indicator.classList.add("visible");
      ensureActivityTicker();
    }

    function beginLocalActivity(label, detail = "", delayMs = 0) {
      const token = Number(state.nextActivityToken || 1);
      state.nextActivityToken = token + 1;
      const entry = {
        label: String(label || "Working..."),
        detail: String(detail || ""),
        startedAt: Date.now(),
        visible: Number(delayMs || 0) <= 0,
        delayTimer: null,
      };
      if (Number(delayMs || 0) > 0) {
        entry.delayTimer = window.setTimeout(() => {
          const active = state.localActivities.get(token);
          if (!active) return;
          active.visible = true;
          renderActivityIndicator(state.snapshot);
        }, Number(delayMs));
      }
      state.localActivities.set(token, entry);
      renderActivityIndicator(state.snapshot);
      return token;
    }

    function endLocalActivity(token) {
      const entry = state.localActivities.get(Number(token));
      if (!entry) return;
      if (entry.delayTimer != null) {
        clearTimeout(entry.delayTimer);
      }
      state.localActivities.delete(Number(token));
      renderActivityIndicator(state.snapshot);
      maybeStopActivityTicker(state.snapshot);
    }

    async function withLocalActivity(label, detail, task, { delayMs = 0 } = {}) {
      const token = beginLocalActivity(label, detail, delayMs);
      try {
        return await task();
      } finally {
        endLocalActivity(token);
      }
    }

    function renderHawaiiStoplight(snapshot) {
      const payload = snapshot?.hawaii || {};
      const mode = String(payload.mode || "go").toLowerCase();
      const running = Boolean(payload.running);
      const stage = String(payload.stage || "").trim();
      const dr = payload.dr == null ? "" : `dr=${Number(payload.dr).toFixed(1)}`;
      const configuredTools = normalizedHawaiiDiscoveryTools(payload.discovery_tools);
      if (!state.hawaiiDiscoveryDirty) {
        setSelectedHawaiiDiscoveryTools(configuredTools, { force: true });
      }
      const selectedTools = getSelectedHawaiiDiscoveryTools();
      const selectedLabel = selectedTools.length ? selectedTools.join(", ") : "none";
      const modeLabel = mode === "yellow" ? "YELLOW" : mode === "red" ? "RED" : "GO";
      const modeEl = document.getElementById("hawaii-mode");
      const noteEl = document.getElementById("hawaii-note");
      const discoverySummaryEl = document.getElementById("hawaii-discovery-summary");
      const goBtn = document.getElementById("hawaii-go");
      const yellowBtn = document.getElementById("hawaii-yellow");
      const redBtn = document.getElementById("hawaii-red");
      if (modeEl) {
        modeEl.textContent = `Mode: ${modeLabel}${running ? " (running)" : " (idle)"}`;
      }
      if (noteEl) {
        const stageText = stage ? ` Stage: ${stage}.` : "";
        const drText = dr ? ` ${dr}.` : "";
        const detail = String(payload.note || "").trim();
        noteEl.textContent = detail
          ? `${detail}${stageText}${drText}`
          : `Green starts autonomous mode with the selected discovery tools. Yellow lets the current stage finish. Red stops immediately.${stageText}${drText}`;
      }
      if (discoverySummaryEl) {
        const pendingSuffix = state.hawaiiDiscoveryDirty ? " (pending send)" : "";
        discoverySummaryEl.textContent = `Selected: ${selectedLabel}${pendingSuffix}`;
      }
      HAWAII_DISCOVERY_TOOL_ORDER.forEach((tool) => {
        const input = document.getElementById(HAWAII_DISCOVERY_TOOL_INPUTS[tool]);
        if (input) input.disabled = state.hawaiiControlInFlight;
      });
      if (goBtn) {
        goBtn.classList.toggle("active", mode === "go");
        goBtn.disabled = state.hawaiiControlInFlight;
      }
      if (yellowBtn) {
        yellowBtn.classList.toggle("active", mode === "yellow");
        yellowBtn.disabled = state.hawaiiControlInFlight || !running;
      }
      if (redBtn) {
        redBtn.classList.toggle("active", mode === "red");
        redBtn.disabled = state.hawaiiControlInFlight || !running;
      }
    }

    function clearPendingLiveActivity() {
      state.pendingLiveActivity = null;
    }

    function buildOptimisticGrowthActivity(nodeId, title, note) {
      const network = state.snapshot?.drive?.network || {};
      const nodes = Array.isArray(network.nodes) ? network.nodes : [];
      const edges = Array.isArray(network.edges) ? network.edges : [];
      const hydratedNodes = nodes.map((node) => ({
        id: Number(node.id),
        label: String(node.label || node.id),
        growing: Number(node.id) === Number(nodeId),
      }));
      if (!hydratedNodes.length) {
        hydratedNodes.push({
          id: Number(nodeId),
          label: String(nodeId),
          growing: true,
        });
      }
      return {
        type: "growth",
        title: title || "Growing Retropaths network",
        note: note || `Growing node ${Number(nodeId)}.`,
        phase: "growing",
        network: {
          nodes: hydratedNodes,
          edges: edges.map((edge) => ({
            source: Number(edge.source),
            target: Number(edge.target),
          })),
        },
      };
    }

    function setButtonsDisabled(disabled) {
      ["initialize-seed", "initialize-grow", "load-workspace", "minimize-all", "add-manual-edge"].forEach((id) => {
        const elem = document.getElementById(id);
        if (elem) elem.disabled = disabled;
      });
      document.querySelectorAll("[data-drive-action]").forEach((elem) => {
        elem.disabled = disabled;
      });
    }

    async function readJsonResponse(response) {
      const rawText = await response.text();
      if (!rawText) return {};
      try {
        return JSON.parse(rawText);
      } catch (_error) {
        const compact = rawText.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
        throw new Error(compact || `Request failed: ${response.status}`);
      }
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify(payload || {}),
      });
      let data = {};
      try {
        data = await readJsonResponse(response);
      } catch (error) {
        if (response.ok) throw error;
        throw new Error(error.message || `Request failed: ${response.status}`);
      }
      if (!response.ok) {
        throw new Error(data.error || `Request failed: ${response.status}`);
      }
      return data;
    }

    function renderStats(snapshot) {
      const stats = document.getElementById("stats");
      if (!snapshot || !snapshot.initialized || !snapshot.drive) {
        stats.innerHTML = "";
        return;
      }
      const queue = snapshot.drive.queue || {};
      const workspace = snapshot.drive.workspace || {};
      const entries = [
        ["Queue Items", queue.items || 0],
        ["Pending", queue.pending || 0],
        ["Running", queue.running || 0],
        ["Completed", queue.completed || 0],
        ["Failed", queue.failed || 0],
        ["Optimized Endpoints", workspace.optimized_endpoints || 0],
        ["NEB-backed Edges", workspace.neb_backed_edges || 0],
        ["Network Nodes", workspace.network_nodes || 0],
        ["Network Edges", workspace.network_edges || 0],
        ["Retropaths Nodes", workspace.retropaths_nodes || 0],
      ];
      stats.innerHTML = entries.map(([label, value]) => (
        `<div class="stat"><div class="muted">${escapeHtml(label)}</div><div style="font-size:24px;">${escapeHtml(value)}</div></div>`
      )).join("");
    }

    function deploymentProgramDefaults(program) {
      const normalized = String(program || "").trim().toLowerCase();
      if (normalized === "crest") {
        return { method: "gfn2", basis: "gfn2" };
      }
      if (normalized === "terachem") {
        return { method: "ub3lyp", basis: "sto-3g" };
      }
      return { method: "", basis: "" };
    }

    function syncTheoryModelFieldsToProgram(program, { force = false } = {}) {
      const theoryMethodField = document.getElementById("theory-method");
      const theoryBasisField = document.getElementById("theory-basis");
      const defaults = deploymentProgramDefaults(program);
      if (!theoryMethodField || !theoryBasisField) {
        return defaults;
      }
      const currentMethod = String(theoryMethodField.value || "").trim().toLowerCase();
      const currentBasis = String(theoryBasisField.value || "").trim().toLowerCase();
      const isCrestPair = currentMethod === "gfn2" && currentBasis === "gfn2";
      const isTerachemPair = currentMethod === "ub3lyp" && (currentBasis === "sto-3g" || currentBasis === "3-21g");
      if (force || !currentMethod || !currentBasis || isCrestPair || isTerachemPair) {
        theoryMethodField.value = defaults.method;
        theoryBasisField.value = defaults.basis;
      }
      return defaults;
    }

    function ensureTheoryProgramOptions(theoryProgramField, defaultsProgram, allowedPrograms) {
      if (!theoryProgramField) return;
      const normalizedAllowed = Array.isArray(allowedPrograms)
        ? allowedPrograms
            .map((value) => String(value || "").trim().toLowerCase())
            .filter((value) => value.length > 0)
        : [];
      if (defaultsProgram && !normalizedAllowed.includes(defaultsProgram)) {
        normalizedAllowed.push(defaultsProgram);
      }
      if (!normalizedAllowed.length) {
        normalizedAllowed.push("crest", "terachem");
      }
      const existing = new Set(
        Array.from(theoryProgramField.options || []).map((option) =>
          String(option.value || "").trim().toLowerCase()
        )
      );
      normalizedAllowed.forEach((program) => {
        if (existing.has(program)) return;
        const option = document.createElement("option");
        option.value = program;
        option.textContent = program;
        theoryProgramField.appendChild(option);
      });
    }

    function renderWorkspaceSummary(snapshot) {
      const inputEl = document.getElementById("workspace-summary");
      const networkEl = document.getElementById("network-summary");
      const defaults = snapshot?.defaults || {};
      const inputsPathField = document.getElementById("inputs-path");
      const reactionsPathField = document.getElementById("reactions-path");
      const environmentField = document.getElementById("environment-smiles");
      const runNameField = document.getElementById("run-name");
      const theoryProgramField = document.getElementById("theory-program");
      const theoryMethodField = document.getElementById("theory-method");
      const theoryBasisField = document.getElementById("theory-basis");
      const theoryEngineField = document.getElementById("theory-engine");
      if (inputsPathField && !inputsPathField.value) {
        inputsPathField.value = String(defaults.inputs_path || "");
      }
      if (reactionsPathField && !reactionsPathField.value) {
        reactionsPathField.value = String(defaults.reactions_fp || "");
      }
      const defaultsProgram = String(defaults.program || "terachem").trim().toLowerCase() || "terachem";
      const defaultsEngine = String(defaults.engine_name || "chemcloud").trim().toLowerCase() || "chemcloud";
      ensureTheoryProgramOptions(theoryProgramField, defaultsProgram, defaults.allowed_programs);
      const programDefaults = deploymentProgramDefaults(defaultsProgram);
      const defaultsMethod = String(defaults.method || programDefaults.method).trim() || programDefaults.method;
      const defaultsBasis = String(defaults.basis || programDefaults.basis).trim() || programDefaults.basis;
      const defaultsKey = `${String(defaults.inputs_path || "")}|${defaultsEngine}|${defaultsProgram}|${defaultsMethod}|${defaultsBasis}`;
      if (state.inputsDefaultsKey !== defaultsKey) {
        state.inputsDefaultsKey = defaultsKey;
        if (theoryEngineField) {
          theoryEngineField.value = defaultsEngine;
        }
        if (theoryProgramField) {
          theoryProgramField.value = defaultsProgram;
        }
        if (theoryMethodField) {
          theoryMethodField.value = defaultsMethod;
        }
        if (theoryBasisField) {
          theoryBasisField.value = defaultsBasis;
        }
      }
      if (theoryProgramField && theoryProgramField.dataset.bound !== "true") {
        theoryProgramField.dataset.bound = "true";
        theoryProgramField.addEventListener("change", () => {
          const selectedProgram = String(theoryProgramField.value || "terachem").trim().toLowerCase() || "terachem";
          syncTheoryModelFieldsToProgram(selectedProgram, { force: true });
        });
      }
      if (theoryProgramField) {
        const selectedProgram = String(theoryProgramField.value || "terachem").trim().toLowerCase() || "terachem";
        syncTheoryModelFieldsToProgram(selectedProgram, { force: false });
      }
      if (!inputEl || !networkEl) return;
      if (!snapshot || !snapshot.initialized || !snapshot.drive) {
        inputEl.textContent = "No workspace initialized yet.";
        networkEl.textContent = "No workspace initialized yet.";
        return;
      }
      const workspace = snapshot.drive.workspace;
      if (inputsPathField && !inputsPathField.value) {
        inputsPathField.value = String(snapshot.drive.inputs?.path || "");
      }
      if (reactionsPathField && !reactionsPathField.value) {
        reactionsPathField.value = String(workspace.reactions_fp || "");
      }
      if (environmentField && !environmentField.value) {
        environmentField.value = String(workspace.environment_smiles || "");
      }
      if (runNameField && !runNameField.value) {
        runNameField.value = String(workspace.run_name || "");
      }
      const html = `
        <span class="badge">${escapeHtml(workspace.run_name)}</span>
        <span class="badge">${escapeHtml(workspace.root_smiles)}</span>
        ${snapshot.product?.smiles ? `<span class="badge">Target: ${escapeHtml(snapshot.product.smiles)}</span>` : ""}
        ${workspace.environment_smiles ? `<span class="badge">Environment: ${escapeHtml(workspace.environment_smiles)}</span>` : ""}
        <div style="margin-top:8px;"><strong>Workspace:</strong> ${escapeHtml(workspace.directory)}</div>
      `;
      inputEl.innerHTML = html;
      networkEl.innerHTML = `
        <div style="margin-bottom:6px;"><strong>${escapeHtml(workspace.run_name)}</strong></div>
        <div class="muted">Root: ${escapeHtml(workspace.root_smiles)}</div>
        ${snapshot.product?.smiles ? `<div class="muted">Target: ${escapeHtml(snapshot.product.smiles)}</div>` : ""}
        <div class="muted">Workspace: ${escapeHtml(workspace.directory)}</div>
      `;
    }

    function canonicalEdgePairKey(a, b) {
      const first = Number(a);
      const second = Number(b);
      return first <= second ? `${first}|${second}` : `${second}|${first}`;
    }

    function getDriveNetwork(snapshot) {
      return snapshot?.drive?.network || null;
    }

    function ensurePathSourceNode(snapshot) {
      const nodes = Array.isArray(getDriveNetwork(snapshot)?.nodes) ? getDriveNetwork(snapshot).nodes : [];
      const nodeIds = new Set(nodes.map((node) => Number(node.id)));
      if (!nodeIds.size) {
        state.pathSourceNodeId = 0;
        return null;
      }
      const current = Number(state.pathSourceNodeId);
      if (Number.isFinite(current) && nodeIds.has(current)) {
        return current;
      }
      state.pathSourceNodeId = nodeIds.has(0) ? 0 : Number(nodes[0].id);
      return state.pathSourceNodeId;
    }

    function buildShortestPathSearch(nodes, edges, sourceNodeId, directed) {
      const adjacency = new Map();
      nodes.forEach((node) => adjacency.set(Number(node.id), []));
      edges.forEach((edge) => {
        const source = Number(edge.source);
        const target = Number(edge.target);
        if (!adjacency.has(source)) adjacency.set(source, []);
        if (!adjacency.has(target)) adjacency.set(target, []);
        adjacency.get(source).push(target);
        if (!directed) adjacency.get(target).push(source);
      });
      const source = Number(sourceNodeId);
      const queue = [source];
      const distances = new Map([[source, 0]]);
      const parents = new Map([[source, []]]);
      while (queue.length) {
        const current = Number(queue.shift());
        const currentDistance = Number(distances.get(current) || 0);
        (adjacency.get(current) || []).forEach((neighbor) => {
          const next = Number(neighbor);
          if (!distances.has(next)) {
            distances.set(next, currentDistance + 1);
            parents.set(next, [current]);
            queue.push(next);
            return;
          }
          if (Number(distances.get(next)) === currentDistance + 1) {
            const branchParents = parents.get(next) || [];
            if (!branchParents.includes(current)) branchParents.push(current);
            parents.set(next, branchParents);
          }
        });
      }
      return { distances, parents, mode: directed ? "directed" : "undirected" };
    }

    function enumerateShortestPaths(parents, sourceNodeId, targetNodeId, limit = 32) {
      const source = Number(sourceNodeId);
      const target = Number(targetNodeId);
      const paths = [];
      const seen = new Set();
      function walk(nodeId, suffix) {
        if (paths.length >= limit) return;
        const nextSuffix = [Number(nodeId), ...suffix];
        if (Number(nodeId) === source) {
          const key = nextSuffix.join("->");
          if (!seen.has(key)) {
            seen.add(key);
            paths.push(nextSuffix);
          }
          return;
        }
        const branchParents = parents.get(Number(nodeId)) || [];
        branchParents.forEach((parentId) => {
          walk(Number(parentId), nextSuffix);
        });
      }
      walk(target, []);
      return paths;
    }

    function buildProductRecords(snapshot, sourceNodeId) {
      const network = getDriveNetwork(snapshot);
      const nodes = Array.isArray(network?.nodes) ? network.nodes : [];
      const edges = Array.isArray(network?.edges) ? network.edges : [];
      const source = Number(sourceNodeId);
      const directedSearch = buildShortestPathSearch(nodes, edges, source, true);
      const undirectedSearch = buildShortestPathSearch(nodes, edges, source, false);
      const records = [];
      nodes.forEach((node) => {
        const nodeId = Number(node.id);
        if (nodeId === source) return;
        const label = String(node.label || node.id);
        const directedDistance = directedSearch.distances.get(nodeId);
        const undirectedDistance = undirectedSearch.distances.get(nodeId);
        const nearestDistance = directedDistance != null ? Number(directedDistance) : (
          undirectedDistance != null ? Number(undirectedDistance) : null
        );
        const mode = directedDistance != null ? "directed" : (undirectedDistance != null ? "undirected" : "");
        records.push({
          key: String(nodeId),
          nodeId,
          label,
          displayLabel: `${label} (${nodeId})`,
          nearestDistance,
          mode,
          reachable: nearestDistance != null,
        });
      });
      return records.sort((a, b) => {
        const aReachable = a.reachable ? 0 : 1;
        const bReachable = b.reachable ? 0 : 1;
        if (aReachable !== bReachable) return aReachable - bReachable;
        if ((a.nearestDistance ?? Infinity) !== (b.nearestDistance ?? Infinity)) {
          return (a.nearestDistance ?? Infinity) - (b.nearestDistance ?? Infinity);
        }
        const labelCmp = String(a.label).localeCompare(String(b.label));
        if (labelCmp !== 0) return labelCmp;
        return Number(a.nodeId) - Number(b.nodeId);
      });
    }

    function computePathHighlight(snapshot, sourceNodeId, productKey) {
      const network = getDriveNetwork(snapshot);
      const nodes = Array.isArray(network?.nodes) ? network.nodes : [];
      const edges = Array.isArray(network?.edges) ? network.edges : [];
      const source = Number(sourceNodeId);
      const targetNodeId = Number(productKey);
      if (!nodes.length || !Number.isFinite(targetNodeId)) return null;
      const targetNode = nodes.find((node) => Number(node.id) === targetNodeId);
      if (!targetNode || targetNodeId === source) return null;
      const targetLabel = String(targetNode.label || targetNode.id);

      let search = buildShortestPathSearch(nodes, edges, source, true);
      if (!search.distances.has(targetNodeId)) {
        search = buildShortestPathSearch(nodes, edges, source, false);
      }
      if (!search.distances.has(targetNodeId)) {
        return {
          productKey: String(targetNodeId),
          productLabel: targetLabel,
          sourceNodeId: source,
          targetNodeIds: [],
          pathNodeIds: [],
          edgePairs: [],
          paths: [],
          mode: "none",
        };
      }

      const paths = enumerateShortestPaths(search.parents, source, targetNodeId, 24);
      const pathNodeIds = Array.from(new Set(paths.flatMap((path) => path.map((nodeId) => Number(nodeId)))));
      const edgePairs = Array.from(new Set(paths.flatMap((path) => (
        path.slice(1).map((nodeId, index) => canonicalEdgePairKey(path[index], nodeId))
      ))));
      return {
        productKey: String(targetNodeId),
        productLabel: targetLabel,
        sourceNodeId: source,
        targetNodeIds: [targetNodeId],
        pathNodeIds,
        edgePairs,
        paths,
        mode: search.mode,
      };
    }

    function setPathSourceNode(nodeId) {
      state.pathSourceNodeId = Number(nodeId);
      const snapshot = state.snapshot;
      const records = buildProductRecords(snapshot, state.pathSourceNodeId);
      if (state.selectedProductKey && !records.some((record) => record.key === state.selectedProductKey)) {
        state.selectedProductKey = "";
      }
      state.pathHighlight = state.selectedProductKey
        ? computePathHighlight(snapshot, state.pathSourceNodeId, state.selectedProductKey)
        : null;
      renderProductPathPanel(snapshot);
      if (snapshot?.drive?.network) renderNetwork(snapshot);
    }

    function clearProductPathHighlight() {
      state.selectedProductKey = "";
      state.pathHighlight = null;
      renderProductPathPanel(state.snapshot);
      if (state.snapshot?.drive?.network) renderNetwork(state.snapshot);
    }

    function selectProductPath(productKey) {
      state.selectedProductKey = String(productKey || "");
      state.pathHighlight = computePathHighlight(state.snapshot, state.pathSourceNodeId, state.selectedProductKey);
      renderProductPathPanel(state.snapshot);
      if (state.snapshot?.drive?.network) renderNetwork(state.snapshot);
    }

    function renderProductPathPanel(snapshot) {
      const sourceSelect = document.getElementById("path-source-node");
      const summaryEl = document.getElementById("product-path-summary");
      const listEl = document.getElementById("product-path-list");
      if (!sourceSelect || !summaryEl || !listEl) return;
      const network = getDriveNetwork(snapshot);
      const nodes = Array.isArray(network?.nodes) ? network.nodes : [];
      if (!snapshot || !snapshot.initialized || !nodes.length) {
        sourceSelect.innerHTML = "";
        summaryEl.textContent = "Initialize or load a workspace to browse created products.";
        listEl.innerHTML = "";
        state.pathHighlight = null;
        return;
      }

      const sourceNodeId = ensurePathSourceNode(snapshot);
      const records = buildProductRecords(snapshot, sourceNodeId);
      const sourceNode = nodes.find((node) => Number(node.id) === Number(sourceNodeId)) || nodes[0];
      sourceSelect.innerHTML = nodes.map((node) => `
        <option value="${Number(node.id)}" ${Number(node.id) === Number(sourceNodeId) ? "selected" : ""}>
          ${escapeHtml(`${Number(node.id)} • ${String(node.label || node.id)}`)}
        </option>
      `).join("");

      if (state.selectedProductKey && !records.some((record) => record.key === state.selectedProductKey)) {
        state.selectedProductKey = "";
      }
      state.pathHighlight = state.selectedProductKey
        ? computePathHighlight(snapshot, sourceNodeId, state.selectedProductKey)
        : null;
      const selectedRecord = records.find((record) => record.key === state.selectedProductKey) || null;

      if (!records.length) {
        summaryEl.innerHTML = `No created products are available beyond structure A (<strong>${escapeHtml(sourceNode?.label || sourceNodeId)}</strong>).`;
        listEl.innerHTML = "";
      } else if (state.pathHighlight && state.selectedProductKey) {
        const overlay = state.pathHighlight;
        const pathCount = Array.isArray(overlay.paths) ? overlay.paths.length : 0;
        const targetCount = Array.isArray(overlay.targetNodeIds) ? overlay.targetNodeIds.length : 0;
        const modeNote = overlay.mode === "undirected"
          ? "No directed route was found; highlighting the nearest topology path instead."
          : "Highlighting shortest directed route(s).";
        summaryEl.innerHTML = `
          <div><strong>A:</strong> ${escapeHtml(sourceNode?.label || sourceNodeId)} (node ${escapeHtml(sourceNodeId)})</div>
          <div><strong>Product:</strong> ${escapeHtml(selectedRecord?.displayLabel || state.selectedProductKey)}</div>
          <div><strong>Matches:</strong> ${escapeHtml(targetCount)} nearest node(s), ${escapeHtml(pathCount)} shortest path(s).</div>
          <div class="muted" style="margin-top:6px;">${escapeHtml(modeNote)}</div>
        `;
      } else {
        summaryEl.innerHTML = `
          <div><strong>A:</strong> ${escapeHtml(sourceNode?.label || sourceNodeId)} (node ${escapeHtml(sourceNodeId)})</div>
          <div class="muted" style="margin-top:6px;">${escapeHtml(records.length)} product node(s) are currently reachable or present in the graph. Select one to highlight path(s) on the network.</div>
        `;
      }

      listEl.innerHTML = records.map((record) => `
        <button
          class="product-row ${record.key === state.selectedProductKey ? "active" : ""} ${record.reachable ? "" : "unreachable"}"
          type="button"
          data-product-key="${escapeHtml(record.key)}"
        >
          <span class="product-row-title">${escapeHtml(record.displayLabel)}</span>
          <span class="product-row-meta">
            ${record.reachable ? `nearest in ${escapeHtml(record.nearestDistance)} step(s)` : "no route from current A"}
          </span>
        </button>
      `).join("");

      sourceSelect.onchange = (event) => setPathSourceNode(event.target.value);
      listEl.querySelectorAll("[data-product-key]").forEach((button) => {
        button.addEventListener("click", () => selectProductPath(button.getAttribute("data-product-key") || ""));
      });
    }

    function formatJsonBlock(value) {
      const text = JSON.stringify(value || {}, null, 2);
      return `<div class="code-block">${escapeHtml(text)}</div>`;
    }

    function isMeaningfulTemplateValue(value) {
      if (value == null) return false;
      if (Array.isArray(value)) return value.length > 0;
      if (typeof value === "object") return Object.keys(value).length > 0;
      if (typeof value === "boolean") return value;
      const text = String(value).trim();
      return text !== "" && text !== "{}" && text !== "[]" && text !== "null" && text !== "None";
    }

    function cleanTemplateText(value) {
      const text = String(value ?? "").trim();
      if ((text.startsWith("'") && text.endsWith("'")) || (text.startsWith('"') && text.endsWith('"'))) {
        return text.slice(1, -1);
      }
      return text;
    }

    function formatTemplatePrimitive(value) {
      if (!isMeaningfulTemplateValue(value)) return "";
      if (typeof value === "boolean") return value ? "yes" : "no";
      return cleanTemplateText(value);
    }

    function renderTemplateList(values) {
      const items = Array.isArray(values) ? values.filter(isMeaningfulTemplateValue) : [];
      if (!items.length) return "";
      return `<ul style="margin:6px 0 0 18px;">${items.map((item) => `<li>${escapeHtml(formatTemplatePrimitive(item))}</li>`).join("")}</ul>`;
    }

    function renderTemplateKvGrid(entries) {
      const usable = entries.filter((entry) => isMeaningfulTemplateValue(entry[1]));
      if (!usable.length) return "";
      return `
        <div class="kv-grid" style="margin-bottom:12px;">
          ${usable.map(([label, value]) => `
            <div class="kv-card">
              <div class="muted">${escapeHtml(label)}</div>
              <div>${escapeHtml(formatTemplatePrimitive(value))}</div>
            </div>
          `).join("")}
        </div>
      `;
    }

    function renderTemplateConditions(conditions) {
      if (!conditions || typeof conditions !== "object") return "";
      const summary = renderTemplateKvGrid([
        ["Catalyst", conditions.catalyst],
        ["Stoichiometry", conditions.stechio],
        ["Label", conditions.label],
        ["Light required", conditions.light],
        ["Remove forward", conditions.remove_forward],
      ]);
      const contextual = renderTemplateKvGrid([
        ["Temperature", conditions.temperature],
        ["pH", conditions.pH],
        ["Solvent", conditions.solvent],
      ]);
      const doiList = Array.isArray(conditions.doi) ? conditions.doi : [];
      return `
        <div style="margin-bottom:14px;">
          <div style="font-weight:600; margin-bottom:6px;">Conditions</div>
          ${summary || contextual ? `${summary}${contextual}` : `<div class="muted">No explicit conditions were attached to this template.</div>`}
          ${doiList.length ? `
            <div style="margin-top:8px;"><strong>References</strong></div>
            ${renderTemplateList(doiList.map(cleanTemplateText))}
          ` : ""}
        </div>
      `;
    }

    function renderTemplateRules(rules) {
      if (!rules || typeof rules !== "object" || !Object.keys(rules).length) return "";
      const sections = [
        ["Avoid", rules.avoid],
        ["Enforce", rules.enforce],
        ["Avoid Formation", rules.avoid_formation],
        ["At Least One", rules.at_least_one],
      ].filter(([, value]) => isMeaningfulTemplateValue(value));
      if (!sections.length) return "";
      return `
        <div style="margin-bottom:14px;">
          <div style="font-weight:600; margin-bottom:6px;">Rules</div>
          ${sections.map(([label, value]) => `
            <details style="margin-bottom:8px;">
              <summary>${escapeHtml(label)}</summary>
              ${formatJsonBlock(value)}
            </details>
          `).join("")}
        </div>
      `;
    }

    function renderTemplateChangeSet(title, changes) {
      const rows = Array.isArray(changes) ? changes : [];
      if (!rows.length) return "";
      return `
        <div style="margin-bottom:14px;">
          <div style="font-weight:600; margin-bottom:6px;">${escapeHtml(title)}</div>
          <div style="display:grid; gap:8px;">
            ${rows.map((change, index) => {
              const entries = [
                ["Delete", change?.delete],
                ["Single", change?.single],
                ["Double", change?.double],
                ["Triple", change?.triple],
                ["Aromatic", change?.aromatic],
                ["Charges", change?.charges],
              ].filter(([, value]) => isMeaningfulTemplateValue(value));
              return `
                <div class="kv-card">
                  <div style="font-weight:600; margin-bottom:6px;">Pattern ${index + 1}</div>
                  ${entries.length
                    ? entries.map(([label, value]) => `<div style="margin-bottom:4px;"><strong>${escapeHtml(label)}:</strong> ${escapeHtml(formatTemplatePrimitive(value))}</div>`).join("")
                    : `<div class="muted">No explicit bond-change data recorded.</div>`}
                </div>
              `;
            }).join("")}
          </div>
        </div>
      `;
    }

    function renderTemplateHtml(templatePayload) {
      const templateData = templatePayload?.data || {};
      const visualizationHtml = String(templatePayload?.visualization_html || "").trim();
      if ((!templateData || typeof templateData !== "object" || !Object.keys(templateData).length) && !visualizationHtml) {
        return `<div class="muted">No reaction-template library data was available for this reaction.</div>`;
      }
      const summary = renderTemplateKvGrid([
        ["Name", templateData.name],
        ["Reactants", templateData.reactants],
        ["Products", templateData.products],
        ["Major Reactants", templateData.major_reactants],
        ["Major Products", templateData.major_products],
        ["Minor Reactants", templateData.minor_reactants],
        ["Minor Products", templateData.minor_products],
        ["Spectators", templateData.spectators],
        ["Side Reaction", templateData.side_reaction],
        ["Reacts With Itself", templateData.react_with_itself],
        ["Template File", templateData.folder],
      ]);
      const unknownFields = Object.fromEntries(
        Object.entries(templateData).filter(([key]) => ![
          "name",
          "reactants",
          "products",
          "major_reactants",
          "major_products",
          "minor_reactants",
          "minor_products",
          "spectators",
          "side_reaction",
          "react_with_itself",
          "folder",
          "conditions",
          "rules",
          "changes_react_to_prod",
          "changes_prod_to_react",
          "MG",
        ].includes(key))
      );
      return `
        ${visualizationHtml ? `
          <div style="margin-bottom:16px;">
            <div style="font-weight:600; margin-bottom:6px;">Template Render</div>
            <div class="mol-card">${visualizationHtml}</div>
          </div>
        ` : ""}
        <div style="margin-bottom:14px;">
          <div style="font-weight:600; margin-bottom:6px;">Template Summary</div>
          ${summary || `<div class="muted">No high-level summary fields were attached to this template.</div>`}
        </div>
        ${renderTemplateConditions(templateData.conditions)}
        ${renderTemplateRules(templateData.rules)}
        ${renderTemplateChangeSet("Reactant → Product Changes", templateData.changes_react_to_prod)}
        ${renderTemplateChangeSet("Product → Reactant Changes", templateData.changes_prod_to_react)}
        ${Object.keys(unknownFields).length ? `
          <details>
            <summary>Additional Template Data</summary>
            ${formatJsonBlock(unknownFields)}
          </details>
        ` : ""}
      `;
    }

    function renderInputsPanels(snapshot) {
      const summary = document.getElementById("inputs-summary");
      const pathmin = document.getElementById("pathmin-config-panel");
      if (!summary || !pathmin) return;
      if (!snapshot || !snapshot.initialized || !snapshot.drive) {
        summary.innerHTML = `<div class="muted">No initialized workspace. Fill in the starting geometries here or load an existing workspace.</div>`;
        pathmin.innerHTML = `<div class="muted">Load or initialize a workspace to inspect the current path minimization inputs.</div>`;
        return;
      }
      const inputs = snapshot.drive.inputs || {};
      summary.innerHTML = `
        <div class="kv-grid">
          <div class="kv-card"><div class="muted">Inputs File</div><div>${escapeHtml(inputs.path || "Unavailable")}</div></div>
          <div class="kv-card"><div class="muted">Path Method</div><div>${escapeHtml(inputs.path_min_method || "Unavailable")}</div></div>
          <div class="kv-card"><div class="muted">Engine</div><div>${escapeHtml(inputs.engine_name || "Unavailable")}</div></div>
          <div class="kv-card"><div class="muted">Program</div><div>${escapeHtml(inputs.program || "Unavailable")}</div></div>
        </div>
        ${inputs.error ? `<div style="margin-top:12px; color: var(--warn);">${escapeHtml(inputs.error)}</div>` : ""}
      `;
      pathmin.innerHTML = `
        <div class="kv-grid" style="margin-bottom:12px;">
          <div class="kv-card"><div class="muted">Path Minimization Method</div><div>${escapeHtml(inputs.path_min_method || "Unavailable")}</div></div>
          <div class="kv-card"><div class="muted">Engine</div><div>${escapeHtml(inputs.engine_name || "Unavailable")}</div></div>
          <div class="kv-card"><div class="muted">Program</div><div>${escapeHtml(inputs.program || "Unavailable")}</div></div>
          <div class="kv-card"><div class="muted">Inputs File</div><div>${escapeHtml(inputs.path || "Unavailable")}</div></div>
        </div>
        ${inputs.error ? `<div style="margin-bottom:12px; color: var(--warn);">${escapeHtml(inputs.error)}</div>` : ""}
        <div style="display:grid; gap:12px;">
          <div>
            <div style="font-weight:600; margin-bottom:6px;">Path Minimization Inputs</div>
            ${formatJsonBlock(inputs.path_min_inputs || {})}
          </div>
          <div>
            <div style="font-weight:600; margin-bottom:6px;">Chain Inputs</div>
            ${formatJsonBlock(inputs.chain_inputs || {})}
          </div>
          <div>
            <div style="font-weight:600; margin-bottom:6px;">Geodesic Interpolation Inputs</div>
            ${formatJsonBlock(inputs.gi_inputs || {})}
          </div>
          <div>
            <div style="font-weight:600; margin-bottom:6px;">Optimizer Inputs</div>
            ${formatJsonBlock(inputs.optimizer_kwds || {})}
          </div>
          <div>
            <div style="font-weight:600; margin-bottom:6px;">Program Inputs</div>
            ${formatJsonBlock(inputs.program_kwds || {})}
          </div>
        </div>
      `;
    }

    function renderLogging(snapshot) {
      const consoleEl = document.getElementById("log-console");
      const stateEl = document.getElementById("log-state");
      const statusEl = document.getElementById("log-status");
      if (!consoleEl || !stateEl || !statusEl) return;
      const activity = snapshot?.live_activity || null;
      const consoleText = activity?.console_text || snapshot?.last_error || snapshot?.last_message || "No log output yet.";
      consoleEl.textContent = consoleText;
      statusEl.innerHTML = `
        <strong>Status:</strong> ${escapeHtml(snapshot?.busy ? "Busy" : "Idle")}
        <div class="muted" style="margin-top:6px;">${escapeHtml(snapshot?.last_message || "Idle.")}</div>
        ${snapshot?.last_error ? `<div style="margin-top:6px; color: var(--warn);">${escapeHtml(snapshot.last_error)}</div>` : ""}
      `;
      stateEl.textContent = JSON.stringify({
        busy: snapshot?.busy || false,
        active_action: snapshot?.active_action || null,
        drive: snapshot?.drive ? {
          workspace: snapshot.drive.workspace || {},
          queue: snapshot.drive.queue || {},
          inputs: snapshot.drive.inputs || {},
        } : null,
      }, null, 2);
    }

    function renderKmcPanel(snapshot) {
      const panel = document.getElementById("kmc-panel");
      const temperatureInput = document.getElementById("kmc-temperature");
      const finalTimeInput = document.getElementById("kmc-final-time");
      const maxStepsInput = document.getElementById("kmc-max-steps");
      const initialConditionsInput = document.getElementById("kmc-initial-conditions");
      if (!panel || !temperatureInput || !finalTimeInput || !maxStepsInput || !initialConditionsInput) return;
      if (!snapshot || !snapshot.initialized || !snapshot.drive) {
        panel.innerHTML = `<div class="muted">Initialize or load a workspace before running kinetics.</div>`;
        return;
      }
      const kmc = snapshot.drive.kmc || {};
      if (kmc.available && !temperatureInput.dataset.synced) {
        temperatureInput.value = String(Number(kmc.temperature_kelvin || 298.15));
        finalTimeInput.value = kmc.default_end_time == null ? "" : String(kmc.default_end_time);
        maxStepsInput.value = String(Number(kmc.default_max_steps || 200));
        initialConditionsInput.value = JSON.stringify(
          Object.fromEntries((kmc.nodes || []).filter((node) => Number(node.initial || 0) !== 0).map((node) => [String(node.id), Number(node.initial)])),
          null,
          2
        );
        temperatureInput.dataset.synced = "true";
      }
      const result = state.kmcResult;
      const defaultsHtml = Array.isArray(kmc.nodes) && kmc.nodes.length
        ? `<div class="code-block">${escapeHtml(JSON.stringify(kmc.nodes, null, 2))}</div>`
        : `<div class="muted">No kinetics-ready NEB barriers are available yet on this network.</div>`;
      const resultHtml = result
        ? `
          <div class="kv-grid" style="margin-bottom:12px;">
            <div class="kv-card"><strong>Events</strong><div>${escapeHtml(result.event_count)}</div></div>
            <div class="kv-card"><strong>Final Time</strong><div>${escapeHtml(Number(result.final_time).toFixed(6))}</div></div>
            <div class="kv-card"><strong>Dominant Node</strong><div>${escapeHtml(result.dominant_node?.label || "n/a")}</div></div>
          </div>
          ${drawKmcPlotSvg(result.plot)}
          <div style="margin-top:12px;"><strong>Final Populations</strong></div>
          <div class="code-block">${escapeHtml(JSON.stringify(result.summary, null, 2))}</div>
        `
        : `<div class="muted">Run the kinetic model to see time evolution over the current NEB-backed network.</div>`;
      panel.innerHTML = `
        <div class="kv-grid" style="margin-bottom:12px;">
          <div class="kv-card"><strong>Kinetics Nodes</strong><div>${escapeHtml(Number(kmc.node_count || 0))}</div></div>
          <div class="kv-card"><strong>Kinetics Edges</strong><div>${escapeHtml(Number(kmc.edge_count || 0))}</div></div>
          <div class="kv-card"><strong>Suppressed Edges</strong><div>${escapeHtml(Number(kmc.suppressed_edge_count || 0))}</div></div>
        </div>
        <div style="margin-bottom:12px;"><strong>Default Kinetics State</strong></div>
        ${defaultsHtml}
        <div style="margin-top:12px;"><strong>Kinetics Result</strong></div>
        ${resultHtml}
      `;
    }

    function renderNetworkToolbar() {
      const toolbar = document.getElementById("network-toolbar");
      if (!toolbar) return;
      const selection = state.selected;
      if (!selection) {
        toolbar.innerHTML = `
          <div class="network-toolbar-title">Network Actions</div>
          <div class="muted">Select a node or edge to see actions here.</div>
          <div class="network-toolbar-actions">
            <button class="network-tool-button ${state.manualNodeInsertMode ? "active" : ""}" data-drive-action="toolbar-insert-node" title="Insert node from XYZ at next click" onclick="toggleManualNodeInsertMode()" ${state.manualNodeRequestInFlight || state.snapshot?.busy ? "disabled" : ""}>◎</button>
          </div>
        `;
        return;
      }
      if (selection.kind === "node") {
        const node = selection.node;
        const connectActive = Number(state.connectSourceNodeId) === Number(node.id);
        const connectDisabled = state.manualEdgeRequestInFlight ? "disabled" : "";
        toolbar.innerHTML = `
          <div class="network-toolbar-title">Node ${escapeHtml(node.id)} Actions</div>
          <div class="muted">${state.manualEdgeRequestInFlight ? "Waiting for the current manual edge request to finish." : connectActive ? "Click a second node to create an edge from this source." : "Node tools are available directly from the graph and via right-click context menu."}</div>
          <div class="network-toolbar-actions">
            <button class="network-tool-button" data-drive-action="toolbar-minimize-node" title="Minimize geometry" onclick="queueMinimizeNode(${Number(node.id)})" ${node.minimizable ? "" : "disabled"}>↓</button>
            <button class="network-tool-button" data-drive-action="toolbar-apply-node" title="Apply reaction templates" onclick="queueApplyReactions(${Number(node.id)})" ${node.can_apply_reactions ? "" : "disabled"}>+</button>
            <button class="network-tool-button" data-drive-action="toolbar-nanoreactor-node" title="Run nanoreactor sampling" onclick="queueNanoreactor(${Number(node.id)})" ${node.can_nanoreactor ? "" : "disabled"}>⊕</button>
            <button class="network-tool-button" data-drive-action="toolbar-hessian-node" title="Run Hessian minima explorer" onclick="queueHessianSampleFromNode(${Number(node.id)})" ${node.can_hessian_sample ? "" : "disabled"}>*</button>
            <button class="network-tool-button ${connectActive ? "active" : ""}" data-drive-action="toolbar-connect-node" title="Connect to new node" onclick="beginConnectMode(${Number(node.id)})" ${connectDisabled}>→</button>
            <button class="network-tool-button ${state.manualNodeInsertMode ? "active" : ""}" data-drive-action="toolbar-insert-node" title="Insert node from XYZ at next click" onclick="toggleManualNodeInsertMode()" ${state.manualNodeRequestInFlight || state.snapshot?.busy ? "disabled" : ""}>◎</button>
          </div>
        `;
        return;
      }

      const edge = selection.edge;
      toolbar.innerHTML = `
        <div class="network-toolbar-title">Edge ${escapeHtml(edge.source)} → ${escapeHtml(edge.target)} Actions</div>
        <div class="muted">Queue work on both endpoints or launch an autosplitting NEB for this edge.</div>
        <div class="network-toolbar-actions">
          <button class="network-tool-button" data-drive-action="toolbar-minimize-edge" title="Minimize both endpoint geometries" onclick="queueMinimizePair(${Number(edge.source)}, ${Number(edge.target)})">↓↓</button>
          <button class="network-tool-button" data-drive-action="toolbar-hessian-edge" title="Run Hessian minima explorer from edge peak" onclick="queueHessianSampleFromEdge(${Number(edge.source)}, ${Number(edge.target)})" ${edge.can_hessian_sample ? "" : "disabled"}>*</button>
          <button class="network-tool-button" data-drive-action="toolbar-neb-edge" title="Queue NEB minimization" onclick="queueEdgeNeb(${Number(edge.source)}, ${Number(edge.target)})" ${edge.can_queue_neb ? "" : "disabled"}>#</button>
          <button class="network-tool-button ${state.manualNodeInsertMode ? "active" : ""}" data-drive-action="toolbar-insert-node" title="Insert node from XYZ at next click" onclick="toggleManualNodeInsertMode()" ${state.manualNodeRequestInFlight || state.snapshot?.busy ? "disabled" : ""}>◎</button>
        </div>
      `;
    }

    function drawLinePlotSvg(plot, ariaLabel) {
      const xVals = Array.isArray(plot?.x) ? plot.x : [];
      const yVals = Array.isArray(plot?.y) ? plot.y : [];
      const title = plot?.title || plot?.caption || "";
      const xLabel = plot?.x_label || "Step";
      const yLabel = plot?.y_label || "Energy";
      if (!xVals.length || !yVals.length || xVals.length !== yVals.length) {
        return `<div class="muted">Waiting for trajectory data...</div>`;
      }
      const width = 760;
      const height = 280;
      const margin = { top: 28, right: 18, bottom: 36, left: 64 };
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const minX = Math.min(...xVals);
      const maxX = Math.max(...xVals) === minX ? minX + 1 : Math.max(...xVals);
      const minY0 = Math.min(...yVals);
      const maxY0 = Math.max(...yVals);
      const minY = minY0 === maxY0 ? minY0 - 1 : minY0;
      const maxY = minY0 === maxY0 ? maxY0 + 1 : maxY0;
      const sx = (x) => margin.left + ((x - minX) / (maxX - minX)) * innerWidth;
      const sy = (y) => margin.top + (1 - (y - minY) / (maxY - minY)) * innerHeight;
      const yTicks = Array.from({ length: 5 }, (_, i) => minY + ((maxY - minY) * i / 4));
      const points = xVals.map((x, i) => `${sx(x).toFixed(2)},${sy(yVals[i]).toFixed(2)}`).join(" ");
      return `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(ariaLabel || title || "Trajectory plot")}">
          ${yTicks.map((tick) => `
            <g>
              <line x1="${margin.left}" y1="${sy(tick)}" x2="${width - margin.right}" y2="${sy(tick)}" stroke="rgba(142,163,194,0.18)" stroke-dasharray="3 3" />
              <line x1="${margin.left - 6}" y1="${sy(tick)}" x2="${margin.left}" y2="${sy(tick)}" stroke="rgba(201,216,240,0.78)" />
              <text x="${margin.left - 10}" y="${sy(tick) + 4}" text-anchor="end" font-size="11" fill="#9eb4d6">${escapeHtml(Number(tick).toFixed(2))}</text>
            </g>
          `).join("")}
          <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="rgba(201,216,240,0.78)" />
          <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="rgba(201,216,240,0.78)" />
          <polyline fill="none" stroke="#7ef0c7" stroke-width="2.5" points="${points}" />
          ${xVals.map((x, i) => `<circle cx="${sx(x)}" cy="${sy(yVals[i])}" r="3.5" fill="#63d5ff" />`).join("")}
          <text x="${width / 2}" y="18" text-anchor="middle" font-size="14" fill="#eef4ff">${escapeHtml(title || "Trajectory plot")}</text>
          <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#9eb4d6">${escapeHtml(xLabel)}</text>
          <text x="16" y="${height / 2}" transform="rotate(-90 16 ${height / 2})" text-anchor="middle" font-size="12" fill="#9eb4d6">${escapeHtml(yLabel)}</text>
        </svg>
      `;
    }

    function drawKmcPlotSvg(plot) {
      const xVals = Array.isArray(plot?.x) ? plot.x : [];
      const series = Array.isArray(plot?.series) ? plot.series : [];
      if (!xVals.length || !series.length) {
        return `<div class="muted">Run the kinetic model to populate the trajectory plot.</div>`;
      }
      const yVals = series.flatMap((item) => Array.isArray(item.y) ? item.y : []);
      if (!yVals.length) {
        return `<div class="muted">Run the kinetic model to populate the trajectory plot.</div>`;
      }
      const width = 760;
      const height = 280;
      const margin = { top: 28, right: 18, bottom: 36, left: 64 };
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const minX = Math.min(...xVals);
      const maxX = Math.max(...xVals) === minX ? minX + 1 : Math.max(...xVals);
      const minY = 0.0;
      const maxY = Math.max(1.0, ...yVals);
      const sx = (x) => margin.left + ((x - minX) / (maxX - minX)) * innerWidth;
      const sy = (y) => margin.top + (1 - (y - minY) / (maxY - minY)) * innerHeight;
      const colors = ["#7ef0c7", "#63d5ff", "#9cafff", "#ff8eb0", "#ffd166", "#59d8b6"];
      return `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Kinetic model population trajectories">
          <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="rgba(201,216,240,0.78)" />
          <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="rgba(201,216,240,0.78)" />
          ${series.map((item, index) => {
            const pts = (item.y || []).map((y, i) => `${sx(xVals[i]).toFixed(2)},${sy(y).toFixed(2)}`).join(" ");
            return `<polyline fill="none" stroke="${colors[index % colors.length]}" stroke-width="2.4" points="${pts}" />`;
          }).join("")}
          <text x="${width / 2}" y="18" text-anchor="middle" font-size="14" fill="#eef4ff">${escapeHtml(plot?.title || "Population vs time")}</text>
          <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#9eb4d6">${escapeHtml(plot?.x_label || "Time")}</text>
          <text x="16" y="${height / 2}" transform="rotate(-90 16 ${height / 2})" text-anchor="middle" font-size="12" fill="#9eb4d6">${escapeHtml(plot?.y_label || "Population")}</text>
        </svg>
        <div class="muted" style="margin-top:10px;">
          ${series.map((item, index) => `<span class="badge" style="border-color:${colors[index % colors.length]}; color:${colors[index % colors.length]}; margin-right:6px;">${escapeHtml(item.label || item.node_id)}</span>`).join("")}
        </div>
      `;
    }

    function drawNebHistorySvg(activity) {
      const history = Array.isArray(activity?.history) ? activity.history : [];
      const current = activity?.plot || null;
      if (!history.length && !current) {
        return `<div class="muted">Waiting for the first NEB optimization update...</div>`;
      }
      const curves = history.length ? history : (current ? [current] : []);
      const xVals = curves.flatMap((curve) => Array.isArray(curve.x) ? curve.x : []);
      const yVals = curves.flatMap((curve) => Array.isArray(curve.y) ? curve.y : []);
      if (!xVals.length || !yVals.length) {
        return `<div class="muted">Waiting for the first NEB optimization update...</div>`;
      }
      const width = 760;
      const height = 280;
      const margin = { top: 28, right: 18, bottom: 36, left: 64 };
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const minX = Math.min(...xVals);
      const maxX = Math.max(...xVals) === minX ? minX + 1 : Math.max(...xVals);
      const minY0 = Math.min(...yVals);
      const maxY0 = Math.max(...yVals);
      const minY = minY0 === maxY0 ? minY0 - 1 : minY0;
      const maxY = minY0 === maxY0 ? maxY0 + 1 : maxY0;
      const sx = (x) => margin.left + ((x - minX) / (maxX - minX)) * innerWidth;
      const sy = (y) => margin.top + (1 - (y - minY) / (maxY - minY)) * innerHeight;
      const yTicks = Array.from({ length: 5 }, (_, i) => minY + ((maxY - minY) * i / 4));
      const backgroundCurves = curves.slice(0, -1);
      const foreground = curves[curves.length - 1];
      return `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Live NEB optimization history">
          ${yTicks.map((tick) => `
            <g>
              <line x1="${margin.left}" y1="${sy(tick)}" x2="${width - margin.right}" y2="${sy(tick)}" stroke="rgba(142,163,194,0.18)" stroke-dasharray="3 3" />
              <line x1="${margin.left - 6}" y1="${sy(tick)}" x2="${margin.left}" y2="${sy(tick)}" stroke="rgba(201,216,240,0.78)" />
              <text x="${margin.left - 10}" y="${sy(tick) + 4}" text-anchor="end" font-size="11" fill="#9eb4d6">${escapeHtml(Number(tick).toFixed(2))}</text>
            </g>
          `).join("")}
          <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="rgba(201,216,240,0.78)" />
          <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="rgba(201,216,240,0.78)" />
          ${backgroundCurves.map((curve) => {
            const pts = (curve.x || []).map((x, i) => `${sx(x).toFixed(2)},${sy(curve.y[i]).toFixed(2)}`).join(" ");
            return `<polyline fill="none" stroke="rgba(142,163,194,0.28)" stroke-width="1.4" opacity="0.4" points="${pts}" />`;
          }).join("")}
          ${foreground ? `<polyline fill="none" stroke="#7ef0c7" stroke-width="2.8" points="${(foreground.x || []).map((x, i) => `${sx(x).toFixed(2)},${sy(foreground.y[i]).toFixed(2)}`).join(" ")}" />` : ""}
          ${foreground ? (foreground.x || []).map((x, i) => `<circle cx="${sx(x)}" cy="${sy(foreground.y[i])}" r="3.2" fill="#63d5ff" />`).join("") : ""}
          <text x="${width / 2}" y="18" text-anchor="middle" font-size="14" fill="#eef4ff">${escapeHtml(activity?.title || "Live NEB optimization history")}</text>
          <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#9eb4d6">Integrated path length</text>
          <text x="16" y="${height / 2}" transform="rotate(-90 16 ${height / 2})" text-anchor="middle" font-size="12" fill="#9eb4d6">Energy</text>
        </svg>
      `;
    }

    function renderNebAsciiMonitors(activity) {
      const monitors = activity?.monitors && typeof activity.monitors === "object"
        ? activity.monitors
        : {};
      const monitorEntries = Object.entries(monitors);
      if (!monitorEntries.length) {
        if (activity?.ascii_plot) {
          return `
            <div class="muted" style="margin-top:10px;">Live chain monitor</div>
            <pre>${escapeHtml(activity.ascii_plot)}</pre>
          `;
        }
        return "";
      }
      const rows = monitorEntries
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
        .map(([monitorId, monitorPayload]) => {
          const payload = monitorPayload || {};
          const caption = String(payload.caption || payload.status_message || "").trim();
          const ascii = String(payload.ascii_plot || "").trim();
          return `
            <div class="job-row running">
              <div><strong>${escapeHtml(monitorId)}</strong>${caption ? ` <span class="badge">${escapeHtml(caption)}</span>` : ""}</div>
              ${ascii ? `<pre style="margin-top:8px;">${escapeHtml(ascii)}</pre>` : `<div class="muted" style="margin-top:8px;">Waiting for first chain update...</div>`}
            </div>
          `;
        })
        .join("");
      return `
        <div class="muted" style="margin-top:10px;">Parallel branch monitors</div>
        <div class="job-list">${rows}</div>
      `;
    }

    function drawGrowthGraphSvg(activity) {
      const payload = activity?.network || {};
      const nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
      const edges = Array.isArray(payload.edges) ? payload.edges : [];
      if (!nodes.length) {
        return `<div class="muted">Waiting for the first growth update...</div>`;
      }

      const width = 760;
      const height = 280;
      const levels = new Map();
      const parentsByNode = new Map();
      const childrenByNode = new Map();
      edges.forEach((edge) => {
        const source = Number(edge.source);
        const target = Number(edge.target);
        const parents = parentsByNode.get(target) || [];
        parents.push(source);
        parentsByNode.set(target, parents);
        const children = childrenByNode.get(source) || [];
        children.push(target);
        childrenByNode.set(source, children);
      });

      const depthMemo = new Map();
      const nodeIds = nodes.map((node) => Number(node.id));
      if (nodeIds.includes(0)) {
        depthMemo.set(0, 0);
      }
      nodeIds.forEach((nodeId) => {
        const parents = parentsByNode.get(nodeId) || [];
        if (!parents.length && !depthMemo.has(nodeId)) {
          depthMemo.set(nodeId, 1);
        }
      });
      const queue = Array.from(depthMemo.keys());
      while (queue.length) {
        const parentId = Number(queue.shift());
        const parentDepth = Number(depthMemo.get(parentId) || 0);
        const children = childrenByNode.get(parentId) || [];
        children.forEach((childId) => {
          const nextDepth = parentDepth + 1;
          const current = depthMemo.get(childId);
          if (current == null || nextDepth < current) {
            depthMemo.set(childId, nextDepth);
            queue.push(childId);
          }
        });
      }
      nodeIds.forEach((nodeId) => {
        if (depthMemo.has(nodeId)) return;
        const parents = (parentsByNode.get(nodeId) || []).filter((parentId) => depthMemo.has(parentId));
        if (parents.length) {
          depthMemo.set(
            nodeId,
            1 + Math.min(...parents.map((parentId) => Number(depthMemo.get(parentId) || 0))),
          );
        } else {
          depthMemo.set(nodeId, 1);
        }
      });

      nodes.forEach((node) => {
        const depth = Number(depthMemo.get(Number(node.id)) || 1);
        const row = levels.get(depth) || [];
        row.push(node);
        levels.set(depth, row);
      });

      const maxDepth = Math.max(...Array.from(levels.keys()));
      const xForDepth = (depth) => 70 + (maxDepth === 0 ? 0 : (depth / Math.max(maxDepth, 1)) * (width - 140));
      const positions = new Map();
      Array.from(levels.entries()).sort((a, b) => a[0] - b[0]).forEach(([depth, levelNodes]) => {
        const count = levelNodes.length;
        levelNodes.sort((a, b) => Number(a.id) - Number(b.id)).forEach((node, index) => {
          const y = count === 1 ? height / 2 : 40 + (index / Math.max(count - 1, 1)) * (height - 80);
          positions.set(Number(node.id), { x: xForDepth(depth), y });
        });
      });

      return `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Live reaction-network growth">
          ${edges.map((edge) => {
            const source = positions.get(Number(edge.source));
            const target = positions.get(Number(edge.target));
            if (!source || !target) return "";
            return `<line x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" stroke="rgba(142,163,194,0.38)" stroke-width="2" />`;
          }).join("")}
          ${nodes.map((node) => {
            const pos = positions.get(Number(node.id));
            if (!pos) return "";
            return `
              <g transform="translate(${pos.x},${pos.y})">
                <circle r="18" fill="${node.growing ? "#ffd166" : (Number(node.id) === 0 ? "#63d5ff" : "#7d94bb")}" stroke="#eef4ff" stroke-width="2.5"></circle>
                ${node.growing ? `
                  <circle r="24" fill="none" stroke="#7ef0c7" stroke-width="3" stroke-dasharray="8 6" stroke-linecap="round">
                    <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="0" to="360" dur="1.2s" repeatCount="indefinite"/>
                  </circle>
                ` : ""}
                <text y="4" text-anchor="middle" font-size="12" fill="#08111f">${escapeHtml(node.label || node.id)}</text>
              </g>
            `;
          }).join("")}
        </svg>
      `;
    }

    function renderLiveActivityContent(activity) {
      if (!activity) return "";
      if (activity.type === "growth") {
        return `
          <div style="font-weight:600; margin-bottom:8px;">Reaction-Network Growth</div>
          <div class="muted" style="margin-bottom:10px;">${escapeHtml(activity.title || "Growing Retropaths network")}</div>
          ${drawGrowthGraphSvg(activity)}
          <div class="muted" style="margin-top:10px;">${escapeHtml(activity.note || "")}</div>
        `;
      }
      if (activity.type === "minimize") {
        const jobs = Array.isArray(activity.jobs) ? activity.jobs : [];
        return `
          <div style="font-weight:600; margin-bottom:8px;">Geometry Optimization Monitor</div>
          <div class="muted" style="margin-bottom:10px;">${escapeHtml(activity.title || "Running geometry minimizations")}</div>
          ${activity.plot ? drawLinePlotSvg(activity.plot, "Geometry optimization trajectory") : `<div class="muted">${escapeHtml(activity.note || "Waiting for geometry optimization updates...")}</div>`}
          <div class="muted" style="margin-top:10px;">${escapeHtml(activity.note || "")}</div>
          <div class="job-list">
            ${jobs.map((job) => `
              <div class="job-row ${escapeHtml(job.status || "pending")}">
                <div><strong>Node ${escapeHtml(job.node_id)}</strong> <span class="badge">${escapeHtml(job.status || "pending")}</span></div>
                ${job.error ? `<div style="color:var(--warn); margin-top:4px;">${escapeHtml(job.error)}</div>` : ""}
              </div>
            `).join("")}
          </div>
        `;
      }
      return `
        <div style="font-weight:600; margin-bottom:8px;">NEB Optimization Monitor</div>
        <div class="muted" style="margin-bottom:10px;">${escapeHtml(activity.title || "Running autosplitting NEB")}</div>
        <div class="live-neb-layout">
          <div>
            <div style="margin-bottom:6px;"><strong>Reactant</strong></div>
            ${renderMoleculeCard(activity.reactant_structure)}
          </div>
          <div>${drawNebHistorySvg(activity)}</div>
          <div>
            <div style="margin-bottom:6px;"><strong>Product</strong></div>
            ${renderMoleculeCard(activity.product_structure)}
          </div>
        </div>
        <div class="muted" style="margin-top:10px;">${escapeHtml(activity.note || "")}</div>
        ${renderNebAsciiMonitors(activity)}
        ${activity.console_text ? `<pre>${escapeHtml(activity.console_text)}</pre>` : ""}
      `;
    }

    function renderLiveActivity(snapshot) {
      const panel = document.getElementById("live-activity-panel");
      const inline = document.getElementById("live-activity-inline");
      const activity = snapshot?.live_activity || state.pendingLiveActivity || null;
      if (!panel || !inline) return;
      if (!activity) {
        panel.style.display = "none";
        panel.innerHTML = "";
        inline.style.display = "none";
        inline.innerHTML = "";
        return;
      }
      const content = renderLiveActivityContent(activity);
      panel.style.display = "block";
      panel.innerHTML = content;
      inline.style.display = "block";
      inline.innerHTML = content;
    }

    function renderDetail(selection) {
      const targeted = document.getElementById("panel-targeted");
      const templateData = document.getElementById("panel-template-data");
      const structures = document.getElementById("panel-structures");
      const title = document.getElementById("detail-title");
      const summary = document.getElementById("detail-summary");

      if (!selection) {
        title.textContent = "Select a node or edge";
        summary.textContent = "Click a node to inspect its geometry or click an edge to inspect the targeted reaction, template data, and queue NEB work.";
        targeted.innerHTML = "";
        templateData.innerHTML = "";
        structures.innerHTML = "";
        renderNetworkToolbar();
        return;
      }

      if (selection.kind === "node") {
        const node = selection.node;
        title.textContent = `Node ${node.id}`;
        summary.innerHTML = `
          <div><strong>Label:</strong> ${escapeHtml(node.label || node.id)}</div>
          <div><strong>Endpoint optimized:</strong> ${node.endpoint_optimized ? "yes" : "no"}</div>
          ${node.endpoint_optimization_error ? `<div><strong>Last optimization error:</strong> <span style="color:var(--warn);">${escapeHtml(node.endpoint_optimization_error)}</span></div>` : ""}
          <div><strong>Can queue minimization:</strong> ${node.minimizable ? "yes" : "no"}</div>
          <div><strong>Can apply reactions:</strong> ${node.can_apply_reactions ? "yes" : "no"}</div>
          <div><strong>Can run nanoreactor:</strong> ${node.can_nanoreactor ? "yes" : "no"}</div>
          <div><strong>Can run Hessian sample:</strong> ${node.can_hessian_sample ? "yes" : "no"}</div>
          ${node.minimize_note ? `<div><strong>Minimization note:</strong> ${escapeHtml(node.minimize_note)}</div>` : ""}
          ${node.apply_reactions_note ? `<div><strong>Reaction note:</strong> ${escapeHtml(node.apply_reactions_note)}</div>` : ""}
          ${node.nanoreactor_note ? `<div><strong>Nanoreactor note:</strong> ${escapeHtml(node.nanoreactor_note)}</div>` : ""}
          ${node.hessian_sample_note ? `<div><strong>Hessian note:</strong> ${escapeHtml(node.hessian_sample_note)}</div>` : ""}
          <div><strong>NEB-backed:</strong> ${node.neb_backed ? "yes" : "no"}</div>
        `;
        targeted.innerHTML = `
          <div style="margin-bottom:10px;"><button class="secondary" data-drive-action="minimize-node" onclick="queueMinimizeNode(${Number(node.id)})" ${node.minimizable ? "" : "disabled"}>Queue Minimization For This Geometry</button></div>
          <div style="margin-bottom:10px;"><button class="secondary" data-drive-action="apply-reactions" onclick="queueApplyReactions(${Number(node.id)})" ${node.can_apply_reactions ? "" : "disabled"}>Apply Reactions To This Node</button></div>
          <div style="margin-bottom:10px;"><button class="secondary" data-drive-action="nanoreactor" onclick="queueNanoreactor(${Number(node.id)})" ${node.can_nanoreactor ? "" : "disabled"}>Run Nanoreactor Sampling From This Geometry</button></div>
          <div style="margin-bottom:10px; display:grid; grid-template-columns:minmax(0, 140px) minmax(0, 180px) minmax(0, 220px) minmax(0, 1fr); gap:8px; align-items:end;">
            <div>
              <label for="hessian-sample-dr">Hessian sample dr</label>
              <input id="hessian-sample-dr" type="number" step="0.01" min="0.0001" value="${escapeHtml(state.hessianSampleDr)}" />
            </div>
            <div>
              <label for="hessian-sample-max-candidates">Max minimizations</label>
              <input id="hessian-sample-max-candidates" type="number" step="1" min="1" value="${escapeHtml(state.hessianSampleMaxCandidates)}" />
            </div>
            <div>
              <label for="hessian-sample-use-bigchem" style="display:flex; align-items:center; gap:8px;">
                <input id="hessian-sample-use-bigchem" type="checkbox" ${state.hessianSampleUseBigchem ? "checked" : ""} />
                <span>Force BigChem (QCOP override)</span>
              </label>
            </div>
            <button class="secondary" data-drive-action="hessian-sample-node" onclick="queueHessianSampleFromNode(${Number(node.id)})" ${node.can_hessian_sample ? "" : "disabled"}>Run Hessian Sample From This Geometry</button>
          </div>
          <div style="margin-bottom:10px;"><button class="secondary" type="button" onclick="setPathSourceNode(${Number(node.id)})">Use As Path Source A</button></div>
          <div style="margin-bottom:10px; display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px;">
            <button class="secondary" onclick="setManualEdgeEndpoint('source', ${Number(node.id)})">Use As Manual Edge Source</button>
            <button class="secondary" onclick="setManualEdgeEndpoint('target', ${Number(node.id)})">Use As Manual Edge Target</button>
          </div>
          ${node.minimize_note ? `<div style="margin-bottom:10px; color:${node.minimizable ? "var(--muted)" : "var(--warn)"};">${escapeHtml(node.minimize_note)}</div>` : ""}
          ${node.apply_reactions_note ? `<div style="margin-bottom:10px; color:${node.can_apply_reactions ? "var(--muted)" : "var(--warn)"};">${escapeHtml(node.apply_reactions_note)}</div>` : ""}
          ${node.nanoreactor_note ? `<div style="margin-bottom:10px; color:${node.can_nanoreactor ? "var(--muted)" : "var(--warn)"};">${escapeHtml(node.nanoreactor_note)}</div>` : ""}
          ${node.hessian_sample_note ? `<div style="margin-bottom:10px; color:${node.can_hessian_sample ? "var(--muted)" : "var(--warn)"};">${escapeHtml(node.hessian_sample_note)}</div>` : ""}
          <pre>${escapeHtml(JSON.stringify(node.data || {}, null, 2))}</pre>
        `;
        templateData.innerHTML = `<pre>${escapeHtml(JSON.stringify(node.data || {}, null, 2))}</pre>`;
        structures.innerHTML = node.structure?.xyz_b64
          ? `<iframe class="structure" srcdoc="${escapeHtml(makeStructureSrcdoc(node.structure.xyz_b64))}"></iframe>`
          : `<div class="muted">No 3D structure is available for this node.</div>`;
        bindHessianSampleDrInput();
        renderNetworkToolbar();
        return;
      }

      const edge = selection.edge;
      const template = edge.template || {};
      const hasTemplateData = Boolean(template.data && Object.keys(template.data).length);
      const hasNebResultData = edge.barrier != null || edge.viewer_href || Number(edge.chains || 0) > 0 || edge.neb_backed;
      title.textContent = `Edge ${edge.source} → ${edge.target}`;
      summary.innerHTML = `
        <div><strong>Reaction:</strong> ${escapeHtml(edge.reaction || "Unknown")}</div>
        <div><strong>Queue status:</strong> ${escapeHtml(edge.queue_status || "not queued")}</div>
        <div><strong>NEB-backed:</strong> ${edge.neb_backed ? "yes" : "no"}</div>
        <div><strong>Can run Hessian sample:</strong> ${edge.can_hessian_sample ? "yes" : "no"}</div>
        ${edge.result_from_reverse_edge ? `<div><strong>Displayed NEB result:</strong> reverse-directed edge</div>` : ""}
        ${edge.result_from_completed_queue ? `<div><strong>Displayed NEB result:</strong> completed queue attempt</div>` : ""}
        ${edge.queue_note ? `<div><strong>Queue note:</strong> ${escapeHtml(edge.queue_note)}</div>` : ""}
        ${edge.hessian_sample_note ? `<div><strong>Hessian note:</strong> ${escapeHtml(edge.hessian_sample_note)}</div>` : ""}
        ${edge.barrier == null ? "" : `<div><strong>Barrier:</strong> ${escapeHtml(Number(edge.barrier).toFixed(3))}</div>`}
      `;
      targeted.innerHTML = `
        <div style="margin-bottom:10px;">
          <button class="primary" data-drive-action="run-neb" onclick="queueEdgeNeb(${Number(edge.source)}, ${Number(edge.target)})" ${edge.can_queue_neb ? "" : "disabled"}>Queue Autosplitting NEB For This Edge</button>
        </div>
        <div style="margin-bottom:10px; display:grid; grid-template-columns:minmax(0, 140px) minmax(0, 180px) minmax(0, 220px) minmax(0, 1fr); gap:8px; align-items:end;">
          <div>
            <label for="hessian-sample-dr">Hessian sample dr</label>
            <input id="hessian-sample-dr" type="number" step="0.01" min="0.0001" value="${escapeHtml(state.hessianSampleDr)}" />
          </div>
          <div>
            <label for="hessian-sample-max-candidates">Max minimizations</label>
            <input id="hessian-sample-max-candidates" type="number" step="1" min="1" value="${escapeHtml(state.hessianSampleMaxCandidates)}" />
          </div>
          <div>
            <label for="hessian-sample-use-bigchem" style="display:flex; align-items:center; gap:8px;">
              <input id="hessian-sample-use-bigchem" type="checkbox" ${state.hessianSampleUseBigchem ? "checked" : ""} />
              <span>Force BigChem (QCOP override)</span>
            </label>
          </div>
          <button class="secondary" data-drive-action="hessian-sample-edge" onclick="queueHessianSampleFromEdge(${Number(edge.source)}, ${Number(edge.target)})" ${edge.can_hessian_sample ? "" : "disabled"}>Run Hessian Sample From Edge Peak</button>
        </div>
        <div style="margin-bottom:10px;">
          <strong>Targeted reaction:</strong> ${escapeHtml(edge.reaction || "Unknown")}
        </div>
        ${edge.result_from_reverse_edge ? `<div style="margin-bottom:10px;" class="muted">Showing completed NEB data reconstructed from the reverse-directed edge because this directed edge does not carry the chain payload directly.</div>` : ""}
        ${edge.result_from_completed_queue ? `<div style="margin-bottom:10px;" class="muted">Showing NEB data from the completed attempted pair because autosplitting did not leave a direct annotated edge for this exact selection.</div>` : ""}
        ${edge.queue_note ? `<div style="margin-bottom:10px; color: ${edge.can_queue_neb ? "var(--muted)" : "var(--warn)"};"><strong>${edge.can_queue_neb ? "Queue note" : "Edge cannot run as-is"}:</strong> ${escapeHtml(edge.queue_note)}</div>` : ""}
        ${edge.hessian_sample_note ? `<div style="margin-bottom:10px; color: ${edge.can_hessian_sample ? "var(--muted)" : "var(--warn)"};"><strong>${edge.can_hessian_sample ? "Hessian note" : "Hessian sampling unavailable"}:</strong> ${escapeHtml(edge.hessian_sample_note)}</div>` : ""}
        ${edge.viewer_href ? `<div style="margin-bottom:10px;"><a href="${escapeHtml(edge.viewer_href)}" target="_blank" rel="noreferrer">Open completed NEB viewer</a></div>` : ""}
        <pre>${escapeHtml(JSON.stringify(edge.data || {}, null, 2))}</pre>
      `;
      templateData.innerHTML = `
        ${hasNebResultData ? `
          <div class="kv-grid" style="margin-bottom:12px;">
            <div class="kv-card"><strong>Barrier</strong><div>${edge.barrier == null ? "n/a" : escapeHtml(Number(edge.barrier).toFixed(3))}</div></div>
            <div class="kv-card"><strong>NEB Chains</strong><div>${escapeHtml(Number(edge.chains || 0))}</div></div>
            <div class="kv-card"><strong>Queue Status</strong><div>${escapeHtml(edge.queue_status || "not queued")}</div></div>
          </div>
          ${edge.viewer_href ? `<div style="margin-bottom:12px;"><strong>Viewer:</strong> <a href="${escapeHtml(edge.viewer_href)}" target="_blank" rel="noreferrer">Open completed NEB viewer</a></div>` : ""}
          <pre style="margin-bottom:12px;">${escapeHtml(JSON.stringify(edge.data || {}, null, 2))}</pre>
        ` : `<div class="muted" style="margin-bottom:12px;">No NEB-derived edge data is available yet for this edge.</div>`}
        ${hasTemplateData
          ? renderTemplateHtml(template)
          : `<div class="muted">No reaction-template library data was available for this reaction.</div>`}
      `;
      structures.innerHTML = `
        <div class="viewer-grid">
          <div>
            <div style="margin-bottom:6px;"><strong>Source geometry</strong></div>
            ${edge.source_structure?.xyz_b64 ? `<iframe class="structure" srcdoc="${escapeHtml(makeStructureSrcdoc(edge.source_structure.xyz_b64))}"></iframe>` : `<div class="muted">No structure available.</div>`}
          </div>
          <div>
            <div style="margin-bottom:6px;"><strong>Target geometry</strong></div>
            ${edge.target_structure?.xyz_b64 ? `<iframe class="structure" srcdoc="${escapeHtml(makeStructureSrcdoc(edge.target_structure.xyz_b64))}"></iframe>` : `<div class="muted">No structure available.</div>`}
          </div>
        </div>
      `;
      bindHessianSampleDrInput();
      renderNetworkToolbar();
    }

    function selectToolTab(groupName, tabName) {
      document.querySelectorAll(`[data-tool-tab="${groupName}"]`).forEach((button) => {
        button.classList.toggle("active", button.getAttribute("data-tool-target") === tabName);
      });
      document.querySelectorAll(`[id^="tool-panel-${groupName}-"]`).forEach((panel) => {
        panel.classList.toggle("active", panel.id === `tool-panel-${groupName}-${tabName}`);
      });
    }

    function selectDetailTab(tabName) {
      document.querySelectorAll(".detail-tab").forEach((button) => {
        button.classList.toggle("active", button.getAttribute("data-tab") === tabName);
      });
      document.querySelectorAll(".detail-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `panel-${tabName}`);
      });
    }

    document.querySelectorAll(".detail-tab").forEach((button) => {
      button.addEventListener("click", () => selectDetailTab(button.getAttribute("data-tab")));
    });

    document.querySelectorAll(".tool-tab").forEach((button) => {
      button.addEventListener("click", () => selectToolTab(
        button.getAttribute("data-tool-tab"),
        button.getAttribute("data-tool-target"),
      ));
    });

    function hideNetworkContextMenu() {
      const menu = document.getElementById("network-context-menu");
      if (!menu) return;
      menu.classList.remove("visible");
      state.contextMenuGraphPoint = null;
    }

    function applyNetworkViewTransform() {
      const canvas = state.networkCanvas;
      const viewport = canvas?.viewport;
      if (!viewport) return;
      const view = state.networkView;
      viewport.setAttribute(
        "transform",
        `translate(${view.x.toFixed(3)},${view.y.toFixed(3)}) scale(${view.scale.toFixed(4)})`,
      );
    }

    function networkSvgCoordsFromEvent(event) {
      const canvas = state.networkCanvas;
      if (!canvas?.svg) return { x: 0, y: 0 };
      const bounds = canvas.svg.getBoundingClientRect();
      if (!bounds.width || !bounds.height) {
        return { x: canvas.width / 2, y: canvas.height / 2 };
      }
      return {
        x: ((event.clientX - bounds.left) / bounds.width) * canvas.width,
        y: ((event.clientY - bounds.top) / bounds.height) * canvas.height,
      };
    }

    function zoomNetworkAt(factor, centerX = null, centerY = null) {
      const canvas = state.networkCanvas;
      if (!canvas) return;
      const view = state.networkView;
      const oldScale = Number(view.scale || 1);
      const nextScale = Math.max(
        Number(view.minScale || 0.35),
        Math.min(Number(view.maxScale || 4.5), oldScale * Number(factor || 1)),
      );
      if (!Number.isFinite(nextScale) || nextScale === oldScale) return;
      const cx = centerX == null ? canvas.width / 2 : Number(centerX);
      const cy = centerY == null ? canvas.height / 2 : Number(centerY);
      const ratio = nextScale / oldScale;
      view.x = cx - ratio * (cx - Number(view.x || 0));
      view.y = cy - ratio * (cy - Number(view.y || 0));
      view.scale = nextScale;
      applyNetworkViewTransform();
    }

    function resetNetworkView() {
      state.networkView.x = 0;
      state.networkView.y = 0;
      state.networkView.scale = 1;
      applyNetworkViewTransform();
    }

    function buildNetworkContextActions(selection) {
      const actions = [];
      const manualNodeXyz = String(document.getElementById("manual-node-xyz")?.value || "").trim();
      const canToggleManualNode = Boolean(manualNodeXyz) && !state.manualNodeRequestInFlight && !state.snapshot?.busy;
      const canInsertManualNode = canToggleManualNode && state.contextMenuGraphPoint != null;
      if (selection?.kind === "node") {
        const node = selection.node;
        const nodeId = Number(node.id);
        actions.push({
          label: `Minimize node ${nodeId}`,
          enabled: Boolean(node.minimizable),
          note: node.minimize_note || "This node cannot be minimized from its current state.",
          run: () => queueMinimizeNode(nodeId),
        });
        actions.push({
          label: `Apply reactions on node ${nodeId}`,
          enabled: Boolean(node.can_apply_reactions),
          note: node.apply_reactions_note || "Reaction-template application is unavailable for this node.",
          run: () => queueApplyReactions(nodeId),
        });
        actions.push({
          label: `Run nanoreactor from node ${nodeId}`,
          enabled: Boolean(node.can_nanoreactor),
          note: node.nanoreactor_note || "Nanoreactor sampling is unavailable for this node.",
          run: () => queueNanoreactor(nodeId),
        });
        actions.push({
          label: `Run Hessian sample from node ${nodeId}`,
          enabled: Boolean(node.can_hessian_sample),
          note: node.hessian_sample_note || "Hessian sampling is unavailable for this node.",
          run: () => queueHessianSampleFromNode(nodeId),
        });
        actions.push({
          label: `Connect from node ${nodeId}`,
          enabled: !state.manualEdgeRequestInFlight,
          note: "A manual edge request is already in progress.",
          run: () => beginConnectMode(nodeId),
        });
      } else if (selection?.kind === "edge") {
        const edge = selection.edge;
        const source = Number(edge.source);
        const target = Number(edge.target);
        actions.push({
          label: `Minimize edge endpoints (${source} and ${target})`,
          enabled: true,
          note: "",
          run: () => queueMinimizePair(source, target),
        });
        actions.push({
          label: `Run Hessian sample from edge ${source} -> ${target}`,
          enabled: Boolean(edge.can_hessian_sample),
          note: edge.hessian_sample_note || "Hessian sampling is unavailable for this edge.",
          run: () => queueHessianSampleFromEdge(source, target),
        });
        actions.push({
          label: `Queue NEB for edge ${source} -> ${target}`,
          enabled: Boolean(edge.can_queue_neb),
          note: edge.queue_note || "This edge cannot be queued from its current state.",
          run: () => queueEdgeNeb(source, target),
        });
      }
      actions.push({
        label: state.manualNodeInsertMode ? "Cancel click-to-insert node mode" : "Enable click-to-insert node mode",
        enabled: canToggleManualNode,
        note: manualNodeXyz
          ? (state.snapshot?.busy ? "Wait for the current background action to finish first." : "A manual node insertion request is already in progress.")
          : "Paste XYZ text in the Manual Edge tab before enabling node insertion mode.",
        run: () => {
          if (state.manualNodeInsertMode) {
            setManualNodeInsertMode(false);
            setSubtext("Node insert mode cleared.");
          } else {
            setManualNodeInsertMode(true);
            setSubtext("Node insert mode active. Click empty space in the network canvas to place the node.");
          }
        },
      });
      actions.push({
        label: "Insert node here from XYZ",
        enabled: canInsertManualNode,
        note: !manualNodeXyz
          ? "Paste XYZ text in the Manual Edge tab before inserting nodes."
          : (
            state.contextMenuGraphPoint == null
              ? "Right-click inside the network canvas to pick an insertion point."
              : (state.snapshot?.busy ? "Wait for the current background action to finish first." : "A manual node insertion request is already in progress.")
          ),
        run: () => {
          const point = state.contextMenuGraphPoint;
          if (!point) return;
          void insertManualNodeAtGraphCoords(Number(point.x), Number(point.y));
        },
      });
      actions.push({
        label: "Zoom in",
        enabled: true,
        note: "",
        run: () => zoomNetworkAt(1.18),
      });
      actions.push({
        label: "Zoom out",
        enabled: true,
        note: "",
        run: () => zoomNetworkAt(1 / 1.18),
      });
      actions.push({
        label: "Reset view",
        enabled: true,
        note: "",
        run: () => resetNetworkView(),
      });
      return actions;
    }

    function renderNetworkContextMenu(event, selectionOverride = null) {
      const menu = document.getElementById("network-context-menu");
      const canvas = state.networkCanvas;
      const host = canvas?.svg?.closest(".network-canvas-shell");
      if (!menu || !host) return;
      const contextCoords = networkSvgCoordsFromEvent(event);
      state.contextMenuGraphPoint = canvasCoordsToGraphCoords(contextCoords.x, contextCoords.y);
      const selection = selectionOverride || state.selected;
      const title = selection?.kind === "node"
        ? `Node ${Number(selection.node.id)} tools`
        : selection?.kind === "edge"
          ? `Edge ${Number(selection.edge.source)} -> ${Number(selection.edge.target)} tools`
          : "Network tools";
      const actions = buildNetworkContextActions(selection);
      menu.innerHTML = "";
      const heading = document.createElement("div");
      heading.className = "network-context-title";
      heading.textContent = title;
      menu.appendChild(heading);
      actions.forEach((action) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = action.label;
        button.disabled = !action.enabled;
        if (!action.enabled && action.note) {
          button.title = action.note;
        }
        button.addEventListener("click", () => {
          hideNetworkContextMenu();
          action.run();
        });
        menu.appendChild(button);
      });
      menu.classList.add("visible");
      const bounds = host.getBoundingClientRect();
      const rawX = event.clientX - bounds.left;
      const rawY = event.clientY - bounds.top;
      const maxX = Math.max(10, bounds.width - menu.offsetWidth - 10);
      const maxY = Math.max(10, bounds.height - menu.offsetHeight - 10);
      const clampedX = Math.max(10, Math.min(maxX, rawX));
      const clampedY = Math.max(10, Math.min(maxY, rawY));
      menu.style.left = `${clampedX}px`;
      menu.style.top = `${clampedY}px`;
    }

    function bindNetworkInteractionHandlers(svg) {
      if (!svg || svg.dataset.interactionsBound === "true") return;
      svg.dataset.interactionsBound = "true";
      svg.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        renderNetworkContextMenu(event);
      });
      svg.addEventListener("click", (event) => {
        if (!state.manualNodeInsertMode || state.manualNodeRequestInFlight) return;
        const target = event.target;
        if (!(target instanceof Element)) return;
        if (
          target.closest(".network-node")
          || target.closest(".network-edge-hitbox")
          || target.closest(".network-edge-line")
          || target.closest(".network-label")
        ) {
          return;
        }
        const canvasCoords = networkSvgCoordsFromEvent(event);
        const graphCoords = canvasCoordsToGraphCoords(canvasCoords.x, canvasCoords.y);
        event.preventDefault();
        hideNetworkContextMenu();
        setManualNodeInsertMode(false);
        void insertManualNodeAtGraphCoords(graphCoords.x, graphCoords.y);
      });
      svg.addEventListener("wheel", (event) => {
        event.preventDefault();
        hideNetworkContextMenu();
        const coords = networkSvgCoordsFromEvent(event);
        zoomNetworkAt(event.deltaY < 0 ? 1.12 : 1 / 1.12, coords.x, coords.y);
      }, { passive: false });
      svg.addEventListener("mousedown", (event) => {
        if (event.button !== 0) return;
        const target = event.target;
        if (!(target instanceof Element)) return;
        if (
          target.closest(".network-node")
          || target.closest(".network-edge-hitbox")
          || target.closest(".network-edge-line")
          || target.closest(".network-label")
        ) {
          return;
        }
        if (state.manualNodeInsertMode) {
          hideNetworkContextMenu();
          event.preventDefault();
          return;
        }
        const canvas = state.networkCanvas;
        if (!canvas) return;
        canvas.panning = true;
        canvas.panOrigin = {
          mouseX: event.clientX,
          mouseY: event.clientY,
          viewX: Number(state.networkView.x || 0),
          viewY: Number(state.networkView.y || 0),
        };
        svg.classList.add("is-panning");
        hideNetworkContextMenu();
        event.preventDefault();
      });
      window.addEventListener("mousemove", (event) => {
        const canvas = state.networkCanvas;
        if (!canvas?.panning || !canvas.panOrigin || !canvas.svg) return;
        const bounds = canvas.svg.getBoundingClientRect();
        if (!bounds.width || !bounds.height) return;
        const dx = ((event.clientX - canvas.panOrigin.mouseX) / bounds.width) * canvas.width;
        const dy = ((event.clientY - canvas.panOrigin.mouseY) / bounds.height) * canvas.height;
        if (Math.abs(dx) + Math.abs(dy) > 1.5) {
          state.networkView.suppressClickUntil = Date.now() + 150;
        }
        state.networkView.x = canvas.panOrigin.viewX + dx;
        state.networkView.y = canvas.panOrigin.viewY + dy;
        applyNetworkViewTransform();
      });
      window.addEventListener("mouseup", (event) => {
        const canvas = state.networkCanvas;
        if (!canvas?.panning || event.button !== 0) return;
        canvas.panning = false;
        canvas.panOrigin = null;
        if (canvas.svg) {
          canvas.svg.classList.remove("is-panning");
        }
      });
      document.addEventListener("click", (event) => {
        const menu = document.getElementById("network-context-menu");
        if (!menu) return;
        if (menu.contains(event.target)) return;
        hideNetworkContextMenu();
      });
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          hideNetworkContextMenu();
          if (state.manualNodeInsertMode) {
            setManualNodeInsertMode(false);
            setSubtext("Node insert mode cleared.");
          }
        }
      });
    }

    function renderNetwork(snapshot) {
      const svg = document.getElementById("network-svg");
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      hideNetworkContextMenu();
      if (!snapshot || !snapshot.initialized || !snapshot.drive?.network) {
        state.networkCanvas = null;
        renderDetail(null);
        return;
      }

      const payload = snapshot.drive.network;
      const nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
      const edges = Array.isArray(payload.edges) ? payload.edges : [];
      const pendingEdge = state.pendingEdgeAddition;
      if (pendingEdge && edges.some((edge) => Number(edge.source) === Number(pendingEdge.source) && Number(edge.target) === Number(pendingEdge.target))) {
        state.pendingEdgeAddition = null;
      }
      if (!nodes.length) {
        state.networkCanvas = null;
        renderDetail(null);
        return;
      }

      function computeTreeNetworkLayout(nodes, edges) {
        const clamp = (value, lower, upper) => Math.max(lower, Math.min(upper, value));
        const parentsByNode = new Map();
        const childrenByNode = new Map();
        const nodeIds = nodes.map((node) => Number(node.id));
        const edgeSignature = edges
          .map((edge) => `${Number(edge.source)}>${Number(edge.target)}`)
          .sort()
          .join(",");
        const layoutVersion = `${nodeIds.slice().sort((a, b) => a - b).join(",")}::${edgeSignature}`;
        const previousPositions = state.networkNodePositions && typeof state.networkNodePositions === "object"
          ? state.networkNodePositions
          : {};
        edges.forEach((edge) => {
          const source = Number(edge.source);
          const target = Number(edge.target);
          const parents = parentsByNode.get(target) || [];
          parents.push(source);
          parentsByNode.set(target, parents);
          const children = childrenByNode.get(source) || [];
          children.push(target);
          childrenByNode.set(source, children);
        });

        const depthByNode = new Map();
        const rootId = nodeIds.includes(0) ? 0 : (nodeIds[0] ?? 0);
        const queue = [rootId];
        depthByNode.set(rootId, 0);
        while (queue.length) {
          const parentId = queue.shift();
          const parentDepth = Number(depthByNode.get(parentId) || 0);
          const children = (childrenByNode.get(parentId) || []).slice().sort((a, b) => a - b);
          children.forEach((childId) => {
            const nextDepth = parentDepth + 1;
            const current = depthByNode.get(childId);
            if (current == null || nextDepth < current) {
              depthByNode.set(childId, nextDepth);
              queue.push(childId);
            }
          });
        }

        const pending = nodeIds.filter((nodeId) => !depthByNode.has(nodeId)).sort((a, b) => a - b);
        pending.forEach((nodeId) => {
          const parents = (parentsByNode.get(nodeId) || []).filter((parentId) => depthByNode.has(parentId));
          if (parents.length) {
            depthByNode.set(nodeId, 1 + Math.min(...parents.map((parentId) => Number(depthByNode.get(parentId) || 0))));
            return;
          }
          depthByNode.set(nodeId, 1 + Math.max(...Array.from(depthByNode.values(), (value) => Number(value || 0))));
        });

        const levelMap = new Map();
        nodes.forEach((node) => {
          const depth = Number(depthByNode.get(Number(node.id)) || 0);
          const row = levelMap.get(depth) || [];
          row.push(node);
          levelMap.set(depth, row);
        });

        const levelDepths = Array.from(levelMap.keys()).sort((a, b) => a - b);
        const orderByNode = new Map();
        levelDepths.forEach((depth) => {
          const levelNodes = levelMap.get(depth) || [];
          if (depth === 0) {
            levelNodes.sort((a, b) => Number(a.id) - Number(b.id));
          } else {
            levelNodes.sort((a, b) => {
              const aParents = (parentsByNode.get(Number(a.id)) || []).filter((parentId) => orderByNode.has(parentId));
              const bParents = (parentsByNode.get(Number(b.id)) || []).filter((parentId) => orderByNode.has(parentId));
              const aCenter = aParents.length
                ? aParents.reduce((sum, parentId) => sum + Number(orderByNode.get(parentId) || 0), 0) / aParents.length
                : Number(a.id);
              const bCenter = bParents.length
                ? bParents.reduce((sum, parentId) => sum + Number(orderByNode.get(parentId) || 0), 0) / bParents.length
                : Number(b.id);
              if (aCenter !== bCenter) return aCenter - bCenter;
              return Number(a.id) - Number(b.id);
            });
          }
          levelNodes.forEach((node, index) => orderByNode.set(Number(node.id), index));
        });

        const maxDepth = Math.max(0, ...levelDepths);
        const maxLevelCount = Math.max(1, ...Array.from(levelMap.values(), (row) => row.length));
        const width = Math.max(1180, 260 + maxDepth * 230);
        const height = Math.max(680, 180 + maxLevelCount * 94);
        const xForDepth = (depth) => 86 + (maxDepth === 0 ? 0 : (depth / maxDepth) * (width - 172));
        const positions = new Map();
        const levelBands = new Map();

        levelDepths.forEach((depth) => {
          const levelNodes = (levelMap.get(depth) || []).slice();
          const count = levelNodes.length;
          const usableHeight = Math.max(1, height - 120);
          const gap = count <= 1 ? 0 : Math.min(110, usableHeight / Math.max(count - 1, 1));
          const blockHeight = gap * Math.max(count - 1, 0);
          const startY = (height - blockHeight) / 2;
          levelBands.set(depth, { startY, gap, count });
          levelNodes.forEach((node, index) => {
            const nodeId = Number(node.id);
            const saved = previousPositions[String(nodeId)] || previousPositions[nodeId];
            const savedX = Number(saved?.x);
            const savedY = Number(saved?.y);
            positions.set(Number(node.id), {
              x: Number.isFinite(savedX) ? clamp(savedX, 44, width - 44) : xForDepth(depth),
              y: Number.isFinite(savedY) ? clamp(savedY, 44, height - 44) : (count === 1 ? height / 2 : startY + index * gap),
            });
          });
        });

        const hessianGroups = new Map();
        nodes.forEach((node) => {
          const nodeId = Number(node.id);
          if (!Number.isFinite(nodeId)) return;
          const data = node && typeof node.data === "object" ? node.data : {};
          if (String(data.generated_by || "") !== "hessian_sample") return;
          const provenance = data.hessian_provenance_edge;
          if (!Array.isArray(provenance) || provenance.length < 2) return;
          const source = Number(provenance[0]);
          const target = Number(provenance[1]);
          if (!Number.isFinite(source) || !Number.isFinite(target)) return;
          const depth = Number(depthByNode.get(nodeId) || 0);
          const key = `${depth}|${Math.min(source, target)}|${Math.max(source, target)}`;
          const group = hessianGroups.get(key) || [];
          group.push({ nodeId, source, target });
          hessianGroups.set(key, group);
        });

        hessianGroups.forEach((group, key) => {
          if (!Array.isArray(group) || group.length <= 1) return;
          const [depthText] = String(key || "").split("|");
          const depth = Number(depthText || 0);
          const band = levelBands.get(depth);
          const yMin = band ? Number(band.startY || 0) : 44;
          const yMax = band
            ? Number((band.startY || 0) + Math.max(0, (band.count - 1) * (band.gap || 0)))
            : (height - 44);
          const sorted = group.slice().sort((left, right) => left.nodeId - right.nodeId);
          const center = (sorted.length - 1) / 2;
          const fanStep = Math.min(64, 34 + sorted.length * 2);
          const xSpread = Math.min(56, 22 + sorted.length * 2);

          sorted.forEach((item, index) => {
            const sourcePos = positions.get(Number(item.source));
            const targetPos = positions.get(Number(item.target));
            const currentPos = positions.get(Number(item.nodeId));
            if (!sourcePos || !targetPos || !currentPos) return;
            const rank = index - center;
            const dx = Number(targetPos.x) - Number(sourcePos.x);
            const dy = Number(targetPos.y) - Number(sourcePos.y);
            const dist = Math.max(1, Math.hypot(dx, dy));
            const normalX = -dy / dist;
            const normalY = dx / dist;
            const anchorY = (Number(sourcePos.y) + Number(targetPos.y)) / 2;
            const nextX = Number(currentPos.x) + rank * xSpread + normalX * rank * fanStep;
            const nextY = anchorY + normalY * rank * fanStep;
            positions.set(Number(item.nodeId), {
              x: clamp(nextX, 44, width - 44),
              y: clamp(nextY, yMin, yMax || (height - 44)),
            });
          });
        });

        const uniqueLinks = [];
        const uniqueLinkKeys = new Set();
        edges.forEach((edge) => {
          const source = Number(edge.source);
          const target = Number(edge.target);
          if (!Number.isFinite(source) || !Number.isFinite(target) || source === target) return;
          const key = source < target ? `${source}|${target}` : `${target}|${source}`;
          if (uniqueLinkKeys.has(key)) return;
          uniqueLinkKeys.add(key);
          uniqueLinks.push([source, target]);
        });

        if (nodeIds.length > 1) {
          const widthInner = Math.max(1, width - 88);
          const heightInner = Math.max(1, height - 88);
          const area = widthInner * heightInner;
          const k = Math.max(8, Math.sqrt(area / Math.max(1, nodeIds.length)));
          const largeGraph = nodeIds.length > 120;
          const iterations = largeGraph
            ? Math.max(8, Math.min(24, 8 + Math.floor(nodeIds.length / 28)))
            : Math.max(24, Math.min(84, 20 + nodeIds.length));
          const repulsionPairBudget = largeGraph
            ? Math.min(2600, Math.max(400, nodeIds.length * 12))
            : 0;
          let temperature = Math.max(14, Math.min(width, height) * 0.11);
          const basePositions = new Map();
          nodeIds.forEach((nodeId) => {
            const current = positions.get(nodeId);
            if (!current) return;
            basePositions.set(nodeId, { x: Number(current.x), y: Number(current.y) });
          });

          for (let iter = 0; iter < iterations; iter += 1) {
            const disp = new Map();
            nodeIds.forEach((nodeId) => disp.set(nodeId, { x: 0, y: 0 }));

            if (largeGraph) {
              const count = nodeIds.length;
              for (let pair = 0; pair < repulsionPairBudget; pair += 1) {
                const i = (pair * 37 + iter * 19) % count;
                const j = (pair * 53 + iter * 29 + 1) % count;
                if (i === j) continue;
                const idA = nodeIds[i];
                const idB = nodeIds[j];
                const posA = positions.get(idA);
                const posB = positions.get(idB);
                if (!posA || !posB) continue;
                let dx = Number(posA.x) - Number(posB.x);
                let dy = Number(posA.y) - Number(posB.y);
                let dist = Math.hypot(dx, dy);
                if (!Number.isFinite(dist) || dist < 0.01) {
                  const jitter = ((i + 1) * (j + 3) * 17 + iter * 13) % 13;
                  const angle = (jitter / 13) * Math.PI * 2;
                  dx = Math.cos(angle) * 0.5;
                  dy = Math.sin(angle) * 0.5;
                  dist = Math.hypot(dx, dy);
                }
                const nx = dx / dist;
                const ny = dy / dist;
                const repulse = Math.min(280, (k * k) / Math.max(dist, 1.0));
                const dispA = disp.get(idA);
                const dispB = disp.get(idB);
                if (!dispA || !dispB) continue;
                dispA.x += nx * repulse;
                dispA.y += ny * repulse;
                dispB.x -= nx * repulse;
                dispB.y -= ny * repulse;
              }
            } else {
              for (let i = 0; i < nodeIds.length; i += 1) {
                const idA = nodeIds[i];
                const posA = positions.get(idA);
                if (!posA) continue;
                for (let j = i + 1; j < nodeIds.length; j += 1) {
                  const idB = nodeIds[j];
                  const posB = positions.get(idB);
                  if (!posB) continue;
                  let dx = Number(posA.x) - Number(posB.x);
                  let dy = Number(posA.y) - Number(posB.y);
                  let dist = Math.hypot(dx, dy);
                  if (!Number.isFinite(dist) || dist < 0.01) {
                    const jitter = ((i + 1) * (j + 3) * 17 + iter * 13) % 13;
                    const angle = (jitter / 13) * Math.PI * 2;
                    dx = Math.cos(angle) * 0.5;
                    dy = Math.sin(angle) * 0.5;
                    dist = Math.hypot(dx, dy);
                  }
                  const nx = dx / dist;
                  const ny = dy / dist;
                  const repulse = Math.min(340, (k * k) / Math.max(dist, 1.0));
                  const dispA = disp.get(idA);
                  const dispB = disp.get(idB);
                  dispA.x += nx * repulse;
                  dispA.y += ny * repulse;
                  dispB.x -= nx * repulse;
                  dispB.y -= ny * repulse;
                }
              }
            }

            uniqueLinks.forEach(([sourceId, targetId]) => {
              const sourcePos = positions.get(sourceId);
              const targetPos = positions.get(targetId);
              if (!sourcePos || !targetPos) return;
              const dx = Number(sourcePos.x) - Number(targetPos.x);
              const dy = Number(sourcePos.y) - Number(targetPos.y);
              const dist = Math.max(0.01, Math.hypot(dx, dy));
              const nx = dx / dist;
              const ny = dy / dist;
              const attract = Math.min(360, (dist * dist) / Math.max(1.0, k));
              const sourceDisp = disp.get(sourceId);
              const targetDisp = disp.get(targetId);
              sourceDisp.x -= nx * attract;
              sourceDisp.y -= ny * attract;
              targetDisp.x += nx * attract;
              targetDisp.y += ny * attract;
            });

            nodeIds.forEach((nodeId) => {
              const pos = positions.get(nodeId);
              const base = basePositions.get(nodeId) || pos;
              const force = disp.get(nodeId);
              if (!pos || !base || !force) return;
              if (nodeId === rootId) {
                pos.x = Number(base.x);
                pos.y = Number(base.y);
                return;
              }
              const depth = Number(depthByNode.get(nodeId) || 0);
              const depthAnchorX = xForDepth(depth);
              force.x += (depthAnchorX - Number(pos.x)) * 0.21;
              force.y += (Number(base.y) - Number(pos.y)) * 0.1;
              force.x += (width / 2 - Number(pos.x)) * 0.008;
              force.y += (height / 2 - Number(pos.y)) * 0.008;
              const dispMag = Math.max(1e-6, Math.hypot(force.x, force.y));
              const step = Math.min(temperature, dispMag);
              const nextX = Number(pos.x) + (force.x / dispMag) * step;
              const nextY = Number(pos.y) + (force.y / dispMag) * step;
              pos.x = clamp(nextX, 44, width - 44);
              pos.y = clamp(nextY, 44, height - 44);
            });

            temperature = Math.max(0.8, temperature * 0.94);
          }
        }

        return { width, height, positions, layoutVersion };
      }

      const layout = computeTreeNetworkLayout(nodes, edges);
      const width = layout.width;
      const height = layout.height;
      state.networkLayoutVersion = String(layout.layoutVersion || "");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const make = (tag) => document.createElementNS(ns, tag);
      const viewport = make("g");
      svg.appendChild(viewport);
      state.networkCanvas = {
        svg,
        viewport,
        width,
        height,
        panning: false,
        panOrigin: null,
      };
      bindNetworkInteractionHandlers(svg);
      applyNetworkViewTransform();
      const nodeElems = new Map();
      const edgeElems = [];
      const simNodes = nodes.map((node) => ({
        id: Number(node.id),
        node,
        x: Number(layout.positions.get(Number(node.id))?.x || width / 2),
        y: Number(layout.positions.get(Number(node.id))?.y || height / 2),
      }));
      const simById = new Map(simNodes.map((node) => [node.id, node]));
      const sourceBuckets = new Map();
      const targetBuckets = new Map();
      const pairBuckets = new Map();
      edges.forEach((edge, renderIndex) => {
        const source = Number(edge.source);
        const target = Number(edge.target);
        const sourceEntries = sourceBuckets.get(source) || [];
        sourceEntries.push({ edge, renderIndex });
        sourceBuckets.set(source, sourceEntries);
        const targetEntries = targetBuckets.get(target) || [];
        targetEntries.push({ edge, renderIndex });
        targetBuckets.set(target, targetEntries);
        const pairKey = source <= target ? `${source}|${target}` : `${target}|${source}`;
        const pairEntries = pairBuckets.get(pairKey) || [];
        pairEntries.push({ edge, renderIndex });
        pairBuckets.set(pairKey, pairEntries);
      });
      const sourceRankByEdge = new Map();
      const sourceCountByEdge = new Map();
      sourceBuckets.forEach((entries) => {
        entries.sort((left, right) => {
          const leftTarget = Number(left.edge.target);
          const rightTarget = Number(right.edge.target);
          if (leftTarget !== rightTarget) return leftTarget - rightTarget;
          const leftReaction = String(left.edge.reaction || "");
          const rightReaction = String(right.edge.reaction || "");
          if (leftReaction !== rightReaction) return leftReaction.localeCompare(rightReaction);
          return left.renderIndex - right.renderIndex;
        });
        const center = (entries.length - 1) / 2;
        entries.forEach((entry, index) => {
          sourceRankByEdge.set(entry.edge, index - center);
          sourceCountByEdge.set(entry.edge, entries.length);
        });
      });
      const targetRankByEdge = new Map();
      const targetCountByEdge = new Map();
      targetBuckets.forEach((entries) => {
        entries.sort((left, right) => {
          const leftSource = Number(left.edge.source);
          const rightSource = Number(right.edge.source);
          if (leftSource !== rightSource) return leftSource - rightSource;
          const leftReaction = String(left.edge.reaction || "");
          const rightReaction = String(right.edge.reaction || "");
          if (leftReaction !== rightReaction) return leftReaction.localeCompare(rightReaction);
          return left.renderIndex - right.renderIndex;
        });
        const center = (entries.length - 1) / 2;
        entries.forEach((entry, index) => {
          targetRankByEdge.set(entry.edge, index - center);
          targetCountByEdge.set(entry.edge, entries.length);
        });
      });
      const pairRankByEdge = new Map();
      const pairCountByEdge = new Map();
      pairBuckets.forEach((entries) => {
        entries.sort((left, right) => {
          const leftSource = Number(left.edge.source);
          const rightSource = Number(right.edge.source);
          if (leftSource !== rightSource) return leftSource - rightSource;
          const leftTarget = Number(left.edge.target);
          const rightTarget = Number(right.edge.target);
          if (leftTarget !== rightTarget) return leftTarget - rightTarget;
          const leftReaction = String(left.edge.reaction || "");
          const rightReaction = String(right.edge.reaction || "");
          if (leftReaction !== rightReaction) return leftReaction.localeCompare(rightReaction);
          return left.renderIndex - right.renderIndex;
        });
        const center = (entries.length - 1) / 2;
        entries.forEach((entry, index) => {
          pairRankByEdge.set(entry.edge, index - center);
          pairCountByEdge.set(entry.edge, entries.length);
        });
      });
      const curvatureByEdge = new Map();
      edges.forEach((edge, renderIndex) => {
        const pairCount = Number(pairCountByEdge.get(edge) || 1);
        const sourceCount = Number(sourceCountByEdge.get(edge) || 1);
        const targetCount = Number(targetCountByEdge.get(edge) || 1);
        let curvature = 0;
        if (pairCount > 1) {
          curvature += Number(pairRankByEdge.get(edge) || 0) * 34;
        }
        if (sourceCount > 1) {
          curvature += Number(sourceRankByEdge.get(edge) || 0) * 12;
        }
        if (targetCount > 1) {
          curvature -= Number(targetRankByEdge.get(edge) || 0) * 12;
        }
        if (Math.abs(curvature) < 6 && edges.length > 1) {
          const seed = (Number(edge.source) * 131 + Number(edge.target) * 37 + renderIndex * 17) % 2 === 0 ? -1 : 1;
          curvature = seed * 8;
        }
        curvatureByEdge.set(edge, curvature);
      });

      function edgePathD(source, target, curvature) {
        const sx = Number(source?.x || 0);
        const sy = Number(source?.y || 0);
        const tx = Number(target?.x || 0);
        const ty = Number(target?.y || 0);
        const dx = tx - sx;
        const dy = ty - sy;
        const distance = Math.max(1, Math.hypot(dx, dy));
        const normalX = -dy / distance;
        const normalY = dx / distance;
        const maxBend = Math.max(16, Math.min(88, distance * 0.42));
        const bend = Math.max(-maxBend, Math.min(maxBend, Number(curvature || 0)));
        const cx = (sx + tx) / 2 + normalX * bend;
        const cy = (sy + ty) / 2 + normalY * bend;
        return `M ${sx.toFixed(2)} ${sy.toFixed(2)} Q ${cx.toFixed(2)} ${cy.toFixed(2)} ${tx.toFixed(2)} ${ty.toFixed(2)}`;
      }

      function applyNetworkDecorations(selection) {
        const overlay = state.pathHighlight || null;
        const highlightedNodeIds = new Set(Array.isArray(overlay?.pathNodeIds) ? overlay.pathNodeIds.map((value) => Number(value)) : []);
        const targetNodeIds = new Set(Array.isArray(overlay?.targetNodeIds) ? overlay.targetNodeIds.map((value) => Number(value)) : []);
        const edgePairs = new Set(Array.isArray(overlay?.edgePairs) ? overlay.edgePairs : []);
        edgeElems.forEach((item) => {
          item.line.classList.toggle("selected", selection?.kind === "edge" && item.edge === selection.edge);
          item.line.classList.toggle("path-highlight", edgePairs.has(canonicalEdgePairKey(item.edge.source, item.edge.target)));
        });
        nodeElems.forEach((circle, nodeId) => {
          const currentNodeId = Number(nodeId);
          circle.classList.toggle("selected", selection?.kind === "node" && Number(selection.node.id) === currentNodeId);
          circle.classList.toggle("connect-source", Number(state.connectSourceNodeId) === currentNodeId);
          circle.classList.toggle("path-highlight", highlightedNodeIds.has(currentNodeId));
          circle.classList.toggle("path-source", overlay != null && Number(overlay.sourceNodeId) === currentNodeId);
          circle.classList.toggle("path-target", targetNodeIds.has(currentNodeId));
        });
      }

      function setSelection(selection) {
        state.selected = selection;
        applyNetworkDecorations(selection);
        renderDetail(selection);
      }

      edges.forEach((edge) => {
        const group = make("g");
        const hitbox = make("path");
        hitbox.setAttribute("class", "network-edge-hitbox");
        const line = make("path");
        line.setAttribute("class", `network-edge-line${edge.neb_backed ? " neb-backed" : ""}`);
        group.appendChild(hitbox);
        group.appendChild(line);
        group.addEventListener("click", () => {
          if (Date.now() < Number(state.networkView.suppressClickUntil || 0)) return;
          state.connectSourceNodeId = null;
          setSelection({ kind: "edge", edge });
        });
        group.addEventListener("contextmenu", (event) => {
          event.preventDefault();
          event.stopPropagation();
          state.connectSourceNodeId = null;
          setSelection({ kind: "edge", edge });
          renderNetworkContextMenu(event, { kind: "edge", edge });
        });
        viewport.appendChild(group);
        edgeElems.push({ edge, line, hitbox, curvature: Number(curvatureByEdge.get(edge) || 0) });
      });

      nodes.forEach((node) => {
        const group = make("g");
        group.style.cursor = "pointer";
        const circle = make("circle");
        const classes = ["network-node"];
        if (Number(node.id) === 0) classes.push("root");
        if (node.neb_backed) classes.push("neb-backed");
        if (node.is_target) classes.push("target");
        circle.setAttribute("r", "18");
        circle.setAttribute("class", classes.join(" "));
        const text = make("text");
        text.setAttribute("class", "network-label");
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("y", "34");
        text.textContent = String(node.id);
        group.appendChild(circle);
        group.appendChild(text);
        group.addEventListener("click", async () => {
          if (Date.now() < Number(state.networkView.suppressClickUntil || 0)) return;
          const connectSource = state.connectSourceNodeId;
          if (connectSource != null && Number(connectSource) !== Number(node.id)) {
            setSelection({ kind: "node", node });
            await completeConnectMode(Number(node.id));
            return;
          }
          setSelection({ kind: "node", node });
        });
        group.addEventListener("contextmenu", (event) => {
          event.preventDefault();
          event.stopPropagation();
          setSelection({ kind: "node", node });
          renderNetworkContextMenu(event, { kind: "node", node });
        });
        viewport.appendChild(group);
        nodeElems.set(Number(node.id), circle);
        simById.get(Number(node.id)).group = group;
      });

      function applySimPositions() {
        edgeElems.forEach((item) => {
          const source = simById.get(Number(item.edge.source));
          const target = simById.get(Number(item.edge.target));
          if (!source || !target) return;
          const d = edgePathD(source, target, item.curvature);
          item.line.setAttribute("d", d);
          item.hitbox.setAttribute("d", d);
        });
        simNodes.forEach((simNode) => {
          if (simNode.group) simNode.group.setAttribute("transform", `translate(${simNode.x},${simNode.y})`);
        });
      }
      applySimPositions();
      state.networkNodePositions = Object.fromEntries(
        simNodes.map((simNode) => [
          String(simNode.id),
          { x: Number(simNode.x), y: Number(simNode.y) },
        ])
      );

      const pending = state.pendingEdgeAddition;
      if (pending) {
        const source = simById.get(Number(pending.source));
        const target = simById.get(Number(pending.target));
        if (source && target) {
          const group = make("g");
          group.setAttribute("aria-hidden", "true");
          group.style.pointerEvents = "none";
          const line = make("line");
          line.setAttribute("class", "network-edge-line pending-add");
          line.setAttribute("x1", String(source.x));
          line.setAttribute("y1", String(source.y));
          line.setAttribute("x2", String(source.x));
          line.setAttribute("y2", String(source.y));
          const animX = make("animate");
          animX.setAttribute("attributeName", "x2");
          animX.setAttribute("from", String(source.x));
          animX.setAttribute("to", String(target.x));
          animX.setAttribute("dur", "1.05s");
          animX.setAttribute("repeatCount", "indefinite");
          const animY = make("animate");
          animY.setAttribute("attributeName", "y2");
          animY.setAttribute("from", String(source.y));
          animY.setAttribute("to", String(target.y));
          animY.setAttribute("dur", "1.05s");
          animY.setAttribute("repeatCount", "indefinite");
          line.appendChild(animX);
          line.appendChild(animY);
          group.appendChild(line);
          viewport.appendChild(group);
        }
      }
      applyNetworkDecorations(state.selected);

      const selected = state.selected;
      if (selected?.kind === "edge") {
        const replacement = edges.find((edge) => Number(edge.source) === Number(selected.edge.source) && Number(edge.target) === Number(selected.edge.target));
        setSelection(replacement ? { kind: "edge", edge: replacement } : { kind: "node", node: nodes[0] });
      } else if (selected?.kind === "node") {
        const replacement = nodes.find((node) => Number(node.id) === Number(selected.node.id));
        setSelection(replacement ? { kind: "node", node: replacement } : { kind: "node", node: nodes[0] });
      } else if (edges.length) {
        setSelection({ kind: "edge", edge: edges[0] });
      } else {
        setSelection({ kind: "node", node: nodes[0] });
      }
    }

    function scheduleRefreshLoop(snapshot) {
      if (state.refreshTimer != null) {
        clearTimeout(state.refreshTimer);
        state.refreshTimer = null;
      }
      const isHawaii = String(snapshot?.active_action?.type || "") === "hawaii"
        && String(snapshot?.active_action?.status || "") === "running";
      const delayMs = isHawaii ? 750 : (snapshot?.busy ? 2000 : 5000);
      state.refreshTimer = setTimeout(refreshState, delayMs);
    }

    async function refreshState() {
      const refreshToken = state.localActivities.size > 0
        ? null
        : beginLocalActivity(
          "Refreshing workspace snapshot...",
          "Updating network and kinetics state from disk.",
          850
        );
      try {
        const response = await fetch("/api/state", { headers: authHeaders() });
        const snapshot = await readJsonResponse(response);
        if (!response.ok) throw new Error(snapshot.error || `State refresh failed: ${response.status}`);
        if (snapshot.active_action && snapshot.active_action.status === "running") {
          clearPendingLiveActivity();
        }
        state.snapshot = snapshot;
        const activeAction = snapshot.active_action || null;
        const activeLabel = activeAction && activeAction.status === "running"
          ? (activeAction.label || snapshot.busy_label)
          : snapshot.busy_label;
        setBanner(snapshot.last_message || (snapshot.busy ? `Running: ${activeLabel}` : "Idle."), Boolean(snapshot.last_error));
        if (snapshot.last_error) setBanner(snapshot.last_error, true);
        if (activeAction && activeAction.status === "running") {
          if (activeAction.type === "initialize") {
            setSubtext("The Retropaths network is being grown. The live growth widget shows existing nodes, edges, and the nodes currently being expanded.");
          } else if (activeAction.type === "minimize") {
            setSubtext("Geometry optimization is running. Updated node structures are written back one-by-one and will appear here as polling refreshes.");
          } else if (activeAction.type === "neb") {
            setSubtext("Autosplitting NEB is running. The edge state and any discovered intermediates will appear after the backend writes them back.");
          } else if (activeAction.type === "apply-reactions") {
            setSubtext("Reaction templates are being applied to the selected node. Any newly grown products will be merged into the current graph after the job finishes.");
          } else if (activeAction.type === "nanoreactor") {
            setSubtext("Nanoreactor sampling is running from the selected node. Distinct minimized products will be merged into the graph after the backend finishes.");
          } else if (activeAction.type === "hessian-sample") {
            setSubtext("Hessian sampling is running from the selected node or edge peak. Displaced-mode minima will be merged into the graph when the backend finishes.");
          } else if (activeAction.type === "hawaii") {
            setSubtext("Hawaii autonomous exploration is running. Use the stoplight controls to continue, stop after stage, or stop immediately.");
          } else {
            setSubtext("Background work is running. The network and counters will refresh automatically.");
          }
        } else {
          setSubtext("Idle.");
        }
        setButtonsDisabled(Boolean(snapshot.busy));
        if (snapshot.busy && state.manualNodeInsertMode) {
          setManualNodeInsertMode(false);
        }
        renderActivityIndicator(snapshot);
        renderWorkspaceSummary(snapshot);
        renderHawaiiStoplight(snapshot);
        renderInputsPanels(snapshot);
        renderStats(snapshot);
        renderProductPathPanel(snapshot);
        renderLiveActivity(snapshot);
        renderLogging(snapshot);
        renderKmcPanel(snapshot);
        const version = snapshot.drive?.version || "";
        if (version !== state.networkVersion) {
          state.kmcResult = null;
          const kmcTemperatureInput = document.getElementById("kmc-temperature");
          if (kmcTemperatureInput) delete kmcTemperatureInput.dataset.synced;
          state.networkVersion = version;
          renderProductPathPanel(snapshot);
          renderNetwork(snapshot);
          renderKmcPanel(snapshot);
        }
      } catch (error) {
        setBanner(error.message || String(error), true);
      } finally {
        if (refreshToken != null) {
          endLocalActivity(refreshToken);
        }
        scheduleRefreshLoop(state.snapshot);
      }
    }

    async function initializeDrive(growRetropaths = false) {
      const mode = document.getElementById("mode").value;
      const seedOnly = !growRetropaths;
      try {
        setBanner("Submitting initialization request...");
        setSubtext(
          seedOnly
            ? "Preparing a fresh workspace and seeding only the provided endpoints."
            : "Preparing a fresh workspace and growing the Retropaths network from the root."
        );
        await postJson("/api/initialize", {
          mode,
          seed_only: seedOnly,
          run_name: document.getElementById("run-name").value,
          inputs_path: document.getElementById("inputs-path").value,
          theory_program: document.getElementById("theory-program").value,
          theory_method: document.getElementById("theory-method").value,
          theory_basis: document.getElementById("theory-basis").value,
          reactions_fp: document.getElementById("reactions-path").value,
          environment_smiles: document.getElementById("environment-smiles").value,
          reactant_smiles: document.getElementById("reactant-smiles").value,
          reactant_xyz: document.getElementById("reactant-xyz").value,
          product_smiles: document.getElementById("product-smiles").value,
          product_xyz: document.getElementById("product-xyz").value,
        });
        setBanner(seedOnly ? "Seed-network initialization request accepted." : "Retropaths growth request accepted.");
        void refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The request failed before it could start.");
      }
    }

    async function loadWorkspace() {
      const loadButton = document.getElementById("load-workspace");
      if (loadButton) loadButton.disabled = true;
      try {
        setBanner("Loading existing workspace...");
        setSubtext("Reading the workspace files and restoring the live network.");
        await withLocalActivity(
          "Loading workspace files...",
          "Parsing workspace, queue, and kinetics metadata.",
          () => postJson("/api/load-workspace", {
            workspace_path: document.getElementById("workspace-path").value,
          }),
          { delayMs: 0 }
        );
        setBanner("Workspace load request accepted.");
        void refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The workspace could not be loaded.");
      } finally {
        if (loadButton) loadButton.disabled = Boolean(state.snapshot?.busy);
      }
    }

    async function queueMinimizeNode(nodeId) {
      try {
        setBanner(`Submitting minimization for node ${nodeId}...`);
        setSubtext("The selected geometry will be optimized and then written back into the live network.");
        await postJson("/api/minimize", { node_ids: [Number(nodeId)] });
        setBanner(`Minimization request accepted for node ${nodeId}.`);
        void refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The minimization request failed before it could start.");
      }
    }

    async function queueMinimizeAll() {
      try {
        setBanner("Submitting minimization for all geometries...");
        setSubtext("Every available endpoint geometry will be optimized.");
        await postJson("/api/minimize", { node_ids: [] });
        setBanner("Minimization request accepted for all geometries.");
        void refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The minimization request failed before it could start.");
      }
    }

    async function queueMinimizePair(sourceNode, targetNode) {
      try {
        setBanner(`Submitting minimization for nodes ${sourceNode} and ${targetNode}...`);
        setSubtext("Both endpoint geometries will be optimized and written back into the live network.");
        await postJson("/api/minimize", { node_ids: [Number(sourceNode), Number(targetNode)] });
        setBanner(`Minimization request accepted for nodes ${sourceNode} and ${targetNode}.`);
        void refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The paired minimization request failed before it could start.");
      }
    }

    async function queueEdgeNeb(sourceNode, targetNode) {
      try {
        setBanner(`Submitting autosplitting NEB for edge ${sourceNode} -> ${targetNode}...`);
        setSubtext("The edge endpoints will be prepared, the NEB will run in the background, and any discovered intermediates will be folded back into this network.");
        await postJson("/api/run-neb", { source_node: Number(sourceNode), target_node: Number(targetNode) });
        setBanner(`Autosplitting NEB request accepted for edge ${sourceNode} -> ${targetNode}.`);
        void refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The NEB request failed before it could start.");
      }
    }

    async function queueApplyReactions(nodeId) {
      try {
        setBanner(`Applying reactions to node ${nodeId}...`);
        setSubtext("The selected node will be regrown with the Retropaths template library and any new products will be merged into the current graph.");
        state.pendingLiveActivity = buildOptimisticGrowthActivity(
          Number(nodeId),
          `Applying reaction templates to node ${Number(nodeId)}...`,
          `Growing node ${Number(nodeId)}.`
        );
        renderLiveActivity(state.snapshot);
        await postJson("/api/apply-reactions", { node_id: Number(nodeId) });
        setBanner(`Reaction-application request accepted for node ${nodeId}.`);
        void refreshState();
      } catch (error) {
        clearPendingLiveActivity();
        renderLiveActivity(state.snapshot);
        setBanner(error.message || String(error), true);
        setSubtext("The reaction-application request failed before it could start.");
      }
    }

    async function queueNanoreactor(nodeId) {
      try {
        setBanner(`Running nanoreactor sampling from node ${nodeId}...`);
        setSubtext("The selected geometry will be sent to the configured nanoreactor backend and distinct minimized products will be merged into the graph.");
        state.pendingLiveActivity = buildOptimisticGrowthActivity(
          Number(nodeId),
          `Running nanoreactor sampling from node ${Number(nodeId)}...`,
          `Sampling products from node ${Number(nodeId)}.`
        );
        renderLiveActivity(state.snapshot);
        await postJson("/api/nanoreactor", { node_id: Number(nodeId) });
        setBanner(`Nanoreactor request accepted for node ${nodeId}.`);
        void refreshState();
      } catch (error) {
        clearPendingLiveActivity();
        renderLiveActivity(state.snapshot);
        setBanner(error.message || String(error), true);
        setSubtext("The nanoreactor request failed before it could start.");
      }
    }

    async function queueHessianSampleFromNode(nodeId) {
      const dr = getHessianSampleDr();
      const maxCandidates = getHessianSampleMaxCandidates();
      const useBigchem = getHessianSampleUseBigchem();
      try {
        setBanner(`Running Hessian sample from node ${nodeId} (dr=${dr}, max=${maxCandidates})...`);
        setSubtext("The selected minimum will be displaced along Hessian normal modes by ±dr, then up to the requested number of candidates will be optimized and merged as unique minima.");
        state.pendingLiveActivity = buildOptimisticGrowthActivity(
          Number(nodeId),
          `Running Hessian sample from node ${Number(nodeId)} (dr=${dr}, max=${maxCandidates})...`,
          `Sampling minima from node ${Number(nodeId)} with dr=${dr}, max=${maxCandidates}.`
        );
        renderLiveActivity(state.snapshot);
        await postJson("/api/hessian-sample", {
          node_id: Number(nodeId),
          dr: Number(dr),
          max_candidates: Number(maxCandidates),
          use_bigchem: Boolean(useBigchem),
        });
        setBanner(`Hessian-sample request accepted for node ${nodeId}.`);
        void refreshState();
      } catch (error) {
        clearPendingLiveActivity();
        renderLiveActivity(state.snapshot);
        setBanner(error.message || String(error), true);
        setSubtext("The Hessian-sample request failed before it could start.");
      }
    }

    async function queueHessianSampleFromEdge(sourceNode, targetNode) {
      const dr = getHessianSampleDr();
      const maxCandidates = getHessianSampleMaxCandidates();
      const useBigchem = getHessianSampleUseBigchem();
      try {
        setBanner(`Running Hessian sample from edge ${sourceNode} -> ${targetNode} peak (dr=${dr}, max=${maxCandidates})...`);
        setSubtext("The highest-energy geometry on the completed edge chain will be displaced by ±dr, then up to the requested number of candidates will be optimized and merged as minima.");
        state.pendingLiveActivity = buildOptimisticGrowthActivity(
          Number(sourceNode),
          `Running Hessian sample from edge ${Number(sourceNode)} -> ${Number(targetNode)} peak (dr=${dr}, max=${maxCandidates})...`,
          `Sampling minima from edge peak ${Number(sourceNode)} -> ${Number(targetNode)} with dr=${dr}, max=${maxCandidates}.`
        );
        renderLiveActivity(state.snapshot);
        await postJson("/api/hessian-sample", {
          source_node: Number(sourceNode),
          target_node: Number(targetNode),
          dr: Number(dr),
          max_candidates: Number(maxCandidates),
          use_bigchem: Boolean(useBigchem),
        });
        setBanner(`Hessian-sample request accepted for edge ${sourceNode} -> ${targetNode}.`);
        void refreshState();
      } catch (error) {
        clearPendingLiveActivity();
        renderLiveActivity(state.snapshot);
        setBanner(error.message || String(error), true);
        setSubtext("The edge Hessian-sample request failed before it could start.");
      }
    }

    function formatKmcInitialConditionsInput({ showErrors = true } = {}) {
      const input = document.getElementById("kmc-initial-conditions");
      if (!input) return false;
      const raw = String(input.value || "").trim();
      if (!raw) {
        input.value = "{}";
        return true;
      }
      try {
        const parsed = JSON.parse(raw);
        input.value = JSON.stringify(parsed, null, 2);
        return true;
      } catch (error) {
        if (showErrors) {
          setBanner("Initial conditions must be valid JSON.", true);
          setSubtext('Use a JSON object like {"0": 1.0, "4": 0.25}.');
        }
        return false;
      }
    }

    async function runKmcModel() {
      const runButton = document.getElementById("run-kmc");
      if (runButton) runButton.disabled = true;
      try {
        if (!formatKmcInitialConditionsInput({ showErrors: true })) {
          return;
        }
        setBanner("Running kinetic model...");
        setSubtext("Simulating population flow across the current NEB-backed network. This may take a while for large systems.");
        const result = await withLocalActivity(
          "Solving kinetic ODE system...",
          "Integrating population trajectories across the active reaction network.",
          () => postJson("/api/run-kinetics", {
            temperature_kelvin: Number(document.getElementById("kmc-temperature").value || 298.15),
            final_time: document.getElementById("kmc-final-time").value,
            max_steps: Number(document.getElementById("kmc-max-steps").value || 200),
            initial_conditions: document.getElementById("kmc-initial-conditions").value,
          }),
          { delayMs: 0 }
        );
        state.kmcResult = result;
        renderKmcPanel(state.snapshot);
        setBanner("Kinetic model finished.");
        setSubtext("Population trajectories were updated from the current reaction network.");
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The kinetic model could not be run.");
      } finally {
        if (runButton) runButton.disabled = Boolean(state.snapshot?.busy);
      }
    }

    async function addManualEdge() {
      if (state.manualEdgeRequestInFlight) {
        setBanner("A manual edge request is already in progress.", true);
        setSubtext("Wait for the current edge add to finish before sending another one.");
        return;
      }
      const sourceRaw = document.getElementById("manual-edge-source").value;
      const targetRaw = document.getElementById("manual-edge-target").value;
      if (sourceRaw === "" || targetRaw === "") {
        setBanner("Both manual edge endpoints are required.", true);
        setSubtext("Fill in both node ids before attempting to add a manual edge.");
        return;
      }
      const sourceNode = Number(sourceRaw);
      const targetNode = Number(targetRaw);
      try {
        setManualEdgeRequestInFlight(true);
        setBanner(`Attempting to add manual edge ${sourceNode} -> ${targetNode}...`);
        setSubtext("The graph edge will be created if needed and prepared for a subsequent autosplitting NEB run.");
        setPendingEdgeAddition(sourceNode, targetNode);
        const result = await postJson("/api/add-edge", {
          source_node: sourceNode,
          target_node: targetNode,
          reaction_label: document.getElementById("manual-edge-label").value,
        });
        setBanner(result.message || `Manual edge ${sourceNode} -> ${targetNode} updated.`);
        await refreshState();
      } catch (error) {
        clearPendingEdgeAddition();
        setBanner(error.message || String(error), true);
        setSubtext("The manual edge could not be added.");
      } finally {
        setManualEdgeRequestInFlight(false);
      }
    }

    async function setHawaiiStoplight(mode) {
      const normalized = String(mode || "").trim().toLowerCase();
      if (!["go", "yellow", "red"].includes(normalized)) return;
      const selectedDiscoveryTools = getSelectedHawaiiDiscoveryTools();
      try {
        state.hawaiiControlInFlight = true;
        renderHawaiiStoplight(state.snapshot);
        const bannerLabel = normalized === "go"
          ? "Sending GREEN (GO) to Hawaii mode..."
          : normalized === "yellow"
            ? "Sending YELLOW stop request to Hawaii mode..."
            : "Sending RED immediate stop request to Hawaii mode...";
        setBanner(bannerLabel);
        setSubtext("Updating Hawaii stoplight control...");
        const result = await postJson("/api/hawaii-control", {
          mode: normalized,
          discovery_tools: selectedDiscoveryTools,
        });
        if (Array.isArray(result?.discovery_tools)) {
          setSelectedHawaiiDiscoveryTools(result.discovery_tools, { force: true });
        } else {
          state.hawaiiDiscoveryDirty = false;
        }
        setBanner(result.message || "Hawaii stoplight updated.");
        void refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("Failed to update Hawaii stoplight.");
      } finally {
        state.hawaiiControlInFlight = false;
        renderHawaiiStoplight(state.snapshot);
      }
    }

    function beginConnectMode(nodeId) {
      if (state.manualEdgeRequestInFlight) {
        setSubtext("Wait for the current manual edge request to finish.");
        return;
      }
      const nextId = Number(nodeId);
      state.connectSourceNodeId = Number(state.connectSourceNodeId) === nextId ? null : nextId;
      setManualEdgeEndpoint("source", nextId);
      if (state.connectSourceNodeId == null) {
        setSubtext("Connect mode cleared.");
      } else {
        setSubtext(`Connect mode active from node ${nextId}. Click a second node in the network to create an edge.`);
      }
      renderNetworkToolbar();
      if (state.snapshot?.drive?.network) renderNetwork(state.snapshot);
    }

    async function completeConnectMode(targetNodeId) {
      if (state.manualEdgeRequestInFlight) {
        setBanner("A manual edge request is already in progress.", true);
        setSubtext("Wait for the current edge add to finish before sending another one.");
        return;
      }
      const sourceNodeId = Number(state.connectSourceNodeId);
      const targetNode = Number(targetNodeId);
      if (!Number.isFinite(sourceNodeId) || sourceNodeId < 0 || sourceNodeId === targetNode) {
        return;
      }
      setManualEdgeEndpoint("source", sourceNodeId);
      setManualEdgeEndpoint("target", targetNode);
      state.connectSourceNodeId = null;
      try {
        setManualEdgeRequestInFlight(true);
        setBanner(`Attempting to add manual edge ${sourceNodeId} -> ${targetNode}...`);
        setSubtext("Creating an edge directly from the graph selection.");
        setPendingEdgeAddition(sourceNodeId, targetNode);
        const result = await postJson("/api/add-edge", {
          source_node: sourceNodeId,
          target_node: targetNode,
          reaction_label: document.getElementById("manual-edge-label").value,
        });
        setBanner(result.message || `Manual edge ${sourceNodeId} -> ${targetNode} updated.`);
        await refreshState();
      } catch (error) {
        clearPendingEdgeAddition();
        setBanner(error.message || String(error), true);
        setSubtext("The graph-directed manual edge could not be added.");
      } finally {
        setManualEdgeRequestInFlight(false);
      }
    }

    window.queueMinimizeNode = queueMinimizeNode;
    window.queueMinimizeAll = queueMinimizeAll;
    window.queueMinimizePair = queueMinimizePair;
    window.queueEdgeNeb = queueEdgeNeb;
    window.queueApplyReactions = queueApplyReactions;
    window.queueNanoreactor = queueNanoreactor;
    window.queueHessianSampleFromNode = queueHessianSampleFromNode;
    window.queueHessianSampleFromEdge = queueHessianSampleFromEdge;
    window.runKmcModel = runKmcModel;
    window.setManualEdgeEndpoint = setManualEdgeEndpoint;
    window.toggleManualNodeInsertMode = toggleManualNodeInsertMode;
    window.beginConnectMode = beginConnectMode;
    window.setPathSourceNode = setPathSourceNode;
    window.setHawaiiStoplight = setHawaiiStoplight;

    document.getElementById("initialize-seed").addEventListener("click", () => initializeDrive(false));
    document.getElementById("initialize-grow").addEventListener("click", () => initializeDrive(true));
    document.getElementById("load-workspace").addEventListener("click", loadWorkspace);
    document.getElementById("minimize-all").addEventListener("click", queueMinimizeAll);
    document.getElementById("add-manual-edge").addEventListener("click", addManualEdge);
    document.getElementById("insert-manual-node-mode").addEventListener("click", toggleManualNodeInsertMode);
    document.getElementById("format-kmc-json").addEventListener("click", () => {
      formatKmcInitialConditionsInput({ showErrors: true });
    });
    document.getElementById("kmc-initial-conditions").addEventListener("blur", () => {
      formatKmcInitialConditionsInput({ showErrors: false });
    });
    document.getElementById("run-kmc").addEventListener("click", runKmcModel);
    document.getElementById("clear-product-path").addEventListener("click", clearProductPathHighlight);
    document.getElementById("hawaii-go").addEventListener("click", () => setHawaiiStoplight("go"));
    document.getElementById("hawaii-yellow").addEventListener("click", () => setHawaiiStoplight("yellow"));
    document.getElementById("hawaii-red").addEventListener("click", () => setHawaiiStoplight("red"));
    Object.values(HAWAII_DISCOVERY_TOOL_INPUTS).forEach((inputId) => {
      const input = document.getElementById(inputId);
      if (!input) return;
      input.addEventListener("change", () => {
        state.hawaiiDiscoveryDirty = true;
        renderHawaiiStoplight(state.snapshot);
      });
    });

    const d3Script = document.createElement("script");
    d3Script.src = "https://d3js.org/d3.v3.min.js";
    d3Script.onload = refreshState;
    document.head.appendChild(d3Script);
  </script>
</body>
</html>"""


@dataclass
class _DriveRuntimeState:
    workspace: RetropathsWorkspace | None = None
    reactant: dict[str, Any] | None = None
    product: dict[str, Any] | None = None
    last_message: str = "Idle."
    last_error: str = ""
    future: Future | None = None
    busy_label: str = ""
    active_action: dict[str, Any] | None = None
    payload_mode_hint: str = ""


class MepdDriveServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        base_directory: Path,
        inputs_fp: Path | None,
        reactions_fp: Path | None,
        timeout_seconds: int,
        max_nodes: int,
        max_depth: int,
        max_parallel_nebs: int,
        parallel_autosplit_nebs: bool = False,
        parallel_autosplit_workers: int = 4,
        network_splits: bool = True,
        hawaii_discovery_tools: Any = None,
        initial_state: dict[str, Any] | None = None,
        require_api_token: bool = False,
        api_token: str | None = None,
        max_request_bytes: int = 1_000_000,
    ):
        self.base_directory = base_directory
        self.inputs_fp = inputs_fp
        self.reactions_fp = reactions_fp
        self.timeout_seconds = timeout_seconds
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_parallel_nebs = max_parallel_nebs
        self.parallel_autosplit_nebs = bool(parallel_autosplit_nebs)
        self.parallel_autosplit_workers = max(1, int(parallel_autosplit_workers))
        self.network_splits = bool(network_splits)
        self.hawaii_discovery_tools = _normalize_hawaii_discovery_tools(
            hawaii_discovery_tools,
            default=_HAWAII_DISCOVERY_TOOL_DEFAULT,
        )
        self.require_api_token = bool(require_api_token)
        self.api_token = str(api_token or "")
        self.ui_token = self.api_token if self.require_api_token else ""
        self.max_request_bytes = int(max_request_bytes)
        self.state_lock = threading.Lock()
        self.runtime = _DriveRuntimeState()
        self._drive_payload_cache_key: tuple[Any, ...] | None = None
        self._drive_payload_cache_value: dict[str, Any] | None = None
        self._prefer_fast_payload_once = False
        self._prefer_fast_payload_until = 0.0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mepd-drive")
        self.process_executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        super().__init__(server_address, _DriveHandler)
        if initial_state is not None:
            workspace_payload = dict(initial_state.get("workspace") or {})
            if workspace_payload:
                self.runtime.workspace = RetropathsWorkspace(**workspace_payload)
            self.runtime.reactant = initial_state.get("reactant")
            self.runtime.product = initial_state.get("product")
            self.runtime.last_message = str(initial_state.get("message") or "Workspace loaded.")
            if self.runtime.workspace is not None and self.runtime.workspace.queue_fp.exists():
                with contextlib.suppress(Exception):
                    runtime = _DriveRuntimeState(
                        workspace=self.runtime.workspace,
                        reactant=self.runtime.reactant,
                        product=self.runtime.product,
                        last_message=self.runtime.last_message,
                        last_error=self.runtime.last_error,
                        future=None,
                        busy_label="",
                        active_action=None,
                        payload_mode_hint="",
                    )
                    self._drive_payload_cache_lookup(
                        workspace=self.runtime.workspace,
                        runtime=runtime,
                    )

    def _reset_process_executor(self) -> None:
        old_executor = getattr(self, "process_executor", None)
        if old_executor is not None:
            processes = dict(getattr(old_executor, "_processes", {}) or {})
            for process in processes.values():
                with contextlib.suppress(Exception):
                    process.terminate()
            for process in processes.values():
                with contextlib.suppress(Exception):
                    process.join(timeout=1.0)
            with contextlib.suppress(Exception):
                old_executor.shutdown(wait=False, cancel_futures=True)
        self.process_executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def _resolve_inputs_fp(self, payload: dict[str, Any]) -> Path:
        configured = getattr(self, "inputs_fp", None)
        payload_value = str(payload.get("inputs_path") or "").strip()
        if payload_value:
            return Path(payload_value).expanduser().resolve()
        if configured is not None:
            return Path(configured).resolve()
        raise ValueError("An inputs TOML path is required. Provide --inputs when launching drive or fill in Inputs TOML in the UI.")

    def _resolve_reactions_fp(self, payload: dict[str, Any]) -> Path | None:
        payload_value = str(payload.get("reactions_fp") or "").strip()
        if payload_value:
            return Path(payload_value).expanduser().resolve()
        configured = getattr(self, "reactions_fp", None)
        if configured is None:
            return None
        return Path(configured).resolve()

    def _drive_payload_cache_lookup(
        self,
        *,
        workspace: RetropathsWorkspace,
        runtime: _DriveRuntimeState,
    ) -> dict[str, Any]:
        network_splits = getattr(self, "network_splits", True)
        builder = _build_drive_payload
        builder_name = "full"
        with self.state_lock:
            prefer_fast_once = bool(getattr(self, "_prefer_fast_payload_once", False))
            prefer_fast_until = float(getattr(self, "_prefer_fast_payload_until", 0.0) or 0.0)
        if (
            runtime.active_action is not None
            and runtime.active_action.get("status") == "running"
            and runtime.active_action.get("type") in {
                "minimize",
                "initialize",
                "apply-reactions",
                "nanoreactor",
                "hessian-sample",
                "hawaii",
                "load-workspace",
            }
        ):
            builder = _build_drive_payload_fast
            builder_name = f"fast-{runtime.active_action.get('type')}"
        elif (
            runtime.active_action is not None
            and runtime.active_action.get("status") == "running"
            and runtime.active_action.get("type") == "neb"
        ):
            builder = _build_drive_payload_fast_neb
            builder_name = "fast-neb"
        elif prefer_fast_once or time.time() < prefer_fast_until:
            builder = _build_drive_payload_fast
            builder_name = "fast-post-manual-edge"

        product_smiles = str((runtime.product or {}).get("smiles") or "")
        active_action = runtime.active_action or {}
        version_key = _drive_network_version(workspace)
        if builder_name.startswith("fast-") and builder_name != "fast-hawaii":
            version_key = f"{builder_name}-active"
        action_progress_key: tuple[Any, ...] = ()
        if builder_name.startswith("fast-") and builder_name != "fast-neb":
            jobs = active_action.get("jobs", []) or []
            action_progress_key = (
                str(active_action.get("type") or ""),
                str(active_action.get("status") or ""),
                str(active_action.get("label") or ""),
                tuple(
                    (
                        int(job.get("node_id", -1)),
                        str(job.get("status") or ""),
                        str(job.get("error") or ""),
                    )
                    for job in jobs
                    if isinstance(job, dict)
                ),
            )
            if builder_name == "fast-hawaii":
                progress_fp = active_action.get("progress_fp")
                progress_stat = ("missing",)
                if progress_fp:
                    with contextlib.suppress(Exception):
                        stat = Path(str(progress_fp)).stat()
                        progress_stat = (int(stat.st_mtime_ns), int(stat.st_size))
                action_progress_key = (*action_progress_key, progress_stat)
        elif builder_name == "fast-neb":
            action_progress_key = (
                str(active_action.get("status") or ""),
                str(active_action.get("label") or ""),
            )
        cache_key = (
            builder_name,
            version_key,
            product_smiles,
            action_progress_key,
            tuple(int(node_id) for node_id in active_action.get("node_ids", []))
            if builder_name.startswith("fast-") and builder_name != "fast-neb"
            else (),
            int(active_action.get("source_node", -1)) if builder_name == "fast-neb" else -1,
            int(active_action.get("target_node", -1)) if builder_name == "fast-neb" else -1,
        )

        with self.state_lock:
            cached_key = getattr(self, "_drive_payload_cache_key", None)
            cached_value = getattr(self, "_drive_payload_cache_value", None)
            if cached_key == cache_key and cached_value is not None:
                return cached_value

        payload = _call_drive_payload_builder(
            builder,
            workspace,
            network_splits=getattr(self, "network_splits", True),
            product_smiles=product_smiles,
            active_job_label=runtime.busy_label if runtime.future is not None and not runtime.future.done() else "",
            active_action=runtime.active_action,
        )
        with self.state_lock:
            self._drive_payload_cache_key = cache_key
            self._drive_payload_cache_value = payload
            if builder_name == "fast-post-manual-edge":
                self._prefer_fast_payload_once = False
        return payload

    def _assert_idle(self) -> None:
        with self.state_lock:
            future = self.runtime.future
            busy_label = self.runtime.busy_label
        if future is not None and not future.done():
            raise ValueError(f"Another drive action is still running: {busy_label or 'background job'}. Wait for it to finish first.")

    def _set_busy(self, label: str, future: Future) -> None:
        with self.state_lock:
            self.runtime.future = future
            self.runtime.busy_label = label
            self.runtime.last_error = ""
            self.runtime.last_message = label

    def _submit_process_action(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        label: str,
        **kwargs: Any,
    ) -> Future:
        def _job() -> Any:
            process_future = self.process_executor.submit(fn, *args, **kwargs)
            return process_future.result()

        future = self.executor.submit(_job)
        self._set_busy(label, future)
        return future

    def _finish_future(self, future: Future) -> None:
        with self.state_lock:
            if self.runtime.future is not future:
                return
            self.runtime.future = None
            self.runtime.busy_label = ""
            if self.runtime.active_action is not None:
                self.runtime.active_action["status"] = "finished"
        try:
            result = future.result()
        except Exception as exc:
            with self.state_lock:
                active_action = self.runtime.active_action or {}
                active_type = str(active_action.get("type") or "")
                control_fp = active_action.get("control_fp") if isinstance(active_action, dict) else None
                if active_type == "hawaii" and _read_hawaii_control_mode(control_fp) == "red":
                    self.runtime.last_error = ""
                    self.runtime.last_message = "Hawaii autonomous exploration stopped immediately."
                    self.runtime.active_action = None
                    return
                self.runtime.last_error = _format_user_facing_error(exc)
                self.runtime.last_message = "Last action failed."
                if self.runtime.active_action is not None:
                    self.runtime.active_action["status"] = "failed"
                    self.runtime.active_action["error"] = _format_user_facing_error(exc)
            return
        message = result.get("message") if isinstance(result, dict) else result
        with self.state_lock:
            self.runtime.last_error = ""
            self.runtime.last_message = str(message or "Action completed.")
            self.runtime.active_action = None

    def _record_request_error(self, message: str) -> None:
        with self.state_lock:
            self.runtime.last_error = str(message)
            self.runtime.last_message = "Last action failed."

    def submit_initialize(self, payload: dict[str, Any]) -> None:
        self._assert_idle()
        charge = int(payload.get("charge", 0) or 0)
        multiplicity = int(payload.get("multiplicity", 1) or 1)
        auto_start_hawaii = bool(payload.get("auto_start_hawaii", False))
        auto_start_hawaii_discovery_tools = payload.get("auto_start_hawaii_discovery_tools")
        reactant = _resolve_species_input(
            smiles=str(payload.get("reactant_smiles") or ""),
            xyz_text=str(payload.get("reactant_xyz") or ""),
            charge=charge,
            multiplicity=multiplicity,
        )
        if not reactant:
            raise ValueError("A reactant SMILES or reactant XYZ block is required.")

        product = _resolve_species_input(
            smiles=str(payload.get("product_smiles") or ""),
            xyz_text=str(payload.get("product_xyz") or ""),
            charge=charge,
            multiplicity=multiplicity,
        )
        if str(payload.get("mode") or "reactant") == "reactant-product" and not product:
            raise ValueError("Reactant/product mode requires a product SMILES or product XYZ block.")

        payload_seed_only = payload.get("seed_only")
        if payload_seed_only is None:
            seed_only = bool(product) or bool(str(reactant.get("xyz_b64") or "").strip()) or bool(
                product and str(product.get("xyz_b64") or "").strip()
            )
        else:
            seed_only = bool(payload_seed_only)
        initialize_label = "Seeding workspace network..." if seed_only else "Building Retropaths network..."

        run_name = _validate_run_name(
            str(payload.get("run_name") or "").strip() or f"mepd-drive-{int(time.time())}"
        )
        workspace_dir = (self.base_directory / run_name).resolve()
        progress_fp: str | None = None
        if not seed_only:
            progress_fp = str((workspace_dir / "drive_growth.progress.json").resolve())
        inputs_fp = self._resolve_inputs_fp(payload)
        inputs_fp = _materialize_deployment_inputs(
            template_fp=inputs_fp,
            output_dir=self.base_directory,
            run_name=run_name,
            theory_program=str(payload.get("theory_program") or "").strip().lower() or None,
            theory_method=str(payload.get("theory_method") or "").strip() or None,
            theory_basis=str(payload.get("theory_basis") or "").strip() or None,
        )
        reactions_fp = self._resolve_reactions_fp(payload)
        environment_smiles = str(payload.get("environment_smiles") or "").strip()

        def _finish_initialize(future: Future) -> None:
            with self.state_lock:
                if self.runtime.future is not future:
                    return
                self.runtime.future = None
                self.runtime.busy_label = ""
                if self.runtime.active_action is not None:
                    self.runtime.active_action["status"] = "finished"
            with self.state_lock:
                self.runtime.last_error = ""
            try:
                result = future.result()
            except Exception as exc:
                with self.state_lock:
                    self.runtime.last_error = _format_user_facing_error(exc)
                    self.runtime.last_message = "Last action failed."
                    if self.runtime.active_action is not None:
                        self.runtime.active_action["status"] = "failed"
                        self.runtime.active_action["error"] = _format_user_facing_error(exc)
                return
            workspace = RetropathsWorkspace(**result["workspace"])
            with self.state_lock:
                self.runtime.workspace = workspace
                self.runtime.reactant = result["reactant"]
                self.runtime.product = result["product"] or None
                self.runtime.last_message = str(result.get("message") or "Action completed.")
                self.runtime.last_error = str(result.get("error") or "")
                self.runtime.active_action = None
            if auto_start_hawaii:
                try:
                    if auto_start_hawaii_discovery_tools is None:
                        self.submit_hawaii()
                    else:
                        self.submit_hawaii(discovery_tools=auto_start_hawaii_discovery_tools)
                except Exception as exc:
                    with self.state_lock:
                        self.runtime.last_error = _format_user_facing_error(exc)
                        self.runtime.last_message = "Last action failed."

        future = self._submit_process_action(
            _initialize_workspace_job,
            reactant=reactant,
            product=product or None,
            run_name=run_name,
            workspace_dir=str(workspace_dir),
            inputs_fp=str(inputs_fp),
            reactions_fp=str(reactions_fp) if reactions_fp else None,
            environment_smiles=environment_smiles,
            timeout_seconds=self.timeout_seconds,
            max_nodes=self.max_nodes,
            max_depth=self.max_depth,
            max_parallel_nebs=self.max_parallel_nebs,
            parallel_autosplit_nebs=self.parallel_autosplit_nebs,
            parallel_autosplit_workers=self.parallel_autosplit_workers,
            seed_only=seed_only,
            network_splits=getattr(self, "network_splits", True),
            progress_fp=progress_fp,
            allowed_base_dir=str(self.base_directory.resolve()),
            label=initialize_label,
        )
        future.add_done_callback(_finish_initialize)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "initialize",
                "status": "running",
                "label": initialize_label,
                "progress_fp": progress_fp,
            }

    def submit_load_workspace(self, payload: dict[str, Any]) -> None:
        self._assert_idle()
        workspace_path = str(payload.get("workspace_path") or "").strip()
        if not workspace_path:
            raise ValueError("A workspace path is required.")
        if bool(getattr(self, "require_api_token", False)):
            requested_path = Path(workspace_path).expanduser().resolve()
            requested_workspace_dir = requested_path.parent if requested_path.is_file() else requested_path
            if not _is_within_directory(requested_workspace_dir, self.base_directory.resolve()):
                raise ValueError("Workspace path must be within the configured drive base directory.")

        def _finish_load(future: Future) -> None:
            with self.state_lock:
                if self.runtime.future is not future:
                    return
                self.runtime.future = None
                self.runtime.busy_label = ""
                if self.runtime.active_action is not None:
                    self.runtime.active_action["status"] = "finished"
            with self.state_lock:
                self.runtime.last_error = ""
            try:
                result = future.result()
            except Exception as exc:
                with self.state_lock:
                    self.runtime.last_error = _format_user_facing_error(exc)
                    self.runtime.last_message = "Last action failed."
                    if self.runtime.active_action is not None:
                        self.runtime.active_action["status"] = "failed"
                        self.runtime.active_action["error"] = _format_user_facing_error(exc)
                return
            workspace = RetropathsWorkspace(**result["workspace"])
            with self.state_lock:
                self.runtime.workspace = workspace
                self.runtime.reactant = result["reactant"]
                self.runtime.product = result["product"] or None
                self.runtime.last_message = str(result.get("message") or "Workspace loaded.")
                self.runtime.active_action = None

        future = self.executor.submit(
            _load_existing_workspace_job,
            workspace_path,
            network_splits=getattr(self, "network_splits", True),
        )
        future.add_done_callback(_finish_load)
        self._set_busy("Loading existing workspace...", future)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "load-workspace",
                "status": "running",
                "label": "Loading existing workspace...",
            }

    def submit_minimize(self, node_ids: list[int]) -> None:
        self._assert_idle()
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize a workspace before queueing minimizations.")
        target_node_ids = _resolve_minimize_target_indices(workspace, node_ids or None)
        if not target_node_ids:
            if node_ids:
                pot = Pot.read_from_disk(workspace.neb_pot_fp)
                requested = int(node_ids[0])
                if requested not in pot.graph.nodes:
                    raise ValueError(f"Node {requested} is not present in the current workspace.")
                _minimizable, note = _node_minimize_status(requested, pot.graph.nodes[requested])
                raise ValueError(note or f"No geometry matched the minimization request for node {requested}.")
            raise ValueError("No geometries matched the minimization request.")

        def _progress(message: str) -> None:
            with self.state_lock:
                self.runtime.busy_label = message
                self.runtime.last_message = message
                if self.runtime.active_action is not None:
                    self.runtime.active_action["label"] = message

        def _on_node_update(payload: dict[str, Any]) -> None:
            with self.state_lock:
                active = self.runtime.active_action
                if active is None or active.get("type") != "minimize":
                    return
                jobs = active.setdefault("jobs", [])
                job = next(
                    (item for item in jobs if int(item.get("node_id", -1)) == int(payload["node_id"])),
                    None,
                )
                if job is None:
                    job = {"node_id": int(payload["node_id"]), "status": "pending"}
                    jobs.append(job)
                job.update(payload)

        def _job() -> str:
            result = _optimize_selected_nodes(
                workspace,
                node_ids or None,
                progress=_progress,
                on_node_update=_on_node_update,
            )
            return str(result.get("message") or "Minimization finished.")

        future = self.executor.submit(_job)
        future.add_done_callback(self._finish_future)
        self._set_busy("Running geometry minimizations...", future)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "minimize",
                "status": "running",
                "label": "Running geometry minimizations...",
                "node_ids": list(target_node_ids),
                "jobs": [{"node_id": int(node_id), "status": "pending"} for node_id in target_node_ids],
            }

    def submit_apply_reactions(self, *, node_id: int) -> None:
        self._assert_idle()
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize a workspace before applying reactions.")
        try:
            ensure_retropaths_available(feature="MEPD Drive reaction-template application (+)")
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

        progress_fp = str((workspace.directory / f"drive_apply_reactions_{int(node_id)}.progress.json").resolve())
        future = self._submit_process_action(
            apply_reactions_to_node,
            RetropathsWorkspace(**dict(workspace.__dict__)),
            int(node_id),
            progress_fp=progress_fp,
            label=f"Applying reaction templates to node {node_id}...",
        )
        future.add_done_callback(self._finish_future)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "apply-reactions",
                "status": "running",
                "label": f"Applying reaction templates to node {node_id}...",
                "node_id": int(node_id),
                "progress_fp": progress_fp,
            }

    def submit_nanoreactor(self, *, node_id: int) -> None:
        self._assert_idle()
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize a workspace before running nanoreactor sampling.")

        progress_fp = str((workspace.directory / f"drive_nanoreactor_{int(node_id)}.progress.json").resolve())
        workspace_copy = RetropathsWorkspace(**dict(workspace.__dict__))

        def _job() -> dict[str, Any]:
            return run_nanoreactor_for_node(
                workspace_copy,
                int(node_id),
                progress_fp=progress_fp,
            )

        future = self.executor.submit(_job)
        future.add_done_callback(self._finish_future)
        self._set_busy(f"Running nanoreactor sampling from node {node_id}...", future)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "nanoreactor",
                "status": "running",
                "label": f"Running nanoreactor sampling from node {node_id}...",
                "node_id": int(node_id),
                "progress_fp": progress_fp,
            }

    def submit_hessian_sample(
        self,
        *,
        dr: float,
        max_candidates: int = 100,
        use_bigchem: bool | None = None,
        node_id: int | None = None,
        source_node: int | None = None,
        target_node: int | None = None,
    ) -> None:
        self._assert_idle()
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize a workspace before running Hessian sampling.")
        if not (float(dr) > 0):
            raise ValueError("`dr` must be a positive number.")
        max_candidates = int(max_candidates)
        if max_candidates <= 0:
            raise ValueError("`max_candidates` must be a positive integer.")

        workspace_copy = RetropathsWorkspace(**dict(workspace.__dict__))
        if node_id is not None:
            progress_fp = str((workspace.directory / f"drive_hessian_node_{int(node_id)}.progress.json").resolve())
            label = (
                f"Running Hessian sample from node {int(node_id)} "
                f"(dr={float(dr):.4f}, max={int(max_candidates)})..."
            )

            def _job() -> dict[str, Any]:
                return run_hessian_sample_for_node(
                    workspace_copy,
                    int(node_id),
                    dr=float(dr),
                    max_candidates=int(max_candidates),
                    use_bigchem=use_bigchem,
                    progress_fp=progress_fp,
                )

            action_payload = {
                "type": "hessian-sample",
                "status": "running",
                "label": label,
                "dr": float(dr),
                "max_candidates": int(max_candidates),
                "use_bigchem": (bool(use_bigchem) if use_bigchem is not None else None),
                "node_id": int(node_id),
                "progress_fp": progress_fp,
            }
        else:
            if source_node is None or target_node is None:
                raise ValueError("Provide either `node_id` or both `source_node` and `target_node` for Hessian sampling.")
            progress_fp = str(
                (
                    workspace.directory
                    / f"drive_hessian_edge_{int(source_node)}_{int(target_node)}.progress.json"
                ).resolve()
            )
            label = (
                f"Running Hessian sample from edge {int(source_node)} -> {int(target_node)} peak "
                f"(dr={float(dr):.4f}, max={int(max_candidates)})..."
            )

            def _job() -> dict[str, Any]:
                return run_hessian_sample_for_edge(
                    workspace_copy,
                    int(source_node),
                    int(target_node),
                    dr=float(dr),
                    max_candidates=int(max_candidates),
                    use_bigchem=use_bigchem,
                    progress_fp=progress_fp,
                )

            action_payload = {
                "type": "hessian-sample",
                "status": "running",
                "label": label,
                "dr": float(dr),
                "max_candidates": int(max_candidates),
                "use_bigchem": (bool(use_bigchem) if use_bigchem is not None else None),
                "source_node": int(source_node),
                "target_node": int(target_node),
                "progress_fp": progress_fp,
            }

        future = self.executor.submit(_job)
        future.add_done_callback(self._finish_future)
        self._set_busy(label, future)
        with self.state_lock:
            self.runtime.active_action = action_payload

    def submit_hawaii(self, *, discovery_tools: Any = None) -> None:
        self._assert_idle()
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize or load a workspace before running --hawaii automation.")

        resolved_discovery_tools = _normalize_hawaii_discovery_tools(
            discovery_tools if discovery_tools is not None else getattr(self, "hawaii_discovery_tools", None),
            default=_HAWAII_DISCOVERY_TOOL_DEFAULT,
        )
        self.hawaii_discovery_tools = list(resolved_discovery_tools)
        progress_fp = str((workspace.directory / "drive_hawaii.progress.json").resolve())
        control_fp = str((workspace.directory / "drive_hawaii.control.json").resolve())
        _write_hawaii_control_payload(
            control_fp,
            mode="go",
            source="server",
            note="Hawaii started.",
            discovery_tools=resolved_discovery_tools,
        )
        label = (
            "Running Hawaii autonomous exploration "
            f"(discovery tools: {', '.join(resolved_discovery_tools) if resolved_discovery_tools else 'none'})..."
        )
        future = self._submit_process_action(
            _run_hawaii_autonomy,
            dict(workspace.__dict__),
            network_splits=getattr(self, "network_splits", True),
            progress_fp=progress_fp,
            control_fp=control_fp,
            max_hessian_candidates=100,
            discovery_tools=list(resolved_discovery_tools),
            label=label,
        )
        future.add_done_callback(self._finish_future)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "hawaii",
                "status": "running",
                "label": label,
                "progress_fp": progress_fp,
                "control_fp": control_fp,
                "discovery_tools": list(resolved_discovery_tools),
            }

    def submit_hawaii_control(self, *, mode: str, discovery_tools: Any = None) -> dict[str, Any]:
        normalized_mode = _normalize_hawaii_control_mode(mode)
        resolved_discovery_tools = (
            _normalize_hawaii_discovery_tools(discovery_tools, default=None)
            if discovery_tools is not None
            else None
        )
        if resolved_discovery_tools is not None:
            self.hawaii_discovery_tools = list(resolved_discovery_tools)
        with self.state_lock:
            workspace = self.runtime.workspace
            active_action = dict(self.runtime.active_action) if isinstance(self.runtime.active_action, dict) else None
            future = self.runtime.future

        if workspace is None:
            raise ValueError("Initialize or load a workspace before controlling Hawaii mode.")
        control_fp = str(_hawaii_control_fp_for_workspace(workspace))
        running_hawaii = bool(
            future is not None
            and not future.done()
            and active_action is not None
            and str(active_action.get("type") or "") == "hawaii"
            and str(active_action.get("status") or "") == "running"
        )

        if normalized_mode == "go" and not running_hawaii:
            if resolved_discovery_tools is None:
                self.submit_hawaii()
            else:
                self.submit_hawaii(discovery_tools=resolved_discovery_tools)
            return {
                "message": "Hawaii autonomous exploration started.",
                "mode": "go",
                "running": True,
                "discovery_tools": list(getattr(self, "hawaii_discovery_tools", [])),
            }

        _write_hawaii_control_payload(
            control_fp,
            mode=normalized_mode,
            source="server",
            note=f"Stoplight set to {normalized_mode}.",
            discovery_tools=resolved_discovery_tools,
        )

        if not running_hawaii:
            with self.state_lock:
                self.runtime.last_error = ""
                self.runtime.last_message = (
                    "Hawaii stoplight set to GO."
                    if normalized_mode == "go"
                    else f"Hawaii stoplight set to {normalized_mode.upper()}."
                )
            return {
                "message": self.runtime.last_message,
                "mode": normalized_mode,
                "running": False,
                "discovery_tools": list(getattr(self, "hawaii_discovery_tools", [])),
            }

        if normalized_mode == "red":
            with self.state_lock:
                self.runtime.last_error = ""
                self.runtime.last_message = "Stopping Hawaii autonomous exploration immediately..."
            self._reset_process_executor()
            return {
                "message": "Stopping Hawaii autonomous exploration immediately.",
                "mode": "red",
                "running": True,
                "discovery_tools": list(getattr(self, "hawaii_discovery_tools", [])),
            }

        with self.state_lock:
            self.runtime.last_error = ""
            if normalized_mode == "yellow":
                self.runtime.last_message = "Hawaii will stop after the current stage finishes."
            else:
                self.runtime.last_message = "Hawaii stoplight set to GO."
        return {
            "message": self.runtime.last_message,
            "mode": normalized_mode,
            "running": True,
            "discovery_tools": list(getattr(self, "hawaii_discovery_tools", [])),
        }

    def submit_run_neb(self, *, source_node: int, target_node: int) -> None:
        self._assert_idle()
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize a workspace before queueing NEBs.")
        pot = Pot.read_from_disk(workspace.neb_pot_fp)
        queue = build_retropaths_neb_queue(
            pot=pot,
            queue_fp=workspace.queue_fp,
            overwrite=False,
        )
        with contextlib.suppress(Exception):
            charge, multiplicity = _workspace_charge_multiplicity(workspace)
            if queue.recover_stale_running_items(
                output_dir=workspace.queue_output_dir,
                charge=charge,
                multiplicity=multiplicity,
            ):
                queue.write_to_disk(workspace.queue_fp)
                if getattr(self, "network_splits", True):
                    load_partial_annotated_pot(workspace)
                queue = build_retropaths_neb_queue(
                    pot=pot,
                    queue_fp=workspace.queue_fp,
                    overwrite=False,
                )
        item = queue.find_item(source_node, target_node)
        if item is None:
            raise ValueError(f"Edge {source_node} -> {target_node} is not present in the NEB queue.")
        can_queue_neb, queue_note = _edge_neb_status(
            {
                "queue_status": item.status,
                "queue_error": item.error,
            }
        )
        if not can_queue_neb:
            raise ValueError(queue_note or f"Edge {source_node} -> {target_node} cannot be queued.")
        log_fp = str((workspace.directory / f"drive_neb_{source_node}_{target_node}.log").resolve())
        progress_fp = str((workspace.directory / f"drive_neb_{source_node}_{target_node}.progress.log").resolve())
        chain_fp = str((workspace.directory / f"drive_neb_{source_node}_{target_node}.chain.json").resolve())
        future = self._submit_process_action(
            _run_selected_edge_neb_logged,
            dict(workspace.__dict__),
            source_node=source_node,
            target_node=target_node,
            network_splits=getattr(self, "network_splits", True),
            log_fp=log_fp,
            progress_fp=progress_fp,
            chain_fp=chain_fp,
            label=f"Running autosplitting NEB for {source_node} -> {target_node}...",
        )
        future.add_done_callback(self._finish_future)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "neb",
                "status": "running",
                "label": f"Running autosplitting NEB for {source_node} -> {target_node}...",
                "source_node": int(source_node),
                "target_node": int(target_node),
                "log_fp": log_fp,
                "progress_fp": progress_fp,
                "chain_fp": chain_fp,
            }

    def submit_add_edge(self, *, source_node: int, target_node: int, reaction_label: str = "") -> dict[str, Any]:
        self._assert_idle()
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize a workspace before adding manual edges.")
        result = add_manual_edge(
            workspace,
            source_node=int(source_node),
            target_node=int(target_node),
            reaction_label=str(reaction_label or ""),
        )
        with self.state_lock:
            self.runtime.last_error = ""
            self.runtime.last_message = str(result.get("message") or "Manual edge updated.")
            self._prefer_fast_payload_once = True
            self._prefer_fast_payload_until = time.time() + 30.0
        return result

    def submit_add_node(
        self,
        *,
        xyz_text: str,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> dict[str, Any]:
        self._assert_idle()
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize a workspace before adding manual nodes.")
        result = add_manual_node(
            workspace,
            xyz_text=str(xyz_text or ""),
            charge=int(charge),
            multiplicity=int(multiplicity),
        )
        with self.state_lock:
            self.runtime.last_error = ""
            self.runtime.last_message = str(result.get("message") or "Manual node added.")
            self._prefer_fast_payload_once = True
            self._prefer_fast_payload_until = time.time() + 30.0
        return result

    def submit_run_kmc(
        self,
        *,
        temperature_kelvin: float,
        final_time: float | None,
        max_steps: int,
        initial_conditions_text: str | None = None,
    ) -> dict[str, Any]:
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize a workspace before running kinetics.")
        result = _run_kmc_payload(
            workspace,
            network_splits=getattr(self, "network_splits", True),
            temperature_kelvin=float(temperature_kelvin),
            final_time=float(final_time) if final_time is not None else None,
            max_steps=int(max_steps),
            initial_conditions=_parse_kmc_initial_conditions(initial_conditions_text),
        )
        with self.state_lock:
            self.runtime.last_error = ""
            self.runtime.last_message = "Kinetic model updated."
        return result

    def snapshot(self) -> dict[str, Any]:
        with self.state_lock:
            runtime = _DriveRuntimeState(
                workspace=self.runtime.workspace,
                reactant=self.runtime.reactant,
                product=self.runtime.product,
                last_message=self.runtime.last_message,
                last_error=self.runtime.last_error,
                future=self.runtime.future,
                busy_label=self.runtime.busy_label,
                active_action=dict(self.runtime.active_action) if self.runtime.active_action is not None else None,
            )
        snapshot: dict[str, Any] = {
            "initialized": runtime.workspace is not None,
            "busy": runtime.future is not None and not runtime.future.done(),
            "busy_label": runtime.busy_label,
            "last_message": runtime.last_message,
            "last_error": runtime.last_error,
            "reactant": runtime.reactant,
            "product": runtime.product,
            "active_action": runtime.active_action,
            "live_activity": None,
            "drive": None,
            "defaults": _drive_defaults_payload(
                getattr(self, "inputs_fp", None),
                getattr(self, "reactions_fp", None),
            ),
            "hawaii": {
                "running": False,
                "mode": "go",
                "stage": "",
                "dr": None,
                "note": "",
                "discovery_tools": list(getattr(self, "hawaii_discovery_tools", [])),
            },
        }
        if runtime.workspace is not None:
            control_fp = _hawaii_control_fp_for_workspace(runtime.workspace)
            mode = _read_hawaii_control_mode(control_fp)
            active_action = runtime.active_action or {}
            progress = _read_growth_progress(active_action.get("progress_fp")) if str(active_action.get("type") or "") == "hawaii" else {}
            configured_discovery_tools = list(
                _normalize_hawaii_discovery_tools(
                    active_action.get("discovery_tools")
                    if isinstance(active_action, dict) and active_action.get("discovery_tools") is not None
                    else getattr(self, "hawaii_discovery_tools", None),
                    default=_HAWAII_DISCOVERY_TOOL_DEFAULT,
                )
            )
            snapshot["hawaii"] = {
                "running": bool(
                    runtime.future is not None
                    and not runtime.future.done()
                    and str(active_action.get("type") or "") == "hawaii"
                    and str(active_action.get("status") or "") == "running"
                ),
                "mode": mode,
                "stage": str((progress or {}).get("stage") or ""),
                "dr": (float((progress or {}).get("dr")) if (progress or {}).get("dr") is not None else None),
                "note": str((progress or {}).get("note") or ""),
                "discovery_tools": configured_discovery_tools,
            }
        if runtime.active_action is not None and runtime.active_action.get("status") == "running":
            with contextlib.suppress(Exception):
                if runtime.active_action.get("type") == "neb":
                    snapshot["live_activity"] = _build_neb_live_payload(runtime.active_action, runtime.workspace)
                elif runtime.active_action.get("type") == "minimize":
                    snapshot["live_activity"] = _build_minimize_live_payload(runtime.active_action)
                elif runtime.active_action.get("type") == "hawaii":
                    snapshot["live_activity"] = _build_hawaii_live_payload(runtime.active_action, runtime.workspace)
                elif runtime.active_action.get("type") in {"initialize", "apply-reactions", "nanoreactor", "hessian-sample"}:
                    snapshot["live_activity"] = _build_growth_live_payload(runtime.active_action)
        if runtime.workspace is not None and runtime.workspace.queue_fp.exists():
            try:
                snapshot["drive"] = self._drive_payload_cache_lookup(
                    workspace=runtime.workspace,
                    runtime=runtime,
                )
            except Exception as exc:
                snapshot["drive_error"] = _format_user_facing_error(exc)
                try:
                    snapshot["drive"] = _call_drive_payload_builder(
                        _build_drive_payload_fast,
                        runtime.workspace,
                        product_smiles=str((runtime.product or {}).get("smiles") or ""),
                        active_job_label=runtime.busy_label if runtime.future is not None and not runtime.future.done() else "",
                        active_action=runtime.active_action,
                        network_splits=getattr(self, "network_splits", True),
                    )
                except Exception as fast_exc:
                    snapshot["drive_error"] = (
                        f"{snapshot.get('drive_error', '')}; fallback failed: "
                        f"{_format_user_facing_error(fast_exc)}"
                    ).strip("; ")
        return snapshot

    def server_close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.process_executor.shutdown(wait=False, cancel_futures=True)
        super().server_close()


class _DriveHandler(BaseHTTPRequestHandler):
    server: MepdDriveServer

    def _split_request_path(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlsplit(self.path)
        return parsed.path, parse_qs(parsed.query, keep_blank_values=False)

    def _request_token(self) -> str:
        bearer = str(self.headers.get("Authorization") or "").strip()
        if bearer.lower().startswith("bearer "):
            return bearer[7:].strip()
        header_token = str(self.headers.get("X-MEPD-Token") or "").strip()
        if header_token:
            return header_token
        _path, query = self._split_request_path()
        token_values = query.get("token") or []
        if token_values:
            return str(token_values[0]).strip()
        return ""

    def _is_authorized(self) -> bool:
        if not bool(getattr(self.server, "require_api_token", False)):
            return True
        expected = str(getattr(self.server, "api_token", "") or "")
        provided = self._request_token()
        if not expected or not provided:
            return False
        return hmac.compare_digest(provided, expected)

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if length < 0:
            raise ValueError("Invalid negative Content-Length header.")
        max_request_bytes = int(getattr(self.server, "max_request_bytes", 1_000_000) or 1_000_000)
        if length > max_request_bytes:
            raise ValueError(
                f"Request body too large ({length} bytes). Limit is {max_request_bytes} bytes."
            )
        raw = self.rfile.read(length) if length else b"{}"
        if len(raw) > max_request_bytes:
            raise ValueError(
                f"Request body too large ({len(raw)} bytes). Limit is {max_request_bytes} bytes."
            )
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        route_path, _query = self._split_request_path()
        if route_path == "/":
            if not self._is_authorized():
                self._write_json({"error": "Unauthorized."}, HTTPStatus.UNAUTHORIZED)
                return
            body = _drive_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route_path == "/api/state":
            if not self._is_authorized():
                self._write_json({"error": "Unauthorized."}, HTTPStatus.UNAUTHORIZED)
                return
            self._write_json(self.server.snapshot())
            return
        if route_path.startswith("/edge_visualizations/"):
            if not self._is_authorized():
                self._write_json({"error": "Unauthorized."}, HTTPStatus.UNAUTHORIZED)
                return
            with self.server.state_lock:
                workspace = self.server.runtime.workspace
            if workspace is None:
                self.send_error(404)
                return
            edge_dir = workspace.edge_visualizations_dir.resolve()
            target = (edge_dir / Path(route_path).name).resolve()
            if not _is_within_directory(target, edge_dir) or not target.is_file():
                self.send_error(404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            route_path, _query = self._split_request_path()
            if not self._is_authorized():
                self._write_json({"error": "Unauthorized."}, HTTPStatus.UNAUTHORIZED)
                return
            payload = self._read_json()
            if route_path == "/api/initialize":
                self.server.submit_initialize(payload)
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if route_path == "/api/load-workspace":
                self.server.submit_load_workspace(payload)
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if route_path == "/api/minimize":
                self.server.submit_minimize([int(value) for value in payload.get("node_ids", [])])
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if route_path == "/api/apply-reactions":
                self.server.submit_apply_reactions(
                    node_id=int(payload["node_id"]),
                )
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if route_path == "/api/nanoreactor":
                self.server.submit_nanoreactor(
                    node_id=int(payload["node_id"]),
                )
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if route_path == "/api/hessian-sample":
                self.server.submit_hessian_sample(
                    dr=float(payload.get("dr", 1.0)),
                    max_candidates=int(payload.get("max_candidates") or 100),
                    use_bigchem=_coerce_optional_bool(payload.get("use_bigchem")),
                    node_id=(int(payload["node_id"]) if payload.get("node_id") is not None else None),
                    source_node=(int(payload["source_node"]) if payload.get("source_node") is not None else None),
                    target_node=(int(payload["target_node"]) if payload.get("target_node") is not None else None),
                )
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if route_path == "/api/run-neb":
                self.server.submit_run_neb(
                    source_node=int(payload["source_node"]),
                    target_node=int(payload["target_node"]),
                )
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if route_path == "/api/add-edge":
                result = self.server.submit_add_edge(
                    source_node=int(payload["source_node"]),
                    target_node=int(payload["target_node"]),
                    reaction_label=str(payload.get("reaction_label") or ""),
                )
                self._write_json({"ok": True, **result}, HTTPStatus.ACCEPTED)
                return
            if route_path == "/api/add-node":
                result = self.server.submit_add_node(
                    xyz_text=str(payload.get("xyz_text") or ""),
                    charge=int(payload.get("charge", 0) or 0),
                    multiplicity=int(payload.get("multiplicity", 1) or 1),
                )
                self._write_json({"ok": True, **result}, HTTPStatus.ACCEPTED)
                return
            if route_path in {"/api/run-kmc", "/api/run-kinetics"}:
                result = self.server.submit_run_kmc(
                    temperature_kelvin=float(payload.get("temperature_kelvin", 298.15)),
                    final_time=(float(payload["final_time"]) if payload.get("final_time") not in {None, ""} else None),
                    max_steps=int(payload.get("max_steps", 200)),
                    initial_conditions_text=str(payload.get("initial_conditions") or ""),
                )
                self._write_json({"ok": True, **result}, HTTPStatus.OK)
                return
            if route_path == "/api/hawaii-control":
                result = self.server.submit_hawaii_control(
                    mode=str(payload.get("mode") or "go"),
                    discovery_tools=payload.get("discovery_tools"),
                )
                self._write_json({"ok": True, **result}, HTTPStatus.OK)
                return
        except Exception as exc:
            error_message = _format_user_facing_error(exc)
            with contextlib.suppress(Exception):
                self.server._record_request_error(error_message)
            self._write_json({"error": error_message}, HTTPStatus.BAD_REQUEST)
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def launch_mepd_drive(
    *,
    directory: str | None,
    inputs_fp: str | None,
    reactions_fp: str | None = None,
    workspace_path: str | None = None,
    smiles: str | None = None,
    product_smiles: str | None = None,
    start_xyz_fp: str | None = None,
    end_xyz_fp: str | None = None,
    environment_smiles: str = "",
    charge: int = 0,
    multiplicity: int = 1,
    run_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    timeout_seconds: int = 30,
    max_nodes: int = 40,
    max_depth: int = 4,
    max_parallel_nebs: int = 1,
    parallel_autosplit_nebs: bool = False,
    parallel_autosplit_workers: int = 4,
    network_splits: bool = True,
    hawaii: bool = False,
    hawaii_discovery_tools: Any = None,
    open_browser: bool = True,
    startup_progress: Callable[[dict[str, Any]], None] | None = None,
) -> MepdDriveServer:
    resolved_hawaii_discovery_tools = _normalize_hawaii_discovery_tools(
        hawaii_discovery_tools,
        default=_HAWAII_DISCOVERY_TOOL_DEFAULT,
    )
    parallel_autosplit_nebs = bool(parallel_autosplit_nebs)
    parallel_autosplit_workers = max(1, int(parallel_autosplit_workers))
    startup_base_steps = 6 if network_splits else 4
    startup_total_steps = startup_base_steps + 2

    def _emit_startup_progress(message: str, completed_steps: float) -> None:
        if startup_progress is None:
            return
        with contextlib.suppress(Exception):
            startup_progress(
                {
                    "message": message,
                    "completed_steps": float(completed_steps),
                    "total_steps": float(startup_total_steps),
                }
            )

    explicit_directory = Path(directory).resolve() if directory else None
    startup_workspace_path = workspace_path
    deferred_initialize_payload: dict[str, Any] | None = None
    reactant_xyz_text = _read_xyz_file_text(start_xyz_fp, label="Start") if start_xyz_fp else ""
    product_xyz_text = _read_xyz_file_text(end_xyz_fp, label="End") if end_xyz_fp else ""
    has_xyz_bootstrap = bool(reactant_xyz_text)
    require_api_token = not _is_loopback_host(host)
    api_token = secrets.token_urlsafe(24) if require_api_token else None
    if (
        startup_workspace_path is None
        and explicit_directory is not None
        and (
            (explicit_directory / "workspace.json").exists()
            or _resolve_network_splits_pot_fp(explicit_directory) is not None
        )
    ):
        startup_workspace_path = str(explicit_directory)

    initial_state: dict[str, Any] | None = None
    if startup_workspace_path:
        workspace_dir = Path(startup_workspace_path).expanduser().resolve()
        if workspace_dir.is_file() and (
            workspace_dir.name == "workspace.json"
            or workspace_dir.name.endswith("_network.json")
            or workspace_dir.name.endswith("_request_manifest.json")
        ):
            workspace_dir = workspace_dir.parent
        base_directory = workspace_dir.parent
        initial_state = _load_existing_workspace_job_compat(
            str(workspace_dir),
            network_splits=network_splits,
            progress=startup_progress,
        )
        _emit_startup_progress("Workspace snapshot loaded. Starting Drive server...", startup_base_steps)
    elif smiles or has_xyz_bootstrap:
        resolved_inputs = Path(str(inputs_fp)).expanduser().resolve() if inputs_fp else None
        if resolved_inputs is None:
            raise ValueError(
                "An inputs TOML path is required to start drive from SMILES or XYZ endpoints."
            )
        reactant_payload = _resolve_species_input(
            smiles=str(smiles or "").strip(),
            xyz_text=reactant_xyz_text,
            charge=int(charge),
            multiplicity=int(multiplicity),
        )
        if not reactant_payload:
            raise ValueError(
                "A bootstrap reactant is required. Provide --smiles, --start-xyz-fp, or both."
            )
        product_payload = _resolve_species_input(
            smiles=str(product_smiles or "").strip(),
            xyz_text=product_xyz_text,
            charge=int(charge),
            multiplicity=int(multiplicity),
        )
        seed_only = (
            bool(product_payload)
            or bool(reactant_xyz_text.strip())
            or bool(product_xyz_text.strip())
        )
        resolved_run_name = _validate_run_name(
            str(run_name or "").strip() or f"mepd-drive-{int(time.time())}"
        )
        if explicit_directory is not None:
            run_dir = explicit_directory
            base_directory = run_dir.parent
        else:
            base_directory = (Path.cwd() / "mepd-drive").resolve()
            run_dir = (base_directory / resolved_run_name).resolve()
        if hawaii:
            deferred_initialize_payload = {
                "mode": "reactant-product" if bool(product_payload) else "reactant",
                "reactant_smiles": str(reactant_payload.get("smiles") or "").strip(),
                "reactant_xyz": reactant_xyz_text,
                "product_smiles": str(product_payload.get("smiles") or "").strip() if product_payload else "",
                "product_xyz": product_xyz_text,
                "run_name": resolved_run_name,
                "inputs_fp": str(resolved_inputs),
                "reactions_fp": str(Path(reactions_fp).expanduser().resolve()) if reactions_fp else "",
                "environment_smiles": str(environment_smiles or ""),
                "charge": int(charge),
                "multiplicity": int(multiplicity),
                "seed_only": bool(seed_only),
                "auto_start_hawaii": True,
                "auto_start_hawaii_discovery_tools": list(resolved_hawaii_discovery_tools),
            }
            initial_state = None
        else:
            initial_state = _initialize_workspace_job(
                reactant=reactant_payload,
                product=(product_payload or None),
                run_name=resolved_run_name,
                workspace_dir=str(run_dir),
                inputs_fp=str(resolved_inputs),
                reactions_fp=str(Path(reactions_fp).expanduser().resolve()) if reactions_fp else None,
                environment_smiles=str(environment_smiles or ""),
                timeout_seconds=timeout_seconds,
                max_nodes=max_nodes,
                max_depth=max_depth,
                max_parallel_nebs=max_parallel_nebs,
                parallel_autosplit_nebs=parallel_autosplit_nebs,
                parallel_autosplit_workers=parallel_autosplit_workers,
                seed_only=bool(seed_only),
                network_splits=network_splits,
                progress_fp=None,
                allowed_base_dir=str(base_directory.resolve()),
            )
    else:
        base_directory = explicit_directory if explicit_directory is not None else (Path.cwd() / "mepd-drive").resolve()
    base_directory.mkdir(parents=True, exist_ok=True)
    _emit_startup_progress("Initializing Drive server...", startup_base_steps + 1)
    server = MepdDriveServer(
        (host, int(port)),
        base_directory=base_directory,
        inputs_fp=Path(inputs_fp).expanduser().resolve() if inputs_fp else None,
        reactions_fp=Path(reactions_fp).resolve() if reactions_fp else None,
        timeout_seconds=timeout_seconds,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_parallel_nebs=max_parallel_nebs,
        parallel_autosplit_nebs=parallel_autosplit_nebs,
        parallel_autosplit_workers=parallel_autosplit_workers,
        network_splits=network_splits,
        hawaii_discovery_tools=list(resolved_hawaii_discovery_tools),
        initial_state=initial_state,
        require_api_token=require_api_token,
        api_token=api_token,
    )
    _emit_startup_progress("Drive server ready.", startup_total_steps)
    if open_browser:
        url = f"http://{host}:{server.server_address[1]}/"
        if require_api_token and api_token:
            url = f"{url}?token={api_token}"
        webbrowser.open(url)
    if deferred_initialize_payload is not None and hasattr(server, "submit_initialize"):
        server.submit_initialize(deferred_initialize_payload)
    elif hawaii and initial_state is not None and hasattr(server, "submit_hawaii"):
        server.submit_hawaii()
    return server
