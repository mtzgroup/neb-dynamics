from __future__ import annotations

import base64
import contextlib
import json
import multiprocessing
import os
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

from qcio import Structure

from neb_dynamics.constants import ANGSTROM_TO_BOHR
from neb_dynamics.inputs import RunInputs
from neb_dynamics.molecule import Molecule
from neb_dynamics.pot import Pot
from neb_dynamics.qcio_structure_helpers import molecule_to_structure, structure_to_molecule
from neb_dynamics.rdkit_draw import moldrawsvg
from neb_dynamics.retropaths_queue import (
    _identical_endpoint_skip_reason,
    _ensure_pair_endpoints_optimized,
    _make_pair_chain,
    _run_single_item_worker,
    RetropathsNEBQueue,
    build_retropaths_neb_queue,
)
from neb_dynamics.retropaths_workflow import (
    _build_network_explorer_payload,
    _json_safe,
    _load_template_payloads,
    _persist_endpoint_optimization_result,
    _quiet_force_smiles,
    _strip_cached_result,
    _write_edge_visualizations,
    RetropathsWorkspace,
    create_workspace,
    load_partial_annotated_pot,
    load_retropaths_pot,
    prepare_neb_workspace,
    summarize_queue,
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
        if attrs.get("td") is None:
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
    return True, ""


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
    return {
        "type": "neb",
        "title": active_action.get("label") or "Running autosplitting NEB",
        "plot": live_chain.get("plot") if live_chain else None,
        "history": history,
        "ascii_plot": live_chain.get("ascii_plot") if live_chain else "",
        "console_text": _read_log_tail(active_action.get("progress_fp"), max_chars=6000),
        "reactant_structure": reactant_structure,
        "product_structure": product_structure,
        "note": "Live optimization-history view. Faded curves are earlier optimization steps; the highlighted curve is the latest chain.",
    }


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


def _drive_network_version(workspace: RetropathsWorkspace) -> str:
    fingerprints: list[str] = []
    for fp in filter(
        None,
        (
            getattr(workspace, "neb_pot_fp", None),
            getattr(workspace, "queue_fp", None),
            getattr(workspace, "annotated_neb_pot_fp", None),
        ),
    ):
        if fp.exists():
            stat = fp.stat()
            fingerprints.append(f"{fp.name}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            fingerprints.append(f"{fp.name}:missing")
    return "|".join(fingerprints)
    try:
        return json.loads(Path(str(chain_fp)).read_text())
    except Exception:
        return None


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


def _merge_drive_pot(workspace: RetropathsWorkspace) -> Pot:
    base_pot = Pot.read_from_disk(workspace.neb_pot_fp)
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


def _neb_backed_nodes(graph) -> set[int]:
    backed: set[int] = set()
    for source, target in graph.edges:
        attrs = graph.edges[(source, target)]
        if attrs.get("list_of_nebs"):
            backed.add(int(source))
            backed.add(int(target))
    return backed


def _build_drive_payload(
    workspace: RetropathsWorkspace,
    *,
    product_smiles: str = "",
    active_job_label: str = "",
    active_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retropaths_pot = load_retropaths_pot(workspace)
    queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)
    pot = _merge_drive_pot(workspace)
    edge_visualizations = _write_edge_visualizations(workspace=workspace, pot=pot)
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
        node["neb_backed"] = node_index in backed_nodes
        node["is_target"] = bool(normalized_product and str(node["label"]) == normalized_product)

    for edge in explorer["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        attrs = pot.graph.edges[(source, target)]
        queue_item = queue_by_edge.get((source, target))
        edge["neb_backed"] = bool(attrs.get("list_of_nebs"))
        edge["queue_status"] = queue_item.status if queue_item is not None else ""
        edge["queue_error"] = queue_item.error if queue_item is not None else ""
        edge["can_queue_neb"], edge["queue_note"] = _edge_neb_status(edge)
        edge["source_structure"] = _node_structure_payload(pot.graph.nodes[source])
        edge["target_structure"] = _node_structure_payload(pot.graph.nodes[target])

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
            "retropaths_nodes": int(retropaths_pot.graph.number_of_nodes()),
            "retropaths_edges": int(retropaths_pot.graph.number_of_edges()),
            "network_nodes": int(pot.graph.number_of_nodes()),
            "network_edges": int(pot.graph.number_of_edges()),
            "optimized_endpoints": int(optimized_endpoints),
            "neb_backed_edges": int(neb_backed_edges),
        },
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
) -> dict[str, Any]:
    queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)
    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    normalized_product = product_smiles.strip()
    explorer = _build_network_explorer_payload(pot.graph)

    queue_by_edge = {
        (int(item.source_node), int(item.target_node)): item
        for item in queue.items
    }
    for node in explorer["nodes"]:
        node_index = int(node["id"])
        attrs = pot.graph.nodes[node_index]
        node["structure"] = _node_structure_payload(attrs)
        node["endpoint_optimized"] = bool(attrs.get("endpoint_optimized"))
        node["endpoint_optimization_error"] = str(attrs.get("endpoint_optimization_error") or "")
        node["minimizable"], node["minimize_note"] = _node_minimize_status(node_index, attrs)
        node["neb_backed"] = False
        node["is_target"] = bool(normalized_product and str(node["label"]) == normalized_product)

    for edge in explorer["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        queue_item = queue_by_edge.get((source, target))
        edge["neb_backed"] = False
        edge["queue_status"] = queue_item.status if queue_item is not None else ""
        edge["queue_error"] = queue_item.error if queue_item is not None else ""
        edge["can_queue_neb"], edge["queue_note"] = _edge_neb_status(edge)
        edge["source_structure"] = _node_structure_payload(pot.graph.nodes[source])
        edge["target_structure"] = _node_structure_payload(pot.graph.nodes[target])

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
            "neb_backed_edges": 0,
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
) -> dict[str, Any]:
    queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)
    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    normalized_product = product_smiles.strip()
    explorer = _build_network_explorer_payload(pot.graph)

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
        node["structure"] = _node_structure_payload(attrs)
        node["endpoint_optimized"] = bool(attrs.get("endpoint_optimized"))
        node["endpoint_optimization_error"] = str(attrs.get("endpoint_optimization_error") or "")
        node["minimizable"], node["minimize_note"] = _node_minimize_status(node_index, attrs)
        node["neb_backed"] = False
        node["is_target"] = bool(normalized_product and str(node["label"]) == normalized_product)

    for edge in explorer["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        queue_item = queue_by_edge.get((source, target))
        edge["neb_backed"] = bool((source, target) == active_edge)
        edge["queue_status"] = queue_item.status if queue_item is not None else ""
        edge["queue_error"] = queue_item.error if queue_item is not None else ""
        edge["can_queue_neb"], edge["queue_note"] = _edge_neb_status(edge)
        edge["source_structure"] = _node_structure_payload(pot.graph.nodes[source])
        edge["target_structure"] = _node_structure_payload(pot.graph.nodes[target])

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
            "neb_backed_edges": 0,
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
    timeout_seconds: int,
    max_nodes: int,
    max_depth: int,
    max_parallel_nebs: int,
) -> dict[str, Any]:
    workspace_path = Path(workspace_dir).resolve()
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace = create_workspace(
        root_smiles=reactant["smiles"],
        environment_smiles="",
        inputs_fp=inputs_fp,
        reactions_fp=reactions_fp,
        name=run_name,
        directory=workspace_path,
        timeout_seconds=timeout_seconds,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_parallel_nebs=max_parallel_nebs,
    )
    prepare_neb_workspace(workspace)
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
        },
        "reactant": reactant,
        "product": product,
        "message": f"Initialized workspace {workspace.run_name}.",
    }


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


def _run_selected_edge_neb(
    workspace: RetropathsWorkspace,
    *,
    source_node: int,
    target_node: int,
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

    try:
        result = _run_single_item_worker(
            pair,
            run_inputs,
            str(result_dir),
            str(output_chain_fp),
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
    load_partial_annotated_pot(workspace)
    return {"message": f"Autosplitting NEB completed for edge {source_node} -> {target_node}."}


def _run_selected_edge_neb_logged(
    workspace_data: dict[str, Any],
    *,
    source_node: int,
    target_node: int,
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


def _load_existing_workspace_job(workspace_path: str) -> dict[str, Any]:
    path = Path(workspace_path).expanduser().resolve()
    workspace_dir = path.parent if path.is_file() and path.name == "workspace.json" else path
    workspace = RetropathsWorkspace.read(workspace_dir)
    required = [
        workspace.workspace_fp,
        workspace.neb_pot_fp,
        workspace.queue_fp,
        workspace.retropaths_pot_fp,
    ]
    missing = [str(fp) for fp in required if not fp.exists()]
    if missing:
        raise ValueError(f"Workspace is missing required files: {', '.join(missing)}")
    # Rebuild the annotated overlay immediately so completed NEB results
    # are visible as soon as the workspace is opened in drive.
    load_partial_annotated_pot(workspace)
    return {
        "workspace": workspace.__dict__,
        "reactant": {"smiles": workspace.root_smiles},
        "product": None,
        "message": f"Loaded existing workspace {workspace.run_name}.",
    }


def _drive_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MEPD Drive</title>
  <style>
    :root {
      --bg: #f2ede5;
      --panel: #fffdf8;
      --line: #d7c8b2;
      --ink: #241d18;
      --muted: #6f6558;
      --accent: #b9652a;
      --accent-2: #1f6a57;
      --backed: #2f7b61;
      --warn: #b03a2e;
    }
    body { margin: 0; font-family: Georgia, serif; background: radial-gradient(circle at top, #fbf7f1, var(--bg)); color: var(--ink); }
    .shell { padding: 20px; display: grid; gap: 16px; }
    .panel { background: var(--panel); border: 1px solid var(--line); padding: 16px; }
    .hero { display: grid; grid-template-columns: 1.1fr 1.7fr; gap: 16px; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; }
    .stat { border: 1px solid var(--line); padding: 10px; background: #fbf7ef; }
    .muted { color: var(--muted); }
    textarea, input, select, button { font: inherit; }
    textarea, input[type="text"], input[type="number"] { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid var(--line); background: #fffdfa; }
    textarea { min-height: 96px; resize: vertical; }
    button { padding: 10px 14px; border: 1px solid #ad875d; background: #f5e8d3; cursor: pointer; }
    button.primary { background: var(--accent); color: #fff9f3; border-color: #8f4b1c; }
    button.secondary { background: #e5f0eb; border-color: #8ab0a3; }
    button:disabled { opacity: 0.55; cursor: default; }
    .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .graph-grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(360px, 1fr); gap: 16px; }
    .explorer-svg { width: 100%; min-height: 620px; border: 1px solid var(--line); background: linear-gradient(180deg, #fffdf8, #f8f0e0); }
    .network-edge-line { stroke: #8f8271; stroke-width: 2.2; fill: none; }
    .network-edge-line.neb-backed { stroke: var(--backed); stroke-width: 3.4; }
    .network-edge-line.selected { stroke: var(--accent); stroke-width: 4.5; }
    .network-edge-hitbox { stroke: transparent; stroke-width: 18; fill: none; cursor: pointer; }
    .network-node { fill: #7b6a57; stroke: #fdf8ef; stroke-width: 2; cursor: pointer; }
    .network-node.root { fill: #8e4d1c; }
    .network-node.neb-backed { fill: var(--backed); }
    .network-node.target { fill: #81467b; }
    .network-node.selected { fill: #d79e35; stroke: #5f3706; stroke-width: 3; }
    .network-label { font-size: 11px; fill: #2d241c; pointer-events: none; }
    .detail-tabs { display: flex; gap: 8px; margin: 10px 0 14px; flex-wrap: wrap; }
    .detail-tab { padding: 7px 10px; border: 1px solid var(--line); background: #f4ecdf; cursor: pointer; }
    .detail-tab.active { background: var(--accent); color: white; border-color: #8f4b1c; }
    .detail-panel { display: none; }
    .detail-panel.active { display: block; }
    .viewer-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    iframe.structure { width: 100%; height: 320px; border: 1px solid var(--line); background: white; }
    .mol-card { width: 100%; min-height: 320px; border: 1px solid var(--line); background: white; display: grid; align-content: start; }
    .mol-card svg { width: 100%; height: auto; display: block; }
    .mol-empty { padding: 18px 14px; color: var(--muted); }
    .mol-meta { padding: 10px 12px 12px; border-top: 1px solid var(--line); font-size: 12px; color: var(--muted); word-break: break-word; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; background: #f7f1e9; border: 1px solid #e4d7c5; padding: 12px; }
    .badge { display: inline-block; padding: 2px 8px; border: 1px solid var(--line); background: #faf2e5; margin-right: 6px; }
    .message { padding: 10px 12px; border: 1px solid var(--line); background: #faf5eb; }
    .live-activity { margin-top: 12px; border: 1px solid var(--line); background: #fbf7ef; padding: 12px; }
    .live-activity svg { width: 100%; height: auto; border: 1px solid var(--line); background: white; }
    .live-activity pre { margin-top: 10px; font-size: 12px; max-height: 220px; overflow: auto; }
    .live-neb-layout { display: grid; grid-template-columns: 190px minmax(0, 1fr) 190px; gap: 12px; align-items: start; }
    .live-neb-layout .mol-card { min-height: 210px; }
    .job-list { display: grid; gap: 8px; margin-top: 12px; }
    .job-row { border: 1px solid var(--line); background: #fffdf8; padding: 8px 10px; }
    .job-row.running { border-color: #1f6a57; }
    .job-row.completed { border-color: #8ab0a3; }
    .job-row.failed { border-color: #b03a2e; }
    @media (max-width: 1080px) {
      .hero, .graph-grid, .form-grid, .viewer-grid { grid-template-columns: 1fr; }
      .explorer-svg { min-height: 480px; }
      .live-neb-layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="panel">
        <h1 style="margin-top:0;">MEPD Drive</h1>
        <div class="muted" style="margin-bottom:12px;">Initialize a Retropaths workspace from reactant-only or reactant/product inputs, inspect the live network, and queue geometry minimizations or autosplitting NEBs directly from the graph.</div>
        <div id="job-banner" class="message">Idle.</div>
        <div id="job-subtext" class="muted" style="margin-top:8px;">No action submitted yet.</div>
      </div>
      <div class="panel">
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
          <button id="initialize" class="primary">Build Retropaths Network</button>
          <button id="load-workspace" class="secondary">Load Existing Workspace</button>
          <button id="minimize-all" class="secondary">Queue Minimization For All Geometries</button>
        </div>
      </div>
    </div>

    <div class="panel">
      <div id="workspace-summary" class="muted">No workspace initialized yet.</div>
      <div id="stats" class="stats" style="margin-top:12px;"></div>
      <div id="live-activity-panel" class="live-activity" style="display:none;"></div>
    </div>

    <div class="graph-grid">
      <div class="panel">
        <svg id="network-svg" class="explorer-svg" viewBox="0 0 1180 680" role="img" aria-label="MEPD Drive network graph"></svg>
      </div>
      <div class="panel">
        <h2 id="detail-title" style="margin-top:0;">Select a node or edge</h2>
        <div id="detail-summary" class="muted">Click a node to inspect its geometry or click an edge to inspect the targeted reaction, template data, and queue NEB work.</div>
        <div class="detail-tabs">
          <button class="detail-tab active" data-tab="targeted">Targeted Reaction</button>
          <button class="detail-tab" data-tab="template-data">Template Data</button>
          <button class="detail-tab" data-tab="structures">3D Structures</button>
        </div>
        <div id="panel-targeted" class="detail-panel active"></div>
        <div id="panel-template-data" class="detail-panel"></div>
        <div id="panel-structures" class="detail-panel"></div>
      </div>
    </div>
  </div>

  <script>
    const ns = "http://www.w3.org/2000/svg";
    const state = {
      snapshot: null,
      selected: null,
      networkVersion: "",
    };

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
        return "<html><body style='font-family:Georgia,serif;padding:12px;color:#665;'>No 3D structure available.</body></html>";
      }
      return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>
    html, body, #viewer { margin: 0; width: 100%; height: 100%; background: white; overflow: hidden; }
    #status { padding: 12px; color: #666; font-family: Georgia, serif; }
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
      const viewer = $3Dmol.createViewer(host, { backgroundColor: "white" });
      viewer.addModel(xyz, "xyz");
      viewer.setStyle({}, { stick: {}, sphere: { scale: 0.32 } });
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

    function setButtonsDisabled(disabled) {
      ["initialize", "load-workspace", "minimize-all"].forEach((id) => {
        const elem = document.getElementById(id);
        if (elem) elem.disabled = disabled;
      });
      document.querySelectorAll("[data-drive-action]").forEach((elem) => {
        elem.disabled = disabled;
      });
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      });
      const data = await response.json().catch(() => ({}));
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

    function renderWorkspaceSummary(snapshot) {
      const el = document.getElementById("workspace-summary");
      if (!snapshot || !snapshot.initialized || !snapshot.drive) {
        el.textContent = "No workspace initialized yet.";
        return;
      }
      const workspace = snapshot.drive.workspace;
      el.innerHTML = `
        <span class="badge">${escapeHtml(workspace.run_name)}</span>
        <span class="badge">${escapeHtml(workspace.root_smiles)}</span>
        ${snapshot.product?.smiles ? `<span class="badge">Target: ${escapeHtml(snapshot.product.smiles)}</span>` : ""}
        <div style="margin-top:8px;"><strong>Workspace:</strong> ${escapeHtml(workspace.directory)}</div>
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
              <line x1="${margin.left}" y1="${sy(tick)}" x2="${width - margin.right}" y2="${sy(tick)}" stroke="#e1ddd6" stroke-dasharray="3 3" />
              <line x1="${margin.left - 6}" y1="${sy(tick)}" x2="${margin.left}" y2="${sy(tick)}" stroke="#555" />
              <text x="${margin.left - 10}" y="${sy(tick) + 4}" text-anchor="end" font-size="11" fill="#555">${escapeHtml(Number(tick).toFixed(2))}</text>
            </g>
          `).join("")}
          <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#555" />
          <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#555" />
          <polyline fill="none" stroke="#1f6a57" stroke-width="2.5" points="${points}" />
          ${xVals.map((x, i) => `<circle cx="${sx(x)}" cy="${sy(yVals[i])}" r="3.5" fill="#b9652a" />`).join("")}
          <text x="${width / 2}" y="18" text-anchor="middle" font-size="14" fill="#222">${escapeHtml(title || "Trajectory plot")}</text>
          <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#444">${escapeHtml(xLabel)}</text>
          <text x="16" y="${height / 2}" transform="rotate(-90 16 ${height / 2})" text-anchor="middle" font-size="12" fill="#444">${escapeHtml(yLabel)}</text>
        </svg>
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
              <line x1="${margin.left}" y1="${sy(tick)}" x2="${width - margin.right}" y2="${sy(tick)}" stroke="#e1ddd6" stroke-dasharray="3 3" />
              <line x1="${margin.left - 6}" y1="${sy(tick)}" x2="${margin.left}" y2="${sy(tick)}" stroke="#555" />
              <text x="${margin.left - 10}" y="${sy(tick) + 4}" text-anchor="end" font-size="11" fill="#555">${escapeHtml(Number(tick).toFixed(2))}</text>
            </g>
          `).join("")}
          <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#555" />
          <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#555" />
          ${backgroundCurves.map((curve) => {
            const pts = (curve.x || []).map((x, i) => `${sx(x).toFixed(2)},${sy(curve.y[i]).toFixed(2)}`).join(" ");
            return `<polyline fill="none" stroke="#8f8271" stroke-width="1.4" opacity="0.22" points="${pts}" />`;
          }).join("")}
          ${foreground ? `<polyline fill="none" stroke="#1f6a57" stroke-width="2.8" points="${(foreground.x || []).map((x, i) => `${sx(x).toFixed(2)},${sy(foreground.y[i]).toFixed(2)}`).join(" ")}" />` : ""}
          ${foreground ? (foreground.x || []).map((x, i) => `<circle cx="${sx(x)}" cy="${sy(foreground.y[i])}" r="3.2" fill="#b9652a" />`).join("") : ""}
          <text x="${width / 2}" y="18" text-anchor="middle" font-size="14" fill="#222">${escapeHtml(activity?.title || "Live NEB optimization history")}</text>
          <text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-size="12" fill="#444">Integrated path length</text>
          <text x="16" y="${height / 2}" transform="rotate(-90 16 ${height / 2})" text-anchor="middle" font-size="12" fill="#444">Energy</text>
        </svg>
      `;
    }

    function renderLiveActivity(snapshot) {
      const panel = document.getElementById("live-activity-panel");
      const activity = snapshot?.live_activity || null;
      if (!panel) return;
      if (!activity) {
        panel.style.display = "none";
        panel.innerHTML = "";
        return;
      }
      panel.style.display = "block";
      if (activity.type === "minimize") {
        const jobs = Array.isArray(activity.jobs) ? activity.jobs : [];
        panel.innerHTML = `
          <div style="font-weight:600; margin-bottom:8px;">Geometry Optimization Monitor</div>
          <div class="muted" style="margin-bottom:10px;">${escapeHtml(activity.title || "Running geometry minimizations")}</div>
          ${activity.plot ? drawLinePlotSvg(activity.plot, "Geometry optimization trajectory") : `<div class="muted">${escapeHtml(activity.note || "Waiting for geometry optimization updates...")}</div>`}
          <div class="muted" style="margin-top:10px;">${escapeHtml(activity.note || "")}</div>
          <div class="job-list">
            ${jobs.map((job) => `
              <div class="job-row ${escapeHtml(job.status || "pending")}">
                <div><strong>Node ${escapeHtml(job.node_id)}</strong> <span class="badge">${escapeHtml(job.status || "pending")}</span></div>
                ${job.error ? `<div style="color:#b03a2e; margin-top:4px;">${escapeHtml(job.error)}</div>` : ""}
              </div>
            `).join("")}
          </div>
        `;
        return;
      }
      panel.innerHTML = `
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
        ${activity.console_text ? `<pre>${escapeHtml(activity.console_text)}</pre>` : ""}
      `;
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
        return;
      }

      if (selection.kind === "node") {
        const node = selection.node;
        title.textContent = `Node ${node.id}`;
        summary.innerHTML = `
          <div><strong>Label:</strong> ${escapeHtml(node.label || node.id)}</div>
          <div><strong>Endpoint optimized:</strong> ${node.endpoint_optimized ? "yes" : "no"}</div>
          ${node.endpoint_optimization_error ? `<div><strong>Last optimization error:</strong> <span style="color:#b03a2e;">${escapeHtml(node.endpoint_optimization_error)}</span></div>` : ""}
          <div><strong>Can queue minimization:</strong> ${node.minimizable ? "yes" : "no"}</div>
          ${node.minimize_note ? `<div><strong>Minimization note:</strong> ${escapeHtml(node.minimize_note)}</div>` : ""}
          <div><strong>NEB-backed:</strong> ${node.neb_backed ? "yes" : "no"}</div>
        `;
        targeted.innerHTML = `
          <div style="margin-bottom:10px;"><button class="secondary" data-drive-action="minimize-node" onclick="queueMinimizeNode(${Number(node.id)})" ${node.minimizable ? "" : "disabled"}>Queue Minimization For This Geometry</button></div>
          ${node.minimize_note ? `<div style="margin-bottom:10px; color:${node.minimizable ? "#6f6558" : "#b03a2e"};">${escapeHtml(node.minimize_note)}</div>` : ""}
          <pre>${escapeHtml(JSON.stringify(node.data || {}, null, 2))}</pre>
        `;
        templateData.innerHTML = `<pre>${escapeHtml(JSON.stringify(node.data || {}, null, 2))}</pre>`;
        structures.innerHTML = node.structure?.xyz_b64
          ? `<iframe class="structure" srcdoc="${escapeHtml(makeStructureSrcdoc(node.structure.xyz_b64))}"></iframe>`
          : `<div class="muted">No 3D structure is available for this node.</div>`;
        return;
      }

      const edge = selection.edge;
      const template = edge.template || {};
      title.textContent = `Edge ${edge.source} → ${edge.target}`;
      summary.innerHTML = `
        <div><strong>Reaction:</strong> ${escapeHtml(edge.reaction || "Unknown")}</div>
        <div><strong>Queue status:</strong> ${escapeHtml(edge.queue_status || "not queued")}</div>
        <div><strong>NEB-backed:</strong> ${edge.neb_backed ? "yes" : "no"}</div>
        ${edge.queue_note ? `<div><strong>Queue note:</strong> ${escapeHtml(edge.queue_note)}</div>` : ""}
        ${edge.barrier == null ? "" : `<div><strong>Barrier:</strong> ${escapeHtml(Number(edge.barrier).toFixed(3))}</div>`}
      `;
      targeted.innerHTML = `
        <div style="margin-bottom:10px;">
          <button class="primary" data-drive-action="run-neb" onclick="queueEdgeNeb(${Number(edge.source)}, ${Number(edge.target)})" ${edge.can_queue_neb ? "" : "disabled"}>Queue Autosplitting NEB For This Edge</button>
        </div>
        <div style="margin-bottom:10px;">
          <strong>Targeted reaction:</strong> ${escapeHtml(edge.reaction || "Unknown")}
        </div>
        ${edge.queue_note ? `<div style="margin-bottom:10px; color: ${edge.can_queue_neb ? "#6f6558" : "#b03a2e"};"><strong>${edge.can_queue_neb ? "Queue note" : "Edge cannot run as-is"}:</strong> ${escapeHtml(edge.queue_note)}</div>` : ""}
        ${edge.viewer_href ? `<div style="margin-bottom:10px;"><a href="${escapeHtml(edge.viewer_href)}" target="_blank" rel="noreferrer">Open completed NEB viewer</a></div>` : ""}
        <pre>${escapeHtml(JSON.stringify(edge.data || {}, null, 2))}</pre>
      `;
      templateData.innerHTML = template.data
        ? `<pre>${escapeHtml(JSON.stringify(template.data, null, 2))}</pre>`
        : `<div class="muted">No template data was available for this reaction.</div>`;
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
    }

    function selectTab(tabName) {
      document.querySelectorAll(".detail-tab").forEach((button) => {
        button.classList.toggle("active", button.getAttribute("data-tab") === tabName);
      });
      document.querySelectorAll(".detail-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `panel-${tabName}`);
      });
    }

    document.querySelectorAll(".detail-tab").forEach((button) => {
      button.addEventListener("click", () => selectTab(button.getAttribute("data-tab")));
    });

    function renderNetwork(snapshot) {
      const svg = document.getElementById("network-svg");
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      if (!snapshot || !snapshot.initialized || !snapshot.drive?.network) {
        renderDetail(null);
        return;
      }

      const payload = snapshot.drive.network;
      const nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
      const edges = Array.isArray(payload.edges) ? payload.edges : [];
      if (!nodes.length) {
        renderDetail(null);
        return;
      }

      const width = 1180;
      const height = 680;
      const make = (tag) => document.createElementNS(ns, tag);
      const nodeElems = new Map();
      const edgeElems = [];

      const simNodes = nodes.map((node, index) => ({
        id: Number(node.id),
        node,
        x: width / 2 + ((index % 6) - 3) * 28,
        y: height / 2 + (Math.floor(index / 6) - 3) * 28,
      }));
      const simById = new Map(simNodes.map((node) => [node.id, node]));

      function setSelection(selection) {
        state.selected = selection;
        edgeElems.forEach((item) => item.line.classList.toggle("selected", selection?.kind === "edge" && item.edge === selection.edge));
        nodeElems.forEach((circle, nodeId) => {
          circle.classList.toggle("selected", selection?.kind === "node" && Number(selection.node.id) === Number(nodeId));
        });
        renderDetail(selection);
      }

      edges.forEach((edge) => {
        const group = make("g");
        const hitbox = make("line");
        hitbox.setAttribute("class", "network-edge-hitbox");
        const line = make("line");
        line.setAttribute("class", `network-edge-line${edge.neb_backed ? " neb-backed" : ""}`);
        group.appendChild(hitbox);
        group.appendChild(line);
        group.addEventListener("click", () => setSelection({ kind: "edge", edge }));
        svg.appendChild(group);
        edgeElems.push({ edge, line, hitbox });
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
        group.addEventListener("click", () => setSelection({ kind: "node", node }));
        svg.appendChild(group);
        nodeElems.set(Number(node.id), circle);
        simById.get(Number(node.id)).group = group;
      });

      if (window.d3 && typeof d3.layout?.force === "function") {
        const force = d3.layout.force()
          .size([width, height])
          .nodes(simNodes)
          .links(edges.map((edge) => ({
            source: simById.get(Number(edge.source)),
            target: simById.get(Number(edge.target)),
          })))
          .charge(Math.max(-1200, -170 - nodes.length * 14))
          .linkDistance(Math.max(100, Math.min(240, 120 + nodes.length * 1.3)))
          .gravity(0.05)
          .friction(0.84)
          .on("tick", () => {
            edgeElems.forEach((item) => {
              const source = simById.get(Number(item.edge.source));
              const target = simById.get(Number(item.edge.target));
              item.line.setAttribute("x1", String(source.x));
              item.line.setAttribute("y1", String(source.y));
              item.line.setAttribute("x2", String(target.x));
              item.line.setAttribute("y2", String(target.y));
              item.hitbox.setAttribute("x1", String(source.x));
              item.hitbox.setAttribute("y1", String(source.y));
              item.hitbox.setAttribute("x2", String(target.x));
              item.hitbox.setAttribute("y2", String(target.y));
            });
            simNodes.forEach((simNode) => {
              simNode.x = Math.max(28, Math.min(width - 28, simNode.x));
              simNode.y = Math.max(28, Math.min(height - 36, simNode.y));
              if (simNode.group) simNode.group.setAttribute("transform", `translate(${simNode.x},${simNode.y})`);
            });
          });
        force.start();
        for (let i = 0; i < Math.max(110, nodes.length * 7); i += 1) force.tick();
        force.stop();
      }

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

    async function refreshState() {
      try {
        const response = await fetch("/api/state");
        const snapshot = await response.json();
        state.snapshot = snapshot;
        const activeAction = snapshot.active_action || null;
        const activeLabel = activeAction && activeAction.status === "running"
          ? (activeAction.label || snapshot.busy_label)
          : snapshot.busy_label;
        setBanner(snapshot.last_message || (snapshot.busy ? `Running: ${activeLabel}` : "Idle."), Boolean(snapshot.last_error));
        if (snapshot.last_error) setBanner(snapshot.last_error, true);
        if (activeAction && activeAction.status === "running") {
          if (activeAction.type === "minimize") {
            setSubtext("Geometry optimization is running. Updated node structures are written back one-by-one and will appear here as polling refreshes.");
          } else if (activeAction.type === "neb") {
            setSubtext("Autosplitting NEB is running. The edge state and any discovered intermediates will appear after the backend writes them back.");
          } else {
            setSubtext("Background work is running. The network and counters will refresh automatically.");
          }
        } else {
          setSubtext("Idle.");
        }
        setButtonsDisabled(Boolean(snapshot.busy));
        renderWorkspaceSummary(snapshot);
        renderStats(snapshot);
        renderLiveActivity(snapshot);
        const version = snapshot.drive?.version || "";
        if (version !== state.networkVersion) {
          state.networkVersion = version;
          renderNetwork(snapshot);
        }
      } catch (error) {
        setBanner(error.message || String(error), true);
      }
    }

    async function initializeDrive() {
      const mode = document.getElementById("mode").value;
      try {
        setBanner("Submitting initialization request...");
        setSubtext("Preparing a fresh workspace and building the Retropaths network.");
        await postJson("/api/initialize", {
          mode,
          run_name: document.getElementById("run-name").value,
          reactant_smiles: document.getElementById("reactant-smiles").value,
          reactant_xyz: document.getElementById("reactant-xyz").value,
          product_smiles: document.getElementById("product-smiles").value,
          product_xyz: document.getElementById("product-xyz").value,
        });
        setBanner("Initialization request accepted.");
        await refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The request failed before it could start.");
      }
    }

    async function loadWorkspace() {
      try {
        setBanner("Loading existing workspace...");
        setSubtext("Reading the workspace files and restoring the live network.");
        await postJson("/api/load-workspace", {
          workspace_path: document.getElementById("workspace-path").value,
        });
        setBanner("Workspace load request accepted.");
        await refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The workspace could not be loaded.");
      }
    }

    async function queueMinimizeNode(nodeId) {
      try {
        setBanner(`Submitting minimization for node ${nodeId}...`);
        setSubtext("The selected geometry will be optimized and then written back into the live network.");
        await postJson("/api/minimize", { node_ids: [Number(nodeId)] });
        setBanner(`Minimization request accepted for node ${nodeId}.`);
        await refreshState();
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
        await refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The minimization request failed before it could start.");
      }
    }

    async function queueEdgeNeb(sourceNode, targetNode) {
      try {
        setBanner(`Submitting autosplitting NEB for edge ${sourceNode} -> ${targetNode}...`);
        setSubtext("The edge endpoints will be prepared, the NEB will run in the background, and any discovered intermediates will be folded back into this network.");
        await postJson("/api/run-neb", { source_node: Number(sourceNode), target_node: Number(targetNode) });
        setBanner(`Autosplitting NEB request accepted for edge ${sourceNode} -> ${targetNode}.`);
        await refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The NEB request failed before it could start.");
      }
    }

    window.queueMinimizeNode = queueMinimizeNode;
    window.queueMinimizeAll = queueMinimizeAll;
    window.queueEdgeNeb = queueEdgeNeb;

    document.getElementById("initialize").addEventListener("click", initializeDrive);
    document.getElementById("load-workspace").addEventListener("click", loadWorkspace);
    document.getElementById("minimize-all").addEventListener("click", queueMinimizeAll);

    const d3Script = document.createElement("script");
    d3Script.src = "https://d3js.org/d3.v3.min.js";
    d3Script.onload = refreshState;
    document.head.appendChild(d3Script);
    setInterval(refreshState, 3000);
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


class MepdDriveServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        base_directory: Path,
        inputs_fp: Path,
        reactions_fp: Path | None,
        timeout_seconds: int,
        max_nodes: int,
        max_depth: int,
        max_parallel_nebs: int,
    ):
        self.base_directory = base_directory
        self.inputs_fp = inputs_fp
        self.reactions_fp = reactions_fp
        self.timeout_seconds = timeout_seconds
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_parallel_nebs = max_parallel_nebs
        self.state_lock = threading.Lock()
        self.runtime = _DriveRuntimeState()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mepd-drive")
        self.process_executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        super().__init__(server_address, _DriveHandler)

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
                self.runtime.last_error = f"{type(exc).__name__}: {exc}"
                self.runtime.last_message = "Last action failed."
                if self.runtime.active_action is not None:
                    self.runtime.active_action["status"] = "failed"
                    self.runtime.active_action["error"] = f"{type(exc).__name__}: {exc}"
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
        reactant = _resolve_species_input(
            smiles=str(payload.get("reactant_smiles") or ""),
            xyz_text=str(payload.get("reactant_xyz") or ""),
        )
        if not reactant:
            raise ValueError("A reactant SMILES or reactant XYZ block is required.")

        product = _resolve_species_input(
            smiles=str(payload.get("product_smiles") or ""),
            xyz_text=str(payload.get("product_xyz") or ""),
        )
        if str(payload.get("mode") or "reactant") == "reactant-product" and not product:
            raise ValueError("Reactant/product mode requires a product SMILES or product XYZ block.")

        run_name = str(payload.get("run_name") or "").strip() or f"mepd-drive-{int(time.time())}"
        workspace_dir = (self.base_directory / run_name).resolve()

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
                    self.runtime.last_error = f"{type(exc).__name__}: {exc}"
                    self.runtime.last_message = "Last action failed."
                    if self.runtime.active_action is not None:
                        self.runtime.active_action["status"] = "failed"
                        self.runtime.active_action["error"] = f"{type(exc).__name__}: {exc}"
                return
            workspace = RetropathsWorkspace(**result["workspace"])
            with self.state_lock:
                self.runtime.workspace = workspace
                self.runtime.reactant = result["reactant"]
                self.runtime.product = result["product"] or None
                self.runtime.last_message = str(result.get("message") or "Action completed.")
                self.runtime.active_action = None

        future = self.process_executor.submit(
            _initialize_workspace_job,
            reactant=reactant,
            product=product or None,
            run_name=run_name,
            workspace_dir=str(workspace_dir),
            inputs_fp=str(self.inputs_fp),
            reactions_fp=str(self.reactions_fp) if self.reactions_fp else None,
            timeout_seconds=self.timeout_seconds,
            max_nodes=self.max_nodes,
            max_depth=self.max_depth,
            max_parallel_nebs=self.max_parallel_nebs,
        )
        future.add_done_callback(_finish_initialize)
        self._set_busy("Building Retropaths network...", future)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "initialize",
                "status": "running",
                "label": "Building Retropaths network...",
            }

    def submit_load_workspace(self, payload: dict[str, Any]) -> None:
        self._assert_idle()
        workspace_path = str(payload.get("workspace_path") or "").strip()
        if not workspace_path:
            raise ValueError("A workspace path is required.")

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
                    self.runtime.last_error = f"{type(exc).__name__}: {exc}"
                    self.runtime.last_message = "Last action failed."
                    if self.runtime.active_action is not None:
                        self.runtime.active_action["status"] = "failed"
                        self.runtime.active_action["error"] = f"{type(exc).__name__}: {exc}"
                return
            workspace = RetropathsWorkspace(**result["workspace"])
            with self.state_lock:
                self.runtime.workspace = workspace
                self.runtime.reactant = result["reactant"]
                self.runtime.product = result["product"] or None
                self.runtime.last_message = str(result.get("message") or "Workspace loaded.")
                self.runtime.active_action = None

        future = self.executor.submit(_load_existing_workspace_job, workspace_path)
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
            if queue.recover_stale_running_items():
                queue.write_to_disk(workspace.queue_fp)
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
        future = self.process_executor.submit(
            _run_selected_edge_neb_logged,
            dict(workspace.__dict__),
            source_node=source_node,
            target_node=target_node,
            log_fp=log_fp,
            progress_fp=progress_fp,
            chain_fp=chain_fp,
        )
        future.add_done_callback(self._finish_future)
        self._set_busy(f"Running autosplitting NEB for {source_node} -> {target_node}...", future)
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
        }
        if runtime.active_action is not None and runtime.active_action.get("status") == "running":
            with contextlib.suppress(Exception):
                if runtime.active_action.get("type") == "neb":
                    snapshot["live_activity"] = _build_neb_live_payload(runtime.active_action, runtime.workspace)
                elif runtime.active_action.get("type") == "minimize":
                    snapshot["live_activity"] = _build_minimize_live_payload(runtime.active_action)
        if runtime.workspace is not None and runtime.workspace.queue_fp.exists():
            with contextlib.suppress(Exception):
                builder = _build_drive_payload
                if (
                    runtime.active_action is not None
                    and runtime.active_action.get("status") == "running"
                    and runtime.active_action.get("type") == "minimize"
                ):
                    builder = _build_drive_payload_fast
                elif (
                    runtime.active_action is not None
                    and runtime.active_action.get("status") == "running"
                    and runtime.active_action.get("type") == "neb"
                ):
                    builder = _build_drive_payload_fast_neb
                snapshot["drive"] = builder(
                    runtime.workspace,
                    product_smiles=str((runtime.product or {}).get("smiles") or ""),
                    active_job_label=runtime.busy_label if runtime.future is not None and not runtime.future.done() else "",
                    active_action=runtime.active_action,
                )
        return snapshot

    def server_close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.process_executor.shutdown(wait=False, cancel_futures=True)
        super().server_close()


class _DriveHandler(BaseHTTPRequestHandler):
    server: MepdDriveServer

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        if self.path == "/":
            body = _drive_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/state":
            self._write_json(self.server.snapshot())
            return
        if self.path.startswith("/edge_visualizations/"):
            with self.server.state_lock:
                workspace = self.server.runtime.workspace
            if workspace is None:
                self.send_error(404)
                return
            target = (workspace.edge_visualizations_dir / Path(self.path).name).resolve()
            if not target.exists():
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
            payload = self._read_json()
            if self.path == "/api/initialize":
                self.server.submit_initialize(payload)
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if self.path == "/api/load-workspace":
                self.server.submit_load_workspace(payload)
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if self.path == "/api/minimize":
                self.server.submit_minimize([int(value) for value in payload.get("node_ids", [])])
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if self.path == "/api/run-neb":
                self.server.submit_run_neb(
                    source_node=int(payload["source_node"]),
                    target_node=int(payload["target_node"]),
                )
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
        except Exception as exc:
            with contextlib.suppress(Exception):
                self.server._record_request_error(f"{type(exc).__name__}: {exc}")
            self._write_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def launch_mepd_drive(
    *,
    directory: str | None,
    inputs_fp: str,
    reactions_fp: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    timeout_seconds: int = 30,
    max_nodes: int = 40,
    max_depth: int = 4,
    max_parallel_nebs: int = 1,
    open_browser: bool = True,
) -> MepdDriveServer:
    base_directory = Path(directory).resolve() if directory else (Path.cwd() / "mepd-drive").resolve()
    base_directory.mkdir(parents=True, exist_ok=True)
    server = MepdDriveServer(
        (host, int(port)),
        base_directory=base_directory,
        inputs_fp=Path(inputs_fp).resolve(),
        reactions_fp=Path(reactions_fp).resolve() if reactions_fp else None,
        timeout_seconds=timeout_seconds,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_parallel_nebs=max_parallel_nebs,
    )
    if open_browser:
        webbrowser.open(f"http://{host}:{server.server_address[1]}/")
    return server
