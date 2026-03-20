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
from neb_dynamics.TreeNode import TreeNode
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
    add_manual_edge,
    apply_reactions_to_node,
    initialize_workspace_with_progress,
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


def _run_geometry_optimization_batch_with_trajectories(
    nodes: list[Any],
    run_inputs: RunInputs,
) -> list[tuple[Any, str | None, list[Any] | None]]:
    if not nodes:
        return []

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

        optimized_nodes: list[tuple[Any, str | None, list[Any] | None]] = []
        for original, traj in zip(nodes, trajectories):
            optimized = traj[-1]
            if getattr(original, "has_molecular_graph", False):
                optimized.graph = structure_to_molecule(optimized.structure)
                optimized.has_molecular_graph = True
            else:
                optimized.graph = original.graph
                optimized.has_molecular_graph = original.has_molecular_graph
            optimized_nodes.append((optimized, None, traj))
        if len(optimized_nodes) == len(nodes):
            return optimized_nodes
    except Exception:
        pass

    return [
        _run_geometry_optimization_with_trajectory(node=node, run_inputs=run_inputs)
        for node in nodes
    ]


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


def _build_growth_live_payload(active_action: dict[str, Any] | None) -> dict[str, Any] | None:
    if active_action is None or active_action.get("type") not in {"initialize", "apply-reactions"}:
        return None
    progress = _read_growth_progress(active_action.get("progress_fp")) or {}
    return {
        "type": "growth",
        "title": str(progress.get("title") or active_action.get("label") or "Growing Retropaths network"),
        "note": str(progress.get("note") or ""),
        "phase": str(progress.get("phase") or "growing"),
        "network": dict(progress.get("network") or {"nodes": [], "edges": []}),
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
        "gi_inputs": {},
        "optimizer_kwds": {},
        "program_kwds": {},
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
    summary["gi_inputs"] = _namespace_payload(getattr(run_inputs, "gi_inputs", None))
    summary["optimizer_kwds"] = _namespace_payload(getattr(run_inputs, "optimizer_kwds", None))
    program_kwds = getattr(run_inputs, "program_kwds", None)
    if isinstance(program_kwds, dict):
        summary["program_kwds"] = _json_safe(program_kwds)
    elif hasattr(program_kwds, "model_dump"):
        with contextlib.suppress(Exception):
            summary["program_kwds"] = _json_safe(program_kwds.model_dump())
    return summary


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


def _write_completed_queue_visualizations(
    workspace: RetropathsWorkspace,
    queue: RetropathsNEBQueue,
) -> list[dict[str, Any]]:
    from neb_dynamics.scripts.main_cli import _build_chain_visualizer_html

    rows: list[dict[str, Any]] = []
    out_dir = workspace.edge_visualizations_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in queue.items:
        if item.status != "completed" or not item.result_dir:
            continue
        filename = f"queue_edge_{int(item.source_node)}_{int(item.target_node)}.html"
        out_fp = out_dir / filename
        meta_fp = out_dir / f"queue_edge_{int(item.source_node)}_{int(item.target_node)}.meta.json"
        cache_key = {
            "result_dir": str(item.result_dir),
            "finished_at": str(item.finished_at or ""),
        }
        if out_fp.exists() and meta_fp.exists():
            with contextlib.suppress(Exception):
                meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                if (
                    meta.get("result_dir") == cache_key["result_dir"]
                    and meta.get("finished_at") == cache_key["finished_at"]
                ):
                    rows.append(
                        {
                            "edge": f"{int(item.source_node)} -> {int(item.target_node)}",
                            "barrier": meta.get("barrier"),
                            "href": filename,
                        }
                    )
                    continue
        try:
            history = TreeNode.read_from_disk(
                folder_name=item.result_dir,
                charge=0,
                multiplicity=1,
            )
        except Exception:
            continue
        chain = getattr(history, "output_chain", None)
        if chain is None:
            continue

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

        barrier = None
        with contextlib.suppress(Exception):
            barrier = float(chain.get_eA_chain())
        with contextlib.suppress(Exception):
            meta_fp.write_text(
                json.dumps(
                    {
                        **cache_key,
                        "barrier": barrier,
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
            }
        )

    return rows


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
        node["neb_backed"] = node_index in backed_nodes
        node["is_target"] = bool(normalized_product and str(node["label"]) == normalized_product)

    for edge in explorer["edges"]:
        source = int(edge["source"])
        target = int(edge["target"])
        attrs = pot.graph.edges[(source, target)]
        display_attrs, used_reverse_result = _resolve_display_edge_attrs(pot.graph, source, target)
        queue_item = queue_by_edge.get((source, target))
        queue_result = queue_result_by_edge.get(f"{source} -> {target}") or (
            queue_result_by_edge.get(f"{target} -> {source}") if used_reverse_result else None
        )
        viewer_edge_key = f"{target} -> {source}" if used_reverse_result else f"{source} -> {target}"
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
            if not edge["viewer_href"] and queue_result.get("href"):
                edge["viewer_href"] = f"edge_visualizations/{queue_result['href']}"
            if edge["chains"] == 0:
                edge["chains"] = 1
            edge["neb_backed"] = edge["neb_backed"] or bool(edge["viewer_href"]) or edge["barrier"] is not None
            edge["result_from_completed_queue"] = True
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
        "inputs": _inputs_summary_payload(workspace),
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
        node["can_apply_reactions"], node["apply_reactions_note"] = _node_apply_reaction_status(node_index, attrs)
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
        "inputs": _inputs_summary_payload(workspace),
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
        node["can_apply_reactions"], node["apply_reactions_note"] = _node_apply_reaction_status(node_index, attrs)
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
        "inputs": _inputs_summary_payload(workspace),
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
    progress_fp: str | None = None,
) -> dict[str, Any]:
    workspace_path = Path(workspace_dir).resolve()
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
    )
    if progress_fp:
        initialize_workspace_with_progress(workspace, progress_fp=progress_fp)
    else:
        prepare_neb_workspace(workspace)
    return _workspace_snapshot_payload(
        workspace,
        reactant=reactant,
        product=product,
        message=f"Initialized workspace {workspace.run_name}.",
    )


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
        },
        "reactant": reactant if reactant is not None else {"smiles": workspace.root_smiles},
        "product": product,
        "message": message or f"Loaded workspace {workspace.run_name}.",
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
    return _workspace_snapshot_payload(
        workspace,
        message=f"Loaded existing workspace {workspace.run_name}.",
    )


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
    .page-title { padding: 20px; background: linear-gradient(135deg, #fff8ee, #f3e7d3); }
    .section-head { display: flex; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
    .section-head h2 { margin: 0; }
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
    .summary-strip { display: grid; grid-template-columns: minmax(260px, 1fr) minmax(0, 1.6fr); gap: 16px; align-items: start; margin-bottom: 14px; }
    .tool-tabs, .detail-tabs { display: flex; gap: 8px; margin: 10px 0 14px; flex-wrap: wrap; }
    .tool-tab, .detail-tab { padding: 7px 10px; border: 1px solid var(--line); background: #f4ecdf; cursor: pointer; }
    .tool-tab.active, .detail-tab.active { background: var(--accent); color: white; border-color: #8f4b1c; }
    .tool-panel, .detail-panel { display: none; }
    .tool-panel.active, .detail-panel.active { display: block; }
    .inputs-grid { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.9fr); gap: 16px; }
    .network-canvas-shell { position: relative; }
    .network-toolbar {
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 3;
      min-width: 220px;
      padding: 12px;
      border: 1px solid var(--line);
      background: rgba(255, 253, 248, 0.96);
      box-shadow: 0 10px 24px rgba(36, 29, 24, 0.12);
    }
    .network-toolbar-title { font-weight: 600; margin-bottom: 6px; }
    .network-toolbar-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .network-tool-button {
      min-width: 44px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      background: #f7efe2;
      font-size: 20px;
      line-height: 1;
    }
    .network-tool-button.active { background: var(--accent-2); color: #fff; border-color: #165546; }
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
    .network-node.connect-source { fill: #1f6a57; stroke: #fdf8ef; stroke-width: 3.2; }
    .network-label { font-size: 11px; fill: #2d241c; pointer-events: none; }
    .viewer-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    iframe.structure { width: 100%; height: 320px; border: 1px solid var(--line); background: white; }
    .mol-card { width: 100%; min-height: 320px; border: 1px solid var(--line); background: white; display: grid; align-content: start; }
    .mol-card svg { width: 100%; height: auto; display: block; }
    .mol-empty { padding: 18px 14px; color: var(--muted); }
    .mol-meta { padding: 10px 12px 12px; border-top: 1px solid var(--line); font-size: 12px; color: var(--muted); word-break: break-word; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; background: #f7f1e9; border: 1px solid #e4d7c5; padding: 12px; }
    .badge { display: inline-block; padding: 2px 8px; border: 1px solid var(--line); background: #faf2e5; margin-right: 6px; }
    .message { padding: 10px 12px; border: 1px solid var(--line); background: #faf5eb; }
    .kv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
    .kv-card { border: 1px solid var(--line); background: #fbf7ef; padding: 10px; }
    .code-block { background: #f7f1e9; border: 1px solid #e4d7c5; padding: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow: auto; white-space: pre-wrap; word-break: break-word; }
    .live-activity { margin-top: 12px; border: 1px solid var(--line); background: #fbf7ef; padding: 12px; }
    .live-activity svg { width: 100%; height: auto; border: 1px solid var(--line); background: white; }
    .live-activity pre { margin-top: 10px; font-size: 12px; max-height: 220px; overflow: auto; }
    .live-activity-inline {
      position: absolute;
      top: 16px;
      right: 16px;
      z-index: 2;
      width: min(460px, calc(100% - 32px));
      max-height: calc(100% - 32px);
      margin-top: 0;
      overflow: auto;
      box-shadow: 0 14px 28px rgba(36, 29, 24, 0.14);
      backdrop-filter: blur(1px);
    }
    .live-neb-layout { display: grid; grid-template-columns: 190px minmax(0, 1fr) 190px; gap: 12px; align-items: start; }
    .live-neb-layout .mol-card { min-height: 210px; }
    .job-list { display: grid; gap: 8px; margin-top: 12px; }
    .job-row { border: 1px solid var(--line); background: #fffdf8; padding: 8px 10px; }
    .job-row.running { border-color: #1f6a57; }
    .job-row.completed { border-color: #8ab0a3; }
    .job-row.failed { border-color: #b03a2e; }
    .log-grid { display: grid; gap: 12px; }
    @media (max-width: 1080px) {
      .summary-strip, .inputs-grid, .form-grid, .viewer-grid { grid-template-columns: 1fr; }
      .explorer-svg { min-height: 480px; }
      .live-neb-layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel page-title">
      <h1 style="margin:0 0 6px;">MEPD Drive</h1>
      <div class="muted" style="margin-bottom:12px;">Set inputs, grow the reaction network, interact directly with the graph, then queue exploration work and inspect logs from a single workspace.</div>
      <div id="job-banner" class="message">Idle.</div>
      <div id="job-subtext" class="muted" style="margin-top:8px;">No action submitted yet.</div>
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
              <button id="initialize" class="primary">Build Retropaths Network</button>
              <button id="load-workspace" class="secondary">Load Existing Workspace</button>
              <button id="minimize-all" class="secondary">Queue Minimization For All Geometries</button>
            </div>
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
      <div class="network-canvas-shell">
        <div id="network-toolbar" class="network-toolbar"></div>
        <svg id="network-svg" class="explorer-svg" viewBox="0 0 1180 680" role="img" aria-label="MEPD Drive network graph"></svg>
        <div id="live-activity-inline" class="live-activity live-activity-inline" style="display:none;"></div>
      </div>
    </div>

    <div class="panel">
      <div class="section-head">
        <h2>Exploration</h2>
        <div class="muted">Queue NEBs, minimizations, reaction-template application, and inspect the selected graph item.</div>
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
    const state = {
      snapshot: null,
      selected: null,
      networkVersion: "",
      connectSourceNodeId: null,
    };

    function setManualEdgeEndpoint(which, nodeId) {
      const input = document.getElementById(which === "source" ? "manual-edge-source" : "manual-edge-target");
      if (input) input.value = String(Number(nodeId));
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
      ["initialize", "load-workspace", "minimize-all", "add-manual-edge"].forEach((id) => {
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
      const inputEl = document.getElementById("workspace-summary");
      const networkEl = document.getElementById("network-summary");
      if (!inputEl || !networkEl) return;
      if (!snapshot || !snapshot.initialized || !snapshot.drive) {
        inputEl.textContent = "No workspace initialized yet.";
        networkEl.textContent = "No workspace initialized yet.";
        return;
      }
      const workspace = snapshot.drive.workspace;
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

    function formatJsonBlock(value) {
      const text = JSON.stringify(value || {}, null, 2);
      return `<div class="code-block">${escapeHtml(text)}</div>`;
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

    function renderNetworkToolbar() {
      const toolbar = document.getElementById("network-toolbar");
      if (!toolbar) return;
      const selection = state.selected;
      if (!selection) {
        toolbar.innerHTML = `
          <div class="network-toolbar-title">Network Actions</div>
          <div class="muted">Select a node or edge to see actions here.</div>
        `;
        return;
      }
      if (selection.kind === "node") {
        const node = selection.node;
        const connectActive = Number(state.connectSourceNodeId) === Number(node.id);
        toolbar.innerHTML = `
          <div class="network-toolbar-title">Node ${escapeHtml(node.id)} Actions</div>
          <div class="muted">${connectActive ? "Click a second node to create an edge from this source." : "Node tools are available directly from the graph."}</div>
          <div class="network-toolbar-actions">
            <button class="network-tool-button" data-drive-action="toolbar-minimize-node" title="Minimize geometry" onclick="queueMinimizeNode(${Number(node.id)})" ${node.minimizable ? "" : "disabled"}>↓</button>
            <button class="network-tool-button" data-drive-action="toolbar-apply-node" title="Apply reaction templates" onclick="queueApplyReactions(${Number(node.id)})" ${node.can_apply_reactions ? "" : "disabled"}>+</button>
            <button class="network-tool-button ${connectActive ? "active" : ""}" data-drive-action="toolbar-connect-node" title="Connect to new node" onclick="beginConnectMode(${Number(node.id)})">→</button>
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
          <button class="network-tool-button" data-drive-action="toolbar-neb-edge" title="Queue NEB minimization" onclick="queueEdgeNeb(${Number(edge.source)}, ${Number(edge.target)})" ${edge.can_queue_neb ? "" : "disabled"}>#</button>
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
      edges.forEach((edge) => {
        const source = Number(edge.source);
        const target = Number(edge.target);
        const parents = parentsByNode.get(source) || [];
        parents.push(target);
        parentsByNode.set(source, parents);
      });

      const depthMemo = new Map();
      function depthFor(nodeId) {
        if (depthMemo.has(nodeId)) return depthMemo.get(nodeId);
        if (Number(nodeId) === 0) {
          depthMemo.set(nodeId, 0);
          return 0;
        }
        const parents = parentsByNode.get(Number(nodeId)) || [];
        if (!parents.length) {
          depthMemo.set(nodeId, 1);
          return 1;
        }
        const depth = 1 + Math.min(...parents.map((parentId) => depthFor(parentId)));
        depthMemo.set(nodeId, depth);
        return depth;
      }

      nodes.forEach((node) => {
        const depth = depthFor(Number(node.id));
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
            return `<line x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" stroke="#8f8271" stroke-width="2" />`;
          }).join("")}
          ${nodes.map((node) => {
            const pos = positions.get(Number(node.id));
            if (!pos) return "";
            return `
              <g transform="translate(${pos.x},${pos.y})">
                <circle r="18" fill="${node.growing ? "#d79e35" : (Number(node.id) === 0 ? "#8e4d1c" : "#7b6a57")}" stroke="#fffaf2" stroke-width="2.5"></circle>
                ${node.growing ? `
                  <circle r="24" fill="none" stroke="#1f6a57" stroke-width="3" stroke-dasharray="8 6" stroke-linecap="round">
                    <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="0" to="360" dur="1.2s" repeatCount="indefinite"/>
                  </circle>
                ` : ""}
                <text y="4" text-anchor="middle" font-size="12" fill="#ffffff">${escapeHtml(node.label || node.id)}</text>
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
                ${job.error ? `<div style="color:#b03a2e; margin-top:4px;">${escapeHtml(job.error)}</div>` : ""}
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
        ${activity.console_text ? `<pre>${escapeHtml(activity.console_text)}</pre>` : ""}
      `;
    }

    function renderLiveActivity(snapshot) {
      const panel = document.getElementById("live-activity-panel");
      const inline = document.getElementById("live-activity-inline");
      const activity = snapshot?.live_activity || null;
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
          ${node.endpoint_optimization_error ? `<div><strong>Last optimization error:</strong> <span style="color:#b03a2e;">${escapeHtml(node.endpoint_optimization_error)}</span></div>` : ""}
          <div><strong>Can queue minimization:</strong> ${node.minimizable ? "yes" : "no"}</div>
          <div><strong>Can apply reactions:</strong> ${node.can_apply_reactions ? "yes" : "no"}</div>
          ${node.minimize_note ? `<div><strong>Minimization note:</strong> ${escapeHtml(node.minimize_note)}</div>` : ""}
          ${node.apply_reactions_note ? `<div><strong>Reaction note:</strong> ${escapeHtml(node.apply_reactions_note)}</div>` : ""}
          <div><strong>NEB-backed:</strong> ${node.neb_backed ? "yes" : "no"}</div>
        `;
        targeted.innerHTML = `
          <div style="margin-bottom:10px;"><button class="secondary" data-drive-action="minimize-node" onclick="queueMinimizeNode(${Number(node.id)})" ${node.minimizable ? "" : "disabled"}>Queue Minimization For This Geometry</button></div>
          <div style="margin-bottom:10px;"><button class="secondary" data-drive-action="apply-reactions" onclick="queueApplyReactions(${Number(node.id)})" ${node.can_apply_reactions ? "" : "disabled"}>Apply Reactions To This Node</button></div>
          <div style="margin-bottom:10px; display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px;">
            <button class="secondary" onclick="setManualEdgeEndpoint('source', ${Number(node.id)})">Use As Manual Edge Source</button>
            <button class="secondary" onclick="setManualEdgeEndpoint('target', ${Number(node.id)})">Use As Manual Edge Target</button>
          </div>
          ${node.minimize_note ? `<div style="margin-bottom:10px; color:${node.minimizable ? "#6f6558" : "#b03a2e"};">${escapeHtml(node.minimize_note)}</div>` : ""}
          ${node.apply_reactions_note ? `<div style="margin-bottom:10px; color:${node.can_apply_reactions ? "#6f6558" : "#b03a2e"};">${escapeHtml(node.apply_reactions_note)}</div>` : ""}
          <pre>${escapeHtml(JSON.stringify(node.data || {}, null, 2))}</pre>
        `;
        templateData.innerHTML = `<pre>${escapeHtml(JSON.stringify(node.data || {}, null, 2))}</pre>`;
        structures.innerHTML = node.structure?.xyz_b64
          ? `<iframe class="structure" srcdoc="${escapeHtml(makeStructureSrcdoc(node.structure.xyz_b64))}"></iframe>`
          : `<div class="muted">No 3D structure is available for this node.</div>`;
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
        ${edge.result_from_reverse_edge ? `<div><strong>Displayed NEB result:</strong> reverse-directed edge</div>` : ""}
        ${edge.result_from_completed_queue ? `<div><strong>Displayed NEB result:</strong> completed queue attempt</div>` : ""}
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
        ${edge.result_from_reverse_edge ? `<div style="margin-bottom:10px;" class="muted">Showing completed NEB data reconstructed from the reverse-directed edge because this directed edge does not carry the chain payload directly.</div>` : ""}
        ${edge.result_from_completed_queue ? `<div style="margin-bottom:10px;" class="muted">Showing NEB data from the completed attempted pair because autosplitting did not leave a direct annotated edge for this exact selection.</div>` : ""}
        ${edge.queue_note ? `<div style="margin-bottom:10px; color: ${edge.can_queue_neb ? "#6f6558" : "#b03a2e"};"><strong>${edge.can_queue_neb ? "Queue note" : "Edge cannot run as-is"}:</strong> ${escapeHtml(edge.queue_note)}</div>` : ""}
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
          ? `<pre>${escapeHtml(JSON.stringify(template.data, null, 2))}</pre>`
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
          circle.classList.toggle("connect-source", Number(state.connectSourceNodeId) === Number(nodeId));
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
        group.addEventListener("click", () => {
          state.connectSourceNodeId = null;
          setSelection({ kind: "edge", edge });
        });
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
        group.addEventListener("click", async () => {
          const connectSource = state.connectSourceNodeId;
          if (connectSource != null && Number(connectSource) !== Number(node.id)) {
            setSelection({ kind: "node", node });
            await completeConnectMode(Number(node.id));
            return;
          }
          setSelection({ kind: "node", node });
        });
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
          if (activeAction.type === "initialize") {
            setSubtext("The Retropaths network is being grown. The live growth widget shows existing nodes, edges, and the nodes currently being expanded.");
          } else if (activeAction.type === "minimize") {
            setSubtext("Geometry optimization is running. Updated node structures are written back one-by-one and will appear here as polling refreshes.");
          } else if (activeAction.type === "neb") {
            setSubtext("Autosplitting NEB is running. The edge state and any discovered intermediates will appear after the backend writes them back.");
          } else if (activeAction.type === "apply-reactions") {
            setSubtext("Reaction templates are being applied to the selected node. Any newly grown products will be merged into the current graph after the job finishes.");
          } else {
            setSubtext("Background work is running. The network and counters will refresh automatically.");
          }
        } else {
          setSubtext("Idle.");
        }
        setButtonsDisabled(Boolean(snapshot.busy));
        renderWorkspaceSummary(snapshot);
        renderInputsPanels(snapshot);
        renderStats(snapshot);
        renderLiveActivity(snapshot);
        renderLogging(snapshot);
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
          inputs_path: document.getElementById("inputs-path").value,
          reactions_fp: document.getElementById("reactions-path").value,
          environment_smiles: document.getElementById("environment-smiles").value,
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

    async function queueMinimizePair(sourceNode, targetNode) {
      try {
        setBanner(`Submitting minimization for nodes ${sourceNode} and ${targetNode}...`);
        setSubtext("Both endpoint geometries will be optimized and written back into the live network.");
        await postJson("/api/minimize", { node_ids: [Number(sourceNode), Number(targetNode)] });
        setBanner(`Minimization request accepted for nodes ${sourceNode} and ${targetNode}.`);
        await refreshState();
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
        await refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The NEB request failed before it could start.");
      }
    }

    async function queueApplyReactions(nodeId) {
      try {
        setBanner(`Applying reactions to node ${nodeId}...`);
        setSubtext("The selected node will be regrown with the Retropaths template library and any new products will be merged into the current graph.");
        await postJson("/api/apply-reactions", { node_id: Number(nodeId) });
        setBanner(`Reaction-application request accepted for node ${nodeId}.`);
        await refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The reaction-application request failed before it could start.");
      }
    }

    async function addManualEdge() {
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
        setBanner(`Attempting to add manual edge ${sourceNode} -> ${targetNode}...`);
        setSubtext("The graph edge will be created if needed and prepared for a subsequent autosplitting NEB run.");
        const result = await postJson("/api/add-edge", {
          source_node: sourceNode,
          target_node: targetNode,
          reaction_label: document.getElementById("manual-edge-label").value,
        });
        setBanner(result.message || `Manual edge ${sourceNode} -> ${targetNode} updated.`);
        await refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The manual edge could not be added.");
      }
    }

    function beginConnectMode(nodeId) {
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
      const sourceNodeId = Number(state.connectSourceNodeId);
      const targetNode = Number(targetNodeId);
      if (!Number.isFinite(sourceNodeId) || sourceNodeId < 0 || sourceNodeId === targetNode) {
        return;
      }
      setManualEdgeEndpoint("source", sourceNodeId);
      setManualEdgeEndpoint("target", targetNode);
      state.connectSourceNodeId = null;
      try {
        setBanner(`Attempting to add manual edge ${sourceNodeId} -> ${targetNode}...`);
        setSubtext("Creating an edge directly from the graph selection.");
        const result = await postJson("/api/add-edge", {
          source_node: sourceNodeId,
          target_node: targetNode,
          reaction_label: document.getElementById("manual-edge-label").value,
        });
        setBanner(result.message || `Manual edge ${sourceNodeId} -> ${targetNode} updated.`);
        await refreshState();
      } catch (error) {
        setBanner(error.message || String(error), true);
        setSubtext("The graph-directed manual edge could not be added.");
      }
    }

    window.queueMinimizeNode = queueMinimizeNode;
    window.queueMinimizeAll = queueMinimizeAll;
    window.queueMinimizePair = queueMinimizePair;
    window.queueEdgeNeb = queueEdgeNeb;
    window.queueApplyReactions = queueApplyReactions;
    window.setManualEdgeEndpoint = setManualEdgeEndpoint;
    window.beginConnectMode = beginConnectMode;

    document.getElementById("initialize").addEventListener("click", initializeDrive);
    document.getElementById("load-workspace").addEventListener("click", loadWorkspace);
    document.getElementById("minimize-all").addEventListener("click", queueMinimizeAll);
    document.getElementById("add-manual-edge").addEventListener("click", addManualEdge);

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
        inputs_fp: Path | None,
        reactions_fp: Path | None,
        timeout_seconds: int,
        max_nodes: int,
        max_depth: int,
        max_parallel_nebs: int,
        initial_state: dict[str, Any] | None = None,
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
        self._drive_payload_cache_key: tuple[Any, ...] | None = None
        self._drive_payload_cache_value: dict[str, Any] | None = None
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
        builder = _build_drive_payload
        builder_name = "full"
        if (
            runtime.active_action is not None
            and runtime.active_action.get("status") == "running"
            and runtime.active_action.get("type") == "minimize"
        ):
            builder = _build_drive_payload_fast
            builder_name = "fast-minimize"
        elif (
            runtime.active_action is not None
            and runtime.active_action.get("status") == "running"
            and runtime.active_action.get("type") == "neb"
        ):
            builder = _build_drive_payload_fast_neb
            builder_name = "fast-neb"

        product_smiles = str((runtime.product or {}).get("smiles") or "")
        active_action = runtime.active_action or {}
        cache_key = (
            builder_name,
            _drive_network_version(workspace),
            product_smiles,
            int(active_action.get("source_node", -1)) if builder_name == "fast-neb" else -1,
            int(active_action.get("target_node", -1)) if builder_name == "fast-neb" else -1,
        )

        with self.state_lock:
            cached_key = getattr(self, "_drive_payload_cache_key", None)
            cached_value = getattr(self, "_drive_payload_cache_value", None)
            if cached_key == cache_key and cached_value is not None:
                return cached_value

        payload = builder(
            workspace,
            product_smiles=product_smiles,
            active_job_label=runtime.busy_label if runtime.future is not None and not runtime.future.done() else "",
            active_action=runtime.active_action,
        )
        with self.state_lock:
            self._drive_payload_cache_key = cache_key
            self._drive_payload_cache_value = payload
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
        progress_fp = str((workspace_dir / "drive_growth.progress.json").resolve())
        inputs_fp = self._resolve_inputs_fp(payload)
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
            inputs_fp=str(inputs_fp),
            reactions_fp=str(reactions_fp) if reactions_fp else None,
            environment_smiles=environment_smiles,
            timeout_seconds=self.timeout_seconds,
            max_nodes=self.max_nodes,
            max_depth=self.max_depth,
            max_parallel_nebs=self.max_parallel_nebs,
            progress_fp=progress_fp,
        )
        future.add_done_callback(_finish_initialize)
        self._set_busy("Building Retropaths network...", future)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "initialize",
                "status": "running",
                "label": "Building Retropaths network...",
                "progress_fp": progress_fp,
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

    def submit_apply_reactions(self, *, node_id: int) -> None:
        self._assert_idle()
        with self.state_lock:
            workspace = self.runtime.workspace
        if workspace is None:
            raise ValueError("Initialize a workspace before applying reactions.")

        progress_fp = str((workspace.directory / f"drive_apply_reactions_{int(node_id)}.progress.json").resolve())
        future = self.process_executor.submit(
            apply_reactions_to_node,
            RetropathsWorkspace(**dict(workspace.__dict__)),
            int(node_id),
            progress_fp=progress_fp,
        )
        future.add_done_callback(self._finish_future)
        self._set_busy(f"Applying reaction templates to node {node_id}...", future)
        with self.state_lock:
            self.runtime.active_action = {
                "type": "apply-reactions",
                "status": "running",
                "label": f"Applying reaction templates to node {node_id}...",
                "node_id": int(node_id),
                "progress_fp": progress_fp,
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
        }
        if runtime.active_action is not None and runtime.active_action.get("status") == "running":
            with contextlib.suppress(Exception):
                if runtime.active_action.get("type") == "neb":
                    snapshot["live_activity"] = _build_neb_live_payload(runtime.active_action, runtime.workspace)
                elif runtime.active_action.get("type") == "minimize":
                    snapshot["live_activity"] = _build_minimize_live_payload(runtime.active_action)
                elif runtime.active_action.get("type") in {"initialize", "apply-reactions"}:
                    snapshot["live_activity"] = _build_growth_live_payload(runtime.active_action)
        if runtime.workspace is not None and runtime.workspace.queue_fp.exists():
            with contextlib.suppress(Exception):
                snapshot["drive"] = self._drive_payload_cache_lookup(
                    workspace=runtime.workspace,
                    runtime=runtime,
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
            if self.path == "/api/apply-reactions":
                self.server.submit_apply_reactions(
                    node_id=int(payload["node_id"]),
                )
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if self.path == "/api/run-neb":
                self.server.submit_run_neb(
                    source_node=int(payload["source_node"]),
                    target_node=int(payload["target_node"]),
                )
                self._write_json({"ok": True}, HTTPStatus.ACCEPTED)
                return
            if self.path == "/api/add-edge":
                result = self.server.submit_add_edge(
                    source_node=int(payload["source_node"]),
                    target_node=int(payload["target_node"]),
                    reaction_label=str(payload.get("reaction_label") or ""),
                )
                self._write_json({"ok": True, **result}, HTTPStatus.ACCEPTED)
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
    inputs_fp: str | None,
    reactions_fp: str | None = None,
    workspace_path: str | None = None,
    smiles: str | None = None,
    environment_smiles: str = "",
    run_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    timeout_seconds: int = 30,
    max_nodes: int = 40,
    max_depth: int = 4,
    max_parallel_nebs: int = 1,
    open_browser: bool = True,
) -> MepdDriveServer:
    explicit_directory = Path(directory).resolve() if directory else None
    startup_workspace_path = workspace_path
    if startup_workspace_path is None and explicit_directory is not None and (explicit_directory / "workspace.json").exists():
        startup_workspace_path = str(explicit_directory)

    initial_state: dict[str, Any] | None = None
    if startup_workspace_path:
        workspace_dir = Path(startup_workspace_path).expanduser().resolve()
        if workspace_dir.is_file() and workspace_dir.name == "workspace.json":
            workspace_dir = workspace_dir.parent
        base_directory = workspace_dir.parent
        initial_state = _load_existing_workspace_job(str(workspace_dir))
    elif smiles:
        resolved_inputs = Path(str(inputs_fp)).expanduser().resolve() if inputs_fp else None
        if resolved_inputs is None:
            raise ValueError("An inputs TOML path is required to start drive from SMILES.")
        resolved_run_name = str(run_name or "").strip() or f"mepd-drive-{int(time.time())}"
        if explicit_directory is not None:
            run_dir = explicit_directory
            base_directory = run_dir.parent
        else:
            base_directory = (Path.cwd() / "mepd-drive").resolve()
            run_dir = (base_directory / resolved_run_name).resolve()
        initial_state = _initialize_workspace_job(
            reactant={"smiles": str(smiles).strip()},
            product=None,
            run_name=resolved_run_name,
            workspace_dir=str(run_dir),
            inputs_fp=str(resolved_inputs),
            reactions_fp=str(Path(reactions_fp).expanduser().resolve()) if reactions_fp else None,
            environment_smiles=str(environment_smiles or ""),
            timeout_seconds=timeout_seconds,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_parallel_nebs=max_parallel_nebs,
            progress_fp=None,
        )
    else:
        base_directory = explicit_directory if explicit_directory is not None else (Path.cwd() / "mepd-drive").resolve()
    base_directory.mkdir(parents=True, exist_ok=True)
    server = MepdDriveServer(
        (host, int(port)),
        base_directory=base_directory,
        inputs_fp=Path(inputs_fp).expanduser().resolve() if inputs_fp else None,
        reactions_fp=Path(reactions_fp).resolve() if reactions_fp else None,
        timeout_seconds=timeout_seconds,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_parallel_nebs=max_parallel_nebs,
        initial_state=initial_state,
    )
    if open_browser:
        webbrowser.open(f"http://{host}:{server.server_address[1]}/")
    return server
