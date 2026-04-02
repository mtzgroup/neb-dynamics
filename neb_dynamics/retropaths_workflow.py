from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import contextlib
import importlib
import io
import json
import os
import re
import sys
import types
from pathlib import Path
from time import time
from typing import Any, Callable

import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics.helper_functions import parse_nma_freq_data
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.inputs import RunInputs
from neb_dynamics.kmc import (
    build_kmc_payload,
    normalize_initial_conditions,
    simulate_kmc,
)
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.nodes.nodehelpers import _is_connectivity_identical, displace_by_dr, is_identical
from neb_dynamics.molecule import Molecule
from neb_dynamics.pot import Pot
from neb_dynamics.qcio_structure_helpers import structure_to_molecule
from neb_dynamics.retropaths_compat import copy_graph_like_molecule, retropaths_pot_to_neb_pot
from neb_dynamics.retropaths_queue import (
    _make_pair_chain,
    build_balanced_endpoints,
    _global_environment_molecule,
    NEBQueueItem,
    RetropathsNEBQueue,
    _trim_balanced_endpoint,
    annotate_pot_with_queue_results,
    build_retropaths_neb_queue,
    ensure_queue_item_for_edge,
    load_completed_queue_chains,
    run_retropaths_neb_queue,
)
from neb_dynamics.retropaths_compat import structure_node_from_graph_like_molecule
from neb_dynamics.TreeNode import TreeNode


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return slug or "netgen"


@dataclass
class RetropathsWorkspace:
    workdir: str
    run_name: str
    root_smiles: str
    environment_smiles: str
    inputs_fp: str
    reactions_fp: str = ""
    timeout_seconds: int = 30
    max_nodes: int = 40
    max_depth: int = 4
    max_parallel_nebs: int = 1

    @property
    def directory(self) -> Path:
        return Path(self.workdir)

    @property
    def workspace_fp(self) -> Path:
        return self.directory / "workspace.json"

    @property
    def retropaths_pot_fp(self) -> Path:
        return self.directory / "retropaths_pot.json"

    @property
    def neb_pot_fp(self) -> Path:
        return self.directory / "neb_pot.json"

    @property
    def annotated_neb_pot_fp(self) -> Path:
        return self.directory / "neb_pot_annotated.json"

    @property
    def queue_fp(self) -> Path:
        return self.directory / "neb_queue.json"

    @property
    def queue_output_dir(self) -> Path:
        return self.directory / "queue_runs"

    @property
    def status_html_fp(self) -> Path:
        return self.directory / "status.html"

    @property
    def edge_visualizations_dir(self) -> Path:
        return self.directory / "edge_visualizations"

    @property
    def reactions_path(self) -> Path:
        if self.reactions_fp:
            return Path(self.reactions_fp)
        return _retropaths_repo() / "data" / "reactions.p"

    def write(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.workspace_fp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def read(cls, workdir: Path | str) -> "RetropathsWorkspace":
        workdir = Path(workdir)
        return cls(**json.loads((workdir / "workspace.json").read_text()))


def default_workspace_name(root_smiles: str, environment_smiles: str = "") -> str:
    env_suffix = f"-env-{_slugify(environment_smiles)}" if environment_smiles else ""
    return f"netgen-{_slugify(root_smiles)}{env_suffix}"


def create_workspace(
    root_smiles: str,
    environment_smiles: str,
    inputs_fp: Path | str,
    reactions_fp: Path | str | None = None,
    name: str | None = None,
    directory: Path | str | None = None,
    timeout_seconds: int = 30,
    max_nodes: int = 40,
    max_depth: int = 4,
    max_parallel_nebs: int = 1,
) -> RetropathsWorkspace:
    run_name = name or default_workspace_name(root_smiles, environment_smiles)
    workdir = Path(directory).resolve() if directory else (Path.cwd() / run_name).resolve()
    workspace = RetropathsWorkspace(
        workdir=str(workdir),
        run_name=run_name,
        root_smiles=root_smiles,
        environment_smiles=environment_smiles,
        inputs_fp=str(Path(inputs_fp).resolve()),
        reactions_fp=str(Path(reactions_fp).resolve()) if reactions_fp else "",
        timeout_seconds=timeout_seconds,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_parallel_nebs=max_parallel_nebs,
    )
    workspace.write()
    return workspace


def _retropaths_repo() -> Path:
    explicit = os.environ.get("RETROPATHS_REPO")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "retropaths"


def _retropaths_unavailable_message(*, feature: str, repo: Path, detail: str = "") -> str:
    msg = (
        f"{feature} requires the optional `retropaths` repository, but it is not available at "
        f"`{repo}`. Set `RETROPATHS_REPO` to a valid checkout and retry."
    )
    if detail:
        msg = f"{msg} ({detail})"
    return msg


def ensure_retropaths_available(*, feature: str = "This action") -> None:
    repo = _retropaths_repo()
    if not repo.exists():
        raise RuntimeError(_retropaths_unavailable_message(feature=feature, repo=repo))
    if not repo.is_dir():
        raise RuntimeError(
            _retropaths_unavailable_message(
                feature=feature,
                repo=repo,
                detail="path exists but is not a directory",
            )
        )

    prepare_retropaths_imports()
    try:
        importlib.import_module("retropaths.helper_functions")
        importlib.import_module("retropaths.molecules.molecule")
        importlib.import_module("retropaths.reactions.pot")
    except Exception as exc:
        raise RuntimeError(
            _retropaths_unavailable_message(
                feature=feature,
                repo=repo,
                detail=f"import failed: {type(exc).__name__}: {exc}",
            )
        ) from exc


def prepare_retropaths_imports() -> None:
    repo = _retropaths_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    if "cairosvg" not in sys.modules:
        cairosvg_stub = types.ModuleType("cairosvg")
        cairosvg_stub.svg2png = lambda *_args, **_kwargs: None
        sys.modules["cairosvg"] = cairosvg_stub

    if "imgkit" not in sys.modules:
        imgkit_stub = types.ModuleType("imgkit")
        imgkit_stub.from_string = lambda *_args, **_kwargs: None
        sys.modules["imgkit"] = imgkit_stub


def _load_retropaths_classes():
    ensure_retropaths_available(feature="Retropaths workflows")
    prepare_retropaths_imports()
    import retropaths.helper_functions as hf
    from retropaths.molecules.molecule import Molecule
    from retropaths.reactions.pot import Pot as RetropathsPot

    return hf, Molecule, RetropathsPot


def load_or_grow_retropaths_pot(workspace: RetropathsWorkspace):
    hf, Molecule, RetropathsPot = _load_retropaths_classes()

    if workspace.retropaths_pot_fp.exists():
        return RetropathsPot.from_json(workspace.retropaths_pot_fp)

    library = hf.pload(workspace.reactions_path)
    root = Molecule.from_smiles(workspace.root_smiles)
    environment = (
        Molecule.from_smiles(workspace.environment_smiles)
        if workspace.environment_smiles
        else Molecule()
    )
    pot = RetropathsPot(root=root, environment=environment, rxn_name=workspace.run_name)
    pot.run_with_timeout_and_error_catching(
        timeout_seconds_pot=workspace.timeout_seconds,
        library=library,
        name=workspace.run_name,
        maximum_number_of_nodes=workspace.max_nodes,
        max_depth=workspace.max_depth,
    )
    pot.to_json(workspace.retropaths_pot_fp)
    return pot


def load_retropaths_pot(workspace: RetropathsWorkspace):
    _hf, _Molecule, RetropathsPot = _load_retropaths_classes()
    return RetropathsPot.from_json(workspace.retropaths_pot_fp)


def prepare_neb_workspace(workspace: RetropathsWorkspace) -> tuple[Pot, RetropathsNEBQueue]:
    retropaths_pot = load_or_grow_retropaths_pot(workspace)
    if workspace.neb_pot_fp.exists():
        neb_pot = Pot.read_from_disk(workspace.neb_pot_fp)
    else:
        neb_pot = retropaths_pot_to_neb_pot(retropaths_pot)
        neb_pot.write_to_disk(workspace.neb_pot_fp)

    queue = build_retropaths_neb_queue(neb_pot, queue_fp=workspace.queue_fp)
    return neb_pot, queue


def _write_growth_progress(
    progress_fp: str | None,
    *,
    graph,
    growing_nodes: list[int] | None = None,
    title: str = "",
    note: str = "",
    phase: str = "",
) -> None:
    if not progress_fp:
        return
    growing = {int(node_id) for node_id in (growing_nodes or [])}
    payload = {
        "title": str(title or ""),
        "note": str(note or ""),
        "phase": str(phase or ""),
        "network": {
            "nodes": [
                {
                    "id": int(node_index),
                    "label": str(node_index),
                    "growing": int(node_index) in growing,
                }
                for node_index in sorted(graph.nodes)
            ],
            "edges": [
                {
                    "source": int(source_node),
                    "target": int(target_node),
                }
                for source_node, target_node in sorted(graph.edges)
            ],
        },
    }
    progress_path = Path(progress_fp)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def initialize_workspace_with_progress(
    workspace: RetropathsWorkspace,
    *,
    progress_fp: str | None = None,
) -> tuple[Pot, RetropathsNEBQueue]:
    hf, MoleculeCls, RetropathsPot = _load_retropaths_classes()
    prepare_retropaths_imports()
    from retropaths.reactions.pot import TimeoutPot
    from timeout_timer import timeout

    library = hf.pload(workspace.reactions_path)
    root = MoleculeCls.from_smiles(workspace.root_smiles)
    environment = (
        MoleculeCls.from_smiles(workspace.environment_smiles)
        if workspace.environment_smiles
        else MoleculeCls()
    )
    pot = RetropathsPot(root=root, environment=environment, rxn_name=workspace.run_name)
    filtered_library = library.filter_compatible(pot.conditions)
    started_at = time()

    _write_growth_progress(
        progress_fp,
        graph=pot.graph,
        growing_nodes=[0],
        title="Growing Retropaths network",
        note="Starting from the root node.",
        phase="initializing",
    )

    try:
        with timeout(workspace.timeout_seconds, exception=TimeoutPot):
            layer_counter = 0
            while pot.any_leaves_growable() and layer_counter < workspace.max_depth:
                growable_nodes = [
                    int(node_index)
                    for node_index in pot.leaves
                    if not pot.is_node_converged(node_index)
                ]
                _write_growth_progress(
                    progress_fp,
                    graph=pot.graph,
                    growing_nodes=growable_nodes,
                    title="Growing Retropaths network",
                    note=f"Growing layer {layer_counter + 1} with {len(growable_nodes)} active node(s).",
                    phase="growing",
                )
                for offset, leaf in enumerate(growable_nodes, start=1):
                    _write_growth_progress(
                        progress_fp,
                        graph=pot.graph,
                        growing_nodes=[leaf],
                        title="Growing Retropaths network",
                        note=f"Growing node {leaf} ({offset}/{len(growable_nodes)}) in layer {layer_counter + 1}.",
                        phase="growing",
                    )
                    pot.grow_this_node(
                        leaf,
                        filtered_library,
                        filter_minor_products=True,
                        use_father_error=False,
                    )
                    pot.check_for_number_of_nodes(workspace.max_nodes, started_at)
                layer_counter += 1
    except TimeoutPot as exc:
        _write_growth_progress(
            progress_fp,
            graph=pot.graph,
            growing_nodes=[],
            title="Growing Retropaths network",
            note=f"Timed out after {workspace.timeout_seconds} seconds.",
            phase="failed",
        )
        raise TimeoutError(
            f"Retropaths growth timed out after {workspace.timeout_seconds} seconds."
        ) from exc
    except Exception as exc:
        _write_growth_progress(
            progress_fp,
            graph=pot.graph,
            growing_nodes=[],
            title="Growing Retropaths network",
            note=f"Growth stopped after a partial network was built: {type(exc).__name__}: {exc}",
            phase="failed",
        )
        with contextlib.suppress(Exception):
            pot.run_time = time() - started_at
            pot.to_json(workspace.retropaths_pot_fp)
            neb_pot = retropaths_pot_to_neb_pot(pot)
            neb_pot.write_to_disk(workspace.neb_pot_fp)
            build_retropaths_neb_queue(neb_pot, queue_fp=workspace.queue_fp)
        raise

    pot.run_time = time() - started_at
    pot.to_json(workspace.retropaths_pot_fp)
    neb_pot = retropaths_pot_to_neb_pot(pot)
    neb_pot.write_to_disk(workspace.neb_pot_fp)
    queue = build_retropaths_neb_queue(neb_pot, queue_fp=workspace.queue_fp)
    _write_growth_progress(
        progress_fp,
        graph=neb_pot.graph,
        growing_nodes=[],
        title="Growing Retropaths network",
        note="Growth finished.",
        phase="finished",
    )
    return neb_pot, queue


def _merge_pot_overlay(base_pot: Pot, overlay_pot: Pot) -> Pot:
    merged = base_pot
    for node_index in overlay_pot.graph.nodes:
        attrs = dict(overlay_pot.graph.nodes[node_index])
        if node_index in merged.graph.nodes:
            merged.graph.nodes[node_index].update(attrs)
        else:
            merged.graph.add_node(node_index, **attrs)
    for source_node, target_node in overlay_pot.graph.edges:
        attrs = dict(overlay_pot.graph.edges[(source_node, target_node)])
        if merged.graph.has_edge(source_node, target_node):
            merged.graph.edges[(source_node, target_node)].update(attrs)
        else:
            merged.graph.add_edge(source_node, target_node, **attrs)
    return merged


def materialize_drive_graph(workspace: RetropathsWorkspace) -> Pot:
    base_pot = Pot.read_from_disk(workspace.neb_pot_fp) if workspace.neb_pot_fp.exists() else None
    try:
        overlay_pot = load_partial_annotated_pot(workspace)
    except Exception:
        if base_pot is None:
            raise
        pot = base_pot
    else:
        pot = _merge_pot_overlay(base_pot, overlay_pot) if base_pot is not None else overlay_pot
    pot.write_to_disk(workspace.neb_pot_fp)
    build_retropaths_neb_queue(pot=pot, queue_fp=workspace.queue_fp, overwrite=False)
    return pot


def _node_graph_like_molecule(node_attrs: dict[str, Any]) -> Molecule | None:
    molecule = node_attrs.get("molecule")
    if molecule is not None:
        return molecule.copy()
    td = node_attrs.get("td")
    graph = getattr(td, "graph", None)
    if graph is not None:
        return graph.copy()
    structure = getattr(td, "structure", None)
    if structure is not None:
        with contextlib.suppress(Exception):
            return structure_to_molecule(structure)
    return None


def _molecule_key(molecule: Molecule | None) -> str:
    if molecule is None:
        return ""
    with contextlib.suppress(Exception):
        return str(molecule.smiles_from_multiple_molecules())
    with contextlib.suppress(Exception):
        return str(molecule.force_smiles())
    return ""


def _find_matching_node_by_molecule(pot: Pot, molecule: Molecule | None) -> int | None:
    key = _molecule_key(molecule)
    if not key:
        return None
    for node_index in pot.graph.nodes:
        existing = _node_graph_like_molecule(pot.graph.nodes[node_index])
        if _molecule_key(existing) == key:
            return int(node_index)
    return None


def _coerce_retropaths_molecule(molecule: Any, MoleculeCls: Any) -> Any | None:
    if molecule is None:
        return None
    if isinstance(molecule, MoleculeCls):
        return molecule.copy()
    smiles = _molecule_key(molecule)
    if smiles:
        with contextlib.suppress(Exception):
            return MoleculeCls.from_smiles(smiles)
    return None


def _dense_retropaths_molecule(molecule: Any, MoleculeCls: Any) -> Any | None:
    if molecule is None:
        return None
    dense = MoleculeCls()
    old_to_new = {
        old_index: new_index for new_index, old_index in enumerate(sorted(molecule.nodes))
    }
    for old_index in sorted(molecule.nodes):
        dense.add_node(old_to_new[old_index], **dict(molecule.nodes[old_index]))
    for atom1, atom2, attrs in molecule.edges(data=True):
        dense.add_edge(old_to_new[atom1], old_to_new[atom2], **dict(attrs))
    if hasattr(dense, "set_neighbors"):
        dense.set_neighbors()
    return dense


def _nanoreactor_support_status(run_inputs: RunInputs) -> tuple[bool, str]:
    engine_name = str(getattr(run_inputs, "engine_name", "") or "").strip().lower()
    program = str(getattr(run_inputs, "program", "") or "").strip().lower()
    if engine_name not in {"chemcloud", "qcop"}:
        return False, "Nanoreactor sampling currently requires a QCOP/ChemCloud-backed inputs file."
    if "terachem" in program or "crest" in program:
        return True, ""
    return False, "Nanoreactor sampling currently requires a CREST or TeraChem-backed inputs file."


def _hessian_sample_support_status(run_inputs: RunInputs) -> tuple[bool, str]:
    engine = getattr(run_inputs, "engine", None)
    if engine is None:
        return False, "The inputs file did not construct an engine, so Hessian sampling cannot be started."
    if not hasattr(engine, "_compute_hessian_result"):
        return (
            False,
            "The configured engine does not expose `_compute_hessian_result`, which is required for Hessian sampling.",
        )
    if not (hasattr(engine, "compute_geometry_optimization") or hasattr(engine, "compute_geometry_optimizations")):
        return (
            False,
            "The configured engine does not expose geometry-optimization methods needed for Hessian sampling.",
        )
    return True, ""


def _extract_normal_modes_from_hessian_result(
    hessres: Any,
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
    result_dir = str(getattr(item, "result_dir", "") or "").strip()
    if not result_dir:
        return None
    with contextlib.suppress(Exception):
        history = TreeNode.read_from_disk(
            folder_name=result_dir,
            charge=0,
            multiplicity=1,
        )
        return getattr(history, "output_chain", None)
    return None


def _coerce_chain_candidate(candidate: Any) -> Any | None:
    if candidate is None:
        return None
    if isinstance(candidate, Chain):
        return candidate
    if hasattr(candidate, "nodes"):
        with contextlib.suppress(Exception):
            if len(candidate.nodes) > 0:
                return candidate
    return None


def _find_completed_chain_for_edge(
    workspace: RetropathsWorkspace,
    pot: Pot,
    *,
    source_node: int,
    target_node: int,
) -> Any | None:
    for source, target in ((source_node, target_node), (target_node, source_node)):
        if not pot.graph.has_edge(source, target):
            continue
        edge_attrs = dict(pot.graph.edges[(source, target)])
        chains = list(edge_attrs.get("list_of_nebs") or [])
        while chains:
            candidate = _coerce_chain_candidate(chains.pop())
            if candidate is not None:
                return candidate

    if not workspace.queue_fp.exists():
        return None
    with contextlib.suppress(Exception):
        queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)
    if "queue" not in locals():
        return None

    matching_items: list[Any] = []
    for item in queue.items:
        if item.status != "completed":
            continue
        pair = (int(item.source_node), int(item.target_node))
        if pair in {(int(source_node), int(target_node)), (int(target_node), int(source_node))}:
            matching_items.append(item)
    if not matching_items:
        return None

    matching_items.sort(key=lambda item: str(getattr(item, "finished_at", "") or ""))
    for item in reversed(matching_items):
        chain = _load_completed_queue_chain(item)
        candidate = _coerce_chain_candidate(chain)
        if candidate is None:
            continue
        with contextlib.suppress(Exception):
            if len(candidate) > 0:
                return candidate
        with contextlib.suppress(Exception):
            if len(getattr(candidate, "nodes", [])) > 0:
                return candidate
    return None


def _safe_node_energy(node: Any) -> float | None:
    with contextlib.suppress(Exception):
        value = node.energy
        if value is not None:
            return float(value)
    with contextlib.suppress(Exception):
        value = getattr(node, "_cached_energy", None)
        if value is not None:
            return float(value)
    with contextlib.suppress(Exception):
        result = getattr(node, "_cached_result", None)
        if result is not None:
            return float(result.results.energy)
    return None


def _peak_node_from_chain(chain: Any) -> StructureNode:
    nodes = list(getattr(chain, "nodes", []) or [])
    if not nodes:
        with contextlib.suppress(Exception):
            nodes = [chain[index] for index in range(len(chain))]
    if not nodes:
        raise ValueError("The selected NEB chain does not contain any geometries.")

    peak_index: int | None = None
    peak_energy: float | None = None
    for index, node in enumerate(nodes):
        energy = _safe_node_energy(node)
        if energy is None:
            continue
        if peak_energy is None or energy > peak_energy:
            peak_energy = float(energy)
            peak_index = int(index)

    if peak_index is None:
        with contextlib.suppress(Exception):
            energies = np.asarray(getattr(chain, "energies"), dtype=float)
            if energies.shape[0] == len(nodes):
                finite_mask = np.isfinite(energies)
                if bool(np.any(finite_mask)):
                    masked = np.where(finite_mask, energies, -np.inf)
                    peak_index = int(np.argmax(masked))
    if peak_index is None:
        peak_index = int(len(nodes) // 2)

    peak_node = nodes[peak_index]
    if not isinstance(peak_node, StructureNode):
        raise ValueError("The selected NEB chain peak is not a StructureNode geometry.")
    return peak_node.copy()


def _ensure_node_graph(node: StructureNode, *, fallback: StructureNode | None = None) -> StructureNode:
    graph_like = getattr(node, "graph", None)
    if graph_like is None and fallback is not None:
        graph_like = getattr(fallback, "graph", None)
    if graph_like is None and getattr(node, "structure", None) is not None:
        with contextlib.suppress(Exception):
            graph_like = structure_to_molecule(node.structure)
    if graph_like is not None:
        node.graph = graph_like
        node.has_molecular_graph = True
    return node


def _final_optimized_node(result: Any) -> StructureNode | None:
    if isinstance(result, StructureNode):
        return result
    nodes = getattr(result, "nodes", None)
    if isinstance(nodes, list) and nodes:
        last = nodes[-1]
        if isinstance(last, StructureNode):
            return last
    with contextlib.suppress(Exception):
        if len(result) > 0:
            last = result[-1]
            if isinstance(last, StructureNode):
                return last
    return None


def _run_hessian_sample(
    workspace: RetropathsWorkspace,
    pot: Pot,
    seed_node: StructureNode,
    *,
    dr: float,
    max_candidates: int,
    source_label: str,
    growing_nodes: list[int],
    provenance_node_index: int | None = None,
    provenance_edge: tuple[int, int] | None = None,
    progress_fp: str | None = None,
) -> dict[str, Any]:
    if dr <= 0:
        raise ValueError("The Hessian-sample displacement (`dr`) must be positive.")
    if int(max_candidates) <= 0:
        raise ValueError("`max_candidates` must be a positive integer.")
    if seed_node is None or getattr(seed_node, "structure", None) is None:
        raise ValueError("The selected geometry has no 3D structure, so Hessian sampling cannot be started.")

    seed_node = _ensure_node_graph(seed_node.copy())
    run_inputs = RunInputs.open(workspace.inputs_fp)
    supported, support_note = _hessian_sample_support_status(run_inputs)
    if not supported:
        raise ValueError(support_note)

    engine = run_inputs.engine
    _write_growth_progress(
        progress_fp,
        graph=pot.graph,
        growing_nodes=list(growing_nodes),
        title=f"Running Hessian sample from {source_label}",
        note=f"Computing Hessian normal modes with dr={float(dr):.4f}.",
        phase="growing",
    )

    try:
        hessres = engine._compute_hessian_result(seed_node)
    except Exception as exc:
        hessres = getattr(exc, "program_output", None)
        if hessres is None:
            raise

    normal_modes, _frequencies = _extract_normal_modes_from_hessian_result(hessres)
    if len(normal_modes) == 0:
        raise ValueError("No normal modes were returned from the Hessian result.")

    max_candidates = int(max_candidates)
    maxiter = 500
    displaced_nodes: list[StructureNode] = []
    clipped = False
    for mode in normal_modes:
        for signed_dr in (float(dr), -float(dr)):
            displaced = displace_by_dr(node=seed_node, displacement=np.array(mode), dr=signed_dr)
            displaced = _ensure_node_graph(displaced, fallback=seed_node)
            displaced_nodes.append(displaced)
            if len(displaced_nodes) >= max_candidates:
                clipped = True
                break
        if clipped:
            break
    if not displaced_nodes:
        raise ValueError("No displaced candidate geometries were generated from the Hessian normal modes.")

    _write_growth_progress(
        progress_fp,
        graph=pot.graph,
        growing_nodes=list(growing_nodes),
        title=f"Running Hessian sample from {source_label}",
        note=f"Optimizing {len(displaced_nodes)} displaced structure(s) into minima.",
        phase="optimizing",
    )

    optimized_nodes: list[StructureNode] = []
    compute_single = getattr(engine, "compute_geometry_optimization", None)
    compute_many = getattr(engine, "compute_geometry_optimizations", None)
    engine_name = str(getattr(run_inputs, "engine_name", "") or "").strip().lower()
    compute_program = str(getattr(engine, "compute_program", "") or "").strip().lower()
    use_chemcloud_batch = engine_name == "chemcloud" or compute_program == "chemcloud"

    if use_chemcloud_batch:
        if not callable(compute_many):
            raise ValueError(
                "ChemCloud Hessian sampling requires batch geometry optimization, but this engine does not expose `compute_geometry_optimizations`."
            )
        _write_growth_progress(
            progress_fp,
            graph=pot.graph,
            growing_nodes=list(growing_nodes),
            title=f"Running Hessian sample from {source_label}",
            note=f"Submitting {len(displaced_nodes)} displaced structure(s) to ChemCloud as one batch.",
            phase="optimizing",
        )
        try:
            try:
                optimized_histories = compute_many(
                    displaced_nodes,
                    keywords={"coordsys": "cart", "maxiter": int(maxiter)},
                )
            except TypeError:
                optimized_histories = compute_many(displaced_nodes)
        except Exception as exc:
            raise ValueError(
                f"ChemCloud batch geometry optimization failed during Hessian sampling: {type(exc).__name__}: {exc}"
            ) from exc
        if len(optimized_histories) != len(displaced_nodes):
            if len(optimized_histories) < len(displaced_nodes):
                optimized_histories = list(optimized_histories) + [
                    [] for _ in range(len(displaced_nodes) - len(optimized_histories))
                ]
            else:
                optimized_histories = list(optimized_histories[: len(displaced_nodes)])
        for candidate, result in zip(displaced_nodes, optimized_histories):
            final_node = _final_optimized_node(result)
            if final_node is None:
                continue
            final_node = _ensure_node_graph(final_node, fallback=candidate)
            optimized_nodes.append(final_node)
    else:
        if callable(compute_many):
            with contextlib.suppress(Exception):
                try:
                    optimized_histories = compute_many(
                        displaced_nodes,
                        keywords={"coordsys": "cart", "maxiter": int(maxiter)},
                    )
                except TypeError:
                    optimized_histories = compute_many(displaced_nodes)
                if len(optimized_histories) == len(displaced_nodes):
                    for candidate, result in zip(displaced_nodes, optimized_histories):
                        final_node = _final_optimized_node(result)
                        if final_node is None:
                            continue
                        final_node = _ensure_node_graph(final_node, fallback=candidate)
                        optimized_nodes.append(final_node)

        if not optimized_nodes:
            for candidate in displaced_nodes:
                result = None
                if compute_single is not None:
                    try:
                        result = compute_single(
                            candidate, keywords={"coordsys": "cart", "maxiter": int(maxiter)}
                        )
                    except Exception:
                        with contextlib.suppress(Exception):
                            result = compute_single(candidate)
                elif compute_many is not None:
                    with contextlib.suppress(Exception):
                        batch = compute_many([candidate]) or []
                        if batch:
                            result = batch[0]
                final_node = _final_optimized_node(result)
                if final_node is None:
                    continue
                final_node = _ensure_node_graph(final_node, fallback=candidate)
                optimized_nodes.append(final_node)

    if not optimized_nodes:
        _write_growth_progress(
            progress_fp,
            graph=pot.graph,
            growing_nodes=[],
            title=f"Running Hessian sample from {source_label}",
            note="The Hessian sample run completed, but no displaced candidate converged to a minimum.",
            phase="finished",
        )
        return {
            "message": f"Hessian sampling from {source_label} returned no converged minima.",
            "added_nodes": 0,
            "added_edges": 0,
            "targets": [],
            "dr": float(dr),
        }

    chain_inputs = getattr(run_inputs, "chain_inputs", ChainInputs())
    network_inputs = getattr(run_inputs, "network_inputs", None)
    collapse_rms_thre = float(getattr(network_inputs, "collapse_node_rms_thre", chain_inputs.node_rms_thre))
    collapse_ene_thre = float(getattr(network_inputs, "collapse_node_ene_thre", chain_inputs.node_ene_thre))

    provenance_sources: list[int] = []
    if provenance_node_index is not None:
        provenance_sources = [int(provenance_node_index)]
    elif provenance_edge is not None:
        provenance_sources = [int(provenance_edge[0]), int(provenance_edge[1])]

    added_nodes = 0
    added_edges = 0
    skipped_duplicates = 0
    merged_targets: list[int] = []
    hessian_reaction_index = 1

    for offset, optimized_node in enumerate(optimized_nodes, start=1):
        _write_growth_progress(
            progress_fp,
            graph=pot.graph,
            growing_nodes=list(growing_nodes),
            title=f"Running Hessian sample from {source_label}",
            note=f"Merging optimized minimum {offset}/{len(optimized_nodes)} into the network.",
            phase="merging",
        )
        if _nodes_same_species(optimized_node, seed_node):
            skipped_duplicates += 1
            continue

        target_index = _find_existing_network_node(
            pot,
            optimized_node,
            chain_inputs,
            collapse_rms_thre,
            collapse_ene_thre,
        )
        if target_index is None:
            target_index = (max(pot.graph.nodes) + 1) if pot.graph.nodes else 0
            graph_like = getattr(optimized_node, "graph", None)
            pot.graph.add_node(
                target_index,
                molecule=copy_graph_like_molecule(graph_like) if graph_like is not None else None,
                converged=True,
                td=optimized_node,
                endpoint_optimized=True,
                generated_by="hessian_sample",
                hessian_sample_dr=float(dr),
                hessian_provenance_node=int(provenance_node_index) if provenance_node_index is not None else None,
                hessian_provenance_edge=[int(provenance_edge[0]), int(provenance_edge[1])] if provenance_edge is not None else None,
            )
            added_nodes += 1
        else:
            target_attrs = pot.graph.nodes[target_index]
            graph_like = getattr(optimized_node, "graph", None)
            if graph_like is not None:
                target_attrs["molecule"] = copy_graph_like_molecule(graph_like)
            target_attrs["td"] = optimized_node
            target_attrs["endpoint_optimized"] = True
            target_attrs.setdefault("generated_by", "hessian_sample")
            target_attrs["hessian_sample_dr"] = float(dr)
            if provenance_node_index is not None:
                target_attrs.setdefault("hessian_provenance_node", int(provenance_node_index))
            if provenance_edge is not None:
                target_attrs.setdefault(
                    "hessian_provenance_edge",
                    [int(provenance_edge[0]), int(provenance_edge[1])],
                )

        for source_index in provenance_sources:
            if int(target_index) == int(source_index):
                skipped_duplicates += 1
                continue
            if not pot.graph.has_edge(source_index, target_index):
                if provenance_node_index is not None:
                    reaction_label = f"Hessian sample {hessian_reaction_index}"
                else:
                    reaction_label = (
                        f"Hessian sample edge {int(provenance_edge[0])}->{int(provenance_edge[1])} "
                        f"{hessian_reaction_index}"
                    )
                pot.graph.add_edge(
                    source_index,
                    target_index,
                    reaction=reaction_label,
                    list_of_nebs=[],
                    generated_by="hessian_sample",
                )
                added_edges += 1
                hessian_reaction_index += 1
            ensure_queue_item_for_edge(
                pot=pot,
                source_node=int(source_index),
                target_node=int(target_index),
                queue_fp=workspace.queue_fp,
                overwrite=False,
            )
        merged_targets.append(int(target_index))

    pot.write_to_disk(workspace.neb_pot_fp)
    build_retropaths_neb_queue(pot=pot, queue_fp=workspace.queue_fp, overwrite=False)
    _write_growth_progress(
        progress_fp,
        graph=pot.graph,
        growing_nodes=[],
        title=f"Running Hessian sample from {source_label}",
        note=(
            f"Merged {added_nodes} new minima node(s), created {added_edges} new edge(s), "
            f"and skipped {skipped_duplicates} duplicate minima."
        ),
        phase="finished",
    )
    return {
        "message": (
            f"Hessian sampling from {source_label} merged {added_nodes} new minima "
            f"and {added_edges} new edges."
        ),
        "added_nodes": int(added_nodes),
        "added_edges": int(added_edges),
        "targets": sorted(set(int(target) for target in merged_targets)),
        "dr": float(dr),
    }


def run_hessian_sample_for_node(
    workspace: RetropathsWorkspace,
    node_index: int,
    *,
    dr: float,
    max_candidates: int = 100,
    progress_fp: str | None = None,
) -> dict[str, Any]:
    pot = materialize_drive_graph(workspace)
    if node_index not in pot.graph.nodes:
        raise ValueError(f"Node {node_index} is not present in the current workspace.")
    source_attrs = pot.graph.nodes[node_index]
    source_td = source_attrs.get("td")
    if source_td is None or getattr(source_td, "structure", None) is None:
        raise ValueError(f"Node {node_index} has no 3D structure, so Hessian sampling cannot be started.")
    return _run_hessian_sample(
        workspace,
        pot,
        source_td,
        dr=float(dr),
        max_candidates=int(max_candidates),
        source_label=f"node {int(node_index)}",
        growing_nodes=[int(node_index)],
        provenance_node_index=int(node_index),
        progress_fp=progress_fp,
    )


def run_hessian_sample_for_edge(
    workspace: RetropathsWorkspace,
    source_node: int,
    target_node: int,
    *,
    dr: float,
    max_candidates: int = 100,
    progress_fp: str | None = None,
) -> dict[str, Any]:
    pot = materialize_drive_graph(workspace)
    forward_exists = pot.graph.has_edge(int(source_node), int(target_node))
    reverse_exists = pot.graph.has_edge(int(target_node), int(source_node))
    if not (forward_exists or reverse_exists):
        raise ValueError(
            f"Edge {int(source_node)} -> {int(target_node)} is not present in the current workspace."
        )

    chain = _find_completed_chain_for_edge(
        workspace,
        pot,
        source_node=int(source_node),
        target_node=int(target_node),
    )
    if chain is None:
        raise ValueError(
            "Hessian sampling from an edge requires a completed NEB chain on that edge (directly or via queue results)."
        )
    peak_node = _peak_node_from_chain(chain)
    if getattr(peak_node, "structure", None) is None:
        raise ValueError("The selected edge's NEB peak has no 3D structure, so Hessian sampling cannot be started.")
    return _run_hessian_sample(
        workspace,
        pot,
        peak_node,
        dr=float(dr),
        max_candidates=int(max_candidates),
        source_label=f"edge {int(source_node)} -> {int(target_node)} peak",
        growing_nodes=[int(source_node), int(target_node)],
        provenance_edge=(int(source_node), int(target_node)),
        progress_fp=progress_fp,
    )

def apply_reactions_to_node(
    workspace: RetropathsWorkspace,
    node_index: int,
    *,
    progress_fp: str | None = None,
) -> dict[str, Any]:
    pot = materialize_drive_graph(workspace)
    if node_index not in pot.graph.nodes:
        raise ValueError(f"Node {node_index} is not present in the current workspace.")

    _write_growth_progress(
        progress_fp,
        graph=pot.graph,
        growing_nodes=[int(node_index)],
        title=f"Applying reaction templates to node {node_index}",
        note=f"Growing node {node_index} with the Retropaths template library.",
        phase="growing",
    )

    source_attrs = pot.graph.nodes[node_index]
    source_molecule = _node_graph_like_molecule(source_attrs)
    if source_molecule is None:
        raise ValueError(f"Node {node_index} has no molecular graph, so reactions cannot be applied to it.")

    charge = 0
    multiplicity = 1
    source_td = source_attrs.get("td")
    if source_td is not None and getattr(source_td, "structure", None) is not None:
        charge = int(source_td.structure.charge)
        multiplicity = int(source_td.structure.multiplicity)

    hf, RetropathsMolecule, RetropathsPot = _load_retropaths_classes()
    library = hf.pload(workspace.reactions_path)
    source_retropaths_molecule = _coerce_retropaths_molecule(source_molecule, RetropathsMolecule)
    environment_retropaths_molecule = _coerce_retropaths_molecule(
        _global_environment_molecule(pot),
        RetropathsMolecule,
    )
    if source_retropaths_molecule is None:
        raise ValueError(
            f"Node {node_index} could not be converted into a Retropaths-compatible molecular graph."
        )
    temp_pot = RetropathsPot(
        root=source_retropaths_molecule,
        environment=environment_retropaths_molecule or RetropathsMolecule(),
        rxn_name=f"{workspace.run_name}-node-{node_index}",
    )
    temp_pot.grow_this_node(0, library, filter_minor_products=True, use_father_error=False)

    added_nodes = 0
    added_edges = 0
    merged_targets: list[int] = []

    for result_index in sorted(temp_pot.graph.nodes):
        if result_index == 0:
            continue
        result_attrs = temp_pot.graph.nodes[result_index]
        result_molecule = result_attrs.get("molecule")
        if result_molecule is None:
            continue

        target_index = _find_matching_node_by_molecule(pot, result_molecule)
        if target_index is None:
            target_index = (max(pot.graph.nodes) + 1) if pot.graph.nodes else 0
            td = structure_node_from_graph_like_molecule(
                result_molecule,
                charge=charge,
                spinmult=multiplicity,
            )
            pot.graph.add_node(
                target_index,
                molecule=copy_graph_like_molecule(result_molecule),
                converged=bool(result_attrs.get("converged", False)),
                td=td,
                endpoint_optimized=False,
                generated_by="retropaths_apply_reactions",
            )
            added_nodes += 1
        else:
            target_attrs = pot.graph.nodes[target_index]
            target_attrs.setdefault("molecule", copy_graph_like_molecule(result_molecule))
            if target_attrs.get("td") is None:
                target_attrs["td"] = structure_node_from_graph_like_molecule(
                    result_molecule,
                    charge=charge,
                    spinmult=multiplicity,
                )
                target_attrs.setdefault("endpoint_optimized", False)

        if not pot.graph.has_edge(target_index, node_index):
            edge_attrs = dict(temp_pot.graph.edges[(result_index, 0)])
            edge_attrs.setdefault("list_of_nebs", [])
            edge_attrs.setdefault("generated_by", "retropaths_apply_reactions")
            pot.graph.add_edge(target_index, node_index, **edge_attrs)
            added_edges += 1
        merged_targets.append(int(target_index))

    pot.write_to_disk(workspace.neb_pot_fp)
    build_retropaths_neb_queue(pot=pot, queue_fp=workspace.queue_fp, overwrite=False)
    _write_growth_progress(
        progress_fp,
        graph=pot.graph,
        growing_nodes=[],
        title=f"Applying reaction templates to node {node_index}",
        note=f"Merged {added_nodes} new node(s) and {added_edges} new edge(s).",
        phase="finished",
    )
    return {
        "message": (
            f"Applied reactions to node {node_index}: merged {added_nodes} new nodes "
            f"and {added_edges} new edges."
        ),
        "node_index": int(node_index),
        "added_nodes": int(added_nodes),
        "added_edges": int(added_edges),
        "targets": merged_targets,
    }


def run_nanoreactor_for_node(
    workspace: RetropathsWorkspace,
    node_index: int,
    *,
    progress_fp: str | None = None,
) -> dict[str, Any]:
    pot = materialize_drive_graph(workspace)
    if node_index not in pot.graph.nodes:
        raise ValueError(f"Node {node_index} is not present in the current workspace.")

    source_attrs = pot.graph.nodes[node_index]
    source_td = source_attrs.get("td")
    if source_td is None or getattr(source_td, "structure", None) is None:
        raise ValueError(f"Node {node_index} has no 3D structure, so nanoreactor sampling cannot be started.")
    provenance_node_index = int(node_index)

    run_inputs = RunInputs.open(workspace.inputs_fp)
    supported, support_note = _nanoreactor_support_status(run_inputs)
    if not supported:
        raise ValueError(support_note)

    program_name = str(getattr(run_inputs, "program", "") or "").strip().lower()
    backend_label = "TeraChem MD nanoreactor" if "terachem" in program_name else "CREST MSREACT nanoreactor"
    _write_growth_progress(
        progress_fp,
        graph=pot.graph,
        growing_nodes=[int(node_index)],
        title=f"Running {backend_label} from node {node_index}",
        note=f"Submitting {backend_label} sampling and collecting candidate structures.",
        phase="growing",
    )

    candidate_nodes = list(
        getattr(run_inputs.engine, "compute_nanoreactor_candidates")(
            source_td,
            nanoreactor_inputs=dict(getattr(run_inputs, "nanoreactor_inputs", {}) or {}),
        )
        or []
    )
    if not candidate_nodes:
        _write_growth_progress(
            progress_fp,
            graph=pot.graph,
            growing_nodes=[],
            title=f"Running {backend_label} from node {node_index}",
            note="The nanoreactor run completed but returned no candidate structures.",
            phase="finished",
        )
        return {
            "message": f"Nanoreactor sampling from node {node_index} returned no candidate minima.",
            "node_index": int(node_index),
            "added_nodes": 0,
            "added_edges": 0,
            "targets": [],
        }

    _write_growth_progress(
        progress_fp,
        graph=pot.graph,
        growing_nodes=[int(node_index)],
        title=f"Running {backend_label} from node {node_index}",
        note=f"Optimizing {len(candidate_nodes)} candidate structure(s) into minima.",
        phase="optimizing",
    )

    try:
        optimized_histories = run_inputs.engine.compute_geometry_optimizations(candidate_nodes)
    except Exception:
        optimized_histories = []
        compute_single = getattr(run_inputs.engine, "compute_geometry_optimization")
        for candidate_node in candidate_nodes:
            with contextlib.suppress(Exception):
                optimized_histories.append(compute_single(candidate_node))
    optimized_nodes: list[StructureNode] = []
    for candidate_node, history in zip(candidate_nodes, optimized_histories):
        if not history:
            continue
        optimized_node = history[-1]
        graph_like = getattr(optimized_node, "graph", None) or getattr(candidate_node, "graph", None)
        if graph_like is None and getattr(optimized_node, "structure", None) is not None:
            with contextlib.suppress(Exception):
                graph_like = structure_to_molecule(optimized_node.structure)
        if graph_like is not None:
            optimized_node.graph = graph_like
            optimized_node.has_molecular_graph = True
        optimized_nodes.append(optimized_node)
    chain_inputs = getattr(run_inputs, "chain_inputs", ChainInputs())
    network_inputs = getattr(run_inputs, "network_inputs", None)
    collapse_rms_thre = float(getattr(network_inputs, "collapse_node_rms_thre", chain_inputs.node_rms_thre))
    collapse_ene_thre = float(getattr(network_inputs, "collapse_node_ene_thre", chain_inputs.node_ene_thre))
    added_nodes = 0
    added_edges = 0
    skipped_duplicates = 0
    merged_targets: list[int] = []
    nanoreactor_reaction_index = 1

    for offset, optimized_node in enumerate(optimized_nodes, start=1):
        _write_growth_progress(
            progress_fp,
            graph=pot.graph,
            growing_nodes=[int(node_index)],
            title=f"Running {backend_label} from node {node_index}",
            note=f"Merging optimized minimum {offset}/{len(optimized_nodes)} into the network.",
            phase="merging",
        )
        if _nodes_same_species(optimized_node, source_td):
            skipped_duplicates += 1
            continue

        target_index = _find_existing_network_node(
            pot,
            optimized_node,
            chain_inputs,
            collapse_rms_thre,
            collapse_ene_thre,
        )
        if target_index is None:
            target_index = (max(pot.graph.nodes) + 1) if pot.graph.nodes else 0
            graph_like = getattr(optimized_node, "graph", None)
            if graph_like is None and getattr(optimized_node, "structure", None) is not None:
                with contextlib.suppress(Exception):
                    graph_like = structure_to_molecule(optimized_node.structure)
                    optimized_node.graph = graph_like
            pot.graph.add_node(
                target_index,
                molecule=copy_graph_like_molecule(graph_like) if graph_like is not None else None,
                converged=True,
                td=optimized_node,
                endpoint_optimized=True,
                generated_by="nanoreactor_sampling",
                nanoreactor_provenance_node=int(provenance_node_index),
            )
            added_nodes += 1
        else:
            target_attrs = pot.graph.nodes[target_index]
            graph_like = getattr(optimized_node, "graph", None)
            if graph_like is not None:
                target_attrs["molecule"] = copy_graph_like_molecule(graph_like)
            target_attrs["td"] = optimized_node
            target_attrs["endpoint_optimized"] = True
            target_attrs.setdefault("nanoreactor_provenance_node", int(provenance_node_index))

        if int(target_index) == int(provenance_node_index):
            skipped_duplicates += 1
            continue
        if not pot.graph.has_edge(provenance_node_index, target_index):
            pot.graph.add_edge(
                provenance_node_index,
                target_index,
                reaction=f"Nanoreactor reaction {nanoreactor_reaction_index}",
                list_of_nebs=[],
                generated_by="nanoreactor_sampling",
            )
            added_edges += 1
            nanoreactor_reaction_index += 1
        ensure_queue_item_for_edge(
            pot=pot,
            source_node=int(provenance_node_index),
            target_node=int(target_index),
            queue_fp=workspace.queue_fp,
            overwrite=False,
        )
        merged_targets.append(int(target_index))

    pot.write_to_disk(workspace.neb_pot_fp)
    build_retropaths_neb_queue(pot=pot, queue_fp=workspace.queue_fp, overwrite=False)
    _write_growth_progress(
        progress_fp,
        graph=pot.graph,
        growing_nodes=[],
        title=f"Running {backend_label} from node {node_index}",
        note=(
            f"Merged {added_nodes} new minima node(s), created {added_edges} new edge(s), "
            f"and skipped {skipped_duplicates} duplicate minima."
        ),
        phase="finished",
    )
    return {
        "message": (
            f"Nanoreactor sampling from node {node_index} merged {added_nodes} new minima "
            f"and {added_edges} new edges."
        ),
        "node_index": int(node_index),
        "added_nodes": int(added_nodes),
        "added_edges": int(added_edges),
        "targets": merged_targets,
    }


def add_manual_edge(
    workspace: RetropathsWorkspace,
    *,
    source_node: int,
    target_node: int,
    reaction_label: str = "",
) -> dict[str, Any]:
    if int(source_node) == int(target_node):
        raise ValueError("Manual edges must connect two distinct nodes.")

    if not workspace.neb_pot_fp.exists():
        raise ValueError("The workspace has no NEB graph yet.")
    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    for node_index in (source_node, target_node):
        if int(node_index) not in pot.graph.nodes:
            raise ValueError(f"Node {node_index} is not present in the current workspace.")

    added = False
    if not pot.graph.has_edge(source_node, target_node):
        pot.graph.add_edge(
            source_node,
            target_node,
            reaction=str(reaction_label).strip() or f"Manual edge {source_node}->{target_node}",
            list_of_nebs=[],
            generated_by="manual_drive_edge",
        )
        added = True

    pot.write_to_disk(workspace.neb_pot_fp)
    queue = ensure_queue_item_for_edge(
        pot=pot,
        source_node=int(source_node),
        target_node=int(target_node),
        queue_fp=workspace.queue_fp,
        overwrite=False,
    )
    item = queue.find_item(source_node, target_node)
    return {
        "message": (
            f"{'Added' if added else 'Refreshed'} manual edge {source_node} -> {target_node}."
        ),
        "source_node": int(source_node),
        "target_node": int(target_node),
        "added": bool(added),
        "queue_status": item.status if item is not None else "",
        "queue_error": item.error if item is not None else "",
    }


def _safe_read_pot(fp: Path) -> Pot | None:
    try:
        return Pot.read_from_disk(fp)
    except Exception:
        return None


def _nodes_identical(node1: StructureNode, node2: StructureNode, chain_inputs: ChainInputs) -> bool:
    try:
        return is_identical(
            node1,
            node2,
            fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
            kcal_mol_cutoff=chain_inputs.node_ene_thre,
            verbose=False,
            collect_comparison=False,
        )
    except Exception:
        return False


def _history_nodes_identical(
    node1: StructureNode,
    node2: StructureNode,
    chain_inputs: ChainInputs,
    collapse_rms_thre: float,
    collapse_ene_thre: float,
) -> bool:
    try:
        return is_identical(
            node1,
            node2,
            fragment_rmsd_cutoff=collapse_rms_thre,
            kcal_mol_cutoff=collapse_ene_thre,
            verbose=False,
            collect_comparison=False,
        )
    except Exception:
        return False


def _nodes_same_species(node1: StructureNode, node2: StructureNode) -> bool:
    if getattr(node1, "has_molecular_graph", False) and getattr(node2, "has_molecular_graph", False):
        try:
            return _is_connectivity_identical(
                node1,
                node2,
                verbose=False,
                collect_comparison=False,
            )
        except Exception:
            return False
    return False


def _structure_node_molecule_key(node: StructureNode | None) -> str:
    if node is None:
        return ""
    graph = getattr(node, "graph", None)
    if graph is None and getattr(node, "structure", None) is not None:
        with contextlib.suppress(Exception):
            graph = structure_to_molecule(node.structure)
    return _molecule_key(graph)


def _network_node_matches_species(node: StructureNode, node_attrs: dict[str, Any]) -> bool:
    node_key = _structure_node_molecule_key(node)
    existing_key = _molecule_key(_node_graph_like_molecule(node_attrs))
    return bool(node_key) and node_key == existing_key


def _normalize_history_endpoint(
    node: StructureNode,
    candidates: list[tuple[int, StructureNode | None, object | None]],
    chain_inputs: ChainInputs,
) -> tuple[StructureNode, int | None]:
    for node_index, reference_node, reference_graph in candidates:
        if reference_node is None:
            continue
        graph = reference_graph
        if graph is not None and hasattr(graph, "is_empty") and graph.is_empty():
            graph = None
        normalized = _trim_balanced_endpoint(node, graph)
        normalized.has_molecular_graph = graph is not None
        if _structure_node_molecule_key(normalized) and _structure_node_molecule_key(normalized) == _structure_node_molecule_key(reference_node):
            return normalized, node_index

    graph = getattr(node, "graph", None)
    if graph is not None and hasattr(graph, "is_empty") and graph.is_empty():
        graph = None
    normalized = _trim_balanced_endpoint(node, graph)
    normalized.has_molecular_graph = graph is not None
    return normalized, None


def _find_existing_network_node(
    pot: Pot,
    node: StructureNode,
    chain_inputs: ChainInputs,
    collapse_rms_thre: float,
    collapse_ene_thre: float,
) -> int | None:
    for node_index in pot.graph.nodes:
        if _network_node_matches_species(node, pot.graph.nodes[node_index]):
            return node_index
    return None


def _register_network_node(
    pot: Pot,
    source_pot: Pot,
    node: StructureNode,
    chain_inputs: ChainInputs,
    collapse_rms_thre: float,
    collapse_ene_thre: float,
    preferred_index: int | None = None,
) -> int:
    if preferred_index is not None and preferred_index in source_pot.graph.nodes:
        if preferred_index not in pot.graph.nodes:
            pot.graph.add_node(preferred_index, **dict(source_pot.graph.nodes[preferred_index]))
        pot.graph.nodes[preferred_index]["td"] = node
        if getattr(node, "graph", None) is not None:
            pot.graph.nodes[preferred_index]["molecule"] = node.graph.copy()
        return preferred_index

    existing_index = _find_existing_network_node(
        pot,
        node,
        chain_inputs,
        collapse_rms_thre,
        collapse_ene_thre,
    )
    if existing_index is not None:
        return existing_index

    next_index = (max(pot.graph.nodes) + 1) if pot.graph.nodes else 0
    pot.graph.add_node(
        next_index,
        molecule=node.graph.copy() if node.graph is not None else None,
        converged=False,
        td=node,
        generated_by="recursive_msmep",
    )
    return next_index


def _history_leaf_chains(history: TreeNode) -> list[Chain]:
    chains: list[Chain] = []
    for leaf in history.ordered_leaves:
        if not leaf.data or not leaf.data.chain_trajectory:
            continue
        final_chain = leaf.data.chain_trajectory[-1]
        if len(final_chain.nodes) == 0:
            continue
        chains.append(final_chain.copy())
    return chains


def _elementary_step_label(base_reaction: str | None, leaf_index: int, total_leaves: int) -> str:
    reaction = (base_reaction or "Elementary-Step").strip() or "Elementary-Step"
    if total_leaves <= 1:
        return reaction
    return f"{reaction}(step {leaf_index})"


def _write_edge_visualizations(workspace: RetropathsWorkspace, pot: Pot) -> list[dict[str, str]]:
    from neb_dynamics.scripts.main_cli import _build_chain_visualizer_html

    rows: list[dict[str, str]] = []
    out_dir = workspace.edge_visualizations_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    display_run_inputs: RunInputs | None = None
    active_files: set[str] = set()

    for source_node, target_node in sorted(pot.graph.edges):
        edge_attrs = pot.graph.edges[(source_node, target_node)]
        filename = f"edge_{source_node}_{target_node}.html"
        meta_filename = f"edge_{source_node}_{target_node}.meta.json"
        out_fp = out_dir / filename
        meta_fp = out_dir / meta_filename
        active_files.add(filename)
        active_files.add(meta_filename)
        chains = edge_attrs.get("list_of_nebs") or []
        cache_key = {
            "source_node": int(source_node),
            "target_node": int(target_node),
            "reaction": str(edge_attrs.get("reaction") or ""),
            "barrier": (
                float(edge_attrs["barrier"])
                if edge_attrs.get("barrier") is not None
                else None
            ),
            "chains": len(chains),
            "source_molecule_key": _molecule_key(_node_graph_like_molecule(pot.graph.nodes[source_node])),
            "target_molecule_key": _molecule_key(_node_graph_like_molecule(pot.graph.nodes[target_node])),
        }
        if out_fp.exists() and meta_fp.exists():
            with contextlib.suppress(Exception):
                meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                if meta.get("cache_key") == cache_key:
                    rows.append(
                        {
                            "edge": str(meta.get("edge") or f"{source_node} -> {target_node}"),
                            "start": str(meta.get("start") or source_node),
                            "end": str(meta.get("end") or target_node),
                            "reaction": str(meta.get("reaction") or cache_key["reaction"]),
                            "barrier": str(meta.get("barrier") or ""),
                            "chains": str(meta.get("chains") or len(chains)),
                            "href": filename,
                            "source_structure": meta.get("source_structure"),
                            "target_structure": meta.get("target_structure"),
                        }
                    )
                    continue

        chain = chains[-1] if chains else None
        if chain is None:
            with contextlib.suppress(Exception):
                if display_run_inputs is None:
                    display_run_inputs = RunInputs.open(workspace.inputs_fp)
                preview_item = NEBQueueItem(
                    job_id=f"{int(source_node)}->{int(target_node)}",
                    source_node=int(source_node),
                    target_node=int(target_node),
                    reaction=str(edge_attrs.get("reaction") or ""),
                    attempt_key="preview",
                )
                chain = _make_pair_chain(
                    pot=pot,
                    item=preview_item,
                    run_inputs=display_run_inputs,
                )
        if chain is None:
            continue

        source_structure = None
        target_structure = None
        if len(chain.nodes) > 0:
            with contextlib.suppress(Exception):
                source_structure = {
                    "xyz_b64": base64.b64encode(chain.nodes[0].structure.to_xyz().encode("utf-8")).decode("ascii"),
                    "symbols": list(chain.nodes[0].structure.symbols),
                }
            with contextlib.suppress(Exception):
                target_structure = {
                    "xyz_b64": base64.b64encode(chain.nodes[-1].structure.to_xyz().encode("utf-8")).decode("ascii"),
                    "symbols": list(chain.nodes[-1].structure.symbols),
                }
        start_smiles = ""
        end_smiles = ""
        if len(chain.nodes) > 0 and getattr(chain[0], "graph", None) is not None:
            with contextlib.suppress(Exception):
                start_smiles = _quiet_force_smiles(chain[0].graph)
        if len(chain.nodes) > 0 and getattr(chain[-1], "graph", None) is not None:
            with contextlib.suppress(Exception):
                end_smiles = _quiet_force_smiles(chain[-1].graph)
        title_html = (
            f"<div style=\"font-family: -apple-system, BlinkMacSystemFont, sans-serif; "
            f"margin: 0 0 12px 0; padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa;\">"
            f"<div><strong>Edge:</strong> {source_node} -&gt; {target_node}</div>"
            f"<div><strong>Start:</strong> {start_smiles or source_node}</div>"
            f"<div><strong>End:</strong> {end_smiles or target_node}</div>"
            f"</div>"
        )
        html = _build_chain_visualizer_html(
            chain=chain,
            chain_trajectory=list(chains) if len(chains) > 1 else None,
        )
        html = html.replace("<body>", f"<body>\n  {title_html}", 1)
        out_fp.write_text(html, encoding="utf-8")
        row = {
            "edge": f"{source_node} -> {target_node}",
            "start": start_smiles or str(source_node),
            "end": end_smiles or str(target_node),
            "reaction": str(edge_attrs.get("reaction") or ""),
            "barrier": (
                f"{float(edge_attrs['barrier']):.3f}"
                if edge_attrs.get("barrier") is not None
                else ""
            ),
            "chains": str(len(chains)),
            "href": filename,
            "source_structure": source_structure,
            "target_structure": target_structure,
        }
        rows.append(row)
        with contextlib.suppress(Exception):
            meta_fp.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        **row,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    for stale_fp in out_dir.glob("edge_*"):
        if stale_fp.name not in active_files:
            stale_fp.unlink(missing_ok=True)

    return rows


def _json_safe(
    value: Any,
    depth: int = 0,
    *,
    _seen: set[int] | None = None,
) -> Any:
    if depth > 3:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return repr(value)
    if isinstance(value, dict):
        _seen.add(value_id)
        return {
            str(key): _json_safe(val, depth + 1, _seen=_seen)
            for key, val in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple, set)):
        _seen.add(value_id)
        return [_json_safe(item, depth + 1, _seen=_seen) for item in list(value)[:40]]
    if hasattr(value, "force_smiles"):
        with contextlib.suppress(Exception):
            return _quiet_force_smiles(value)
    if hasattr(value, "smiles"):
        with contextlib.suppress(Exception):
            return str(value.smiles)
    if hasattr(value, "to_dict"):
        with contextlib.suppress(Exception):
            _seen.add(value_id)
            return _json_safe(value.to_dict(), depth + 1, _seen=_seen)
    if hasattr(value, "model_dump"):
        with contextlib.suppress(Exception):
            _seen.add(value_id)
            return _json_safe(value.model_dump(), depth + 1, _seen=_seen)
    if hasattr(value, "__dict__"):
        with contextlib.suppress(Exception):
            _seen.add(value_id)
            return {
                str(key): _json_safe(val, depth + 1, _seen=_seen)
                for key, val in list(vars(value).items())[:40]
                if not str(key).startswith("_") and not callable(val)
            }
    return repr(value)


def _load_template_payloads(workspace: RetropathsWorkspace) -> dict[str, dict[str, Any]]:
    try:
        hf, _Molecule, _RetropathsPot = _load_retropaths_classes()
        library = hf.pload(workspace.reactions_path)
    except Exception:
        return {}

    payloads: dict[str, dict[str, Any]] = {}
    for reaction_name, reaction_obj in getattr(library, "items", lambda: [])():
        visualization_html = ""
        draw = getattr(reaction_obj, "draw", None)
        if callable(draw):
            with contextlib.suppress(Exception):
                visualization_html = str(draw(string_mode=True, size=(420, 320)))
        payloads[str(reaction_name)] = {
            "name": str(reaction_name),
            "data": _json_safe(reaction_obj),
            "visualization_html": visualization_html,
        }
    return payloads


def _quiet_force_smiles(molecule_like: Any) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return str(molecule_like.force_smiles())


def _node_label_for_explorer(node_index: int, attrs: dict[str, Any]) -> str:
    molecule = attrs.get("molecule")
    if molecule is not None:
        if isinstance(molecule, str):
            return molecule
        if hasattr(molecule, "force_smiles"):
            with contextlib.suppress(Exception):
                return _quiet_force_smiles(molecule)
        if hasattr(molecule, "smiles"):
            with contextlib.suppress(Exception):
                return str(molecule.smiles)
    return str(node_index)


def _build_network_explorer_payload(
    graph,
    *,
    template_payloads: dict[str, dict[str, Any]] | None = None,
    edge_visualizations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    template_payloads = template_payloads or {}
    viewer_by_edge = {
        item["edge"]: f"edge_visualizations/{item['href']}"
        for item in (edge_visualizations or [])
    }

    nodes = []
    for node_index in sorted(graph.nodes):
        attrs = graph.nodes[node_index]
        nodes.append(
            {
                "id": int(node_index),
                "label": _node_label_for_explorer(node_index, attrs),
                "data": _json_safe(dict(attrs)),
            }
        )

    edges = []
    for source, target in sorted(graph.edges):
        attrs = graph.edges[(source, target)]
        reaction_name = str(attrs.get("reaction") or "")
        template_payload = _json_safe(template_payloads.get(reaction_name) or {})
        edge_key = f"{source} -> {target}"
        edges.append(
            {
                "source": int(source),
                "target": int(target),
                "reaction": reaction_name,
                "barrier": (
                    float(attrs["barrier"]) if attrs.get("barrier") is not None else None
                ),
                "chains": len(attrs.get("list_of_nebs") or []),
                "viewer_href": viewer_by_edge.get(edge_key),
                "data": _json_safe(
                    {
                        key: value
                        for key, value in dict(attrs).items()
                        if key != "list_of_nebs"
                    }
                ),
                "template": template_payload,
            }
        )

    return {"nodes": nodes, "edges": edges}


def summarize_queue(queue: RetropathsNEBQueue) -> dict[str, int]:
    statuses = ["pending", "running", "completed", "failed", "incompatible", "skipped_attempted", "missing_td"]
    counts = {status: 0 for status in statuses}
    for item in queue.items:
        counts[item.status] = counts.get(item.status, 0) + 1
    counts["items"] = len(queue.items)
    return counts


def _count_unoptimized_endpoints(pot: Pot) -> int:
    count = 0
    for node_index in pot.graph.nodes:
        node_attrs = pot.graph.nodes[node_index]
        if node_attrs.get("td") is None:
            continue
        if not node_attrs.get("endpoint_optimized"):
            count += 1
    return count


def _endpoint_optimization_dir(workspace: RetropathsWorkspace) -> Path:
    return workspace.directory / "endpoint_optimizations"


def _persist_endpoint_optimization_result(
    workspace: RetropathsWorkspace,
    node_index: int,
    optimized_td: StructureNode,
) -> str | None:
    result = getattr(optimized_td, "_cached_result", None)
    if result is None or not hasattr(result, "save"):
        return None

    out_dir = _endpoint_optimization_dir(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"node_{node_index}.qcio"
    result.save(out_fp)
    return str(out_fp.resolve())


def _strip_cached_result(node: StructureNode) -> StructureNode:
    return StructureNode(
        structure=node.structure,
        graph=node.graph,
        has_molecular_graph=node.has_molecular_graph,
        converged=node.converged,
        _cached_energy=node._cached_energy,
        _cached_gradient=node._cached_gradient,
        _cached_result=None,
    )


def _optimize_single_endpoint(
    node: StructureNode, run_inputs: RunInputs
) -> tuple[StructureNode, str | None]:
    try:
        try:
            traj = run_inputs.engine.compute_geometry_optimization(
                node, keywords={"coordsys": "cart", "maxiter": 500}
            )
        except TypeError:
            traj = run_inputs.engine.compute_geometry_optimization(node)
        optimized = traj[-1]
        if node.has_molecular_graph:
            optimized.graph = structure_to_molecule(optimized.structure)
            optimized.has_molecular_graph = True
        else:
            optimized.graph = node.graph
            optimized.has_molecular_graph = node.has_molecular_graph
        return optimized, None
    except Exception as exc:
        fallback = node.copy()
        fallback._cached_energy = None
        fallback._cached_gradient = None
        fallback._cached_result = None
        return fallback, f"{type(exc).__name__}: {exc}"


def _optimize_endpoint_batch(
    nodes: list[StructureNode], run_inputs: RunInputs
) -> list[tuple[StructureNode, str | None]]:
    if not nodes:
        return []

    batch_optimizer = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    if callable(batch_optimizer):
        try:
            try:
                trajectories = batch_optimizer(
                    nodes, keywords={"coordsys": "cart", "maxiter": 500}
                )
            except TypeError:
                trajectories = batch_optimizer(nodes)

            optimized_nodes: list[tuple[StructureNode, str | None]] = []
            for original, traj in zip(nodes, trajectories):
                optimized = traj[-1]
                if original.has_molecular_graph:
                    optimized.graph = structure_to_molecule(optimized.structure)
                    optimized.has_molecular_graph = True
                else:
                    optimized.graph = original.graph
                    optimized.has_molecular_graph = original.has_molecular_graph
                optimized_nodes.append((optimized, None))
            if len(optimized_nodes) == len(nodes):
                return optimized_nodes
        except Exception:
            pass

    return [
        _optimize_single_endpoint(node=node, run_inputs=run_inputs)
        for node in nodes
    ]


def prepare_optimized_neb_pot(workspace: RetropathsWorkspace, run_inputs: RunInputs) -> Pot:
    pot = Pot.read_from_disk(workspace.neb_pot_fp)
    return prepare_optimized_neb_pot_from_pot(pot=pot, workspace=workspace, run_inputs=run_inputs)


def prepare_optimized_neb_pot_from_pot(
    pot: Pot,
    workspace: RetropathsWorkspace,
    run_inputs: RunInputs,
    progress: Callable[[str], None] | None = None,
) -> Pot:
    if progress is None:
        progress = lambda _msg: None

    pending_indices: list[int] = []
    pending_nodes: list[StructureNode] = []
    for node_index in pot.graph.nodes:
        node_attrs = pot.graph.nodes[node_index]
        if node_attrs.get("endpoint_optimized"):
            continue
        td = node_attrs.get("td")
        if td is None:
            continue
        pending_indices.append(node_index)
        pending_nodes.append(td)

    if not pending_nodes:
        progress("All endpoint structures are already optimized.")
        pot.write_to_disk(workspace.neb_pot_fp)
        return pot

    progress(f"Optimizing {len(pending_nodes)} endpoint structures before queueing NEBs.")
    batch_optimizer = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    if callable(batch_optimizer):
        progress(
            f"Submitting {len(pending_nodes)} endpoint geometry optimizations as one batch."
        )
    else:
        progress(
            f"Batch endpoint optimization unavailable; running {len(pending_nodes)} optimizations sequentially."
        )

    for node_index, (optimized_td, error) in zip(
        pending_indices, _optimize_endpoint_batch(nodes=pending_nodes, run_inputs=run_inputs)
    ):
        node_attrs = pot.graph.nodes[node_index]
        result_fp = None
        if error is None:
            result_fp = _persist_endpoint_optimization_result(
                workspace=workspace,
                node_index=node_index,
                optimized_td=optimized_td,
            )
            if result_fp is not None:
                optimized_td = _strip_cached_result(optimized_td)
        node_attrs["td"] = optimized_td
        node_attrs["endpoint_optimized"] = error is None
        if error is None:
            node_attrs.pop("endpoint_optimization_error", None)
            if result_fp is not None:
                node_attrs["endpoint_optimization_result_fp"] = result_fp
        else:
            node_attrs["endpoint_optimization_error"] = error
            node_attrs.pop("endpoint_optimization_result_fp", None)

    succeeded = sum(
        bool(pot.graph.nodes[node_index].get("endpoint_optimized"))
        for node_index in pending_indices
    )
    failed = len(pending_indices) - succeeded
    progress(
        f"Endpoint optimization finished: {succeeded} succeeded, {failed} failed."
    )
    pot.write_to_disk(workspace.neb_pot_fp)
    return pot


def load_partial_annotated_pot(workspace: RetropathsWorkspace) -> Pot:
    source_pot = Pot.read_from_disk(workspace.neb_pot_fp)
    queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)

    pot = Pot(
        root=source_pot.root.copy(),
        target=source_pot.target.copy(),
        multiplier=source_pot.multiplier,
        rxn_name=source_pot.rxn_name,
    )
    pot.graph = source_pot.graph.copy()
    pot.run_time = source_pot.run_time

    run_inputs = RunInputs.open(workspace.inputs_fp) if Path(workspace.inputs_fp).exists() else RunInputs()
    chain_inputs = run_inputs.chain_inputs
    collapse_rms_thre = float(run_inputs.network_inputs.collapse_node_rms_thre)
    collapse_ene_thre = float(run_inputs.network_inputs.collapse_node_ene_thre)
    chains_by_edge: dict[tuple[int, int], list[Chain]] = {}

    for item in queue.items:
        if item.status != "completed" or not item.result_dir:
            continue

        result_dir = Path(item.result_dir)
        if not result_dir.exists() or not (result_dir / "adj_matrix.txt").exists():
            continue

        history = TreeNode.read_from_disk(
            folder_name=result_dir,
            charge=0,
            multiplicity=1,
        )
        leaf_chains = _history_leaf_chains(history)
        if not leaf_chains:
            continue

        source_attrs = source_pot.graph.nodes.get(item.source_node, {})
        target_attrs = source_pot.graph.nodes.get(item.target_node, {})
        source_td = source_attrs.get("td")
        target_td = target_attrs.get("td")
        source_graph = source_attrs.get("molecule")
        target_graph = target_attrs.get("molecule")
        candidates = [
            (item.source_node, source_td, source_graph),
            (item.target_node, target_td, target_graph),
        ]

        total_leaves = len(leaf_chains)
        for leaf_index, leaf_chain in enumerate(leaf_chains, start=1):
            start_node, start_index_hint = _normalize_history_endpoint(
                leaf_chain[0], candidates, chain_inputs
            )
            end_node, end_index_hint = _normalize_history_endpoint(
                leaf_chain[-1], candidates, chain_inputs
            )
            start_index = _register_network_node(
                pot=pot,
                source_pot=source_pot,
                node=start_node,
                chain_inputs=chain_inputs,
                collapse_rms_thre=collapse_rms_thre,
                collapse_ene_thre=collapse_ene_thre,
                preferred_index=start_index_hint,
            )
            end_index = _register_network_node(
                pot=pot,
                source_pot=source_pot,
                node=end_node,
                chain_inputs=chain_inputs,
                collapse_rms_thre=collapse_rms_thre,
                collapse_ene_thre=collapse_ene_thre,
                preferred_index=end_index_hint,
            )

            chain_copy = leaf_chain.copy()
            chain_copy.nodes[0] = start_node
            chain_copy.nodes[-1] = end_node
            chains_by_edge.setdefault((start_index, end_index), []).append(chain_copy)

            if not pot.graph.has_edge(start_index, end_index):
                edge_attrs = {}
                if source_pot.graph.has_edge(item.source_node, item.target_node):
                    edge_attrs = dict(source_pot.graph.edges[(item.source_node, item.target_node)])
                edge_attrs["reaction"] = _elementary_step_label(
                    base_reaction=item.reaction,
                    leaf_index=leaf_index,
                    total_leaves=total_leaves,
                )
                pot.graph.add_edge(start_index, end_index, **edge_attrs)

    from neb_dynamics.retropaths_compat import annotate_pot_with_neb_results
    annotate_pot_with_neb_results(pot=pot, chains_by_edge=chains_by_edge)
    pot.write_to_disk(workspace.annotated_neb_pot_fp)
    return pot


def _kmc_summary_rows(final_populations: dict[int, float], labels: dict[int, str]) -> str:
    rows = []
    for node_index, value in sorted(
        final_populations.items(),
        key=lambda item: (-float(item[1]), int(item[0])),
    ):
        rows.append(
            "<tr>"
            f"<td>{node_index}</td>"
            f"<td>{labels.get(int(node_index), str(node_index))}</td>"
            f"<td>{float(value):.6f}</td>"
            "</tr>"
        )
    return "".join(rows) or "<tr><td colspan='3'>No data</td></tr>"


def _kmc_suppressed_rows(items: list[dict[str, str | float]]) -> str:
    if not items:
        return "<tr><td colspan='5'>None</td></tr>"
    rows = []
    for item in items:
        rows.append(
            "<tr>"
            f"<td>{item['source']} -&gt; {item['target']}</td>"
            f"<td>{item['reaction']}</td>"
            f"<td>{float(item['barrier']):.6f}</td>"
            f"<td>{item['reason']}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_status_html(
    workspace: RetropathsWorkspace,
    retropaths_pot,
    pot: Pot,
    queue: RetropathsNEBQueue,
    optimized_nodes: int,
    prepared_edge_count: int,
    edge_visualizations: list[dict[str, str]],
    kmc_temperature_kelvin: float = 298.15,
    kmc_initial_conditions: dict[int, float] | None = None,
) -> str:
    counts = summarize_queue(queue)
    running_items = [item for item in queue.items if item.status == "running"][:5]
    failed_items = [item for item in queue.items if item.status == "failed"][:10]
    completed_items = [item for item in queue.items if item.status == "completed"][:10]
    incompatible_items = [item for item in queue.items if item.status == "incompatible"][:10]
    retropaths_html = retropaths_pot.draw(string_mode=True, leaves=False)
    network_html = pot.draw(string_mode=True, leaves=False)
    normalized_initial_conditions = normalize_initial_conditions(
        pot=pot,
        initial_conditions=kmc_initial_conditions,
    )
    kmc_payload = build_kmc_payload(
        pot=pot,
        temperature_kelvin=kmc_temperature_kelvin,
        initial_conditions=normalized_initial_conditions,
    )
    kmc_default_result = simulate_kmc(
        pot=pot,
        temperature_kelvin=kmc_temperature_kelvin,
        initial_conditions=normalized_initial_conditions,
    )
    kmc_labels = {
        int(node["id"]): str(node["label"])
        for node in kmc_payload["nodes"]
    }
    template_payloads = _load_template_payloads(workspace)
    retropaths_explorer_payload = _build_network_explorer_payload(
        retropaths_pot.graph,
        template_payloads=template_payloads,
    )
    neb_explorer_payload = _build_network_explorer_payload(
        pot.graph,
        template_payloads=template_payloads,
        edge_visualizations=edge_visualizations,
    )
    kmc_payload_json = json.dumps(kmc_payload)
    kmc_default_result_json = json.dumps(kmc_default_result)
    retropaths_explorer_payload_json = json.dumps(retropaths_explorer_payload)
    neb_explorer_payload_json = json.dumps(neb_explorer_payload)
    kmc_initial_table_rows = "".join(
        "<tr>"
        f"<td>{int(node['id'])}</td>"
        f"<td>{node['label']}</td>"
        f"<td><input type=\"number\" step=\"0.01\" data-node-id=\"{int(node['id'])}\" value=\"{float(normalized_initial_conditions.get(int(node['id']), 0.0)):.6f}\" /></td>"
        "</tr>"
        for node in kmc_payload["nodes"]
    ) or "<tr><td colspan='3'>No completed NEB nodes available.</td></tr>"
    kmc_default_summary_rows = _kmc_summary_rows(
        final_populations={
            int(node_index): float(value)
            for node_index, value in kmc_default_result["final_populations"].items()
        },
        labels=kmc_labels,
    )
    kmc_suppressed_rows = _kmc_suppressed_rows(kmc_payload.get("suppressed_edges", []))

    def _rows(items):
        if not items:
            return "<tr><td colspan='5'>None</td></tr>"
        rows = []
        for item in items:
            rows.append(
                "<tr>"
                f"<td>{item.job_id}</td>"
                f"<td>{item.status}</td>"
                f"<td>{item.reaction or ''}</td>"
                f"<td>{item.started_at or ''}</td>"
                f"<td>{item.error or ''}</td>"
                "</tr>"
            )
        return "".join(rows)

    def _edge_rows(items: list[dict[str, str]]) -> str:
        if not items:
            return "<tr><td colspan='7'>None</td></tr>"
        rows = []
        for item in items:
            rows.append(
                "<tr>"
                f"<td>{item['edge']}</td>"
                f"<td>{item['start']}</td>"
                f"<td>{item['end']}</td>"
                f"<td>{item['reaction']}</td>"
                f"<td>{item['barrier']}</td>"
                f"<td>{item['chains']}</td>"
                f"<td><a href=\"edge_visualizations/{item['href']}\">Open Viewer</a></td>"
                "</tr>"
            )
        return "".join(rows)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{workspace.run_name} Status</title>
  <script src="https://d3js.org/d3.v3.min.js"></script>
  <style>
    :root {{
      --bg: #08111f;
      --panel: rgba(13, 24, 41, 0.9);
      --panel-soft: rgba(12, 23, 39, 0.78);
      --line: rgba(114, 146, 197, 0.2);
      --line-strong: rgba(120, 170, 233, 0.38);
      --ink: #eef4ff;
      --ink-soft: #c9d8f0;
      --muted: #8ea3c2;
      --accent: #63d5ff;
      --accent-2: #7ef0c7;
      --backed: #59d8b6;
      --target: #ff8eb0;
      --warn: #ff7a8f;
      --radius-lg: 22px;
      --radius-md: 16px;
      --radius-sm: 12px;
      --shadow: 0 18px 44px rgba(2, 8, 18, 0.45);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font: 500 14px/1.5 "IBM Plex Sans", "Aptos", "Segoe UI Variable", "Avenir Next", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(93, 213, 255, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(126, 240, 199, 0.08), transparent 24%),
        linear-gradient(180deg, #0a1425 0%, #08111f 100%);
      color: var(--ink);
    }}
    a {{ color: var(--accent); }}
    .hero {{
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, rgba(17, 32, 54, 0.94), rgba(11, 22, 38, 0.9));
      margin-bottom: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card {{
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, rgba(16, 30, 50, 0.92), rgba(10, 20, 34, 0.84));
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      background: linear-gradient(180deg, rgba(16, 30, 50, 0.92), rgba(10, 20, 34, 0.84));
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      overflow: hidden;
    }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; vertical-align: top; text-align: left; }}
    td + td, th + th {{ border-left: 1px solid var(--line); }}
    tr:last-child td {{ border-bottom: none; }}
    th {{ background: rgba(99, 213, 255, 0.08); color: var(--ink-soft); }}
    h1, h2 {{ margin: 0 0 10px 0; letter-spacing: -0.02em; }}
    h3 {{ letter-spacing: -0.01em; }}
    .section {{ margin-top: 24px; }}
    code {{
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 2px 6px;
      color: var(--ink-soft);
    }}
    input, button {{ font: inherit; }}
    input {{
      min-height: 38px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: rgba(8, 16, 28, 0.74);
      color: var(--ink);
    }}
    button {{
      min-height: 38px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--ink);
      cursor: pointer;
    }}
    .tab-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .tab-button {{
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--muted);
      padding: 8px 12px;
      cursor: pointer;
    }}
    .tab-button.active {{ background: rgba(99, 213, 255, 0.12); color: var(--ink); border-color: rgba(99, 213, 255, 0.34); }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .explorer-layout {{ display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(340px, 0.95fr); gap: 16px; }}
    .explorer-card {{
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, rgba(16, 30, 50, 0.92), rgba(10, 20, 34, 0.84));
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    .explorer-svg {{
      width: 100%;
      height: auto;
      min-height: 420px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background:
        radial-gradient(circle at 20% 18%, rgba(99, 213, 255, 0.11), transparent 20%),
        radial-gradient(circle at 82% 12%, rgba(126, 240, 199, 0.08), transparent 18%),
        linear-gradient(180deg, #0d1728 0%, #08111f 100%);
    }}
    .info-tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }}
    .info-tab-button {{
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.03);
      color: var(--muted);
      padding: 6px 10px;
      cursor: pointer;
    }}
    .info-tab-button.active {{ background: rgba(99, 213, 255, 0.12); color: var(--ink); border-color: rgba(99, 213, 255, 0.34); }}
    .info-tab-panel {{ display: none; }}
    .info-tab-panel.active {{ display: block; }}
    .placeholder {{ color: var(--muted); font-style: italic; }}
    .network-node {{
      fill: #7d94bb;
      stroke: rgba(238, 244, 255, 0.85);
      stroke-width: 1.8;
      cursor: pointer;
      filter: drop-shadow(0 8px 14px rgba(4, 9, 18, 0.3));
    }}
    .network-node.root {{ fill: var(--accent); }}
    .network-node.selected {{ fill: #ffd166; stroke: #fff7de; stroke-width: 3; }}
    .network-edge-hitbox {{ stroke: transparent; stroke-width: 16; cursor: pointer; fill: none; }}
    .network-edge-line {{ stroke: rgba(128, 154, 194, 0.42); stroke-width: 2.5; fill: none; }}
    .network-edge-line.selected {{ stroke: var(--accent); stroke-width: 4; }}
    .network-label {{ font-size: 12px; fill: rgba(233, 241, 255, 0.94); pointer-events: none; }}
    .template-visualization svg {{ max-width: 100%; height: auto; }}
    pre.json-block {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(4, 10, 18, 0.72);
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: 12px;
      color: var(--ink-soft);
    }}
    @media (max-width: 960px) {{
      .explorer-layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>{workspace.run_name}</h1>
    <div><strong>Root:</strong> <code>{workspace.root_smiles}</code></div>
    <div><strong>Environment:</strong> <code>{workspace.environment_smiles or '(none)'}</code></div>
    <div><strong>Inputs:</strong> <code>{workspace.inputs_fp}</code></div>
    <div><strong>Reactions File:</strong> <code>{workspace.reactions_path}</code></div>
    <div><strong>Workspace:</strong> <code>{workspace.workdir}</code></div>
  </div>

  <div class="grid">
    <div class="card"><strong>Queue Items</strong><br />{counts['items']}</div>
    <div class="card"><strong>Pending</strong><br />{counts.get('pending', 0)}</div>
    <div class="card"><strong>Running</strong><br />{counts.get('running', 0)}</div>
    <div class="card"><strong>Completed</strong><br />{counts.get('completed', 0)}</div>
    <div class="card"><strong>Failed</strong><br />{counts.get('failed', 0)}</div>
    <div class="card"><strong>Incompatible</strong><br />{counts.get('incompatible', 0)}</div>
    <div class="card"><strong>Network Nodes</strong><br />{pot.graph.number_of_nodes()}</div>
    <div class="card"><strong>Network Edges</strong><br />{pot.graph.number_of_edges()}</div>
    <div class="card"><strong>Optimized Endpoints</strong><br />{optimized_nodes}</div>
    <div class="card"><strong>Edges With NEBs</strong><br />{sum(bool(pot.graph.edges[e].get('list_of_nebs')) for e in pot.graph.edges)}</div>
    <div class="card"><strong>Prepared Queue Edges</strong><br />{prepared_edge_count}</div>
  </div>

  <div class="section">
    <h2>Retropaths Pot</h2>
    {retropaths_html}
  </div>

  <div class="section">
    <h2>NEB Pot</h2>
    {network_html}
  </div>

  <div class="section">
    <h2>Graph Delta</h2>
    <div class="card">
      <strong>Retropaths nodes:</strong> {retropaths_pot.graph.number_of_nodes()}<br />
      <strong>Retropaths edges:</strong> {retropaths_pot.graph.number_of_edges()}<br />
      <strong>NEB nodes:</strong> {pot.graph.number_of_nodes()}<br />
      <strong>NEB edges:</strong> {pot.graph.number_of_edges()}<br />
      <strong>NEB edges with chains:</strong> {sum(bool(pot.graph.edges[e].get('list_of_nebs')) for e in pot.graph.edges)}<br />
    </div>
  </div>

  <div class="section">
    <h2>Network Explorer</h2>
    <div class="tab-row" data-tab-group="network-explorer">
      <button class="tab-button active" data-tab-target="retropaths-network">Retropaths Network</button>
      <button class="tab-button" data-tab-target="neb-network">NEB Network</button>
    </div>

    <div id="retropaths-network" class="tab-panel active">
      <div class="explorer-layout">
        <div class="explorer-card">
          <h3 style="margin-top: 0;">Retropaths Reaction Network</h3>
          <svg id="retropaths-network-svg" class="explorer-svg" viewBox="0 0 960 560" role="img" aria-label="Retropaths network graph"></svg>
        </div>
        <div class="explorer-card">
          <h3 id="retropaths-network-title" style="margin-top: 0;">Select an edge or node</h3>
          <div id="retropaths-network-summary" class="placeholder">Click a network edge to inspect the targeted reaction and template details.</div>
          <div class="info-tabs" data-info-tabs="retropaths-network">
            <button class="info-tab-button active" data-info-target="retropaths-network-targeted">Targeted Reaction</button>
            <button class="info-tab-button" data-info-target="retropaths-network-template-data">Template Data</button>
            <button class="info-tab-button" data-info-target="retropaths-network-template-visualization">Template Visualization</button>
          </div>
          <div id="retropaths-network-targeted" class="info-tab-panel active"></div>
          <div id="retropaths-network-template-data" class="info-tab-panel"></div>
          <div id="retropaths-network-template-visualization" class="info-tab-panel template-visualization"></div>
        </div>
      </div>
    </div>

    <div id="neb-network" class="tab-panel">
      <div class="explorer-layout">
        <div class="explorer-card">
          <h3 style="margin-top: 0;">NEB Reaction Network</h3>
          <svg id="neb-network-svg" class="explorer-svg" viewBox="0 0 960 560" role="img" aria-label="NEB network graph"></svg>
        </div>
        <div class="explorer-card">
          <h3 id="neb-network-title" style="margin-top: 0;">Select an edge or node</h3>
          <div id="neb-network-summary" class="placeholder">Click an annotated NEB edge to inspect barrier, chain count, and any linked viewer.</div>
          <div class="info-tabs" data-info-tabs="neb-network">
            <button class="info-tab-button active" data-info-target="neb-network-targeted">Targeted Reaction</button>
            <button class="info-tab-button" data-info-target="neb-network-template-data">Template Data</button>
            <button class="info-tab-button" data-info-target="neb-network-template-visualization">Template Visualization</button>
          </div>
          <div id="neb-network-targeted" class="info-tab-panel active"></div>
          <div id="neb-network-template-data" class="info-tab-panel"></div>
          <div id="neb-network-template-visualization" class="info-tab-panel template-visualization"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Running</h2>
    <table>
      <tr><th>Job</th><th>Status</th><th>Reaction</th><th>Started</th><th>Error</th></tr>
      {_rows(running_items)}
    </table>
  </div>

  <div class="section">
    <h2>Completed</h2>
    <table>
      <tr><th>Job</th><th>Status</th><th>Reaction</th><th>Started</th><th>Error</th></tr>
      {_rows(completed_items)}
    </table>
  </div>

  <div class="section">
    <h2>Completed NEB Edges</h2>
    <table>
      <tr><th>Edge</th><th>Start</th><th>End</th><th>Reaction</th><th>Barrier</th><th>Chains</th><th>Viewer</th></tr>
      {_edge_rows(edge_visualizations)}
    </table>
  </div>

  <div class="section">
    <h2>Kinetic Monte Carlo</h2>
    <div class="card">
      <div style="margin-bottom: 12px;">
        <label><strong>Temperature (K)</strong>
          <input id="kmc-temperature" type="number" step="0.1" value="{float(kmc_temperature_kelvin):.2f}" style="margin-left: 8px;" />
        </label>
        <label style="margin-left: 16px;"><strong>End Time</strong>
          <input id="kmc-end-time" type="number" step="any" min="0" value="{float(kmc_payload['default_end_time']):.6g}" style="margin-left: 8px;" />
        </label>
        <label style="margin-left: 16px;"><strong>Time Points</strong>
          <input id="kmc-max-steps" type="number" step="1" min="1" value="{int(kmc_default_result['max_steps'])}" style="margin-left: 8px;" />
        </label>
        <button id="kmc-run" style="margin-left: 16px;">Run KMC</button>
        <button id="kmc-reset" style="margin-left: 8px;">Reset Defaults</button>
      </div>
      <div style="margin-bottom: 12px;">
        Default initial conditions are all nodes at 0.0 M and node 0 at 1.0 M.
      </div>
      <table>
        <tr><th>Node</th><th>Label</th><th>Initial Concentration (M)</th></tr>
        {kmc_initial_table_rows}
      </table>
    </div>
    <div class="grid" style="margin-top: 12px;">
      <div class="card"><strong>Time Points</strong><br /><span id="kmc-events">{max(len(kmc_default_result["history"]) - 1, 0)}</span></div>
      <div class="card"><strong>Final Time (a.u.)</strong><br /><span id="kmc-final-time">{float(kmc_default_result["history"][-1]["time"]):.6f}</span></div>
      <div class="card"><strong>Dominant Node</strong><br /><span id="kmc-dominant">{kmc_labels.get(int(max(kmc_default_result["final_populations"], key=kmc_default_result["final_populations"].get)) if kmc_default_result["final_populations"] else 0, "n/a")}</span></div>
    </div>
    <div class="card" style="margin-top: 12px;">
      <h3 style="margin-top: 0;">Final Populations</h3>
      <table>
        <tr><th>Node</th><th>Label</th><th>Final Population (M)</th></tr>
        <tbody id="kmc-summary-body">{kmc_default_summary_rows}</tbody>
      </table>
    </div>
    <div class="card" style="margin-top: 12px;">
      <h3 style="margin-top: 0;">Trajectories</h3>
      <svg id="kmc-plot" width="960" height="360" viewBox="0 0 960 360" style="width: 100%; height: auto; border: 1px solid var(--line); border-radius: 16px; background: linear-gradient(180deg, #101b2b 0%, #0a1321 100%);"></svg>
    </div>
    <div class="card" style="margin-top: 12px;">
      <h3 style="margin-top: 0;">Suppressed KMC Edges</h3>
      <div style="margin-bottom: 8px;">Edges are excluded from kinetics when the directed chain starts at the highest-energy endpoint and therefore produces an artificial zero barrier.</div>
      <table>
        <tr><th>Edge</th><th>Reaction</th><th>Barrier</th><th>Reason</th></tr>
        {kmc_suppressed_rows}
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Failed</h2>
    <table>
      <tr><th>Job</th><th>Status</th><th>Reaction</th><th>Started</th><th>Error</th></tr>
      {_rows(failed_items)}
    </table>
  </div>

  <div class="section">
    <h2>Incompatible</h2>
    <table>
      <tr><th>Job</th><th>Status</th><th>Reaction</th><th>Started</th><th>Error</th></tr>
      {_rows(incompatible_items)}
    </table>
  </div>
  <script id="kmc-payload" type="application/json">{kmc_payload_json}</script>
  <script id="kmc-default-result" type="application/json">{kmc_default_result_json}</script>
  <script id="retropaths-network-payload" type="application/json">{retropaths_explorer_payload_json}</script>
  <script id="neb-network-payload" type="application/json">{neb_explorer_payload_json}</script>
  <script>
    (function() {{
      function escapeHtml(value) {{
        return String(value ?? "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }}

      document.querySelectorAll("[data-tab-group]").forEach((group) => {{
        const buttons = Array.from(group.querySelectorAll("[data-tab-target]"));
        buttons.forEach((button) => {{
          button.addEventListener("click", () => {{
            const targetId = button.getAttribute("data-tab-target");
            buttons.forEach((elem) => {{
              elem.classList.toggle("active", elem === button);
              const panelId = elem.getAttribute("data-tab-target");
              const panel = panelId ? document.getElementById(panelId) : null;
              if (panel) {{
                panel.classList.toggle("active", panelId === targetId);
              }}
            }});
          }});
        }});
      }});

      document.querySelectorAll("[data-info-tabs]").forEach((group) => {{
        const buttons = Array.from(group.querySelectorAll("[data-info-target]"));
        buttons.forEach((button) => {{
          button.addEventListener("click", () => {{
            const targetId = button.getAttribute("data-info-target");
            buttons.forEach((elem) => {{
              elem.classList.toggle("active", elem === button);
              const panelId = elem.getAttribute("data-info-target");
              const panel = panelId ? document.getElementById(panelId) : null;
              if (panel) {{
                panel.classList.toggle("active", panelId === targetId);
              }}
            }});
          }});
        }});
      }});

      function renderNetworkExplorer(prefix, payload) {{
        const svg = document.getElementById(`${{prefix}}-svg`);
        const titleEl = document.getElementById(`${{prefix}}-title`);
        const summaryEl = document.getElementById(`${{prefix}}-summary`);
        const targetedEl = document.getElementById(`${{prefix}}-targeted`);
        const templateDataEl = document.getElementById(`${{prefix}}-template-data`);
        const templateVisualizationEl = document.getElementById(`${{prefix}}-template-visualization`);
        if (!svg) return;

        const nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
        const edges = Array.isArray(payload.edges) ? payload.edges : [];
        if (!nodes.length) {{
          summaryEl.innerHTML = '<div class="placeholder">No network data available.</div>';
          targetedEl.innerHTML = "";
          templateDataEl.innerHTML = "";
          templateVisualizationEl.innerHTML = "";
          return;
        }}

        while (svg.firstChild) svg.removeChild(svg.firstChild);
        const width = 960;
        const height = 560;
        const nodeElems = new Map();
        const edgeElems = [];

        function makeSvg(tag) {{
          return document.createElementNS("http://www.w3.org/2000/svg", tag);
        }}

        function setInfo(selection) {{
          edgeElems.forEach((item) => item.line.classList.toggle("selected", item.edge === selection.edge));
          nodeElems.forEach((elem, nodeId) => {{
            const selectedNodeId = selection.node ? Number(selection.node.id) : null;
            elem.classList.toggle("selected", selectedNodeId === Number(nodeId));
          }});

          if (selection.edge) {{
            const edge = selection.edge;
            const template = edge.template || {{}};
            titleEl.textContent = `Edge ${{edge.source}} -> ${{edge.target}}`;
            summaryEl.innerHTML = `
              <div><strong>Reaction:</strong> ${{escapeHtml(edge.reaction || "Unknown")}}</div>
              <div><strong>Barrier:</strong> ${{edge.barrier == null ? "n/a" : escapeHtml(edge.barrier.toFixed(3))}}</div>
              <div><strong>Chains:</strong> ${{escapeHtml(edge.chains ?? 0)}}</div>
              <div><strong>Targeted Reaction:</strong> ${{escapeHtml(edge.reaction || "Unknown")}}</div>
              ${{edge.viewer_href ? `<div><strong>Viewer:</strong> <a href="${{escapeHtml(edge.viewer_href)}}">Open edge viewer</a></div>` : ""}}
            `;
            targetedEl.innerHTML = `
              <div><strong>Targeted reaction:</strong> ${{escapeHtml(edge.reaction || "Unknown")}}</div>
              <div style="margin-top: 8px;"><strong>Raw edge data:</strong></div>
              <pre class="json-block">${{escapeHtml(JSON.stringify(edge.data || {{}}, null, 2))}}</pre>
            `;
            templateDataEl.innerHTML = template.data
              ? `<pre class="json-block">${{escapeHtml(JSON.stringify(template.data, null, 2))}}</pre>`
              : '<div class="placeholder">No template data was available for this reaction.</div>';
            templateVisualizationEl.innerHTML = template.visualization_html
              ? template.visualization_html
              : '<div class="placeholder">No template visualization was available for this reaction.</div>';
            return;
        }}

          if (selection.node) {{
            const node = selection.node;
            titleEl.textContent = `Node ${{node.id}}`;
            summaryEl.innerHTML = `
              <div><strong>Label:</strong> ${{escapeHtml(node.label || String(node.id))}}</div>
              <div><strong>Node:</strong> ${{escapeHtml(node.id)}}</div>
            `;
            targetedEl.innerHTML = '<div class="placeholder">Select an edge to inspect the targeted reaction.</div>';
            templateDataEl.innerHTML = `<pre class="json-block">${{escapeHtml(JSON.stringify(node.data || {{}}, null, 2))}}</pre>`;
            templateVisualizationEl.innerHTML = '<div class="placeholder">Template visualization is only available for edge selections.</div>';
          }}
        }}

        edges.forEach((edge) => {{
          const group = makeSvg("g");
          const hitbox = makeSvg("line");
          hitbox.setAttribute("class", "network-edge-hitbox");
          const line = makeSvg("line");
          line.setAttribute("class", "network-edge-line");
          group.appendChild(hitbox);
          group.appendChild(line);
          group.addEventListener("click", () => setInfo({{ edge }}));
          svg.appendChild(group);
          edgeElems.push({{ edge, line, hitbox }});
        }});

        const simulationNodes = nodes.map((node, index) => ({{
          id: Number(node.id),
          node,
          x: width / 2 + ((index % 5) - 2) * 24,
          y: height / 2 + (Math.floor(index / 5) - 2) * 24,
          fixed: false,
        }}));
        const simulationNodeById = new Map(simulationNodes.map((node) => [node.id, node]));

        nodes.forEach((node) => {{
          const simNode = simulationNodeById.get(Number(node.id));
          if (!simNode) return;
          const group = makeSvg("g");
          group.style.cursor = "pointer";
          const circle = makeSvg("circle");
          circle.setAttribute("r", "18");
          circle.setAttribute("class", `network-node${{Number(node.id) === 0 ? " root" : ""}}`);
          const text = makeSvg("text");
          text.setAttribute("y", "36");
          text.setAttribute("text-anchor", "middle");
          text.setAttribute("class", "network-label");
          text.textContent = String(node.id);
          group.appendChild(circle);
          group.appendChild(text);
          group.addEventListener("click", () => setInfo({{ node }}));
          svg.appendChild(group);
          nodeElems.set(Number(node.id), circle);
          simNode.group = group;
        }});

        if (window.d3 && typeof d3.layout?.force === "function") {{
          const force = d3.layout.force()
            .size([width, height])
            .nodes(simulationNodes)
            .links(edges.map((edge) => ({{
              source: simulationNodeById.get(Number(edge.source)),
              target: simulationNodeById.get(Number(edge.target)),
              edge,
            }})))
            .charge(Math.max(-900, -140 - nodes.length * 12))
            .linkDistance(Math.max(90, Math.min(220, 110 + nodes.length * 1.5)))
            .gravity(0.06)
            .friction(0.82)
            .on("tick", () => {{
              edgeElems.forEach((item) => {{
                const source = simulationNodeById.get(Number(item.edge.source));
                const target = simulationNodeById.get(Number(item.edge.target));
                if (!source || !target) return;
                item.line.setAttribute("x1", String(source.x));
                item.line.setAttribute("y1", String(source.y));
                item.line.setAttribute("x2", String(target.x));
                item.line.setAttribute("y2", String(target.y));
                item.hitbox.setAttribute("x1", String(source.x));
                item.hitbox.setAttribute("y1", String(source.y));
                item.hitbox.setAttribute("x2", String(target.x));
                item.hitbox.setAttribute("y2", String(target.y));
              }});
              simulationNodes.forEach((simNode) => {{
                simNode.x = Math.max(28, Math.min(width - 28, simNode.x));
                simNode.y = Math.max(28, Math.min(height - 42, simNode.y));
                if (simNode.group) {{
                  simNode.group.setAttribute("transform", `translate(${{simNode.x}},${{simNode.y}})`);
                }}
              }});
            }});
          force.start();
          for (let i = 0; i < Math.max(80, nodes.length * 6); i += 1) {{
            force.tick();
          }}
          force.stop();
        }} else {{
          simulationNodes.forEach((simNode, index) => {{
            const cols = Math.max(1, Math.ceil(Math.sqrt(nodes.length)));
            const row = Math.floor(index / cols);
            const col = index % cols;
            const x = 80 + col * Math.max(110, Math.floor((width - 160) / cols));
            const y = 80 + row * 100;
            if (simNode.group) {{
              simNode.group.setAttribute("transform", `translate(${{x}},${{y}})`);
            }}
          }});
          edgeElems.forEach((item) => {{
            const source = simulationNodeById.get(Number(item.edge.source));
            const target = simulationNodeById.get(Number(item.edge.target));
            if (!source || !target || !source.group || !target.group) return;
            const sourceTransform = source.group.getAttribute("transform") || "translate(0,0)";
            const targetTransform = target.group.getAttribute("transform") || "translate(0,0)";
            const sourceMatch = /translate\\(([^,]+),([^\\)]+)\\)/.exec(sourceTransform);
            const targetMatch = /translate\\(([^,]+),([^\\)]+)\\)/.exec(targetTransform);
            const sx = sourceMatch ? Number(sourceMatch[1]) : 0;
            const sy = sourceMatch ? Number(sourceMatch[2]) : 0;
            const tx = targetMatch ? Number(targetMatch[1]) : 0;
            const ty = targetMatch ? Number(targetMatch[2]) : 0;
            item.line.setAttribute("x1", String(sx));
            item.line.setAttribute("y1", String(sy));
            item.line.setAttribute("x2", String(tx));
            item.line.setAttribute("y2", String(ty));
            item.hitbox.setAttribute("x1", String(sx));
            item.hitbox.setAttribute("y1", String(sy));
            item.hitbox.setAttribute("x2", String(tx));
            item.hitbox.setAttribute("y2", String(ty));
          }});
        }}

        if (edges.length) {{
          setInfo({{ edge: edges[0] }});
        }} else {{
          setInfo({{ node: nodes[0] }});
        }}
      }}

      const payload = JSON.parse(document.getElementById("kmc-payload").textContent);
      const defaultResult = JSON.parse(document.getElementById("kmc-default-result").textContent);
      const retropathsNetworkPayload = JSON.parse(document.getElementById("retropaths-network-payload").textContent);
      const nebNetworkPayload = JSON.parse(document.getElementById("neb-network-payload").textContent);
      const nodeLabels = new Map(payload.nodes.map(node => [Number(node.id), String(node.label)]));
      const temperatureInput = document.getElementById("kmc-temperature");
      const endTimeInput = document.getElementById("kmc-end-time");
      const maxStepsInput = document.getElementById("kmc-max-steps");
      const runButton = document.getElementById("kmc-run");
      const resetButton = document.getElementById("kmc-reset");
      const summaryBody = document.getElementById("kmc-summary-body");
      const eventsEl = document.getElementById("kmc-events");
      const finalTimeEl = document.getElementById("kmc-final-time");
      const dominantEl = document.getElementById("kmc-dominant");
      const svg = document.getElementById("kmc-plot");

      function eyringRate(barrier, temperature) {{
        const kB = 1.380649e-23;
        const h = 6.62607015e-34;
        const r = 0.00198720425864083;
        return (kB * temperature / h) * Math.exp(-barrier / (r * temperature));
      }}

      function readInitialConditions() {{
        const values = {{}};
        document.querySelectorAll("[data-node-id]").forEach((input) => {{
          values[Number(input.getAttribute("data-node-id"))] = Number(input.value || 0);
        }});
        return values;
      }}

      function resetDefaults() {{
        payload.nodes.forEach((node) => {{
          const input = document.querySelector(`[data-node-id="${{node.id}}"]`);
          if (input) {{
            input.value = Number(node.initial).toFixed(6);
          }}
        }});
        temperatureInput.value = Number(payload.temperature_kelvin).toFixed(2);
        endTimeInput.value = Number(payload.default_end_time || 1).toPrecision(6);
        maxStepsInput.value = Number(defaultResult.max_steps || 200);
        render(defaultResult);
      }}

      function buildRateMatrix(edges, nodeIds) {{
        const indexMap = new Map(nodeIds.map((nodeId, idx) => [nodeId, idx]));
        const matrix = nodeIds.map(() => nodeIds.map(() => 0));
        edges.forEach((edge) => {{
          const sourceIdx = indexMap.get(edge.source);
          const targetIdx = indexMap.get(edge.target);
          const rate = Number(edge.rate_constant || 0);
          if (sourceIdx === undefined || targetIdx === undefined || rate <= 0) {{
            return;
          }}
          matrix[targetIdx][sourceIdx] += rate;
          matrix[sourceIdx][sourceIdx] -= rate;
        }});
        return matrix;
      }}

      function solveLinearSystem(matrix, rhs) {{
        const n = rhs.length;
        const a = matrix.map((row, rowIdx) => row.map((value) => Number(value)).concat([Number(rhs[rowIdx] || 0)]));
        for (let pivot = 0; pivot < n; pivot += 1) {{
          let maxRow = pivot;
          for (let row = pivot + 1; row < n; row += 1) {{
            if (Math.abs(a[row][pivot]) > Math.abs(a[maxRow][pivot])) {{
              maxRow = row;
            }}
          }}
          if (Math.abs(a[maxRow][pivot]) < 1e-18) {{
            return rhs.slice();
          }}
          if (maxRow !== pivot) {{
            const tmp = a[pivot];
            a[pivot] = a[maxRow];
            a[maxRow] = tmp;
          }}
          const pivotValue = a[pivot][pivot];
          for (let col = pivot; col <= n; col += 1) {{
            a[pivot][col] /= pivotValue;
          }}
          for (let row = 0; row < n; row += 1) {{
            if (row === pivot) {{
              continue;
            }}
            const factor = a[row][pivot];
            if (factor === 0) {{
              continue;
            }}
            for (let col = pivot; col <= n; col += 1) {{
              a[row][col] -= factor * a[pivot][col];
            }}
          }}
        }}
        return a.map((row) => row[n]);
      }}

      function implicitEulerStep(matrix, state, dt) {{
        if (dt <= 0) {{
          return state.slice();
        }}
        const lhs = matrix.map((row, rowIdx) => row.map((value, colIdx) => (
          (rowIdx === colIdx ? 1 : 0) - dt * Number(value)
        )));
        const nextState = solveLinearSystem(lhs, state);
        for (let i = 0; i < nextState.length; i += 1) {{
          if (nextState[i] < 0) {{
            nextState[i] = 0;
          }}
        }}
        const total = nextState.reduce((sum, value) => sum + value, 0);
        const referenceTotal = state.reduce((sum, value) => sum + value, 0);
        if (total > 0 && referenceTotal > 0) {{
          for (let i = 0; i < nextState.length; i += 1) {{
            nextState[i] *= referenceTotal / total;
          }}
        }}
        return nextState;
      }}

      function simulate() {{
        const temperature = Number(temperatureInput.value || payload.temperature_kelvin);
        const maxSteps = Math.max(1, Number(maxStepsInput.value || 1000));
        const finalTime = Math.max(0, Number(endTimeInput.value || payload.default_end_time || 1));
        const populations = readInitialConditions();
        const nodeIds = payload.nodes.map(node => Number(node.id));
        const edges = payload.edges.map(edge => ({{
          source: Number(edge.source),
          target: Number(edge.target),
          barrier: Number(edge.barrier),
          reaction: String(edge.reaction || ""),
          rate_constant: eyringRate(Number(edge.barrier), temperature),
        }}));
        const matrix = buildRateMatrix(edges, nodeIds);
        let state = nodeIds.map(nodeId => Number(populations[nodeId] || 0));
        const history = [{{ time: 0, event: null, populations: Object.fromEntries(nodeIds.map((nodeId, idx) => [nodeId, state[idx]])) }}];
        let time = 0;
        const dt = finalTime / maxSteps;
        for (let step = 0; step < maxSteps; step += 1) {{
          if (dt > 0) {{
            state = implicitEulerStep(matrix, state, dt);
          }}
          time += dt;
          history.push({{
            time,
            event: null,
            populations: Object.fromEntries(nodeIds.map((nodeId, idx) => [nodeId, state[idx]])),
          }});
        }}

        return {{
          temperature_kelvin: temperature,
          max_steps: maxSteps,
          final_time: finalTime,
          seed: null,
          history,
          final_populations: Object.assign({{}}, history[history.length - 1].populations),
        }};
      }}

      function render(result) {{
        const finalPopulations = result.final_populations || {{}};
        const entries = Object.keys(finalPopulations)
          .map(key => [Number(key), Number(finalPopulations[key] || 0)])
          .sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]));
        summaryBody.innerHTML = entries.map(([nodeId, value]) => (
          `<tr><td>${{nodeId}}</td><td>${{nodeLabels.get(nodeId) || nodeId}}</td><td>${{value.toFixed(6)}}</td></tr>`
        )).join("") || "<tr><td colspan='3'>No data</td></tr>";
        eventsEl.textContent = String(Math.max((result.history || []).length - 1, 0));
        const finalTime = (result.history || []).length ? Number(result.history[result.history.length - 1].time || 0) : 0;
        finalTimeEl.textContent = finalTime.toFixed(6);
        if (entries.length > 0) {{
          dominantEl.textContent = nodeLabels.get(entries[0][0]) || String(entries[0][0]);
        }} else {{
          dominantEl.textContent = "n/a";
        }}
        drawPlot(result.history || []);
      }}

      function drawPlot(history) {{
        while (svg.firstChild) {{
          svg.removeChild(svg.firstChild);
        }}
        const width = 960;
        const height = 360;
        const margin = {{ top: 20, right: 20, bottom: 40, left: 56 }};
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        const nodeIds = payload.nodes.map(node => Number(node.id));
        const maxTime = history.length ? Math.max(...history.map(step => Number(step.time || 0))) : 1;
        const maxPopulation = Math.max(
          1,
          ...history.flatMap(step => nodeIds.map(nodeId => Number((step.populations || {{}})[nodeId] || 0)))
        );

        const ns = "http://www.w3.org/2000/svg";
        const g = document.createElementNS(ns, "g");
        g.setAttribute("transform", `translate(${{margin.left}},${{margin.top}})`);
        svg.appendChild(g);

        function sx(value) {{
          return maxTime === 0 ? 0 : (value / maxTime) * innerWidth;
        }}
        function sy(value) {{
          return innerHeight - (value / maxPopulation) * innerHeight;
        }}

        const axisColor = "rgba(201,216,240,0.78)";
        const xAxis = document.createElementNS(ns, "line");
        xAxis.setAttribute("x1", "0");
        xAxis.setAttribute("y1", String(innerHeight));
        xAxis.setAttribute("x2", String(innerWidth));
        xAxis.setAttribute("y2", String(innerHeight));
        xAxis.setAttribute("stroke", axisColor);
        g.appendChild(xAxis);

        const yAxis = document.createElementNS(ns, "line");
        yAxis.setAttribute("x1", "0");
        yAxis.setAttribute("y1", "0");
        yAxis.setAttribute("x2", "0");
        yAxis.setAttribute("y2", String(innerHeight));
        yAxis.setAttribute("stroke", axisColor);
        g.appendChild(yAxis);

        const palette = ["#7ef0c7", "#63d5ff", "#9cafff", "#ff8eb0", "#ffd166", "#59d8b6", "#b89cff", "#ff9d6c"];
        nodeIds.forEach((nodeId, idx) => {{
          const color = palette[idx % palette.length];
          const points = history.map(step => `${{sx(Number(step.time || 0)).toFixed(2)}},${{sy(Number((step.populations || {{}})[nodeId] || 0)).toFixed(2)}}`).join(" ");
          const polyline = document.createElementNS(ns, "polyline");
          polyline.setAttribute("fill", "none");
          polyline.setAttribute("stroke", color);
          polyline.setAttribute("stroke-width", "2");
          polyline.setAttribute("points", points);
          g.appendChild(polyline);

          const legend = document.createElementNS(ns, "text");
          legend.setAttribute("x", String(innerWidth - 160));
          legend.setAttribute("y", String(16 + idx * 18));
          legend.setAttribute("fill", color);
          legend.setAttribute("font-size", "12");
          legend.textContent = nodeLabels.get(nodeId) || String(nodeId);
          g.appendChild(legend);
        }});

        const xLabel = document.createElementNS(ns, "text");
        xLabel.setAttribute("x", String(innerWidth / 2));
        xLabel.setAttribute("y", String(innerHeight + 32));
        xLabel.setAttribute("text-anchor", "middle");
        xLabel.setAttribute("fill", "#9eb4d6");
        xLabel.textContent = "Simulation time (a.u.)";
        g.appendChild(xLabel);

        const yLabel = document.createElementNS(ns, "text");
        yLabel.setAttribute("transform", `translate(-40,${{innerHeight / 2}}) rotate(-90)`);
        yLabel.setAttribute("text-anchor", "middle");
        yLabel.setAttribute("fill", "#9eb4d6");
        yLabel.textContent = "Population / concentration (M)";
        g.appendChild(yLabel);
      }}

      runButton.addEventListener("click", () => render(simulate()));
      resetButton.addEventListener("click", resetDefaults);
      renderNetworkExplorer("retropaths-network", retropathsNetworkPayload);
      renderNetworkExplorer("neb-network", nebNetworkPayload);
      render(defaultResult);
    }})();
  </script>
</body>
</html>
"""


def write_status_html(
    workspace: RetropathsWorkspace,
    kmc_temperature_kelvin: float = 298.15,
    kmc_initial_conditions: dict[int, float] | None = None,
) -> tuple[RetropathsNEBQueue, Pot, Path]:
    queue = RetropathsNEBQueue.read_from_disk(workspace.queue_fp)
    retropaths_pot = load_retropaths_pot(workspace)
    try:
        pot = load_partial_annotated_pot(workspace)
    except FileNotFoundError:
        cached_pot = _safe_read_pot(workspace.annotated_neb_pot_fp)
        if cached_pot is None:
            raise
        pot = cached_pot
    full_pot = _safe_read_pot(workspace.neb_pot_fp)
    optimized_nodes = 0
    if full_pot is not None:
        optimized_nodes = sum(
            bool(full_pot.graph.nodes[node_index].get("endpoint_optimized"))
            for node_index in full_pot.graph.nodes
        )
    prepared_edge_count = sum(
        item.status in {"pending", "running", "completed", "failed", "skipped_attempted"}
        for item in queue.items
    )
    edge_visualizations = _write_edge_visualizations(workspace=workspace, pot=pot)
    html = build_status_html(
        workspace=workspace,
        retropaths_pot=retropaths_pot,
        pot=pot,
        queue=queue,
        optimized_nodes=optimized_nodes,
        prepared_edge_count=prepared_edge_count,
        edge_visualizations=edge_visualizations,
        kmc_temperature_kelvin=kmc_temperature_kelvin,
        kmc_initial_conditions=kmc_initial_conditions,
    )
    workspace.status_html_fp.write_text(html, encoding="utf-8")
    return queue, pot, workspace.status_html_fp


def run_netgen_smiles_workflow(
    workspace: RetropathsWorkspace,
    progress: Callable[[str], None] | None = None,
) -> tuple[RetropathsNEBQueue, Pot]:
    if progress is None:
        progress = lambda _msg: None

    progress("Loading run inputs.")
    run_inputs = RunInputs.open(workspace.inputs_fp)
    progress("Preparing retropaths pot, converted NEB pot, and queue.")
    pot, queue = prepare_neb_workspace(workspace)
    progress(
        f"Prepared network with {pot.graph.number_of_nodes()} nodes, {pot.graph.number_of_edges()} edges, and {summarize_queue(queue)['items']} queued NEB jobs."
    )
    pot = prepare_optimized_neb_pot_from_pot(
        pot=pot, workspace=workspace, run_inputs=run_inputs, progress=progress
    )
    progress("Starting recursive NEB queue execution.")
    queue = run_retropaths_neb_queue(
        pot=pot,
        run_inputs=run_inputs,
        queue_fp=workspace.queue_fp,
        output_dir=workspace.queue_output_dir,
        pot_fp=workspace.neb_pot_fp,
        max_parallel_nebs=workspace.max_parallel_nebs,
    )
    progress("Reconstructing partial NEB pot from completed queue results.")
    annotated = load_partial_annotated_pot(workspace)
    return queue, annotated
