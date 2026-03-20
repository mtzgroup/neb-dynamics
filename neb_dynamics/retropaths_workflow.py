from __future__ import annotations

from dataclasses import asdict, dataclass
import contextlib
import io
import json
import re
import sys
import types
from pathlib import Path
from time import time
from typing import Any, Callable

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.inputs import RunInputs
from neb_dynamics.kmc import (
    build_kmc_payload,
    normalize_initial_conditions,
    simulate_kmc,
)
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.nodes.nodehelpers import is_identical
from neb_dynamics.molecule import Molecule
from neb_dynamics.pot import Pot
from neb_dynamics.qcio_structure_helpers import structure_to_molecule
from neb_dynamics.retropaths_compat import retropaths_pot_to_neb_pot
from neb_dynamics.retropaths_queue import (
    _global_environment_molecule,
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
    return Path(__file__).resolve().parents[3] / "retropaths"


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

    hf, _RetropathsMolecule, RetropathsPot = _load_retropaths_classes()
    library = hf.pload(workspace.reactions_path)
    temp_pot = RetropathsPot(
        root=source_molecule.copy(),
        environment=_global_environment_molecule(pot) or Molecule(),
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
                molecule=result_molecule.copy(),
                converged=bool(result_attrs.get("converged", False)),
                td=td,
                endpoint_optimized=False,
                generated_by="retropaths_apply_reactions",
            )
            added_nodes += 1
        else:
            target_attrs = pot.graph.nodes[target_index]
            target_attrs.setdefault("molecule", result_molecule.copy())
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
        if _nodes_identical(normalized, reference_node, chain_inputs):
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
) -> int | None:
    for node_index in pot.graph.nodes:
        existing = pot.graph.nodes[node_index].get("td")
        if existing is None:
            continue
        if _nodes_identical(node, existing, chain_inputs):
            return node_index
    return None


def _register_network_node(
    pot: Pot,
    source_pot: Pot,
    node: StructureNode,
    chain_inputs: ChainInputs,
    preferred_index: int | None = None,
) -> int:
    if preferred_index is not None and preferred_index in source_pot.graph.nodes:
        if preferred_index not in pot.graph.nodes:
            pot.graph.add_node(preferred_index, **dict(source_pot.graph.nodes[preferred_index]))
        return preferred_index

    existing_index = _find_existing_network_node(pot, node, chain_inputs)
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

    for source_node, target_node in sorted(pot.graph.edges):
        edge_attrs = pot.graph.edges[(source_node, target_node)]
        chains = edge_attrs.get("list_of_nebs") or []
        if not chains:
            continue

        filename = f"edge_{source_node}_{target_node}.html"
        out_fp = out_dir / filename
        chain = chains[-1]
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
            chain=chains[-1],
            chain_trajectory=list(chains) if len(chains) > 1 else None,
        )
        html = html.replace("<body>", f"<body>\n  {title_html}", 1)
        out_fp.write_text(html, encoding="utf-8")
        rows.append(
            {
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
            }
        )

    return rows


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(val, depth + 1)
            for key, val in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in list(value)[:40]]
    if hasattr(value, "force_smiles"):
        with contextlib.suppress(Exception):
            return _quiet_force_smiles(value)
    if hasattr(value, "smiles"):
        with contextlib.suppress(Exception):
            return str(value.smiles)
    if hasattr(value, "to_dict"):
        with contextlib.suppress(Exception):
            return _json_safe(value.to_dict(), depth + 1)
    if hasattr(value, "model_dump"):
        with contextlib.suppress(Exception):
            return _json_safe(value.model_dump(), depth + 1)
    if hasattr(value, "__dict__"):
        with contextlib.suppress(Exception):
            return {
                str(key): _json_safe(val, depth + 1)
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
        template_payload = template_payloads.get(reaction_name) or {}
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
    pot.graph.clear()
    pot.run_time = source_pot.run_time
    if 0 in source_pot.graph.nodes:
        pot.graph.add_node(0, **dict(source_pot.graph.nodes[0]))

    chain_inputs = ChainInputs()
    chains_by_edge: dict[tuple[int, int], list[Chain]] = {}

    for item in queue.items:
        if item.status != "completed" or not item.result_dir:
            continue

        history = TreeNode.read_from_disk(
            folder_name=item.result_dir,
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
                preferred_index=start_index_hint,
            )
            end_index = _register_network_node(
                pot=pot,
                source_pot=source_pot,
                node=end_node,
                chain_inputs=chain_inputs,
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
    body {{ font-family: Georgia, serif; margin: 24px; background: #f4f0e8; color: #1f1b16; }}
    .hero {{ padding: 20px 24px; border: 1px solid #b9a98f; background: linear-gradient(135deg, #efe6d5, #f8f5ef); margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card {{ border: 1px solid #c8b99f; background: #fffdf8; padding: 14px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fffdf8; }}
    th, td {{ border: 1px solid #d8ccb9; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #efe2c8; }}
    h1, h2 {{ margin: 0 0 10px 0; }}
    .section {{ margin-top: 24px; }}
    code {{ background: #f0e7da; padding: 2px 4px; }}
    .tab-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .tab-button {{ border: 1px solid #b9a98f; background: #f8f5ef; color: #3b3026; padding: 8px 12px; cursor: pointer; }}
    .tab-button.active {{ background: #d8a13b; color: #23170b; border-color: #b07f29; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .explorer-layout {{ display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(320px, 1fr); gap: 16px; }}
    .explorer-card {{ border: 1px solid #c8b99f; background: #fffdf8; padding: 14px; }}
    .explorer-svg {{ width: 100%; height: auto; min-height: 360px; border: 1px solid #d8ccb9; background: linear-gradient(180deg, #fffdf8, #f6efe2); }}
    .info-tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }}
    .info-tab-button {{ border: 1px solid #c8b99f; background: #f8f0e2; padding: 6px 10px; cursor: pointer; }}
    .info-tab-button.active {{ background: #d8a13b; border-color: #b07f29; }}
    .info-tab-panel {{ display: none; }}
    .info-tab-panel.active {{ display: block; }}
    .placeholder {{ color: #6b6157; font-style: italic; }}
    .network-node {{ fill: #7b6b58; stroke: #fdf7ee; stroke-width: 2; cursor: pointer; }}
    .network-node.root {{ fill: #b35c1e; }}
    .network-node.selected {{ fill: #d8a13b; stroke: #6b3f07; stroke-width: 3; }}
    .network-edge-hitbox {{ stroke: transparent; stroke-width: 16; cursor: pointer; fill: none; }}
    .network-edge-line {{ stroke: #8e7f6d; stroke-width: 2.5; fill: none; }}
    .network-edge-line.selected {{ stroke: #d8a13b; stroke-width: 4; }}
    .network-label {{ font-size: 12px; fill: #2d241c; pointer-events: none; }}
    .template-visualization svg {{ max-width: 100%; height: auto; }}
    pre.json-block {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: #f6f0e7; border: 1px solid #e0d4c3; padding: 12px; }}
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
      <svg id="kmc-plot" width="960" height="360" viewBox="0 0 960 360" style="width: 100%; height: auto; border: 1px solid #d8ccb9; background: #fff;"></svg>
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

        const axisColor = "#5f5447";
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

        const palette = ["#8b5e3c", "#326b5b", "#8e3b46", "#3a5683", "#b06c3b", "#5b4b8a", "#56733d", "#a4486f"];
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
        xLabel.textContent = "Simulation time (a.u.)";
        g.appendChild(xLabel);

        const yLabel = document.createElementNS(ns, "text");
        yLabel.setAttribute("transform", `translate(-40,${{innerHeight / 2}}) rotate(-90)`);
        yLabel.setAttribute("text-anchor", "middle");
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
