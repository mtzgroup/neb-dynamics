from __future__ import annotations

from dataclasses import asdict, dataclass
import contextlib
import json
import re
import sys
import types
from pathlib import Path
from typing import Callable

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
from neb_dynamics.pot import Pot
from neb_dynamics.retropaths_compat import retropaths_pot_to_neb_pot
from neb_dynamics.retropaths_queue import (
    RetropathsNEBQueue,
    _trim_balanced_endpoint,
    annotate_pot_with_queue_results,
    build_retropaths_neb_queue,
    load_completed_queue_chains,
    run_retropaths_neb_queue,
)
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
            start_smiles = chain[0].graph.force_smiles()
        if len(chain.nodes) > 0 and getattr(chain[-1], "graph", None) is not None:
            end_smiles = chain[-1].graph.force_smiles()
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
    kmc_payload_json = json.dumps(kmc_payload)
    kmc_default_result_json = json.dumps(kmc_default_result)
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
  <script>
    (function() {{
      const payload = JSON.parse(document.getElementById("kmc-payload").textContent);
      const defaultResult = JSON.parse(document.getElementById("kmc-default-result").textContent);
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
