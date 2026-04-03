from __future__ import annotations

import contextlib
import copy
import concurrent.futures
import multiprocessing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs, RunInputs
from neb_dynamics.molecule import Molecule
from neb_dynamics.msmep import MSMEP
from neb_dynamics.nodes.nodehelpers import _is_connectivity_identical
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.pot import Pot
from neb_dynamics.qcio_structure_helpers import structure_to_molecule
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.retropaths_compat import annotate_pot_with_neb_results
from neb_dynamics.retropaths_compat import structure_node_from_graph_like_molecule
try:
    from neb_dynamics.retropaths_pseudoalign import pseudoalign_reaction_pair
except ModuleNotFoundError:
    def pseudoalign_reaction_pair(start: StructureNode, end: StructureNode, _reaction: Any) -> tuple[StructureNode, StructureNode]:
        return start, end


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_payload(node: StructureNode) -> dict[str, Any]:
    payload = {
        "symbols": list(node.structure.symbols),
        "charge": int(node.structure.charge),
        "multiplicity": int(node.structure.multiplicity),
        "geometry": np.round(np.asarray(node.coords, dtype=float), 8).tolist(),
    }
    if node.graph is not None:
        payload["graph"] = node.graph.to_serializable()
    return payload


def structure_node_attempt_signature(node: StructureNode) -> str:
    payload = json.dumps(_node_payload(node), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_attempt_key(node1: StructureNode, node2: StructureNode) -> str:
    signatures = sorted(
        [
            structure_node_attempt_signature(node1),
            structure_node_attempt_signature(node2),
        ]
    )
    return hashlib.sha256("::".join(signatures).encode("utf-8")).hexdigest()


def pair_is_direct_neb_compatible(node1: StructureNode, node2: StructureNode) -> tuple[bool, str | None]:
    symbols1 = list(node1.structure.symbols)
    symbols2 = list(node2.structure.symbols)
    if len(symbols1) != len(symbols2):
        return False, f"Atom count mismatch: {len(symbols1)} != {len(symbols2)}"
    if sorted(symbols1) != sorted(symbols2):
        return False, "Element inventory mismatch between endpoints."
    return True, None


def _symbol_counter_from_node(node: StructureNode) -> Counter:
    return Counter(node.structure.symbols)


def _counter_difference(larger: Counter, smaller: Counter) -> Counter | None:
    diff = Counter()
    for key in larger.keys() | smaller.keys():
        delta = larger[key] - smaller[key]
        if delta < 0:
            return None
        if delta > 0:
            diff[key] = delta
    return diff


def _molecule_atom_count(molecule: Molecule | None) -> int:
    if molecule is None:
        return 0
    with contextlib.suppress(Exception):
        return len(molecule.list_of_elements())
    return 0


def _structure_node_from_graph_if_mismatch(
    node: StructureNode,
    graph: Molecule | None,
    *,
    charge: int,
    spinmult: int,
) -> StructureNode:
    if graph is None:
        return node
    graph_atom_count = _molecule_atom_count(graph)
    if graph_atom_count <= 0 or graph_atom_count == len(node.structure.symbols):
        return node
    rebuilt = structure_node_from_graph_like_molecule(
        graph,
        charge=charge,
        spinmult=spinmult,
    )
    rebuilt._cached_energy = getattr(node, "_cached_energy", None)
    rebuilt._cached_gradient = getattr(node, "_cached_gradient", None)
    rebuilt._cached_result = getattr(node, "_cached_result", None)
    rebuilt.converged = node.converged
    return rebuilt


def _piece_signature(piece: Molecule) -> str:
    with contextlib.suppress(Exception):
        return str(piece.force_smiles())
    return ".".join(sorted(piece.list_of_elements()))


def _piece_counter(pieces: list[Molecule]) -> Counter:
    return Counter(_piece_signature(piece) for piece in pieces)


def _missing_graph_pieces(
    larger_graph: Molecule | None,
    smaller_graph: Molecule | None,
) -> list[Molecule] | None:
    if larger_graph is None or smaller_graph is None:
        return None

    larger_pieces = list(larger_graph.separate_graph_in_pieces())
    smaller_pieces = list(smaller_graph.separate_graph_in_pieces())
    larger_counts = _piece_counter(larger_pieces)
    smaller_counts = _piece_counter(smaller_pieces)
    missing_counts = _counter_difference(larger_counts, smaller_counts)
    if missing_counts is None:
        return None

    missing: list[Molecule] = []
    remaining = Counter(missing_counts)
    for piece in larger_pieces:
        signature = _piece_signature(piece)
        if remaining[signature] <= 0:
            continue
        missing.append(piece)
        remaining[signature] -= 1
    if any(value > 0 for value in remaining.values()):
        return None
    return missing


def _graph_with_appended_fragment(base_graph: Molecule | None, fragment_graph: Molecule | None) -> Molecule | None:
    if base_graph is None and fragment_graph is None:
        return None
    if base_graph is None:
        return fragment_graph.copy()
    if fragment_graph is None:
        return base_graph.copy()

    combined = base_graph.copy()
    start_index = (max(combined.nodes) + 1) if len(combined.nodes) else 0
    mapping = {
        old_index: start_index + offset
        for offset, old_index in enumerate(sorted(fragment_graph.nodes))
    }
    for old_index in sorted(fragment_graph.nodes):
        combined.add_node(mapping[old_index], **dict(fragment_graph.nodes[old_index]))
    for atom1, atom2, attrs in fragment_graph.edges(data=True):
        combined.add_edge(mapping[atom1], mapping[atom2], **dict(attrs))
    if hasattr(combined, "set_neighbors"):
        combined.set_neighbors()
    return combined


def _append_fragment_structure(base_node: StructureNode, fragment_node: StructureNode, offset: float = 4.0) -> StructureNode:
    base_coords = np.asarray(base_node.coords, dtype=float)
    frag_coords = np.asarray(fragment_node.coords, dtype=float)

    if len(base_coords) == 0:
        shifted = frag_coords.copy()
    else:
        max_x = float(np.max(base_coords[:, 0]))
        min_x = float(np.min(frag_coords[:, 0]))
        shifted = frag_coords.copy()
        shifted[:, 0] = shifted[:, 0] - min_x + max_x + offset

    geometry = np.concatenate([base_coords, shifted], axis=0)
    symbols = list(base_node.structure.symbols) + list(fragment_node.structure.symbols)
    extras = dict(getattr(base_node.structure, "extras", {}) or {})
    extras.setdefault("retropaths_original_atom_count", len(base_node.structure.symbols))

    new_structure = base_node.structure.__class__(
        geometry=geometry,
        symbols=symbols,
        charge=base_node.structure.charge + fragment_node.structure.charge,
        multiplicity=base_node.structure.multiplicity,
        extras=extras,
    )
    return StructureNode(
        structure=new_structure,
        graph=_graph_with_appended_fragment(base_node.graph, fragment_node.graph),
        converged=base_node.converged,
        _cached_energy=None,
        _cached_gradient=None,
        _cached_result=None,
    )


def _find_environment_piece_combo(
    target_counter: Counter,
    environment_pieces: list[Molecule],
    max_depth: int = 6,
) -> list[Molecule] | None:
    piece_counters = [Counter(piece.list_of_elements()) for piece in environment_pieces]
    memo: dict[tuple[tuple[tuple[str, int], ...], int], list[Molecule] | None] = {}

    def _search(remaining: Counter, depth: int) -> list[Molecule] | None:
        if sum(remaining.values()) == 0:
            return []
        if depth >= max_depth:
            return None
        key = (tuple(sorted((k, v) for k, v in remaining.items() if v > 0)), depth)
        if key in memo:
            return memo[key]

        for piece, piece_counter in zip(environment_pieces, piece_counters):
            next_remaining = _counter_difference(remaining, piece_counter)
            if next_remaining is None:
                continue
            result = _search(next_remaining, depth + 1)
            if result is not None:
                memo[key] = [piece] + result
                return memo[key]

        memo[key] = None
        return None

    return _search(target_counter, 0)


def _global_environment_molecule(pot: Pot) -> Molecule | None:
    root_node = pot.graph.nodes.get(0, {})
    environment = root_node.get("environment")
    if environment is None or environment.is_empty():
        return None
    return environment


def _augment_node_with_environment_fragments(
    node: StructureNode,
    fragments: list[Molecule],
    charge: int,
    spinmult: int,
) -> StructureNode:
    augmented = node.copy()
    for fragment in fragments:
        fragment_node = structure_node_from_graph_like_molecule(
            fragment, charge=charge, spinmult=spinmult
        )
        augmented = _append_fragment_structure(augmented, fragment_node)
    return augmented


def build_balanced_endpoints(
    source_td: StructureNode,
    target_td: StructureNode,
    environment: Molecule | None,
    source_graph: Molecule | None = None,
    target_graph: Molecule | None = None,
    charge: int = 0,
    spinmult: int = 1,
) -> tuple[StructureNode, StructureNode, str | None]:
    source_td = _structure_node_from_graph_if_mismatch(
        source_td,
        source_graph,
        charge=charge,
        spinmult=spinmult,
    )
    target_td = _structure_node_from_graph_if_mismatch(
        target_td,
        target_graph,
        charge=charge,
        spinmult=spinmult,
    )

    compatible, reason = pair_is_direct_neb_compatible(source_td, target_td)
    if compatible:
        return source_td, target_td, None

    source_missing_graph_pieces = _missing_graph_pieces(target_graph, source_graph)
    if source_missing_graph_pieces:
        augmented_source = _augment_node_with_environment_fragments(
            source_td, source_missing_graph_pieces, charge=charge, spinmult=spinmult
        )
        compatible, reason = pair_is_direct_neb_compatible(augmented_source, target_td)
        if compatible:
            return augmented_source, target_td, None

    target_missing_graph_pieces = _missing_graph_pieces(source_graph, target_graph)
    if target_missing_graph_pieces:
        augmented_target = _augment_node_with_environment_fragments(
            target_td, target_missing_graph_pieces, charge=charge, spinmult=spinmult
        )
        compatible, reason = pair_is_direct_neb_compatible(source_td, augmented_target)
        if compatible:
            return source_td, augmented_target, None

    if environment is None or environment.is_empty():
        return source_td, target_td, reason

    source_counts = _symbol_counter_from_node(source_td)
    target_counts = _symbol_counter_from_node(target_td)
    env_pieces = environment.separate_graph_in_pieces()

    source_missing = _counter_difference(target_counts, source_counts)
    if source_missing is not None:
        source_fragments = _find_environment_piece_combo(source_missing, env_pieces)
        if source_fragments is not None:
            augmented_source = _augment_node_with_environment_fragments(
                source_td, source_fragments, charge=charge, spinmult=spinmult
            )
            return augmented_source, target_td, None

    target_missing = _counter_difference(source_counts, target_counts)
    if target_missing is not None:
        target_fragments = _find_environment_piece_combo(target_missing, env_pieces)
        if target_fragments is not None:
            augmented_target = _augment_node_with_environment_fragments(
                target_td, target_fragments, charge=charge, spinmult=spinmult
            )
            return source_td, augmented_target, None

    return source_td, target_td, reason


@dataclass
class NEBQueueItem:
    job_id: str
    source_node: int
    target_node: int
    attempt_key: str
    reaction: str | None = None
    status: str = "pending"
    created_at: str = field(default_factory=_utcnow_iso)
    started_at: str | None = None
    finished_at: str | None = None
    result_dir: str | None = None
    output_chain_xyz: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NEBQueueItem":
        return cls(**data)


@dataclass
class RetropathsNEBQueue:
    items: list[NEBQueueItem] = field(default_factory=list)
    attempted_pairs: dict[str, dict[str, Any]] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "attempted_pairs": self.attempted_pairs,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetropathsNEBQueue":
        return cls(
            version=data.get("version", 1),
            attempted_pairs=dict(data.get("attempted_pairs", {})),
            items=[
                NEBQueueItem.from_dict(item)
                for item in data.get("items", [])
            ],
        )

    @classmethod
    def read_from_disk(cls, fp: Path) -> "RetropathsNEBQueue":
        fp = Path(fp)
        return cls.from_dict(json.loads(fp.read_text()))

    def write_to_disk(self, fp: Path) -> None:
        fp = Path(fp)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    def find_item(self, source_node: int, target_node: int) -> NEBQueueItem | None:
        for item in self.items:
            if item.source_node == source_node and item.target_node == target_node:
                return item
        return None

    def replace_item(self, new_item: NEBQueueItem) -> None:
        for index, item in enumerate(self.items):
            if item.source_node == new_item.source_node and item.target_node == new_item.target_node:
                self.items[index] = new_item
                return
        self.items.append(new_item)

    def recover_stale_running_items(
        self,
        *,
        output_dir: Path | None = None,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> bool:
        changed = False
        for item in self.items:
            if item.status != "running":
                continue
            attempt = self.attempted_pairs.get(item.attempt_key)

            recovered = _recover_interrupted_queue_item_outputs(
                item,
                output_dir=output_dir,
                charge=charge,
                multiplicity=multiplicity,
            )
            if recovered:
                if attempt is not None:
                    attempt["status"] = "completed"
                    attempt["finished_at"] = item.finished_at
                    attempt["result_dir"] = item.result_dir
                    attempt["output_chain_xyz"] = item.output_chain_xyz
                    attempt.pop("error", None)
                changed = True
                continue

            item.status = "failed"
            item.finished_at = _utcnow_iso()
            if item.error is None:
                if output_dir is None:
                    item.error = "Recovered stale running queue item before restart."
                else:
                    item.error = (
                        "Recovered stale running queue item before restart. "
                        "No recoverable NEB result artifacts were written."
                    )
            if attempt is not None:
                attempt["status"] = "failed"
                attempt["finished_at"] = item.finished_at
                attempt["error"] = item.error
            changed = True
        return changed


def _recover_interrupted_queue_item_outputs(
    item: NEBQueueItem,
    *,
    output_dir: Path | None,
    charge: int,
    multiplicity: int,
) -> bool:
    candidate_result_dirs: list[Path] = []
    if item.result_dir:
        candidate_result_dirs.append(Path(item.result_dir))
    if output_dir is not None:
        candidate_result_dirs.append(output_dir / f"pair_{item.source_node}_{item.target_node}_msmep")

    result_dir_path: Path | None = None
    for candidate in candidate_result_dirs:
        if not candidate.exists() or not candidate.is_dir():
            continue
        try:
            history = TreeNode.read_from_disk(
                folder_name=candidate,
                charge=charge,
                multiplicity=multiplicity,
            )
        except Exception:
            continue
        if getattr(history, "output_chain", None) is None:
            continue
        result_dir_path = candidate.resolve()
        break

    if result_dir_path is None:
        return False

    candidate_chain_fps: list[Path] = []
    if item.output_chain_xyz:
        candidate_chain_fps.append(Path(item.output_chain_xyz))
    if output_dir is not None:
        candidate_chain_fps.append(output_dir / f"pair_{item.source_node}_{item.target_node}.xyz")

    output_chain_fp: str | None = None
    for candidate in candidate_chain_fps:
        if candidate.exists() and candidate.is_file():
            output_chain_fp = str(candidate.resolve())
            break

    item.status = "completed"
    item.finished_at = _utcnow_iso()
    item.result_dir = str(result_dir_path)
    item.output_chain_xyz = output_chain_fp
    item.error = None
    return True


def _queue_item_for_edge(
    pot: Pot,
    source_node: int,
    target_node: int,
    reaction: str | None,
    attempted_pairs: dict[str, dict[str, Any]],
) -> NEBQueueItem:
    source_td = pot.graph.nodes[source_node].get("td")
    target_td = pot.graph.nodes[target_node].get("td")

    if source_td is None or target_td is None:
        return NEBQueueItem(
            job_id=f"{source_node}->{target_node}",
            source_node=source_node,
            target_node=target_node,
            attempt_key="",
            reaction=reaction,
            status="missing_td",
            error="Both queue endpoints need node['td'] structures.",
        )

    balance_reason = None
    compatible, reason = pair_is_direct_neb_compatible(source_td, target_td)
    if not compatible:
        balanced_source, balanced_target, balance_reason = build_balanced_endpoints(
            source_td=source_td,
            target_td=target_td,
            environment=_global_environment_molecule(pot),
            source_graph=pot.graph.nodes[source_node].get("molecule"),
            target_graph=pot.graph.nodes[target_node].get("molecule"),
            charge=source_td.structure.charge,
            spinmult=source_td.structure.multiplicity,
        )
        compatible, reason = pair_is_direct_neb_compatible(balanced_source, balanced_target)
        if compatible:
            source_td = balanced_source
            target_td = balanced_target

    if not compatible:
        return NEBQueueItem(
            job_id=f"{source_node}->{target_node}",
            source_node=source_node,
            target_node=target_node,
            attempt_key="",
            reaction=reaction,
            status="incompatible",
            error=balance_reason or reason,
        )

    attempt_key = pair_attempt_key(source_td, target_td)
    pair_key = tuple(sorted((int(source_node), int(target_node))))
    has_terminal_pair_attempt = False
    for attempt in attempted_pairs.values():
        if not isinstance(attempt, dict):
            continue
        with contextlib.suppress(Exception):
            attempt_pair_key = tuple(
                sorted((int(attempt.get("source_node")), int(attempt.get("target_node"))))
            )
            if attempt_pair_key != pair_key:
                continue
            status = str(attempt.get("status") or "").strip().lower()
            if status in {"completed", "running", "skipped_identical"}:
                has_terminal_pair_attempt = True
                break

    status = "skipped_attempted" if (attempt_key in attempted_pairs or has_terminal_pair_attempt) else "pending"
    return NEBQueueItem(
        job_id=f"{source_node}->{target_node}",
        source_node=source_node,
        target_node=target_node,
        attempt_key=attempt_key,
        reaction=reaction,
        status=status,
    )


def _is_retryable_legacy_failure(item: NEBQueueItem) -> bool:
    if item.status == "incompatible":
        return True
    if item.status != "failed" or not item.error:
        return False
    retryable_patterns = (
        "setting an array element with a sequence",
        "Atom count mismatch",
        "Element inventory mismatch",
        "is not JSON serializable",
        "Recovered stale running queue item before restart.",
        "signal only works in main thread",
    )
    return any(pattern in item.error for pattern in retryable_patterns)


def _should_refresh_existing_item(
    existing: NEBQueueItem,
    candidate: NEBQueueItem,
) -> bool:
    if str(existing.status or "").strip().lower() in {"completed", "running", "skipped_identical"}:
        return False
    if _is_retryable_legacy_failure(existing):
        return True
    if existing.reaction != candidate.reaction:
        return True
    if existing.attempt_key != candidate.attempt_key:
        return True
    if existing.status in {"pending", "skipped_attempted", "incompatible", "missing_td"}:
        if existing.status != candidate.status:
            return True
        if (existing.error or "") != (candidate.error or ""):
            return True
    return False


def _run_single_item_worker(
    pair: Chain,
    run_inputs: RunInputs,
    result_dir: str,
    output_chain_xyz: str,
) -> dict[str, str]:
    msmep = MSMEP(inputs=run_inputs)
    history = msmep.run_recursive_minimize(pair)
    output_chain = getattr(history, "output_chain", None)
    if output_chain is None:
        raise RuntimeError("Recursive MSMEP returned no output_chain.")

    result_dir_path = Path(result_dir)
    output_chain_fp = Path(output_chain_xyz)
    result_dir_path.parent.mkdir(parents=True, exist_ok=True)
    output_chain_fp.parent.mkdir(parents=True, exist_ok=True)
    history.write_to_disk(result_dir_path)
    output_chain.write_to_disk(output_chain_fp)
    return {
        "result_dir": str(result_dir_path),
        "output_chain_xyz": str(output_chain_fp),
    }


def build_retropaths_neb_queue(
    pot: Pot,
    queue_fp: Path | None = None,
    overwrite: bool = False,
) -> RetropathsNEBQueue:
    queue = RetropathsNEBQueue()
    if queue_fp is not None:
        queue_fp = Path(queue_fp)
        if queue_fp.exists() and not overwrite:
            queue = RetropathsNEBQueue.read_from_disk(queue_fp)

    for source_node, target_node, attrs in pot.graph.edges(data=True):
        existing = queue.find_item(source_node, target_node)
        candidate = _queue_item_for_edge(
            pot=pot,
            source_node=source_node,
            target_node=target_node,
            reaction=attrs.get("reaction"),
            attempted_pairs=queue.attempted_pairs,
        )
        if existing is not None and not _should_refresh_existing_item(existing, candidate):
            continue
        if existing is not None and _is_retryable_legacy_failure(existing) and existing.attempt_key:
            queue.attempted_pairs.pop(existing.attempt_key, None)
            candidate = _queue_item_for_edge(
                pot=pot,
                source_node=source_node,
                target_node=target_node,
                reaction=attrs.get("reaction"),
                attempted_pairs=queue.attempted_pairs,
            )
        queue.replace_item(candidate)

    if queue_fp is not None:
        queue.write_to_disk(queue_fp)
    return queue


def ensure_queue_item_for_edge(
    pot: Pot,
    *,
    source_node: int,
    target_node: int,
    queue_fp: Path | None = None,
    overwrite: bool = False,
) -> RetropathsNEBQueue:
    queue = RetropathsNEBQueue()
    if queue_fp is not None:
        queue_fp = Path(queue_fp)
        if queue_fp.exists() and not overwrite:
            queue = RetropathsNEBQueue.read_from_disk(queue_fp)

    reaction = ""
    if pot.graph.has_edge(source_node, target_node):
        reaction = pot.graph.edges[(source_node, target_node)].get("reaction")

    existing = queue.find_item(source_node, target_node)
    candidate = _queue_item_for_edge(
        pot=pot,
        source_node=source_node,
        target_node=target_node,
        reaction=reaction,
        attempted_pairs=queue.attempted_pairs,
    )
    if existing is not None and not _should_refresh_existing_item(existing, candidate):
        return queue
    if existing is not None and _is_retryable_legacy_failure(existing) and existing.attempt_key:
        queue.attempted_pairs.pop(existing.attempt_key, None)
        candidate = _queue_item_for_edge(
            pot=pot,
            source_node=source_node,
            target_node=target_node,
            reaction=reaction,
            attempted_pairs=queue.attempted_pairs,
        )

    queue.replace_item(candidate)
    if queue_fp is not None:
        queue.write_to_disk(queue_fp)
    return queue


def _make_pair_chain(pot: Pot, item: NEBQueueItem, run_inputs: RunInputs) -> Chain:
    start = pot.graph.nodes[item.source_node]["td"].copy()
    end = pot.graph.nodes[item.target_node]["td"].copy()
    start, end, balance_reason = build_balanced_endpoints(
        source_td=start,
        target_td=end,
        environment=_global_environment_molecule(pot),
        source_graph=pot.graph.nodes[item.source_node].get("molecule"),
        target_graph=pot.graph.nodes[item.target_node].get("molecule"),
        charge=start.structure.charge,
        spinmult=start.structure.multiplicity,
    )
    start, end = pseudoalign_reaction_pair(start, end, item.reaction)
    compatible, reason = pair_is_direct_neb_compatible(start, end)
    if not compatible:
        raise ValueError(balance_reason or reason or "Endpoints are not direct-NEB-compatible.")
    return Chain.model_validate(
        {"nodes": [start, end], "parameters": run_inputs.chain_inputs}
    )


def _identical_endpoint_skip_reason(pair: Chain, run_inputs: RunInputs) -> str | None:
    start = pair.nodes[0]
    end = pair.nodes[-1]
    if (
        getattr(run_inputs.path_min_inputs, "skip_identical_graphs", False)
        and getattr(start, "has_molecular_graph", False)
        and getattr(end, "has_molecular_graph", False)
    ):
        try:
            if _is_connectivity_identical(start, end, verbose=False, collect_comparison=False):
                return "Skipped NEB because optimized endpoints have identical molecular graphs."
        except Exception:
            pass
    return None


def _optimize_single_node(node: StructureNode, run_inputs: RunInputs) -> tuple[StructureNode, str | None]:
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


def _optimize_nodes(
    nodes: list[StructureNode], run_inputs: RunInputs
) -> list[tuple[StructureNode, str | None]]:
    if not nodes:
        return []

    batch_optimizer = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    if callable(batch_optimizer) and len(nodes) > 1:
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

    return [_optimize_single_node(node=node, run_inputs=run_inputs) for node in nodes]


def _ensure_pair_endpoints_optimized(
    pot: Pot,
    item: NEBQueueItem,
    run_inputs: RunInputs,
) -> None:
    pending_indices: list[int] = []
    pending_nodes: list[StructureNode] = []
    for node_index in (item.source_node, item.target_node):
        node_attrs = pot.graph.nodes[node_index]
        if node_attrs.get("endpoint_optimized"):
            continue
        td = node_attrs.get("td")
        if td is None:
            continue
        pending_indices.append(node_index)
        pending_nodes.append(td)

    for node_index, (optimized_td, error) in zip(
        pending_indices, _optimize_nodes(nodes=pending_nodes, run_inputs=run_inputs)
    ):
        node_attrs = pot.graph.nodes[node_index]
        node_attrs["td"] = optimized_td
        node_attrs["endpoint_optimized"] = error is None
        if error is not None:
            node_attrs["endpoint_optimization_error"] = error


def run_retropaths_neb_queue(
    pot: Pot,
    run_inputs: RunInputs,
    queue_fp: Path,
    output_dir: Path,
    pot_fp: Path | None = None,
    stop_on_error: bool = False,
    max_parallel_nebs: int = 1,
) -> RetropathsNEBQueue:
    queue_fp = Path(queue_fp)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queue = build_retropaths_neb_queue(pot=pot, queue_fp=queue_fp, overwrite=False)
    if queue.recover_stale_running_items(
        output_dir=output_dir,
        charge=0,
        multiplicity=1,
    ):
        queue.write_to_disk(queue_fp)
    run_inputs.path_min_inputs.do_elem_step_checks = True
    max_parallel_nebs = max(1, int(max_parallel_nebs))

    runnable_items: list[NEBQueueItem] = []

    for item in queue.items:
        if item.status != "pending":
            continue

        source_td = pot.graph.nodes[item.source_node].get("td")
        target_td = pot.graph.nodes[item.target_node].get("td")
        balanced_source, balanced_target, balance_reason = build_balanced_endpoints(
            source_td=source_td,
            target_td=target_td,
            environment=_global_environment_molecule(pot),
            source_graph=pot.graph.nodes[item.source_node].get("molecule"),
            target_graph=pot.graph.nodes[item.target_node].get("molecule"),
            charge=source_td.structure.charge,
            spinmult=source_td.structure.multiplicity,
        )
        compatible, reason = pair_is_direct_neb_compatible(balanced_source, balanced_target)
        if not compatible:
            item.status = "incompatible"
            item.finished_at = _utcnow_iso()
            item.error = balance_reason or reason
            queue.write_to_disk(queue_fp)
            continue

        if item.attempt_key in queue.attempted_pairs:
            item.status = "skipped_attempted"
            item.finished_at = _utcnow_iso()
            queue.write_to_disk(queue_fp)
            continue

        item.status = "running"
        item.started_at = _utcnow_iso()
        queue.attempted_pairs[item.attempt_key] = {
            "job_id": item.job_id,
            "source_node": item.source_node,
            "target_node": item.target_node,
            "status": "running",
            "started_at": item.started_at,
        }
        queue.write_to_disk(queue_fp)
        _ensure_pair_endpoints_optimized(pot=pot, item=item, run_inputs=run_inputs)
        if pot_fp is not None:
            pot.write_to_disk(pot_fp)
        pair = _make_pair_chain(pot=pot, item=item, run_inputs=run_inputs)
        skip_reason = _identical_endpoint_skip_reason(pair, run_inputs)
        if skip_reason:
            item.status = "skipped_identical"
            item.finished_at = _utcnow_iso()
            item.error = skip_reason
            queue.attempted_pairs[item.attempt_key].update(
                {
                    "status": "skipped_identical",
                    "finished_at": item.finished_at,
                    "error": skip_reason,
                }
            )
            queue.write_to_disk(queue_fp)
            continue
        runnable_items.append(item)

    def _item_job_payload(item: NEBQueueItem) -> tuple[Chain, RunInputs, str, str]:
        local_inputs = copy.deepcopy(run_inputs)
        pair = _make_pair_chain(pot=pot, item=item, run_inputs=local_inputs)
        result_dir = output_dir / f"pair_{item.source_node}_{item.target_node}_msmep"
        output_chain_fp = output_dir / f"pair_{item.source_node}_{item.target_node}.xyz"
        return pair, local_inputs, str(result_dir), str(output_chain_fp)

    def _mark_completed(item: NEBQueueItem, result: dict[str, str]) -> None:
        item.status = "completed"
        item.finished_at = _utcnow_iso()
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
        queue.write_to_disk(queue_fp)

    def _mark_failed(item: NEBQueueItem, exc: Exception) -> None:
        item.status = "failed"
        item.finished_at = _utcnow_iso()
        item.error = f"{type(exc).__name__}: {exc}"
        queue.attempted_pairs[item.attempt_key].update(
            {
                "status": "failed",
                "finished_at": item.finished_at,
                "error": item.error,
            }
        )
        queue.write_to_disk(queue_fp)

    if max_parallel_nebs == 1:
        for item in runnable_items:
            try:
                result = _run_single_item_worker(*_item_job_payload(item))
            except Exception as exc:
                _mark_failed(item, exc)
                if stop_on_error:
                    raise
            else:
                _mark_completed(item, result)
        return queue

    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_parallel_nebs,
        mp_context=ctx,
    ) as executor:
        future_to_item = {
            executor.submit(_run_single_item_worker, *_item_job_payload(item)): item
            for item in runnable_items
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
            except Exception as exc:
                _mark_failed(item, exc)
                if stop_on_error:
                    raise
            else:
                _mark_completed(item, result)

    return queue


def load_completed_queue_chains(
    queue_fp: Path,
    charge: int = 0,
    multiplicity: int = 1,
) -> dict[tuple[int, int], list[Chain]]:
    queue = RetropathsNEBQueue.read_from_disk(queue_fp)
    chains_by_edge: dict[tuple[int, int], list[Chain]] = {}
    seen_records: set[tuple[int, int, str, str]] = set()
    completed_records: list[tuple[int, int, str, str]] = []

    for item in queue.items:
        if item.status != "completed":
            continue
        source_node = int(item.source_node)
        target_node = int(item.target_node)
        result_dir = str(item.result_dir or "").strip()
        output_chain_xyz = str(item.output_chain_xyz or "").strip()
        record = (source_node, target_node, result_dir, output_chain_xyz)
        if record in seen_records or (not result_dir and not output_chain_xyz):
            continue
        seen_records.add(record)
        completed_records.append(record)

    for attempt in queue.attempted_pairs.values():
        if str((attempt or {}).get("status") or "").strip().lower() != "completed":
            continue
        with contextlib.suppress(Exception):
            source_node = int((attempt or {}).get("source_node"))
            target_node = int((attempt or {}).get("target_node"))
        result_dir = str((attempt or {}).get("result_dir") or "").strip()
        output_chain_xyz = str((attempt or {}).get("output_chain_xyz") or "").strip()
        record = (source_node, target_node, result_dir, output_chain_xyz)
        if record in seen_records or (not result_dir and not output_chain_xyz):
            continue
        seen_records.add(record)
        completed_records.append(record)

    for source_node, target_node, result_dir, output_chain_xyz in completed_records:
        chain = None
        if result_dir:
            with contextlib.suppress(Exception):
                history = TreeNode.read_from_disk(
                    folder_name=result_dir,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                chain = getattr(history, "output_chain", None)
        if chain is None and output_chain_xyz:
            with contextlib.suppress(Exception):
                chain = Chain.from_xyz(output_chain_xyz, parameters=ChainInputs())
        if chain is None:
            continue
        chains_by_edge.setdefault((source_node, target_node), []).append(chain)

    return chains_by_edge


def _trim_balanced_endpoint(node: StructureNode, graph: Molecule | None) -> StructureNode:
    original_atom_count = None
    extras = getattr(node.structure, "extras", {}) or {}
    if "retropaths_original_atom_count" in extras:
        original_atom_count = int(extras["retropaths_original_atom_count"])
    if original_atom_count is None:
        return node.copy()

    trimmed_structure = node.structure.__class__(
        geometry=np.asarray(node.coords, dtype=float)[:original_atom_count],
        symbols=list(node.structure.symbols)[:original_atom_count],
        charge=node.structure.charge,
        multiplicity=node.structure.multiplicity,
        extras={k: v for k, v in extras.items() if k != "retropaths_original_atom_count"},
    )
    return StructureNode(
        structure=trimmed_structure,
        graph=graph.copy() if graph is not None else None,
        converged=node.converged,
        _cached_energy=node._cached_energy,
        _cached_gradient=node._cached_gradient,
        _cached_result=node._cached_result,
    )


def annotate_pot_with_queue_results(
    pot: Pot,
    queue_fp: Path,
    charge: int = 0,
    multiplicity: int = 1,
    maximum_barrier_height: float = 1000.0,
) -> Pot:
    chains_by_edge = load_completed_queue_chains(
        queue_fp=queue_fp,
        charge=charge,
        multiplicity=multiplicity,
    )

    for (source_node, target_node), chains in chains_by_edge.items():
        source_graph = pot.graph.nodes[source_node].get("molecule")
        target_graph = pot.graph.nodes[target_node].get("molecule")
        for chain in chains:
            if len(chain.nodes) == 0:
                continue
            if source_graph is not None and isinstance(chain.nodes[0], StructureNode):
                chain.nodes[0] = _trim_balanced_endpoint(chain.nodes[0], source_graph)
                chain.nodes[0].has_molecular_graph = True
            if target_graph is not None and isinstance(chain.nodes[-1], StructureNode):
                chain.nodes[-1] = _trim_balanced_endpoint(chain.nodes[-1], target_graph)
                chain.nodes[-1].has_molecular_graph = True

    return annotate_pot_with_neb_results(
        pot=pot,
        chains_by_edge=chains_by_edge,
        maximum_barrier_height=maximum_barrier_height,
    )
