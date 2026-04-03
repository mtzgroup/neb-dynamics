import concurrent.futures
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import networkx as nx
import numpy as np
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.molecule import Molecule
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.pot import Pot
from neb_dynamics.retropaths_compat import structure_node_from_graph_like_molecule
from neb_dynamics.retropaths_queue import (
    NEBQueueItem,
    RetropathsNEBQueue,
    _optimize_single_node,
    build_balanced_endpoints,
    build_retropaths_neb_queue,
    load_completed_queue_chains,
    pair_is_direct_neb_compatible,
    pair_attempt_key,
    run_retropaths_neb_queue,
)


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def _node_at_x(x: float, energy: float = 0.0) -> StructureNode:
    node = StructureNode(structure=_structure_at_x(x))
    node.has_molecular_graph = False
    node.graph = None
    node._cached_energy = float(energy)
    node._cached_gradient = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return node


def _test_pot(nodes_by_index: dict[int, StructureNode], edges: list[tuple[int, int]]) -> Pot:
    pot = Pot(root=Molecule(), target=Molecule())
    pot.graph = nx.DiGraph()
    for node_index, td in nodes_by_index.items():
        pot.graph.add_node(node_index, td=td)
    for source_node, target_node in edges:
        pot.graph.add_edge(source_node, target_node, reaction=f"{source_node}->{target_node}")
    return pot


def _queue_state_with_attempted_pair(attempt_key: str) -> RetropathsNEBQueue:
    return RetropathsNEBQueue(
        attempted_pairs={
            attempt_key: {
                "job_id": "prior",
                "source_node": 99,
                "target_node": 100,
                "status": "failed",
                "started_at": "2026-03-11T00:00:00+00:00",
                "finished_at": "2026-03-11T00:01:00+00:00",
            }
        }
    )


def test_build_retropaths_neb_queue_marks_prior_attempts_as_skipped(tmp_path):
    nodes = {0: _node_at_x(0.0), 1: _node_at_x(1.0), 2: _node_at_x(2.0)}
    pot = _test_pot(nodes, [(1, 0), (2, 0)])
    queue_fp = tmp_path / "queue.json"

    prior_attempt_key = pair_attempt_key(nodes[1], nodes[0])
    _queue_state_with_attempted_pair(prior_attempt_key).write_to_disk(queue_fp)

    queue = build_retropaths_neb_queue(pot=pot, queue_fp=queue_fp)

    assert [item.status for item in queue.items] == ["skipped_attempted", "pending"]


def test_build_retropaths_neb_queue_refreshes_legacy_atom_mismatch_failures(tmp_path):
    source = structure_node_from_graph_like_molecule(Molecule.from_smiles("CCO"))
    target = structure_node_from_graph_like_molecule(Molecule.from_smiles("C=C"))
    pot = _test_pot({0: target, 1: source}, [(1, 0)])
    pot.graph.nodes[0]["environment"] = Molecule.from_smiles("O")
    queue_fp = tmp_path / "queue.json"

    RetropathsNEBQueue(
        items=[
            NEBQueueItem(
                job_id="1->0",
                source_node=1,
                target_node=0,
                attempt_key="legacy",
                status="failed",
                error="ValidationError: setting an array element with a sequence",
            )
        ],
        attempted_pairs={"legacy": {"status": "failed"}},
    ).write_to_disk(queue_fp)

    queue = build_retropaths_neb_queue(pot=pot, queue_fp=queue_fp)

    assert queue.items[0].status == "pending"
    assert queue.items[0].attempt_key != "legacy"
    assert "legacy" not in queue.attempted_pairs


def test_build_retropaths_neb_queue_refreshes_stale_attempt_signature_after_endpoint_changes(tmp_path):
    original_nodes = {0: _node_at_x(0.0), 1: _node_at_x(1.0)}
    original_attempt_key = pair_attempt_key(original_nodes[1], original_nodes[0])
    queue_fp = tmp_path / "queue.json"

    RetropathsNEBQueue(
        items=[
            NEBQueueItem(
                job_id="1->0",
                source_node=1,
                target_node=0,
                attempt_key=original_attempt_key,
                status="skipped_attempted",
            )
        ],
        attempted_pairs={
            original_attempt_key: {
                "job_id": "prior",
                "source_node": 1,
                "target_node": 0,
                "status": "completed",
            }
        },
    ).write_to_disk(queue_fp)

    optimized_nodes = {0: _node_at_x(0.25), 1: _node_at_x(1.5)}
    pot = _test_pot(optimized_nodes, [(1, 0)])

    queue = build_retropaths_neb_queue(pot=pot, queue_fp=queue_fp)

    assert queue.items[0].attempt_key != original_attempt_key
    assert queue.items[0].status == "pending"
    assert original_attempt_key in queue.attempted_pairs


def test_load_completed_queue_chains_includes_completed_attempted_pairs(tmp_path, monkeypatch):
    queue_fp = tmp_path / "queue.json"
    queue = RetropathsNEBQueue(
        items=[
            NEBQueueItem(
                job_id="1->0",
                source_node=1,
                target_node=0,
                attempt_key="new-attempt",
                status="pending",
            )
        ],
        attempted_pairs={
            "old-attempt": {
                "job_id": "1->0-old",
                "source_node": 1,
                "target_node": 0,
                "status": "completed",
                "result_dir": str(tmp_path / "old_result"),
                "finished_at": "2026-04-03T07:00:00",
            }
        },
    )
    queue.write_to_disk(queue_fp)

    expected_chain = Chain.model_validate(
        {"nodes": [_node_at_x(0.0), _node_at_x(1.0)], "parameters": ChainInputs()}
    )

    monkeypatch.setattr(
        "neb_dynamics.retropaths_queue.TreeNode.read_from_disk",
        lambda folder_name, charge, multiplicity: SimpleNamespace(output_chain=expected_chain),
    )

    chains_by_edge = load_completed_queue_chains(queue_fp=queue_fp)

    assert (1, 0) in chains_by_edge
    assert chains_by_edge[(1, 0)][0] is expected_chain


def test_pair_is_direct_neb_compatible_rejects_atom_count_mismatch():
    compatible, reason = pair_is_direct_neb_compatible(_node_at_x(0.0), _energetic_like_three_atom_node())

    assert compatible is False
    assert "Atom count mismatch" in reason


def test_build_balanced_endpoints_adds_environment_fragment_to_smaller_side():
    source = structure_node_from_graph_like_molecule(Molecule.from_smiles("CCO"))
    target = structure_node_from_graph_like_molecule(Molecule.from_smiles("C=C"))
    environment = Molecule.from_smiles("O")

    balanced_source, balanced_target, reason = build_balanced_endpoints(
        source_td=source,
        target_td=target,
        environment=environment,
    )

    compatible, _ = pair_is_direct_neb_compatible(balanced_source, balanced_target)
    assert compatible is True
    assert reason is None
    assert len(balanced_target.structure.symbols) > len(target.structure.symbols)
    assert balanced_target.structure.extras["retropaths_original_atom_count"] == len(target.structure.symbols)


def test_build_balanced_endpoints_rebuilds_missing_geometry_from_graph():
    source_graph = Molecule.from_smiles("C=C.O")
    target_graph = Molecule.from_smiles("C=C.O")
    source = structure_node_from_graph_like_molecule(Molecule.from_smiles("C=C"))
    target = structure_node_from_graph_like_molecule(target_graph)

    balanced_source, balanced_target, reason = build_balanced_endpoints(
        source_td=source,
        target_td=target,
        environment=Molecule(),
        source_graph=source_graph,
        target_graph=target_graph,
    )

    compatible, _ = pair_is_direct_neb_compatible(balanced_source, balanced_target)
    assert compatible is True
    assert reason is None
    assert len(balanced_source.structure.symbols) == len(balanced_target.structure.symbols)


class _FakeHistory:
    def __init__(self, output_chain: Chain):
        self.output_chain = output_chain

    def write_to_disk(self, folder_name):
        folder_name.mkdir(parents=True, exist_ok=True)
        (folder_name / "adj_matrix.txt").write_text("1\n")
        self.output_chain.write_to_disk(folder_name / "node_0.xyz")


class _FakeMSMEP:
    calls: list[Chain] = []

    def __init__(self, inputs):
        self.inputs = inputs

    def run_recursive_minimize(self, chain: Chain):
        _FakeMSMEP.calls.append(chain.copy())
        return _FakeHistory(output_chain=chain.copy())


class _NoOpEngine:
    def compute_geometry_optimization(self, node, keywords=None):
        return [node.copy()]


def _fake_run_inputs():
    return SimpleNamespace(
        engine=_NoOpEngine(),
        chain_inputs=ChainInputs(),
        path_min_inputs=SimpleNamespace(do_elem_step_checks=False),
    )


def _energetic_like_three_atom_node() -> StructureNode:
    node = StructureNode(
        structure=Structure(
            geometry=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
            symbols=["H", "H", "H"],
            charge=0,
            multiplicity=1,
        )
    )
    node.has_molecular_graph = False
    node.graph = None
    node._cached_energy = 0.0
    node._cached_gradient = [[0.0, 0.0, 0.0]] * 3
    return node


def test_run_retropaths_neb_queue_uses_recursive_runner_and_persists_completion(
    monkeypatch, tmp_path
):
    nodes = {0: _node_at_x(0.0), 1: _node_at_x(1.0)}
    pot = _test_pot(nodes, [(1, 0)])
    queue_fp = tmp_path / "queue.json"
    output_dir = tmp_path / "out"

    monkeypatch.setattr("neb_dynamics.retropaths_queue.MSMEP", _FakeMSMEP)
    _FakeMSMEP.calls = []

    queue = run_retropaths_neb_queue(
        pot=pot,
        run_inputs=_fake_run_inputs(),
        queue_fp=queue_fp,
        output_dir=output_dir,
    )

    assert len(_FakeMSMEP.calls) == 1
    assert queue.items[0].status == "completed"
    assert queue.items[0].result_dir is not None
    assert queue.items[0].output_chain_xyz is not None
    assert queue.attempted_pairs[queue.items[0].attempt_key]["status"] == "completed"
    assert (output_dir / "pair_1_0_msmep" / "adj_matrix.txt").exists()
    assert (output_dir / "pair_1_0.xyz").exists()


def test_run_retropaths_neb_queue_skips_duplicate_pairs_before_execution(
    monkeypatch, tmp_path
):
    nodes = {0: _node_at_x(0.0), 1: _node_at_x(1.0), 2: _node_at_x(1.0)}
    pot = _test_pot(nodes, [(1, 0), (2, 0)])
    queue_fp = tmp_path / "queue.json"
    output_dir = tmp_path / "out"

    monkeypatch.setattr("neb_dynamics.retropaths_queue.MSMEP", _FakeMSMEP)
    _FakeMSMEP.calls = []

    queue = run_retropaths_neb_queue(
        pot=pot,
        run_inputs=_fake_run_inputs(),
        queue_fp=queue_fp,
        output_dir=output_dir,
    )

    assert len(_FakeMSMEP.calls) == 1
    assert queue.items[0].status == "completed"
    assert queue.items[1].status == "skipped_attempted"


def test_run_retropaths_neb_queue_skips_identical_endpoints_after_optimization(
    monkeypatch, tmp_path
):
    source = structure_node_from_graph_like_molecule(Molecule.from_smiles("CCO"))
    target = structure_node_from_graph_like_molecule(Molecule.from_smiles("COC"))
    pot = _test_pot({0: target, 1: source}, [(1, 0)])
    queue_fp = tmp_path / "queue.json"
    output_dir = tmp_path / "out"

    identical = structure_node_from_graph_like_molecule(Molecule.from_smiles("COC"))

    class _GraphCollapsingEngine:
        def compute_geometry_optimization(self, node, keywords=None):
            return [identical.copy()]

    monkeypatch.setattr("neb_dynamics.retropaths_queue.MSMEP", _FakeMSMEP)
    _FakeMSMEP.calls = []

    queue = run_retropaths_neb_queue(
        pot=pot,
        run_inputs=SimpleNamespace(
            engine=_GraphCollapsingEngine(),
            chain_inputs=ChainInputs(),
            path_min_inputs=SimpleNamespace(do_elem_step_checks=False, skip_identical_graphs=True),
        ),
        queue_fp=queue_fp,
        output_dir=output_dir,
        pot_fp=tmp_path / "neb_pot.json",
    )

    assert len(_FakeMSMEP.calls) == 0
    assert queue.items[0].status == "skipped_identical"
    assert "identical" in (queue.items[0].error or "").lower()


def test_optimize_single_node_refreshes_graph_from_optimized_structure():
    start = structure_node_from_graph_like_molecule(Molecule.from_smiles("CC"))
    optimized = structure_node_from_graph_like_molecule(Molecule.from_smiles("C=C"))

    class _ChangingEngine:
        def compute_geometry_optimization(self, node, keywords=None):
            return [optimized.copy()]

    out, error = _optimize_single_node(
        start,
        SimpleNamespace(engine=_ChangingEngine()),
    )

    assert error is None
    assert out.has_molecular_graph is True
    assert out.graph.is_bond_isomorphic_to(Molecule.from_smiles("C=C"))
    assert not out.graph.is_bond_isomorphic_to(start.graph)


def test_recover_stale_running_items_marks_interrupted_jobs_failed():
    item = NEBQueueItem(
        job_id="1->0",
        source_node=1,
        target_node=0,
        attempt_key="abc",
        status="running",
        started_at="2026-03-11T00:00:00+00:00",
    )
    queue = RetropathsNEBQueue(
        attempted_pairs={"abc": {"status": "running"}},
        items=[
            item
        ]
    )

    changed = queue.recover_stale_running_items()

    assert changed is True
    assert queue.items[0].status == "failed"
    assert queue.items[0].error is not None
    assert queue.attempted_pairs["abc"]["status"] == "failed"


def test_recover_stale_running_items_promotes_saved_result_tree(tmp_path, monkeypatch):
    result_dir = tmp_path / "queue_runs" / "pair_1_0_msmep"
    result_dir.mkdir(parents=True)
    output_chain = tmp_path / "queue_runs" / "pair_1_0.xyz"
    output_chain.write_text("2\n\nH 0 0 0\nH 1 0 0\n")

    item = NEBQueueItem(
        job_id="1->0",
        source_node=1,
        target_node=0,
        attempt_key="abc",
        status="running",
        started_at="2026-03-11T00:00:00+00:00",
    )
    queue = RetropathsNEBQueue(
        attempted_pairs={"abc": {"status": "running"}},
        items=[item],
    )

    monkeypatch.setattr(
        "neb_dynamics.retropaths_queue.TreeNode.read_from_disk",
        lambda folder_name, charge, multiplicity: SimpleNamespace(output_chain=object()),
    )

    changed = queue.recover_stale_running_items(output_dir=tmp_path / "queue_runs")

    assert changed is True
    assert queue.items[0].status == "completed"
    assert queue.items[0].result_dir == str(result_dir.resolve())
    assert queue.items[0].output_chain_xyz == str(output_chain.resolve())
    assert queue.attempted_pairs["abc"]["status"] == "completed"


def test_run_retropaths_neb_queue_supports_parallel_workers(
    monkeypatch, tmp_path
):
    nodes = {0: _node_at_x(0.0), 1: _node_at_x(1.0), 2: _node_at_x(2.0)}
    pot = _test_pot(nodes, [(1, 0), (2, 0)])
    queue_fp = tmp_path / "queue.json"
    output_dir = tmp_path / "out"

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def _fake_worker(pair, run_inputs, result_dir, output_chain_xyz):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(0.05)
            history = _FakeHistory(output_chain=pair.copy())
            history.write_to_disk(Path(result_dir))
            history.output_chain.write_to_disk(Path(output_chain_xyz))
            return {"result_dir": result_dir, "output_chain_xyz": output_chain_xyz}
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr("neb_dynamics.retropaths_queue._run_single_item_worker", _fake_worker)
    class _CompatThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, max_workers=None, mp_context=None):
            super().__init__(max_workers=max_workers)
    monkeypatch.setattr(
        "neb_dynamics.retropaths_queue.concurrent.futures.ProcessPoolExecutor",
        _CompatThreadPoolExecutor,
    )

    queue = run_retropaths_neb_queue(
        pot=pot,
        run_inputs=_fake_run_inputs(),
        queue_fp=queue_fp,
        output_dir=output_dir,
        max_parallel_nebs=2,
    )

    assert [item.status for item in queue.items] == ["completed", "completed"]
    assert state["max_active"] >= 2
