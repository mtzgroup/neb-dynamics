import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.scripts import main_cli


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def _node_at_x(x: float) -> StructureNode:
    node = StructureNode(structure=_structure_at_x(x))
    node._cached_energy = float(x)
    node._cached_gradient = np.zeros_like(node.coords)
    return node


def _chain_from_xs(xs: list[float], params: ChainInputs) -> Chain:
    return Chain.model_validate(
        {"nodes": [_node_at_x(x) for x in xs], "parameters": params}
    )


class _FakeNEB:
    def __init__(self, chain: Chain):
        self.chain_trajectory = [chain]
        self.optimized = chain
        self.grad_calls_made = 0

    def write_to_disk(self, fp, write_history=True, write_qcio=False):
        self.chain_trajectory[-1].write_to_disk(fp)


def _history_from_segments(
    segments: list[tuple[float, float]], params: ChainInputs
) -> TreeNode:
    root_chain = _chain_from_xs([segments[0][0], segments[-1][-1]], params)
    root = TreeNode(data=_FakeNEB(root_chain), children=[], index=0)
    for i, (start, end) in enumerate(segments, start=1):
        leaf_chain = _chain_from_xs([start, end], params)
        root.children.append(TreeNode(data=_FakeNEB(leaf_chain), children=[], index=i))
    return root


class _FakePot:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node(0, td=_node_at_x(0.0))
        self.graph.add_node(1, td=_node_at_x(4.0))
        self.graph.add_edge(0, 1, barrier=1.0, list_of_nebs=[_chain_from_xs([0.0, 4.0], ChainInputs())])

    def write_to_disk(self, fp):
        fp.write_text(
            json.dumps(
                {
                    "nodes": {
                        str(node): {
                            "root": bool(data.get("root")),
                            "requested_target": bool(data.get("requested_target")),
                        }
                        for node, data in self.graph.nodes(data=True)
                    }
                }
            )
        )


class _RegularFakeMSMEP:
    def __init__(self, inputs):
        self.inputs = inputs
        self.recursive_calls = 0
        self.regular_calls = 0

    def run_recursive_minimize(self, input_chain: Chain):
        self.recursive_calls += 1
        return _history_from_segments([(0.0, 1.0), (1.0, 2.0)], self.inputs.chain_inputs)

    def run_minimize_chain(self, input_chain: Chain):
        self.regular_calls += 1
        chain = _chain_from_xs([0.0, 1.0, 2.0], self.inputs.chain_inputs)
        neb = _FakeNEB(chain)
        neb.geom_grad_calls_made = 0
        return neb, SimpleNamespace(is_elem_step=True)


def test_run_recursive_network_splits_enqueues_intermediate_targets(monkeypatch, tmp_path):
    params = ChainInputs()
    expensive_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )

    histories = {
        (0.0, 4.0): _history_from_segments(
            [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)], params
        ),
        (1.0, 4.0): _history_from_segments([(1.0, 5.0), (5.0, 4.0)], params),
        (2.0, 4.0): _history_from_segments([(2.0, 4.0)], params),
    }
    calls: list[tuple[float, float]] = []
    built_dirs: list[str] = []

    class _FakeMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_recursive_minimize(self, input_chain: Chain):
            pair = (
                float(input_chain[0].coords[1][0]),
                float(input_chain[-1].coords[1][0]),
            )
            calls.append(pair)
            return histories[pair]

    class _FakeNetworkBuilder:
        def __init__(self, data_dir, start, end, network_inputs, chain_inputs):
            self.data_dir = data_dir
            self.msmep_data_dir = None

        def create_rxn_network(self, file_pattern="*_msmep"):
            nonlocal built_dirs
            built_dirs = sorted(
                p.name for p in self.msmep_data_dir.glob(file_pattern) if p.is_dir()
            )
            return _FakePot()

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: expensive_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(main_cli, "NetworkBuilder", _FakeNetworkBuilder)
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(4.0)],
    )

    monkeypatch.chdir(tmp_path)
    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=True,
        network_splits=True,
        name="rgs",
    )

    assert calls == [(0.0, 4.0), (1.0, 4.0), (2.0, 4.0)]
    assert built_dirs == ["request_0_msmep", "request_1_msmep", "request_2_msmep"]

    manifest = json.loads(
        (tmp_path / "rgs_network_splits" / "rgs_request_manifest.json").read_text()
    )
    assert manifest["run_state"] == "completed"
    assert manifest["total_requests"] == 3
    assert manifest["counts"]["completed"] == 3
    assert [row["request_id"] for row in manifest["requests"]] == [0, 1, 2]
    assert [row["status"] for row in manifest["requests"]] == [
        "completed",
        "completed",
        "completed",
    ]

    network_dump = json.loads(
        (tmp_path / "rgs_network_splits" / "rgs_network.json").read_text()
    )
    assert network_dump["nodes"]["0"]["root"] is True
    assert network_dump["nodes"]["1"]["requested_target"] is True


def test_run_network_splits_forces_recursive_mode(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _RegularFakeMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "NetworkBuilder", lambda *args, **kwargs: SimpleNamespace(msmep_data_dir=None, create_rxn_network=lambda file_pattern="*_msmep": _FakePot()))
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)])
    monkeypatch.setattr(main_cli, "is_identical", lambda a, b, **kwargs: np.allclose(a.coords, b.coords))
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        network_splits=True,
        name="forced",
    )

    assert runner.recursive_calls == 1
    assert runner.regular_calls == 0
    status_payload = json.loads((tmp_path / "forced_status.json").read_text())
    assert status_payload["recursive"] is True
    assert status_payload["network_splits"] is True


def test_nonrecursive_run_writes_status_snapshot(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="FNEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _RegularFakeMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)])
    monkeypatch.setattr(main_cli, "_write_neb_results_with_history", lambda n, filename: filename.write_text("ok") or True)
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        network_splits=False,
        name="plain",
    )

    snapshot = main_cli._load_status_snapshot(str(tmp_path / "plain.xyz"))
    assert snapshot["run_status"]["recursive"] is False
    assert snapshot["run_status"]["network_splits"] is False
    assert snapshot["run_status"]["path_min_method"] == "FNEB"
    assert snapshot["run_status"]["run_state"] == "completed"


def test_load_status_snapshot_prefers_run_status_and_manifest(tmp_path):
    status_fp = tmp_path / "rgs_status.json"
    manifest_fp = tmp_path / "rgs_network_splits" / "rgs_request_manifest.json"
    manifest_fp.parent.mkdir()

    manifest_fp.write_text(
        json.dumps(
            {
                "base_name": "rgs",
                "run_state": "running",
                "current_request_id": 2,
                "total_requests": 4,
                "counts": {"completed": 1, "queued": 2, "running": 1},
                "requests": [],
                "network_summary": {"node_count": 3, "edge_count": 2, "edges": [["0", "1"], ["1", "2"]]},
            }
        )
    )
    status_fp.write_text(
        json.dumps(
            {
                "base_name": "rgs",
                "run_state": "running",
                "phase": "network_splits",
                "manifest_path": str(manifest_fp),
            }
        )
    )

    snapshot = main_cli._load_status_snapshot(str(status_fp))
    assert snapshot["artifact_kind"] == "run_status"
    assert snapshot["run_status"]["phase"] == "network_splits"
    assert snapshot["manifest"]["current_request_id"] == 2


def test_load_status_snapshot_resolves_xyz_to_status_files(tmp_path):
    xyz_fp = tmp_path / "rgs.xyz"
    xyz_fp.write_text("dummy")
    status_fp = tmp_path / "rgs_status.json"
    status_fp.write_text(
        json.dumps(
            {
                "base_name": "rgs",
                "run_state": "completed",
                "phase": "complete",
            }
        )
    )

    snapshot = main_cli._load_status_snapshot(str(xyz_fp))
    assert snapshot["artifact_kind"] == "run_status"
    assert snapshot["run_status"]["run_state"] == "completed"


def test_load_status_snapshot_resolves_missing_output_target_to_status_file(tmp_path):
    status_fp = tmp_path / "rgs_status.json"
    status_fp.write_text(
        json.dumps(
            {
                "base_name": "rgs",
                "run_state": "running",
                "phase": "initial_recursive_request",
            }
        )
    )

    snapshot = main_cli._load_status_snapshot(str(tmp_path / "rgs.xyz"))
    assert snapshot["artifact_kind"] == "run_status"
    assert snapshot["run_status"]["phase"] == "initial_recursive_request"
