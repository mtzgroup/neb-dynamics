from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from neb_dynamics.mepd_drive import (
    _build_neb_live_payload,
    _build_drive_payload,
    _initialize_workspace_job,
    _load_existing_workspace_job,
    _merge_drive_pot,
    _optimize_selected_nodes,
    _parse_xyz_text_to_structure,
    MepdDriveServer,
)
from neb_dynamics.scripts import main_cli
from neb_dynamics.scripts.progress import ProgressPrinter
from neb_dynamics.retropaths_queue import _run_single_item_worker


def test_parse_xyz_text_to_structure_reads_single_frame():
    structure = _parse_xyz_text_to_structure(
        """2
comment
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""
    )

    assert list(structure.symbols) == ["H", "H"]
    assert len(structure.geometry) == 2


def test_merge_drive_pot_overlays_annotated_edges(monkeypatch, tmp_path):
    base = SimpleNamespace(graph=nx.DiGraph())
    base.graph.add_node(0, molecule="root")
    base.graph.add_node(1, molecule="target")
    base.graph.add_edge(0, 1, reaction="base")

    annotated = SimpleNamespace(graph=nx.DiGraph())
    annotated.graph.add_node(0, molecule="root")
    annotated.graph.add_node(2, molecule="intermediate", generated_by="recursive_msmep")
    annotated.graph.add_edge(0, 2, reaction="step-1", list_of_nebs=["chain-a"])

    workspace = SimpleNamespace(neb_pot_fp=tmp_path / "neb_pot.json")

    monkeypatch.setattr("neb_dynamics.mepd_drive.Pot.read_from_disk", lambda _fp: base)
    monkeypatch.setattr("neb_dynamics.mepd_drive.load_partial_annotated_pot", lambda _workspace: annotated)

    merged = _merge_drive_pot(workspace)

    assert merged.graph.has_edge(0, 1)
    assert merged.graph.has_edge(0, 2)
    assert merged.graph.nodes[2]["generated_by"] == "recursive_msmep"


def test_drive_command_launches_server(monkeypatch, tmp_path):
    calls = {}

    class _FakeServer:
        server_address = ("127.0.0.1", 48123)

        def serve_forever(self):
            calls["served"] = True

        def shutdown(self):
            calls["shutdown"] = True

        def server_close(self):
            calls["closed"] = True

    monkeypatch.setattr(
        main_cli,
        "launch_mepd_drive",
        lambda **kwargs: calls.update(kwargs) or _FakeServer(),
    )

    main_cli.drive(
        inputs=str(tmp_path / "inputs.toml"),
        reactions_fp=str(tmp_path / "reactions.p"),
        directory=str(tmp_path / "drive"),
        no_open=True,
    )

    assert Path(calls["inputs_fp"]).name == "inputs.toml"
    assert calls["open_browser"] is False
    assert calls["served"] is True
    assert calls["shutdown"] is True
    assert calls["closed"] is True


def test_format_drive_access_panel_includes_ssh_tunnel():
    panel = main_cli._format_drive_access_panel(
        actual_host="127.0.0.1",
        actual_port=48123,
        ssh_login="jane@cluster",
        local_port=9000,
    )

    rendered = panel.renderable
    assert "ssh -N -L 9000:127.0.0.1:48123 jane@cluster" in rendered
    assert "http://127.0.0.1:9000/" in rendered


def test_initialize_workspace_job_clears_existing_workspace(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "drive-run"
    workspace_dir.mkdir()
    stale = workspace_dir / "retropaths_pot.json"
    stale.write_text("stale")

    created = {}

    workspace = SimpleNamespace(
        workdir=str(workspace_dir),
        run_name="drive-run",
        root_smiles="C=C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
        reactions_fp="",
        timeout_seconds=30,
        max_nodes=40,
        max_depth=4,
        max_parallel_nebs=1,
    )

    def _fake_create_workspace(**kwargs):
        created.update(kwargs)
        Path(kwargs["directory"]).mkdir(parents=True, exist_ok=True)
        return workspace

    monkeypatch.setattr("neb_dynamics.mepd_drive.create_workspace", _fake_create_workspace)
    monkeypatch.setattr("neb_dynamics.mepd_drive.prepare_neb_workspace", lambda _workspace: None)

    result = _initialize_workspace_job(
        reactant={"smiles": "C=C"},
        product=None,
        run_name="drive-run",
        workspace_dir=str(workspace_dir),
        inputs_fp=str(tmp_path / "inputs.toml"),
        reactions_fp=None,
        timeout_seconds=30,
        max_nodes=40,
        max_depth=4,
        max_parallel_nebs=1,
    )

    assert created["directory"] == workspace_dir.resolve()
    assert stale.exists() is False
    assert result["workspace"]["root_smiles"] == "C=C"


def test_load_existing_workspace_job_reads_workspace(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "drive-run"
    workspace_dir.mkdir()
    workspace_json = workspace_dir / "workspace.json"
    workspace_json.write_text(
        """
{
  "workdir": "%s",
  "run_name": "drive-run",
  "root_smiles": "C=C",
  "environment_smiles": "",
  "inputs_fp": "%s",
  "reactions_fp": "",
  "timeout_seconds": 30,
  "max_nodes": 40,
  "max_depth": 4,
  "max_parallel_nebs": 1
}
"""
        % (workspace_dir, tmp_path / "inputs.toml")
    )
    (workspace_dir / "neb_pot.json").write_text("{}")
    (workspace_dir / "neb_queue.json").write_text('{"items": [], "attempted_pairs": {}, "version": 1}')
    (workspace_dir / "retropaths_pot.json").write_text("{}")

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_partial_annotated_pot", lambda workspace: None)
    result = _load_existing_workspace_job(str(workspace_dir))

    assert result["workspace"]["run_name"] == "drive-run"
    assert result["reactant"]["smiles"] == "C=C"


def test_load_existing_workspace_job_rebuilds_annotated_overlay(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "drive-run"
    workspace_dir.mkdir()
    workspace_json = workspace_dir / "workspace.json"
    workspace_json.write_text(
        """
{
  "workdir": "%s",
  "run_name": "drive-run",
  "root_smiles": "C=C",
  "environment_smiles": "",
  "inputs_fp": "%s",
  "reactions_fp": "",
  "timeout_seconds": 30,
  "max_nodes": 40,
  "max_depth": 4,
  "max_parallel_nebs": 1
}
"""
        % (workspace_dir, tmp_path / "inputs.toml")
    )
    (workspace_dir / "neb_pot.json").write_text("{}")
    (workspace_dir / "neb_queue.json").write_text('{"items": [], "attempted_pairs": {}, "version": 1}')
    (workspace_dir / "retropaths_pot.json").write_text("{}")

    calls = []
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.load_partial_annotated_pot",
        lambda workspace: calls.append(Path(workspace.workdir)),
    )

    _load_existing_workspace_job(str(workspace_dir))

    assert calls == [workspace_dir]


def test_drive_server_rejects_overlapping_actions():
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()

    class _PendingFuture:
        def done(self):
            return False

    server.runtime = SimpleNamespace(future=_PendingFuture(), busy_label="Running geometry minimizations...")

    try:
        server._assert_idle()
    except ValueError as exc:
        assert "Another drive action is still running" in str(exc)
    else:
        raise AssertionError("Expected _assert_idle to reject overlapping actions.")


def test_build_drive_payload_hides_unstarted_queue_items(monkeypatch, tmp_path):
    queue = SimpleNamespace(
        items=[
            SimpleNamespace(
                source_node=0,
                target_node=1,
                status="pending",
                started_at=None,
                finished_at=None,
                error=None,
            ),
            SimpleNamespace(
                source_node=1,
                target_node=2,
                status="completed",
                started_at="2026-03-18T12:00:00",
                finished_at="2026-03-18T12:05:00",
                error=None,
            ),
        ]
    )
    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0, molecule="C")
    pot.graph.add_node(1, molecule="CC", endpoint_optimized=True)
    pot.graph.add_node(2, molecule="CCC")
    pot.graph.add_edge(0, 1, reaction="r1")
    pot.graph.add_edge(1, 2, reaction="r2", list_of_nebs=["chain"])
    workspace = SimpleNamespace(
        queue_fp=tmp_path / "queue.json",
        neb_pot_fp=tmp_path / "neb_pot.json",
        workdir=str(tmp_path),
        run_name="drive",
        root_smiles="C",
        environment_smiles="",
        reactions_path=tmp_path / "reactions.p",
    )
    retropaths_pot = SimpleNamespace(graph=nx.DiGraph())
    retropaths_pot.graph.add_node(0)
    retropaths_pot.graph.add_edge(0, 1)

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_retropaths_pot", lambda _workspace: retropaths_pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive.RetropathsNEBQueue.read_from_disk", lambda _fp: queue)
    monkeypatch.setattr("neb_dynamics.mepd_drive._merge_drive_pot", lambda _workspace: pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_edge_visualizations", lambda workspace, pot: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._load_template_payloads", lambda workspace: {})
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_network_explorer_payload",
        lambda graph, template_payloads=None, edge_visualizations=None: {
            "nodes": [{"id": 0, "label": "C"}, {"id": 1, "label": "CC"}, {"id": 2, "label": "CCC"}],
            "edges": [{"source": 0, "target": 1, "reaction": "r1"}, {"source": 1, "target": 2, "reaction": "r2"}],
        },
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._node_structure_payload", lambda _attrs: None)

    payload = _build_drive_payload(
        workspace,
        active_job_label="Running geometry minimizations...",
        active_action={"type": "minimize", "status": "running", "label": "Optimizing geometry 1/1: node 0"},
    )

    assert payload["queue"]["items"] == 1
    assert payload["queue"]["completed"] == 1
    assert payload["queue"]["pending"] == 0
    assert payload["queue"]["running"] == 1


def test_snapshot_includes_active_action(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    class _PendingFuture:
        def done(self):
            return False

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast_neb",
        lambda workspace, product_smiles="", active_job_label="", active_action=None: {
            "queue": {"running": 1},
            "workspace": {"run_name": "drive"},
            "network": {},
            "echo_active_action": active_action,
        },
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Running autosplitting NEB for 1 -> 0",
        last_error="",
        future=_PendingFuture(),
        busy_label="Running autosplitting NEB for 1 -> 0",
        active_action={"type": "neb", "status": "running", "label": "Running autosplitting NEB for 1 -> 0", "source_node": 1, "target_node": 0},
    )

    snapshot = server.snapshot()

    assert snapshot["busy"] is True
    assert snapshot["active_action"]["type"] == "neb"
    assert snapshot["drive"]["echo_active_action"]["label"] == "Running autosplitting NEB for 1 -> 0"
    assert snapshot["live_activity"]["type"] == "neb"


def test_snapshot_uses_fast_builder_for_running_minimization(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    class _PendingFuture:
        def done(self):
            return False

    called = {}

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast",
        lambda workspace, product_smiles="", active_job_label="", active_action=None: called.update({"fast": True, "active_action": active_action}) or {"queue": {}, "workspace": {}, "network": {}},
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("slow builder should not be used")),
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Optimizing geometry 1/1: node 0",
        last_error="",
        future=_PendingFuture(),
        busy_label="Optimizing geometry 1/1: node 0",
        active_action={"type": "minimize", "status": "running", "label": "Optimizing geometry 1/1: node 0"},
    )

    snapshot = server.snapshot()

    assert snapshot["busy"] is True
    assert called["fast"] is True
    assert called["active_action"]["type"] == "minimize"


def test_snapshot_uses_fast_builder_for_running_neb(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    class _PendingFuture:
        def done(self):
            return False

    called = {}

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast_neb",
        lambda workspace, product_smiles="", active_job_label="", active_action=None: called.update({"fast_neb": True, "active_action": active_action}) or {"queue": {}, "workspace": {}, "network": {}},
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("slow builder should not be used")),
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("minimize fast builder should not be used")),
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Running autosplitting NEB for 1 -> 0",
        last_error="",
        future=_PendingFuture(),
        busy_label="Running autosplitting NEB for 1 -> 0",
        active_action={"type": "neb", "status": "running", "label": "Running autosplitting NEB for 1 -> 0", "source_node": 1, "target_node": 0},
    )

    snapshot = server.snapshot()

    assert snapshot["busy"] is True
    assert called["fast_neb"] is True
    assert called["active_action"]["type"] == "neb"


def test_snapshot_includes_live_activity_for_running_neb(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    class _PendingFuture:
        def done(self):
            return False

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast_neb",
        lambda workspace, product_smiles="", active_job_label="", active_action=None: {"queue": {}, "workspace": {}, "network": {}},
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._read_chain_payload",
        lambda _fp: {
            "ascii_plot": "plot",
            "caption": "step 4",
            "plot": {"x": [0.0, 1.0], "y": [0.0, 2.0]},
            "history": [{"x": [0.0, 1.0], "y": [0.5, 2.5], "caption": "step 3"}],
        },
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Running autosplitting NEB for 1 -> 0",
        last_error="",
        future=_PendingFuture(),
        busy_label="Running autosplitting NEB for 1 -> 0",
        active_action={"type": "neb", "status": "running", "label": "Running autosplitting NEB for 1 -> 0", "source_node": 1, "target_node": 0, "chain_fp": "dummy.json"},
    )

    snapshot = server.snapshot()

    assert snapshot["live_activity"]["type"] == "neb"
    assert snapshot["live_activity"]["plot"]["x"] == [0.0, 1.0]
    assert snapshot["live_activity"]["history"][0]["caption"] == "step 3"


def test_snapshot_includes_live_activity_for_running_minimization(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    class _PendingFuture:
        def done(self):
            return False

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast",
        lambda workspace, product_smiles="", active_job_label="", active_action=None: {"queue": {}, "workspace": {}, "network": {}},
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Optimizing geometry 1/2: node 0",
        last_error="",
        future=_PendingFuture(),
        busy_label="Optimizing geometry 1/2: node 0",
        active_action={
            "type": "minimize",
            "status": "running",
            "label": "Optimizing geometry 1/2: node 0",
            "jobs": [
                {"node_id": 0, "status": "running"},
                {"node_id": 1, "status": "pending"},
            ],
        },
    )

    snapshot = server.snapshot()

    assert snapshot["live_activity"]["type"] == "minimize"
    assert snapshot["live_activity"]["jobs"][0]["status"] == "running"


def test_optimize_selected_nodes_reports_progress_and_persists_each_node(monkeypatch, tmp_path):
    writes = []
    progress_messages = []
    node_updates = []

    class _FakePot:
        def __init__(self):
            self.graph = nx.DiGraph()
            self.graph.add_node(0, td="node-0")
            self.graph.add_node(1, td="node-1")

        def write_to_disk(self, fp):
            writes.append(Path(fp))

    fake_pot = _FakePot()
    workspace = SimpleNamespace(inputs_fp=str(tmp_path / "inputs.toml"), neb_pot_fp=tmp_path / "neb_pot.json")

    monkeypatch.setattr("neb_dynamics.mepd_drive.RunInputs.open", lambda _fp: SimpleNamespace(engine="fake"))
    monkeypatch.setattr("neb_dynamics.mepd_drive.Pot.read_from_disk", lambda _fp: fake_pot)
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._run_geometry_optimization_with_trajectory",
        lambda node, run_inputs: (f"optimized-{node}", None, [SimpleNamespace(energy=0.0), SimpleNamespace(energy=1.0)]),
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._persist_endpoint_optimization_result",
        lambda workspace, node_index, optimized_td: None,
    )

    result = _optimize_selected_nodes(
        workspace,
        None,
        progress=progress_messages.append,
        on_node_update=node_updates.append,
    )

    assert result["node_indices"] == [0, 1]
    assert progress_messages[0] == "Optimizing geometry 1/2: node 0"
    assert progress_messages[1] == "Finished geometry 1/2: node 0"
    assert progress_messages[2] == "Optimizing geometry 2/2: node 1"
    assert progress_messages[3] == "Finished geometry 2/2: node 1"
    assert len(writes) == 2
    assert fake_pot.graph.nodes[0]["td"] == "optimized-node-0"
    assert fake_pot.graph.nodes[1]["td"] == "optimized-node-1"
    assert node_updates[0]["status"] == "running"
    assert node_updates[1]["status"] == "completed"
    assert node_updates[1]["plot"]["x"] == [0, 1]
    assert node_updates[1]["plot"]["y"] == [0.0, 1.0]


def test_progress_printer_writes_live_chain_payload_file(monkeypatch, tmp_path):
    chain_fp = tmp_path / "live_chain.json"
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_JSON", str(chain_fp))
    printer = ProgressPrinter(use_rich=False)
    chain = SimpleNamespace(
        integrated_path_length=[0.0, 1.0, 2.0],
        energies_kcalmol=[0.0, 3.0, 1.0],
    )
    monkeypatch.setattr(
        "neb_dynamics.scripts.progress._endpoint_smiles_for_chain",
        lambda _chain: ("C=C", "CCO"),
    )

    printer.record_chain_plot(chain, "step 5")
    printer.print_chain_ascii("chain ascii", "step 5", force_update=True)

    payload = __import__("json").loads(chain_fp.read_text())
    assert payload["caption"] == "step 5"
    assert payload["plot"]["x"] == [0.0, 1.0, 2.0]
    assert payload["plot"]["y"] == [0.0, 3.0, 1.0]
    assert payload["plot"]["reactant_smiles"] == "C=C"
    assert payload["plot"]["product_smiles"] == "CCO"
    assert payload["ascii_plot"] == "chain ascii"


def test_run_single_item_worker_creates_queue_output_parent(monkeypatch, tmp_path):
    writes = {}

    class _FakeHistory:
        output_chain = SimpleNamespace(write_to_disk=lambda fp: writes.setdefault("chain", Path(fp)))

        def write_to_disk(self, folder_name, write_qcio=False):
            folder = Path(folder_name)
            folder.mkdir(parents=True, exist_ok=True)
            writes["history"] = folder

    class _FakeMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_recursive_minimize(self, pair):
            return _FakeHistory()

    monkeypatch.setattr("neb_dynamics.retropaths_queue.MSMEP", _FakeMSMEP)

    result = _run_single_item_worker(
        pair="pair",
        run_inputs=SimpleNamespace(),
        result_dir=str(tmp_path / "queue_runs" / "pair_1_0_msmep"),
        output_chain_xyz=str(tmp_path / "queue_runs" / "pair_1_0.xyz"),
    )

    assert Path(result["result_dir"]).name == "pair_1_0_msmep"
    assert writes["history"].name == "pair_1_0_msmep"
    assert writes["chain"].parent.name == "queue_runs"


def test_build_neb_live_payload_prefers_live_chain_endpoints(monkeypatch, tmp_path):
    chain_fp = tmp_path / "drive_neb_1_0.chain.json"
    chain_fp.write_text(
        __import__("json").dumps(
            {
                "plot": {
                    "x": [0.0, 1.0],
                    "y": [0.0, 2.0],
                    "reactant_smiles": "C=C.O",
                    "product_smiles": "CCO",
                },
                "history": [],
            }
        )
    )

    payload_calls = []

    def _fake_molecule_visual_payload(smiles):
        payload_calls.append(smiles)
        return {"smiles": smiles, "svg": f"<svg>{smiles}</svg>", "render_error": ""}

    monkeypatch.setattr("neb_dynamics.mepd_drive._molecule_visual_payload", _fake_molecule_visual_payload)

    live = _build_neb_live_payload(
        {
            "type": "neb",
            "chain_fp": str(chain_fp),
            "source_node": 1,
            "target_node": 0,
            "progress_fp": str(tmp_path / "progress.log"),
        },
        workspace=None,
    )

    assert payload_calls == ["C=C.O", "CCO"]
    assert live["reactant_structure"]["smiles"] == "C=C.O"
    assert live["product_structure"]["smiles"] == "CCO"


def test_submit_minimize_surfaces_requested_node_error(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server.runtime = SimpleNamespace(
        workspace=SimpleNamespace(neb_pot_fp=tmp_path / "neb_pot.json", inputs_fp=tmp_path / "inputs.toml"),
        last_error="",
        last_message="Idle.",
        future=None,
        busy_label="",
    )

    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0, td=None)
    monkeypatch.setattr("neb_dynamics.mepd_drive.Pot.read_from_disk", lambda _fp: pot)

    try:
        server.submit_minimize([0])
        assert False, "submit_minimize should have raised for node 0 without geometry"
    except ValueError as exc:
        assert "Node 0 has no geometry attached" in str(exc)


def test_submit_run_neb_surfaces_non_queueable_edge(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server.runtime = SimpleNamespace(
        workspace=SimpleNamespace(
            neb_pot_fp=tmp_path / "neb_pot.json",
            queue_fp=tmp_path / "neb_queue.json",
            directory=tmp_path,
        ),
        last_error="",
        last_message="Idle.",
        future=None,
        busy_label="",
    )

    monkeypatch.setattr("neb_dynamics.mepd_drive.Pot.read_from_disk", lambda _fp: SimpleNamespace())
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.build_retropaths_neb_queue",
        lambda **kwargs: SimpleNamespace(
            recover_stale_running_items=lambda: False,
            write_to_disk=lambda _fp: None,
            find_item=lambda source, target: SimpleNamespace(
                source_node=source,
                target_node=target,
                status="completed",
                error=None,
            ),
        ),
    )

    try:
        server.submit_run_neb(source_node=1, target_node=0)
        assert False, "submit_run_neb should have raised for completed edge"
    except ValueError as exc:
        assert "already has a completed NEB result" in str(exc)
