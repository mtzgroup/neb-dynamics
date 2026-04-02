from pathlib import Path
from types import SimpleNamespace

import networkx as nx
from qcio import Structure

from neb_dynamics.mepd_drive import (
    _build_growth_live_payload,
    _build_neb_live_payload,
    _bootstrap_product_endpoint,
    _build_drive_payload,
    _drive_network_version,
    _run_kmc_payload,
    _drive_html,
    _initialize_workspace_job,
    _load_existing_workspace_job,
    _merge_drive_pot,
    _optimize_selected_nodes,
    _parse_xyz_text_to_structure,
    _write_completed_queue_visualizations,
    MepdDriveServer,
)
from neb_dynamics.scripts import main_cli
from neb_dynamics.scripts.progress import ProgressPrinter
from neb_dynamics.retropaths_queue import _run_single_item_worker
from neb_dynamics.retropaths_workflow import RetropathsWorkspace, apply_reactions_to_node, run_nanoreactor_for_node
from neb_dynamics.molecule import Molecule
from neb_dynamics.pot import Pot
from neb_dynamics.retropaths_compat import structure_node_from_graph_like_molecule


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


def test_drive_command_passes_smiles_bootstrap_options(monkeypatch, tmp_path):
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
        smiles="C=C",
        product_smiles="CC",
        environment="O",
        charge=1,
        multiplicity=2,
        name="smiles-run",
        inputs=str(tmp_path / "inputs.toml"),
        directory=str(tmp_path / "drive"),
        no_open=True,
    )

    assert calls["smiles"] == "C=C"
    assert calls["product_smiles"] == "CC"
    assert calls["environment_smiles"] == "O"
    assert calls["charge"] == 1
    assert calls["multiplicity"] == 2
    assert calls["run_name"] == "smiles-run"
    assert Path(calls["inputs_fp"]).name == "inputs.toml"
    assert calls["served"] is True


def test_submit_apply_reactions_errors_when_retropaths_unavailable(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server.runtime = SimpleNamespace(
        workspace=RetropathsWorkspace(
            workdir=str(tmp_path),
            run_name="demo",
            root_smiles="C",
            environment_smiles="",
            inputs_fp=str(tmp_path / "inputs.toml"),
        ),
        reactant=None,
        product=None,
        last_message="",
        last_error="",
        future=None,
        busy_label="",
        active_action=None,
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.ensure_retropaths_available",
        lambda feature="This action": (_ for _ in ()).throw(
            RuntimeError("MEPD Drive reaction-template application (+) requires retropaths")
        ),
    )

    try:
        server.submit_apply_reactions(node_id=0)
        assert False, "Expected submit_apply_reactions to fail when retropaths is unavailable."
    except ValueError as exc:
        assert "requires retropaths" in str(exc)


def test_drive_command_loads_existing_workspace_without_inputs(monkeypatch, tmp_path):
    calls = {}
    workspace_dir = tmp_path / "existing-run"
    workspace_dir.mkdir()
    (workspace_dir / "workspace.json").write_text("{}")

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
        directory=str(workspace_dir),
        no_open=True,
    )

    assert calls["workspace_path"] == str(workspace_dir.resolve())
    assert calls["inputs_fp"] is None
    assert calls["served"] is True


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
    monkeypatch.setattr("neb_dynamics.mepd_drive._apply_bootstrap_species_overrides", lambda _workspace, reactant, product: None)

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


def test_initialize_workspace_job_returns_partial_workspace_on_growth_error(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "drive-run"
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
        workspace_fp=workspace_dir / "workspace.json",
        neb_pot_fp=workspace_dir / "neb_pot.json",
        queue_fp=workspace_dir / "neb_queue.json",
        retropaths_pot_fp=workspace_dir / "retropaths_pot.json",
    )

    def _fake_create_workspace(**kwargs):
        workspace_dir.mkdir(parents=True, exist_ok=True)
        workspace.workspace_fp.write_text("{}")
        return workspace

    def _fake_initialize(_workspace, progress_fp=None):
        workspace.retropaths_pot_fp.write_text("{}")
        workspace.neb_pot_fp.write_text("{}")
        workspace.queue_fp.write_text('{"items": [], "attempted_pairs": {}, "version": 1}')
        raise RuntimeError("Too many iterations")

    monkeypatch.setattr("neb_dynamics.mepd_drive.create_workspace", _fake_create_workspace)
    monkeypatch.setattr("neb_dynamics.mepd_drive.initialize_workspace_with_progress", _fake_initialize)
    monkeypatch.setattr("neb_dynamics.mepd_drive._bootstrap_product_endpoint", lambda _workspace, _product: None)

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
        progress_fp=str(workspace_dir / "growth.json"),
    )

    assert result["workspace"]["root_smiles"] == "C=C"
    assert result["message"] == "Initialized partial workspace drive-run."
    assert result["error"] == "RuntimeError: Too many iterations"


def test_bootstrap_product_endpoint_adds_target_node_and_queue(tmp_path):
    workspace_dir = tmp_path / "drive-run"
    workspace_dir.mkdir()
    workspace = SimpleNamespace(
        neb_pot_fp=workspace_dir / "neb_pot.json",
        queue_fp=workspace_dir / "neb_queue.json",
    )

    root_mol = Molecule.from_smiles("C=C")
    pot = Pot(root=root_mol, target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(
        0,
        molecule=root_mol,
        td=structure_node_from_graph_like_molecule(root_mol),
        endpoint_optimized=False,
    )
    pot.write_to_disk(workspace.neb_pot_fp)
    workspace.queue_fp.write_text('{"items": [], "attempted_pairs": {}, "version": 1}')

    _bootstrap_product_endpoint(workspace, {"smiles": "CC"})

    updated = Pot.read_from_disk(workspace.neb_pot_fp)
    assert sorted(updated.graph.nodes) == [0, 1]
    assert updated.graph.nodes[1]["generated_by"] == "drive_product_smiles"
    assert updated.graph.has_edge(0, 1)
    assert updated.graph.edges[(0, 1)]["reaction"] == "Product target"

    queue = __import__("json").loads(workspace.queue_fp.read_text())
    assert queue["items"][0]["source_node"] == 0
    assert queue["items"][0]["target_node"] == 1


def test_apply_bootstrap_species_overrides_sets_charge_and_multiplicity(tmp_path):
    workspace_dir = tmp_path / "drive-run"
    workspace_dir.mkdir()
    workspace = SimpleNamespace(
        neb_pot_fp=workspace_dir / "neb_pot.json",
        queue_fp=workspace_dir / "neb_queue.json",
    )

    root_mol = Molecule.from_smiles("C=C")
    pot = Pot(root=root_mol, target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(
        0,
        molecule=root_mol,
        td=structure_node_from_graph_like_molecule(root_mol),
        endpoint_optimized=True,
    )
    pot.write_to_disk(workspace.neb_pot_fp)
    workspace.queue_fp.write_text('{"items": [], "attempted_pairs": {}, "version": 1}')

    from neb_dynamics.mepd_drive import _apply_bootstrap_species_overrides

    _apply_bootstrap_species_overrides(
        workspace,
        reactant={"smiles": "C=C", "charge": 1, "multiplicity": 2},
        product={"smiles": "CC", "charge": 1, "multiplicity": 2},
    )

    updated = Pot.read_from_disk(workspace.neb_pot_fp)
    assert updated.graph.nodes[0]["td"].structure.charge == 1
    assert updated.graph.nodes[0]["td"].structure.multiplicity == 2
    assert updated.graph.nodes[0]["endpoint_optimized"] is False
    assert updated.graph.nodes[1]["td"].structure.charge == 1
    assert updated.graph.nodes[1]["td"].structure.multiplicity == 2


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


def test_load_existing_workspace_job_recovers_stale_queue_items(monkeypatch, tmp_path):
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

    state = {"recover_kwargs": None, "write_fp": None}

    class _Queue:
        def recover_stale_running_items(self, **kwargs):
            state["recover_kwargs"] = kwargs
            return True

        def write_to_disk(self, fp):
            state["write_fp"] = Path(fp)

    monkeypatch.setattr("neb_dynamics.mepd_drive.RetropathsNEBQueue.read_from_disk", lambda _fp: _Queue())
    monkeypatch.setattr("neb_dynamics.mepd_drive.load_partial_annotated_pot", lambda workspace: None)

    _load_existing_workspace_job(str(workspace_dir))

    assert state["write_fp"] == workspace_dir / "neb_queue.json"
    assert state["recover_kwargs"]["output_dir"] == workspace_dir / "queue_runs"


def test_load_existing_workspace_job_imports_network_splits_directory(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "demo_network_splits"
    workspace_dir.mkdir()

    root_mol = Molecule.from_smiles("C=C")
    target_mol = Molecule.from_smiles("CC")
    pot = Pot(root=root_mol)
    pot.graph.nodes[0].update(
        molecule=root_mol,
        td=structure_node_from_graph_like_molecule(root_mol),
        endpoint_optimized=False,
    )
    pot.graph.add_node(
        1,
        molecule=target_mol,
        td=structure_node_from_graph_like_molecule(target_mol),
        endpoint_optimized=False,
    )
    pot.graph.add_edge(0, 1, reaction="Hydrogenation", list_of_nebs=[])
    pot.write_to_disk(workspace_dir / "demo_network.json")

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_partial_annotated_pot", lambda workspace: None)

    result = _load_existing_workspace_job(str(workspace_dir))

    assert result["workspace"]["run_name"] == "demo"
    assert result["reactant"]["smiles"] == "C=C"
    assert (workspace_dir / "workspace.json").exists()
    assert (workspace_dir / "neb_pot.json").exists()
    assert (workspace_dir / "neb_queue.json").exists()
    assert (workspace_dir / "retropaths_pot.json").exists()
    queue_payload = __import__("json").loads((workspace_dir / "neb_queue.json").read_text())
    assert any(
        int(item.get("source_node", -1)) == 0 and int(item.get("target_node", -1)) == 1
        for item in queue_payload.get("items", [])
    )


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


def test_drive_html_renders_inline_live_activity_mount():
    html = _drive_html()

    assert 'id="live-activity-panel"' in html
    assert 'id="live-activity-inline"' in html
    assert 'class="network-canvas-shell"' in html
    assert 'class="path-browser"' in html
    assert ">Inputs</h2>" in html
    assert ">Reaction Network</h2>" in html
    assert ">Exploration</h2>" in html
    assert ">Logging</h2>" in html
    assert 'id="inputs-path"' in html
    assert 'id="reactions-path"' in html
    assert 'id="environment-smiles"' in html
    assert 'id="pathmin-config-panel"' in html
    assert 'id="log-console"' in html
    assert 'id="network-toolbar"' in html
    assert 'id="path-source-node"' in html
    assert 'id="product-path-list"' in html
    assert 'id="clear-product-path"' in html
    assert 'data-tab="kinetics"' in html
    assert 'id="run-kmc"' in html
    assert "function renderLiveActivityContent(activity)" in html
    assert "function renderProductPathPanel(snapshot)" in html
    assert "function computePathHighlight(snapshot, sourceNodeId, productLabel)" in html
    assert "function buildOptimisticGrowthActivity(nodeId, title, note)" in html
    assert "pendingEdgeAddition" in html
    assert "function setPendingEdgeAddition(sourceNodeId, targetNodeId)" in html
    assert "manualEdgeRequestInFlight" in html
    assert "function setManualEdgeRequestInFlight(inFlight)" in html
    assert "A manual edge request is already in progress." in html
    assert "network-edge-line pending-add" in html
    assert "function computeTreeNetworkLayout(nodes, edges)" in html
    assert "function runKmcModel()" in html
    assert "pendingLiveActivity" in html
    assert "function renderNetworkToolbar()" in html
    assert "queueNanoreactor" in html
    assert "queueHessianSampleFromNode" in html
    assert "queueHessianSampleFromEdge" in html
    assert "Run Nanoreactor Sampling From This Geometry" in html
    assert "Run Hessian Sample From This Geometry" in html
    assert "Use As Path Source A" in html
    assert "toolbar-nanoreactor-node" in html
    assert "toolbar-hessian-node" in html
    assert "toolbar-hessian-edge" in html
    assert "function renderTemplateHtml(templatePayload)" in html
    assert "Template Render" in html
    assert "Template Summary" in html
    assert "function beginConnectMode(nodeId)" in html
    assert "No NEB-derived edge data is available yet for this edge." in html
    assert "<strong>Viewer:</strong>" in html


def test_run_kmc_payload_returns_population_plot(monkeypatch):
    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0)
    pot.graph.add_node(1)

    monkeypatch.setattr("neb_dynamics.mepd_drive._merge_drive_pot", lambda workspace, **kwargs: pot)
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.build_kmc_payload",
        lambda pot, temperature_kelvin=298.15, initial_conditions=None: {
            "temperature_kelvin": temperature_kelvin,
            "nodes": [
                {"id": 0, "label": "0: A", "initial": 1.0},
                {"id": 1, "label": "1: B", "initial": 0.0},
            ],
            "edges": [{"source": 0, "target": 1, "barrier": 1.0, "rate_constant": 2.0, "reaction": "A->B"}],
            "suppressed_edges": [],
            "initial_conditions": {0: 1.0, 1: 0.0},
            "default_end_time": 5.0,
        },
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.simulate_kmc",
        lambda pot, temperature_kelvin=298.15, initial_conditions=None, max_steps=200, final_time=None: {
            "temperature_kelvin": temperature_kelvin,
            "final_time": final_time if final_time is not None else 5.0,
            "max_steps": max_steps,
            "history": [
                {"time": 0.0, "populations": {0: 1.0, 1: 0.0}},
                {"time": 1.0, "populations": {0: 0.4, 1: 0.6}},
            ],
            "final_populations": {0: 0.4, 1: 0.6},
        },
    )

    result = _run_kmc_payload(
        SimpleNamespace(),
        temperature_kelvin=350.0,
        final_time=1.0,
        max_steps=10,
        initial_conditions={0: 1.0},
    )

    assert result["temperature_kelvin"] == 350.0
    assert result["plot"]["x"] == [0.0, 1.0]
    assert result["plot"]["series"][0]["label"] == "1: B"
    assert result["dominant_node"]["label"] == "1: B"


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
        inputs_fp=tmp_path / "inputs.toml",
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
    monkeypatch.setattr("neb_dynamics.mepd_drive._merge_drive_pot", lambda _workspace, **kwargs: pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_edge_visualizations", lambda workspace, pot: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_completed_queue_visualizations", lambda workspace, queue: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._load_template_payloads", lambda workspace: {})
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.RunInputs.open",
        lambda _fp: SimpleNamespace(
            engine_name="chemcloud",
            program="xtb",
            path_min_method="NEB",
            path_min_inputs=SimpleNamespace(max_steps=123),
            chain_inputs=SimpleNamespace(k=0.1),
            gi_inputs=SimpleNamespace(nimages=9),
            optimizer_kwds={"timestep": 0.5},
            program_kwds={"model": {"method": "GFN2xTB"}},
        ),
    )
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
    assert payload["inputs"]["path_min_method"] == "NEB"
    assert payload["inputs"]["path_min_inputs"]["max_steps"] == 123


def test_build_drive_payload_uses_reverse_edge_neb_data_when_selected_direction_lacks_it(monkeypatch, tmp_path):
    queue = SimpleNamespace(
        items=[
            SimpleNamespace(
                source_node=1,
                target_node=0,
                status="completed",
                started_at="2026-03-18T12:00:00",
                finished_at="2026-03-18T12:05:00",
                error=None,
            ),
        ]
    )
    retropaths_pot = SimpleNamespace(graph=nx.DiGraph())
    retropaths_pot.graph.add_node(0)
    retropaths_pot.graph.add_node(1)

    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0, td=SimpleNamespace(structure="xyz"), molecule="C")
    pot.graph.add_node(1, td=SimpleNamespace(structure="xyz"), molecule="CC")
    pot.graph.add_edge(1, 0, reaction="r1")
    pot.graph.add_edge(0, 1, reaction="r1", list_of_nebs=["chain"], barrier=12.5, exp_neg_barrier=0.0)

    workspace = SimpleNamespace(
        queue_fp=tmp_path / "queue.json",
        neb_pot_fp=tmp_path / "neb_pot.json",
        inputs_fp=tmp_path / "inputs.toml",
        workdir=str(tmp_path),
        run_name="drive",
        root_smiles="C",
        environment_smiles="",
        reactions_path=tmp_path / "reactions.p",
    )
    workspace.queue_fp.write_text("{}")

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_retropaths_pot", lambda _workspace: retropaths_pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive.RetropathsNEBQueue.read_from_disk", lambda _fp: queue)
    monkeypatch.setattr("neb_dynamics.mepd_drive._merge_drive_pot", lambda _workspace, **kwargs: pot)
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._write_edge_visualizations",
        lambda workspace, pot: [{"edge": "0 -> 1", "href": "edge_0_1.html"}],
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_completed_queue_visualizations", lambda workspace, queue: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._load_template_payloads", lambda workspace: {})
    monkeypatch.setattr("neb_dynamics.mepd_drive.RunInputs.open", lambda _fp: SimpleNamespace(engine_name="fake"))
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_network_explorer_payload",
        lambda graph, template_payloads=None, edge_visualizations=None: {
            "nodes": [
                {"id": 0, "label": "C", "data": {}},
                {"id": 1, "label": "CC", "data": {}},
            ],
            "edges": [
                {"source": 1, "target": 0, "reaction": "r1", "barrier": None, "chains": 0, "viewer_href": None, "data": {}, "template": {}},
            ],
        },
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._node_structure_payload", lambda _attrs: None)

    payload = _build_drive_payload(workspace)
    edge = payload["network"]["edges"][0]

    assert edge["queue_status"] == "completed"
    assert edge["neb_backed"] is True
    assert edge["barrier"] == 12.5
    assert edge["viewer_href"] == "edge_visualizations/edge_0_1.html"
    assert edge["result_from_reverse_edge"] is True


def test_build_drive_payload_uses_completed_queue_result_when_no_direct_neb_edge_exists(monkeypatch, tmp_path):
    queue = SimpleNamespace(
        items=[
            SimpleNamespace(
                source_node=1,
                target_node=0,
                status="completed",
                started_at="2026-03-18T12:00:00",
                finished_at="2026-03-18T12:05:00",
                error=None,
                result_dir=str(tmp_path / "result"),
                output_chain_xyz=None,
            ),
        ]
    )
    retropaths_pot = SimpleNamespace(graph=nx.DiGraph())
    retropaths_pot.graph.add_node(0)
    retropaths_pot.graph.add_node(1)

    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0, td=SimpleNamespace(structure="xyz"), molecule="C")
    pot.graph.add_node(1, td=SimpleNamespace(structure="xyz"), molecule="CC")
    pot.graph.add_edge(1, 0, reaction="r1")

    workspace = SimpleNamespace(
        queue_fp=tmp_path / "queue.json",
        neb_pot_fp=tmp_path / "neb_pot.json",
        inputs_fp=tmp_path / "inputs.toml",
        workdir=str(tmp_path),
        run_name="drive",
        root_smiles="C",
        environment_smiles="",
        reactions_path=tmp_path / "reactions.p",
        edge_visualizations_dir=tmp_path / "edge_visualizations",
    )
    workspace.queue_fp.write_text("{}")

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_retropaths_pot", lambda _workspace: retropaths_pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive.RetropathsNEBQueue.read_from_disk", lambda _fp: queue)
    monkeypatch.setattr("neb_dynamics.mepd_drive._merge_drive_pot", lambda _workspace, **kwargs: pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_edge_visualizations", lambda workspace, pot: [])
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._write_completed_queue_visualizations",
        lambda workspace, queue: [{"edge": "1 -> 0", "href": "queue_edge_1_0.html", "barrier": 9.75}],
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._load_template_payloads", lambda workspace: {})
    monkeypatch.setattr("neb_dynamics.mepd_drive.RunInputs.open", lambda _fp: SimpleNamespace(engine_name="fake"))
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_network_explorer_payload",
        lambda graph, template_payloads=None, edge_visualizations=None: {
            "nodes": [
                {"id": 0, "label": "C", "data": {}},
                {"id": 1, "label": "CC", "data": {}},
            ],
            "edges": [
                {"source": 1, "target": 0, "reaction": "r1", "barrier": None, "chains": 0, "viewer_href": None, "data": {}, "template": {}},
            ],
        },
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._node_structure_payload", lambda _attrs: None)

    payload = _build_drive_payload(workspace)
    edge = payload["network"]["edges"][0]

    assert edge["queue_status"] == "completed"
    assert edge["neb_backed"] is True
    assert edge["barrier"] == 9.75
    assert edge["viewer_href"] == "edge_visualizations/queue_edge_1_0.html"
    assert edge["result_from_completed_queue"] is True


def test_build_drive_payload_prefers_completed_queue_viewer_over_reconstructed_edge_viewer(monkeypatch, tmp_path):
    queue = SimpleNamespace(
        items=[
            SimpleNamespace(
                source_node=4,
                target_node=1,
                status="completed",
                started_at="2026-03-20T12:00:00",
                finished_at="2026-03-20T12:05:00",
                error=None,
                result_dir=str(tmp_path / "result"),
                output_chain_xyz=None,
            ),
        ]
    )
    retropaths_pot = SimpleNamespace(graph=nx.DiGraph())
    retropaths_pot.graph.add_node(1)
    retropaths_pot.graph.add_node(4)

    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(1, td=SimpleNamespace(structure="xyz"), molecule="A")
    pot.graph.add_node(4, td=SimpleNamespace(structure="xyz"), molecule="B")
    pot.graph.add_edge(4, 1, reaction="r1", list_of_nebs=["reconstructed"], barrier=5.0)

    workspace = SimpleNamespace(
        queue_fp=tmp_path / "queue.json",
        neb_pot_fp=tmp_path / "neb_pot.json",
        inputs_fp=tmp_path / "inputs.toml",
        workdir=str(tmp_path),
        run_name="drive",
        root_smiles="C",
        environment_smiles="",
        reactions_path=tmp_path / "reactions.p",
        edge_visualizations_dir=tmp_path / "edge_visualizations",
    )
    workspace.queue_fp.write_text("{}")

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_retropaths_pot", lambda _workspace: retropaths_pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive.RetropathsNEBQueue.read_from_disk", lambda _fp: queue)
    monkeypatch.setattr("neb_dynamics.mepd_drive._merge_drive_pot", lambda _workspace, **kwargs: pot)
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._write_edge_visualizations",
        lambda workspace, pot: [{"edge": "4 -> 1", "href": "edge_4_1.html"}],
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._write_completed_queue_visualizations",
        lambda workspace, queue: [{"edge": "4 -> 1", "href": "queue_edge_4_1.html", "barrier": 6.5}],
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._load_template_payloads", lambda workspace: {})
    monkeypatch.setattr("neb_dynamics.mepd_drive.RunInputs.open", lambda _fp: SimpleNamespace(engine_name="fake"))
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_network_explorer_payload",
        lambda graph, template_payloads=None, edge_visualizations=None: {
            "nodes": [
                {"id": 1, "label": "A", "data": {}},
                {"id": 4, "label": "B", "data": {}},
            ],
            "edges": [
                {"source": 4, "target": 1, "reaction": "r1", "barrier": 5.0, "chains": 1, "viewer_href": "edge_visualizations/edge_4_1.html", "data": {}, "template": {}},
            ],
        },
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._node_structure_payload", lambda _attrs: None)

    payload = _build_drive_payload(workspace)
    edge = payload["network"]["edges"][0]

    assert edge["viewer_href"] == "edge_visualizations/queue_edge_4_1.html"
    assert edge["result_from_completed_queue"] is True


def test_write_completed_queue_visualizations_reuses_cached_metadata(monkeypatch, tmp_path):
    workspace = SimpleNamespace(edge_visualizations_dir=tmp_path / "edge_visualizations")
    queue = SimpleNamespace(
        items=[
            SimpleNamespace(
                source_node=1,
                target_node=0,
                status="completed",
                result_dir=str(tmp_path / "result"),
                finished_at="2026-03-20T10:00:00",
            )
        ]
    )
    workspace.edge_visualizations_dir.mkdir(parents=True, exist_ok=True)
    (workspace.edge_visualizations_dir / "queue_edge_1_0.html").write_text("<html></html>", encoding="utf-8")
    (workspace.edge_visualizations_dir / "queue_edge_1_0.meta.json").write_text(
        """
{
  "barrier": 7.5,
  "finished_at": "2026-03-20T10:00:00",
  "is_elementary_result": true,
  "result_dir": "%s",
  "source_node": 1,
  "target_node": 0,
  "source_structure": {"xyz_b64": "source-xyz"},
  "target_structure": {"xyz_b64": "target-xyz"}
}
"""
        % (tmp_path / "result"),
        encoding="utf-8",
    )

    def _fail_read(*_args, **_kwargs):
        raise AssertionError("TreeNode.read_from_disk should not be called when cached metadata is valid")

    monkeypatch.setattr("neb_dynamics.mepd_drive.TreeNode.read_from_disk", _fail_read)

    rows = _write_completed_queue_visualizations(workspace, queue)

    assert rows == [
        {
            "edge": "1 -> 0",
            "barrier": 7.5,
            "href": "queue_edge_1_0.html",
            "source_node": 1,
            "target_node": 0,
            "source_structure": {"xyz_b64": "source-xyz"},
            "target_structure": {"xyz_b64": "target-xyz"},
        }
    ]


def test_write_completed_queue_visualizations_prefers_saved_output_chain_xyz(monkeypatch, tmp_path):
    workspace = SimpleNamespace(edge_visualizations_dir=tmp_path / "edge_visualizations")
    queue = SimpleNamespace(
        items=[
            SimpleNamespace(
                source_node=4,
                target_node=1,
                status="completed",
                result_dir=str(tmp_path / "result"),
                output_chain_xyz=str(tmp_path / "pair_4_1.xyz"),
                finished_at="2026-03-20T10:00:00",
            )
        ]
    )

    used = {}

    class _FakeStructure:
        def __init__(self, xyz):
            self._xyz = xyz
            self.symbols = ["H", "H"]

        def to_xyz(self):
            return self._xyz

    class _FakeChain:
        def __init__(self):
            self.nodes = [
                SimpleNamespace(structure=_FakeStructure("2\nsource\nH 0 0 0\nH 0 0 1\n"), graph="A"),
                SimpleNamespace(structure=_FakeStructure("2\ntarget\nH 0 0 0\nH 0 0 2\n"), graph="B"),
            ]

        def get_eA_chain(self):
            return 4.25

        def __len__(self):
            return len(self.nodes)

        def __getitem__(self, index):
            return self.nodes[index]

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.Chain.from_xyz",
        lambda fp, parameters: used.update({"xyz": str(fp)}) or _FakeChain(),
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.TreeNode.read_from_disk",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("TreeNode.read_from_disk should not be used when output_chain_xyz exists")),
    )
    monkeypatch.setattr(
        "neb_dynamics.scripts.main_cli._build_chain_visualizer_html",
        lambda chain, chain_trajectory=None: "<html><body>viewer</body></html>",
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._molecule_visual_payload",
        lambda molecule_like: {"smiles": str(molecule_like)},
    )

    rows = _write_completed_queue_visualizations(workspace, queue)

    assert used["xyz"] == str(tmp_path / "pair_4_1.xyz")
    assert rows[0]["edge"] == "4 -> 1"
    assert rows[0]["barrier"] == 4.25
    assert rows[0]["href"] == "queue_edge_4_1.html"
    assert rows[0]["source_structure"]["xyz_b64"] != rows[0]["target_structure"]["xyz_b64"]
    assert rows[0]["source_node"] == 4
    assert rows[0]["target_node"] == 1


def test_write_completed_queue_visualizations_skips_split_results(monkeypatch, tmp_path):
    workspace = SimpleNamespace(edge_visualizations_dir=tmp_path / "edge_visualizations")
    queue = SimpleNamespace(
        items=[
            SimpleNamespace(
                source_node=1,
                target_node=0,
                status="completed",
                result_dir=str(tmp_path / "result"),
                output_chain_xyz=str(tmp_path / "pair_1_0.xyz"),
                finished_at="2026-03-20T10:00:00",
            )
        ]
    )
    workspace.edge_visualizations_dir.mkdir(parents=True, exist_ok=True)

    class _FakeHistory:
        output_chain = object()

    monkeypatch.setattr("neb_dynamics.mepd_drive.TreeNode.read_from_disk", lambda *args, **kwargs: _FakeHistory())
    monkeypatch.setattr("neb_dynamics.mepd_drive._history_leaf_chains", lambda history: [object(), object()])
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.Chain.from_xyz",
        lambda fp, parameters: (_ for _ in ()).throw(AssertionError("split result should not build queue viewer")),
    )

    rows = _write_completed_queue_visualizations(workspace, queue)

    assert rows == []


def test_build_drive_payload_prefers_completed_queue_structures_for_structures_tab(monkeypatch, tmp_path):
    queue = SimpleNamespace(
        items=[
            SimpleNamespace(
                source_node=4,
                target_node=1,
                status="completed",
                started_at="2026-03-20T12:00:00",
                finished_at="2026-03-20T12:05:00",
                error=None,
                result_dir=str(tmp_path / "result"),
                output_chain_xyz=None,
            ),
        ]
    )
    retropaths_pot = SimpleNamespace(graph=nx.DiGraph())
    retropaths_pot.graph.add_node(1)
    retropaths_pot.graph.add_node(4)

    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(1, td=SimpleNamespace(structure="graph-target"), molecule="graph-target")
    pot.graph.add_node(4, td=SimpleNamespace(structure="graph-source"), molecule="graph-source")
    pot.graph.add_edge(4, 1, reaction="r1")

    workspace = SimpleNamespace(
        queue_fp=tmp_path / "queue.json",
        neb_pot_fp=tmp_path / "neb_pot.json",
        inputs_fp=tmp_path / "inputs.toml",
        workdir=str(tmp_path),
        run_name="drive",
        root_smiles="C",
        environment_smiles="",
        reactions_path=tmp_path / "reactions.p",
        edge_visualizations_dir=tmp_path / "edge_visualizations",
    )
    workspace.queue_fp.write_text("{}")

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_retropaths_pot", lambda _workspace: retropaths_pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive.RetropathsNEBQueue.read_from_disk", lambda _fp: queue)
    monkeypatch.setattr("neb_dynamics.mepd_drive._merge_drive_pot", lambda _workspace, **kwargs: pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_edge_visualizations", lambda workspace, pot: [])
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._write_completed_queue_visualizations",
        lambda workspace, queue: [
            {
                "edge": "4 -> 1",
                "href": "queue_edge_4_1.html",
                "barrier": 6.5,
                "source_node": 4,
                "target_node": 1,
                "source_structure": {"xyz_b64": "queue-source"},
                "target_structure": {"xyz_b64": "queue-target"},
            }
        ],
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._load_template_payloads", lambda workspace: {})
    monkeypatch.setattr("neb_dynamics.mepd_drive.RunInputs.open", lambda _fp: SimpleNamespace(engine_name="fake"))
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_network_explorer_payload",
        lambda graph, template_payloads=None, edge_visualizations=None: {
            "nodes": [
                {"id": 1, "label": "A", "data": {}},
                {"id": 4, "label": "B", "data": {}},
            ],
            "edges": [
                {"source": 4, "target": 1, "reaction": "r1", "barrier": None, "chains": 0, "viewer_href": None, "data": {}, "template": {}},
            ],
        },
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._node_structure_payload",
        lambda attrs: {"xyz_b64": f"graph-{attrs['td'].structure}"},
    )

    payload = _build_drive_payload(workspace)
    edge = payload["network"]["edges"][0]

    assert edge["source_structure"]["xyz_b64"] == "queue-source"
    assert edge["target_structure"]["xyz_b64"] == "queue-target"


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
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: {
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
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: called.update({"fast": True, "active_action": active_action}) or {"queue": {}, "workspace": {}, "network": {}},
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


def test_snapshot_reuses_cached_drive_payload_for_running_minimization(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server._drive_payload_cache_key = None
    server._drive_payload_cache_value = None
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    class _PendingFuture:
        def done(self):
            return False

    calls = {"count": 0}

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._drive_network_version",
        lambda _workspace: "same-version",
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast",
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: calls.update({"count": calls["count"] + 1}) or {"queue": {}, "workspace": {}, "network": {}},
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Optimizing geometry 1/2: node 0",
        last_error="",
        future=_PendingFuture(),
        busy_label="Optimizing geometry 1/2: node 0",
        active_action={"type": "minimize", "status": "running", "label": "Optimizing geometry 1/2: node 0"},
    )

    first = server.snapshot()
    second = server.snapshot()

    assert first["busy"] is True
    assert second["busy"] is True
    assert calls["count"] == 1


def test_snapshot_reuses_cached_full_drive_payload_for_idle_workspace(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server._drive_payload_cache_key = None
    server._drive_payload_cache_value = None
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    calls = {"count": 0}

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._drive_network_version",
        lambda _workspace: "stable-version",
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload",
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: calls.update({"count": calls["count"] + 1}) or {"queue": {}, "workspace": {}, "network": {}},
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Idle.",
        last_error="",
        future=None,
        busy_label="",
        active_action=None,
    )

    first = server.snapshot()
    second = server.snapshot()

    assert first["busy"] is False
    assert second["busy"] is False
    assert calls["count"] == 1


def test_snapshot_reuses_cached_drive_payload_for_running_minimization_when_version_changes(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server._drive_payload_cache_key = None
    server._drive_payload_cache_value = None
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    class _PendingFuture:
        def done(self):
            return False

    versions = iter(["version-1", "version-2"])
    calls = {"count": 0}

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._drive_network_version",
        lambda _workspace: next(versions),
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast",
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: calls.update({"count": calls["count"] + 1}) or {"queue": {}, "workspace": {}, "network": {}},
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
            "node_ids": [0, 1],
        },
    )

    first = server.snapshot()
    second = server.snapshot()

    assert first["busy"] is True
    assert second["busy"] is True
    assert calls["count"] == 1


def test_snapshot_uses_fast_builder_for_running_initialize(monkeypatch, tmp_path):
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
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: called.update({"fast": True, "active_action": active_action}) or {"queue": {}, "workspace": {}, "network": {}},
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("slow builder should not be used")),
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Building Retropaths network...",
        last_error="",
        future=_PendingFuture(),
        busy_label="Building Retropaths network...",
        active_action={"type": "initialize", "status": "running", "label": "Building Retropaths network..."},
    )

    snapshot = server.snapshot()

    assert snapshot["busy"] is True
    assert called["fast"] is True
    assert called["active_action"]["type"] == "initialize"


def test_snapshot_uses_fast_builder_for_running_apply_reactions(monkeypatch, tmp_path):
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
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: called.update({"fast": True, "active_action": active_action}) or {"queue": {}, "workspace": {}, "network": {}},
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("slow builder should not be used")),
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Applying reaction templates to node 0...",
        last_error="",
        future=_PendingFuture(),
        busy_label="Applying reaction templates to node 0...",
        active_action={"type": "apply-reactions", "status": "running", "label": "Applying reaction templates to node 0..."},
    )

    snapshot = server.snapshot()

    assert snapshot["busy"] is True
    assert called["fast"] is True
    assert called["active_action"]["type"] == "apply-reactions"


def test_snapshot_rebuilds_running_minimization_payload_when_job_progress_changes(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server._drive_payload_cache_key = None
    server._drive_payload_cache_value = None
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    class _PendingFuture:
        def done(self):
            return False

    calls = {"count": 0}

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._drive_network_version",
        lambda _workspace: "version-1",
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast",
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: calls.update({"count": calls["count"] + 1}) or {"queue": {}, "workspace": {}, "network": {}},
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
            "node_ids": [0, 1],
            "jobs": [
                {"node_id": 0, "status": "running"},
                {"node_id": 1, "status": "pending"},
            ],
        },
    )

    first = server.snapshot()
    with server.state_lock:
        server.runtime.active_action["label"] = "Finished geometry 1/2: node 0"
        server.runtime.active_action["jobs"][0]["status"] = "completed"
        server.runtime.active_action["jobs"][1]["status"] = "running"
    second = server.snapshot()

    assert first["busy"] is True
    assert second["busy"] is True
    assert calls["count"] == 2


def test_snapshot_falls_back_to_fast_payload_when_full_payload_build_fails(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server._drive_payload_cache_key = None
    server._drive_payload_cache_value = None
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    monkeypatch.setattr(
        MepdDriveServer,
        "_drive_payload_cache_lookup",
        lambda self, workspace, runtime: (_ for _ in ()).throw(RuntimeError("full payload failed")),
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast",
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: {"queue": {}, "workspace": {"network_nodes": 3}, "network": {"nodes": [{"id": 0}], "edges": []}},
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Initialized partial workspace.",
        last_error="RuntimeError: Too many iterations",
        future=None,
        busy_label="",
        active_action=None,
    )

    snapshot = server.snapshot()

    assert snapshot["initialized"] is True
    assert snapshot["drive"]["workspace"]["network_nodes"] == 3
    assert snapshot["drive"]["network"]["nodes"][0]["id"] == 0


def test_submit_initialize_defers_process_pool_submission_off_request_path(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server.runtime = SimpleNamespace(
        workspace=None,
        reactant=None,
        product=None,
        last_message="",
        last_error="",
        future=None,
        busy_label="",
        active_action=None,
    )
    server.base_directory = tmp_path
    server.inputs_fp = tmp_path / "inputs.toml"
    server.reactions_fp = None
    server.timeout_seconds = 30
    server.max_nodes = 40
    server.max_depth = 4
    server.max_parallel_nebs = 1
    server._drive_payload_cache_key = None
    server._drive_payload_cache_value = None

    class _FakeDeferredFuture:
        def add_done_callback(self, _callback):
            return None

        def done(self):
            return False

    thread_calls = {"count": 0}
    process_calls = {"count": 0}

    class _FakeThreadExecutor:
        def submit(self, fn, *args, **kwargs):
            thread_calls["count"] += 1
            return _FakeDeferredFuture()

    class _FakeProcessExecutor:
        def submit(self, fn, *args, **kwargs):
            process_calls["count"] += 1
            return SimpleNamespace(result=lambda: None)

    server.executor = _FakeThreadExecutor()
    server.process_executor = _FakeProcessExecutor()

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._resolve_species_input",
        lambda smiles="", xyz_text="": {"smiles": smiles} if smiles else None,
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._materialize_deployment_inputs",
        lambda template_fp, output_dir, run_name, theory_program=None, theory_method=None, theory_basis=None: Path(template_fp),
    )

    server.submit_initialize(
        {
            "reactant_smiles": "C=CCOCC=C",
            "run_name": "claisen-2",
        }
    )

    assert thread_calls["count"] == 1
    assert process_calls["count"] == 0
    assert server.runtime.busy_label == "Building Retropaths network..."
    assert server.runtime.active_action["type"] == "initialize"


def test_drive_network_version_ignores_annotated_overlay_file(tmp_path):
    workspace = SimpleNamespace(
        neb_pot_fp=tmp_path / "neb_pot.json",
        queue_fp=tmp_path / "neb_queue.json",
        retropaths_pot_fp=tmp_path / "retropaths_pot.json",
        annotated_neb_pot_fp=tmp_path / "neb_pot_annotated.json",
    )
    workspace.neb_pot_fp.write_text("{}")
    workspace.queue_fp.write_text("{}")
    workspace.retropaths_pot_fp.write_text("{}")
    workspace.annotated_neb_pot_fp.write_text("{}")

    version = _drive_network_version(workspace)

    assert "neb_pot.json" in version
    assert "neb_queue.json" in version
    assert "retropaths_pot.json" in version
    assert "neb_pot_annotated.json" not in version


def test_launch_mepd_drive_loads_existing_workspace_on_startup(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "existing-run"
    workspace_dir.mkdir()
    (workspace_dir / "workspace.json").write_text("{}")

    loaded = {
        "workspace": {
            "workdir": str(workspace_dir),
            "run_name": "existing-run",
            "root_smiles": "C=C",
            "environment_smiles": "",
            "inputs_fp": str(tmp_path / "inputs.toml"),
            "reactions_fp": "",
            "timeout_seconds": 30,
            "max_nodes": 40,
            "max_depth": 4,
            "max_parallel_nebs": 1,
        },
        "reactant": {"smiles": "C=C"},
        "product": None,
        "message": "Loaded existing workspace existing-run.",
    }
    captured = {}

    class _FakeServer:
        def __init__(self, _server_address, **kwargs):
            captured.update(kwargs)
            self.server_address = ("127.0.0.1", 48123)

    monkeypatch.setattr("neb_dynamics.mepd_drive._load_existing_workspace_job", lambda path, **kwargs: loaded)
    monkeypatch.setattr("neb_dynamics.mepd_drive.MepdDriveServer", _FakeServer)
    monkeypatch.setattr("neb_dynamics.mepd_drive.webbrowser.open", lambda _url: True)

    server = main_cli.launch_mepd_drive(
        directory=str(workspace_dir),
        inputs_fp=None,
        open_browser=False,
    )

    assert server.server_address == ("127.0.0.1", 48123)
    assert captured["base_directory"] == workspace_dir.parent.resolve()
    assert captured["initial_state"] == loaded
    assert captured["inputs_fp"] is None
    assert captured["network_splits"] is True


def test_launch_mepd_drive_loads_network_splits_directory_on_startup(monkeypatch, tmp_path):
    network_dir = tmp_path / "existing_network_splits"
    network_dir.mkdir()
    (network_dir / "existing_network.json").write_text("{}")

    loaded = {
        "workspace": {
            "workdir": str(network_dir),
            "run_name": "existing",
            "root_smiles": "C=C",
            "environment_smiles": "",
            "inputs_fp": str(tmp_path / "inputs.toml"),
            "reactions_fp": "",
            "timeout_seconds": 30,
            "max_nodes": 40,
            "max_depth": 4,
            "max_parallel_nebs": 1,
        },
        "reactant": {"smiles": "C=C"},
        "product": None,
        "message": "Loaded existing workspace existing.",
    }
    captured = {}

    class _FakeServer:
        def __init__(self, _server_address, **kwargs):
            captured.update(kwargs)
            self.server_address = ("127.0.0.1", 48123)

    monkeypatch.setattr("neb_dynamics.mepd_drive._load_existing_workspace_job", lambda path, **kwargs: loaded)
    monkeypatch.setattr("neb_dynamics.mepd_drive.MepdDriveServer", _FakeServer)
    monkeypatch.setattr("neb_dynamics.mepd_drive.webbrowser.open", lambda _url: True)

    server = main_cli.launch_mepd_drive(
        directory=str(network_dir),
        inputs_fp=None,
        open_browser=False,
    )

    assert server.server_address == ("127.0.0.1", 48123)
    assert captured["base_directory"] == network_dir.parent.resolve()
    assert captured["initial_state"] == loaded
    assert captured["inputs_fp"] is None
    assert captured["network_splits"] is True


def test_launch_mepd_drive_bootstraps_smiles_workspace_on_startup(monkeypatch, tmp_path):
    initialized = {
        "workspace": {
            "workdir": str(tmp_path / "drive" / "smiles-run"),
            "run_name": "smiles-run",
            "root_smiles": "C=C",
            "environment_smiles": "O",
            "inputs_fp": str(tmp_path / "inputs.toml"),
            "reactions_fp": "",
            "timeout_seconds": 30,
            "max_nodes": 40,
            "max_depth": 4,
            "max_parallel_nebs": 1,
        },
        "reactant": {"smiles": "C=C", "charge": 1, "multiplicity": 2},
        "product": {"smiles": "CC", "charge": 1, "multiplicity": 2},
        "message": "Initialized workspace smiles-run.",
    }
    captured = {}

    class _FakeServer:
        def __init__(self, _server_address, **kwargs):
            captured.update(kwargs)
            self.server_address = ("127.0.0.1", 48123)

    monkeypatch.setattr("neb_dynamics.mepd_drive._initialize_workspace_job", lambda **kwargs: initialized)
    monkeypatch.setattr("neb_dynamics.mepd_drive.MepdDriveServer", _FakeServer)
    monkeypatch.setattr("neb_dynamics.mepd_drive.webbrowser.open", lambda _url: True)

    server = main_cli.launch_mepd_drive(
        directory=str(tmp_path / "drive" / "smiles-run"),
        inputs_fp=str(tmp_path / "inputs.toml"),
        smiles="C=C",
        product_smiles="CC",
        environment_smiles="O",
        charge=1,
        multiplicity=2,
        run_name="smiles-run",
        open_browser=False,
    )

    assert server.server_address == ("127.0.0.1", 48123)
    assert captured["initial_state"] == initialized
    assert captured["base_directory"] == (tmp_path / "drive").resolve()
    assert captured["inputs_fp"] == (tmp_path / "inputs.toml").resolve()
    assert captured["network_splits"] is True


def test_build_drive_payload_tolerates_missing_inputs_and_retropaths(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "drive-run"
    workspace_dir.mkdir()
    workspace = RetropathsWorkspace(
        workdir=str(workspace_dir),
        run_name="drive-run",
        root_smiles="C=C",
        environment_smiles="",
        inputs_fp=str(workspace_dir / "missing_inputs.toml"),
        reactions_fp="",
    )

    root_mol = Molecule.from_smiles("C=C")
    target_mol = Molecule.from_smiles("CC")
    pot = Pot(root=root_mol)
    pot.graph.nodes[0].update(
        molecule=root_mol,
        td=structure_node_from_graph_like_molecule(root_mol),
    )
    pot.graph.add_node(
        1,
        molecule=target_mol,
        td=structure_node_from_graph_like_molecule(target_mol),
    )
    pot.graph.add_edge(0, 1, reaction="Hydrogenation", list_of_nebs=[])
    pot.write_to_disk(workspace.neb_pot_fp)
    workspace.queue_fp.write_text('{"items": [], "attempted_pairs": {}, "version": 1}')

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_retropaths_pot", lambda _workspace: (_ for _ in ()).throw(FileNotFoundError("missing")))
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_edge_visualizations", lambda workspace, pot: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_completed_queue_visualizations", lambda workspace, queue: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._load_template_payloads", lambda workspace: {})

    payload = _build_drive_payload(workspace)

    assert payload["workspace"]["network_nodes"] == 2
    assert payload["workspace"]["retropaths_nodes"] == 0
    assert payload["inputs"]["error"]


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
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: called.update({"fast_neb": True, "active_action": active_action}) or {"queue": {}, "workspace": {}, "network": {}},
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
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: {"queue": {}, "workspace": {}, "network": {}},
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
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: {"queue": {}, "workspace": {}, "network": {}},
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


def test_build_growth_live_payload_reads_progress_file(tmp_path):
    progress_fp = tmp_path / "growth.json"
    progress_fp.write_text(
        """{
  "title": "Growing Retropaths network",
  "note": "Growing node 0.",
  "phase": "growing",
  "network": {
    "nodes": [{"id": 0, "label": "0", "growing": true}],
    "edges": []
  }
}"""
    )

    payload = _build_growth_live_payload(
        {"type": "initialize", "status": "running", "progress_fp": str(progress_fp)}
    )

    assert payload["type"] == "growth"
    assert payload["network"]["nodes"][0]["growing"] is True
    assert payload["note"] == "Growing node 0."


def test_snapshot_includes_live_activity_for_running_initialize(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()

    class _PendingFuture:
        def done(self):
            return False

    progress_fp = tmp_path / "growth.json"
    progress_fp.write_text(
        """{
  "title": "Growing Retropaths network",
  "note": "Growing node 0.",
  "phase": "growing",
  "network": {
    "nodes": [{"id": 0, "label": "0", "growing": true}],
    "edges": []
  }
}"""
    )

    server.runtime = SimpleNamespace(
        workspace=None,
        reactant=None,
        product=None,
        last_message="Building Retropaths network...",
        last_error="",
        future=_PendingFuture(),
        busy_label="Building Retropaths network...",
        active_action={"type": "initialize", "status": "running", "label": "Building Retropaths network...", "progress_fp": str(progress_fp)},
    )

    snapshot = server.snapshot()

    assert snapshot["busy"] is True
    assert snapshot["live_activity"]["type"] == "growth"
    assert snapshot["live_activity"]["network"]["nodes"][0]["growing"] is True


def test_apply_reactions_to_node_converts_drive_molecules_to_retropaths_compatible_graphs(monkeypatch, tmp_path):
    class _DriveMolecule:
        def __init__(self, smiles):
            self._smiles = smiles

        def copy(self):
            return _DriveMolecule(self._smiles)

        def smiles_from_multiple_molecules(self):
            return self._smiles

    class _FakeRetropathsMolecule:
        def __init__(self, smiles=""):
            self._smiles = smiles

        @classmethod
        def from_smiles(cls, smiles):
            return cls(smiles)

        def copy(self):
            return _FakeRetropathsMolecule(self._smiles)

        def substitute_group(self, *_args, **_kwargs):
            return self

    class _FakeRetropathsPot:
        def __init__(self, root, environment, rxn_name):
            assert isinstance(root, _FakeRetropathsMolecule)
            assert isinstance(environment, _FakeRetropathsMolecule)
            self.root = root
            self.environment = environment
            self.rxn_name = rxn_name
            self.graph = nx.DiGraph()
            self.graph.add_node(0, molecule=root)

        def grow_this_node(self, *_args, **_kwargs):
            product = _FakeRetropathsMolecule.from_smiles("C=CO")
            self.graph.add_node(1, molecule=product, converged=False)
            self.graph.add_edge(1, 0, reaction="demo")

    class _FakePot:
        def __init__(self):
            self.graph = nx.DiGraph()
            self.graph.add_node(
                0,
                molecule=_DriveMolecule("C=CC"),
                td=SimpleNamespace(
                    structure=SimpleNamespace(charge=0, multiplicity=1),
                    graph=_DriveMolecule("C=CC"),
                ),
            )

        def write_to_disk(self, _fp):
            return None

    workspace = SimpleNamespace(
        run_name="drive",
        reactions_path=tmp_path / "reactions.p",
        neb_pot_fp=tmp_path / "neb_pot.json",
        queue_fp=tmp_path / "neb_queue.json",
    )

    fake_pot = _FakePot()
    progress_fp = tmp_path / "apply.progress.json"

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: fake_pot)
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow._load_retropaths_classes",
        lambda: (
            SimpleNamespace(pload=lambda _fp: SimpleNamespace()),
            _FakeRetropathsMolecule,
            _FakeRetropathsPot,
        ),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow._global_environment_molecule",
        lambda _pot: _DriveMolecule("O"),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.copy_graph_like_molecule",
        lambda mol: SimpleNamespace(copied_smiles=getattr(mol, "_smiles", "")),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.structure_node_from_graph_like_molecule",
        lambda molecule, charge=0, spinmult=1: SimpleNamespace(graph=molecule, structure=SimpleNamespace(charge=charge, multiplicity=spinmult)),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.build_retropaths_neb_queue",
        lambda pot, queue_fp, overwrite=False: None,
    )

    result = apply_reactions_to_node(workspace, 0, progress_fp=str(progress_fp))

    assert result["added_nodes"] == 1
    assert fake_pot.graph.nodes[1]["molecule"].copied_smiles == "C=CO"
    assert "Applied reactions to node 0" in result["message"]


def test_run_nanoreactor_for_node_merges_minima(monkeypatch, tmp_path):
    class _FakeGraph:
        def __init__(self, label):
            self.label = label

        def copy(self):
            return _FakeGraph(self.label)

    class _FakePot:
        def __init__(self):
            self.graph = nx.DiGraph()
            self.graph.add_node(
                0,
                molecule=_FakeGraph("source"),
                td=SimpleNamespace(
                    structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
                    graph=_FakeGraph("source"),
                ),
                endpoint_optimized=True,
            )

        def write_to_disk(self, _fp):
            return None

    candidate = SimpleNamespace(
        structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
        graph=_FakeGraph("candidate"),
    )
    optimized = SimpleNamespace(
        structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
        graph=_FakeGraph("optimized"),
    )

    workspace = SimpleNamespace(
        inputs_fp=str(tmp_path / "inputs.toml"),
        neb_pot_fp=tmp_path / "neb_pot.json",
        queue_fp=tmp_path / "neb_queue.json",
    )
    fake_pot = _FakePot()
    ensured = []

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: fake_pot)
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.RunInputs.open",
        lambda _fp: SimpleNamespace(
            engine_name="chemcloud",
            program="crest",
            nanoreactor_inputs={},
            engine=SimpleNamespace(
                compute_nanoreactor_candidates=lambda node, nanoreactor_inputs=None: [candidate],
                compute_geometry_optimizations=lambda nodes: [[optimized]],
            ),
            chain_inputs=SimpleNamespace(node_rms_thre=5.0, node_ene_thre=5.0),
        ),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow._nodes_identical",
        lambda node1, node2, chain_inputs: False,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.copy_graph_like_molecule",
        lambda mol: mol.copy() if hasattr(mol, "copy") else mol,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.ensure_queue_item_for_edge",
        lambda pot, source_node, target_node, queue_fp, overwrite=False: ensured.append((source_node, target_node)),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.build_retropaths_neb_queue",
        lambda pot, queue_fp, overwrite=False: None,
    )

    result = run_nanoreactor_for_node(workspace, 0, progress_fp=str(tmp_path / "nanoreactor.progress.json"))

    assert result["added_nodes"] == 1
    assert result["added_edges"] == 1
    assert ensured == [(0, 1)]
    assert fake_pot.graph.has_edge(0, 1)
    assert fake_pot.graph.nodes[1]["endpoint_optimized"] is True
    assert fake_pot.graph.edges[(0, 1)]["reaction"] == "Nanoreactor reaction 1"


def test_run_nanoreactor_for_node_attaches_follow_on_runs_to_clicked_node(monkeypatch, tmp_path):
    class _FakeGraph:
        def __init__(self, label):
            self.label = label

        def copy(self):
            return _FakeGraph(self.label)

    class _FakePot:
        def __init__(self):
            self.graph = nx.DiGraph()
            self.graph.add_node(
                0,
                molecule=_FakeGraph("source"),
                td=SimpleNamespace(
                    structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
                    graph=_FakeGraph("source"),
                ),
                endpoint_optimized=True,
            )

        def write_to_disk(self, _fp):
            return None

    fake_pot = _FakePot()
    workspace = SimpleNamespace(
        inputs_fp=str(tmp_path / "inputs.toml"),
        neb_pot_fp=tmp_path / "neb_pot.json",
        queue_fp=tmp_path / "neb_queue.json",
    )
    engine_calls = []

    def _compute_nanoreactor_candidates(node, nanoreactor_inputs=None):
        engine_calls.append(getattr(getattr(node, "graph", None), "label", None))
        if engine_calls == ["source"]:
            return [SimpleNamespace(
                structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
                graph=_FakeGraph("candidate-1"),
            )]
        return [SimpleNamespace(
            structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
            graph=_FakeGraph("candidate-2"),
        )]

    optimized_without_graph = SimpleNamespace(
        structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
        graph=None,
    )

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: fake_pot)
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.RunInputs.open",
        lambda _fp: SimpleNamespace(
            engine_name="chemcloud",
            program="crest",
            nanoreactor_inputs={},
            engine=SimpleNamespace(
                compute_nanoreactor_candidates=_compute_nanoreactor_candidates,
                compute_geometry_optimizations=lambda nodes: [[SimpleNamespace(
                    structure=optimized_without_graph.structure,
                    graph=None,
                )] for _node in nodes],
            ),
            chain_inputs=SimpleNamespace(node_rms_thre=5.0, node_ene_thre=5.0),
        ),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow._nodes_identical",
        lambda node1, node2, chain_inputs: False,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.copy_graph_like_molecule",
        lambda mol: mol.copy() if hasattr(mol, "copy") else mol,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.ensure_queue_item_for_edge",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.build_retropaths_neb_queue",
        lambda pot, queue_fp, overwrite=False: None,
    )

    run_nanoreactor_for_node(workspace, 0, progress_fp=str(tmp_path / "nanoreactor_1.progress.json"))
    assert getattr(fake_pot.graph.nodes[1]["td"].graph, "label", None) == "candidate-1"

    run_nanoreactor_for_node(workspace, 1, progress_fp=str(tmp_path / "nanoreactor_2.progress.json"))
    assert engine_calls == ["source", "candidate-1"]
    assert not fake_pot.graph.has_edge(0, 2)
    assert fake_pot.graph.has_edge(1, 2)
    assert fake_pot.graph.nodes[1]["nanoreactor_provenance_node"] == 0
    assert fake_pot.graph.nodes[2]["nanoreactor_provenance_node"] == 1


def test_run_nanoreactor_for_node_falls_back_to_single_optimizations(monkeypatch, tmp_path):
    class _FakeGraph:
        def __init__(self, label):
            self.label = label

        def copy(self):
            return _FakeGraph(self.label)

    class _FakePot:
        def __init__(self):
            self.graph = nx.DiGraph()
            self.graph.add_node(
                0,
                molecule=_FakeGraph("source"),
                td=SimpleNamespace(
                    structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
                    graph=_FakeGraph("source"),
                ),
                endpoint_optimized=True,
            )

        def write_to_disk(self, _fp):
            return None

    candidate = SimpleNamespace(
        structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
        graph=_FakeGraph("candidate"),
    )
    optimized = SimpleNamespace(
        structure=SimpleNamespace(symbols=["H"], charge=0, multiplicity=1),
        graph=_FakeGraph("optimized"),
    )
    fake_pot = _FakePot()
    workspace = SimpleNamespace(
        inputs_fp=str(tmp_path / "inputs.toml"),
        neb_pot_fp=tmp_path / "neb_pot.json",
        queue_fp=tmp_path / "neb_queue.json",
    )
    calls = {"single": 0}

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: fake_pot)
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.RunInputs.open",
        lambda _fp: SimpleNamespace(
            engine_name="chemcloud",
            program="terachem",
            nanoreactor_inputs={},
            engine=SimpleNamespace(
                compute_nanoreactor_candidates=lambda node, nanoreactor_inputs=None: [candidate],
                compute_geometry_optimizations=lambda nodes: (_ for _ in ()).throw(RuntimeError("batch failed")),
                compute_geometry_optimization=lambda node: calls.__setitem__("single", calls["single"] + 1) or [optimized],
            ),
            chain_inputs=SimpleNamespace(node_rms_thre=5.0, node_ene_thre=5.0),
        ),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow._nodes_identical",
        lambda node1, node2, chain_inputs: False,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.copy_graph_like_molecule",
        lambda mol: mol.copy() if hasattr(mol, "copy") else mol,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.ensure_queue_item_for_edge",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.build_retropaths_neb_queue",
        lambda pot, queue_fp, overwrite=False: None,
    )

    result = run_nanoreactor_for_node(workspace, 0, progress_fp=str(tmp_path / "nanoreactor_fallback.progress.json"))

    assert calls["single"] == 1
    assert result["added_nodes"] == 1


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


def test_optimize_selected_nodes_batches_chemcloud_requests(monkeypatch, tmp_path):
    writes = []
    progress_messages = []
    node_updates = []
    batch_calls = []

    class _FakePot:
        def __init__(self):
            self.graph = nx.DiGraph()
            self.graph.add_node(0, td=SimpleNamespace(has_molecular_graph=False, graph="g0"))
            self.graph.add_node(1, td=SimpleNamespace(has_molecular_graph=False, graph="g1"))

        def write_to_disk(self, fp):
            writes.append(Path(fp))

    class _FakeEngine:
        compute_program = "chemcloud"

        def compute_geometry_optimizations(self, nodes, keywords=None):
            batch_calls.append({"nodes": list(nodes), "keywords": dict(keywords or {})})
            return [
                [SimpleNamespace(energy=0.0), SimpleNamespace(energy=1.0, graph=None, has_molecular_graph=False)],
                [SimpleNamespace(energy=2.0), SimpleNamespace(energy=3.0, graph=None, has_molecular_graph=False)],
            ]

    fake_pot = _FakePot()
    workspace = SimpleNamespace(inputs_fp=str(tmp_path / "inputs.toml"), neb_pot_fp=tmp_path / "neb_pot.json")

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.RunInputs.open",
        lambda _fp: SimpleNamespace(engine=_FakeEngine(), engine_name="chemcloud"),
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive.Pot.read_from_disk", lambda _fp: fake_pot)
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
    assert len(batch_calls) == 1
    assert len(batch_calls[0]["nodes"]) == 2
    assert batch_calls[0]["keywords"] == {"coordsys": "cart", "maxiter": 500}
    assert progress_messages[0] == "Submitting 2 geometry optimizations to ChemCloud in parallel."
    assert "Finished geometry 1/2: node 0" in progress_messages


def test_build_drive_payload_marks_nanoreactor_available_for_chemcloud_crest(monkeypatch, tmp_path):
    queue = SimpleNamespace(items=[])
    retropaths_pot = SimpleNamespace(graph=nx.DiGraph())
    retropaths_pot.graph.add_node(0)

    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0, td=SimpleNamespace(structure=SimpleNamespace(symbols=["H"])), molecule="mol")

    workspace = SimpleNamespace(
        queue_fp=tmp_path / "queue.json",
        neb_pot_fp=tmp_path / "neb_pot.json",
        inputs_fp=tmp_path / "inputs.toml",
        workdir=str(tmp_path),
        run_name="drive",
        root_smiles="C",
        environment_smiles="",
        reactions_path=tmp_path / "reactions.p",
        edge_visualizations_dir=tmp_path / "edge_visualizations",
    )
    workspace.queue_fp.write_text("{}")

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_retropaths_pot", lambda _workspace: retropaths_pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive.RetropathsNEBQueue.read_from_disk", lambda _fp: queue)
    monkeypatch.setattr("neb_dynamics.mepd_drive._merge_drive_pot", lambda _workspace, **kwargs: pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_edge_visualizations", lambda workspace, pot: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_completed_queue_visualizations", lambda workspace, queue: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._load_template_payloads", lambda workspace: {})
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.RunInputs.open",
        lambda _fp: SimpleNamespace(
            engine_name="chemcloud",
            program="crest",
            path_min_method="neb",
            path_min_inputs={},
            chain_inputs={},
            gi_inputs={},
            optimizer_kwds={},
            program_kwds={},
            nanoreactor_inputs={},
        ),
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_network_explorer_payload",
        lambda graph, template_payloads=None, edge_visualizations=None: {
            "nodes": [{"id": 0, "label": "A", "data": {}}],
            "edges": [],
        },
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._node_structure_payload", lambda attrs: {"xyz_b64": "node-xyz"})

    payload = _build_drive_payload(workspace)

    assert payload["network"]["nodes"][0]["can_nanoreactor"] is True


def test_build_drive_payload_marks_hessian_sample_available_when_engine_supports_hessian(monkeypatch, tmp_path):
    queue = SimpleNamespace(items=[])
    retropaths_pot = SimpleNamespace(graph=nx.DiGraph())
    retropaths_pot.graph.add_node(0)

    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0, td=SimpleNamespace(structure=SimpleNamespace(symbols=["H"])), molecule="mol")

    workspace = SimpleNamespace(
        queue_fp=tmp_path / "queue.json",
        neb_pot_fp=tmp_path / "neb_pot.json",
        inputs_fp=tmp_path / "inputs.toml",
        workdir=str(tmp_path),
        run_name="drive",
        root_smiles="C",
        environment_smiles="",
        reactions_path=tmp_path / "reactions.p",
        edge_visualizations_dir=tmp_path / "edge_visualizations",
    )
    workspace.queue_fp.write_text("{}")

    monkeypatch.setattr("neb_dynamics.mepd_drive.load_retropaths_pot", lambda _workspace: retropaths_pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive.RetropathsNEBQueue.read_from_disk", lambda _fp: queue)
    monkeypatch.setattr("neb_dynamics.mepd_drive._merge_drive_pot", lambda _workspace, **kwargs: pot)
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_edge_visualizations", lambda workspace, pot: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._write_completed_queue_visualizations", lambda workspace, queue: [])
    monkeypatch.setattr("neb_dynamics.mepd_drive._load_template_payloads", lambda workspace: {})

    class _Engine:
        def _compute_hessian_result(self, _node):
            return None

        def compute_geometry_optimization(self, _node):
            return []

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.RunInputs.open",
        lambda _fp: SimpleNamespace(
            engine_name="qcop",
            program="xtb",
            engine=_Engine(),
            path_min_method="neb",
            path_min_inputs={},
            chain_inputs={},
            gi_inputs={},
            optimizer_kwds={},
            program_kwds={},
            nanoreactor_inputs={},
        ),
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_network_explorer_payload",
        lambda graph, template_payloads=None, edge_visualizations=None: {
            "nodes": [{"id": 0, "label": "A", "data": {}}],
            "edges": [],
        },
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._node_structure_payload", lambda attrs: {"xyz_b64": "node-xyz"})

    payload = _build_drive_payload(workspace)

    assert payload["network"]["nodes"][0]["can_hessian_sample"] is True


def test_snapshot_uses_fast_builder_for_running_nanoreactor(monkeypatch, tmp_path):
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
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: called.update({"fast": True, "active_action": active_action}) or {"queue": {}, "workspace": {}, "network": {}},
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("slow builder should not be used")),
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Running nanoreactor sampling from node 0...",
        last_error="",
        future=_PendingFuture(),
        busy_label="Running nanoreactor sampling from node 0...",
        active_action={"type": "nanoreactor", "status": "running", "label": "Running nanoreactor sampling from node 0..."},
    )

    snapshot = server.snapshot()

    assert snapshot["busy"] is True
    assert called["fast"] is True
    assert called["active_action"]["type"] == "nanoreactor"


def test_snapshot_uses_fast_builder_for_running_hessian_sample(monkeypatch, tmp_path):
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
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: called.update({"fast": True, "active_action": active_action}) or {"queue": {}, "workspace": {}, "network": {}},
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("slow builder should not be used")),
    )

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Running Hessian sample from node 0...",
        last_error="",
        future=_PendingFuture(),
        busy_label="Running Hessian sample from node 0...",
        active_action={"type": "hessian-sample", "status": "running", "label": "Running Hessian sample from node 0..."},
    )

    snapshot = server.snapshot()

    assert snapshot["busy"] is True
    assert called["fast"] is True
    assert called["active_action"]["type"] == "hessian-sample"


def test_snapshot_uses_fast_builder_once_after_manual_edge(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server._drive_payload_cache_key = None
    server._drive_payload_cache_value = None
    server._prefer_fast_payload_once = True
    server.network_splits = True
    workspace = SimpleNamespace(queue_fp=tmp_path / "neb_queue.json")
    workspace.queue_fp.write_text("{}")

    called = {"fast": 0, "full": 0}

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload_fast",
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: called.__setitem__("fast", called["fast"] + 1) or {"queue": {}, "workspace": {}, "network": {}},
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive._build_drive_payload",
        lambda workspace, product_smiles="", active_job_label="", active_action=None, **kwargs: called.__setitem__("full", called["full"] + 1) or {"queue": {}, "workspace": {}, "network": {}},
    )
    monkeypatch.setattr("neb_dynamics.mepd_drive._drive_defaults_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr("neb_dynamics.mepd_drive._drive_network_version", lambda workspace: "v1")

    server.runtime = SimpleNamespace(
        workspace=workspace,
        reactant={"smiles": "C=C.O"},
        product=None,
        last_message="Manual edge updated.",
        last_error="",
        future=None,
        busy_label="",
        active_action=None,
    )

    first = server.snapshot()
    second = server.snapshot()

    assert first["drive"] == {"queue": {}, "workspace": {}, "network": {}}
    assert second["drive"] == {"queue": {}, "workspace": {}, "network": {}}
    assert called["fast"] == 1
    assert called["full"] == 1
    assert server._prefer_fast_payload_once is False


def test_submit_nanoreactor_uses_thread_executor_not_process_pool(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    workspace = SimpleNamespace(
        directory=workspace_dir,
        workdir=str(workspace_dir),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
        reactions_fp=str(tmp_path / "reactions.pkl"),
        timeout_seconds=10,
        max_nodes=10,
        max_depth=3,
        max_parallel_nebs=1,
    )
    server.runtime = SimpleNamespace(
        workspace=workspace,
        future=None,
        busy_label="",
        last_error="",
        last_message="",
        active_action=None,
    )

    class _DoneFuture:
        def add_done_callback(self, _cb):
            return None

        def done(self):
            return False

    called = {"executor": 0, "process": 0}

    class _Executor:
        def submit(self, fn, *args, **kwargs):
            called["executor"] += 1
            return _DoneFuture()

    class _ProcessExecutor:
        def submit(self, fn, *args, **kwargs):
            called["process"] += 1
            raise AssertionError("process pool should not be used for nanoreactor")

    server.executor = _Executor()
    server.process_executor = _ProcessExecutor()
    server._set_busy = MepdDriveServer._set_busy.__get__(server, MepdDriveServer)
    server._finish_future = lambda future: None
    server._assert_idle = lambda: None

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.RetropathsWorkspace",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    server.submit_nanoreactor(node_id=3)

    assert called["executor"] == 1
    assert called["process"] == 0


def test_submit_hessian_sample_uses_thread_executor_not_process_pool(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    workspace = SimpleNamespace(
        directory=workspace_dir,
        workdir=str(workspace_dir),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
        reactions_fp=str(tmp_path / "reactions.pkl"),
        timeout_seconds=10,
        max_nodes=10,
        max_depth=3,
        max_parallel_nebs=1,
    )
    server.runtime = SimpleNamespace(
        workspace=workspace,
        future=None,
        busy_label="",
        last_error="",
        last_message="",
        active_action=None,
    )

    class _DoneFuture:
        def add_done_callback(self, _cb):
            return None

        def done(self):
            return False

    called = {"executor": 0, "process": 0}

    class _Executor:
        def submit(self, fn, *args, **kwargs):
            called["executor"] += 1
            return _DoneFuture()

    class _ProcessExecutor:
        def submit(self, fn, *args, **kwargs):
            called["process"] += 1
            raise AssertionError("process pool should not be used for Hessian sampling")

    server.executor = _Executor()
    server.process_executor = _ProcessExecutor()
    server._set_busy = MepdDriveServer._set_busy.__get__(server, MepdDriveServer)
    server._finish_future = lambda future: None
    server._assert_idle = lambda: None

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.RetropathsWorkspace",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    server.submit_hessian_sample(node_id=3, dr=0.15)

    assert called["executor"] == 1
    assert called["process"] == 0


def test_submit_hessian_sample_forwards_max_candidates_to_workflow(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    workspace = SimpleNamespace(
        directory=workspace_dir,
        workdir=str(workspace_dir),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
        reactions_fp=str(tmp_path / "reactions.pkl"),
        timeout_seconds=10,
        max_nodes=10,
        max_depth=3,
        max_parallel_nebs=1,
    )
    server.runtime = SimpleNamespace(
        workspace=workspace,
        future=None,
        busy_label="",
        last_error="",
        last_message="",
        active_action=None,
    )

    class _DoneFuture:
        def add_done_callback(self, _cb):
            return None

        def done(self):
            return False

    forwarded = {}

    class _Executor:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)
            return _DoneFuture()

    class _ProcessExecutor:
        def submit(self, fn, *args, **kwargs):
            raise AssertionError("process pool should not be used for Hessian sampling")

    def _fake_run_hessian_sample_for_node(
        workspace_obj,
        node_id,
        *,
        dr,
        max_candidates,
        progress_fp=None,
    ):
        forwarded["node_id"] = int(node_id)
        forwarded["dr"] = float(dr)
        forwarded["max_candidates"] = int(max_candidates)
        forwarded["progress_fp"] = str(progress_fp or "")
        return {"message": "ok"}

    server.executor = _Executor()
    server.process_executor = _ProcessExecutor()
    server._set_busy = MepdDriveServer._set_busy.__get__(server, MepdDriveServer)
    server._finish_future = lambda future: None
    server._assert_idle = lambda: None

    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.RetropathsWorkspace",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "neb_dynamics.mepd_drive.run_hessian_sample_for_node",
        _fake_run_hessian_sample_for_node,
    )

    server.submit_hessian_sample(node_id=3, dr=0.15, max_candidates=42)

    assert forwarded["node_id"] == 3
    assert forwarded["dr"] == 0.15
    assert forwarded["max_candidates"] == 42
    assert server.runtime.active_action["max_candidates"] == 42


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


def test_submit_minimize_rejects_already_optimized_node(monkeypatch, tmp_path):
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
    pot.graph.add_node(0, td=SimpleNamespace(structure="xyz"), endpoint_optimized=True)
    monkeypatch.setattr("neb_dynamics.mepd_drive.Pot.read_from_disk", lambda _fp: pot)

    try:
        server.submit_minimize([0])
        assert False, "submit_minimize should reject an already optimized node"
    except ValueError as exc:
        assert "already geometry-optimized" in str(exc)


def test_submit_run_neb_surfaces_non_queueable_edge(monkeypatch, tmp_path):
    server = object.__new__(MepdDriveServer)
    server.state_lock = __import__("threading").Lock()
    server.runtime = SimpleNamespace(
        workspace=SimpleNamespace(
            neb_pot_fp=tmp_path / "neb_pot.json",
            queue_fp=tmp_path / "neb_queue.json",
            directory=tmp_path,
            queue_output_dir=tmp_path / "queue_runs",
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
            recover_stale_running_items=lambda **kwargs: False,
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
