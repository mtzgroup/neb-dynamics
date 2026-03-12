from types import SimpleNamespace
from pathlib import Path

import networkx as nx

from neb_dynamics.retropaths_queue import RetropathsNEBQueue
from neb_dynamics.retropaths_workflow import create_workspace, default_workspace_name
from neb_dynamics.scripts import main_cli


def test_default_workspace_name_includes_environment_suffix():
    name = default_workspace_name("C=CC(O)CC=C", "O")
    assert name.startswith("netgen-")
    assert "env-o" in name


def test_create_workspace_writes_workspace_json(tmp_path):
    workspace = create_workspace(
        root_smiles="C=CC(O)CC=C",
        environment_smiles="O",
        inputs_fp=tmp_path / "inputs.toml",
        reactions_fp=tmp_path / "reactions.p",
        directory=tmp_path / "run",
    )

    assert workspace.workspace_fp.exists()
    assert workspace.directory == (tmp_path / "run").resolve()
    assert workspace.reactions_path == (tmp_path / "reactions.p").resolve()


def test_status_command_reads_workspace_and_writes_summary(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "run"
    workspace_dir.mkdir()
    (workspace_dir / "workspace.json").write_text("{}")
    status_fp = workspace_dir / "status.html"
    status_fp.write_text("<html>ok</html>")

    queue = RetropathsNEBQueue()
    queue.items = []
    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0)
    pot.graph.add_edge(0, 0)

    monkeypatch.setattr(
        main_cli.RetropathsWorkspace,
        "read",
        staticmethod(
            lambda _path: SimpleNamespace(
                workdir=str(workspace_dir),
                root_smiles="C=CC(O)CC=C",
                environment_smiles="O",
            )
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "write_status_html",
        lambda workspace, kmc_temperature_kelvin=298.15, kmc_initial_conditions=None: (queue, pot, status_fp),
    )
    monkeypatch.setattr(
        main_cli,
        "summarize_queue",
        lambda _queue: {"items": 0, "completed": 0, "running": 0, "pending": 0, "failed": 0, "incompatible": 0},
    )
    monkeypatch.setattr(main_cli.webbrowser, "open", lambda _url: True)

    main_cli.status_cmd(directory=str(workspace_dir), no_open=True)

    assert status_fp.read_text() == "<html>ok</html>"


def test_status_command_parses_kmc_inputs(monkeypatch, tmp_path):
    workspace_dir = tmp_path / "run"
    workspace_dir.mkdir()
    (workspace_dir / "workspace.json").write_text("{}")
    status_fp = workspace_dir / "status.html"
    status_fp.write_text("<html>ok</html>")

    queue = RetropathsNEBQueue()
    queue.items = []
    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0)
    pot.graph.add_edge(0, 0)

    captured = {}

    monkeypatch.setattr(
        main_cli.RetropathsWorkspace,
        "read",
        staticmethod(
            lambda _path: SimpleNamespace(
                workdir=str(workspace_dir),
                root_smiles="C=CC(O)CC=C",
                environment_smiles="O",
            )
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "write_status_html",
        lambda workspace, kmc_temperature_kelvin=298.15, kmc_initial_conditions=None: captured.update(
            {"temperature": kmc_temperature_kelvin, "initial_conditions": kmc_initial_conditions}
        ) or (queue, pot, status_fp),
    )
    monkeypatch.setattr(
        main_cli,
        "summarize_queue",
        lambda _queue: {"items": 0, "completed": 0, "running": 0, "pending": 0, "failed": 0, "incompatible": 0},
    )

    main_cli.status_cmd(
        directory=str(workspace_dir),
        temperature=425.0,
        initial_conditions=["0=0.4", "2=0.6"],
        no_open=True,
    )

    assert captured == {"temperature": 425.0, "initial_conditions": {0: 0.4, 2: 0.6}}


def test_netgen_smiles_passes_reactions_fp(monkeypatch, tmp_path):
    reactions_fp = tmp_path / "reactions.p"
    reactions_fp.write_text("x")
    status_fp = tmp_path / "status.html"
    status_fp.write_text("<html>ok</html>")

    created = {}

    monkeypatch.setattr(
        main_cli,
        "create_workspace",
        lambda **kwargs: created.update(kwargs) or SimpleNamespace(
            workdir=str(tmp_path),
            root_smiles=kwargs["root_smiles"],
            environment_smiles=kwargs["environment_smiles"],
            reactions_path=reactions_fp.resolve(),
        ),
    )
    monkeypatch.setattr(main_cli, "prepare_neb_workspace", lambda _workspace: None)
    monkeypatch.setattr(
        main_cli,
        "write_status_html",
        lambda workspace, kmc_temperature_kelvin=298.15, kmc_initial_conditions=None: (
            RetropathsNEBQueue(),
            SimpleNamespace(graph=nx.DiGraph()),
            status_fp,
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "run_netgen_smiles_workflow",
        lambda workspace, progress=None: (
            RetropathsNEBQueue(),
            SimpleNamespace(graph=nx.DiGraph()),
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "summarize_queue",
        lambda _queue: {"items": 0, "completed": 0, "running": 0, "pending": 0, "failed": 0, "incompatible": 0},
    )

    pot = SimpleNamespace(graph=nx.DiGraph())
    pot.graph.add_node(0, endpoint_optimized=False)
    monkeypatch.setattr(
        main_cli,
        "run_netgen_smiles_workflow",
        lambda workspace, progress=None: (RetropathsNEBQueue(), pot),
    )

    main_cli.netgen_smiles(
        smiles="C",
        inputs=str(tmp_path / "inputs.toml"),
        reactions_fp=str(reactions_fp),
        no_open=True,
    )

    assert Path(created["reactions_fp"]).resolve() == reactions_fp.resolve()
