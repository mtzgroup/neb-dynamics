from types import SimpleNamespace

from neb_dynamics.scripts import progress


class _Node:
    def __init__(self, has_molecular_graph: bool):
        self.has_molecular_graph = has_molecular_graph
        self.graph = object() if has_molecular_graph else None
        self.structure = object()


class _Chain:
    def __init__(self, has_molecular_graph: bool):
        self.nodes = [_Node(has_molecular_graph), _Node(has_molecular_graph)]
        self.energies_kcalmol = [0.0, 1.0]


def test_ascii_profile_skips_smiles_for_graphless_nodes(monkeypatch):
    def _fail(_structure):
        raise AssertionError("structure_to_smiles should not be called")

    monkeypatch.setattr(progress, "structure_to_smiles", _fail)
    out = progress.ascii_profile_for_chain(_Chain(has_molecular_graph=False))

    assert "start SMILES: N/A" in out
    assert "end SMILES:   N/A" in out


def test_ascii_profile_uses_smiles_for_graph_nodes(monkeypatch):
    monkeypatch.setattr(progress, "structure_to_smiles", lambda _structure: "C")
    out = progress.ascii_profile_for_chain(_Chain(has_molecular_graph=True))

    assert "start SMILES: C" in out
    assert "end SMILES:   C" in out


def test_ascii_profile_tolerates_non_finite_energies():
    chain = _Chain(has_molecular_graph=False)
    chain.energies_kcalmol = [float("nan"), 1.0]

    out = progress.ascii_profile_for_chain(chain)

    assert "node index" in out


def test_ascii_profile_axis_prefix_width_is_stable():
    plot = progress._build_ascii_energy_profile(
        energies=[-1.0e12, 2.5e12],
        labels=["0", "1"],
        width=12,
        height=5,
    )
    data_lines = plot.splitlines()[:5]
    bar_columns = {line.index("|") for line in data_lines}
    assert len(bar_columns) == 1


def test_monitor_label_is_bounded_to_fixed_width():
    printer = progress.ProgressPrinter(use_rich=False)
    printer._monitor_column_width = 20
    label = printer._format_monitor_label(
        monitor_id="branch-001",
        caption="step 125 | TS gperp: 0.0123 | max rms: 0.0456",
    )
    assert len(label) <= 20
    assert label.startswith("branch-001")


def test_compact_ascii_for_live_uses_payload_series():
    printer = progress.ProgressPrinter(use_rich=False)
    state = {
        "chain_plot_payload": {"y": [0.0, 1.0, 0.3, 1.2]},
        "ascii_plot": "unused",
    }

    out = printer._compact_ascii_for_live(state)

    assert "node index" in out


def test_visible_monitor_ids_paginates_and_rotates():
    printer = progress.ProgressPrinter(use_rich=False)
    printer._monitor_page_size = 2
    printer._monitor_page_rotate_seconds = 1.0
    monitor_ids = ["branch-0", "branch-1", "branch-2", "branch-3", "branch-4"]

    ids0, meta0 = printer._visible_monitor_ids(monitor_ids, now=100.0)
    ids1, meta1 = printer._visible_monitor_ids(monitor_ids, now=100.3)
    ids2, meta2 = printer._visible_monitor_ids(monitor_ids, now=101.1)
    ids3, meta3 = printer._visible_monitor_ids(monitor_ids, now=102.2)

    assert ids0 == ["branch-0", "branch-1"]
    assert ids1 == ["branch-0", "branch-1"]
    assert ids2 == ["branch-2", "branch-3"]
    assert ids3 == ["branch-4"]
    assert meta0["total_pages"] == 3
    assert meta2["page"] == 2
    assert meta3["page"] == 3


def test_monitor_active_state_toggles_without_dropping_payload_state():
    printer = progress.ProgressPrinter(use_rich=False)
    printer.mark_monitor_active("branch-7")
    state = printer._state_for_monitor("branch-7")
    state["ascii_plot"] = "plot"

    printer.mark_monitor_inactive("branch-7")

    payload = printer._monitors_payload()
    assert "branch-7" in payload
    assert payload["branch-7"]["active"] is False
