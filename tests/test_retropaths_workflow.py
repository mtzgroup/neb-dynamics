from types import SimpleNamespace
from pathlib import Path

import networkx as nx
import numpy as np
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.molecule import Molecule
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.pot import Pot
from neb_dynamics.retropaths_workflow import (
    RetropathsWorkspace,
    load_partial_annotated_pot,
    prepare_optimized_neb_pot,
    prepare_optimized_neb_pot_from_pot,
    write_status_html,
)


def _node(x: float) -> StructureNode:
    node = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
            symbols=["H", "H"],
            charge=0,
            multiplicity=1,
        )
    )
    node.graph = Molecule.from_smiles("[H][H]")
    node._cached_energy = 0.0
    node._cached_gradient = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return node


def _graphless_node(x: float) -> StructureNode:
    node = _node(x)
    node.has_molecular_graph = False
    node.graph = None
    return node


def _fake_run_inputs(engine) -> SimpleNamespace:
    return SimpleNamespace(
        engine=engine,
        chain_inputs=ChainInputs(),
        path_min_inputs=SimpleNamespace(do_elem_step_checks=False),
    )


def test_prepare_optimized_neb_pot_batches_all_unoptimized_nodes(tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule(), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.5))
    pot.graph.add_node(1, td=_node(0.9))
    pot.graph.add_edge(1, 0, reaction="1->0")
    pot.write_to_disk(workspace.neb_pot_fp)

    calls = {"n": 0}

    class _BatchEngine:
        def compute_geometry_optimizations(self, nodes, keywords=None):
            calls["n"] += 1
            out = []
            for node in nodes:
                moved = node.update_coords(
                    node.coords + np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
                )
                moved._cached_energy = 0.0
                moved._cached_gradient = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                out.append([moved])
            return out

    optimized = prepare_optimized_neb_pot(
        workspace=workspace,
        run_inputs=_fake_run_inputs(_BatchEngine()),
    )

    assert calls["n"] == 1
    assert optimized.graph.nodes[0]["endpoint_optimized"] is True
    assert optimized.graph.nodes[1]["endpoint_optimized"] is True
    persisted = Pot.read_from_disk(workspace.neb_pot_fp)
    assert np.allclose(
        persisted.graph.nodes[0]["td"].coords[0],
        np.array([2.0, 0.0, 0.0]),
    )


def test_prepare_optimized_neb_pot_reports_progress(tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule(), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.5))
    pot.graph.add_node(1, td=_node(0.9))

    class _BatchEngine:
        def compute_geometry_optimizations(self, nodes, keywords=None):
            return [[node.copy()] for node in nodes]

    messages = []
    prepare_optimized_neb_pot_from_pot(
        pot=pot,
        workspace=workspace,
        run_inputs=_fake_run_inputs(_BatchEngine()),
        progress=messages.append,
    )

    assert any("Optimizing 2 endpoint structures" in msg for msg in messages)
    assert any("Submitting 2 endpoint geometry optimizations as one batch." in msg for msg in messages)
    assert any("Endpoint optimization finished: 2 succeeded, 0 failed." in msg for msg in messages)


def test_prepare_optimized_neb_pot_persists_endpoint_results(tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule(), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.5))

    class _SavableResult:
        def __init__(self):
            self.saved_to = None

        def model_dump(self):
            return {"files": {"stdout": Path("/tmp/fake.out")}}

        def save(self, fp):
            self.saved_to = Path(fp)
            Path(fp).write_text("{}")

    class _BatchEngine:
        def compute_geometry_optimizations(self, nodes, keywords=None):
            out = []
            for node in nodes:
                moved = node.copy()
                moved._cached_result = _SavableResult()
                out.append([moved])
            return out

    optimized = prepare_optimized_neb_pot_from_pot(
        pot=pot,
        workspace=workspace,
        run_inputs=_fake_run_inputs(_BatchEngine()),
    )

    result_fp = optimized.graph.nodes[0]["endpoint_optimization_result_fp"]
    assert result_fp.endswith("endpoint_optimizations/node_0.qcio")
    assert Path(result_fp).exists()


def test_write_status_html_writes_visible_pot_sections(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="O",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    neb_pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    chain = Chain.model_validate(
        {"nodes": [_node(0.8), _node(1.0), _node(1.2)], "parameters": ChainInputs()}
    )
    reverse_bad = Chain.model_validate(
        {"nodes": [_node(1.2), _node(1.0), _node(0.8)], "parameters": ChainInputs()}
    )
    reverse_bad.nodes[0]._cached_energy = 1.2
    reverse_bad.nodes[1]._cached_energy = 1.0
    reverse_bad.nodes[2]._cached_energy = 0.8
    neb_pot.graph.add_node(0, molecule=_node(0.8).graph, td=_node(0.8))
    neb_pot.graph.add_node(1, molecule=_node(1.2).graph, td=_node(1.2))
    neb_pot.graph.add_edge(0, 1, reaction="0->1", list_of_nebs=[chain], barrier=1.23)
    neb_pot.graph.add_edge(1, 0, reaction="1->0", list_of_nebs=[reverse_bad], barrier=0.0)
    neb_pot.write_to_disk(workspace.neb_pot_fp)

    workspace.queue_fp.write_text('{"attempted_pairs": {}, "items": [], "version": 1}')

    class _FakeRetropathsPot:
        def __init__(self):
            self.graph = nx.DiGraph()
            self.graph.add_node(0)

        def draw(self, string_mode=True, leaves=False):
            return "<div>retropaths-graph</div>"

    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.load_retropaths_pot",
        lambda _workspace: _FakeRetropathsPot(),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.load_partial_annotated_pot",
        lambda _workspace: neb_pot,
    )
    monkeypatch.setattr(
        "neb_dynamics.scripts.main_cli._build_chain_visualizer_html",
        lambda chain, chain_trajectory=None, tree_layers=None: "<html>viewer</html>",
    )

    _queue, _pot, status_fp = write_status_html(
        workspace,
        kmc_temperature_kelvin=350.0,
        kmc_initial_conditions={1: 0.25},
    )

    html = status_fp.read_text()
    assert "Retropaths Pot" in html
    assert "NEB Pot" in html
    assert "Graph Delta" in html
    assert "Network Explorer" in html
    assert "Targeted Reaction" in html
    assert "Template Data" in html
    assert "Template Visualization" in html
    assert "Prepared Queue Edges" in html
    assert "Completed NEB Edges" in html
    assert "Kinetic Monte Carlo" in html
    assert 'id="retropaths-network-payload"' in html
    assert 'id="neb-network-payload"' in html
    assert 'id="kmc-temperature"' in html
    assert 'data-node-id="0"' in html
    assert 'data-node-id="1"' in html
    assert "350.00" in html
    assert "Suppressed KMC Edges" in html
    assert "start_endpoint_is_chain_maximum" in html
    assert "edge_visualizations/edge_0_1.html" in html
    assert (workspace.edge_visualizations_dir / "edge_0_1.html").read_text() == "<html>viewer</html>"


def test_load_partial_annotated_pot_expands_recursive_history_into_elementary_edges(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    source_pot = Pot(root=Molecule(), target=Molecule())
    source_pot.graph = nx.DiGraph()
    source_pot.graph.add_node(0, molecule=None, td=_graphless_node(60.0))
    source_pot.graph.add_node(1, molecule=None, td=_graphless_node(0.5))
    source_pot.graph.add_edge(1, 0, reaction="1->0")
    source_pot.write_to_disk(workspace.neb_pot_fp)

    workspace.queue_fp.write_text(
        """{
  "attempted_pairs": {},
  "items": [
    {
      "job_id": "1->0",
      "source_node": 1,
      "target_node": 0,
      "attempt_key": "abc",
      "status": "completed",
      "result_dir": "fake_history",
      "reaction": "1->0"
    }
  ],
  "version": 1
}"""
    )

    mid = _graphless_node(30.0)
    ts_a = _graphless_node(12.0)
    ts_a._cached_energy = 0.2
    ts_b = _graphless_node(45.0)
    ts_b._cached_energy = 0.2
    chain_a = Chain.model_validate(
        {"nodes": [_graphless_node(0.5), ts_a, mid.copy()], "parameters": ChainInputs()}
    )
    chain_b = Chain.model_validate(
        {"nodes": [mid.copy(), ts_b, _graphless_node(60.0)], "parameters": ChainInputs()}
    )

    class _LeafData:
        def __init__(self, chain):
            self.chain_trajectory = [chain]

    class _Leaf:
        def __init__(self, chain):
            self.data = _LeafData(chain)

    class _History:
        ordered_leaves = [_Leaf(chain_a), _Leaf(chain_b)]

    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.TreeNode.read_from_disk",
        staticmethod(lambda **kwargs: _History()),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow._nodes_identical",
        lambda node1, node2, _chain_inputs: np.allclose(node1.coords, node2.coords),
    )

    annotated = load_partial_annotated_pot(workspace)

    assert annotated.graph.number_of_nodes() == 3
    assert annotated.graph.number_of_edges() == 4
    assert (1, 2) in annotated.graph.edges
    assert (2, 0) in annotated.graph.edges
    assert (2, 1) in annotated.graph.edges
    assert (0, 2) in annotated.graph.edges
    assert len(annotated.graph.edges[(1, 2)]["list_of_nebs"]) == 1
    assert len(annotated.graph.edges[(2, 0)]["list_of_nebs"]) == 1
    assert len(annotated.graph.edges[(2, 1)]["list_of_nebs"]) == 1
    assert len(annotated.graph.edges[(0, 2)]["list_of_nebs"]) == 1
    assert annotated.graph.edges[(1, 2)]["reaction"] == "1->0(step 1)"
    assert annotated.graph.edges[(2, 0)]["reaction"] == "1->0(step 2)"
    assert annotated.graph.edges[(2, 1)]["reaction"] == "1->0(step 1)"
    assert annotated.graph.edges[(0, 2)]["reaction"] == "1->0(step 2)"
