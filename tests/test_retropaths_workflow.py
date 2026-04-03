from types import SimpleNamespace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from qcio import Structure

import neb_dynamics.retropaths_workflow as workflow
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.molecule import Molecule
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.pot import Pot
from neb_dynamics.retropaths_workflow import (
    RetropathsWorkspace,
    _find_existing_network_node,
    _build_network_explorer_payload,
    _write_edge_visualizations,
    add_manual_edge,
    apply_reactions_to_node,
    load_partial_annotated_pot,
    materialize_drive_graph,
    prepare_optimized_neb_pot,
    prepare_optimized_neb_pot_from_pot,
    run_hessian_sample_for_edge,
    run_hessian_sample_for_node,
    write_status_html,
)
from neb_dynamics.retropaths_compat import structure_node_from_graph_like_molecule


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


def _smiles_node(smiles: str, energy: float = 0.0) -> StructureNode:
    molecule = Molecule.from_smiles(smiles)
    node = structure_node_from_graph_like_molecule(molecule)
    node.graph = molecule.copy()
    node.has_molecular_graph = True
    node._cached_energy = energy
    node._cached_gradient = np.zeros_like(node.coords).tolist()
    return node


def _fake_run_inputs(engine) -> SimpleNamespace:
    return SimpleNamespace(
        engine=engine,
        chain_inputs=ChainInputs(),
        path_min_inputs=SimpleNamespace(do_elem_step_checks=False),
    )


def test_ensure_retropaths_available_reports_missing_repo(monkeypatch, tmp_path):
    missing_repo = tmp_path / "missing-retropaths"
    monkeypatch.setattr(workflow, "_retropaths_repo", lambda: missing_repo)

    with pytest.raises(RuntimeError, match="optional `retropaths` repository"):
        workflow.ensure_retropaths_available(feature="`netgen-smiles`")


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
    assert "Right-click inside the graph for tools." in html
    assert 'data-network-tool="zoom-in"' in html
    assert 'addEventListener("contextmenu"' in html
    assert 'addEventListener("wheel"' in html
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
    root_mol = Molecule.from_smiles("CCCO")
    product_mol = Molecule.from_smiles("COCC")
    mid_mol = Molecule.from_smiles("CC(O)C")
    source_pot.graph.add_node(0, molecule=root_mol, td=_smiles_node("CCCO", 0.0))
    source_pot.graph.add_node(1, molecule=product_mol, td=_smiles_node("COCC", 0.0))
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

    mid = _smiles_node("CC(O)C", 0.1)
    ts_a = _smiles_node("CC(O)C", 0.2)
    ts_b = _smiles_node("CC(O)C", 0.2)
    chain_a = Chain.model_validate(
        {"nodes": [_smiles_node("COCC", 0.0), ts_a, mid.copy()], "parameters": ChainInputs()}
    )
    chain_b = Chain.model_validate(
        {"nodes": [mid.copy(), ts_b, _smiles_node("CCCO", 0.0)], "parameters": ChainInputs()}
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
    annotated = load_partial_annotated_pot(workspace)

    assert annotated.graph.number_of_nodes() == 3
    assert annotated.graph.number_of_edges() == 5
    assert (1, 0) in annotated.graph.edges
    assert (1, 2) in annotated.graph.edges
    assert (2, 0) in annotated.graph.edges
    assert (2, 1) in annotated.graph.edges
    assert (0, 2) in annotated.graph.edges
    assert len(annotated.graph.edges[(1, 0)]["list_of_nebs"]) == 0
    assert len(annotated.graph.edges[(1, 2)]["list_of_nebs"]) == 1
    assert len(annotated.graph.edges[(2, 0)]["list_of_nebs"]) == 1
    assert len(annotated.graph.edges[(2, 1)]["list_of_nebs"]) == 1
    assert len(annotated.graph.edges[(0, 2)]["list_of_nebs"]) == 1
    assert annotated.graph.edges[(1, 2)]["reaction"] == "1->0(step 1)"
    assert annotated.graph.edges[(2, 0)]["reaction"] == "1->0(step 2)"
    assert annotated.graph.edges[(2, 1)]["reaction"] == "1->0(step 1)"
    assert annotated.graph.edges[(0, 2)]["reaction"] == "1->0(step 2)"


def test_load_partial_annotated_pot_preserves_existing_target_ids_during_split(monkeypatch, tmp_path):
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
    root_mol = Molecule.from_smiles("CCCO")
    product_mol = Molecule.from_smiles("COCC")
    mid_mol = Molecule.from_smiles("CC(O)C")
    other_mol = Molecule.from_smiles("CCCN")
    source_pot.graph.add_node(0, molecule=root_mol, td=_smiles_node("CCCO", 0.0))
    source_pot.graph.add_node(1, molecule=product_mol, td=_smiles_node("COCC", 0.0))
    source_pot.graph.add_node(2, molecule=other_mol, td=_smiles_node("CCCN", 0.0))
    source_pot.graph.add_edge(0, 1, reaction="0->1")
    source_pot.graph.add_edge(0, 2, reaction="0->2")
    source_pot.write_to_disk(workspace.neb_pot_fp)

    workspace.queue_fp.write_text(
        """{
  "attempted_pairs": {},
  "items": [
    {
      "job_id": "0->1",
      "source_node": 0,
      "target_node": 1,
      "attempt_key": "abc",
      "status": "completed",
      "result_dir": "fake_history",
      "reaction": "0->1"
    }
  ],
  "version": 1
}"""
    )

    mid = _smiles_node("CC(O)C", 0.1)
    ts_a = _smiles_node("CC(O)C", 0.2)
    ts_b = _smiles_node("CC(O)C", 0.2)
    chain_a = Chain.model_validate(
        {"nodes": [_smiles_node("CCCO", 0.0), ts_a, mid.copy()], "parameters": ChainInputs()}
    )
    chain_b = Chain.model_validate(
        {"nodes": [mid.copy(), ts_b, _smiles_node("COCC", 0.0)], "parameters": ChainInputs()}
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
    annotated = load_partial_annotated_pot(workspace)

    assert sorted(annotated.graph.nodes) == [0, 1, 2, 3]
    assert annotated.graph.nodes[1]["molecule"].force_smiles() == product_mol.force_smiles()
    assert annotated.graph.nodes[3]["molecule"].force_smiles() == mid_mol.force_smiles()
    assert (0, 3) in annotated.graph.edges
    assert (3, 1) in annotated.graph.edges
    assert (1, 3) in annotated.graph.edges
    assert (3, 0) in annotated.graph.edges
    assert annotated.graph.edges[(0, 3)]["reaction"] == "0->1(step 1)"
    assert annotated.graph.edges[(3, 1)]["reaction"] == "0->1(step 2)"


def test_load_partial_annotated_pot_reuses_existing_same_species_intermediate(monkeypatch, tmp_path):
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
    root_mol = Molecule.from_smiles("CCCO")
    product_mol = Molecule.from_smiles("COCC")
    mid_mol = Molecule.from_smiles("CC(O)C")
    source_pot.graph.add_node(0, molecule=root_mol, td=_smiles_node("CCCO", 0.0))
    source_pot.graph.add_node(1, molecule=product_mol, td=_smiles_node("COCC", 0.0))
    source_pot.graph.add_node(5, molecule=mid_mol, td=_smiles_node("CC(O)C", 0.0))
    source_pot.graph.add_edge(0, 1, reaction="0->1")
    source_pot.write_to_disk(workspace.neb_pot_fp)

    workspace.queue_fp.write_text(
        """{
  "attempted_pairs": {},
  "items": [
    {
      "job_id": "0->1",
      "source_node": 0,
      "target_node": 1,
      "attempt_key": "abc",
      "status": "completed",
      "result_dir": "fake_history",
      "reaction": "0->1"
    }
  ],
  "version": 1
}"""
    )

    chain_a = Chain.model_validate(
        {
            "nodes": [
                _smiles_node("CCCO", 0.0),
                _smiles_node("CC(O)C", 0.2),
                _smiles_node("CC(O)C", 0.1),
            ],
            "parameters": ChainInputs(),
        }
    )
    chain_b = Chain.model_validate(
        {
            "nodes": [
                _smiles_node("CC(O)C", 0.1),
                _smiles_node("CC(O)C", 0.2),
                _smiles_node("COCC", 0.0),
            ],
            "parameters": ChainInputs(),
        }
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

    annotated = load_partial_annotated_pot(workspace)

    assert 5 in annotated.graph.nodes
    assert sorted(annotated.graph.nodes) == [0, 1, 5]
    assert (0, 5) in annotated.graph.edges
    assert (5, 1) in annotated.graph.edges


def test_write_edge_visualizations_removes_stale_edge_files(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()
    workspace.inputs_fp = str(tmp_path / "inputs.toml")
    Path(workspace.inputs_fp).write_text("")

    stale_fp = workspace.edge_visualizations_dir / "edge_9_9.html"
    workspace.edge_visualizations_dir.mkdir(parents=True, exist_ok=True)
    stale_fp.write_text("stale", encoding="utf-8")

    pot = Pot(root=Molecule(), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_graphless_node(0.0), molecule=None)
    pot.graph.add_node(1, td=_graphless_node(1.0), molecule=None)
    chain = Chain.model_validate(
        {"nodes": [_graphless_node(0.0), _graphless_node(1.0)], "parameters": ChainInputs()}
    )
    pot.graph.add_edge(0, 1, list_of_nebs=[chain], reaction="demo")

    monkeypatch.setattr(
        "neb_dynamics.scripts.main_cli._build_chain_visualizer_html",
        lambda chain, chain_trajectory=None: "<html>viewer</html>",
    )

    rows = _write_edge_visualizations(workspace=workspace, pot=pot)

    assert rows[0]["href"] == "edge_0_1.html"
    assert rows[0]["source_structure"]["symbols"] == ["H", "H"]
    assert rows[0]["target_structure"]["symbols"] == ["H", "H"]
    assert not stale_fp.exists()
    assert (workspace.edge_visualizations_dir / "edge_0_1.html").read_text() == "<html>viewer</html>"
    assert (workspace.edge_visualizations_dir / "edge_0_1.meta.json").exists()


def test_write_edge_visualizations_reuses_unchanged_cached_viewers(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()
    workspace.inputs_fp = str(tmp_path / "inputs.toml")
    Path(workspace.inputs_fp).write_text("")

    pot = Pot(root=Molecule(), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_graphless_node(0.0), molecule=None)
    pot.graph.add_node(1, td=_graphless_node(1.0), molecule=None)
    chain = Chain.model_validate(
        {"nodes": [_graphless_node(0.0), _graphless_node(1.0)], "parameters": ChainInputs()}
    )
    pot.graph.add_edge(0, 1, list_of_nebs=[chain], reaction="demo")

    calls = {"count": 0}

    def _build_html(chain, chain_trajectory=None):
        calls["count"] += 1
        return "<html>viewer</html>"

    monkeypatch.setattr(
        "neb_dynamics.scripts.main_cli._build_chain_visualizer_html",
        _build_html,
    )

    rows_first = _write_edge_visualizations(workspace=workspace, pot=pot)
    rows_second = _write_edge_visualizations(workspace=workspace, pot=pot)

    assert calls["count"] == 1
    assert rows_first == rows_second
    assert rows_second[0]["source_structure"]["symbols"] == ["H", "H"]
    assert rows_second[0]["target_structure"]["symbols"] == ["H", "H"]


def test_find_existing_network_node_uses_graph_identity_only():
    pot = Pot(root=Molecule(), target=Molecule())
    pot.graph = nx.DiGraph()
    existing = _node(0.0)
    existing.graph = Molecule.from_smiles("CC")
    existing.has_molecular_graph = True
    pot.graph.add_node(0, td=existing, molecule=existing.graph.copy())

    candidate = _node(999.0)
    candidate.graph = Molecule.from_smiles("CC")
    candidate.has_molecular_graph = True

    chain_inputs = ChainInputs(node_rms_thre=0.5, node_ene_thre=1.0)

    assert _find_existing_network_node(pot, candidate, chain_inputs, 0.01, 0.01) == 0


def test_apply_reactions_to_node_merges_new_products(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(
        0,
        molecule=Molecule.from_smiles("C"),
        td=structure_node_from_graph_like_molecule(Molecule.from_smiles("C")),
        converged=False,
    )

    class _FakeRetropathsPot:
        def __init__(self, root, environment, rxn_name=None):
            self.graph = nx.DiGraph()
            self.graph.add_node(0, molecule=root, converged=False)

        def grow_this_node(self, leaf, library, filter_minor_products=True, use_father_error=False):
            self.graph.add_node(1, molecule=Molecule.from_smiles("CC"), converged=False)
            self.graph.add_edge(1, 0, reaction="Fake Growth")

    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.load_partial_annotated_pot",
        lambda _workspace: pot,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.build_retropaths_neb_queue",
        lambda pot, queue_fp, overwrite=False: SimpleNamespace(find_item=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow._load_retropaths_classes",
        lambda: (
            SimpleNamespace(pload=lambda _fp: {"fake": object()}),
            Molecule,
            _FakeRetropathsPot,
        ),
    )

    result = apply_reactions_to_node(workspace, 0)
    persisted = Pot.read_from_disk(workspace.neb_pot_fp)

    assert result["added_nodes"] == 1
    assert persisted.graph.number_of_nodes() == 2
    assert persisted.graph.has_edge(1, 0)
    assert persisted.graph.edges[(1, 0)]["reaction"] == "Fake Growth"


def test_apply_reactions_to_node_uses_canonical_child_structure(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    source_td = structure_node_from_graph_like_molecule(Molecule.from_smiles("C"))
    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, molecule=Molecule.from_smiles("C"), td=source_td, converged=False)

    class _FakeRetropathsPot:
        def __init__(self, root, environment, rxn_name=None):
            self.graph = nx.DiGraph()
            self.graph.add_node(0, molecule=root, converged=False)

        def grow_this_node(self, leaf, library, filter_minor_products=True, use_father_error=False):
            self.graph.add_node(1, molecule=Molecule.from_smiles("CC"), converged=False)
            self.graph.add_edge(1, 0, reaction="Fake Growth")

    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.load_partial_annotated_pot",
        lambda _workspace: pot,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.build_retropaths_neb_queue",
        lambda pot, queue_fp, overwrite=False: SimpleNamespace(find_item=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow._load_retropaths_classes",
        lambda: (
            SimpleNamespace(pload=lambda _fp: {"Fake Growth": object()}),
            Molecule,
            _FakeRetropathsPot,
        ),
    )

    apply_reactions_to_node(workspace, 0)

    assert pot.graph.nodes[1]["td"] is not None
    assert pot.graph.nodes[1]["td"].graph.force_smiles() == Molecule.from_smiles("CC").force_smiles()
    assert pot.graph.nodes[1]["endpoint_optimized"] is False


def test_build_network_explorer_payload_handles_cyclic_metadata():
    graph = nx.DiGraph()
    node_cycle: dict[str, object] = {}
    node_cycle["self"] = node_cycle
    edge_cycle: dict[str, object] = {}
    edge_cycle["self"] = edge_cycle
    template_cycle: dict[str, object] = {"name": "Fake Growth"}
    template_cycle["self"] = template_cycle

    graph.add_node(0, molecule="C", metadata=node_cycle)
    graph.add_node(1, molecule="CC")
    graph.add_edge(1, 0, reaction="Fake Growth", metadata=edge_cycle)

    payload = _build_network_explorer_payload(
        graph,
        template_payloads={
            "Fake Growth": {
                "name": "Fake Growth",
                "data": template_cycle,
                "visualization_html": "",
            }
        },
    )

    assert payload["nodes"][0]["data"]["metadata"]["self"].startswith("{")
    assert payload["edges"][0]["data"]["metadata"]["self"].startswith("{")
    assert payload["edges"][0]["template"]["data"]["self"].startswith("{")


def test_add_manual_edge_persists_new_edge(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, molecule=Molecule.from_smiles("C"), td=structure_node_from_graph_like_molecule(Molecule.from_smiles("C")))
    pot.graph.add_node(1, molecule=Molecule.from_smiles("CC"), td=structure_node_from_graph_like_molecule(Molecule.from_smiles("CC")))
    pot.write_to_disk(workspace.neb_pot_fp)

    fake_item = SimpleNamespace(status="pending", error=None)
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.ensure_queue_item_for_edge",
        lambda pot, source_node, target_node, queue_fp, overwrite=False: SimpleNamespace(find_item=lambda *_args, **_kwargs: fake_item),
    )

    result = add_manual_edge(
        workspace,
        source_node=1,
        target_node=0,
        reaction_label="Manual test edge",
    )
    persisted = Pot.read_from_disk(workspace.neb_pot_fp)

    assert result["added"] is True
    assert result["queue_status"] == "pending"
    assert persisted.graph.has_edge(1, 0)
    assert persisted.graph.edges[(1, 0)]["reaction"] == "Manual test edge"


def test_materialize_drive_graph_merges_overlay_without_dropping_base_nodes(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    base_pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    base_pot.graph = nx.DiGraph()
    base_pot.graph.add_node(0, molecule=Molecule.from_smiles("C"))
    base_pot.graph.add_node(1, molecule=Molecule.from_smiles("CC"))
    base_pot.graph.add_node(2, molecule=Molecule.from_smiles("CCC"))
    base_pot.graph.add_edge(1, 0, reaction="1->0")
    base_pot.graph.add_edge(2, 1, reaction="2->1")
    base_pot.write_to_disk(workspace.neb_pot_fp)

    overlay_pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    overlay_pot.graph = nx.DiGraph()
    overlay_pot.graph.add_node(0, molecule=Molecule.from_smiles("C"))

    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.load_partial_annotated_pot",
        lambda _workspace: overlay_pot,
    )
    monkeypatch.setattr(
        "neb_dynamics.retropaths_workflow.build_retropaths_neb_queue",
        lambda pot, queue_fp, overwrite=False: SimpleNamespace(find_item=lambda *_args, **_kwargs: None),
    )

    merged = materialize_drive_graph(workspace)

    assert merged.graph.number_of_nodes() == 3
    assert merged.graph.has_edge(1, 0)
    assert merged.graph.has_edge(2, 1)


def test_run_hessian_sample_for_node_rejects_nonpositive_dr(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.3), molecule=Molecule.from_smiles("[H][H]"))

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: pot)

    with pytest.raises(ValueError, match="must be positive"):
        run_hessian_sample_for_node(workspace, 0, dr=0.0)


def test_run_hessian_sample_for_node_accepts_compute_hessian_only_engine(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.3), molecule=Molecule.from_smiles("[H][H]"))

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: pot)

    class _Engine:
        def compute_hessian(self, node):
            return np.eye(node.coords.size)

        def compute_geometry_optimization(self, node, keywords=None):
            optimized = node.copy()
            optimized.graph = Molecule.from_smiles("C")
            optimized.has_molecular_graph = True
            optimized._cached_energy = 0.0
            optimized._cached_gradient = np.zeros_like(optimized.coords).tolist()
            return [optimized]

    fake_inputs = SimpleNamespace(
        engine=_Engine(),
        engine_name="ase",
        program="omol25",
        chain_inputs=ChainInputs(),
        network_inputs=SimpleNamespace(
            collapse_node_rms_thre=5.0,
            collapse_node_ene_thre=5.0,
        ),
    )
    monkeypatch.setattr(workflow.RunInputs, "open", staticmethod(lambda _fp: fake_inputs))

    out = run_hessian_sample_for_node(workspace, 0, dr=0.1, max_candidates=2)

    assert isinstance(out, dict)
    assert out["added_nodes"] >= 1


def test_run_hessian_sample_for_edge_requires_completed_chain(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.0), molecule=Molecule.from_smiles("[H][H]"))
    pot.graph.add_node(1, td=_node(0.6), molecule=Molecule.from_smiles("[H][H]"))
    pot.graph.add_edge(0, 1, reaction="0->1", list_of_nebs=[])

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: pot)

    with pytest.raises(ValueError, match="requires a completed NEB chain"):
        run_hessian_sample_for_edge(workspace, 0, 1, dr=0.1)


def test_run_hessian_sample_for_edge_batches_all_candidates_for_chemcloud(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.0), molecule=Molecule.from_smiles("[H][H]"))
    pot.graph.add_node(1, td=_node(0.8), molecule=Molecule.from_smiles("[H][H]"))
    peak_chain = Chain.model_validate(
        {"nodes": [_node(0.0), _node(0.4), _node(0.8)], "parameters": ChainInputs()}
    )
    pot.graph.add_edge(0, 1, reaction="0->1", list_of_nebs=[peak_chain])

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: pot)

    calls = {"batch": 0, "batch_size": 0, "serial": 0}

    class _HessianResult:
        def __init__(self, modes):
            self.results = SimpleNamespace(
                normal_modes_cartesian=modes,
                freqs_wavenumber=[-100.0 for _ in modes],
            )

    class _Engine:
        compute_program = "chemcloud"

        def _compute_hessian_result(self, _node):
            mode = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
            # 60 modes -> 120 displaced candidates, clipped to 100 in workflow.
            return _HessianResult([mode.copy() for _ in range(60)])

        def compute_geometry_optimizations(self, nodes, keywords=None):
            calls["batch"] += 1
            calls["batch_size"] = len(nodes)
            return [[node.copy()] for node in nodes]

        def compute_geometry_optimization(self, node, keywords=None):
            calls["serial"] += 1
            raise AssertionError("Serial optimization should not be called for ChemCloud Hessian edge sampling.")

    fake_inputs = SimpleNamespace(
        engine=_Engine(),
        engine_name="chemcloud",
        chain_inputs=ChainInputs(),
        network_inputs=SimpleNamespace(
            collapse_node_rms_thre=5.0,
            collapse_node_ene_thre=5.0,
        ),
    )
    monkeypatch.setattr(workflow.RunInputs, "open", staticmethod(lambda _fp: fake_inputs))

    out = run_hessian_sample_for_edge(workspace, 0, 1, dr=0.1)

    assert isinstance(out, dict)
    assert calls["batch"] == 1
    assert calls["batch_size"] == 100
    assert calls["serial"] == 0


def test_run_hessian_sample_for_edge_respects_max_candidates(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.0), molecule=Molecule.from_smiles("[H][H]"))
    pot.graph.add_node(1, td=_node(0.8), molecule=Molecule.from_smiles("[H][H]"))
    peak_chain = Chain.model_validate(
        {"nodes": [_node(0.0), _node(0.4), _node(0.8)], "parameters": ChainInputs()}
    )
    pot.graph.add_edge(0, 1, reaction="0->1", list_of_nebs=[peak_chain])

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: pot)

    calls = {"batch": 0, "batch_size": 0}

    class _HessianResult:
        def __init__(self, modes):
            self.results = SimpleNamespace(
                normal_modes_cartesian=modes,
                freqs_wavenumber=[-100.0 for _ in modes],
            )

    class _Engine:
        compute_program = "chemcloud"

        def _compute_hessian_result(self, _node):
            mode = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
            return _HessianResult([mode.copy() for _ in range(60)])

        def compute_geometry_optimizations(self, nodes, keywords=None):
            calls["batch"] += 1
            calls["batch_size"] = len(nodes)
            return [[node.copy()] for node in nodes]

    fake_inputs = SimpleNamespace(
        engine=_Engine(),
        engine_name="chemcloud",
        chain_inputs=ChainInputs(),
        network_inputs=SimpleNamespace(
            collapse_node_rms_thre=5.0,
            collapse_node_ene_thre=5.0,
        ),
    )
    monkeypatch.setattr(workflow.RunInputs, "open", staticmethod(lambda _fp: fake_inputs))

    out = run_hessian_sample_for_edge(workspace, 0, 1, dr=0.1, max_candidates=37)

    assert isinstance(out, dict)
    assert calls["batch"] == 1
    assert calls["batch_size"] == 37


def test_run_hessian_sample_for_edge_keeps_partial_batch_success(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.0), molecule=Molecule.from_smiles("[H][H]"))
    pot.graph.add_node(1, td=_node(0.8), molecule=Molecule.from_smiles("[H][H]"))
    peak_chain = Chain.model_validate(
        {"nodes": [_node(0.0), _node(0.4), _node(0.8)], "parameters": ChainInputs()}
    )
    pot.graph.add_edge(0, 1, reaction="0->1", list_of_nebs=[peak_chain])

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: pot)

    calls = {"batch": 0, "batch_size": 0}

    class _HessianResult:
        def __init__(self, modes):
            self.results = SimpleNamespace(
                normal_modes_cartesian=modes,
                freqs_wavenumber=[-100.0 for _ in modes],
            )

    class _Engine:
        compute_program = "chemcloud"

        def _compute_hessian_result(self, _node):
            mode = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
            return _HessianResult([mode.copy()])

        def compute_geometry_optimizations(self, nodes, keywords=None):
            calls["batch"] += 1
            calls["batch_size"] = len(nodes)
            successful = nodes[1].copy()
            successful.graph = Molecule.from_smiles("C")
            successful.has_molecular_graph = True
            return [[], [successful]]

    fake_inputs = SimpleNamespace(
        engine=_Engine(),
        engine_name="chemcloud",
        chain_inputs=ChainInputs(),
        network_inputs=SimpleNamespace(
            collapse_node_rms_thre=5.0,
            collapse_node_ene_thre=5.0,
        ),
    )
    monkeypatch.setattr(workflow.RunInputs, "open", staticmethod(lambda _fp: fake_inputs))

    out = run_hessian_sample_for_edge(workspace, 0, 1, dr=0.1, max_candidates=2)

    assert isinstance(out, dict)
    assert calls["batch"] == 1
    assert calls["batch_size"] == 2
    assert out["added_nodes"] >= 1


def test_run_hessian_sample_for_edge_handles_batch_count_mismatch(monkeypatch, tmp_path):
    workspace = RetropathsWorkspace(
        workdir=str(tmp_path),
        run_name="demo",
        root_smiles="C",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, td=_node(0.0), molecule=Molecule.from_smiles("[H][H]"))
    pot.graph.add_node(1, td=_node(0.8), molecule=Molecule.from_smiles("[H][H]"))
    peak_chain = Chain.model_validate(
        {"nodes": [_node(0.0), _node(0.4), _node(0.8)], "parameters": ChainInputs()}
    )
    pot.graph.add_edge(0, 1, reaction="0->1", list_of_nebs=[peak_chain])

    monkeypatch.setattr("neb_dynamics.retropaths_workflow.materialize_drive_graph", lambda _workspace: pot)

    calls = {"batch": 0, "batch_size": 0}

    class _HessianResult:
        def __init__(self, modes):
            self.results = SimpleNamespace(
                normal_modes_cartesian=modes,
                freqs_wavenumber=[-100.0 for _ in modes],
            )

    class _Engine:
        compute_program = "chemcloud"

        def _compute_hessian_result(self, _node):
            mode = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
            return _HessianResult([mode.copy()])

        def compute_geometry_optimizations(self, nodes, keywords=None):
            calls["batch"] += 1
            calls["batch_size"] = len(nodes)
            successful = nodes[0].copy()
            successful.graph = Molecule.from_smiles("C")
            successful.has_molecular_graph = True
            # Return fewer histories than submitted candidates.
            return [[successful]]

    fake_inputs = SimpleNamespace(
        engine=_Engine(),
        engine_name="chemcloud",
        chain_inputs=ChainInputs(),
        network_inputs=SimpleNamespace(
            collapse_node_rms_thre=5.0,
            collapse_node_ene_thre=5.0,
        ),
    )
    monkeypatch.setattr(workflow.RunInputs, "open", staticmethod(lambda _fp: fake_inputs))

    out = run_hessian_sample_for_edge(workspace, 0, 1, dr=0.1, max_candidates=2)

    assert isinstance(out, dict)
    assert calls["batch"] == 1
    assert calls["batch_size"] == 2
    assert out["added_nodes"] >= 1
