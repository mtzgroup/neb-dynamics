from types import SimpleNamespace

import numpy as np
import pytest
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.scripts import main_cli


def _node(x: float, e: float) -> StructureNode:
    n = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
            symbols=["H", "H"],
            charge=0,
            multiplicity=1,
        )
    )
    n._cached_energy = e
    n._cached_gradient = np.zeros((2, 3))
    n.has_molecular_graph = False
    n.graph = None
    return n


def _chain_with_energies(xs: list[float], energies: list[float]) -> Chain:
    nodes = [_node(x, e) for x, e in zip(xs, energies)]
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def _chain() -> Chain:
    return Chain.model_validate(
        {
            "nodes": [_node(0.8, 0.0), _node(1.0, 0.01), _node(1.2, 0.0)],
            "parameters": ChainInputs(),
        }
    )


def test_load_chain_for_visualization_detects_tree_folder(monkeypatch, tmp_path):
    folder = tmp_path / "res_tree"
    folder.mkdir()
    (folder / "adj_matrix.txt").write_text("1")
    expected = _chain()

    monkeypatch.setattr(
        main_cli.TreeNode,
        "read_from_disk",
        staticmethod(lambda **kwargs: SimpleNamespace(output_chain=expected)),
    )
    monkeypatch.setattr(
        main_cli,
        "_collect_tree_layers_for_visualization",
        lambda tree: [
            {
                "depth": 0,
                "groups": [
                    {
                        "label": "Node 0",
                        "node_index": 0,
                        "parent_index": None,
                        "chains": [expected],
                    }
                ],
            }
        ],
    )

    out = main_cli._load_visualization_data(folder)
    assert out.chain is expected
    assert out.chain_trajectory is None
    assert out.tree_layers is not None


def test_load_chain_for_visualization_detects_neb_file(monkeypatch, tmp_path):
    neb_fp = tmp_path / "mep_output.xyz"
    neb_fp.write_text("dummy")
    hist = tmp_path / "mep_output_history"
    hist.mkdir()
    expected = _chain()

    older = _chain()
    monkeypatch.setattr(
        main_cli.NEB,
        "read_from_disk",
        staticmethod(lambda **kwargs: SimpleNamespace(chain_trajectory=[older, expected])),
    )

    out = main_cli._load_visualization_data(neb_fp)
    assert out.chain is expected
    assert out.chain_trajectory == [older, expected]


def test_load_chain_for_visualization_detects_network_json(monkeypatch, tmp_path):
    json_fp = tmp_path / "network.json"
    json_fp.write_text("{}")
    expected = _chain()
    fake_pot = SimpleNamespace(graph=SimpleNamespace(edges=[]))

    monkeypatch.setattr(
        main_cli.Pot,
        "read_from_disk",
        staticmethod(lambda path: fake_pot),
    )
    monkeypatch.setattr(main_cli, "_find_pot_root_node_index", lambda pot: 0)
    monkeypatch.setattr(main_cli, "_find_pot_target_node_index", lambda pot, target_idx_hint=None: 1)
    monkeypatch.setattr(main_cli.nx, "has_path", lambda *args, **kwargs: True)
    monkeypatch.setattr(main_cli, "_best_path_by_apparent_barrier", lambda pot, root_idx, target_idx: ([0, 1], 1.0))
    monkeypatch.setattr(main_cli, "_path_chain_from_pot", lambda pot, path: expected)

    out = main_cli._load_visualization_data(json_fp)
    assert out.chain is expected
    assert out.network_pot is fake_pot


def test_load_network_visualization_uses_sibling_manifest_endpoint_hints(monkeypatch, tmp_path):
    json_fp = tmp_path / "rgs_network.json"
    json_fp.write_text("{}")
    (tmp_path / "rgs_request_manifest.json").write_text(
        '{"requests":[{"request_id":0,"start_index":0,"end_index":1}]}'
    )
    expected = _chain()
    fake_pot = SimpleNamespace(graph=SimpleNamespace(edges=[]))

    monkeypatch.setattr(
        main_cli.Pot,
        "read_from_disk",
        staticmethod(lambda path: fake_pot),
    )
    monkeypatch.setattr(main_cli, "_find_pot_root_node_index", lambda pot: 3)
    monkeypatch.setattr(main_cli, "_find_pot_target_node_index", lambda pot, target_idx_hint=None: target_idx_hint)
    monkeypatch.setattr(main_cli.nx, "has_path", lambda *args, **kwargs: True)

    captured = {}

    def _fake_best_path(pot, root_idx, target_idx):
        captured["root_idx"] = root_idx
        captured["target_idx"] = target_idx
        return [root_idx, target_idx], 1.0

    monkeypatch.setattr(main_cli, "_best_path_by_apparent_barrier", _fake_best_path)
    monkeypatch.setattr(main_cli, "_path_chain_from_pot", lambda pot, path: expected)

    out = main_cli._load_visualization_data(json_fp)
    assert out.network_endpoint_hints == {
        "root_index": 0,
        "target_index": 1,
        "manifest_path": str(tmp_path / "rgs_request_manifest.json"),
    }
    assert captured["root_idx"] == 0
    assert captured["target_idx"] == 1


def test_load_network_visualization_prefers_connectivity_matched_endpoints_over_manifest(monkeypatch, tmp_path):
    json_fp = tmp_path / "rgs_network.json"
    json_fp.write_text("{}")
    (tmp_path / "rgs_request_manifest.json").write_text(
        '{"requests":[{"request_id":0,"start_index":0,"end_index":1}]}'
    )
    (tmp_path / "start_rgs.xyz").write_text("dummy")
    (tmp_path / "end_rgs.xyz").write_text("dummy")
    expected = _chain()
    fake_pot = SimpleNamespace(graph=SimpleNamespace(edges=[]))

    monkeypatch.setattr(
        main_cli.Pot,
        "read_from_disk",
        staticmethod(lambda path: fake_pot),
    )
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda fp, *args, **kwargs: [_structure_at_x(2.0)] if "start_" in str(fp) else [_structure_at_x(1.0)])
    monkeypatch.setattr(main_cli, "_match_network_endpoint_indices_by_connectivity", lambda pot, start_node, end_node: {"root_index": 2, "target_index": 1})
    monkeypatch.setattr(main_cli, "_find_pot_root_node_index", lambda pot: 9)
    monkeypatch.setattr(main_cli, "_find_pot_target_node_index", lambda pot, target_idx_hint=None: target_idx_hint)
    monkeypatch.setattr(main_cli.nx, "has_path", lambda *args, **kwargs: True)

    captured = {}

    def _fake_best_path(pot, root_idx, target_idx):
        captured["root_idx"] = root_idx
        captured["target_idx"] = target_idx
        return [root_idx, target_idx], 1.0

    monkeypatch.setattr(main_cli, "_best_path_by_apparent_barrier", _fake_best_path)
    monkeypatch.setattr(main_cli, "_path_chain_from_pot", lambda pot, path: expected)

    out = main_cli._load_visualization_data(json_fp)
    assert out.network_endpoint_hints["root_index"] == 2
    assert out.network_endpoint_hints["target_index"] == 1
    assert captured["root_idx"] == 2
    assert captured["target_idx"] == 1


def test_load_chain_for_visualization_falls_back_to_chain_xyz(monkeypatch, tmp_path):
    chain_fp = tmp_path / "plain_chain.xyz"
    chain_fp.write_text("dummy")
    expected = _chain()

    monkeypatch.setattr(
        main_cli.Chain,
        "from_xyz",
        staticmethod(lambda *args, **kwargs: expected),
    )

    out = main_cli._load_visualization_data(chain_fp)
    assert out.chain is expected
    assert out.chain_trajectory is None


def test_visualize_command_writes_html_and_can_skip_open(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "result.xyz"
    src.write_text("x")
    (tmp_path / "result_history").mkdir()
    expected = _chain()

    monkeypatch.setattr(
        main_cli,
        "_load_visualization_data",
        lambda **kwargs: main_cli._VisualizationData(chain=expected, chain_trajectory=[expected]),
    )
    monkeypatch.setattr(
        main_cli,
        "_build_chain_visualizer_html",
        lambda chain, chain_trajectory=None, tree_layers=None, network_payload=None: "<html>ok</html>",
    )

    opened = {"called": False}

    def _fake_open(url):
        opened["called"] = True
        return True

    monkeypatch.setattr(main_cli.webbrowser, "open", _fake_open)

    main_cli.visualize(result_path=str(src), no_open=True)

    assert (tmp_path / "result_visualize.html").read_text() == "<html>ok</html>"
    assert opened["called"] is False


def test_load_chain_for_visualization_raises_for_unknown_dir(tmp_path):
    folder = tmp_path / "unknown"
    folder.mkdir()
    with pytest.raises(ValueError):
        main_cli._load_visualization_data(folder)


def test_build_chain_visualizer_html_includes_chain_dropdown_default_last(monkeypatch):
    c0 = _chain()
    c1 = _chain()

    monkeypatch.setattr(
        main_cli, "generate_structure_viewer_html", lambda *_args, **_kwargs: "<div>struct</div>"
    )
    monkeypatch.setattr(main_cli, "generate_neb_plot", lambda *_args, **_kwargs: "plotb64")
    monkeypatch.setattr(main_cli, "_generate_opt_history_plot_b64", lambda *_args, **_kwargs: "histb64")

    html = main_cli._build_chain_visualizer_html(chain=c1, chain_trajectory=[c0, c1])
    assert 'id="treePanel"' in html
    assert 'id="treeSvg"' in html
    assert 'id="chainSelect"' in html
    assert "default_chain_index" in html
    assert "Optimization History" in html
    assert "<\\/script>" in html


def test_build_chain_visualizer_html_tree_mode_has_layer_options(monkeypatch):
    c0 = _chain()
    c1 = _chain()

    monkeypatch.setattr(
        main_cli, "generate_structure_viewer_html", lambda *_args, **_kwargs: "<div>struct</div>"
    )
    monkeypatch.setattr(main_cli, "generate_neb_plot", lambda *_args, **_kwargs: "plotb64")
    monkeypatch.setattr(main_cli, "_generate_opt_history_plot_b64", lambda *_args, **_kwargs: "histb64")

    tree_layers = [
        {
            "depth": 0,
            "groups": [
                {"label": "Node 0", "node_index": 0, "parent_index": None, "chains": [c0]}
            ],
        },
        {
            "depth": 1,
            "groups": [
                {"label": "Node 3", "node_index": 3, "parent_index": 0, "chains": [c0, c1]}
            ],
        },
    ]
    html = main_cli._build_chain_visualizer_html(chain=c1, tree_layers=tree_layers)
    assert 'id="treeSvg"' in html
    assert "treeNodeMap" in html
    assert "Node 3" in html


def test_build_chain_visualizer_html_network_mode_has_edge_graph(monkeypatch):
    c0 = _chain()

    monkeypatch.setattr(
        main_cli, "generate_structure_viewer_html", lambda *_args, **_kwargs: "<div>struct</div>"
    )
    monkeypatch.setattr(main_cli, "generate_neb_plot", lambda *_args, **_kwargs: "plotb64")
    monkeypatch.setattr(main_cli, "_generate_opt_history_plot_b64", lambda *_args, **_kwargs: "histb64")

    network_payload = {
        "nodes": [
            {"id": 0, "label": "Node 0", "x": 0.0, "y": 0.0, "is_root": True, "is_target": False},
            {"id": 1, "label": "Node 1", "x": 1.0, "y": 1.0, "is_root": False, "is_target": True},
        ],
        "edges": [
            {
                "id": "0->1",
                "source": 0,
                "target": 1,
                "barrier": 4.2,
                "reverse_barrier": 3.8,
                "pair_barrier_sum": 8.0,
                "highlight": True,
                "viz": {"chains": [{"index": 0, "frames": [{"xyz_b64": "WA=="}], "plot": {"x": [0.0], "y": [0.0]}}], "default_chain_index": 0},
            }
        ],
        "root_index": 0,
        "target_index": 1,
        "highlighted_path": [0, 1],
    }
    html = main_cli._build_chain_visualizer_html(chain=c0, network_payload=network_payload)
    assert 'id="networkSvg"' in html
    assert "currentNetworkEdgeId" in html
    assert "best overall path is highlighted in gold" in html
    assert "0->1" in html


def test_best_chain_for_directed_edge_orients_chain_to_source_target():
    graph = main_cli.nx.DiGraph()
    graph.add_node(0, td=_node(0.0, 0.0), root=True)
    graph.add_node(1, td=_node(1.0, 0.0))
    reversed_chain = _chain_with_energies([1.0, 0.5, 0.0], [0.0, 0.01, 0.0])
    graph.add_edge(0, 1, list_of_nebs=[reversed_chain], barrier=1.0)
    pot = SimpleNamespace(graph=graph, target=None)

    directed = main_cli._best_chain_for_directed_edge(pot, 0, 1)
    assert np.isclose(directed.nodes[0].coords[1][0], 0.0)
    assert np.isclose(directed.nodes[-1].coords[1][0], 1.0)


def test_network_visualization_uses_lowest_apparent_barrier_path():
    graph = main_cli.nx.DiGraph()
    graph.add_node(0, td=_node(0.0, 0.0), root=True)
    graph.add_node(1, td=_node(1.0, 0.0))
    graph.add_node(2, td=_node(2.0, 0.0))
    graph.add_node(3, td=_node(3.0, 0.0))
    graph.add_edge(
        0,
        1,
        list_of_nebs=[_chain_with_energies([0.0, 0.4, 1.0], [0.0, 0.020, 0.001])],
        barrier=1.0,
    )
    graph.add_edge(
        1,
        3,
        list_of_nebs=[_chain_with_energies([1.0, 2.2, 3.0], [0.001, 0.018, 0.0])],
        barrier=1.0,
    )
    graph.add_edge(
        0,
        2,
        list_of_nebs=[_chain_with_energies([0.0, 1.4, 2.0], [0.0, 0.008, 0.001])],
        barrier=2.0,
    )
    graph.add_edge(
        2,
        3,
        list_of_nebs=[_chain_with_energies([2.0, 2.6, 3.0], [0.001, 0.007, 0.0])],
        barrier=2.0,
    )
    pot = SimpleNamespace(graph=graph, target=None)

    payload = main_cli._build_network_visualization_payload(pot)
    assert payload["highlighted_path"] == [0, 2, 3]
    assert payload["best_apparent_barrier"] is not None

    path_chain = main_cli._path_chain_from_pot(pot, payload["highlighted_path"])
    assert path_chain is not None
    assert np.isclose(path_chain.nodes[0].coords[1][0], 0.0)
    assert np.isclose(path_chain.nodes[-1].coords[1][0], 3.0)


def test_network_visualization_collapses_reversible_pair_toward_product():
    graph = main_cli.nx.DiGraph()
    graph.add_node(0, td=_node(0.0, 0.0), root=True)
    graph.add_node(1, td=_node(1.0, 0.0))
    graph.add_node(3, td=_node(3.0, 0.0))
    graph.add_edge(
        0,
        3,
        list_of_nebs=[_chain_with_energies([0.0, 2.0, 3.0], [0.0, 0.010, 0.001])],
        barrier=5.0,
    )
    graph.add_edge(
        3,
        0,
        list_of_nebs=[_chain_with_energies([3.0, 2.0, 0.0], [0.0, 0.015, 0.001])],
        barrier=7.0,
    )
    graph.add_edge(
        3,
        1,
        list_of_nebs=[_chain_with_energies([3.0, 2.0, 1.0], [0.0, 0.006, 0.0])],
        barrier=4.0,
    )
    graph.add_edge(
        0,
        1,
        list_of_nebs=[_chain_with_energies([0.0, 0.5, 1.0], [0.0, 0.030, 0.0])],
        barrier=20.0,
    )
    pot = SimpleNamespace(graph=graph, target=None)

    payload = main_cli._build_network_visualization_payload(
        pot, endpoint_hints={"root_index": 0, "target_index": 1}
    )
    ids = {edge["id"] for edge in payload["edges"]}
    assert "0->3" in ids
    assert "3->0" not in ids


def test_parse_visualize_atom_indices_from_file(tmp_path):
    fp = tmp_path / "qmindices.dat"
    fp.write_text("3\n1\n3\n")
    assert main_cli._parse_visualize_atom_indices(qminds_fp=str(fp)) == [1, 3]


def test_subset_chain_for_visualization_keeps_energies():
    chain = _chain()
    chain.nodes[0]._cached_energy = 10.0
    chain.nodes[1]._cached_energy = 12.5
    chain.nodes[2]._cached_energy = 11.0
    out = main_cli._subset_chain_for_visualization(chain, [1])

    assert len(out.nodes[0].structure.symbols) == 1
    assert out.nodes[0].energy == 10.0
    assert out.nodes[1].energy == 12.5
    assert out.nodes[2].energy == 11.0
