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
        lambda chain, chain_trajectory=None, tree_layers=None: "<html>ok</html>",
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
