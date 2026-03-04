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

    out = main_cli._load_chain_for_visualization(folder)
    assert out is expected


def test_load_chain_for_visualization_detects_neb_file(monkeypatch, tmp_path):
    neb_fp = tmp_path / "mep_output.xyz"
    neb_fp.write_text("dummy")
    hist = tmp_path / "mep_output_history"
    hist.mkdir()
    expected = _chain()

    monkeypatch.setattr(
        main_cli.NEB,
        "read_from_disk",
        staticmethod(lambda **kwargs: SimpleNamespace(chain_trajectory=[expected])),
    )

    out = main_cli._load_chain_for_visualization(neb_fp)
    assert out is expected


def test_visualize_command_writes_html_and_can_skip_open(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "result.xyz"
    src.write_text("x")
    (tmp_path / "result_history").mkdir()
    expected = _chain()

    monkeypatch.setattr(main_cli, "_load_chain_for_visualization", lambda **kwargs: expected)
    monkeypatch.setattr(main_cli, "_build_chain_visualizer_html", lambda chain: "<html>ok</html>")

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
        main_cli._load_chain_for_visualization(folder)
