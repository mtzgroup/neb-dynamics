import numpy as np
from qcio import Structure
from types import SimpleNamespace

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.scripts import main_cli


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def test_write_chain_with_nan_fallback_writes_nan_sidecars(tmp_path):
    nodes = [StructureNode(structure=_structure_at_x(0.8)), StructureNode(structure=_structure_at_x(1.2))]
    chain = Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})

    out_fp = tmp_path / "chain.xyz"
    main_cli._write_chain_with_nan_fallback(chain, out_fp)

    ene = np.loadtxt(tmp_path / "chain.energies")
    grads = np.loadtxt(tmp_path / "chain.gradients")
    shape = np.loadtxt(tmp_path / "chain_grad_shapes.txt")

    assert np.all(np.isnan(ene))
    assert np.all(np.isnan(grads))
    assert np.array_equal(shape.astype(int), np.array([2, 2, 3]))


def test_write_neb_results_with_history_fallback_writes_history(tmp_path):
    nodes0 = [StructureNode(structure=_structure_at_x(0.8)), StructureNode(structure=_structure_at_x(1.0))]
    nodes1 = [StructureNode(structure=_structure_at_x(0.9)), StructureNode(structure=_structure_at_x(1.1))]
    chain0 = Chain.model_validate({"nodes": nodes0, "parameters": ChainInputs()})
    chain1 = Chain.model_validate({"nodes": nodes1, "parameters": ChainInputs()})
    fake_neb = SimpleNamespace(chain_trajectory=[chain0, chain1], optimized=chain1)

    out_fp = tmp_path / "neb.xyz"
    wrote = main_cli._write_neb_results_with_history(fake_neb, out_fp)

    assert wrote is True
    assert out_fp.exists()
    assert (tmp_path / "neb_history" / "traj_0.xyz").exists()
    assert (tmp_path / "neb_history" / "traj_1.xyz").exists()


def test_write_neb_results_with_history_fallback_writes_qcio_when_enabled(tmp_path):
    class _SavedResult:
        def save(self, path):
            path.write_text("stub")

    nodes0 = [StructureNode(structure=_structure_at_x(0.8)), StructureNode(structure=_structure_at_x(1.0))]
    nodes1 = [StructureNode(structure=_structure_at_x(0.9)), StructureNode(structure=_structure_at_x(1.1))]
    for node in nodes1:
        node._cached_result = _SavedResult()
    chain0 = Chain.model_validate({"nodes": nodes0, "parameters": ChainInputs()})
    chain1 = Chain.model_validate({"nodes": nodes1, "parameters": ChainInputs()})
    fake_neb = SimpleNamespace(chain_trajectory=[chain0, chain1], optimized=chain1)

    out_fp = tmp_path / "neb.xyz"
    wrote = main_cli._write_neb_results_with_history(fake_neb, out_fp, write_qcio=True)

    assert wrote is True
    assert (tmp_path / "neb_node_0.qcio").exists()
    assert (tmp_path / "neb_node_1.qcio").exists()
    assert (tmp_path / "neb_history" / "traj_1_node_0.qcio").exists()
    assert (tmp_path / "neb_history" / "traj_1_node_1.qcio").exists()
