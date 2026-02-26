import numpy as np
from qcio import Structure

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
