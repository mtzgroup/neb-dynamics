from types import SimpleNamespace

import numpy as np
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.msmep import MSMEP
from neb_dynamics.nodes.node import StructureNode


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def test_qmmm_run_minimize_disables_molecular_graphs(monkeypatch):
    class QMMMEngine:
        def compute_gradients(self, chain):
            for node in chain:
                node._cached_gradient = np.zeros_like(node.coords)
            return np.array([node._cached_gradient for node in chain])

    inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(v=False),
        chain_inputs=ChainInputs(),
        gi_inputs=SimpleNamespace(nimages=2),
        optimizer_kwds={"name": "cg", "timestep": 0.1},
        engine=QMMMEngine(),
    )

    m = MSMEP(inputs=inputs)
    nodes = [StructureNode(structure=_structure_at_x(0.7)), StructureNode(structure=_structure_at_x(1.3))]
    chain = Chain.model_validate({"nodes": nodes, "parameters": inputs.chain_inputs})

    # Ensure the test starts with graphing enabled.
    assert all(node.has_molecular_graph for node in chain)

    class _FakeMinimizer:
        def __init__(self, initial_chain):
            self.initial_chain = initial_chain
            self.optimized = initial_chain
            self.chain_trajectory = [initial_chain]

        def optimize_chain(self):
            return SimpleNamespace(is_elem_step=True)

    monkeypatch.setattr(MSMEP, "_create_interpolation", lambda self, c: c)
    monkeypatch.setattr(MSMEP, "_construct_path_minimizer", lambda self, initial_chain: _FakeMinimizer(initial_chain))

    m.run_minimize_chain(chain)

    assert all(node.has_molecular_graph is False for node in chain)
    assert all(node.graph is None for node in chain)
