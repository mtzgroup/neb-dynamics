import numpy as np
from types import SimpleNamespace
from qcio import Structure

import neb_dynamics.neb as neb_module
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.neb import _endpoint_energy_inversion_warning_text, NEB


def test_endpoint_energy_inversion_warning_triggers_for_qmmm():
    energies = np.array([0.020, 0.000, 0.005, 0.018])
    msg = _endpoint_energy_inversion_warning_text(
        energies=energies,
        is_qmmm_engine=True,
        frozen_atom_indices=[],
    )
    assert msg is not None
    assert "QM/MM" in msg
    assert "frozen atoms" in msg


def test_endpoint_energy_inversion_warning_not_triggered_when_ts_interior():
    energies = np.array([0.000, 0.020, 0.005, 0.001])
    msg = _endpoint_energy_inversion_warning_text(
        energies=energies,
        is_qmmm_engine=True,
        frozen_atom_indices=[0, 1],
    )
    assert msg is None


def test_neb_warning_path_handles_parameters_without_frozen_indices(monkeypatch):
    def _node(x: float, e: float) -> StructureNode:
        node = StructureNode(
            structure=Structure(
                geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
                symbols=["H", "H"],
                charge=0,
                multiplicity=1,
            )
        )
        node._cached_energy = e
        node._cached_gradient = np.zeros((2, 3))
        node.has_molecular_graph = False
        node.graph = None
        return node

    # Endpoint is highest energy to force warning branch evaluation.
    prepared_chain = Chain.model_validate(
        {
            "nodes": [_node(0.8, 0.020), _node(1.0, 0.000), _node(1.2, 0.010)],
            "parameters": ChainInputs(),
        }
    )

    class QMMMEngine:
        pass

    params = SimpleNamespace(
        max_steps=2,
        v=False,
        do_elem_step_checks=False,
        negative_steps_thre=10,
        positive_steps_thre=10,
    )
    optimizer = SimpleNamespace(timestep=0.1, g_old=None, reset=lambda: None)
    neb = NEB(
        initial_chain=prepared_chain.copy(),
        optimizer=optimizer,
        parameters=params,
        engine=QMMMEngine(),
    )

    monkeypatch.setattr(NEB, "update_chain", lambda self, chain: prepared_chain.copy())
    monkeypatch.setattr(neb_module, "chain_converged", lambda **kwargs: True)
    monkeypatch.setattr(neb_module.ch, "_gradient_correlation", lambda a, b: 1.0)
    monkeypatch.setattr(neb_module, "format_neb_caption", lambda **kwargs: "")
    monkeypatch.setattr(neb_module, "print_chain_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(neb_module, "update_status", lambda *args, **kwargs: None)

    # Regression: this used to crash with
    # AttributeError: 'types.SimpleNamespace' object has no attribute 'frozen_atom_indices'
    result = neb.optimize_chain()
    assert result.is_elem_step is True
