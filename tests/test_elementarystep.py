from types import SimpleNamespace

import numpy as np
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.elementarystep import check_if_elem_step
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import StructureNode


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def test_check_if_elem_step_skips_pseudo_irc_for_chemcloud(monkeypatch):
    chain = Chain.model_validate(
        {
            "nodes": [
                StructureNode(structure=_structure_at_x(0.0)),
                StructureNode(structure=_structure_at_x(1.0)),
                StructureNode(structure=_structure_at_x(2.0)),
            ],
            "parameters": ChainInputs(),
        }
    )
    for index, node in enumerate(chain):
        node._cached_energy = float(index)

    monkeypatch.setattr(
        "neb_dynamics.elementarystep._chain_is_concave",
        lambda chain, engine, verbose=True: SimpleNamespace(
            is_not_concave=False,
            is_concave=True,
            minimization_results=[],
            number_grad_calls=0,
        ),
    )
    monkeypatch.setattr(
        "neb_dynamics.elementarystep.pseudo_irc",
        lambda chain, engine: (_ for _ in ()).throw(AssertionError("pseudo_irc should be skipped for chemcloud")),
    )

    engine = SimpleNamespace(compute_program="chemcloud")
    result = check_if_elem_step(chain, engine=engine, verbose=False)

    assert result.is_elem_step is True
