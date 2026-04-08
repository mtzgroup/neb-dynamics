import math

import networkx as nx
import pytest

from neb_dynamics.kmc import (
    build_kmc_payload,
    default_initial_conditions,
    edge_rate_constant,
    normalize_initial_conditions,
    simulate_kmc,
)
from neb_dynamics.molecule import Molecule
from neb_dynamics.pot import Pot
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import StructureNode
from qcio import Structure
import numpy as np


def _pot_with_barrier() -> Pot:
    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, molecule=Molecule.from_smiles("C"))
    pot.graph.add_node(1, molecule=Molecule.from_smiles("CO"))
    pot.graph.add_edge(0, 1, barrier=1.0, reaction="0->1")
    return pot


def _node(x: float, energy: float) -> StructureNode:
    node = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
            symbols=["H", "H"],
            charge=0,
            multiplicity=1,
        )
    )
    node._cached_energy = energy
    node._cached_gradient = np.zeros((2, 3))
    node.has_molecular_graph = False
    node.graph = None
    return node


def test_default_initial_conditions_seed_node_zero():
    pot = _pot_with_barrier()
    assert default_initial_conditions(pot) == {0: 1.0, 1: 0.0}


def test_normalize_initial_conditions_overrides_known_nodes_only():
    pot = _pot_with_barrier()
    assert normalize_initial_conditions(pot, {1: 0.3, 99: 2.0}) == {0: 0.0, 1: 0.3}


def test_normalize_initial_conditions_uses_defaults_only_when_unspecified():
    pot = _pot_with_barrier()
    assert normalize_initial_conditions(pot, None) == {0: 1.0, 1: 0.0}


def test_build_kmc_payload_contains_rate_constants():
    pot = _pot_with_barrier()
    payload = build_kmc_payload(pot, temperature_kelvin=300.0)

    assert payload["temperature_kelvin"] == 300.0
    assert len(payload["nodes"]) == 2
    assert len(payload["edges"]) == 1
    assert payload["edges"][0]["rate_constant"] == edge_rate_constant(1.0, 300.0)


def test_simulate_kmc_moves_population_downhill():
    pot = _pot_with_barrier()
    result = simulate_kmc(
        pot,
        temperature_kelvin=300.0,
        initial_conditions={0: 1.0, 1: 0.0},
        max_steps=5,
        seed=1,
    )

    assert len(result["history"]) >= 2
    assert result["final_populations"][1] > result["final_populations"][0]
    assert math.isclose(
        result["final_populations"][0] + result["final_populations"][1],
        1.0,
        rel_tol=1e-6,
    )


def test_simulate_kmc_respects_explicit_nondefault_seed():
    pot = _pot_with_barrier()
    result = simulate_kmc(
        pot,
        temperature_kelvin=300.0,
        initial_conditions={1: 1.0},
        max_steps=1,
    )

    assert result["history"][0]["populations"][0] == 0.0
    assert result["history"][0]["populations"][1] == 1.0


def test_simulate_kmc_handles_stiff_rates_without_trapping_population():
    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, molecule=Molecule.from_smiles("C"))
    pot.graph.add_node(1, molecule=Molecule.from_smiles("CO"))
    pot.graph.add_edge(0, 1, barrier=0.0, reaction="fast")

    result = simulate_kmc(
        pot,
        temperature_kelvin=800.0,
        initial_conditions={0: 1.0},
        max_steps=200,
        final_time=0.1,
    )

    assert result["final_populations"][1] > 0.99
    assert result["final_populations"][0] < 0.01


def test_simulate_kmc_reuses_prebuilt_payload(monkeypatch):
    pot = _pot_with_barrier()
    payload = build_kmc_payload(pot, temperature_kelvin=300.0)

    monkeypatch.setattr(
        "neb_dynamics.kmc.build_kmc_payload",
        lambda *args, **kwargs: pytest.fail("build_kmc_payload should not be called when payload is provided"),
    )
    result = simulate_kmc(
        pot,
        temperature_kelvin=300.0,
        initial_conditions={0: 1.0, 1: 0.0},
        max_steps=2,
        payload=payload,
    )

    assert result["max_steps"] == 2


def test_build_kmc_payload_suppresses_suspicious_zero_barrier_edge():
    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, molecule=Molecule.from_smiles("C"))
    pot.graph.add_node(1, molecule=Molecule.from_smiles("CO"))

    chain_forward = Chain.model_validate(
        {
            "nodes": [_node(0.0, 0.0), _node(1.0, 0.5), _node(2.0, 1.0)],
            "parameters": ChainInputs(),
        }
    )
    chain_reverse = Chain.model_validate(
        {
            "nodes": [_node(2.0, 1.0), _node(1.0, 0.5), _node(0.0, 0.0)],
            "parameters": ChainInputs(),
        }
    )
    pot.graph.add_edge(0, 1, barrier=627.5, reaction="0->1", list_of_nebs=[chain_forward])
    pot.graph.add_edge(1, 0, barrier=0.0, reaction="1->0", list_of_nebs=[chain_reverse])

    payload = build_kmc_payload(pot, temperature_kelvin=300.0)

    assert [(edge["source"], edge["target"]) for edge in payload["edges"]] == [(0, 1)]
    assert payload["suppressed_edges"] == [
        {
            "source": 1,
            "target": 0,
            "barrier": 0.0,
            "reaction": "1->0",
            "reason": "start_endpoint_is_chain_maximum",
        }
    ]
