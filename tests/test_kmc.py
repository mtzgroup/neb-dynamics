import math

import networkx as nx

from neb_dynamics.kmc import (
    build_kmc_payload,
    default_initial_conditions,
    edge_rate_constant,
    normalize_initial_conditions,
    simulate_kmc,
)
from neb_dynamics.molecule import Molecule
from neb_dynamics.pot import Pot


def _pot_with_barrier() -> Pot:
    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, molecule=Molecule.from_smiles("C"))
    pot.graph.add_node(1, molecule=Molecule.from_smiles("CO"))
    pot.graph.add_edge(0, 1, barrier=1.0, reaction="0->1")
    return pot


def test_default_initial_conditions_seed_node_zero():
    pot = _pot_with_barrier()
    assert default_initial_conditions(pot) == {0: 1.0, 1: 0.0}


def test_normalize_initial_conditions_overrides_known_nodes_only():
    pot = _pot_with_barrier()
    assert normalize_initial_conditions(pot, {1: 0.3, 99: 2.0}) == {0: 1.0, 1: 0.3}


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
