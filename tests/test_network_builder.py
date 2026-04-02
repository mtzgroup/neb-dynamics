from pathlib import Path

import numpy as np
from qcio import Structure

from neb_dynamics.NetworkBuilder import NetworkBuilder
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs, NetworkInputs
from neb_dynamics.pot import Pot
from neb_dynamics.nodes.node import StructureNode


def _structure(distance: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def _topology_change_structure(kind: str, offset: float = 0.0) -> Structure:
    if kind == "chain":
        geometry = np.array(
            [[0.0, 0.0, 0.0], [1.20 + offset, 0.0, 0.0], [2.16 + (2 * offset), 0.0, 0.0]]
        )
    elif kind == "pair_plus_single":
        geometry = np.array(
            [[0.0, 0.0, 0.0], [1.20 + offset, 0.0, 0.0], [5.20 + offset, 0.0, 0.0]]
        )
    else:
        raise ValueError(f"Unknown topology kind: {kind}")
    return Structure(
        geometry=geometry,
        symbols=["C", "O", "H"],
        charge=0,
        multiplicity=1,
    )


def _node(distance: float, energy: float) -> StructureNode:
    node = StructureNode(structure=_structure(distance))
    node._cached_energy = energy
    node._cached_gradient = np.zeros_like(node.coords)
    return node


def _chain(start_distance: float, peak_distance: float, end_distance: float, peak_energy: float) -> Chain:
    return Chain.model_validate(
        {
            "nodes": [
                _node(start_distance, 0.0),
                _node(peak_distance, peak_energy),
                _node(end_distance, 0.01),
            ],
            "parameters": ChainInputs(),
        }
    )


def _topology_node(kind: str, energy: float, offset: float = 0.0) -> StructureNode:
    node = StructureNode(structure=_topology_change_structure(kind, offset))
    node._cached_energy = energy
    node._cached_gradient = np.zeros_like(node.coords)
    return node


def _topology_chain(chain_offset: float, product_offset: float, peak_energy: float) -> Chain:
    return Chain.model_validate(
        {
            "nodes": [
                _topology_node("chain", 0.0, chain_offset),
                _topology_node("chain", peak_energy, chain_offset + 0.15),
                _topology_node("pair_plus_single", 0.01, product_offset),
            ],
            "parameters": ChainInputs(),
        }
    )


def test_load_network_data_collapses_nodes_by_graph_identity(monkeypatch):
    builder = NetworkBuilder(
        data_dir=Path("."),
        chain_inputs=ChainInputs(),
        network_inputs=NetworkInputs(tolerate_kinks=True, verbose=False),
    )
    leaves_by_path = {
        "run_a": [_topology_chain(0.0, 0.0, 0.030)],
        "run_b": [_topology_chain(0.08, 0.12, 0.045)],
    }

    monkeypatch.setattr(
        builder,
        "_get_relevant_leaves",
        lambda fp: leaves_by_path[Path(fp).name],
    )

    structures, edges = builder._load_network_data([Path("run_a"), Path("run_b")])

    assert len(structures) == 2
    assert sorted(edges) == ["0-1", "1-0"]
    assert len(edges["0-1"]) == 2
    assert len(edges["1-0"]) == 2


def test_add_all_edges_keeps_only_lowest_barrier_chain_per_direction():
    builder = NetworkBuilder(
        data_dir=Path("."),
        chain_inputs=ChainInputs(),
        network_inputs=NetworkInputs(verbose=False),
    )
    reactant = _node(0.74, 0.0)
    product = _node(3.0, 0.01)
    low_chain = _chain(0.74, 1.5, 3.0, 0.020)
    high_chain = _chain(0.74, 1.7, 3.0, 0.060)
    low_chain_rev = low_chain.copy()
    low_chain_rev.nodes.reverse()
    high_chain_rev = high_chain.copy()
    high_chain_rev.nodes.reverse()

    builder.leaf_objects = {
        "0-1": [high_chain, low_chain],
        "1-0": [high_chain_rev, low_chain_rev],
    }
    edges = {
        "0-1": [high_chain.get_eA_chain(), low_chain.get_eA_chain()],
        "1-0": [high_chain_rev.get_eA_chain(), low_chain_rev.get_eA_chain()],
    }

    pot = Pot.model_validate({"root": reactant.graph})
    pot = builder._add_all_nodes(pot, structures=[reactant, product])
    pot = builder._add_all_edges(pot, structures=[reactant, product], edges=edges)

    assert pot.graph.edges[(0, 1)]["barrier"] == low_chain.get_eA_chain()
    assert pot.graph.edges[(0, 1)]["list_of_nebs"] == [low_chain]
    assert pot.graph.edges[(1, 0)]["barrier"] == low_chain_rev.get_eA_chain()
    assert pot.graph.edges[(1, 0)]["list_of_nebs"] == [low_chain_rev]
