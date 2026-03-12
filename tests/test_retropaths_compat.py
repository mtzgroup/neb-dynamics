import networkx as nx
import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.pot import Pot
from neb_dynamics.retropaths_compat import (
    annotate_pot_with_neb_results,
    copy_graph_like_molecule,
    retropaths_pot_to_neb_pot,
    structure_node_from_graph_like_molecule,
)


class _ExternalMolecule(nx.Graph):
    def __init__(self):
        super().__init__()
        self.chemical_name = "indexed test"
        self._smiles = "[H]C([H])([H])[O-]"


def _build_external_molecule() -> _ExternalMolecule:
    molecule = _ExternalMolecule()
    molecule.add_node(0, element="C", charge=0, neighbors=4)
    molecule.add_node(1, element="O", charge=-1, neighbors=1)
    molecule.add_node(2, element="H", charge=0, neighbors=1)
    molecule.add_node(3, element="H", charge=0, neighbors=1)
    molecule.add_node(4, element="H", charge=0, neighbors=1)
    molecule.add_edge(0, 1, bond_order="single")
    molecule.add_edge(0, 2, bond_order="single")
    molecule.add_edge(0, 3, bond_order="single")
    molecule.add_edge(0, 4, bond_order="single")
    return molecule


def _build_noncontiguous_external_molecule() -> _ExternalMolecule:
    molecule = _ExternalMolecule()
    molecule.add_node(0, element="C", charge=0, neighbors=4)
    molecule.add_node(2, element="O", charge=-1, neighbors=1)
    molecule.add_node(5, element="H", charge=0, neighbors=1)
    molecule.add_node(9, element="H", charge=0, neighbors=1)
    molecule.add_node(11, element="H", charge=0, neighbors=1)
    molecule.add_edge(0, 2, bond_order="single")
    molecule.add_edge(0, 5, bond_order="single")
    molecule.add_edge(0, 9, bond_order="single")
    molecule.add_edge(0, 11, bond_order="single")
    return molecule


def test_copy_graph_like_molecule_preserves_atom_indices_and_attrs():
    external = _build_external_molecule()

    converted = copy_graph_like_molecule(external)

    assert list(converted.nodes) == [0, 1, 2, 3, 4]
    assert converted.nodes[1]["element"] == "O"
    assert converted.nodes[1]["charge"] == -1
    assert converted.edges[(0, 1)]["bond_order"] == "single"
    assert converted.chemical_name == external.chemical_name
    assert converted._smiles == external._smiles


def test_structure_node_from_graph_like_molecule_preserves_graph_indices():
    external = _build_external_molecule()

    node = structure_node_from_graph_like_molecule(external, charge=-1, spinmult=1)

    assert isinstance(node, StructureNode)
    assert list(node.graph.nodes) == [0, 1, 2, 3, 4]
    assert node.graph.nodes[1]["element"] == "O"
    assert node.graph.nodes[1]["charge"] == -1
    assert node.structure.charge == -1
    assert len(node.structure.symbols) == 5
    assert np.array(node.structure.geometry).shape == (5, 3)


def test_structure_node_roundtrip_keeps_supplied_graph():
    external = _build_external_molecule()
    node = structure_node_from_graph_like_molecule(external, charge=-1, spinmult=1)

    roundtrip = StructureNode.from_serializable(node.to_serializable())

    assert list(roundtrip.graph.nodes) == [0, 1, 2, 3, 4]
    assert roundtrip.graph.nodes[1]["element"] == "O"
    assert roundtrip.graph.nodes[1]["charge"] == -1


def test_structure_node_from_graph_like_molecule_handles_noncontiguous_indices():
    external = _build_noncontiguous_external_molecule()

    node = structure_node_from_graph_like_molecule(external, charge=-1, spinmult=1)

    assert list(node.graph.nodes) == [0, 2, 5, 9, 11]
    assert len(node.structure.symbols) == 5


class _ExternalPot:
    def __init__(self):
        self.root = _build_external_molecule()
        self.target = _build_external_molecule()
        self.environment = _build_external_molecule()
        self.multiplier = 1
        self.rxn_name = "compat"
        self.run_time = 12.0
        self.graph = nx.DiGraph()
        self.graph.add_node(
            0, molecule=self.root, environment=self.environment, converged=False, root=True
        )
        self.graph.add_node(4, molecule=self.target, converged=True)
        self.graph.add_edge(4, 0, reaction="demo")


def test_retropaths_pot_to_neb_pot_preserves_topology_and_td_graphs():
    source = _ExternalPot()

    converted = retropaths_pot_to_neb_pot(source, charge=-1, spinmult=1)

    assert isinstance(converted, Pot)
    assert sorted(converted.graph.nodes) == [0, 4]
    assert converted.graph.edges[(4, 0)]["reaction"] == "demo"
    assert converted.graph.edges[(4, 0)]["list_of_nebs"] == []
    assert list(converted.graph.nodes[0]["molecule"].nodes) == [0, 1, 2, 3, 4]
    assert list(converted.graph.nodes[0]["environment"].nodes) == [0, 1, 2, 3, 4]
    assert list(converted.graph.nodes[4]["td"].graph.nodes) == [0, 1, 2, 3, 4]
    assert converted.run_time == 12.0
    roundtrip = Pot.from_dict(converted.model_dump())
    assert list(roundtrip.graph.nodes[0]["environment"].nodes) == [0, 1, 2, 3, 4]


def _energetic_node(energy: float) -> StructureNode:
    node = structure_node_from_graph_like_molecule(
        _build_external_molecule(), charge=-1, spinmult=1
    )
    node._cached_energy = energy
    node._cached_gradient = np.zeros((5, 3))
    return node


def test_annotate_pot_with_neb_results_populates_networkbuilder_style_fields():
    source = _ExternalPot()
    converted = retropaths_pot_to_neb_pot(source, charge=-1, spinmult=1)

    chain = Chain.model_validate(
        {
            "nodes": [_energetic_node(0.0), _energetic_node(0.02), _energetic_node(0.01)],
            "parameters": ChainInputs(),
        }
    )

    annotate_pot_with_neb_results(converted, {(4, 0): [chain]})

    edge_data = converted.graph.edges[(4, 0)]
    assert edge_data["list_of_nebs"] == [chain]
    assert edge_data["barrier"] == chain.get_eA_chain()
    assert edge_data["exp_neg_barrier"] == np.exp(-chain.get_eA_chain())

    reverse_edge_data = converted.graph.edges[(0, 4)]
    reverse_chain = reverse_edge_data["list_of_nebs"][0]
    assert reverse_chain[0].energy == chain[-1].energy
    assert reverse_chain[-1].energy == chain[0].energy
    assert reverse_edge_data["barrier"] == reverse_chain.get_eA_chain()
    assert reverse_edge_data["barrier"] != edge_data["barrier"]
    assert edge_data["reaction"] == "demo"
    assert reverse_edge_data["reaction"] == "demo"

    node0 = converted.graph.nodes[0]
    node4 = converted.graph.nodes[4]
    assert node4["td"].energy == 0.0
    assert node4["node_energy"] == 0.0
    assert node4["node_energies"] == [0.0, 0.0]
    assert node0["td"].energy == 0.01
    assert node0["node_energies"] == [0.01, 0.01]
