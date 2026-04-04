import json
from pathlib import Path

import networkx as nx

from neb_dynamics.molecule import Molecule
from neb_dynamics.pot import Pot


def test_pot_from_local_fixture_roundtrip():
    fixture_path = Path(__file__).parent / "data.json"
    with fixture_path.open() as f:
        loaded = json.load(f)

    pot = Pot.from_dict(loaded)
    dumped = pot.model_dump()
    pot_roundtrip = Pot.from_dict(dumped)

    for node_id in pot_roundtrip.graph.nodes:
        assert isinstance(pot_roundtrip.graph.nodes[node_id]["molecule"], Molecule)


def test_pot_model_dump_handles_forward_only_edge_without_reverse_lookup():
    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, molecule=Molecule.from_smiles("C"))
    pot.graph.add_node(1, molecule=Molecule.from_smiles("CO"))
    pot.graph.add_edge(0, 1, list_of_nebs=[{"placeholder": True}])

    dumped = pot.model_dump()

    assert dumped["graph"]["links"][0]["source"] == 0
    assert dumped["graph"]["links"][0]["target"] == 1
    assert dumped["graph"]["links"][0]["list_of_nebs"] == [{"placeholder": True}]


def test_pot_from_dict_accepts_networkx_edges_key():
    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, molecule=Molecule.from_smiles("C"))
    pot.graph.add_node(1, molecule=Molecule.from_smiles("CO"))
    pot.graph.add_edge(0, 1, reaction="demo", list_of_nebs=[])

    dumped = pot.model_dump()
    dumped["graph"]["edges"] = dumped["graph"].pop("links")

    loaded = Pot.from_dict(dumped)
    assert loaded.graph.has_edge(0, 1)
    assert loaded.graph.edges[(0, 1)]["reaction"] == "demo"


def test_pot_from_dict_handles_edges_without_list_of_nebs():
    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    pot.graph = nx.DiGraph()
    pot.graph.add_node(0, molecule=Molecule.from_smiles("C"))
    pot.graph.add_node(1, molecule=Molecule.from_smiles("CO"))
    pot.graph.add_edge(0, 1, reaction="demo")

    dumped = pot.model_dump()
    dumped["graph"]["links"][0].pop("list_of_nebs", None)

    loaded = Pot.from_dict(dumped)
    assert loaded.graph.has_edge(0, 1)


def test_pot_from_dict_preserves_existing_root_node_attributes():
    pot = Pot(root=Molecule.from_smiles("C"), target=Molecule())
    dumped = pot.model_dump()
    dumped["graph"]["nodes"][0]["converged"] = True
    dumped["graph"]["nodes"][0]["root"] = False

    loaded = Pot.from_dict(dumped)

    assert loaded.graph.nodes[0]["converged"] is True
    assert loaded.graph.nodes[0]["root"] is False
