from __future__ import annotations

import networkx as nx
from typing import Any

import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics.molecule import Molecule
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.pot import Pot
from neb_dynamics.qcio_structure_helpers import molecule_to_structure


def _reversed_chain(chain: Chain) -> Chain:
    reversed_chain = chain.copy()
    reversed_chain.nodes = list(reversed(reversed_chain.nodes))
    if getattr(reversed_chain, "velocity", None):
        reversed_chain.velocity = list(reversed(reversed_chain.velocity))
    return reversed_chain


def copy_graph_like_molecule(source_molecule: Any) -> Molecule:
    """
    Copy a retropaths-like molecule graph into a neb-dynamics Molecule without
    renumbering atom indices.
    """
    molecule = Molecule(
        name=getattr(source_molecule, "chemical_name", ""),
        smi=getattr(source_molecule, "_smiles", ""),
    )

    for node_index, attrs in source_molecule.nodes(data=True):
        molecule.add_node(node_index, **dict(attrs))

    for atom1, atom2, attrs in source_molecule.edges(data=True):
        molecule.add_edge(atom1, atom2, **dict(attrs))

    if hasattr(molecule, "set_neighbors"):
        molecule.set_neighbors()

    return molecule


def _dense_structure_copy(source_molecule: Molecule) -> Molecule:
    """
    Create a temporary dense-index copy for 3D embedding while leaving the
    source graph labels untouched on the stored graph.
    """
    dense = Molecule(
        name=getattr(source_molecule, "chemical_name", ""),
        smi=getattr(source_molecule, "_smiles", ""),
    )
    old_to_new = {
        old_index: new_index for new_index, old_index in enumerate(sorted(source_molecule.nodes))
    }

    for old_index in sorted(source_molecule.nodes):
        dense.add_node(old_to_new[old_index], **dict(source_molecule.nodes[old_index]))

    for atom1, atom2, attrs in source_molecule.edges(data=True):
        dense.add_edge(old_to_new[atom1], old_to_new[atom2], **dict(attrs))

    dense.set_neighbors()
    return dense


def structure_node_from_graph_like_molecule(
    source_molecule: Any,
    charge: int = 0,
    spinmult: int = 1,
) -> StructureNode:
    """
    Build a StructureNode from a retropaths-like molecule while preserving the
    original atom-indexed graph on the node.
    """
    molecule = copy_graph_like_molecule(source_molecule)
    structure = molecule_to_structure(
        _dense_structure_copy(molecule),
        charge=charge,
        spinmult=spinmult,
    )
    return StructureNode(structure=structure, graph=molecule)


def retropaths_pot_to_neb_pot(
    source_pot: Any,
    charge: int = 0,
    spinmult: int = 1,
    node_to_structure_node: Any | None = None,
) -> Pot:
    """
    Convert a retropaths-like Pot into a neb-dynamics Pot while preserving the
    source graph topology and atom indices on each molecular graph.
    """
    converter = node_to_structure_node or (
        lambda source_molecule, _node_index, _node_attrs: (
            structure_node_from_graph_like_molecule(
                source_molecule,
                charge=charge,
                spinmult=spinmult,
            )
        )
    )

    root = copy_graph_like_molecule(source_pot.root)
    target = copy_graph_like_molecule(source_pot.target)
    converted = Pot(
        root=root,
        target=target,
        multiplier=getattr(source_pot, "multiplier", 1),
        rxn_name=getattr(source_pot, "rxn_name", None),
    )
    converted.graph = nx.DiGraph()
    converted.run_time = getattr(source_pot, "run_time", None)

    for node_index, attrs in source_pot.graph.nodes(data=True):
        node_attrs = dict(attrs)
        source_molecule = node_attrs.get("molecule")
        if source_molecule is not None:
            node_attrs["molecule"] = copy_graph_like_molecule(source_molecule)
            td = converter(source_molecule, node_index, attrs)
            if td is not None:
                node_attrs.setdefault("td", td)
        if node_attrs.get("environment") is not None:
            node_attrs["environment"] = copy_graph_like_molecule(
                node_attrs["environment"]
            )
        converted.graph.add_node(node_index, **node_attrs)

    for node1, node2, attrs in source_pot.graph.edges(data=True):
        edge_attrs = dict(attrs)
        edge_attrs.setdefault("list_of_nebs", [])
        converted.graph.add_edge(node1, node2, **edge_attrs)

    return converted


def annotate_pot_with_neb_results(
    pot: Pot,
    chains_by_edge: dict[tuple[int, int], list[Chain]],
    maximum_barrier_height: float = 1000.0,
) -> Pot:
    """
    Populate a converted Pot with NEB-derived edge and node metadata using the
    same data shape used by NetworkBuilder.
    """
    node_conformers: dict[int, list[StructureNode]] = {
        node_index: [] for node_index in pot.graph.nodes
    }

    all_chains_by_edge: dict[tuple[int, int], list[Chain]] = {}
    edge_reaction_labels: dict[tuple[int, int], str] = {}
    for (node1, node2), chains in chains_by_edge.items():
        if len(chains) == 0:
            continue
        all_chains_by_edge.setdefault((node1, node2), []).extend(chains)
        if pot.graph.has_edge(node1, node2):
            reaction = pot.graph.edges[(node1, node2)].get("reaction")
            if reaction:
                edge_reaction_labels[(node1, node2)] = str(reaction)
        reverse_key = (node2, node1)
        reverse_chains = [_reversed_chain(chain) for chain in chains]
        all_chains_by_edge.setdefault(reverse_key, []).extend(reverse_chains)
        if (node1, node2) in edge_reaction_labels:
            edge_reaction_labels[reverse_key] = edge_reaction_labels[(node1, node2)]

    for (node1, node2), chains in all_chains_by_edge.items():
        if len(chains) == 0:
            continue

        barriers = [chain.get_eA_chain() for chain in chains]
        barrier = min(barriers)
        if barrier > maximum_barrier_height:
            continue

        if not pot.graph.has_edge(node1, node2):
            pot.graph.add_edge(node1, node2)
        edge_attrs = pot.graph.edges[(node1, node2)]
        edge_attrs["list_of_nebs"] = chains
        edge_attrs["barrier"] = barrier
        edge_attrs["exp_neg_barrier"] = np.exp(-barrier)
        edge_attrs["reaction"] = edge_reaction_labels.get(
            (node1, node2),
            edge_attrs.get("reaction") or f"eA ({node1}-{node2}): {barrier}",
        )

        for chain in chains:
            node_conformers[node1].append(chain[0])
            node_conformers[node2].append(chain[-1])

    for node_index, conformers in node_conformers.items():
        if len(conformers) == 0:
            continue

        node_attrs = pot.graph.nodes[node_index]
        node_attrs["conformers"] = conformers
        node_attrs["td"] = min(conformers, key=lambda node: node.energy)
        node_attrs["node_energy"] = node_attrs["td"].energy
        node_attrs["node_energies"] = [node.energy for node in conformers]

    return pot
