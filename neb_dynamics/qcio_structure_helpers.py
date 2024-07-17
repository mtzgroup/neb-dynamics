from collections.abc import Iterable
from typing import List, Tuple
import numpy as np
from pathlib import Path

from openbabel import openbabel
from qcio.models.structure import Structure

from neb_dynamics.helper_functions import from_number_to_element, bond_ord_number_to_string
from neb_dynamics.molecule import Molecule
import tempfile
import os


def split_structure_into_frags(structure: Structure) -> list[Structure]:
    """
    will take a Structure and split it into a list of Structures
    corresponding to each fragment present.

    !!! Warning
        When spliting a structure, if one of your fragments is a RADICAL
        you NEED to manually specify the multiplicity for this structure.
        Otherwise, it will default to whatever the total multipliciy
        of the system was in in the input `structure`.
    """
    root_mol = structure_to_molecule(structure=structure)
    mols = root_mol.separate_graph_in_pieces()

    struct_list = []
    for mol in mols:
        # networkx uses NodeView object for indices,
        # which needs to be converted to a list
        struct = Structure(
            symbols=np.array(structure.symbols)[np.array(mol.nodes)],
            geometry=np.array(structure.geometry)[np.array(mol.nodes)],
            charge=mol.charge,
            multiplicity=structure.multiplicity

        )
        struct_list.append(struct)

    return struct_list


def structure_to_molecule(structure: Structure) -> Molecule:
    """
    converts a Structure object to a Molecule object (i.e. a grapical
    2D representation)

   !!! Warning
        Currently uses openbabel to generate edges and approximate element charges.
        Needs work but will have to do for now...
    """
    # write structure object to disk in order to approximate connectivity info
    # with openbabel
    with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w+", delete=False) as tmp:
        tmp.write(structure.to_xyz())
    obmol = load_obmol_from_fp(Path(tmp.name))
    print(tmp.name)
    # os.remove(tmp.name)

    # create molecule object from this OBmol
    new_mol = Molecule()
    for i, (y, z) in enumerate(_atom_iter(obmol)):
        new_mol.add_node(
            i, neighbors=0, element=from_number_to_element(y), charge=z
        )
    for i, j, k in _edge_iter(obmol=obmol):
        if k == 4:
            k_prime = 1.5
        else:
            k_prime = k
        new_mol.add_edge(
            i, j, bond_order=bond_ord_number_to_string(k_prime))
    new_mol.set_neighbors()
    return new_mol


def _atom_iter(obmol: openbabel.OBMol) -> List[Tuple[int, int]]:
    """
    iterates through atoms in openbabel molecule returning
    a list of Tuples

    openbabel is 1-indexed hence the 'i+1' in the code.
    """
    return [
        (
            obmol.GetAtom(i + 1).GetAtomicNum(),
            obmol.GetAtom(i + 1).GetFormalCharge(),
        )
        for i in range(obmol.NumAtoms())
    ]


def _edge_iter(obmol: openbabel.OBMol) -> Iterable:
    return (
        (
            bond.GetBeginAtomIdx() - 1,
            bond.GetEndAtomIdx() - 1,
            1.5 if bond.IsAromatic() else bond.GetBondOrder(),
        )
        for bond in openbabel.OBMolBondIter(obmol)
    )


def load_obmol_from_fp(fp: Path) -> openbabel.OBMol:
    """
    takes in a pathlib file path as input and reads it in as an openbabel molecule
    """
    if not isinstance(fp, Path):
        assert isinstance(fp, str), 'input fp but be a string or a Path'
        fp = Path(fp)
        assert fp.exists(), f"input file path {fp} does not exist"
    file_type = fp.suffix[1:]  # get what type of file this is

    obmol = openbabel.OBMol()
    obconversion = openbabel.OBConversion()
    obconversion.SetInFormat(file_type)
    obconversion.ReadFile(obmol, str(fp.resolve()))

    return obmol