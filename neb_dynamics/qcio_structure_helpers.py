from __future__ import annotations
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from openbabel import openbabel, pybel
from qcio.models.inputs import ProgramInput
from qcio.models.structure import Structure
from ase import Atoms


from neb_dynamics.constants import ANGSTROM_TO_BOHR
from neb_dynamics.geodesic_interpolation.fileio import read_xyz
from neb_dynamics.helper_functions import (
    bond_ord_number_to_string,
    from_number_to_element,
    atomic_number_to_symbol,
)
from neb_dynamics.elements import symbol_to_atomic_number

from neb_dynamics.molecule import Molecule


def read_multiple_structure_from_file(
    fp: Union[Path, str], charge: int = 0, spinmult: int = 1
) -> List[Structure]:
    """
    will take a file path and return a list of the Structures contained
    """
    symbols, coords = read_xyz(fp)
    coords_bohr = [c * ANGSTROM_TO_BOHR for c in coords]
    return [
        Structure(geometry=c, symbols=symbols,
                  charge=charge, multiplicity=spinmult)
        for c in coords_bohr
    ]


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
            multiplicity=structure.multiplicity,
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
    os.remove(tmp.name)

    # create molecule object from this OBmol
    new_mol = Molecule()
    for i, (y, z) in enumerate(_atom_iter(obmol)):
        new_mol.add_node(
            i, neighbors=0, element=from_number_to_element(y), charge=z)
    for i, j, k in _edge_iter(obmol=obmol):
        if k == 4:
            k_prime = 1.5
        else:
            k_prime = k
        new_mol.add_edge(i, j, bond_order=bond_ord_number_to_string(k_prime))
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
        assert isinstance(fp, str), "input fp but be a string or a Path"
        fp = Path(fp)
        assert fp.exists(), f"input file path {fp} does not exist"
    file_type = fp.suffix[1:]  # get what type of file this is

    obmol = openbabel.OBMol()
    obconversion = openbabel.OBConversion()
    obconversion.SetInFormat(file_type)
    obconversion.ReadFile(obmol, str(fp.resolve()))

    return obmol


def molecule_to_structure(rp_mol: Molecule, charge: int = 0, spinmult: int = 1):
    """Instantiate object from `Molecule` object. see [link](https://mtzgroup.github.io/neb-dynamics/molecule/)

    Args:
        rp_mol (Molecule): Molecule object to build Structure from
        charge (int, optional): charge of molecule. Defaults to 0.
        spinmult (int, optional): spin multiplicity of molecule. Defaults to 1.

    """

    d = {"single": 1, "double": 2, "triple": 3, "aromatic": 1.5}

    obmol = openbabel.OBMol()

    for i, _ in enumerate(rp_mol.nodes):

        node = rp_mol.nodes[i]

        atom_symbol = node["element"]

        atom_number = symbol_to_atomic_number(atom_symbol)
        atom = openbabel.OBAtom()
        atom.SetVector(0, 0, 0)
        atom.SetAtomicNum(atom_number)
        atom.SetFormalCharge(node["charge"])
        atom.SetId(i + 1)
        obmol.AddAtom(atom)

    for rp_atom1_id, rp_atom2_id in rp_mol.edges:
        atom1_id = rp_atom1_id
        atom2_id = rp_atom2_id
        if type(atom1_id) is np.int64:
            atom1_id = int(atom1_id)
        if type(atom2_id) is np.int64:
            atom2_id = int(atom2_id)

        atom1 = obmol.GetAtom(atom1_id + 1)
        atom2 = obmol.GetAtom(atom2_id + 1)

        bond = openbabel.OBBond()
        bond.SetBegin(atom1)
        bond.SetEnd(atom2)

        bond_order = rp_mol.edges[(rp_atom1_id, rp_atom2_id)]["bond_order"]
        if d[bond_order] == 1.5:  # i.e., if an aromatic bond
            bond.SetAromatic(True)
            bond.SetBondOrder(4)
        else:
            bond.SetBondOrder(d[bond_order])

        obmol.AddBond(bond)

    arg = pybel.Molecule(obmol)
    arg.make3D()
    arg.localopt("mmff94", steps=2000)
    arg.localopt("uff", steps=2000)
    arg.localopt("gaff", steps=2000)
    arg.localopt("mmff94", steps=2000)

    xyz_list = []
    atom_nums = []
    for atom in openbabel.OBMolAtomIter(arg.OBMol):
        x = atom.GetX()
        y = atom.GetY()
        z = atom.GetZ()
        atom_nums.append(atom.GetAtomicNum())
        xyz_list.append([x, y, z])
    xyz = np.array(xyz_list)
    symbols = [atomic_number_to_symbol(n) for n in atom_nums]
    geometry = xyz * ANGSTROM_TO_BOHR

    structure = Structure(
        geometry=geometry, symbols=symbols, charge=charge, multiplicity=spinmult
    )

    return structure


def _change_prog_input_property(
    prog_inp: ProgramInput, key: str, value: Union[str, Structure]
):
    prog_dict = prog_inp.__dict__.copy()
    if prog_dict[key] is not value:
        prog_dict[key] = value
        new_prog_inp = ProgramInput(**prog_dict)
    else:
        new_prog_inp = prog_inp

    return new_prog_inp


def structure_to_ase_atoms(structure: Structure):

    symbs = structure.symbols
    pos = structure.geometry_angstrom
    # ASE uses the sum of the partial charges and magnetic moments
    # to determine what the total charge and spinnmultiplicity
    # of the system is, thus the hacky workaround
    atoms = Atoms(
        symbols=symbs,
        positions=pos,
        charges=[structure.charge] + [0] * (len(pos) - 1),
        magmoms=[structure.multiplicity] + [0] * (len(pos) - 1),
    )

    # this is for OMOL25 compatibility
    atoms.info["charge"] = structure.charge
    atoms.info["spin"] = structure.multiplicity
    return atoms


def ase_atoms_to_structure(atoms: Atoms, charge: int = 0, multiplicity: int = 1):
    symbols = atoms.symbols
    positions_angstroms = atoms.positions
    positions_bohr = positions_angstroms * ANGSTROM_TO_BOHR
    structure = Structure(
        symbols=symbols,
        geometry=positions_bohr,
        charge=charge,
        multiplicity=multiplicity,
    )
    return structure
