from collections.abc import Iterable

from openbabel import openbabel
from qcio.models.structure import Structure

from neb_dynamics.helper_functions import from_number_to_element, bond_ord_number_to_string
from neb_dynamics.molecule import Molecule
from neb_dynamics.OBH import obmol_from_coords


def split_td_into_frags(structure: Structure) -> list[Structure]:
    """
    will take a Structure and split it into a list of Structures
    corresponding to each fragment present.
    """
    root_mol = structure_to_molecule(structure=structure)
    mols = root_mol.separate_graph_in_pieces()

    td_list = []
    for mol in mols:
        Structure()
        struct = TDStructure.from_coords_symbols(
            coords=self.coords[mol.nodes],
            symbols=self.symbols[mol.nodes],
            tot_charge=mol.charge,
        )
        td.update_tc_parameters(td_ref=self)
        td_list.append(td)

    return td_list


def structure_to_molecule(structure: Structure) -> Molecule:
    """
    converts a Structure object to a Molecule object (i.e. a grapical
    2D representation)

   !!! Warning
        Currently uses openbabel to generate edges and approximate element charges.
        Needs work but will have to do for now...
    """

    obmol = obmol_from_coords(coords=structure.geometry,
                              symbols=structure.symbols,
                              charge=structure.charge,
                              spinmult=structure.multiplicity)

    new_mol = Molecule()
    for i, (y, z) in enumerate(_atom_iter(obmol)):
        new_mol.add_node(
            i, neighbors=0, element=from_number_to_element(y), charge=z
        )
    for i, j, k in self.edge_iter:
        if k == 4:
            k_prime = 1.5
        else:
            k_prime = k
        new_mol.add_edge(
            i, j, bond_order=bond_ord_number_to_string(k_prime))
    new_mol.set_neighbors()
    return new_mol


def _atom_iter(obmol: openbabel.OBMol) -> Iterable:
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
