# This file is OBH: OBMol Helpers. Contains the methods
# for making the OBMOL transformations easier
from pathlib import Path

from openbabel import openbabel, pybel
from elements import ElementData


def get_bond_between_atoms(mol, atom1_id, atom2_id):
    """
    this takes in an obmol object and atomic indices and returns the bond
    between them
    """
    atom1 = mol.GetAtomById(atom1_id)
    atom2 = mol.GetAtomById(atom2_id)

    bond = atom1.GetBond(atom2)

    # check if there was no bond between the atoms
    if bond is None:
        raise ValueError("No bond was given. Returning <None>.")

    return bond


def load_obmol_from_fp(fp: Path) -> openbabel.OBMol:
    """
    takes in a pathlib file path as input and reads it in as an openbabel molecule
    """
    assert isinstance(fp, Path), "fp must be a Path object"
    file_type = fp.suffix[1:]  # get what type of file this is

    obmol = openbabel.OBMol()
    obconversion = openbabel.OBConversion()
    print("using:", fp.resolve())
    obconversion.SetInFormat(file_type)

    obconversion.ReadFile(obmol, str(fp.resolve()))

    return make_copy(obmol)


def from_xyz(coords, symbols):
    ed = ElementData()
    obmol = openbabel.OBMol()
    for i in range(len(coords)):
        x, y, z = coords[i]

        symbol = symbols[i]
        atomic_num = ed.from_symbol(symbol).atomic_num
        atom = openbabel.OBAtom()
        atom.SetVector(x, y, z)
        atom.SetAtomicNum(atomic_num)
        obmol.AddAtom(atom)

    return make_copy(obmol)


def assign_charges_obmol(obmol):
    """
    this function ideally is the one that would calculate atomic charge based on an
    obmol molecule loaded from an XYZ. but for now, because we are using CML files,
    it's just making sure the formal charge on each atom is exactly what it is loaded
    to be
    """
    for atom in openbabel.OBMolAtomIter(obmol):

        obmol.GetAtom(atom.GetIdx()).SetFormalCharge(atom.GetFormalCharge())

    return obmol


def make_copy(obmol):
    copy_obmol = openbabel.OBMol()
    for atom in openbabel.OBMolAtomIter(obmol):
        copy_obmol.AddAtom(atom)

    for bond in openbabel.OBMolBondIter(obmol):
        copy_obmol.AddBond(bond)

    copy_obmol.SetTotalCharge(obmol.GetTotalCharge())
    copy_obmol.SetTotalSpinMultiplicity(obmol.GetTotalSpinMultiplicity())

    return copy_obmol


def atomic_number_to_symbol(n):
    ed = ElementData()
    return ed.from_atomic_number(n).symbol


def symbol_to_atomic_number(str):
    ed = ElementData()
    return ed.from_symbol(str).atomic_num


def obmol_to_coords(obmol):
    return [atom.coords for atom in pybel.Molecule(obmol).atoms]


def obmol_to_symbs(obmol):
    return [atomic_number_to_symbol(atom.atomicnum) for atom in pybel.Molecule(obmol).atoms]


def add_charges(input_mol, charges_list):
    """
    this adds the charge change from retropaths to the openbabel molecule
    """
    for val in charges_list:
        atom_ind, charge = val
        atom = input_mol.GetAtomById(atom_ind)
        atom.SetFormalCharge(atom.GetFormalCharge() + charge)

    input_mol.SetPartialChargesPerceived()


def add_bonds(input_mol, bond_list, bond_order):
    """
    this takes in an openbabel molecule and a list of bonds to create,
    then creates them inplace
    """
    # if instead of a list of bonds I only gave it a single bond...
    if not isinstance(bond_list, list):
        bond_list = [bond_list]  # make it into a list

    # add bonds
    for bond_to_form in bond_list:

        # get start and end atom of the bond
        atom1 = input_mol.GetAtomById(bond_to_form[0])
        atom2 = input_mol.GetAtomById(bond_to_form[1])

        if atom1.GetBond(
            atom2
        ):  # i.e. if there already exists a bond between these two....
            input_mol.DeleteBond(atom1.GetBond(atom2))  # ... then delete it

        # create a new bond object
        bond = openbabel.OBBond()

        # set the properties of the bond
        bond.SetBegin(atom1)
        bond.SetEnd(atom2)
        if bond_order == 1.5:  # i.e., if an aromatic bond
            bond.SetAromatic(True)
            bond.SetBondOrder(4)
        else:
            bond.SetBondOrder(bond_order)

        # add the new bond to the inputted molecule
        input_mol.AddBond(bond)


def align_molecules(structure_obj, changes_info, v=False):
    """
    this function creates a pseudo-alignment between fragments that will be bonded.
    In other words, it will apply all bonds between atoms that were not bonded before,
    then it will apply FF to get a geometry, then it will break these bonds again and
    rerun the FF to have the atoms in the orientation for the bonds that will form.
    """
    # create a RP representation of the obmol molecule so we can check for bonds that are happening between fragments
    obmol_rp = structure_obj.molecule_rp
    obmol = make_copy(structure_obj.molecule_obmol)

    # add bonds
    for bond in changes_info.bonds.single:
        if not obmol_rp.get_edge_data(
            bond[0], bond[1]
        ):  # i.e. if this connection did not exist before,
            # i.e. if it's a new fragment bond
            add_bonds(obmol, bond, 1)

    for bond in changes_info.bonds.double:
        if not obmol_rp.get_edge_data(bond[0], bond[1]):
            add_bonds(obmol, bond, 2)

    for bond in changes_info.bonds.triple:
        if not obmol_rp.get_edge_data(bond[0], bond[1]):
            add_bonds(obmol, bond, 3)

    # JDE: I don't 'align' relative to aromatic bonds because this was breaking the process somehow.

    # run local MM optimization using GAFF and UFF
    pybel_out_mol = pybel.Molecule(obmol)
    pybel_out_mol.localopt("gaff", steps=500)
    pybel_out_mol.localopt("uff", steps=500)

    # delete all the bonds we just formed for the purpose of alignment
    for bond_to_delete in changes_info.bonds.single:
        if not obmol_rp.get_edge_data(bond_to_delete[0], bond_to_delete[1]):
            bond = get_bond_between_atoms(obmol, bond_to_delete[0], bond_to_delete[1])
            obmol.DeleteBond(bond)

    for bond_to_delete in changes_info.bonds.double:
        if not obmol_rp.get_edge_data(bond_to_delete[0], bond_to_delete[1]):
            bond = get_bond_between_atoms(obmol, bond_to_delete[0], bond_to_delete[1])
            obmol.DeleteBond(bond)

    for bond_to_delete in changes_info.bonds.triple:
        if not obmol_rp.get_edge_data(bond_to_delete[0], bond_to_delete[1]):
            bond = get_bond_between_atoms(obmol, bond_to_delete[0], bond_to_delete[1])
            obmol.DeleteBond(bond)

    # run local MM optimization using GAFF and UFF
    pybel_out_mol = pybel.Molecule(obmol)
    pybel_out_mol.localopt("gaff", steps=500)
    pybel_out_mol.localopt("uff", steps=500)

    return pybel_out_mol.OBMol


def output_obmol_to_file(obmol, file_path):
    if not isinstance(file_path, Path):
        fp_obj = Path(file_path)
    else:
        fp_obj = file_path
    out_mol = pybel.Molecule(obmol)
    suffix = fp_obj.suffix
    outfile = pybel.Outputfile(
        suffix[1:],
        str(fp_obj.resolve()),
        overwrite=True,
    )
    outfile.write(out_mol)


def obmol_from_coords(coords, symbols, charge=0, spinmult=1):
    obmol = openbabel.OBMol()
    obmol.SetTotalCharge(charge)
    obmol.SetTotalSpinMultiplicity(spinmult)
    for i in range(len(coords)):
        x, y, z = coords[i]
        atom_symbol = symbols[i]

        atom_number = symbol_to_atomic_number(atom_symbol)
        atom = openbabel.OBAtom()
        atom.SetVector(x, y, z)
        atom.SetAtomicNum(atom_number)
        obmol.AddAtom(atom)
    return obmol


def obmol_from_coords_and_reference(coords, symbols, reference):
    obmol = openbabel.OBMol()
    for i in range(len(coords)):
        x, y, z = coords[i]
        atom_symbol = symbols[i]

        atom_number = symbol_to_atomic_number(atom_symbol)
        atom = openbabel.OBAtom()
        atom.SetVector(x, y, z)
        atom.SetAtomicNum(atom_number)
        obmol.AddAtom(atom)

    for bond in openbabel.OBMolBondIter(reference):
        obmol.AddBond(bond)

    return obmol
