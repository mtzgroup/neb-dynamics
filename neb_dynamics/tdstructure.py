from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import py3Dmol
from ase import Atoms
from ase.optimize import BFGS, LBFGS, FIRE
from chemcloud import CCClient
# from chemcloud.models import AtomicInput
from qcio import ProgramInput, DualProgramInput
from qcio import Molecule as TCMolecule
from IPython.core.display import HTML
from openbabel import openbabel, pybel
from openeye import oechem
from xtb.ase.calculator import XTB
from xtb.interface import Calculator, XTBException
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics import OBH
from retropaths.abinitio.abinitio import write_xyz
from neb_dynamics.constants import angstroms_to_bohr
from neb_dynamics.geodesic_interpolation.coord_utils import align_geom
from retropaths.molecules.elements import (
    atomic_number_to_symbol,
    symbol_to_atomic_number,
)
from retropaths.molecules.molecule import Molecule
from retropaths.molecules.smiles_tools import (
    bond_ord_number_to_string,
    from_number_to_element,
)
from retropaths.reactions.changes import Changes3D, Changes3DList, ChargeChanges
from retropaths.reactions.library import Library
from retropaths.reactions.r_handler import RHandler
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.template_utilities import (
    give_me_molecule_with_random_replacement_from_rules,
)


from qcparse import parse
import tempfile
import subprocess
import shutil

angstroms_to_bohr = 1.8897259886
bohr_to_angstroms = 1 / angstroms_to_bohr
# q = 'private'
q = None

# ES_PROGRAM = 'psi4'
ES_PROGRAM = 'terachem'

@dataclass
class TDStructure:
    """
    this is the object that will contains all the info for a single Three D structure
    """

    molecule_obmol: openbabel.OBMol

    tc_model_method: str = "b3lyp"
    tc_model_basis: str = "6-31g"

    tc_kwds: dict = field(default_factory=dict)
    tc_geom_opt_kwds: dict = field(default_factory= lambda: {"maxiter": 300, 'trust': 0.005})
    _cached_nma: list = None
    _cached_freqs: list = None

    @property
    def tc_d_model(self):
        return {"method": self.tc_model_method, "basis": self.tc_model_basis}

    @classmethod
    def from_RP(self, rp_mol: Molecule, charge=0, spinmult=1):
        """
        ideally, I don't want to have to give in a total charge as in input. But
        when a TDStructure is read from an xyz, it doesnt know which atom holds the
        charge the way a graph does, so RP charge will end up calculating to 0
        """

        d = {"single": 1, "double": 2, "triple": 3, "aromatic": 1.5}

        ind_mapping = {}

        obmol = openbabel.OBMol()

        for n in range(len(rp_mol.nodes)):
            node_ind = list(rp_mol.nodes)[n]

            ind_mapping[
                node_ind
            ] = n  # dict for knowing what the RP index is in my index

            node = rp_mol.nodes[node_ind]

            atom_symbol = node["element"]

            atom_number = symbol_to_atomic_number(atom_symbol)
            atom = openbabel.OBAtom()
            atom.SetVector(0, 0, 0)
            atom.SetAtomicNum(atom_number)
            atom.SetFormalCharge(node["charge"])
            atom.SetId(n + 1)
            obmol.AddAtom(atom)

        for rp_atom1_id, rp_atom2_id in rp_mol.edges:
            atom1_id = ind_mapping[rp_atom1_id]
            atom2_id = ind_mapping[rp_atom2_id]
            # TODO: figure out why you would pass in a np int64...
            if type(atom1_id) == np.int64:
                atom1_id = int(atom1_id)
            if type(atom2_id) == np.int64:
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
        arg.localopt("uff", steps=2000)
        arg.localopt("gaff", steps=2000)
        arg.localopt("mmff94", steps=2000)

        obmol3D = arg.OBMol
        obmol3D.SetTotalCharge(charge)
        obmol3D.SetTotalSpinMultiplicity(spinmult)

        return TDStructure(obmol3D)

    @property
    def coords(self) -> np.array:
        """
        this return coordinates in Angstroms
        """
        return np.array(
            [
                (atom.x(), atom.y(), atom.z())
                for atom in openbabel.OBMolAtomIter(self.molecule_obmol)
            ]
        )

    @property
    def coords_bohr(self) -> np.array:
        return self.coords * angstroms_to_bohr

    def update_coords(self, coords: np.array):
        # for i, (x, y, z) in enumerate(coords):

        #     atom = self.molecule_obmol.GetAtom(i+1)
        #     atom.SetVector(x, y, z)

        string = write_xyz(self.symbols, coords)

        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as tmp:
            tmp.write(string)

        td = TDStructure.from_xyz(
            tmp.name, tot_charge=self.charge, tot_spinmult=self.spinmult
        )
        os.remove(tmp.name)
        return td

    def remove_Hs_3d(self):

        atoms_to_del = []
        for atom in openbabel.OBMolAtomIter(self.molecule_obmol):
            if atom.GetAtomicNum() == 1:
                atoms_to_del.append(atom)
        [self.molecule_obmol.DeleteAtom(a) for a in atoms_to_del]

    @property
    def symbols(self):
        return np.array(
            [
                self._atomic_number_to_symbol(atom.atomicnum)
                for atom in pybel.Molecule(self.molecule_obmol).atoms
            ]
        )

    def align_to_td(self, other_td: TDStructure):
        __, new_coords = align_geom(other_td.coords, self.coords)
        new_td = self.update_coords(new_coords)
        return new_td

    @property
    def atomic_numbers(self):
        return np.array([self._symbol_to_atomic_number(s) for s in self.symbols])

    def pseudoalign(self, changes_list):
        mol1_3d = self.copy()

        mol1_3d.add_bonds_from_changes3d(c3d=changes_list)
        mol1_3d.gum_mm_optimization()

        mol1_3d.delete_formed_from_changes3d(c3d=changes_list)

        mol1_3d.gum_mm_optimization()

        return mol1_3d

    def gum_mm_optimization(self):
        self.mm_optimization("gaff")
        self.mm_optimization("uff")
        self.mm_optimization("mmff94")

    @property
    def atom_iter(self):
        # return (
        #     (atom.GetAtomicNum(), atom.GetFormalCharge())
        #     for atom in self.molecule_obmol.GetAtomi))
        obmol = self.molecule_obmol
        return [
            (
                obmol.GetAtom(i + 1).GetAtomicNum(),
                obmol.GetAtom(i + 1).GetFormalCharge(),
            )
            for i in range(obmol.NumAtoms())
        ]

    @property
    def atomn(self):
        """
        returns number of atoms
        """
        return self.molecule_obmol.NumAtoms()

    @property
    def symbols(self):
        return np.array(
            [
                atomic_number_to_symbol(atom.atomicnum)
                for atom in pybel.Molecule(self.molecule_obmol).atoms
            ]
        )

    @property
    def atomic_numbers(self):
        return np.array([symbol_to_atomic_number(s) for s in self.symbols])

    def view_mol(self, string_mode=False, style="sphere", center=True, custom_image=""):
        if center:

            def center(nested_array_list):
                a = np.array(nested_array_list)
                return np.mean(a, axis=0)

            coords = self.coords - center(self.coords)
        else:
            coords = self.coords

        frame = write_xyz(self.symbols, coords)
        viewer = py3Dmol.view(width=400, height=400)
        viewer.addModel(frame, "xyz")

        if style == "stick":
            viewer.setStyle({"stick": {}})
        else:
            viewer.setStyle({"sphere": {"scale": "0.5"}})

        rp = self.molecule_rp
        s = f"""
        <p style="text-align: left; font-weight: bold;">{rp.smiles}</p>
        <div style="width: 70%; display: table;"> <div style="display: table-row;">
        <div style="width: 20%; display: table-cell;">
        {viewer._make_html()}
        </div>
        <div style="width: 20%; display: table-cell; border: 1px solid black;">
        {rp.draw(string_mode=True)}
        </div>
        <div style="width: 40%; display: table-cell;">
        {custom_image}
        </div>
        </div>

        """
        if string_mode:
            return s
        else:
            return HTML(s)

    def _repr_html_(self):
        return self.view_mol(string_mode=True)

    @property
    def formal_charges(self):
        return [
            atom.GetFormalCharge()
            for atom in openbabel.OBMolAtomIter(self.molecule_obmol)
        ]

    @property
    def atom_charge_iter(self):
        """
        iter through atomic number and charge for every atom in OBmolecule
        """
        return (
            (atom.GetAtomicNum(), atom.GetFormalCharge())
            for atom in openbabel.OBMolAtomIter(self.molecule_obmol)
        )

    def apply_charges_changes(self, charges_list: ChargeChanges):
        """
        this apply the charge change from retropaths to the openbabel molecule
        """
        for val in charges_list.charges:
            atom_ind, charge = val
            atom = self.molecule_obmol.GetAtom(atom_ind + 1)
            atom.SetFormalCharge(atom.GetFormalCharge() + charge)

        self.molecule_obmol.SetPartialChargesPerceived()

    @property
    def edge_iter(self):
        return (
            (
                bond.GetBeginAtomIdx() - 1,
                bond.GetEndAtomIdx() - 1,
                1.5 if bond.IsAromatic() else bond.GetBondOrder(),
            )
            for bond in openbabel.OBMolBondIter(self.molecule_obmol)
        )

    @property
    def charge(self):
        return self.molecule_obmol.GetTotalCharge()

    @property
    def spinmult(self):
        return self.molecule_obmol.GetTotalSpinMultiplicity()
        # return self.tot_spinmult

    @property
    def molecule_rp(self):
        """
        converts an openbabel molecule to a retropaths molecule
        """
        new_mol = Molecule()
        for i, (y, z) in enumerate(self.atom_iter):
            new_mol.add_node(
                i, neighbors=0, element=from_number_to_element(y), charge=z
            )
        for i, j, k in self.edge_iter:
            if k == 4:
                k_prime = 1.5
            else:
                k_prime = k
            new_mol.add_edge(i, j, bond_order=bond_ord_number_to_string(k_prime))
        new_mol.set_neighbors()
        return new_mol

    
    def gfn1xtb_geom_optimization(self):
        # XTB api is summing the initial charges from the ATOM object.
        # it returns a vector of charges (maybe Mulliken), but to initialize the calculation,
        # it literally sums this vector up. So we create a zero vector (natoms long) and we
        # modify the charge of the first atom to be total charge.
        charges = np.zeros(self.atomn)
        charges[0] = self.charge

        # JAN IS GONNA EXPLAIN THIS
        spins = np.zeros(self.atomn)
        spins[0] = self.spinmult - 1

        atoms = Atoms(
            symbols=self.symbols.tolist(),
            positions=self.coords,
            charges=charges,
            magmoms=spins,
        )

        atoms.calc = XTB(method="GFN1-xTB", accuracy=0.1)
        opt = LBFGS(atoms, logfile=None)
        # opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.001)
        new_tds = TDStructure.from_ase_Atoms(
            atoms=atoms, charge=self.charge, spinmult=self.spinmult
        )
        return new_tds

    def xtb_geom_optimization(self):
        # XTB api is summing the initial charges from the ATOM object.
        # it returns a vector of charges (maybe Mulliken), but to initialize the calculation,
        # it literally sums this vector up. So we create a zero vector (natoms long) and we
        # modify the charge of the first atom to be total charge.
        charges = np.zeros(self.atomn)
        charges[0] = self.charge

        # JAN IS GONNA EXPLAIN THIS
        spins = np.zeros(self.atomn)
        spins[0] = self.spinmult - 1

        atoms = Atoms(
            symbols=self.symbols.tolist(),
            positions=self.coords,
            charges=charges,
            magmoms=spins,
        )

        atoms.calc = XTB(method="GFN2-xTB", accuracy=0.001)
        # opt = BFGS(atoms, logfile=None)
        opt = FIRE(atoms, logfile=None)
        # opt = /(atoms, logfile=None)
        opt.run(fmax=0.005)
        new_tds = TDStructure.from_ase_Atoms(
            atoms=atoms, charge=self.charge, spinmult=self.spinmult
        )
        return new_tds

    def get_changes_in_3d_with_target(self, rxn, final_target):
        """
        Sorry bad Alessio.
        """
        target_molecule = self.molecule_rp
        templ_graph_to_match = rxn.reactants

        for changes in rxn.changes_react_to_prod:

            isomorphisms = templ_graph_to_match.get_subgraph_isomorphisms_of(
                target_molecule
            )
            assert (
                len(isomorphisms) > 0
            ), "You gave the get_changes_in_3d function a template that does not match."
            this_isomorphism = isomorphisms[0]
            r_handler = RHandler(
                this_isomorphism, target_molecule, templ_graph_to_match
            )
            new_isomorphism = this_isomorphism.reverse_mapping_update(
                r_handler.missing_isomorphism_mapping
            )
            changes_in_3d_mol = changes.translate_changes(
                new_isomorphism.reverse_mapping
            )
            new_mol = target_molecule.update_edges_and_charges(changes_in_3d_mol)

            for x, y in changes_in_3d_mol.bonds.delete:
                assert self.was_there_a_bond_here(
                    x, y
                ), "U trin' to get rid of a bond which is not there"

            deleted = [Changes3D(x, y, 0) for x, y in changes_in_3d_mol.bonds.delete]
            forming = [
                Changes3D(x, y, type_of_bond)
                for x, y, type_of_bond in changes_in_3d_mol.list_tuple_of_forming_bonds()
                if not self.was_there_a_bond_here(x, y)
            ]

            final_3d_changes = Changes3DList(
                deleted=deleted, forming=forming, charges=changes_in_3d_mol.charges
            )
            if new_mol.is_isomorphic_to(final_target):
                return final_3d_changes

    def get_changes_in_3d(self, rxn):
        """
        Sorry bad Alessio.
        """
        target_molecule = self.molecule_rp
        templ_graph_to_match = rxn.reactants

        all_3d_changes = []

        for changes in rxn.changes_react_to_prod:

            isomorphisms = templ_graph_to_match.get_subgraph_isomorphisms_of(
                target_molecule
            )
            assert (
                len(isomorphisms) > 0
            ), "You gave the get_changes_in_3d function a template that does not match."
            this_isomorphism = isomorphisms[0]
            r_handler = RHandler(
                this_isomorphism, target_molecule, templ_graph_to_match
            )
            new_isomorphism = this_isomorphism.reverse_mapping_update(
                r_handler.missing_isomorphism_mapping
            )
            changes_in_3d_mol = changes.translate_changes(
                new_isomorphism.reverse_mapping
            )
            new_mol = target_molecule.update_edges_and_charges(changes_in_3d_mol)

            for x, y in changes_in_3d_mol.bonds.delete:
                assert self.was_there_a_bond_here(
                    x, y
                ), "U tryin' to get rid of a bond which is not there"

            deleted = [Changes3D(x, y, 0) for x, y in changes_in_3d_mol.bonds.delete]
            forming = [
                Changes3D(x, y, type_of_bond)
                for x, y, type_of_bond in changes_in_3d_mol.list_tuple_of_forming_bonds()
                if not self.was_there_a_bond_here(x, y)
            ]

        return Changes3DList(
            deleted=deleted, forming=forming, charges=changes_in_3d_mol.charges
        )

    def get_changes_in_3d_backwards(self, rxn):
        """
        Sorry bad Alessio.
        """
        target_molecule = self.molecule_rp
        templ_graph_to_match = rxn.products

        all_3d_changes = []

        for changes in rxn.changes_prod_to_react:

            isomorphisms = templ_graph_to_match.get_subgraph_isomorphisms_of(
                target_molecule
            )
            assert (
                len(isomorphisms) > 0
            ), "You gave the get_changes_in_3d function a template that does not match."
            this_isomorphism = isomorphisms[0]
            r_handler = RHandler(
                this_isomorphism, target_molecule, templ_graph_to_match
            )
            new_isomorphism = this_isomorphism.reverse_mapping_update(
                r_handler.missing_isomorphism_mapping
            )
            changes_in_3d_mol = changes.translate_changes(
                new_isomorphism.reverse_mapping
            )
            new_mol = target_molecule.update_edges_and_charges(changes_in_3d_mol)

            for x, y in changes_in_3d_mol.bonds.delete:
                assert self.was_there_a_bond_here(
                    x, y
                ), "U tryin' to get rid of a bond which is not there"

            deleted = [Changes3D(x, y, 0) for x, y in changes_in_3d_mol.bonds.delete]
            forming = [
                Changes3D(x, y, type_of_bond)
                for x, y, type_of_bond in changes_in_3d_mol.list_tuple_of_forming_bonds()
                if not self.was_there_a_bond_here(x, y)
            ]

        return Changes3DList(
            deleted=deleted, forming=forming, charges=changes_in_3d_mol.charges
        )

    # @classmethod
    # def from_smiles(cls, smi, tot_spinmult=1, tot_charge=0):
    #     pybel_mol = pybel.readstring("smi", smi)
    #     pybel_mol.make3D()
    #     pybel_mol.localopt("gaff")
    #     obmol = pybel_mol.OBMol
    #     obmol.SetTotalSpinMultiplicity(tot_spinmult)
    #     obmol.SetTotalCharge(tot_charge)
    #     return cls(molecule_obmol=obmol)

    def was_there_a_bond_here(self, x, y):
        """
        I need this function to understand which bonds are created and which are destroyed.
        """
        try:
            self.molecule_rp.edges[x, y]
            return True
        except KeyError:
            return False

    def add_bonds_from_changes3d(self, c3d: Changes3DList):
        self.add_bonds(c3d.forming)

    def delete_formed_from_changes3d(self, c3d: Changes3DList):
        self.delete_bonds(c3d.forming)

    def apply_changed3d_list(self, c3d_list: Changes3DList):
        self.add_bonds(c3d_list.forming)
        self.delete_bonds(c3d_list.deleted)
        self.apply_charges_changes(c3d_list.charges)
        self.gum_mm_optimization()

    def add_bonds(self, c3d_list: list[Changes3D]):
        """
        this takes in an openbabel molecule and a list of bonds to create,
        then creates them inplace
        """
        for c3d in c3d_list:
            atom1 = self.molecule_obmol.GetAtom(c3d.start + 1)
            atom2 = self.molecule_obmol.GetAtom(c3d.end + 1)
            bond_order = c3d.bond_order

            bond = openbabel.OBBond()
            bond.SetBegin(atom1)
            bond.SetEnd(atom2)
            if bond_order == 1.5:  # i.e., if an aromatic bond
                bond.SetAromatic(True)
                bond.SetBondOrder(4)
            else:
                bond.SetBondOrder(bond_order)
            self.molecule_obmol.AddBond(bond)
        

    def delete_bonds(self, c3d_list: list[Changes3D]):
        original_charge = self.charge

        for c3d in c3d_list:
            if self.was_there_a_bond_here(c3d.start, c3d.end):
                bond = self.get_bond_between_atoms(c3d.start, c3d.end)
                self.molecule_obmol.DeleteBond(bond)
            else:
                print(
                    f"I am not deleting a bond that is not there: {c3d.start} -> {c3d.end}"
                )

        ### JDEP 09/10/2023: for some reason, openbabel changes the total charge
        # when a bond is deleted. Need to add a hack here to make sure charge
        # stays the same. 
        self.molecule_obmol.SetTotalCharge(original_charge)

    def copy(self):
        obmol = self.molecule_obmol
        copy_obmol = openbabel.OBMol()
        for atom in openbabel.OBMolAtomIter(obmol):
            copy_obmol.AddAtom(atom)

        for bond in openbabel.OBMolBondIter(obmol):
            copy_obmol.AddBond(bond)

        copy_obmol.SetTotalCharge(obmol.GetTotalCharge())
        copy_obmol.SetTotalSpinMultiplicity(obmol.GetTotalSpinMultiplicity())

        tc_model_method = self.tc_model_method
        tc_model_basis = self.tc_model_basis

        tc_kwds = self.tc_kwds.copy()
        tc_geom_opt_kwds = self.tc_geom_opt_kwds.copy()

        return TDStructure(
            molecule_obmol=copy_obmol,
            tc_model_method=tc_model_method,
            tc_model_basis=tc_model_basis,
            tc_kwds=tc_kwds,
            tc_geom_opt_kwds=tc_geom_opt_kwds,
        )
        
    def get_bond_between_atoms(self, atom1_id, atom2_id):
        """
        this takes in an obmol object and atomic indices and returns the bond
        between them
        """
        atom1 = self.molecule_obmol.GetAtom(atom1_id + 1)
        atom2 = self.molecule_obmol.GetAtom(atom2_id + 1)
        return atom1.GetBond(atom2)

    def mm_optimization(self, method="gaff", steps=2000):
        """
        in place MM optimization
        """
        pybel_mol = pybel.Molecule(self.molecule_obmol)
        pybel_mol.localopt(method, steps=steps)
        self.molecule_obmol = pybel_mol.OBMol

    @property
    def xyz(self):
        return write_xyz(self.symbols, self.coords)

    def to_xyz(self, fn: Path):
        with open(fn, "w+") as f:
            f.write(self.xyz)

    def reaction(self, rxn: ReactionTemplate):
        changes3d_list = self.get_changes_in_3d(rxn)
        assert len(changes3d_list) > 0, "Check this rxn, which does not apply to tds."
        changes3d = changes3d_list[0]
        tds = self.copy()
        tds.apply_changed3d_list(changes3d)
        tds.mm_optimization(method="uff")
        return tds

    def move_atom(self, atom_index, new_x, new_y, new_z):
        """
        this method moves the atom to the new coordinates
        """
        atom = self.molecule_obmol.GetAtom(atom_index + 1)
        atom.SetVector(new_x, new_y, new_z)

    def energy_xtb(self):
        try:
            calc = Calculator(
                get_method("GFN2-xTB"),
                self.atomic_numbers,
                self.coords_bohr,
                charge=self.charge,
                uhf=self.spinmult - 1,
            )
            calc.set_verbosity(VERBOSITY_MUTED)
            res = calc.singlepoint()
            return res.get_energy()
        except XTBException:
            return None

    def gradient_xtb(self):
        calc = Calculator(
            get_method("GFN2-xTB"),
            self.atomic_numbers,
            self.coords_bohr,
            charge=self.charge,
            uhf=self.spinmult - 1,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()
        return res.get_gradient() * bohr_to_angstroms

    def set_charge(self, charge):
        self.molecule_obmol.SetTotalCharge(charge)

    def set_spinmult(self, tot_spinmult):
        self.molecule_obmol.SetTotalSpinMultiplicity(tot_spinmult)

    @classmethod
    def from_ase_Atoms(cls, atoms: Atoms, charge: int, spinmult: int):
        atomT = np.asarray([from_number_to_element(x) for x in atoms.numbers])
        string = write_xyz(atomT, atoms.get_positions())

        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as tmp:
            tmp.write(string)

        td = cls.from_xyz(tmp.name, tot_charge=charge, tot_spinmult=spinmult)
        os.remove(tmp.name)
        return td

    @classmethod
    def from_ase_Atoms2(cls, atoms_object: Atoms):
        coords = atoms_object.get_positions()
        numbers = atoms_object.symbols.numbers
        charge = int(sum(atoms_object.get_initial_charges()))
        # the get_initial_magnetic_moments vector should be already 2S and we need to add 1 to get spinmult.
        spinmult = int(sum(atoms_object.get_initial_magnetic_moments())) + 1

        obmol = openbabel.OBMol()
        obmol.SetTotalCharge(charge)
        obmol.SetTotalSpinMultiplicity(spinmult)
        for (x, y, z), atom_number in zip(coords, numbers):
            atom = openbabel.OBAtom()
            atom.SetVector(x, y, z)
            atom.SetAtomicNum(int(atom_number))
            obmol.AddAtom(atom)
        return cls(molecule_obmol=obmol)

    @classmethod
    def from_smiles(cls, smi, tot_spinmult=1):
        pybel_mol = pybel.readstring("smi", smi)
        pybel_mol.make3D()
        pybel_mol.localopt("gaff")
        obmol = pybel_mol.OBMol
        obmol.SetTotalSpinMultiplicity(tot_spinmult)
        obj =  cls(molecule_obmol=obmol)
        obj.gum_mm_optimization()
        return obj

    def update_charges_from_reference(self, reference: TDStructure):
        mol_ref = reference.molecule_rp
        mol = self.molecule_rp

        mapping = mol_ref.get_bond_subgraph_isomorphisms_of(mol)
        for key, val in mapping[0].reverse_mapping.items():
            charge = mol_ref.nodes[key]["charge"]
            self.molecule_obmol.GetAtom(val + 1).SetFormalCharge(charge)

    @classmethod
    def from_oe(cls, oe_mol, tot_charge=None, tot_spinmult=None):

        if tot_charge is not None:
            charge = tot_charge
        else:
            charge = sum(x.GetFormalCharge() for x in oe_mol.GetAtoms())
            print(
                "Warning! Input total charge was None. Guessing the charge from formal charges!"
            )
        if tot_spinmult is not None:
            spinmult = tot_spinmult
        else:
            spinmult = 1

        numbers = [atom.GetAtomicNum() for atom in oe_mol.GetAtoms()]
        coords = [oe_mol.GetCoords(atom) for atom in oe_mol.GetAtoms()]

        obmol = openbabel.OBMol()
        obmol.SetTotalCharge(charge)
        obmol.SetTotalSpinMultiplicity(spinmult)

        for (x, y, z), atom_number in zip(coords, numbers):
            atom = openbabel.OBAtom()
            atom.SetVector(x, y, z)
            atom.SetAtomicNum(int(atom_number))
            obmol.AddAtom(atom)

        for oe_bond in oe_mol.GetBonds():
            atom1_id = oe_bond.GetBgnIdx()
            atom2_id = oe_bond.GetEndIdx()
            bond_order = oe_bond.GetOrder()
            atom1 = obmol.GetAtom(atom1_id + 1)
            atom2 = obmol.GetAtom(atom2_id + 1)

            bond = openbabel.OBBond()
            bond.SetBegin(atom1)
            bond.SetEnd(atom2)

            if oe_bond.IsAromatic():  # i.e., if an aromatic bond
                bond.SetAromatic(True)
                bond.SetBondOrder(4)
            else:
                bond.SetBondOrder(bond_order)

            obmol.AddBond(bond)

        return cls(obmol)

    @classmethod
    def from_coords_symbols2(cls, coords, symbols, tot_charge=0, tot_spinmult=1):
        obmol = openbabel.OBMol()
        obmol.SetTotalCharge(tot_charge)
        obmol.SetTotalSpinMultiplicity(tot_spinmult)
        for i in range(len(coords)):
            x, y, z = coords[i]
            atom_symbol = symbols[i]

            atom_number = symbol_to_atomic_number(atom_symbol)
            atom = openbabel.OBAtom()
            atom.SetVector(x, y, z)
            atom.SetAtomicNum(atom_number)
            obmol.AddAtom(atom)
        return cls(molecule_obmol=obmol)

    @classmethod
    def from_xyz(cls, fp: Path, tot_charge=0, tot_spinmult=1):
        if isinstance(fp, str):
            fp = Path(fp)
        if 'OE_LICENSE' not in os.environ:
            obmol = OBH.load_obmol_from_fp(fp)
            obmol.SetTotalCharge(tot_charge)
            obmol.SetTotalSpinMultiplicity(tot_spinmult)
            return cls(molecule_obmol=obmol)
        
        else:
            ifs = oechem.oemolistream()
            ifs.SetFormat(oechem.OEFormat_XYZ)
            if ifs.open(str(fp.resolve())):
                for mol in ifs.GetOEGraphMols():
                    return cls.from_oe(
                        mol, tot_charge=tot_charge, tot_spinmult=tot_spinmult
                    )
    
    @classmethod
    def from_cc_result(cls, result):
        if hasattr(result, 'final_molecule'):
            mol = result.final_molecule
        else:
            mol = result.molecule
        
        coords = mol.geometry
        symbols = mol.symbols
        
        
        td = TDStructure.from_coords_symbols(
            coords=coords * (bohr_to_angstroms),
            symbols=symbols,
            tot_charge=int(mol.charge),
            tot_spinmult=mol.multiplicity,
        )

        return td

    @classmethod
    def from_xyz_string(cls, string, tot_charge=0, tot_spinmult=1):
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as tmp:
            tmp.write(string)

        td = cls.from_xyz(tmp.name, tot_charge=tot_charge, tot_spinmult=tot_spinmult)
        os.remove(tmp.name)
        return td

    @classmethod
    def from_coords_symbols(cls, coords, symbols, tot_charge=0, tot_spinmult=1):
        string = write_xyz(symbols, coords)
        return cls.from_xyz_string(
            string, tot_charge=tot_charge, tot_spinmult=tot_spinmult
        )
    
    
    def update_tc_parameters(self, td_ref: TDStructure):
        tc_model_method = td_ref.tc_model_method
        tc_model_basis = td_ref.tc_model_basis
        tc_kwds = td_ref.tc_kwds.copy()
        tc_geom_opt_kwds = td_ref.tc_geom_opt_kwds.copy()
        
        self.tc_model_method = tc_model_method
        self.tc_model_basis = tc_model_basis
        self.tc_kwds = tc_kwds
        self.tc_geom_opt_kwds = tc_geom_opt_kwds
        
        
        
    @classmethod
    def from_rxn_name(
        cls,
        rxn_name: str,
        library: Library,
        tot_spinmult=1,
    ):
        rxn = library[rxn_name]
        mol = give_me_molecule_with_random_replacement_from_rules(rxn, library.matching)
        smi = mol.force_smiles()
        return cls.from_smiles(smi, tot_spinmult=tot_spinmult)

    def remove_water(self, count="all"):
        water = Molecule.from_smiles("O")
        if count == "all":
            new_mol_rp = self.molecule_rp.remove_all_of_molecule(water)
        elif isinstance(count, int):
            new_mol_rp = self.molecule_rp.remove_n_of_molecule(water, n=count)
        else:
            raise ValueError("WTF")
        new_td = TDStructure.from_RP(
            new_mol_rp, charge=self.charge, spinmult=self.spinmult
        )

        inds_of_solute = new_mol_rp.nodes
        new_coords = self.coords[inds_of_solute]
        new_td = new_td.update_coords(new_coords)
        return new_td

    def split_td_into_frags(self):
        root_mol = self.molecule_rp
        mols = root_mol.separate_graph_in_pieces()

        td_list = [
            TDStructure.from_coords_symbols(
                coords=self.coords[mol.nodes],
                symbols=self.symbols[mol.nodes],
                tot_charge=mol.charge,
            )
            for mol in mols
        ]

        return td_list

    @property
    def tc_client(self):
        client = CCClient()
        return client

    def _prepare_input(self, method):
        allowed_methods = ["energy", "optimization", "gradient", "transition_state", "hessian"]

        if method.lower() not in allowed_methods:
            print(f"{method} not allowed. Methods: {allowed_methods}")
            return

        tc_mol = self._as_tc_molecule()
        if method == "energy":
            prog_input = ProgramInput(
                calctype="energy",
                molecule=tc_mol,
                model=self.tc_d_model,
                keywords=self.tc_kwds,
            )
            inp = prog_input

        if method == "gradient":
            prog_input = ProgramInput(
                calctype="gradient",
                molecule=tc_mol,
                model=self.tc_d_model,
                keywords=self.tc_kwds,
            )
            inp = prog_input

        elif method == "optimization":
            opt_input = DualProgramInput(
                calctype="optimization",
                molecule=tc_mol,
                keywords=self.tc_geom_opt_kwds,
                subprogram=ES_PROGRAM,
                subprogram_args={'model':self.tc_d_model, 'keywords':self.tc_kwds}
            )
            inp = opt_input
            
        elif method == "transition_state":
            opt_input = DualProgramInput(
                calctype='transition_state',
                molecule=tc_mol,
                keywords=self.tc_geom_opt_kwds,
                subprogram=ES_PROGRAM,
                subprogram_args={'model':self.tc_d_model, 'keywords':self.tc_kwds}
            )
            inp = opt_input
            
        elif method == "hessian":
            prog_input = DualProgramInput(
                calctype='hessian',
                # keywords={'temperature':300.0},
                molecule=tc_mol,
                subprogram=ES_PROGRAM,
                subprogram_args={'model':self.tc_d_model, 'keywords':self.tc_kwds},
            )
            inp = prog_input
            


        return inp

    def _as_tc_molecule(self):
        d = {
            "symbols": self.symbols,
            "geometry": self.coords_bohr,
            "multiplicity": self.spinmult,
            "charge": self.charge,
        }
        tc_mol = TCMolecule(**d)
        return tc_mol
    
    # compute_tc('geometric', 'optimization') # geom opt
    # compute_geom_opt()
    # compute_tc("bigchem", 'hessian')
    

    def compute_tc(self, program: str, calctype: str):
        prog_input = self._prepare_input(method=calctype)
        
        future_result = self.tc_client.compute(
            program, prog_input, queue=q
        )
        output = future_result.get()
        # return output # https://github.com/coltonbh/qcio/blob/f8dd7ad3608c07b3118be791ca563493bebd997a/qcio/models/outputs.py#L152
    
        # TODO: reduce n of functions

        if output.success:
            return output.return_result
            
        else:
            output.ptraceback
            print(f"TeraChem {calctype} failed.")
            return None
        
    def tc_freq_calculation(self):
        freqs, _ = self.tc_freq_nma_calculation()
        return freqs
    
    def tc_nma_calculation(self):
        _, nmas = self.tc_freq_nma_calculation()

        nmas_flat = nmas
        nmas_reshaped = []
        for nma in nmas_flat:
            nma_arr = np.array(nma)
            nmas_reshaped.append(nma_arr.reshape(self.coords.shape))
            
        return nmas_reshaped
        
    def tc_freq_nma_calculation(self):
        if self._cached_nma is None or self._cached_freqs is None:
            prog_input = self._prepare_input(method='hessian')
            future_result = self.tc_client.compute(
                'bigchem', prog_input, queue=q
            )
            output = future_result.get()

            if output.success:
                freqs, nmas = (output.results.freqs_wavenumber, 
                               output.results.normal_modes_cartesian)
                
                self._cached_nma = nmas
                self._cached_freqs = freqs
                return freqs, nmas
            else:
                output.ptraceback
                return None, None
        else:
            return self._cached_freqs, self._cached_nma
        
    def energy_tc(self):
        return self.compute_tc(ES_PROGRAM,'energy')
        
    def gradient_tc(self):
        return self.compute_tc(ES_PROGRAM,'gradient')

    def tc_geom_optimization(self, method='minima'):
        if method == 'minima':
            opt_input = self._prepare_input(method="optimization")
        elif method == 'ts':
            opt_input = self._prepare_input(method="transition_state")
        else:
            raise ValueError(f"Unrecognized method: {method}. Use either: 'minima', or 'ts'")
            
        future_result = self.tc_client.compute(
            "geometric", opt_input, queue=q, propagate_wfn=False # this cannot be true is using psi4 for some reason...
        )
        output = future_result.get()
        result = output.results

        if output.success:
            print("Optimization succeeded!")
        else:
            print("Optimization failed!")
            output.ptraceback

        coords = result.final_molecule.geometry
        symbols = result.final_molecule.symbols
        td_opt_tc = TDStructure.from_coords_symbols(
            coords=coords * (1 / angstroms_to_bohr),
            symbols=symbols,
            tot_charge=int(result.final_molecule.charge),
            tot_spinmult=result.final_molecule.multiplicity,
        )
        
        td_opt_tc.update_tc_parameters(self)

        return td_opt_tc
    
        

    # def gradient_tc(self):
    #     atomic_input = self._prepare_input(method="grad")
    #     future_result = self.tc_client.compute(
    #         atomic_input, engine="terachem_fe", queue=q
    #     )
    #     result = future_result.get()
    #     return result.return_result
    
    # def _result_energy_tc_tcpb(self):
    #     atomic_input = self._prepare_input(method="energy")
        
    #     with TCPBClient(host='127.0.0.1', port=8888) as client:
    #         with suppress_stdout_stderr():
    #             result = client.compute(atomic_input)

    #     return result

    # def _result_gradient_tc_tcpb(self):
    #     atomic_input = self._prepare_input(method="grad")
        
    #     with TCPBClient(host='127.0.0.1', port=8888) as client:
    #         with suppress_stdout_stderr():
    #             result = client.compute(atomic_input)

    #     return result
    
    
    
    def energy_tc_tcpb(self):
        result = self._result_energy_tc_tcpb()
        return result.return_result

    def gradient_tc_tcpb(self):
        result = self._result_gradient_tc_tcpb()
        return result.return_result
    
    
    
    
    
    
    
    
    
    
    def energy_tc_local(self, **kwargs):
        return self.run_tc_local(calculation='energy',
                                method=self.tc_model_method, 
                                basis=self.tc_model_basis, **kwargs)
    
    def gradient_tc_local(self, **kwargs):
        return self.run_tc_local(calculation='gradient',
                                method=self.tc_model_method, 
                                basis=self.tc_model_basis, **kwargs)
    
    def tc_local_geom_optimization(self, **kwargs):
        return self.run_tc_local(calculation='minimize',
                                method=self.tc_model_method, 
                                basis=self.tc_model_basis, **kwargs)
        
    def tc_local_ts_optimization(self, **kwargs):
        return self.run_tc_local(calculation='ts',
                                method=self.tc_model_method, 
                                basis=self.tc_model_basis, **kwargs)
    
    
    def run_tc_local(self, 
                method,
                basis, 
                calculation='energy',
                remove_all=True,
                return_object=False):
        # make the geometry file
        with tempfile.NamedTemporaryFile(suffix='.xyz', mode="w+", delete=False) as tmp:
                    self.to_xyz(tmp.name)

        # make the tc input file
        inp = f"""run {calculation}
        coordinates {tmp.name}
        method {method}
        basis {basis}
        charge {self.charge}
        spinmult {self.spinmult}
        scrdir {tmp.name[:-4]}
        
        maxiter 500
        """
        
        if 'wf_guess' in self.tc_kwds:
            guess_path = self.tc_kwds['wf_guess'] # this must be a string
            assert isinstance(guess_path, str), f'wavefunction guess in tc_kwds needs to be a string, not a {type(guess_path)}'
            inp+=f"\nguess {guess_path}"
        
        
        with tempfile.NamedTemporaryFile(suffix='.in',mode="w+", delete=False) as tmp_inp:
                    tmp_inp.write(inp)

        # run the tc calc
        with tempfile.NamedTemporaryFile(suffix='.out',mode="w+", delete=False) as tmp_out:
            out = subprocess.run([f"terachem {tmp_inp.name}"], shell=True, 
                                capture_output=True)
            tmp_out.write(out.stdout.decode())
            
        if calculation == 'minimize':
            optim_fp = Path(tmp.name[:-4]) / 'optim.xyz'
            data_list = open(optim_fp).read().splitlines()
            result = TDStructure.from_xyz_string("\n".join(data_list[-(self.atomn+2):]),tot_charge=self.charge, tot_spinmult=self.spinmult)
            result.update_tc_parameters(self)
        else:
            result_obj = parse(tmp_out.name)  
            if return_object:
                result = result_obj
            else:
                result = result_obj.return_result
        
        
        # remove everything
        if remove_all:
            Path(tmp.name).unlink()
            Path(tmp_inp.name).unlink()
            Path(tmp_out.name).unlink()

            shutil.rmtree(tmp.name[:-4]) # delete scratch dir
        return result
        
        
    def make_geom_and_inp_file(self,
            calculation='gradient'):
        """
        writes the geometry to disk and an input file. 
        Returns the file paths to each
        """

        # make the geometry file
        with tempfile.NamedTemporaryFile(suffix='.xyz', mode="w+", delete=False) as tmp:
                    self.to_xyz(tmp.name)

        # make the tc input file
        inp = f"""run {calculation}
        coordinates {tmp.name}
        method {self.tc_model_method}
        basis {self.tc_model_basis}
        charge {self.charge}
        spinmult {self.spinmult}
        scrdir {tmp.name[:-4]}
        gpus  1
        """
        with tempfile.NamedTemporaryFile(suffix='.in',mode="w+", delete=False) as tmp_inp:
                    tmp_inp.write(inp)

        return tmp.name, tmp_inp.name
    
    
    def displace_by_dr(self, dr):
        ts_displaced = self.copy()
        ts_displaced_by_dr = ts_displaced.update_coords(ts_displaced.coords + dr)
        return ts_displaced_by_dr
    