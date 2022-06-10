from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openbabel import openbabel, pybel
from neb_dynamics import OBH
from neb_dynamics.constants import angstroms_to_bohr
from neb_dynamics.elements import ElementData


@dataclass
class TDStructure:
    """
    this is the object that will contains all the info for a single Three D structure

    this all exists within the container of a data folder, which is unique to a reaction
    """

    molecule_obmol: openbabel.OBMol

    @property
    def coords(self):
        return np.array(
            [atom.coords for atom in pybel.Molecule(self.molecule_obmol).atoms]
        )

    def update_coords(self, coords:np.array):
        for i, (x, y, z) in enumerate(coords):
            
            
            atom = self.molecule_obmol.GetAtom(i+1)
            atom.SetVector(x, y, z)
        

    @property
    def coords_bohr(self):
        return self.coords*angstroms_to_bohr

    @property
    def symbols(self):
        return np.array(
            [
                self._atomic_number_to_symbol(atom.atomicnum)
                for atom in pybel.Molecule(self.molecule_obmol).atoms
            ]
        )

    @property
    def atomic_numbers(self):
        return np.array([self._symbol_to_atomic_number(s) for s in self.symbols])

    @property
    def atom_iter(self):
        return (
            (atom.GetAtomicNum(), atom.GetFormalCharge())
            for atom in openbabel.OBMolAtomIter(self.molecule_obmol)
        )

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

    @classmethod
    def from_fp(cls, fp: Path, tot_charge=0, tot_spinmult=1):
        obmol = OBH.load_obmol_from_fp(fp)
        obmol.SetTotalCharge(tot_charge)
        obmol.SetTotalSpinMultiplicity(tot_spinmult)
        return cls(molecule_obmol=obmol)

    @classmethod
    def from_smiles(cls, smi, tot_charge=0, tot_spinmult=1):
        pybel_mol = pybel.readstring("smi", smi)
        pybel_mol.make3D()
        pybel_mol.localopt()

        obmol = pybel_mol.OBMol
        obmol.SetTotalCharge(tot_charge)
        obmol.SetTotalSpinMultiplicity(tot_spinmult)

        return cls(molecule_obmol=obmol)

    @classmethod
    def from_coords_symbs(cls, coords, symbs, tot_charge=0, tot_spinmult=1):
        obmol = OBH.obmol_from_coords(coords=coords, symbols=symbs, charge=tot_charge, spinmult=tot_spinmult)
        return cls(molecule_obmol=obmol)

    @classmethod
    def from_coords_symbs_reference(cls, coords, symbs, reference):
        obmol = OBH.obmol_from_coords_and_reference(
            coords=coords, symbols=symbs, reference=reference
        )
        return cls(obmol)

    def _atomic_number_to_symbol(self, n):
        ed = ElementData()
        return ed.from_atomic_number(n).symbol

    def _symbol_to_atomic_number(self, str):
        ed = ElementData()
        return ed.from_symbol(str).atomic_num

    def load_obmol_from_coords(self):
        """
        takes in an object that has coordinates and symbols
        and returns an OBMol object of those coordinates+symbols
        """
        coords = self.coords
        symbols = self.symbols
        obmol = openbabel.OBMol()
        OBH.obmol_from_coords(obmol, coords, symbols)
        return obmol

    def copy(self):
        obmol_copy = OBH.make_copy(self.molecule_obmol)
        return TDStructure(molecule_obmol=obmol_copy)

    def write_to_disk(self, fp):
        OBH.output_obmol_to_file(self.molecule_obmol, fp)
