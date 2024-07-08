from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import py3Dmol
import qcop
from ase import Atoms
# from ase.optimize import LBFGS, LBFGSLineSearch
from ase.optimize.sciopt import SciPyFminCG
from chemcloud import CCClient
from IPython.core.display import HTML
from openbabel import openbabel, pybel
from qcio import DualProgramInput
from qcio import Molecule as TCMolecule
from qcio import ProgramInput
from qcparse import parse
from xtb.ase.calculator import XTB
from xtb.interface import Calculator, XTBException
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.elements import ElementData
from neb_dynamics.geodesic_interpolation.coord_utils import align_geom
from neb_dynamics.helper_functions import (atomic_number_to_symbol,
                                           bond_ord_number_to_string,
                                           from_number_to_element,
                                           load_obmol_from_fp,
                                           run_tc_local_optimization,
                                           get_mass,
                                           write_xyz)
from neb_dynamics.molecule import Molecule
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


# JDEP: 05282024: Have commented out all features using tc_c0 until it stabilized.

# q = 'private'
q = None

# ES_PROGRAM = 'psi4'
ES_PROGRAM = "terachem"


@dataclass
class TDStructure:
    """
    this is the object that will contains all the info for a single Three D structure
    """

    molecule_obmol: openbabel.OBMol

    tc_model_method: str = "b3lyp"
    tc_model_basis: str = "6-31g"

    tc_kwds: dict = field(default_factory=dict)
    tc_geom_opt_kwds: dict = field(
        default_factory=lambda: {"maxiter": 300, "trust": 0.005}
    )
    # tc_c0: bytes = b""

    _cached_nma: list = None
    _cached_freqs: list = None

    @property
    def tc_d_model(self):
        return {"method": self.tc_model_method, "basis": self.tc_model_basis}

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
    def n_fragments(self) -> int:
        """computes the number of separate molecules in TDStructure

        Returns:
            int: number of molecules in TDStructure
        """

        n_frags = self.molecule_rp.separate_graph_in_pieces()
        return len(n_frags)

    @property
    def mass_weight_coords(self):
        labels = self.symbols
        coords = self.coords
        weights = np.array([np.sqrt(get_mass(s)) for s in labels])
        weights = weights / sum(weights)
        coords = coords * weights.reshape(-1, 1)
        return coords

    @property
    def coords_bohr(self) -> np.array:
        return self.coords * ANGSTROM_TO_BOHR

    def update_coords(self, coords: np.array):
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
                atomic_number_to_symbol(atom.atomicnum)
                for atom in pybel.Molecule(self.molecule_obmol).atoms
            ]
        )

    def align_to_td(self, other_td: TDStructure):
        __, new_coords = align_geom(other_td.coords, self.coords)
        new_td = self.update_coords(new_coords)
        return new_td

    def _symbol_to_atomic_number(self, s):
        d = ElementData().from_symbol(s)
        return d.atomic_num

    @property
    def atomic_numbers(self):
        return np.array([self._symbol_to_atomic_number(s) for s in self.symbols])

    def gum_mm_optimization(self):
        self.mm_optimization("gaff")
        self.mm_optimization("uff")
        self.mm_optimization("mmff94")

    @property
    def atom_iter(self):
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
            new_mol.add_edge(
                i, j, bond_order=bond_ord_number_to_string(k_prime))
        new_mol.set_neighbors()
        return new_mol

    def to_ASE_atoms(self):
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
        return atoms

    def xtb_geom_optimization(self, return_traj=False):
        from ase.io.trajectory import Trajectory as ASETraj

        from neb_dynamics.trajectory import Trajectory

        tmp = tempfile.NamedTemporaryFile(
            suffix=".traj", mode="w+", delete=False)

        atoms = self.to_ASE_atoms()
        # print(tmp.name)

        atoms.calc = XTB(method="GFN2-xTB", accuracy=0.001)
        # opt = LBFGSLineSearch(atoms, logfile=None, trajectory=tmp.name)
        opt = SciPyFminCG(atoms, logfile=None, trajectory=tmp.name)
        # opt = LBFGS(atoms, logfile=None, trajectory='/tmp/log.traj')
        # opt = FIRE(atoms, logfile=None)
        opt.run(fmax=0.01)
        # opt.run(fmax=0.5)

        aT = ASETraj(tmp.name)
        traj_list = []
        for i, _ in enumerate(aT):
            traj_list.append(
                TDStructure.from_ase_Atoms(
                    aT[i], charge=self.charge, spinmult=self.spinmult
                )
            )
        traj = Trajectory(traj_list)
        traj.update_tc_parameters(self)

        Path(tmp.name).unlink()
        if return_traj:
            print("len opt traj: ", len(traj))
            return traj
        else:
            return traj[-1]

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
        # tc_c0 = self.tc_c0

        tds = TDStructure(
            molecule_obmol=copy_obmol,
            tc_model_method=tc_model_method,
            tc_model_basis=tc_model_basis,
            tc_kwds=tc_kwds,
            tc_geom_opt_kwds=tc_geom_opt_kwds,
            # tc_c0=tc_c0,
        )

        tds._cached_freqs = self._cached_freqs
        tds._cached_nma = self._cached_nma

        return tds

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

    def to_pdb(self, fn: Path):
        mol_pybel = pybel.Molecule(self.molecule_obmol)
        mol_pybel.write(format="pdb", filename=str(fn), overwrite=True)

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
        return res.get_gradient() * BOHR_TO_ANGSTROMS

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
    def from_smiles(cls, smi, tot_spinmult=1):
        pybel_mol = pybel.readstring("smi", smi)
        pybel_mol.make3D()
        pybel_mol.localopt("gaff")
        obmol = pybel_mol.OBMol
        obmol.SetTotalSpinMultiplicity(tot_spinmult)
        obj = cls(molecule_obmol=obmol)
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
    def from_xyz(cls, fp: Path, tot_charge=0, tot_spinmult=1):
        if isinstance(fp, str):
            fp = Path(fp)
        # if "OE_LICENSE" not in os.environ:
        obmol = load_obmol_from_fp(fp)
        obmol.SetTotalCharge(tot_charge)
        obmol.SetTotalSpinMultiplicity(tot_spinmult)
        return cls(molecule_obmol=obmol)

        # else:
        #     ifs = oechem.oemolistream()
        #     ifs.SetFormat(oechem.OEFormat_XYZ)
        #     if ifs.open(str(fp.resolve())):
        #         for mol in ifs.GetOEGraphMols():
        #             return cls.from_oe(
        #                 mol, tot_charge=tot_charge, tot_spinmult=tot_spinmult
        #             )

    @classmethod
    def from_cc_result(cls, result):
        if hasattr(result, "final_molecule"):
            mol = result.final_molecule
        else:
            mol = result.molecule

        coords = mol.geometry
        symbols = mol.symbols

        td = TDStructure.from_coords_symbols(
            coords=coords * (BOHR_TO_ANGSTROMS),
            symbols=symbols,
            tot_charge=int(mol.charge),
            tot_spinmult=mol.multiplicity,
        )

        return td

    @classmethod
    def from_xyz_string(cls, string, tot_charge=0, tot_spinmult=1):
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as tmp:
            tmp.write(string)

        td = cls.from_xyz(tmp.name, tot_charge=tot_charge,
                          tot_spinmult=tot_spinmult)
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

    @property
    def tc_client(self):
        client = CCClient()
        return client

    def _prepare_input(self, method):
        allowed_methods = [
            "energy",
            "optimization",
            "gradient",
            "transition_state",
            "hessian",
        ]

        if method.lower() not in allowed_methods:
            print(f"{method} not allowed. Methods: {allowed_methods}")
            return

        # if self.tc_c0:
        #     self.tc_kwds["guess"] = "c0"

        tc_mol = self._as_tc_molecule()
        if method in ["energy", "gradient"]:
            # if self.tc_c0:
            #     prog_input = ProgramInput(
            #         calctype=method,
            #         molecule=tc_mol,
            #         model=self.tc_d_model,
            #         keywords=self.tc_kwds,
            #         files={"c0": self.tc_c0},
            #     )
            #     inp = prog_input
            # else:
            prog_input = ProgramInput(
                calctype=method,
                molecule=tc_mol,
                model=self.tc_d_model,
                keywords=self.tc_kwds,
            )
            inp = prog_input

        elif method in ["optimization", "transition_state", "hessian"]:
            # if self.tc_c0:
            #     opt_input = DualProgramInput(
            #         calctype=method,
            #         molecule=tc_mol,
            #         keywords=self.tc_geom_opt_kwds,
            #         subprogram=ES_PROGRAM,
            #         subprogram_args={
            #             "model": self.tc_d_model,
            #             "keywords": self.tc_kwds,
            #         },
            #         files={"c0": self.tc_c0},
            #     )

            # else:
            if method != "hessian":
                inp_kwds = self.tc_kwds
            else:
                inp_kwds = {}

            opt_input = DualProgramInput(
                calctype=method,
                molecule=tc_mol,
                keywords=inp_kwds,
                subprogram=ES_PROGRAM,
                subprogram_args={
                    "model": self.tc_d_model,
                    "keywords": self.tc_kwds,
                },
            )
            inp = opt_input

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

    def _tcpb_input_string(self):
        tc_inp_str = f"""method {self.tc_model_method}
        basis {self.tc_model_basis}
        spinmult {self.spinmult}
        charge {self.charge}
        """

        # if bool(self.tc_c0):
        #     with tempfile.NamedTemporaryFile(
        #         suffix=".xyz", mode="wb", delete=False
        #     ) as tmp:
        #         tmp.write(self.tc_c0)

        #     tc_inp_str += f"\nguess {str(tmp.name)}"

        kwds_strings = "\n".join(
            f"{pair[0]}  {pair[1]}\n" for pair in self.tc_kwds.items()
        )
        tc_inp_str += kwds_strings

        return tc_inp_str

    def compute_tc(self, program: str, calctype: str):
        prog_input = self._prepare_input(method=calctype)

        future_result = self.tc_client.compute(
            program, prog_input, queue=q, collect_files=True, propagate_wfn=True
        )
        output = future_result.get()

        if output.success:
            # if "scr.geometry/c0" in output.files.keys():
            #     self.tc_c0 = output.files["scr.geometry/c0"]
            return output.return_result

        else:
            output.ptraceback
            print(f"TeraChem {calctype} failed.")
            return None

    def compute_tc_local(
        self, program: str, calctype: str, return_object: bool = False
    ):
        prog_input = self._prepare_input(method=calctype)

        output = qcop.compute(
            program, prog_input, propagate_wfn=True, collect_files=True
        )

        if output.success:
            # if "scr.geometry/c0" in output.files.keys():
            #     self.tc_c0 = output.files["scr.geometry/c0"]
            if return_object:
                return output
            return output.return_result

        else:
            output.ptraceback
            print(f"TeraChem {calctype} failed.")
            if return_object:
                return output
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
            prog_input = self._prepare_input(method="hessian")
            future_result = self.tc_client.compute(
                "bigchem", prog_input, queue=q)
            output = future_result.get()

            if output.success:
                freqs, nmas = (
                    output.results.freqs_wavenumber,
                    output.results.normal_modes_cartesian,
                )

                self._cached_nma = nmas
                self._cached_freqs = freqs
                return freqs, nmas
            else:
                output.ptraceback
                return None, None
        else:
            return self._cached_freqs, self._cached_nma

    def energy_tc(self):
        return self.compute_tc(ES_PROGRAM, "energy")

    def energy_tc_local(self):
        return self.compute_tc_local(ES_PROGRAM, "energy")

    def gradient_tc(self):
        return self.compute_tc(ES_PROGRAM, "gradient")

    def gradient_tc_local(self):
        return self.compute_tc_local(ES_PROGRAM, "gradient")

    def tc_geom_optimization(self, method="minima"):
        if method == "minima":
            opt_input = self._prepare_input(method="optimization")
        elif method == "ts":
            opt_input = self._prepare_input(method="transition_state")
        else:
            raise ValueError(
                f"Unrecognized method: {method}. Use either: 'minima', or 'ts'"
            )
        pwfn_bool = ES_PROGRAM == "terachem"
        future_result = self.tc_client.compute(
            "geometric",
            opt_input,
            queue=q,
            # this cannot be true is using psi4 for some reason...
            propagate_wfn=pwfn_bool,
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
            coords=coords * (1 / ANGSTROM_TO_BOHR),
            symbols=symbols,
            tot_charge=int(result.final_molecule.charge),
            tot_spinmult=result.final_molecule.multiplicity,
        )

        td_opt_tc.update_tc_parameters(self)

        return td_opt_tc

    def tc_local_geom_optimization(self, method="minima"):
        if method == "minima":
            # opt_input = self._prepare_input(method="optimization")
            return self.run_tc_local(calculation="minimize")
        elif method == "ts":
            opt_input = self._prepare_input(method="transition_state")
        else:
            raise ValueError(
                f"Unrecognized method: {method}. Use either: 'minima', or 'ts'"
            )

        output = qcop.compute(
            "geometric",
            opt_input,
            queue=q,
            # this cannot be true is using psi4 for some reason...
            propagate_wfn=True,
        )

        result = output.results

        if output.success:
            print("Optimization succeeded!")
        else:
            print("Optimization failed!")
            output.ptraceback

        coords = result.final_molecule.geometry
        symbols = result.final_molecule.symbols
        td_opt_tc = TDStructure.from_coords_symbols(
            coords=coords * (1 / ANGSTROM_TO_BOHR),
            symbols=symbols,
            tot_charge=int(result.final_molecule.charge),
            tot_spinmult=result.final_molecule.multiplicity,
        )

        td_opt_tc.update_tc_parameters(self)

        return td_opt_tc

    def tc_local_ts_optimization(self, **kwargs):
        return self.run_tc_local(
            calculation="ts",
            method=self.tc_model_method,
            basis=self.tc_model_basis,
            **kwargs,
        )

    def run_tc_local(self, calculation="energy", remove_all=True, return_object=False):
        # make the geometry file
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w+", delete=False) as tmp:
            self.to_xyz(tmp.name)

        # make the tc input file
        inp = f"""run {calculation}\ncoordinates {tmp.name}\n"""
        inp += self._tcpb_input_string()
        inp += f"""scrdir {tmp.name[:-4]}\nmaxiter 500\n"""

        if "wf_guess" in self.tc_kwds:
            guess_path = self.tc_kwds["wf_guess"]  # this must be a string
            assert isinstance(
                guess_path, str
            ), f"wavefunction guess in tc_kwds needs to be a string, not a {type(guess_path)}"
            inp += f"\nguess {guess_path}"

        with tempfile.NamedTemporaryFile(
            suffix=".in", mode="w+", delete=False
        ) as tmp_inp:
            tmp_inp.write(inp)

        # run the tc calc
        with tempfile.NamedTemporaryFile(
            suffix=".out", mode="w+", delete=False
        ) as tmp_out:
            out = subprocess.run(
                [f"terachem {tmp_inp.name}"], shell=True, capture_output=True
            )
            tmp_out.write(out.stdout.decode())

        if calculation == "minimize":
            result = run_tc_local_optimization(
                td=self, tmp=tmp, return_optim_traj=return_object
            )

        else:
            result_obj = parse(tmp_out.name, program="terachem")
            if calculation == "energy":
                result = result_obj.energy
            elif calculation == "gradient":
                result = result_obj.gradient
            if return_object:
                result = result_obj
            else:
                result = result_obj.return_result

        # remove everything
        if remove_all:
            Path(tmp.name).unlink()
            Path(tmp_inp.name).unlink()
            Path(tmp_out.name).unlink()

            shutil.rmtree(tmp.name[:-4])  # delete scratch dir
        elif not remove_all:
            print(f"{tmp.name=} {tmp_inp.name=} {tmp_out.name=}")
        return result

    def make_geom_and_inp_file(self, calculation="gradient"):
        """
        writes the geometry to disk and an input file.
        Returns the file paths to each
        """

        # make the geometry file
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w+", delete=False) as tmp:
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
        with tempfile.NamedTemporaryFile(
            suffix=".in", mode="w+", delete=False
        ) as tmp_inp:
            tmp_inp.write(inp)

        return tmp.name, tmp_inp.name

    def displace_by_dr(self, dr):
        ts_displaced = self.copy()
        ts_displaced_by_dr = ts_displaced.update_coords(
            ts_displaced.coords + dr)
        return ts_displaced_by_dr

    def split_td_into_frags(self):
        root_mol = self.molecule_rp
        mols = root_mol.separate_graph_in_pieces()

        td_list = []
        for mol in mols:
            td = TDStructure.from_coords_symbols(
                coords=self.coords[mol.nodes],
                symbols=self.symbols[mol.nodes],
                tot_charge=mol.charge,
            )
            td.update_tc_parameters(td_ref=self)
            td_list.append(td)

        return td_list

    def _get_points_in_cavity(self, step=.5):
        """
        returns a set of points that are inside the solvent cavity formed by
        the structure. points are generated from a 3D grid with step size 'step'
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.get_xyz_lims()

        x_ = np.arange(xmin, xmax, step)
        y_ = np.arange(ymin, ymax, step)
        z_ = np.arange(zmin, zmax, step)

        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

        @np.vectorize
        def is_in_cavity(x, y, z):
            for atom in openbabel.OBMolAtomIter(self.molecule_obmol):
                vdw = openbabel.GetVdwRad(atom.GetAtomicNum())
                atom_coords = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
                dist_to_atom = np.linalg.norm(
                    np.array([x, y, z]) - atom_coords)
                if dist_to_atom <= vdw:
                    return x, y, z
            return None

        out = is_in_cavity(x, y, z).flatten()

        p_in_cav = out[out != None]  # does not work if logic is changed to 'is not None'

        arr = []
        for p in p_in_cav:
            arr.append(p)

        p_in_cav = np.array(arr)
        return p_in_cav

    def _get_vdwr_lim(td, col_ind, sign=1):
        """
        td: TDStructure
        atom_ind: the index of the atom at the corner
        col_ind: either 0, 1, or 2 correspoding to X, Y, Z. Assuming td.coords is shaped (Natom, 3)
        sign: either +1 or -1 corresponding to whether the Vanderwals radius should be added or subtracted.
                E.g. if atom is the xmin, sign should be -1.
        """
        if sign == -1:
            atom_ind = int(td.coords[:, col_ind].argmin())
        elif sign == 1:
            atom_ind = int(td.coords[:, col_ind].argmax())

        atom = td.molecule_obmol.GetAtomById(atom_ind)
        vdw_r = openbabel.GetVdwRad(atom.GetAtomicNum())
        xlim = td.coords[:, col_ind][atom_ind] + (sign*vdw_r)
        return xlim

    def get_xyz_lims(self):
        xmin = self._get_vdwr_lim(col_ind=0, sign=-1)
        xmax = self._get_vdwr_lim(col_ind=0, sign=1)

        ymax = self._get_vdwr_lim(col_ind=1, sign=1)
        ymin = self._get_vdwr_lim(col_ind=1, sign=-1)

        zmin = self._get_vdwr_lim(col_ind=2, sign=-1)
        zmax = self._get_vdwr_lim(col_ind=2, sign=1)

        return xmin, xmax, ymin, ymax, zmin, zmax

    def _get_points_in_both_cavities(self, other_td, step=1):

        xmin1, xmax1, ymin1, ymax1, zmin1, zmax1 = self.get_xyz_lims()
        xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = other_td.get_xyz_lims()

        xmin = min([xmin1, xmin2])
        ymin = min([ymin1, ymin2])
        zmin = min([zmin1, zmin2])

        xmax = max([xmax1, xmax2])
        ymax = max([ymax1, ymax2])
        zmax = max([zmax1, zmax2])

        # print(f"{xmin=}, {xmax=},{ymin=}, {ymax=}, {zmin=}, {zmax=}")
        x_ = np.arange(xmin, xmax, step)
        y_ = np.arange(ymin, ymax, step)
        z_ = np.arange(zmin, zmax, step)

        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

        @np.vectorize
        def is_in_cavity(x, y, z):
            flag1 = False
            for atom in openbabel.OBMolAtomIter(self.molecule_obmol):
                vdw = openbabel.GetVdwRad(atom.GetAtomicNum())
                atom_coords = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
                dist_to_atom = np.linalg.norm(
                    np.array([x, y, z]) - atom_coords)
                if dist_to_atom <= vdw:
                    flag1 = True

            for atom in openbabel.OBMolAtomIter(other_td.molecule_obmol):
                vdw = openbabel.GetVdwRad(atom.GetAtomicNum())
                atom_coords = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
                dist_to_atom = np.linalg.norm(
                    np.array([x, y, z]) - atom_coords)
                if dist_to_atom <= vdw:
                    if flag1:
                        return x, y, z

            return None

        out = is_in_cavity(x, y, z).flatten()
        # print(out)
        p_in_cav = out[out != None]  # does not work if written as 'is not None'

        arr = []
        for p in p_in_cav:
            arr.append(p)

        p_in_cav = np.array(arr)
        return p_in_cav

    def compute_volume(self, step=1):
        p_in_cav = self._get_points_in_cavity(step=step)
        hull = ConvexHull(p_in_cav)
        return hull.volume

    def get_optimal_volume_step(self, initial_step=1, threshold=1, shrink_factor=0.5):
        step = initial_step
        prev_volume = self.compute_volume(step=step)
        step_found = False
        while not step_found:
            step *= shrink_factor
            vol = self.compute_volume(step=step)

            delta = abs(vol - prev_volume)
            print(f"{step=} {vol=} {delta=}")

            if delta <= threshold:
                step_found = True

            prev_volume = vol

        return step

    def plot_overlap_hulls(self, other_td, step=None, just_overlap=True,
                           initial_step=1, threshold=1, shrink_factor=0.8):
        if step is None:
            step1 = self.get_optimal_volume_step(
                initial_step=initial_step,
                threshold=threshold,
                shrink_factor=shrink_factor)

            step2 = other_td.get_optimal_volume_step(
                initial_step=initial_step,
                threshold=threshold,
                shrink_factor=shrink_factor)

            step = min([step1, step2])

        xmin1, xmax1, ymin1, ymax1, zmin1, zmax1 = self.get_xyz_lims()
        xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = other_td.get_xyz_lims()

        xmin = min([xmin1, xmin2])
        ymin = min([ymin1, ymin2])
        zmin = min([zmin1, zmin2])

        xmax = max([xmax1, xmax2])
        ymax = max([ymax1, ymax2])
        zmax = max([zmax1, zmax2])

        if just_overlap:
            p_in_cav = self._get_points_in_both_cavities(other_td=other_td,
                                                         step=step)

            hull = ConvexHull(p_in_cav)

        else:
            p_in_cav1 = self._get_points_in_cavity(step=step)
            p_in_cav2 = other_td._get_points_in_cavity(step=step)

            hull1 = ConvexHull(p_in_cav1)
            hull2 = ConvexHull(p_in_cav2)

        s = 5
        fig = plt.figure(figsize=(1.6*s, s))
        ax = fig.add_subplot(111, projection='3d')

        if just_overlap:

            for simplex in hull.simplices:
                plt.plot(p_in_cav[simplex, 0], p_in_cav[simplex,
                         1], p_in_cav[simplex, 2], 'k--')
        else:
            for simplex in hull1.simplices:
                plt.plot(p_in_cav1[simplex, 0], p_in_cav1[simplex, 1],
                         p_in_cav1[simplex, 2], 'k--', color='blue')

            for simplex in hull2.simplices:
                plt.plot(p_in_cav2[simplex, 0], p_in_cav2[simplex, 1],
                         p_in_cav2[simplex, 2], 'k--', color='red')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        if just_overlap:
            plt.title(f'Volume: {hull.volume}')
        else:
            plt.title(f'Volumes: {hull1.volume}, {hull2.volume}')

        plt.show()

    def compute_overlap_volume(self, other_td, step):
        p_in_cav = self._get_points_in_both_cavities(
            other_td=other_td, step=step)
        hull = ConvexHull(p_in_cav)
        return hull.volume

    def plot_convex_hull(self, step=None, plot_grid=False, plot_hull=True,
                         initial_step=1, threshold=1, shrink_factor=0.8):
        if step is None:
            step = self.get_optimal_volume_step(
                initial_step=initial_step, threshold=threshold, shrink_factor=shrink_factor)

        xmin, xmax, ymin, ymax, zmin, zmax = self.get_xyz_lims()
        p_in_cav = self._get_points_in_cavity(step=step)
        hull = ConvexHull(p_in_cav)

        s = 5
        fig = plt.figure(figsize=(1.6*s, s))
        ax = fig.add_subplot(projection='3d')

        x_ = np.arange(xmin, xmax, step)
        y_ = np.arange(ymin, ymax, step)
        z_ = np.arange(zmin, zmax, step)

        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

        if plot_grid:
            for x, y, z in p_in_cav:
                ax.scatter3D(xs=x, ys=y, zs=z, color='gray', alpha=.3)
        # ax.scatter3D(xs=x,ys=y, zs=z, color='gray', alpha=.3)
        if plot_hull:
            for simplex in hull.simplices:
                plt.plot(p_in_cav[simplex, 0], p_in_cav[simplex,
                         1], p_in_cav[simplex, 2], 'k--')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        plt.title(f'Volume: {round(hull.volume,3)}')

        return fig
