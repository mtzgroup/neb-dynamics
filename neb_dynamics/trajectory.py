import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import HTML
from ipywidgets import IntSlider, interact
from openeye import oechem

from retropaths.abinitio import OBH

from neb_dynamics.geodesic_interpolation.fileio import read_xyz
from neb_dynamics.geodesic_interpolation.geodesic import run_geodesic_py
from neb_dynamics.tdstructure import TDStructure
from chemcloud import CCClient

# oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet) # oechem VERBOSITY controller
ES_PROGRAM = 'psi4'
# ES_PROGRAM = 'terachem'

HARTREE_TO_KCALM = 627.5

@dataclass
class Trajectory:
    """
    this is the object that will contain all the info of a trajectory from
    one structure to another

    - will probably be some special form of an array which contains TDStructures
    from which I can access energies or other properties that might be useful
    """

    traj: list[TDStructure]

    @property
    def charge(self):
        if self.traj:
            return self.traj[0].charge
        else:
            print("Trajectory is empty! Returning 0 for charge")
            return 0
    
    @property
    def spinmult(self):
        if self.traj:
            return self.traj[0].spinmult
        else:
            print("Trajectory is empty! Returning 1 for spinmult")
            return 1


    def __iter__(self):
        return self.traj.__iter__()

    def __getitem__(self, index):
        return self.traj.__getitem__(index)

    def __len__(self):
        return len(self.traj)

    @property
    def energies(self):
        return [t.energy_xtb() for t in self.traj]

    @classmethod
    def extend(cls, first, other):
        return cls(
            traj=np.concatenate((first.traj, other.traj)),
            tot_charge=first.tot_charge,
            tot_spinmult=first.tot_spinmult,
        )

    @classmethod
    def from_coords_symbols(cls, coords, symbs, tot_charge=0, tot_spinmult=1):
        traj = np.array(
            [
                TDStructure.from_coords_symbols(
                    i_coords, symbs, tot_charge=tot_charge, tot_spinmult=tot_spinmult
                )
                for i_coords in coords
            ]
        )
        return cls(traj)

    @classmethod
    def from_list_of_trajs(cls, list_of_trajs):
        new_traj = []
        for traj in list_of_trajs:
            for frame in traj.traj:
                new_traj.append(frame)

        return Trajectory(np.array(new_traj))
    
    @classmethod
    def from_list_of_cc_results(cls, cc_results):
        tds = [TDStructure.from_cc_result(r) for r in cc_results]
        traj = cls(tds)
        return traj

    def copy(self):
        list_of_trajs = [td.copy() for td in self.traj]
        return Trajectory(
            traj=list_of_trajs
        )

    def insert(self, index, structure):
        self.traj.insert(index, structure)

    def append_all(self, array, start_ind=0, end_ind=None):
        """
        will append each frame in the array to the traj

        can also specify a subset of the array to append
        """
        if end_ind is None:
            end_ind = len(array)
        for frame in array[start_ind:end_ind]:
            self.traj.append(frame)

    def as_xyz(self):
        """
        returns the trajectory array as a list of xyz coordinates and
        a list of symbols for each coord
        """
        xyz_arr = []
        for molecule in self.traj:
            molecule_coords = OBH.obmol_to_coords(molecule.molecule_obmol)
            xyz_arr.append(molecule_coords)

        symbols = OBH.obmol_to_symbs(molecule.molecule_obmol)
        return xyz_arr, symbols

    def write_trajectory(self, file_path: Path):
        """
        writes a list of xyz coordinates to 'file_path'
        """
        from neb_dynamics.geodesic_interpolation.fileio import write_xyz

        xyz_arr, symbs = self.as_xyz()
        write_xyz(filename=file_path, atoms=symbs, coords=xyz_arr)

    def set_implicit_solvent(self,model='cosmo',epsilon='80'):
        for td in self.traj:
            td.tc_kwds['pcm'] = model
            td.tc_kwds["epsilon"] = epsilon
        
    @property
    def coords(self) -> np.array:
        return np.array([x.coords for x in self])

    @property
    def symbols(self) -> np.array:
        assert len(self) > 0, "This is empty"
        return self[0].symbols

    @property
    def xyz(self) -> str:
        return "".join(x.xyz for x in self)

    def draw(self, energy=False):
        frames = len(self)

        if energy:
            ene = self.draw_energies()
        else:
            ene = ""

        def wrap(frame):
            return self[frame].view_mol(
                string_mode=False, style="sphere", custom_image=ene
            )

        return interact(
            wrap,
            frame=IntSlider(
                min=0, max=frames - 1, step=1, description="Trajectory frames"
            ),
        )

    def _repr_html_(self):
        return self.draw()

    def draw_energies(self, string_mode=True):
        f = io.BytesIO()
        fig = plt.figure()
        _ = plt.plot(self.energies_xtb())
        plt.savefig(f, format="svg")
        plt.close()
        stringa = f.getvalue()
        if string_mode:
            return stringa.decode()
        else:
            return HTML(stringa.decode())

    @property
    def smiles(self):
        """
        returns non redundant smiles
        """
        return {x.molecule_rp.smiles for x in self}

        
        
    def update_tc_parameters(self, reference: TDStructure):
        for td in self:
            td.update_tc_parameters(reference)

    def run_geodesic(self, **kwargs):
        new_traj = self.copy()
        xyz_coords = run_geodesic_py(self, **kwargs)
        tds = [TDStructure.from_coords_symbols(x, self.symbols, tot_charge=self.charge, tot_spinmult=self.spinmult) for x in xyz_coords]
        
        # update terachem params
        for t in tds:
            t.update_tc_parameters(self[0])
            
        new_traj.traj = tds
        
        
        return new_traj

    def get_conv_gi(
        self, gi_guess, prev_eA=None, conv_thre=1, history=[]
    ):  # conv_thre is in kcal/mol

        gi = gi_guess.run_geodesic(nimages=15)  # the new gi
        history.append(gi)

        if not prev_eA:  # gi is the root gi then.
            prev_eA = max(gi.energies_xtb())
            return self.get_conv_gi(gi_guess=gi, prev_eA=prev_eA, history=history)

        # otherwise we have a gi to compare to
        new_eA = max(gi.energies_xtb())

        print(prev_eA, new_eA)
        if np.abs(new_eA - prev_eA) <= conv_thre:
            print(f"converged in {len(history)-1} iterations")
            return gi, history

        else:
            return self.get_conv_gi(gi_guess=gi, prev_eA=new_eA, history=history)

    def energies_xtb(self):
        array = np.array([x.energy_xtb() for x in self])
        array = array[array != None]
        return (array - array[0]) * HARTREE_TO_KCALM
    
    def energies_tc(self):
        inputs = [td._prepare_input(method='grad') for td in self]
        future_result = self.traj[0].tc_client.compute(inputs, engine="terachem_fe")
        results = future_result.get()
        for r in results:
            if not r.success:
                print(r.error.error_message)
                return None
        array = np.array([result.properties.return_energy for result in results])
        return (array - array[0]) * HARTREE_TO_KCALM
    
    
    def tc_geom_opt_all_geometries(self):
        prog_inp = [td._prepare_input(method='opt') for td in self]
        future_results = self.traj[0].tc_client.compute('geometric',prog_inp)
        outputs = [op.get() for op in future_results]
        results = [TDStructure.from_cc_result(out) for out in outputs if out.success]
        return Trajectory(results)

    def energies_and_gradients_tc(self):
        client = CCClient()
        prog_input = [td._prepare_input(method='gradient') for td in self.traj]
        
        future_result = client.compute(
            ES_PROGRAM, prog_input
        )
        output_list = future_result.get()
            
        for res in output_list:
            if not res.success:
                res.ptraceback
                print(f"{ES_PROGRAM} failed.")
                return None
            
        
        grads = np.array([output.return_result for output in output_list])
        ens = np.array([output.results.energy for output in output_list])
        return ens, grads
    
    

    @classmethod
    def from_xyz_string(cls, string):
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as tmp:
            tmp.write(string)
        return cls.from_xyz(Path(tmp.name))

    @classmethod
    def from_xyz(cls, fn: Path, tot_charge=0, tot_spinmult=1):
        tdss = []
        
        if 'OE_LICENSE' not in os.environ:
            file_path = fn
            symbols, coords = read_xyz(str(file_path.resolve()))
            return cls.from_coords_symbols(coords=np.array(coords), symbs=symbols, tot_charge=tot_charge, tot_spinmult=tot_spinmult)
        else:
            ifs = oechem.oemolistream()
            ifs.SetFormat(oechem.OEFormat_XYZ)
            if ifs.open(str(fn)):
                for mol in ifs.GetOEGraphMols():
                    tds = TDStructure.from_oe(
                        mol, tot_charge=tot_charge, tot_spinmult=tot_spinmult
                    )
                    tdss.append(tds)
        return cls(tdss)
