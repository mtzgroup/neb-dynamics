# TODO: Jan, you have a big problem. TDStructures **need** to have charge and multiplicity information to be
# initialized. Otherwise you have serious energy calculation issues with TCCloud. Fix this.

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neb_dynamics import OBH
from neb_dynamics.fileio import read_xyz, write_xyz
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.constants import angstroms_to_bohr

BOHR_TO_ANGSTROMS = 1/angstroms_to_bohr
@dataclass
class Trajectory:
    """
    this is the object that will contain all the info of a trajectory from
    one structure to another

    - will probably be some special form of an array which contains TDStructures
    from which I can access energies or other properties that might be useful
    """

    traj_array: np.array
    tot_charge: int = 0
    tot_spinmult: int = 1

    def __iter__(self):
        return self.traj_array.__iter__()

    def __getitem__(self, index):
        return self.traj_array.__getitem__(index)

    def __len__(self):
        return len(self.traj_array)

    def to_list(self):
        return self.traj_array.tolist()

    @classmethod
    def from_xyz(cls, file_path, tot_charge=0, tot_spinmult=1):

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        symbols, coords = read_xyz(str(file_path.resolve()))
        return cls.from_coords_symbs(coords=np.array(coords)*angstroms_to_bohr, symbs=symbols, tot_charge=tot_charge, tot_spinmult=tot_spinmult)

    @classmethod
    def from_coords_symbs(cls, coords, symbs, tot_charge=0, tot_spinmult=1):
        traj_array = np.array([TDStructure.from_coords_symbs(i_coords, symbs, tot_charge=tot_charge, tot_spinmult=tot_spinmult) for i_coords in coords])
        return cls(traj_array, tot_charge=tot_charge, tot_spinmult=tot_spinmult)

    @classmethod
    def from_list_of_trajs(cls, list_of_trajs):
        new_traj_array = []
        for traj in list_of_trajs:
            for frame in traj.traj_array:
                new_traj_array.append(frame)

        return Trajectory(np.array(new_traj_array))

    def copy(self):
        return self.traj_array.copy()

    def insert(self, index, structure):
        self.traj_array.insert(index, structure)

    def insert_multiple_frames(self, index, list_of_structures):
        # first flip the list so we can insert them back to front
        flipped_list = np.flip(np.array(list_of_structures))
        for struct in flipped_list:
            self.traj_array.insert(index, struct)

    def append_all(self, array, start_ind=0, end_ind=None):
        """
        will append each frame in the array to the traj_array

        can also specify a subset of the array to append
        """
        if end_ind is None:
            end_ind = len(array)
        for frame in array[start_ind:end_ind]:
            self.traj_array.append(frame)

    def as_xyz(self):
        """
        returns the trajectory array as a list of xyz coordinates and
        a list of symbols for each coord
        """
        xyz_arr = []
        for molecule in self.traj_array:
            molecule_coords = OBH.obmol_to_coords(molecule.molecule_obmol)
            xyz_arr.append(molecule_coords)

        symbols = OBH.obmol_to_symbs(molecule.molecule_obmol)
        return np.array(xyz_arr), symbols

    def write_trajectory(self, file_path: Path):
        """
        writes a list of xyz coordinates to 'file_path'
        """
        xyz_arr, symbs = self.as_xyz()
        write_xyz(filename=str(file_path.resolve()), atoms=symbs, coords=xyz_arr*BOHR_TO_ANGSTROMS)

    def write_traj_energies(self, file_path: Path, energies: list):
        fout = open(str(file_path.resolve()), "w+")
        for val in energies:
            fout.write(f"{val}\n")
