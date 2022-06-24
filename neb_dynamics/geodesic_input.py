from dataclasses import dataclass
from functools import cached_property

from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.elements import ElementData
from openbabel import pybel
from neb_dynamics.geodesic_interpolation.geodesic import run_geodesic_py
import numpy as np


@dataclass
class GeodesicInput:
    trajectory: Trajectory


    @classmethod
    def from_endpoints(cls, initial, final):
        traj = Trajectory([initial, final], tot_charge=initial.charge, tot_spinmult=initial.spinmult)
        return cls(trajectory=traj)

    @property
    def coords(self):
        computed_coords = np.array(
            [struct.coords for struct in self.trajectory]
        )
        if np.array_equal(computed_coords[0], computed_coords[1]):
            raise ValueError("Input coordinates are identical")
        return computed_coords

    @property
    def charge(self):
        return self.trajectory[0].charge
    
    @property
    def spinmult(self):
        return self.trajectory[0].spinmult

    @property
    def symbols(self):
        return self.trajectory[0].symbols
        
    def run(self, **kwargs):
        xyz_coords = run_geodesic_py(self, **kwargs)
        return Trajectory.from_coords_symbs(coords=xyz_coords, symbs=self.symbols, tot_charge=self.charge, tot_spinmult=self.spinmult)

