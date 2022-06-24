# +
from neb_dynamics.NEB import NEB, Chain, Node3D
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.tdstructure import TDStructure
from xtb.ase.calculator import XTB
from neb_dynamics.ALS import ArmijoLineSearch
from neb_dynamics.geodesic_input import GeodesicInput
from neb_dynamics.constants import BOHR_TO_ANGSTROMS, ANGSTROM_TO_BOHR

import numpy as np

from ase.atoms import Atoms
from ase.optimize.lbfgs import LBFGS

import matplotlib.pyplot as plt

from pathlib import Path
# -

# # 1. Optimize Endpoints with XTB

fp = Path("../example_cases/DA_geodesic_opt.xyz")
bad_traj = Trajectory.from_xyz(fp)
start_struct = Node3D(bad_traj[0])
end_struct = Node3D(bad_traj[14])


# +
def optimize_structure(node):
    coords = node.tdstructure.coords

    atoms = Atoms(
            symbols = start_struct.tdstructure.symbols.tolist(),
            positions = coords*BOHR_TO_ANGSTROMS, ### THIS P.O.S USES ANGSTRONGS INSTEAD OF BOHR
        )

    atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
    opt = LBFGS(atoms)
    opt.run(fmax=0.1)

    opt_struct = TDStructure.from_coords_symbs(
        coords=atoms.positions*ANGSTROM_TO_BOHR,
        symbs=node.tdstructure.symbols,
        tot_charge=node.tdstructure.charge,
        tot_spinmult=node.tdstructure.spinmult)
    return Node3D(opt_struct)

opt_start = optimize_structure(start_struct)
opt_end = optimize_structure(end_struct)
# -

start_struct.tdstructure.write_to_disk(Path("./bad_start.xyz"))
end_struct.tdstructure.write_to_disk(Path("./bad_end.xyz"))

opt_start.tdstructure.write_to_disk(Path("./opt_start.xyz"))
opt_end.tdstructure.write_to_disk(Path("./opt_end.xyz"))


# # Make Geodesic Interpolation between endpoints

gi = GeodesicInput.from_endpoints(initial=opt_start.tdstructure, final=opt_end.tdstructure)
opt_traj = gi.run(nimages=15, friction=0.00001)


chain = Chain.from_traj(opt_traj, k=0.1, delta_k=0.09)
plt.plot(chain.energies, 'o-')

opt_traj.write_trajectory(Path("./opt_geodesic.xyz"))

# # DO cNEB

chain = Chain.from_traj(opt_traj, k=0.1, delta_k=0.09)
n = NEB(initial_chain=chain,climb=True, mag_grad_thre=10, en_thre=0.0001, grad_thre=0.0001)
n.optimize_chain()

chain_02 = Chain.from_traj(opt_traj, k=0.2, delta_k=0.19)
n_02 = NEB(initial_chain=chain_02,climb=True, mag_grad_thre=10, en_thre=0.0001, grad_thre=0.0001)
n_02.optimize_chain()

chain_03 = Chain.from_traj(opt_traj, k=0.3, delta_k=0.29)
n_03 = NEB(initial_chain=chain_03,climb=True, mag_grad_thre=10, en_thre=0.0001, grad_thre=0.0001)
n_03.optimize_chain()

chain_1 = Chain.from_traj(opt_traj, k=1, delta_k=0.9)
n_1 = NEB(initial_chain=chain_1,climb=True, mag_grad_thre=10, en_thre=0.0001, grad_thre=0.0001)
n_1.optimize_chain()

chain_10 = Chain.from_traj(opt_traj, k=10, delta_k=9)
n_10 = NEB(initial_chain=chain_1,climb=True, mag_grad_thre=10, en_thre=0.0001, grad_thre=0.0001)
n_10.optimize_chain()

opt_chain = n.optimized
opt_chain_02 = n_02.optimized
opt_chain_03 = n_03.optimized
opt_chain_1 = n_1.optimized
opt_chain_10 = n_10.optimized

plt.plot((chain.energies- opt_chain[0].energy)*627.5, '--', label='GI')
plt.plot((opt_chain.energies- opt_chain[0].energy)*627.5,"o-", label='cNEB (k=0.1)')
plt.plot((opt_chain_02.energies- opt_chain[0].energy)*627.5,"o-", label='cNEB (k=0.2)')
plt.plot((opt_chain_03.energies- opt_chain[0].energy)*627.5,"o-", label='cNEB (k=0.3)')
plt.plot((opt_chain_1.energies- opt_chain[0].energy)*627.5,"o-", label='cNEB (k=1)')
plt.plot((opt_chain_1.energies- opt_chain[0].energy)*627.5,"o-", label='cNEB (k=10)')
plt.legend()
plt.show()

n_02.write_to_disk(Path("./cneb_DA.xyz"))


