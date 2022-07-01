# -*- coding: utf-8 -*-
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
opt_traj = gi.run(nimages=15, friction=0.001)


chain = Chain.from_traj(opt_traj, k=0.1, delta_k=0.09,step_size=0.37, node_class=Node3D)
plt.plot(chain.energies, 'o-')

opt_traj.write_trajectory(Path("./opt_geodesic.xyz"))

opt_traj = Trajectory.from_xyz(Path("./opt_geodesic.xyz"))

# # DO NEB

chain = Chain.from_traj(opt_traj, k=.001, delta_k=0, step_size=1, node_class=Node3D)
n = NEB(initial_chain=chain,climb=False,grad_thre=0.02, vv_force_thre=0.0)
n.optimize_chain()

chain01 = Chain.from_traj(opt_traj, k=.01, delta_k=0, step_size=1, node_class=Node3D)
n_01 = NEB(initial_chain=chain01,climb=False,grad_thre=0.02, vv_force_thre=0.0)
n_01.optimize_chain()

chain1 = Chain.from_traj(opt_traj, k=.1, delta_k=0, step_size=1, node_class=Node3D)
n_1 = NEB(initial_chain=chain1,climb=False,grad_thre=0.02, vv_force_thre=0.0)
n_1.optimize_chain()

chain10 = Chain.from_traj(opt_traj, k=1.0, delta_k=0, step_size=1, node_class=Node3D)
n_10 = NEB(initial_chain=chain10,climb=False,grad_thre=0.02, vv_force_thre=0.0)
n_10.optimize_chain()

chain_09 = Chain.from_traj(opt_traj, k=0.0009, delta_k=0, step_size=.01)
n_09 = NEB(initial_chain=chain_09,climb=False,grad_thre=0.02, vv_force_thre=0.5)
n_09.optimize_chain()

chain_07 = Chain.from_traj(opt_traj, k=0.0007, delta_k=0, step_size=.01)
n_07 = NEB(initial_chain=chain_07,climb=False,grad_thre=0.02, vv_force_thre=0.5)
n_07.optimize_chain()

chain_05 = Chain.from_traj(opt_traj, k=0.0005, delta_k=0, step_size=.01)
n_05 = NEB(initial_chain=chain_05,climb=False,grad_thre=0.02, vv_force_thre=0.5)
n_05.optimize_chain()

chain_03 = Chain.from_traj(opt_traj, k=0.0003, delta_k=0, step_size=.01)
n_03 = NEB(initial_chain=chain_03,climb=False,grad_thre=0.02, vv_force_thre=0.5)
n_03.optimize_chain()

opt_chain = n.optimized
opt_chain_01 = n_01.optimized
opt_chain_1 = n_1.optimized
opt_chain_10 = n_10.optimized


n.write_to_disk(Path("../example_cases/neb_DA_k0.001.xyz"))
n_01.write_to_disk(Path("../example_cases/neb_DA_k0.01.xyz"))
n_1.write_to_disk(Path("../example_cases/neb_DA_k0.1.xyz"))
n_10.write_to_disk(Path("../example_cases/neb_DA_k1.xyz"))

# +
opt_chain_09 = n_09.optimized

opt_chain_07 = n_07.optimized

opt_chain_05 = n_05.optimized

opt_chain_03 = n_03.optimized


# +
plt.plot((chain.energies-opt_chain[0].energy)*627.5, '--', label='GI')
plt.plot((opt_chain.energies- opt_chain[0].energy)*627.5,"o-", label='NEB (k=0.001)')
plt.plot((opt_chain_01.energies- opt_chain_01[0].energy)*627.5,"o-", label='NEB (k=0.01)')
plt.plot((opt_chain_1.energies- opt_chain_1[0].energy)*627.5,"o-", label='NEB (k=0.1)')
plt.plot((opt_chain_10.energies- opt_chain_10[0].energy)*627.5,"o-", label='NEB (k=1.0)')

# plt.plot((opt_chain_03.energies- opt_chain[0].energy)*627.5,"o-", label='NEB (k=0.003)')
# plt.plot((opt_chain_05.energies- opt_chain[0].energy)*627.5,"o-", label='NEB (k=0.005)')
# plt.plot((opt_chain_07.energies- opt_chain[0].energy)*627.5,"o-", label='NEB (k=0.007)')
# plt.plot((opt_chain_09.energies- opt_chain[0].energy)*627.5,"o-", label='NEB (k=0.009)')

plt.legend()
plt.show()

# +
# n_02.write_to_disk(Path("./cneb_DA.xyz"))
# -
# # Do cNEB


chain = Chain.from_traj(opt_traj, k=1, delta_k=0, step_size=1, node_class=Node3D)
n = NEB(initial_chain=chain,climb=True,grad_thre=0.02, vv_force_thre=0.0)
n.optimize_chain()

neb_k10_traj = Trajectory.from_xyz(Path("../example_cases/neb_DA_k1.xyz"))
neb_k10_chain = Chain.from_traj(neb_k1_traj, k=1, delta_k=0, step_size=0.01, node_class=Node3D)

neb_k1_traj = Trajectory.from_xyz(Path("../example_cases/neb_DA_k0.1.xyz"))
neb_k1_chain = Chain.from_traj(neb_k1_traj, k=1, delta_k=0, step_size=0.01, node_class=Node3D)

neb_k01_traj = Trajectory.from_xyz(Path("../example_cases/neb_DA_k0.01.xyz"))
neb_k01_chain = Chain.from_traj(neb_k01_traj, k=1, delta_k=0, step_size=0.01, node_class=Node3D)

neb_k001_traj = Trajectory.from_xyz(Path("../example_cases/neb_DA_k0.001.xyz"))
neb_k001_chain = Chain.from_traj(neb_k001_traj, k=1, delta_k=0, step_size=0.01, node_class=Node3D)

# +
s=10
f,ax = plt.subplots(figsize=(1.16*s, s))
plt.plot((chain.energies- chain[0].energy)*627.5,"o-", label='GI')
plt.plot((neb_k10_chain.energies- chain[0].energy)*627.5,"o-", label='NEB (k=1)')
# plt.plot((neb_k1_chain.energies- chain[0].energy)*627.5,"o-", label='NEB (k=0.1)')
# plt.plot((neb_k01_chain.energies- chain[0].energy)*627.5,"o-", label='NEB (k=0.01)')
# plt.plot((neb_k001_chain.energies- chain[0].energy)*627.5,"o-", label='NEB (k=0.001)')

plt.plot((n.optimized.energies- chain[0].energy)*627.5,"o-", label='cNEB (k=1)')

# plt.plot((opt_chain_10.energies- opt_chain_10[0].energy)*627.5,"o-", label='NEB (k=1.0)')

# plt.plot((opt_chain_03.energies- opt_chain[0].energy)*627.5,"o-", label='NEB (k=0.003)')
# plt.plot((opt_chain_05.energies- opt_chain[0].energy)*627.5,"o-", label='NEB (k=0.005)')
# plt.plot((opt_chain_07.energies- opt_chain[0].energy)*627.5,"o-", label='NEB (k=0.007)')
# plt.plot((opt_chain_09.energies- opt_chain[0].energy)*627.5,"o-", label='NEB (k=0.009)')

plt.legend()
plt.show()
# -

n.write_to_disk(Path("../example_cases/cneb_DA_k1.xyz"))

# # Claisen?

orig = Trajectory.from_xyz("../example_cases/debug_geodesic_claisen.xyz")
orig_chain = Chain.from_traj(orig, k=1, delta_k=0, step_size=1, node_class=Node3D)

plt.plot(orig_chain.energies, '--')

n_CR = NEB(initial_chain=orig_chain, climb=True, grad_thre_per_atom=0.0016, vv_force_thre=0)
n_CR.optimize_chain()

n_CR.write_to_disk(Path("../example_cases/cneb_claisen_k1_gt_0008.xyz"))

cr_neb_k02_traj = Trajectory.from_xyz(Path("../example_cases/cneb_claisen_k0.2.xyz"))
cr_neb_k02_chain = Chain.from_traj(cr_neb_k02_traj, k=1, delta_k=0, step_size=1, node_class=Node3D)

cr_neb_k001_traj = Trajectory.from_xyz(Path("../example_cases/cneb_claisen_k01_gt_0016.xyz"))
cr_neb_k001_chain = Chain.from_traj(cr_neb_k001_traj, k=1, delta_k=0, step_size=1, node_class=Node3D)

cr_neb_k01_traj = Trajectory.from_xyz(Path("../example_cases/cneb_claisen_k0.1_gt_0016.xyz"))
cr_neb_k01_chain = Chain.from_traj(cr_neb_k01_traj, k=1, delta_k=0, step_size=1, node_class=Node3D)

cr_neb_k1_traj = Trajectory.from_xyz(Path("../example_cases/cneb_claisen_k1_gt_0016.xyz"))
cr_neb_k1_chain = Chain.from_traj(cr_neb_k1_traj, k=1, delta_k=0, step_size=1, node_class=Node3D)

cr_neb_k1_d0_traj = Trajectory.from_xyz(Path("../example_cases/cneb_claisen_k1_xyz"))
cr_neb_k1_d0_chain = Chain.from_traj(cr_neb_k1_d0_traj, k=1, delta_k=0, step_size=1, node_class=Node3D)

cr_neb_k1_vv2_traj = Trajectory.from_xyz(Path("../example_cases/cneb_claisen_k1_vv0.2.xyz"))
cr_neb_k1_vv2_chain = Chain.from_traj(cr_neb_k1_vv2_traj, k=1, delta_k=0, step_size=1, node_class=Node3D)



# +
s=8
fs=20
f,ax = plt.subplots(figsize=(1.618*s,s))

plt.plot((orig_chain.energies-orig_chain[0].energy)*627.5, '--', label='GI')
plt.plot((cr_neb_k001_chain.energies-orig_chain[0].energy)*627.5, 'o-', label='cNEB (k=0.01)')
plt.plot((cr_neb_k01_chain.energies-orig_chain[0].energy)*627.5, 'o-', label='cNEB (k=0.1)')
plt.plot((cr_neb_k1_chain.energies-orig_chain[0].energy)*627.5, 'o-', label='cNEB (k=1)')


# plt.plot((cr_neb_k02_chain.energies-orig_chain[0].energy)*627.5, 'o-', label='cNEB (k=0.2)')

# plt.plot((cr_neb_k1_d0_chain.energies-orig_chain[0].energy)*627.5, 'o-', label='cNEB (k=1)(∆0.9)')
# plt.plot((cr_neb_k1_vv2_chain.energies-orig_chain[0].energy)*627.5, 'o-', label='cNEB (k=1)(vv=0.2)')
plt.legend(fontsize=fs)

plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

plt.show()
# -

# ### output the TS-approximations

cr_neb_k001_chain[3].tdstructure.write_to_disk("ts_cr_neb_k001_chain.xyz")
cr_neb_k01_chain[4].tdstructure.write_to_disk("ts_cr_neb_k01_chain.xyz")
cr_neb_k02_chain[3].tdstructure.write_to_disk("ts_cr_neb_k02_chain.xyz")
cr_neb_k1_chain[3].tdstructure.write_to_disk("ts_cr_neb_k1_chain.xyz")
cr_neb_k1_d0_chain[3].tdstructure.write_to_disk("ts_cr_neb_k1_d0_chain.xyz")
cr_neb_k1_vv2_chain[3].tdstructure.write_to_disk("ts_cr_neb_k1_vv2_chain.xyz")

(cr_neb_k001_chain[3].energy-cr_neb_k001_chain[0].energy)*627.5

(cr_neb_k1_vv2_chain[3].energy-cr_neb_k1_vv2_chain[0].energy)*627.5

(cr_neb_k1_chain[3].energy-cr_neb_k1_chain[0].energy)*627.5

n_CR.write_to_disk(Path("./cneb_claisen_k1.xyz"))

(n_CR.optimized.energies[3] - n_CR.optimized.energies[0])*627.5

# n_interations, k value, delta_k value, vv_force_thre
results = [
    [232, 1, 0, 0],
    [208, 1, 0.9, 0],
    [422, 1, 0, 0.2]
]


