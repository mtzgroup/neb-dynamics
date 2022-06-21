# +
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.NEB import Chain

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# +
data_dir = Path("../example_cases/")

ref_geo_traj = Trajectory.from_xyz(data_dir / "DA_geodesic_opt.xyz")

k1_traj = Trajectory.from_xyz(data_dir / "neb_DA_k1.xyz")
k09_traj = Trajectory.from_xyz(data_dir / "neb_DA_k0.9.xyz")
k07_traj = Trajectory.from_xyz(data_dir / "neb_DA_k0.7.xyz")
k05_traj = Trajectory.from_xyz(data_dir / "neb_DA_k0.5.xyz")
k03_traj = Trajectory.from_xyz(data_dir / "neb_DA_k0.3.xyz")
k01_traj = Trajectory.from_xyz(data_dir / "neb_DA_k0.1.xyz")
k0_traj = Trajectory.from_xyz(data_dir / "neb_DA_k0.xyz")
# -

ref_geo_chain = Chain.from_traj(ref_geo_traj)
ref_geo_ens = ref_geo_chain.energies

k1_chain = Chain.from_traj(k1_traj)
k1_chain_ens = k1_chain.energies

k09_chain = Chain.from_traj(k09_traj)
k09_chain_ens = k09_chain.energies

k07_chain = Chain.from_traj(k07_traj)
k07_chain_ens = k07_chain.energies

k05_chain = Chain.from_traj(k05_traj)
k05_chain_ens = k05_chain.energies

k03_chain = Chain.from_traj(k03_traj)
k03_chain_ens = k03_chain.energies

k01_chain = Chain.from_traj(k01_traj)
k01_chain_ens = k01_chain.energies

k0_chain = Chain.from_traj(k0_traj)
k0_chain_ens = k0_chain.energies

# +
s=8
fs=20

f, ax = plt.subplots(figsize=(1.618*s,s))
plt.plot((ref_geo_ens-k1_chain_ens[0])*627.5, "--", label="GI")

plt.plot((k1_chain_ens-k1_chain_ens[0])*627.5, 'o--', label='k=1.0')
plt.plot((k09_chain_ens-k1_chain_ens[0])*627.5, 'o--', label='k=0.9')
plt.plot((k07_chain_ens-k1_chain_ens[0])*627.5, 'o--', label='k=0.7')
plt.plot((k05_chain_ens-k1_chain_ens[0])*627.5, 'o--', label='k=0.5')
plt.plot((k03_chain_ens-k1_chain_ens[0])*627.5, 'o--', label='k=0.3')
plt.plot((k01_chain_ens-k1_chain_ens[0])*627.5, 'o--', label='k=0.1')
plt.plot((k0_chain_ens-k1_chain_ens[0])*627.5, 'o--', label='k=0')

plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()
# -

results = np.array([[1, 5],
           [0.9, 6],
           [0.7, 7],
           [0.5,26],
           [0.3, 84],
           [0.1, 113],
           [0, 323]         
          ]) # col1: k value; col2: nsteps it took to converge

# +
s=8
fs=20

f, ax = plt.subplots(figsize=(1.618*s,s))

plt.plot(results[:, 0], results[:, 1], 'o--', linewidth=10, markersize=25)

plt.ylabel("N steps until convergence", fontsize=fs)
plt.xlabel("K value", fontsize=fs)

plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()
# -

# # cNEB on this?

from neb_dynamics.NEB import NEB, Chain, Node3D

chain_k05 = Chain.from_traj(k0_traj, k=0.5)
chain_k01 = Chain.from_traj(k0_traj, k=0.1)
chain_k1 = Chain.from_traj(k0_traj, k=1)
chain_k03 = Chain.from_traj(k0_traj, k=0.3)
n_k05 = NEB(initial_chain=chain_k05, climb=True, mag_grad_thre=1, steps_until_climb=3)
n_k01 = NEB(initial_chain=chain_k01, climb=True, mag_grad_thre=1, steps_until_climb=3)
n_k1 = NEB(initial_chain=chain_k1, climb=True, mag_grad_thre=1, steps_until_climb=3)
n_k03 = NEB(initial_chain=chain_k03, climb=True, mag_grad_thre=1, steps_until_climb=3)

n_k05.optimize_chain()
n_k01.optimize_chain()
n_k1.optimize_chain()
n_k03.optimize_chain()

k05_cneb_ens = n_k05.optimized.energies

k01_cneb_ens = n_k01.optimized.energies

k1_cneb_ens = n_k1.optimized.energies

k03_cneb_ens = n_k03.optimized.energies

plt.plot((k05_cneb_ens-(k05_cneb_ens[0]))*627.5, 'x--', label='c-neb (k=0.5)')
plt.plot((k01_cneb_ens-(k01_cneb_ens[0]))*627.5, 'x--', label='c-neb (k=0.1)')
plt.plot((k03_cneb_ens-(k03_cneb_ens[0]))*627.5, 'x--', label='c-neb (k=0.3)')
plt.plot((k1_cneb_ens-(k1_cneb_ens[0]))*627.5, 'x--', label='c-neb (k=1.0)')
plt.plot((k01_chain_ens-k01_cneb_ens[0])*627.5, '--', label='neb (k=0.1)')
plt.legend()


