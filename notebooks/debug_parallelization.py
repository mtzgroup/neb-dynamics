# +
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory

from neb_dynamics.NEB import NEB
from neb_dynamics.Chain import Chain
from neb_dynamics.Node3D import Node3D

import matplotlib.pyplot as plt 
# -

start_ind = 0
end_ind = 0
inx = 0
ref_geo = Trajectory.from_xyz(f"../../T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_{start_ind}-{end_ind}_{inx}.xyz")
# ref_cneb = Trajectory.from_xyz(f"../../T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_{start_ind}-{end_ind}_{inx}_cneb.xyz")

# ## repro

init_chain = Chain.from_traj(ref_geo,k=0.1,delta_k=0,step_size=2,node_class=Node3D)

tol = 4.5e-3
n = NEB(initial_chain=init_chain, grad_thre=tol, en_thre=tol/450, rms_grad_thre=tol*(2/3), climb=True, vv_force_thre=0, max_steps=10000,v=1)

# %%time
n.optimize_chain()

plt.plot(ref_geo.energies, '--',label='geo')
# plt.plot(ref_cneb.energies,'o--',label='cneb_ref')
plt.plot(n.optimized.energies, 'o-', label='cneb_parallel')
plt.legend()

ref_geo[0].energy_xtb()

n.optimized[0].tdstructure.energy_xtb()

# +

init_chain[0].energy
# -

n.optimized[0].energy
