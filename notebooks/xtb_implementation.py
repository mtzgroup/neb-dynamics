# +
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from retropaths.abinitio.geodesic_input import GeodesicInput
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory


from neb_dynamics.NEB import NEB, Chain, Node3D

# +

ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROMS = 1 / ANGSTROM_TO_BOHR
# -

# # DA attempt

data_dir = Path("../example_cases/")
traj = Trajectory.from_xyz(data_dir/"DA_geodesic_opt.xyz")

coords = list(traj)
chain = Chain.from_list_of_coords(k=10, list_of_coords=coords, node_class=Node3D)

n = NEB(initial_chain=chain, grad_thre=0.0001, mag_grad_thre=0.1)
# n.optimize_chain()

end = n.update_chain(chain)
end2 = n.update_chain(end)

end3 = n.update_chain(end2)

foo = chain.coordinates - chain.gradients*chain.displacements

np.mean([np.linalg.norm(grad) for grad in chain.gradients])

i=10
foo_struct = TDStructure.from_coords_symbs(coords=end3[i].coords, symbs=end3[i].tdstructure.symbols)
foo_struct.write_to_disk(data_dir/"WTFFFFF.xyz")

chain.displacements

original_chain_ens = [Node3D.en_func(node) for node in chain]
maybe_ens = [node.energy for node in end3]
plt.plot(original_chain_ens)
plt.plot(maybe_ens, "o")



# # PDDA repro

data_dir = Path("../example_cases/")
start = TDStructure.from_fp(data_dir / "root_opt.xyz")

end = TDStructure.from_fp(data_dir / "end_opt.xyz")

gi = GeodesicInput.from_endpoints(initial=start, final=end)
traj = gi.run(nimages=31, friction=0.01)

coords = list(traj)
chain = Chain.from_list_of_coords(k=10, list_of_coords=coords, node_class=Node3D)
n = NEB(initial_chain=chain, grad_thre=0.0001, mag_grad_thre=0.1)

# +

original_chain_ens = [Node3D.en_func(node) for node in chain]
# -

plt.plot(original_chain_ens)

n.optimize_chain()

opt_ens = [Node3D.en_func(node) for node in n.chain_trajectory[-1]]

plt.plot(list(range(len(opt_ens))), opt_ens, "o--")

foo = Trajectory([x.tdstructure for x in n.optimized.nodes])
foo.write_trajectory(Path("./wtf.xyz"))

# +
foo1 = np.array([[1, 2, 3]])
foo2 = np.array([[3, 2, 1]])

np.tensordot(foo1, foo2)
# -

# #### start time: 1045am
# #### end time: <=11:22am

opt_chain, opt_chain_traj = n.optimize_chain(chain=traj, grad_func=n.grad_func, en_func=n.en_func, k=10, max_steps=1000)

# +
# start_pdda, end_pdda = opt_chain[0], opt_chain[-1]

# +
# start_pdda.write_to_disk(data_dir/"root_pdda.xyz")

# +
# end_pdda.write_to_disk(data_dir/'end_pdda.xyz')
# -

opt_chain_ens = [n.en_func(s) for s in opt_chain]

ref_traj = Trajectory.from_xyz(data_dir / "ref_neb_pdda.xyz")
ref_ens = [n.en_func(s) for s in ref_traj]

plt.scatter(list(range(len(ref_ens))), ref_ens)

original_chain_ens_scaled = 627.5 * np.array(original_chain_ens)
original_chain_ens_scaled -= original_chain_ens_scaled[0]

opt_chain_ens_scaled = 627.5 * np.array(opt_chain_ens)
opt_chain_ens_scaled -= opt_chain_ens_scaled[0]

s = 8
fs = 22
f, ax = plt.subplots(figsize=(1.618 * s, s))
plt.plot(list(range(len(original_chain_ens))), original_chain_ens_scaled, "x--", label="orig")
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.plot(list(range(len(original_chain_ens))), opt_chain_ens_scaled, "o--", label="neb")
plt.ylabel("Relative energy (kcal/mol)", fontsize=fs + 5)
# plt.plot(list(range(len(ref_ens))), ref_ens, 'o--',label='neb reference')
plt.legend(fontsize=fs)

opt_chain_ens_scaled[7]

opt_chain_ens_scaled[21]

opt_chain_ens_scaled[25]

out = Trajectory(opt_chain)


out.write_trajectory(data_dir / "pdda_neb_1000_steps_k_10_corrected.xyz")

# # Figure

n = neb()

Trajectory.from_xyz(data_dir / "pdda_neb_1000_steps_k_10.xyz")
