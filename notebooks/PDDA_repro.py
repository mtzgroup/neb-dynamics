# -*- coding: utf-8 -*-
ref_old = Trajectory.from_xyz(Path("/Users/janestrada/neb_2/neb_dynamics/neb_dynamics/reference_DA_neb.xyz"))
ref_new = Trajectory.from_xyz(Path("/Users/janestrada/neb_dynamics/example_cases/neb_converted_pre_redistr.xyz"))

chain_old = Chain.from_traj(ref_old)

chain_new = Chain.from_traj(ref_new)

plt.plot(chain_old.energies, 'o--', label='old')
plt.plot(chain_new.energies, 'x--', label='new')
plt.legend()

chain_old.energies 

chain_new.energies

# +
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from retropaths.abinitio.geodesic_input import GeodesicInput
from retropaths.abinitio.inputs import Inputs
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.NEB import NEB, Node3D, Chain

out_dir = Path("/Users/janestrada/neb_dynamics/example_cases")

# -

# load xtb-minimized structures
root = TDStructure.from_fp(out_dir/"root_opt.xyz")
end = TDStructure.from_fp(out_dir/"end_opt.xyz")

# ### do geodesic interpolation
gi = GeodesicInput.from_endpoints(initial=root, final=end)
traj = gi.run(nimages=31, friction=0.01, nudge=0.01)

traj.write_trajectory(out_dir/"pdda_traj_xtb_optmized_geo_att2.xyz")
# traj = Trajectory.from_xyz(out_dir/"pdda_traj_xtb_optmized_geo_att2.xyz")

nodes = [Node3D(s) for s in traj]
chain = Chain(nodes, k=10)
ens = [Node3D.en_func(node) for node in chain]

plt.plot(ens)

# +
n = NEB(initial_chain=chain, mag_grad_thre=1)

opt_chain = n.optimize_chain()
# -

opt_chain_energies = [n.en_func(s) for s in opt_chain[0]]

plt.title(f"{rn}")
plt.plot(ens, label="geodesic")
plt.scatter(list(range(len(opt_chain_energies))), opt_chain_energies, label="neb", color="orange")
plt.legend()

traj.write_trajectory(out_dir / f"{rn}_geodesic_opt.xyz")

opt_traj = Trajectory(opt_chain[0])

opt_traj.write_trajectory(out_dir / f"{rn}_neb_opt.xyz")

# ## Result

# +
geo_ref = Trajectory.from_xyz(Path("../example_cases/pdda_traj_xtb_optmized_geo_att2.xyz"))

traj_k05 = Trajectory.from_xyz(Path("../example_cases/neb_PDDA_k0.5.xyz"))
traj_k03 = Trajectory.from_xyz(Path("../example_cases/neb_PDDA_k0.3.xyz"))
traj_k01 = Trajectory.from_xyz(Path("../example_cases/neb_PDDA_k0.1.xyz"))
traj_k001 = Trajectory.from_xyz(Path("../example_cases/neb_PDDA_k0.01.xyz"))
# -

geo_chain = Chain.from_traj(geo_ref)
geo_ens = geo_chain.energies

chain_k05 = Chain.from_traj(traj_k05)
chain_k05_ens = chain_k05.energies

chain_k03 = Chain.from_traj(traj_k03)
chain_k03_ens = chain_k03.energies

chain_k01 = Chain.from_traj(traj_k01)
chain_k01_ens = chain_k01.energies

chain_k001 = Chain.from_traj(traj_k001)
chain_k001_ens = chain_k001.energies

plt.plot(geo_ens, '--', label='GI')
plt.plot(chain_k05_ens, 'o-', label='k=0.5')
plt.plot(chain_k03_ens, 'o-', label='k=0.3')
plt.plot(chain_k01_ens, 'o-', label='k=0.1')
plt.plot(chain_k001_ens, 'o-', label='k=0.01')
plt.legend()


