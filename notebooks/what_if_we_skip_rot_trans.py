import matplotlib.pyplot as plt
import numpy as np
from neb_dynamics.NEB import Chain, Node3D, NEB
from neb_dynamics.geodesic_input import GeodesicInput
from pathlib import Path
# from neb_dynamics.trajectory import Trajectory
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from retropaths.abinitio.trajectory import Trajectory

roots = Trajectory.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v4/scr/root/crest_conformers.xyz")
trans = Trajectory.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v4/scr/transformed/crest_conformers.xyz")

r = roots[0]
p = trans[0]

gi = GeodesicInput.from_endpoints(initial=r, final=p)
r.coords, p.coords

traj = gi.run(nimages=15)

# +
# traj[-1].update_coords(traj[-1].coords + 5)
# -

for i, td in enumerate(traj):
    # print("B4",td.coords)
    td.update_coords(td.coords*ANGSTROM_TO_BOHR)
    # print("Post",td.coords)

c = Chain.from_traj(traj, k=0.1, delta_k=0, step_size=2, node_class=Node3D)
c.plot_chain()

tol = 0.0045
n = NEB(initial_chain=c,grad_thre=tol, en_thre=tol/450, rms_grad_thre=tol*(2/3), climb=True, vv_force_thre=0, max_steps=500)
n.optimize_chain()


