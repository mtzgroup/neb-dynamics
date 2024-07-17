# +
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from IPython.core.display import HTML
from neb_dynamics.MSMEP import MSMEP
import retropaths.helper_functions as hf
from neb_dynamics.CompetitorAnalyzer import CompetitorAnalyzer
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.Inputs import NEBInputs, GIInputs, ChainInputs
from chain import Chain
from neb_dynamics.NEB import NEB
from neb_dynamics.helper_functions import RMSD
from kneed import KneeLocator
from neb_dynamics.TreeNode import TreeNode
from retropaths.molecules.elements import ElementData
from retropaths.abinitio.tdstructure import TDStructure
import warnings
warnings.filterwarnings('ignore')
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

# +
h = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/production_results/initial_guess_msmep"))

neb = NEB.read_from_disk(Path('/home/jdep/T3D_data/msmep_draft/comparisons/nebd/Wittig/production_results/initial_guess_neb'))
# -


len(h.data.chain_trajectory)

# +
### 3D Plot
s=5
fs=18


all_chains = neb.chain_trajectory


ens = np.array([c.energies-c.energies[0] for c in all_chains])
all_integrated_path_lengths = np.array([c.integrated_path_length for c in all_chains])
opt_step = np.array(list(range(len(all_chains))))
ax = plt.figure(figsize=(2.0*s, s)).add_subplot(projection='3d')

# Plot a sin curve using the x and y axes.
x = opt_step
ys = all_integrated_path_lengths
zs = ens
for i, (xind, y) in enumerate(zip(x, ys)):
    if i == len(h.data.chain_trajectory):
        ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='red',markersize=5, linewidth=3, label="early stop chain")

    elif i < len(ys) -1:
        ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='gray',markersize=1,alpha=.1)

    else:
        ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='blue',markersize=5, label='optimized chain')
ax.grid(False)

ax.set_xlabel('optimization step',fontsize=fs)
ax.set_ylabel('normalized path length',fontsize=fs)
ax.set_zlabel('Energy (a.u.)',fontsize=fs)

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-45, roll=0)
plt.tight_layout()

plt.legend(fontsize=fs)




plt.savefig("/home/jdep/T3D_data/msmep_draft/figures/early_stop_fig_a.svg",  bbox_inches="tight")
plt.show()

# +
obj = neb
distances = obj._calculate_chain_distances()
forces = [c.get_maximum_gperp() for c in obj.chain_trajectory]
# forces = [c.get_maximum_grad_magnitude() for c in obj.chain_trajectory]

fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)


ax.plot(distances, 'o-',color='green', label='chain distances')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)


ax2 = ax.twinx()

ax2.plot(forces, 'o-',color='orange',label='max(|∇$_{\perp}$|)')
plt.yticks(fontsize=fs)
ax.set_ylabel("Distance between chains",fontsize=fs)
ax2.set_ylabel("Maximum gradient component absolute value",fontsize=fs)


ymin, ymax = ax2.get_ylim()
plt.vlines(x=len(h.data.chain_trajectory),ymin=ymin, ymax=ymax, color='red', linewidth=5, label='early stop', linestyle='--')
# ax.set_yticks(np.linspace(0, ax.get_ybound()[1], 5))
# ax2.set_yticks(np.linspace(0, ax2.get_ybound()[1], 5))


f.legend(fontsize=fs, loc='upper right')
plt.savefig("/home/jdep/T3D_data/msmep_draft/figures/early_stop_fig_b.svg",bbox_inches="tight")
plt.show()
# -


