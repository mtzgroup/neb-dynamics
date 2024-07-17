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
# -

""
long_fixed_gi = neb_short.initial_chain.to_trajectory().run_geodesic(nimages=45)

""
# rn = 'Lobry-de-Bruyn-Van-Ekenstein-Transformation'
rn = 'Robinson-Gabriel-Synthesis'

asneb = TreeNode.read_from_disk(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/asneb/{rn}/production_results/initial_guess_msmep/"))
# asneb = TreeNode.read_from_disk(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/asneb/{rn}/initial_guess_msmep/"))

# asneb = TreeNode.read_from_disk(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/asneb/{rn}/initial_guess_fixed_msmep/"))

""
# neb_short = NEB.read_from_disk(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/nebd/{rn}/initial_guess_neb"))
neb_short = NEB.read_from_disk(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/nebd/{rn}/initial_guess_fixed_neb"))

""
neb_long = NEB.read_from_disk(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/nebd/{rn}/initial_guess_long_neb"))

""
h = asneb

""
# root_chain = h.data.optimized
root_chain = neb_short.optimized
child1 = h.children[0].data.optimized
# leaf1 = h.children[0].children[0].data.optimized
leaf1 = h.children[0].data.optimized
leaf2 = h.children[1].data.optimized

out = asneb.output_chain

""
# +
s=5
fs=18
f, ax = plt.subplots(figsize=(2.16*s, s))

ms = 7
lw = 2
hexvals=[
    '#ff1b1b',
'#dd0090',
'#5855c3']

root_pathlen = root_chain.integrated_path_length
plt.plot(root_pathlen, (root_chain.energies-root_chain.energies[0])*627.5,'o--', label='NEB(15 nodes)'
         ,markersize=ms,linewidth=lw, color=hexvals[0])
plt.plot(root_pathlen, (root_chain.energies-root_chain.energies[0])*627.5,'-'
         ,markersize=ms,linewidth=lw+5, color=hexvals[0], alpha=.3)



# neb_long_pathlen = neb_long.optimized.integrated_path_length
# long_chain = neb_long.optimized
# plt.plot(neb_long_pathlen, (long_chain.energies-root_chain.energies[0])*627.5-0,'*-',label="NEB (30 nodes)"
#         ,markersize=ms+5,linewidth=lw, color='orange')


out_pathlen = out.integrated_path_length
plt.plot(out.integrated_path_length, (out.energies-root_chain.energies[0])*627.5-0,'o-', label="AS-NEB"
        ,markersize=ms,linewidth=lw, color=hexvals[1])


plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.xlabel("Normalized path length",fontsize=fs)

plt.legend(fontsize=fs)


plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/fixed_{rn}_energy_paths.svg")


plt.show()
# +

""
n_steps_msmep = sum([len(obj.data.chain_trajectory) -1 for obj in h.depth_first_ordered_nodes])
n_steps_long_neb = len(neb_long.chain_trajectory) -1
n_steps_short_neb = len(neb_short.chain_trajectory) -1

n_grad_msmep = n_steps_msmep*(15-2)
n_grad_long_neb = n_steps_long_neb*(30-2)
n_grad_short_neb = n_steps_short_neb*(15-2)

""
fig = 7
min_val = -5.3
max_val = 5.3
fs = 18
f,ax = plt.subplots(figsize=(1.16*fig,fig))

bars = ax.bar(x=[f'NEB({len(neb_short.optimized)} nodes)', f'NEB({len(neb_long.optimized)} nodes)',"AS-NEB"],
       height=[n_grad_short_neb, n_grad_long_neb,n_grad_msmep])

ax.bar_label(bars,fontsize=fs)


plt.yticks(fontsize=fs)
plt.text(.03,.95, f"{round(n_grad_long_neb / n_grad_msmep, 2 )}x improvement",transform=ax.transAxes,fontsize=fs-3,
        bbox={'visible':True,'fill':False})
plt.ylabel("Number of gradient calls",fontsize=fs)
plt.xticks(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/fixed_{rn}_gradient_comparison.svg")
plt.show()

# +

""
n_steps_explore_msmep = sum([len(obj.data.chain_trajectory) -1 for obj in h.depth_first_ordered_nodes if obj.is_leaf ] )
n_steps_refine_msmep = sum([len(obj.data.chain_trajectory) -1 for obj in h.depth_first_ordered_nodes if not obj.is_leaf] )


n_steps_long_neb = len(neb_long.chain_trajectory) -1
n_steps_short_neb = len(neb_short.chain_trajectory) -1

n_grad_explore_msmep = n_steps_explore_msmep*(15-2)
n_grad_refine_msmep = n_steps_refine_msmep*(15-2)

n_grad_long_neb = n_steps_long_neb*(30-2)
n_grad_short_neb = n_steps_short_neb*(15-2)

""
fig = 7
min_val = -5.3
max_val = 5.3
fs = 18
f,ax = plt.subplots(figsize=(1.16*fig,fig))

x_labels = [f'NEB({len(neb_short.optimized)} nodes)', f'NEB({len(neb_long.optimized)} nodes)',"AS-NEB", "AS-NEB"]
heights = [n_grad_short_neb, n_grad_long_neb,n_grad_explore_msmep, n_grad_refine_msmep]
bottoms = [0,0,0, n_grad_explore_msmep]


for xl, hv, bt in zip(x_labels, heights, bottoms):
    if bt == 0:
        ax.bar(x=xl, height=hv, bottom=bt, color='blue')
    else:
        ax.bar(x=xl, height=hv, bottom=bt, color='orange')


plt.yticks(fontsize=fs)
plt.text(.03,.95, f"{round(n_grad_long_neb / n_grad_msmep, 2 )}x improvement",transform=ax.transAxes,fontsize=fs-3,
        bbox={'visible':True,'fill':False})
plt.ylabel("Number of gradient calls",fontsize=fs)
plt.xticks(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/fixed_{rn}_gradient_comparison.svg")
plt.show()

# +

""
s=5
fs=18
f, ax = plt.subplots(figsize=(2.16*s, s))

ms = 7
lw = 2
hexvals=[
    '#ff1b1b',
'#dd0090',
'#5855c3']



# plot initial guesses
short_init_guess = Chain.from_xyz(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/initial_guess.xyz"), ChainInputs())
plt.plot(short_init_guess.integrated_path_length, (short_init_guess.energies-root_chain.energies[0])*627.5,'o--', label='15 nodes'
         ,markersize=ms,linewidth=lw, color=hexvals[0])







long_init_guess = Chain.from_xyz(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/initial_guess_long.xyz"), ChainInputs())
plt.plot(long_init_guess.integrated_path_length, (long_init_guess.energies-root_chain.energies[0])*627.5,'o--', label='30 nodes'
         ,markersize=ms,linewidth=lw, color='orange')






gi_leaf1 = asneb.ordered_leaves[0].data.initial_chain
plt.plot(gi_leaf1.integrated_path_length*out_pathlen[14], (gi_leaf1.energies-root_chain.energies[0])*627.5,'o--', label='Leaf1'
         ,markersize=ms,linewidth=lw, color='green')
# plt.plot(asneb.children[0].data.optimized.integrated_path_length*out_pathlen[14],
#          (asneb.children[0].data.optimized.energies-root_chain.energies[0])*627.5,'-', label='Leaf1 opt'
#          ,markersize=ms,linewidth=lw, color='green')




gi_leaf2 = asneb.ordered_leaves[1].data.initial_chain
plt.plot((gi_leaf2.integrated_path_length)*(out_pathlen[-1] - out_pathlen[14])+out_pathlen[14],
         (gi_leaf2.energies-root_chain.energies[0])*627.5,'o--', label='Leaf2'
         ,markersize=ms,linewidth=lw, color='blue')
# plt.plot(asneb.children[1].data.optimized.integrated_path_length*(out_pathlen[-1] - out_pathlen[14])+out_pathlen[14],
#          (asneb.children[1].data.optimized.energies-root_chain.energies[1])*627.5,'-', label='Leaf2 opt'
#          ,markersize=ms,linewidth=lw, color='blue')


out_pathlen = out.integrated_path_length
plt.plot(out.integrated_path_length, (out.energies-root_chain.energies[0])*627.5-0,'-', label="MEP"
        ,markersize=ms,linewidth=lw, color='black')



plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.xlabel("Normalized path length",fontsize=fs)

plt.legend(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/fixed_{rn}_energy_paths_initial_guesses.svg")
plt.show()

