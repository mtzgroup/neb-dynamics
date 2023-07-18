# +
from pathlib import Path
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
import numpy as np
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.Node2d import Node2D_Flower, Node2D
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Node3D import Node3D

from neb_dynamics.Node3D_gfn1xtb import Node3D_gfn1xtb
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS

from neb_dynamics.TreeNode import TreeNode

from itertools import product
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from neb_dynamics.CompetitorAnalyzer import CompetitorAnalyzer

import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
# import os
# del os.environ['OE_LICENSE']

from neb_dynamics.Janitor import Janitor

from neb_dynamics.MSMEP import MSMEP
from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

NIMAGES = 15


# # Helper Functions

def get_eA(chain_energies):
    return max(chain_energies) - chain_energies[0]


# +
def plot_chain(chain,linestyle='--',ax=None, marker='o',**kwds):
    if ax:
        ax.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)
    else:
        plt.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)

        
def plot_neb(neb,linestyle='--',marker='o',ax=None,**kwds):
    plot_chain(chain=neb.chain_trajectory[-1],linestyle='-',marker=marker,ax=ax,**kwds)
    plot_chain(chain=neb.initial_chain,linestyle='--',marker=marker,ax=ax,**kwds)


# +
def plot_chain2d(chain,linestyle='--',ax=None, marker='o',**kwds):
    if ax:
        ax.plot(chain.integrated_path_length,chain.energies,linestyle=linestyle,marker=marker,**kwds)
    else:
        plt.plot(chain.integrated_path_length,chain.energies,linestyle=linestyle,marker=marker,**kwds)

        
def plot_neb2d(neb,linestyle='--',marker='o',ax=None,**kwds):
    plot_chain2d(chain=neb.chain_trajectory[-1],linestyle='-',marker=marker,ax=ax,**kwds)
    plot_chain2d(chain=neb.initial_chain,linestyle='--',marker=marker,ax=ax,**kwds)


# -

# # 2D potentials

# +
ind = 0

the_noise = [-1,1]

noises_bool = [
    True,
    False

]




start_points = [
     [-2.59807434, -1.499999  ],
    [-3.77931026, -3.283186  ]
]

end_points = [
    [2.5980755 , 1.49999912],
    [2.99999996, 1.99999999]

]
tols = [
    0.1,
    0.05,

]

step_sizes = [
    1,
    1
]


k_values = [
    1,#.05,
    50

]



nodes = [Node2D_Flower, Node2D]
node_to_use = nodes[ind]
start_point = start_points[ind]
end_point = end_points[ind]
tol = tols[ind]

ss = step_sizes[ind]
ks = k_values[ind]
do_noise = noises_bool[ind]

# +
nimages = NIMAGES
np.random.seed(0)



coords = np.linspace(start_point, end_point, nimages)
if do_noise:
    coords[1:-1] += the_noise # i.e. good initial guess

    
cni_ref = ChainInputs(
    k=ks,
    node_class=node_to_use,
    delta_k=0,
    step_size=ss,
    # step_size=.01,
    do_parallel=False,
    use_geodesic_interpolation=False,
    min_step_size=.001
)
gii = GIInputs(nimages=nimages)
nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=0, node_freezing=False, 
               vv_force_thre=0)
chain_ref = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni_ref)
# -

n_ref = NEB(initial_chain=chain_ref,parameters=nbi)
n_ref.optimize_chain()

gii = GIInputs(nimages=nimages)
nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_chain_rms_thre=0.02, early_stop_force_thre=1, node_freezing=False)
# nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=3, node_freezing=False)
m = MSMEP(neb_inputs=nbi_msmep,chain_inputs=cni_ref, gi_inputs=gii)
history, out_chain = m.find_mep_multistep(chain_ref)

obj = n_ref
distances = obj._calculate_chain_distances()
forces = [c.get_maximum_grad_magnitude() for c in obj.chain_trajectory]

# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)

ax.plot(distances, 'o-',color='blue', label='chain distances')
plt.xticks(fontsize=fs)

plt.yticks(fontsize=fs)

ax2 = ax.twinx()
ax2.plot(forces, 'o-',color='orange',label='max(|∇$_{\perp}$|)')
plt.yticks(fontsize=fs)
ax.set_ylabel("Distance between chains",fontsize=fs)
ax2.set_ylabel("Maximum gradient component absolute value",fontsize=fs)
f.legend(fontsize=fs, loc='upper left')

# +
nimages_long = len(out_chain)

coords_long = np.linspace(start_point, end_point, nimages_long)
if do_noise:
    coords_long[1:-1] += the_noise # i.e. good initial guess
gii = GIInputs(nimages=nimages_long)
nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=0, node_freezing=False)
cni_ref2 = ChainInputs(
    k=1,
    node_class=node_to_use,
    delta_k=0,
    step_size=1,
    do_parallel=False,
    use_geodesic_interpolation=False,
    min_step_size=0.001
)
# chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref)
chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref2)

n_ref_long = NEB(initial_chain=chain_ref_long,parameters=nbi)
n_ref_long.optimize_chain()

# +
#### get energies for countourplot
gridsize = 100
min_val = -4
max_val = 4
# min_val = -.05
# max_val = .05

x = np.linspace(start=min_val, stop=max_val, num=gridsize)
y = x.reshape(-1, 1)
# -

h_flat_ref = np.array([node_to_use.en_func_arr(pair) for pair in product(x,x)])
h_ref = h_flat_ref.reshape(gridsize,gridsize).T

# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
# x = np.linspace(start=min_val, stop=max_val, num=1000)
# y = x.reshape(-1, 1)

cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
# cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

plot_chain(n_ref.initial_chain, c='orange',label='initial guess')
plot_chain(n_ref.chain_trajectory[-1], c='green',linestyle='-',label=f'NEB({nimages} nodes)')
plot_chain(n_ref_long.chain_trajectory[-1], c='gold',linestyle='-',label=f'NEB({nimages_long} nodes)', marker='*', ms=12)
plot_chain(out_chain, c='red',marker='o',linestyle='-',label='AS-NEB')

plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/results_2D_potential_ind{ind}.svg")
plt.show()

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
# ax.set_facecolor("lightgray")

plot_chain2d(n_ref.initial_chain, c='orange',label='initial guess')
plot_chain2d(n_ref.chain_trajectory[-1], c='green',linestyle='-',label=f'neb({nimages} nodes)')
plot_chain2d(out_chain, c='red',marker='o',linestyle='-',label='as-neb')
plot_chain2d(n_ref_long.chain_trajectory[-1], c='silver',linestyle='-',label=f'neb({nimages_long} nodes)')
plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# +
n_steps_orig_neb = len(n_ref.chain_trajectory)-1
n_steps_msmep = sum([len(obj.chain_trajectory) for obj in history.get_optimization_history()])-1 
n_steps_long_neb = len(n_ref_long.chain_trajectory)-1

n_grad_orig_neb = n_steps_orig_neb*(NIMAGES-2)
n_grad_msmep = n_steps_msmep*(NIMAGES-2)
n_grad_long_neb = n_steps_long_neb*(nimages_long-2)

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f,ax = plt.subplots(figsize=(1.16*fig,fig))
# plt.bar(x=["AS-NEB",f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)'],
#        height=[n_steps_msmep, n_steps_orig_neb, n_steps_long_neb])

bars = ax.bar(x=[f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)',"AS-NEB",],
       height=[n_grad_orig_neb, n_grad_long_neb, n_grad_msmep])

ax.bar_label(bars,fontsize=fs)


plt.yticks(fontsize=fs)
# plt.ylabel("Number of optimization steps",fontsize=fs)
plt.text(.63,.95, f"{round((n_grad_long_neb / n_grad_msmep), 2)}x improvement",transform=ax.transAxes,fontsize=fs,
        bbox={'visible':True,'fill':False})
plt.ylabel("Number of gradient calls",fontsize=fs)

plt.xticks(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/results_2D_potential_ind{ind}_barplot.svg")
plt.show()
# -

# # Visualize Big data

ca = CompetitorAnalyzer(comparisons_dir=Path("/home/jdep/T3D_data/msmep_draft/comparisons/"),method='asneb')

# +
# ca.submit_all_jobs()
# -

rns = ca.available_reaction_names

# +
rn = rns[0]

def get_relevant_chain(rn):
    data_dir = ca.out_folder / rn
    clean_chain = data_dir / 'initial_guess_msmep_clean.xyz'
    msmep_chain = data_dir / 'initial_guess_msmep.xyz'

    if clean_chain.exists():
        chain_to_use = Chain.from_xyz(clean_chain, ChainInputs())
    elif not clean_chain.exists() and msmep_chain.exists():
        chain_to_use = Chain.from_xyz(msmep_chain,ChainInputs())
    else: # somehow the thing failed
        print(f"{rn} failed.")
        chain_to_use = None

    return chain_to_use

def get_relevant_tree(rn):
    data_dir = ca.out_folder / rn
    fp = data_dir / 'initial_guess_msmep'
    tree = TreeNode.read_from_disk(fp)
    return tree

def get_relevant_leaves(rn):
    data_dir = ca.out_folder / rn
    fp = data_dir / 'initial_guess_msmep'
    adj_mat_fp = fp / 'adj_matrix.txt'
    adj_mat = np.loadtxt(adj_mat_fp)
    if adj_mat.size == 1:
        return [Chain.from_xyz(fp / f'node_0.xyz', ChainInputs(k=0.1, delta_k=0.09))]
    else:
    
        a = np.sum(adj_mat,axis=1)
        inds_leaves = np.where(a == 1)[0] 
        chains = [Chain.from_xyz(fp / f'node_{ind}.xyz',ChainInputs(k=0.1, delta_k=0.09)) for ind in inds_leaves]
        return chains


# +
# ChainInputs(k=0.1, delta_k=0.09)
# -

def get_eA_chain(chain):
    eA = max(chain.energies_kcalmol)
    return eA


def get_eA_leaf(leaf):
    chain = leaf.data.optimized
    eA = max(chain.energies_kcalmol)
    return eA


cs = get_relevant_leaves('Semmler-Wolff-Reaction')
# cs = get_relevant_chain("Semmler-Wolff-Reaction")

a, b = cs[0].pe_grads_spring_forces_nudged()

ind = 6
np.dot(a[ind].flatten(),b[ind].flatten())


def get_maximum_gperp(self):
    gperp, gspring = self.pe_grads_spring_forces_nudged()
    max_gperps = []

    for gp, node in zip(gperp, self):
        # remove rotations and translations
        if gp.shape[1] >= 3:  # if we have at least 3 atoms
            gp[0, :] = 0  # this atom cannot move
            gp[1, :2] = 0  # this atom can only move in a line
            gp[2, :1] = 0  # this atom can only move in a plane
        print(gp)
        if not node.converged:
            max_gperps.append(np.amax(np.abs(gp)))
            
    print(max_gperps)
    return np.max(max_gperps)


cs[0].get_maximum_gperp()

# +
gperp, gspring = cs[0].pe_grads_spring_forces_nudged()
# gperp[5, 0, :] = 0  # this atom cannot move
# gperp[5, 1, :2] = 0  # this atom can only move in a line
# gperp[5, 2, :1] = 0  # this atom can only move in a plane

# gspring[5, 0, :] = 0  # this atom cannot move
# gspring[5, 1, :2] = 0  # this atom can only move in a line
# gspring[5, 2, :1] = 0  # this atom can only move in a plane


# -

ind = 1
np.dot(gperp[ind].flatten(), gspring[ind].flatten())

np.amax(np.abs(gperp[5]))

cs[0].get_maximum_gperp()

np.amax(a[ind] - b[ind])

cs[0].get_maximum_grad_magnitude()

# rn = rns[10]
all_max_barriers = []
all_n_steps = []
failed = []
for i, rn in enumerate(rns):
    try:
        cs = get_relevant_leaves(rn)
        eAs = [get_eA_chain(c) for c in cs]
        maximum_barrier = max(eAs)
        n_steps = len(eAs)


        all_max_barriers.append((i, maximum_barrier))
        all_n_steps.append((i, n_steps))
    except:
        failed.append(rn)

plt.hist([x[1] for x in all_max_barriers])

plt.hist([x[1] for x in all_n_steps])

look_at_me = rns[118]
look_at_me

ca.submit_a_job_by_name('Bamberger-Rearrangement')

tr_opt = tr.get_conv_gi(tr)

tr_opt[0]

cs = get_relevant_leaves(look_at_me)

tr = Trajectory.from_xyz(Path('/home/jdep/T3D_data/msmep_draft/comparisons/structures/Bamberger-Rearrangement/gi_fric0.001.xyz'))

ca.submit_a_job_by_name("Bamberger-Rearrangement")

tr.draw()

tr.gradient_xtb()

[x.get_maximum_grad_magnitude() for x in cs]

c = Chain.from_list_of_chains(cs,ChainInputs())

c.plot_chain()

sorted(all_n_steps, key=lambda x: x[1], reverse=True)

# +
# c = get_relevant_chain(rn)
# -

rn

c.plot_chain()

argrelmin(c.energies)[0]


