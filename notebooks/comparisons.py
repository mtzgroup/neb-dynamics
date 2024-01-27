from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.Linesearch import Linesearch

# +
from pathlib import Path
# from retropaths.abinitio.trajectory import Trajectory
# from retropaths.abinitio.tdstructure import TDStructure

# from neb_dynamics.trajectory import Trajectory
from neb_dynamics.tdstructure import TDStructure
import numpy as np
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.NEB_TCDLF import NEB_TCDLF
from neb_dynamics.nodes.Node2d import Node2D_Flower, Node2D, Node2D_LEPS
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from neb_dynamics.nodes.Node3D import Node3D

from neb_dynamics.nodes.Node3D_gfn1xtb import Node3D_gfn1xtb
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.potential_functions import sorry_func_0, sorry_func_1, sorry_func_2,sorry_func_3, flower_func

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
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from chemcloud import CCClient

from neb_dynamics.MSMEP import MSMEP
from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from neb_dynamics.nodes.Node3D_Water import Node3D_Water
from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB
from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local

# rn = 'Benzimidazolone-Synthesis-1-X-Iodine'
rn = 'Aza-Grob-Fragmentation-X-Bromine'
orig = TreeNode.read_from_disk(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/start_opt_2024_msmep/"))

orig_obj = [obj for obj in orig.get_optimization_history()]

cni = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D_TC_TCPB, do_parallel=True, node_freezing=True)
# cni = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D_TC_TCPB, do_parallel=True, node_freezing=True)
nbi = NEBInputs(tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, v=1, _use_dlf_conv=False, climb=True, 
                early_stop_chain_rms_thre=1, early_stop_force_thre=0.01, early_stop_still_steps_thre=100, preopt_with_xtb=True)

optimizer = BFGS(step_size=3, min_step_size=.5, use_linesearch=False, bfgs_flush_thre=0.80, 
                 activation_tol=0.1, bfgs_flush_steps=200)

# +
# optimizer = VelocityProjectedOptimizer(timestep=0.5)

# +
# h = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Claisen-Rearrangement/start_opt_2024_msmep"), neb_parameters=nbi, chain_parameters=cni)

# +
# chain  = h.data.initial_chain
# -

from neb_dynamics.trajectory import Trajectory

# tr = Trajectory.from_xyz('/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Claisen-Rearrangement/initial_guess.xyz')
tr = orig.data.initial_chain.to_trajectory()

ref = tr[0]

ref.tc_model_method = 'wb97xd3'
ref.tc_model_basis = 'def2-svp'

tr.update_tc_parameters(ref)

initial_chain = Chain.from_traj(tr, parameters=cni)

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=12), optimizer=optimizer)

h, out = m.find_mep_multistep(initial_chain)

h.write_to_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Aza-Grob-Fragmentation-X-Bromine/xtb_seed_2024"))

len(h.get_optimization_history())

orig.output_chain.plot_chain()

out[0].energy

orig.output_chain[0].energy

orig.output_chain.get_eA_chain()

out.get_eA_chain()

# plt.plot(orig.output_chain.path_length, orig.output_chain.energies, 'o-',label='orig')
plt.plot(h.output_chain.path_length, h.output_chain.energies, 'o-',label='xtb-seeded')
plt.legend()

for obj in orig.get_optimization_history():
    if obj:
        print(len(obj.chain_trajectory))

for obj in h.get_optimization_history():
    if obj:
        print(len(obj.chain_trajectory))

tsg = out.get_ts_guess()

out.to_trajectory()

plt.plot(n.initial_chain.integrated_path_length, n.initial_chain.energies_kcalmol)
plt.plot(n.chain_trajectory[-1].integrated_path_length, n.chain_trajectory[-1].energies_kcalmol)
# n.chain_trajectory[-1].to_trajectory()

n.chain_trajectory[-1].to_trajectory()

plt.plot(n.initial_chain.integrated_path_length, n.initial_chain.energies_kcalmol,'o-')
plt.plot(n.chain_trajectory[-1].integrated_path_length, n.chain_trajectory[-1].energies_kcalmol,'o-')
# n.chain_trajectory[-1].to_trajectory()

# +
# m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=12), optimizer=optimizer)
# -

h, out = m.find_mep_multistep(initial_chain)

out.plot_chain()

n2.plot_projector_history(var='gradients')

n.plot_opt_history(1)

n2.optimized[0].is_identical(n.optimized[0])

n2.plot_opt_history(1)

len(n2.chain_trajectory)

t[-1]

n.chain_trajectory[-1].get_ts_guess()

# %%time
try:
    neb.optimize_chain(remove_all=False)
except Exception as e:
    print(e)


def get_ens_from_fp(fp):
        lines = open(fp).read().splitlines()
        atomn = int(lines[0])
        inds = list(range(1, len(lines), atomn+2))
        ens = np.array([float(line.split()[0]) for line in np.array(lines)[inds]])
        return ens


# +
def upsample_images_and_ens(nested_imgs_list, nested_ens_list):
    """
    will take a list of lists that contain variable number of images, and will upsample them to tot_number 
    if they are below the number. 
    """
    tot_number = get_number_to_upsample_to(nested_imgs_list)
    
    
    output_list = []
    output_ens = []
    for l, l_ens in zip(nested_imgs_list, nested_ens_list):
        if len(l) < tot_number:
            n_to_add = tot_number - len(l)
            new_list = l
            new_list_ens = list(l_ens)
            
            new_list.extend([l[-1]]*n_to_add)
            new_list_ens.extend([l_ens[-1]]*n_to_add)
            
            
            
            output_list.append(new_list)
            output_ens.append(new_list_ens)
        else:
            output_list.append(l)
            output_ens.append(l_ens)
    return output_list, output_ens
            
    
def get_number_to_upsample_to(nested_imgs_list):
    max_n = 0
    for l in nested_imgs_list:
        if len(l) > max_n:
            max_n = len(l)
    return max_n
    


# -

def get_chain_trajectory(data_dir, parameters):
    
        # start = self.initial_chain[0].tdstructure
        # start_en = self.initial_chain[0].energy
        # end = self.initial_chain[-1].tdstructure
        # end_en = self.initial_chain[-1].energy
        
        
        all_paths = list(data_dir.glob("neb_*.xyz"))
        max_ind = len(all_paths)
        
        all_fps = [data_dir / f'neb_{ind+1}.xyz' for ind in range(1, max_ind)]
        
        img_trajs = [Trajectory.from_xyz(fp).traj for fp in all_fps]
        img_ens = [get_ens_from_fp(fp) for fp in all_fps]
        img_trajs, img_ens = upsample_images_and_ens(img_trajs, img_ens)
        
        
        
        chains_imgs = list(zip(*img_trajs))
        chains_ens = list(zip(*img_ens))
        
        start = img_trajs[0][0]
        start_en = img_ens[0][0]
        end = img_trajs[-1][0]
        end_en = img_ens[-1][0]
        
        
        
        all_trajs = []
        for imgs in chains_imgs:
            t = Trajectory([start]+list(imgs)+[end])
            t.update_tc_parameters(start)
            all_trajs.append(t)
                    
        chain_trajectory = []
        for t, raw_ens in zip(all_trajs, chains_ens):
            ens = [start_en]+list(raw_ens)+[end_en] # dlf gives only the middle image energies
            c = Chain.from_traj(t, parameters)
            for node, en in zip(c.nodes, ens):
                node._cached_energy = en
            chain_trajectory.append(c) 
        
        return chain_trajectory

ct = get_chain_trajectory(Path('/tmp/tmphtv8v0q9'), ChainInputs())

neb.plot_opt_history(1)

t1 = Trajectory.from_xyz('/tmp/tmphtv8v0q9/neb_1.xyz')

tr[0]

t1

neb.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_bfgs_new_conv"))

neb.optimized.is_elem_step()

neb.optimized.get_ts_guess()

neb.plot_opt_history(1)

tsg = neb.optimized.get_ts_guess()

tsg

ts = tsg.tc_geom_optimization(method='ts')

ts

neb.optimized.plot_chain()

neb.optimized.get_ts_guess()

neb.plot_opt_history(0)

# +


cni2 = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D, do_parallel=True, node_freezing=True)
nbi2 = NEBInputs(tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, v=1, _use_dlf_conv=False, climb=False)
chain2 = Chain.from_traj(tr, parameters=cni2)
# -

optimizer2 = VelocityProjectedOptimizer(timestep=1, activation_tol=0.5)

neb2 = NEB(initial_chain=chain2, parameters=nbi2, optimizer=optimizer)

# %%time
try:
    neb2.optimize_chain()
except Exception as e:
    print(e)
    print("done")

plt.plot(neb.optimized.path_length, neb.optimized.energies,'o-', label='cNEB')
plt.plot(neb2.optimized.path_length, neb2.optimized.energies,'o-',label='NEB')
plt.legend()

# tr[2].tc_kwds = {'pcm':'cosmo','epsilon':80}
tr[2].tc_kwds = {}

# # Helper Functions

# +

NIMAGES = 15


# -

def animate_this_mf(neb_obj, potential_func,
                   saveasgif=False, fn='anim.gif'):
    # %matplotlib notebook
    import matplotlib.pyplot as plt
    import matplotlib.animation
    import numpy as np


    n_nodes = len(neb_obj.initial_chain.nodes)
    en_func = neb_obj.initial_chain[0].en_func
    chain_traj = neb_obj.chain_trajectory
    # plt.style.use("seaborn-pastel")

    figsize = 5
    s=4

    fig, ax = plt.subplots(figsize=(1.618 * figsize, figsize))

    min_val = -s
    max_val = s

    x = np.linspace(start=min_val, stop=max_val, num=1000)
    y = x.reshape(-1, 1)



    h = potential_func([x, y])
    cs = plt.contourf(x, x, h)
    _ = fig.colorbar(cs, ax=ax)

    (line,) = ax.plot([], [], "o--", lw=3)

    arrows = [
            ax.arrow(0, 0, 0, 0, head_width=0.05, facecolor="black") for _ in range(n_nodes)
        ]

    def animate(chain):

            x = chain.coordinates[:, 0]
            y = chain.coordinates[:, 1]
            
            
            
            color = 'lightblue'

            for arrow, (x_i, y_i), (dx_i, dy_i) in zip(
                arrows, chain.coordinates, chain.gradients
            ):
                arrow.set_data(x=x_i, y=y_i, dx=-1 * dx_i, dy=-1 * dy_i)

            hess = chain.bfgs_hess
            if np.all(hess == np.eye(hess.shape[0])):
                # line.set_color='red'
                
                line.set_data(x, y)
                line.set_color("red")
            else:
                line.set_data(x, y)
                line.set_color("skyblue")
            
            
            
                
            return (x for x in arrows)

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=chain_traj)

    if saveasgif:
        ani.save(fn)
    
    
    from IPython.display import HTML
    return HTML(ani.to_jshtml())


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
ind = 1

the_noise = [-1,1]

noises_bool = [
    True,
    False,
    False

]




start_points = [
     [-2.59807434, -1.499999  ],
    [-3.77931026, -3.283186  ],
    [0.74200203, 4]
]


end_points = [
    [2.5980755 , 1.49999912],
    [2.99999996, 1.99999999],
    [4, 0.74200311]

]
tols = [
    0.1,
    0.05,
    0.01, 

]

step_sizes = [
    1,
    .1,
    .1
]


k_values = [
    1,#.05,
    50,
    .1

]



nodes = [Node2D_Flower, Node2D, Node2D_LEPS]
node_to_use = nodes[ind]
start_point = start_points[ind]
end_point = end_points[ind]
tol = tols[ind]

ss = step_sizes[ind]
ks = k_values[ind]
do_noise = noises_bool[ind]
# -

from neb_dynamics.optimizers.Linesearch import Linesearch
from neb_dynamics.optimizers.SD import SteepestDescent
from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer

NIMAGES=15

# +
nimages = NIMAGES
# NIMAGES=30
np.random.seed(0)



coords = np.linspace(start_point, end_point, NIMAGES)
if do_noise:
    coords[1:-1] += the_noise # i.e. good initial guess

    
# cni_ref = ChainInputs(
#     k=ks,
#     node_class=node_to_use,
#     delta_k=0,
#     # step_size=ss,
#     # step_size=.01,
#     do_parallel=False,
#     use_geodesic_interpolation=False,
#     # min_step_size=.001,
#     # als_max_steps=3
# )

cni_ref = ChainInputs(
    k=ks,
    node_class=node_to_use,
    delta_k=0.0,
    # step_size=ss,
    # step_size=.01,
    do_parallel=False,
    use_geodesic_interpolation=False,
    node_freezing=True,
    # min_step_size=.001,
    # als_max_steps=3
)

gii = GIInputs(nimages=nimages)
nbi = NEBInputs(tol=tol, v=1, max_steps=500, climb=False, early_stop_force_thre=0,
               vv_force_thre=0)
chain_ref = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni_ref)
chain_ref.parameters.hess_prev = np.eye(chain_ref.gradients.flatten().shape[0])
# -

optimizer = Linesearch(step_size=ss, min_step_size=.001, als_max_steps=3)

# +
# optimizer2 = BFGS(step_size=ss, min_step_size=.001, als_max_steps=3, 
#                   bfgs_flush_steps=20, bfgs_flush_thre=0.90, update_using_gperp=False)

# +
# optimizer_BFGS = BFGS(step_size=ss, min_step_size=.01, als_max_steps=3, 
#                   bfgs_flush_steps=1000, bfgs_flush_thre=-2, update_using_gperp=False)
# -

optimizer = VelocityProjectedOptimizer(timestep=0.01, activation_tol=.05)
# optimizer = SteepestDescent(step_size_per_atom=0.001)

# +
# optimizer3 = VelocityProjectedOptimizer(step_size=ss, min_step_size=.001, als_max_steps=3)
# -

n_ref = NEB(initial_chain=chain_ref,parameters=nbi, optimizer=optimizer)
n_ref.optimize_chain()

n_ref.grad_calls_made

animate_this_mf(n_ref, potential_func=sorry_func_0)

n_ref2 = NEB(initial_chain=chain_ref,parameters=nbi, optimizer=optimizer2)
n_ref2.optimize_chain()

n_ref_BFGS = NEB(initial_chain=chain_ref,parameters=nbi, optimizer=optimizer_BFGS)
n_ref_BFGS.optimize_chain()

# +
# n_ref3 = NEB(initial_chain=chain_ref, parameters=nbi, optimizer=optimizer3)
# n_ref3.optimize_chain()
# -

animate_this_mf(n_ref2, sorry_func_0, saveasgif=False, fn='/home/jdep/T3D_data/bfgs_results/reference.gif')

gii = GIInputs(nimages=nimages)
nbi_msmep = nbi
nbi_msmep.early_stop_chain_rms_thre=10
nbi_msmep.early_stop_force_thre=1
#NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_chain_rms_thre=10, early_stop_force_thre=1)
# nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, ear|ly_stop_force_thre=3, node_freezing=False)
m = MSMEP(neb_inputs=nbi_msmep,chain_inputs=cni_ref, gi_inputs=gii, optimizer=optimizer2)
history, out_chain = m.find_mep_multistep(chain_ref)

obj = n_ref2
distances = obj._calculate_chain_distances()
forces = [c.get_maximum_grad_magnitude() for c in obj.chain_trajectory]

n_ref2.chain_trajectory[0].bfgs_hess.shape

n_ref2.chain_trajectory[0].bfgs_hess.shape

np.eye(60)

did_flush = [np.all(n_ref2.chain_trajectory[i].bfgs_hess == np.eye(60)) for i in range(len(n_ref2.chain_trajectory))]

len(n_ref.chain_trajectory)

len(n_ref2.chain_trajectory)

plt.plot(did_flush)

n_ref2.plot_projector_history()

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
nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=0, bfgs_flush_steps=20, bfgs_flush_thre=.90, do_bfgs=False)
# cni_ref2 = ChainInputs(
#     k=30,
#     node_class=node_to_use,
#     delta_k=0,
#     step_size=.1,
#     do_parallel=False,
#     use_geodesic_interpolation=False,
#     min_step_size=0.001, als_max_steps=3,
# )
chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref)
# chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref2)

n_ref_long = NEB(initial_chain=chain_ref_long,parameters=nbi, optimizer=optimizer)
n_ref_long.optimize_chain()
# -



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

from neb_dynamics.TreeNode import TreeNode

wtf = NEB.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Wittig/wb97xd3-def2svp_msmep/node_0.xyz")

wtf.optimized.to_trajectory()[2]

wtf.optimized.to_trajectory()

# +
# fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
# x = np.linspace(start=min_val, stop=max_val, num=1000)
# y = x.reshape(-1, 1)

cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
# cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

# plot_chain(n_ref.initial_chain, c='orange',label='initial guess')

plot_chain(n_ref2.chain_trajectory[-1], c='blue',linestyle='-',marker='*',label=f'NEB({nimages} nodes)', ms=20)
plot_chain(n_ref.chain_trajectory[-1], c='green',linestyle='-',label=f'NEB({nimages} nodes)')
# plot_chain(n_ref_long.chain_trajectory[-1], c='gold',linestyle='-',label=f'NEB({nimages_long} nodes)', marker='*', ms=12)
# plot_chain(out_chain, c='red',marker='o',linestyle='-',label='AS-NEB')

plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/results_2D_potential_ind{ind}.svg")
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
n_steps_msmep = sum([len(obj.chain_trajectory) for obj in history.get_optimization_history() if obj])-1 
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
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/results_2D_potential_ind{ind}_barplot.svg")
plt.show()
# -

# # BFGS vs Linesearch

from itertools import product

# +
flush_steps = [1, 10, 20, 50, 100]
flush_thre = [0, 0.1, 0.2, 0.3, 0.4, 0.50, 0.6, 0.7, 0.8, 0.9, 0.99]

# nimgs = [5, 10, 15, 20, 50]
nimgs = [50]


conditions = list(product(flush_steps, flush_thre))
# conditions = conditions[6:] # only keep one control group conditions
conditions = conditions[10:] # only keep one control group conditions
# -

from neb_dynamics.NEB import NoneConvergedException

# +
ALL_RESULTS = []

for nimg in nimgs:
    results_nebs = []

    coords = np.linspace(start_point, end_point, nimg)
    for  (f_steps, f_thre) in conditions:
        print(f"{nimg=} {f_steps=} {f_thre=}\n")
        cni = ChainInputs(
            k=ks,
            node_class=node_to_use,
            delta_k=0,
            do_parallel=False,
            use_geodesic_interpolation=False,
            )
        gii = GIInputs(nimages=nimg)

        nbi = NEBInputs(tol=tol, v=1, max_steps=1000, climb=False, early_stop_force_thre=0,
                       vv_force_thre=0, rms_grad_thre=tol)

        chain = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni)
        optimizer = BFGS(step_size=ss, min_step_size=.001, als_max_steps=3, bfgs_flush_steps=f_steps, bfgs_flush_thre=f_thre,
                        update_using_gperp=False)
        # print(f'\t{sum(chain.bfgs_hess.diagonal())}')

        n = NEB(initial_chain=chain,parameters=nbi, optimizer=optimizer)
        try:
            n.optimize_chain()
            results_nebs.append(n)
        except NoneConvergedException as e:
            results_nebs.append(e.obj)
    
    ALL_RESULTS.append(results_nebs)

# +
all_dfs = []

for results_nebs, nimg  in zip(ALL_RESULTS, nimgs):
    # results_nebs = ALL_RESULTS[0]

    results = [] # f_steps, f_thre, n_steps
    for (f_steps, f_thre), n_result in zip(conditions, results_nebs):
        results.append([f_steps, f_thre, len(n_result.chain_trajectory)])

    import pandas as pd

    df = pd.DataFrame(results, columns=['f_steps','f_thre','n_steps'])

    df = df.sort_values(by='n_steps')
    df["nimg"] = [nimg]*len(results_nebs)
    all_dfs.append(df)


# +
# end_point2 = -2.80511811,  3.1313125

coords = np.linspace(start_point, end_point, 165)

cni = ChainInputs(
    k=1,
    node_class=node_to_use,
    delta_k=0,
    do_parallel=False,
    use_geodesic_interpolation=False,
    )
gii = GIInputs(nimages=165)

nbi = NEBInputs(tol=tol, v=1, max_steps=1000, climb=False, early_stop_force_thre=0,
               vv_force_thre=0, rms_grad_thre=tol)

chain = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni)
optimizer = BFGS(step_size=0.001, min_step_size=.0001, als_max_steps=3, bfgs_flush_steps=10000, bfgs_flush_thre=0.40,
                update_using_gperp=False)
# print(f'\t{sum(chain.bfgs_hess.diagonal())}')

n_2d = NEB(initial_chain=chain,parameters=nbi, optimizer=optimizer)

# -

n_2d.optimize_chain()

# +
# animate_this_mf(n_2d, sorry_func_0)
# -

t = Trajectory.from_xyz('/home/jdep/T3D_data/geometry_spawning/claisen_results/claisen_ts_profile.xyz')

r,p = t[0], t[-1]

gi = Trajectory([r,p]).run_geodesic(nimages=10)

r.tc_model_method = 'wb97xd3'
r.tc_model_basis = 'def2-svp'

from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.optimizers.BFGS import BFGS

gi.update_tc_parameters(r)

# +
print(f"doing xtb")
cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D, node_freezing=True)
optimizer = BFGS(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn, 
                 bfgs_flush_steps=10000, bfgs_flush_thre=0.40)

chain = Chain.from_traj(gi, cni)

nbi = NEBInputs(v=True,tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, climb=False,
               _use_dlf_conv=True)
# -

n = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)
n.optimize_chain()

xtb_seed_tr = n.optimized.to_trajectory()

xtb_seed_tr.update_tc_parameters(r)

# +
print(f"doing xtb-seed results")
cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D_TC, node_freezing=True)
optimizer = BFGS(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn, 
                 bfgs_flush_steps=10000, bfgs_flush_thre=0.40)

chain = Chain.from_traj(xtb_seed_tr, cni)

nbi = NEBInputs(v=True,tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, climb=False,
               _use_dlf_conv=True)
# -

n_dft = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)
n_dft.optimize_chain()

n_dft.optimized.plot_chain()

n_dft.optimized.plot_chain()

n_dft.plot_opt_history(do_3d=True)

n.write_to_disk(Path("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen_bfgs_xtb"))

n_dft.write_to_disk(Path("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen_bfgs_xtb_seed"))

n_dft_cont = NEB(initial_chain=n_dft.chain_trajectory[-1], parameters=nbi, optimizer=optimizer)
n_dft_cont.optimize_chain()

len(ALL_RESULTS[3][1].chain_trajectory)

animate_this_mf(ALL_RESULTS[3][1], sorry_func_0, saveasgif=False, fn='/home/jdep/T3D_data/flashing_neb.gif')

ALL_DF = pd.concat(all_dfs)

ft = .75
fs = 100
sub = ALL_DF[(ALL_DF['f_steps']==fs)&(ALL_DF['f_thre']==ft)]
ref = ALL_DF[(ALL_DF['f_steps']==1)]

f,ax = plt.subplots()
sub.plot(x='nimg',y='n_steps',marker='o', ax=ax, label='bfgs w/linesearch')
ref.plot(x='nimg',y='n_steps',marker='o', ax=ax, label='steep. desc. w/linsearch')
plt.title(f"Flush_steps={fs} || Flush_grad_threshold={ft}")
plt.axhline(y=1000, linestyle='--', linewidth=2, label='did not converge', color='gray')
plt.ylabel("nsteps")
plt.legend()

import pandas as pd

all_df = pd.read_csv("/home/jdep/T3D_data/bfgs_results/ALL_DF.csv")


def get_rows_with_percent_improvement(percent, dataframe):
    ref = dataframe[dataframe['f_steps']==1]
    ref_steps = ref['n_steps'].values
    
    
    better_than_ref =  dataframe[((-1*(dataframe['n_steps'] - ref_steps) / ref_steps) >= percent)]
    print(better_than_ref)


# +
ind = 4
nimg = all_df['nimg'].unique()[ind]

sub = all_df[all_df['nimg']==nimg]

get_rows_with_percent_improvement(0.25, sub)
# -

# def scan_along_var(var):
var = 'f_thre'
f, ax = plt.subplots()
vals = np.unique(df['f_thre'])
for val in vals:
    df_sub = df[df[var]==val]
    df_sub.plot('f_steps','n_steps', label=f'{val}',ax=ax)


# # Visualize Big data

# +
# ca = CompetitorAnalyzer(comparisons_dir=Path("/home/jdep/T3D_data/msmep_draft/comparisons/"),method='asneb')

# +
# ca.submit_all_jobs()

# +
# rns = ca.available_reaction_names
# -

from dataclasses import dataclass

from neb_dynamics.helper_functions import RMSD, qRMSD_distance


def get_all_tsg_in_opt_history(opt_history):
    all_tsg = []
    for neb_obj in reversed(opt_history):
        start = neb_obj.initial_chain[0]
        end = neb_obj.initial_chain[-1]

        # gi = Node3D(neb_obj.initial_chain.get_ts_guess())
        tsg = Node3D(neb_obj.optimized.get_ts_guess())
        if any([ref_tsg.is_identical(tsg) for ref_tsg in all_tsg]):
            continue
        else:
            all_tsg.append(tsg)

    return all_tsg


@dataclass
class MSMEPAnalyzer:
    parent_dir: Path
    msmep_root_name: str
    

    def get_relevant_chain(self, folder_name):
        data_dir = self.parent_dir / folder_name
        clean_chain = data_dir / f'{self.msmep_root_name}_msmep_clean.xyz'
        msmep_chain = data_dir / f'{self.msmep_root_name}_msmep.xyz'
        
        if clean_chain.exists():
            chain_to_use = Chain.from_xyz(clean_chain, ChainInputs())
        elif not clean_chain.exists() and msmep_chain.exists():
            chain_to_use = Chain.from_xyz(msmep_chain,ChainInputs())
        else: # somehow the thing failed
            print(f"{folder_name} unavailable.")
            chain_to_use = None

        return chain_to_use
    
    def get_relevant_saddle_point(self, folder_name):
        data_dir = self.parent_dir / folder_name
        sp_fp = data_dir / 'sp.xyz'
        sp = TDStructure.from_xyz(str(sp_fp))
        return sp
    
    def _distance_to_sp(self, chain: Chain, sp):
        ts_guess = chain.get_ts_guess()
        ts_guess = ts_guess.align_to_td(sp)
        
        
        return qRMSD_distance(ts_guess.coords, sp.coords)
    
    def _distance_to_sp_tsg(self, ts_guess, sp):
        ts_guess = ts_guess.align_to_td(sp)
        return qRMSD_distance(ts_guess.coords, sp.coords)
    
    
    def get_relevant_leaves(self, folder_name):
        data_dir = self.parent_dir / folder_name
        fp = data_dir / f'{self.msmep_root_name}_msmep'
        adj_mat_fp = fp / 'adj_matrix.txt'
        adj_mat = np.loadtxt(adj_mat_fp)
        if adj_mat.size == 1:
            return [Chain.from_xyz(fp / f'node_0.xyz', ChainInputs(k=0.1, delta_k=0.09))]
        else:

            a = np.sum(adj_mat,axis=1)
            inds_leaves = np.where(a == 1)[0] 
            chains = [Chain.from_xyz(fp / f'node_{ind}.xyz',ChainInputs(k=0.1, delta_k=0.09)) for ind in inds_leaves]
            return chains
        
    def get_optimization_history(self, folder_name):
        data_dir = self.parent_dir / folder_name
        fp = data_dir / f'{self.msmep_root_name}_msmep'
        adj_mat_fp = fp / 'adj_matrix.txt'
        adj_mat = np.loadtxt(adj_mat_fp)
        if adj_mat.size == 1:
            return [NEB.read_from_disk(fp / f'node_0.xyz')]
        else:

            h = TreeNode.read_from_disk(fp)
            return h.get_optimization_history()
        
    
    def get_relevant_leaves_nebs(self, folder_name):
        data_dir = self.parent_dir / folder_name
        fp = data_dir / f'{self.msmep_root_name}_msmep'
        adj_mat_fp = fp / 'adj_matrix.txt'
        adj_mat = np.loadtxt(adj_mat_fp)
        if adj_mat.size == 1:
            return [NEB.read_from_disk(fp / f'node_0.xyz')]
        else:

            a = np.sum(adj_mat,axis=1)
            inds_leaves = np.where(a == 1)[0] 
            nebs = [NEB.read_from_disk(fp / f'node_{ind}.xyz') for ind in inds_leaves]
            return nebs
        
    def get_relevant_tree(self, folder_name):
        data_dir = self.parent_dir / folder_name
        fp = data_dir / f'{self.msmep_root_name}_msmep'
        h = TreeNode.read_from_disk(fp)
        return h


p = Path('/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/')
msma = MSMEPAnalyzer(parent_dir=p, msmep_root_name='react')
msma_ES = MSMEPAnalyzer(parent_dir=p, msmep_root_name='xtb_early_stop')

msma_dft = MSMEPAnalyzer(parent_dir=p, msmep_root_name='dft_early_stop')

from scipy.signal import argrelextrema

# +
dists = []
en_dists = []

dists_xtb = []
en_dists_xtb = []

dists_xtbES = []
en_dists_xtbES = []


dists_gi = []
en_dists_gi = []

dists_asnebbundle = []
en_dists_asnebbundle = []

n_multi = 0
n_single = 0
multis = []

names = []

msma_obj = msma_dft

all_sys = p.glob("sys*")

for sys_fp in all_sys:
    name = sys_fp.stem
    try:
        out = msma_obj.get_relevant_chain(name)
        sp = msma_obj.get_relevant_saddle_point(name)
        dists.append(msma_obj._distance_to_sp(out, sp)) 
        if out:


            out_xtb = msma.get_relevant_chain(name)
            dists_xtb.append(msma._distance_to_sp(out_xtb, sp)) 

            out_xtbES = msma_ES.get_relevant_chain(name)
            dists_xtbES.append(msma_ES._distance_to_sp(out_xtbES, sp)) 


            n = NEB.read_from_disk(msma_obj.parent_dir / name /  (str(msma_obj.msmep_root_name)+"_msmep") / 'node_0.xyz')
            out_gi = n.initial_chain 
            dists_gi.append(msma._distance_to_sp(out_gi, sp)) 

#             opt_history = msma.get_optimization_history(name)
#             all_tsg_nodes = get_all_tsg_in_opt_history(opt_history)
#             all_tsg = [node.tdstructure for node in all_tsg_nodes]
#             all_tsg_distances = [msma._distance_to_sp_tsg(tsg, sp) for tsg in all_tsg]
#             all_tsg_en_dists = [abs((tsg.energy_xtb() - sp.energy_xtb())*627.5) for tsg in all_tsg]

#             dists_asnebbundle.append(min(all_tsg_distances))
#             en_dists_asnebbundle.append(min(all_tsg_en_dists))




            en_dists.append(abs((out.get_ts_guess().energy_xtb() - sp.energy_xtb())*627.5))
            en_dists_xtb.append(abs((out_xtb.get_ts_guess().energy_xtb() - sp.energy_xtb())*627.5))
            en_dists_xtbES.append(abs((out_xtbES.get_ts_guess().energy_xtb() - sp.energy_xtb())*627.5))
            en_dists_gi.append(abs((out_gi.get_ts_guess().energy_xtb() - sp.energy_xtb())*627.5))

        #         all_ens_tr = Trajectory([out.get_ts_guess(), out_xtb.get_ts_guess(), out_xtbES.get_ts_guess(), sp])
        #         all_ens_tr.update_tc_parameters(ref)
        #         all_ens, _ = all_ens_tr.energies_and_gradients_tc()

        #         en_dft, en_xtb, en_xtbES, sp_en = all_ens


            # en_dists.append(abs(en_dft - sp_en))
            # en_dists_xtb.append(abs(en_xtb - sp_en))
            # en_dists_xtbES.append(abs(en_xtbES - sp_en))

            names.append(sys_fp.stem)

            # if len(argrelextrema(out.energies, np.greater)[0])==1:

            if len(out.energies) <= 8:
            # if len(out) <= 9:
                n_single+=1
            else:
                n_multi+=1
                multis.append([out, name])
    except:
        continue
# -

len(dists)

n_single

# +
ref = TDStructure.from_smiles("O")

ref.tc_model_method = 'B3LYP-D3BJ'
ref.tc_model_basis = 'def2-svp'
ref.tc_kwds = {'reference':'uks'}

# +
# freqs_xtb = []
# freqs_dft = []
# for name in names:
#     out_xtb = msma.get_relevant_chain(name)
#     out_dft = msma_dft.get_relevant_chain(name)
    
#     tsg_xtb = out_xtb.get_ts_guess()
#     tsg_dft = out_dft.get_ts_guess()
    
    
#     tsg_xtb.update_tc_parameters(ref)
#     tsg_dft.update_tc_parameters(ref)
    
#     freqs_xtb.append(tsg_xtb.tc_freq_calculation())
#     freqs_dft.append(tsg_dft.tc_freq_calculation())

# +
# n_imag_xtb = [sum(np.array(freq) < 0) for freq in freqs_xtb]
# n_imag_dft = [sum(np.array(freq) < 0) for freq in freqs_dft]

# +
# xs, heights = np.unique(n_imag_xtb, return_counts=True)

# +
# xs_2, heights_2 = np.unique(n_imag_dft, return_counts=True)

# +
# plt.bar(xs_2, heights_2)
# plt.bar(xs, heights, alpha=.8)
# # plt.hist(n_imag_dft)
# -

len(dists)

n_single

# +
# sorted([(names[i], d )for i, d in enumerate(dists)], key=lambda x:x[1])

# +
# sorted([(names[i], d )for i, d in enumerate(en_dists)], key=lambda x:x[1])
# -

len(multis)

ind = 4
print(multis[ind][1])
c = multis[ind][0]
c.plot_chain()

c.to_trajectory()

# +
fs = 18
output = plt.hist(dists, label='ASNEB-DFT(tol=0.01)')
plt.hist(dists_xtb, label='ASNEB-XTB(tol=0.001)', alpha=.8, bins=output[1])
# plt.hist(dists_xtbES, label='ASNEB-XTB(tol=0.01)', alpha=.5, hatch='x', bins=output[1])
# plt.hist(dists_gi, label='GI', alpha=.5, hatch='/', bins=output[1])
plt.hist(dists_asnebbundle, label='ASNEB-XTB-Bundle', alpha=.5, hatch='/', bins=output[1])

plt.ylabel("Number of reactions", fontsize=fs)
plt.xlabel("qRMSD between TS_guess and TS", fontsize=fs)
plt.xticks(fontsize=fs-3)
plt.yticks(fontsize=fs-3)
plt.legend(fontsize=f1s-3)
plt.show()
# -



len(multis)

# +
fs = 18
bn=70
output = plt.hist(en_dists, label='ASNEB-DFT(tol=0.01)',bins=bn)
plt.hist(en_dists_xtb,alpha=.9, label='ASNEB-XTB(tol=0.001)', bins=output[1])
# plt.hist(en_dists_xtbES,alpha=.5, label='ASNEB-XTB(tol=0.01)', hatch='x')
plt.hist(en_dists_asnebbundle,alpha=.5, label='ASNEB-XTB-Bundle', hatch='/', bins=output[1])

plt.ylabel("Number of reactions", fontsize=fs)
plt.xlabel("|(E$_{TSguess}$ - E$_{TS}$)| kcal/mol", fontsize=fs)
plt.xticks(fontsize=fs-3)
plt.yticks(fontsize=fs-3)
plt.legend(fontsize=fs)
plt.show()


# -

def plot_results_list_names(tuple_inputs, input_names):
    mult_chain_dists, mult_chain_ES_dists, early_stop_dists = tuple_inputs
    s=10
    fs = 18
    f,ax = plt.subplots(figsize=(2*s,s))
    plt.plot(input_names, mult_chain_dists,'o-', label='ASNEB-XTB(tol=0.001)')
    plt.plot(input_names, early_stop_dists, 'x-', label='NEB-XTB(tol=0.01)')
    # plt.plot(input_names, mult_chain_ES_dists, '*-', label='mult_chain_ES_dists')

    plt.xticks(rotation=90,fontsize=fs-9)
    plt.legend(fontsize=fs)
    plt.ylabel("distance to true TS",fontsize=fs)


def calc_results_for_lists_names(input_names, msma_obj):
    mult_chain_dists = []
    gi_chain_dists = []
    early_stop_dists = []


    for ind, name in enumerate(input_names):
        print(f"sysname: {name}")
        # h = msma.get_relevant_tree(name)
        # n = NEB.read_from_disk(p / name /'react_msmep'/ 'node_0.xyz')
        # n = NEB.read_from_disk(p / name /'react_neb')
        n = NEB.read_from_disk(msma_obj.parent_dir / name /  (str(msma_obj.msmep_root_name)+"_msmep") / 'node_0.xyz')
        # early_stop_chain = get_chain_at_cutoff(n.chain_trajectory, 0.01)
        # gi_chain = n.chain_trajectory[0]
        gi_chain = n.optimized
        out = msma_obj.get_relevant_chain(name)
        sp = msma_obj.get_relevant_saddle_point(name)
        # nebs = msma_obj.get_relevant_leaves_nebs(name)

        # mult_chain_ES = Chain.from_list_of_chains([get_chain_at_cutoff(obj.chain_trajectory) for obj in nebs], ChainInputs())

        
        # print(f"MultiChain TSguess, dist to truth: {msma._distance_to_sp(out, sp)}")
        mult_chain_ts_guess_dist = msma_obj._distance_to_sp(out, sp)
        mult_chain_dists.append(mult_chain_ts_guess_dist)

        gi_chain_ts_guess_dist = msma_obj._distance_to_sp(gi_chain, sp)
        gi_chain_dists.append(gi_chain_ts_guess_dist)

        # early_stop_chain_dist = msma_obj._distance_to_sp(early_stop_chain, sp)
        # early_stop_dists.append(early_stop_chain_dist)
    # return mult_chain_dists, gi_chain_dists, early_stop_dists
    return mult_chain_dists, gi_chain_dists, []


def show_results_for_list_names(input_names, msma_obj):
    mult_chain_dists, mult_chain_ES_dists, early_stop_dists = calc_results_for_lists_names(input_names, msma_obj)
    plot_results_list_names((mult_chain_dists, mult_chain_ES_dists, early_stop_dists),input_names)


def get_chain_at_cutoff(chain_traj, cut=0.01):
    beep = None
    for chain in chain_traj:
        if chain.get_maximum_gperp() <= cut:
            beep = chain
            break
    return beep

# system14 looking sus
# system117 looking sus
# system75 got a new minima not in dataset


mult_chain_dists_dft, gi_chain_dists_dft, early_stop_dists_dft = calc_results_for_lists_names(names, msma_dft)

mult_chain_dists_xtb, gi_chain_dists_xtb, early_stop_dists_xtb = calc_results_for_lists_names(names, msma)

mult_chain_dists_xtbES, gi_chain_dists_xtbES, early_stop_dists_xtbES = calc_results_for_lists_names(names, msma_ES)

# +
s=10
fs = 18
f,ax = plt.subplots(figsize=(2*s,s))
plt.plot(names, en_dists_xtb,'o-', label='ASNEB-XTB(tol=0.001)')
plt.plot(names, en_dists,'o-', label='ASNEB-DFT(tol=0.01)')
# plt.plot(names, gi_chain_dists_xtb, 'x-', label='GI')
# plt.plot(names, gi_chain_dists_xtb, 'x-', label='GI-DFT')
# plt.plot(input_names, mult_chain_ES_dists, '*-', label='mult_chain_ES_dists')

plt.xticks(rotation=90,fontsize=fs-9)
plt.legend(fontsize=fs)
plt.ylabel("|(E$_{TSguess}$ - E$_{TS}$)| kcal/mol",fontsize=fs)

# +
s=10
fs = 18
f,ax = plt.subplots(figsize=(2*s,s))
plt.plot(names, mult_chain_dists_xtb,'o-', label='ASNEB-XTB(tol=0.001)')
plt.plot(names, mult_chain_dists_dft,'o-', label='ASNEB-DFT(tol=0.01)')
# plt.plot(names, gi_chain_dists_xtb, 'x-', label='GI')
# plt.plot(names, gi_chain_dists_xtb, 'x-', label='GI-DFT')
# plt.plot(input_names, mult_chain_ES_dists, '*-', label='mult_chain_ES_dists')

plt.xticks(rotation=90,fontsize=fs-9)
plt.legend(fontsize=fs)
plt.ylabel("distance to true TS",fontsize=fs)
# -

sum(np.array(mult_chain_dists_xtb) < np.array(mult_chain_dists_dft))

sum(np.array(en_dists_xtb) < np.array(en_dists))

len(mult_chain_dists_dft)

sum(np.array(gi_chain_dists_xtb) < np.array(mult_chain_dists_dft))

sum(np.array(gi_chain_dists_xtb) < np.array(mult_chain_dists_xtb))

ind_failed = np.array(mult_chain_dists_xtb) > np.array(mult_chain_dists_dft)

inds_sorted_by_ens = np.argsort(np.array(en_dists_xtb))

inds_sorted_by_dists = np.argsort(np.array(mult_chain_dists_xtb))

inds_sorted_by_deltas = np.argsort(np.array(mult_chain_dists_dft) - np.array(mult_chain_dists_xtb))

sum(np.array(en_dists_xtb) < 10)

# +
s=10
fs = 25
f,ax = plt.subplots(figsize=(2*s,s))
plt.plot(np.array(names)[inds_sorted_by_ens], np.array(en_dists_xtb)[inds_sorted_by_ens],'o-', label='ASNEB-XTB(tol=0.001)')
plt.plot(np.array(names)[inds_sorted_by_ens], np.array(en_dists)[inds_sorted_by_ens],'o-', label='ASNEB-DFT(tol=0.01)')
# plt.plot(names, gi_chain_dists_xtb, 'x-', label='GI-DFT')
# plt.plot(input_names, mult_chain_ES_dists, '*-', label='mult_chain_ES_dists')

plt.xticks(rotation=90,fontsize=fs-14)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.ylabel("|(E$_{TSguess}$ - E$_{TS}$)| kcal/mol",fontsize=fs)

# +
s=10
fs = 25
f,ax = plt.subplots(figsize=(2*s,s))
plt.plot(np.array(names)[inds_sorted_by_dists], np.array(mult_chain_dists_xtb)[inds_sorted_by_dists],'o-', label='ASNEB-XTB(tol=0.001)')
plt.plot(np.array(names)[inds_sorted_by_dists], np.array(mult_chain_dists_dft)[inds_sorted_by_dists],'o-', label='ASNEB-DFT(tol=0.01)')
plt.plot(np.array(names)[inds_sorted_by_dists], np.array(gi_chain_dists_dft)[inds_sorted_by_dists], 'x-', label='GI')
# plt.plot(names, gi_chain_dists_xtb, 'x-', label='GI-DFT')
# plt.plot(input_names, mult_chain_ES_dists, '*-', label='mult_chain_ES_dists')

plt.xticks(rotation=90,fontsize=fs-14)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.ylabel("distance to true TS",fontsize=fs)

# +
s=10
fs = 25
f,ax = plt.subplots(figsize=(2*s,s))
plt.plot(np.array(names)[inds_sorted_by_deltas], np.array(mult_chain_dists_xtb)[inds_sorted_by_deltas],'o-', label='ASNEB-XTB(tol=0.001)')
plt.plot(np.array(names)[inds_sorted_by_deltas], np.array(mult_chain_dists_dft)[inds_sorted_by_deltas],'o-', label='ASNEB-DFT(tol=0.01)')

plt.xticks(rotation=90,fontsize=fs-14)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.ylabel("distance to true TS",fontsize=fs)
# -

len(multis)

# +
s=10
fs = 25
f,ax = plt.subplots(figsize=(2*s,s))
plt.plot(np.array(names)[ind_failed], np.array(mult_chain_dists_xtb)[ind_failed],'o-', label='ASNEB-XTB(tol=0.001)')
plt.plot(np.array(names)[ind_failed], np.array(mult_chain_dists_dft)[ind_failed],'o-', label='ASNEB-DFT(tol=0.01)')
plt.plot(np.array(names)[ind_failed], np.array(gi_chain_dists_dft)[ind_failed], 'x-', label='GI')
# plt.plot(names, gi_chain_dists_xtb, 'x-', label='GI-DFT')
# plt.plot(input_names, mult_chain_ES_dists, '*-', label='mult_chain_ES_dists')

plt.xticks(rotation=90,fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.ylabel("distance to true TS",fontsize=fs)
# -

gi_chain_dists_dft[-2]

arr = np.array(mult_chain_dists) - np.array(early_stop_dists) 

interesting_names = np.array(names)[arr > 0]
interesting_ens = arr[arr > 0]

hmm_inds = np.argsort(interesting_ens)

show_results_for_list_names(np.array(names)[inds_sorted_by_dists], msma_ES)

plot_results_list_names((mult_chain_dists[inds_sorted_by_dists], gi_chain_dists[inds_sorted_by_dists], early_stop_dists[inds_sorted_by_dists]), np.array(names)[inds_sorted_by_dists])

show_results_for_list_names(np.array(names)[inds_not_really_good], msma)

show_results_for_list_names(np.array(names)[inds_really_good], msma_ES)

msma._distance_to_sp(h.data.optimized, sp)

msma._distance_to_sp(h.ordered_leaves[0].data.optimized, sp)

# +
s=10
f,ax = plt.subplots(figsize=(2*s,s))
plt.plot(names, mult_chain_dists,'o-', label='mult_chain_dists')
# plt.plot(names, early_stop_dists, 'x-', label='early_stop_dists')
plt.plot(names, gi_chain_dists, '*-', color='green',label='gi_chain_dists')

plt.xticks(rotation=90)
plt.legend()
plt.ylabel("distance to true TS")
# -


ind=5
print(multis[ind][1])
multis[ind][0].to_trajectory().draw();

# +

print(multis[0][1])
multis[0][0].plot_chain()
# -

msma._distance_to_sp(n.initial_chain, sp), msma._distance_to_sp(out_dft, sp), msma._distance_to_sp(out, sp)



n.initial_chain.get_ts_guess()

sp

# +
name = 'system17'
msma_obj = msma
# h = msma_obj.get_relevant_tree(name)
n = NEB.read_from_disk(p / name / 'react_msmep' / 'node_0.xyz')
reac = TDStructure.from_xyz(f'/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/{name}/react.xyz')
prod = TDStructure.from_xyz(f'/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/{name}/prod.xyz')



out = msma_obj.get_relevant_chain(name)
sp = msma_obj.get_relevant_saddle_point(name)
nebs = msma_obj.get_relevant_leaves_nebs(name)

out_dft = msma_dft.get_relevant_chain(name)
# nebs_dft = msma_dft.get_relevant_leaves_nebs(name)


# +
ref = TDStructure.from_smiles("O")

ref.tc_model_method = 'B3LYP-D3BJ'
# ref.tc_model_method = 'b3lyp'
ref.tc_model_basis = 'def2-svp'
ref.tc_kwds = {'reference':'uks'}
# -

prod = TDStructure.from_xyz(f'/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system24/prod.xyz')

traj = out_dft.to_trajectory()

traj.update_tc_parameters(ref)

traj[-1].energy_tc()

opt = traj[11].tc_geom_optimization()

opt

opt

opt.energy_tc()

# +
# traj
# -

traj[9].energy_tc()

sp.update_tc_parameters(ref)

sp.energy_tc()

huh.update_tc_parameters(ref)

huh.energy_xtb()

out_dft[0].energy

out_dft.plot_chain()

out.plot_chain()

out_dft.plot_chain()

out_dft.to_trajectory()

sp

tsg = out.get_ts_guess()

tsg.update_tc_parameters(ref)

ts = tsg.tc_geom_optimization('ts')

ts

sp

out.to_trajectory()[0]

out.get_ts_guess()

sp

out_dft.get_ts_guess()

sp

# +
print(f"dft {msma._distance_to_sp(out_dft, sp)}, xtb {msma._distance_to_sp(out, sp)}")


# tsg_dft = out_dft.get_ts_guess()
# tsg_dft.tc_model_method = 'B3LYP-D3BJ'
# tsg_dft.tc_model_basis = 'def2-svp'
# tsg_dft.tc_kwds = {'reference':'uks'}

# tsg_xtb = out.get_ts_guess()
# tsg_xtb.update_tc_parameters(tsg_dft)
# -

tsg = out.get_ts_guess()
tsg.tc_model_method = 'B3LYP-D3BJ'
tsg.tc_model_basis = 'def2-svp'
tsg.tc_kwds = {'reference':'uks'}

ts = tsg.tc_geom_optimization('ts')

ts

sp


# # Not Benchmark Dataset

def plot_hist(data_list, label, **kwargs):
    fs = 18
    plt.hist([x[1] for x in data_list], **kwargs)
    plt.ylabel("Count",fontsize=fs)
    plt.xlabel(label,fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.show()


def get_eA_chain(chain):
    eA = max(chain.energies_kcalmol)
    return eA


def get_eA_leaf(leaf):
    chain = leaf.data.optimized
    eA = max(chain.energies_kcalmol)
    return eA


def descending_order(data_list):
    return sorted(data_list, key=lambda x: x[1], reverse=True)


# +
from openbabel import pybel

ob_log_handler = pybel.ob.OBMessageHandler()
pybel.ob.obErrorLog.StopLogging()
# -


succ=['Semmler-Wolff-Reaction', 'Grob-Fragmentation-X-Fluorine', 'Elimination-Lg-Alkoxide', 'Elimination-Alkene-Lg-Bromine', 'Elimination-with-Alkyl-Shift-Lg-Chlorine', 'Aza-Grob-Fragmentation-X-Bromine', 'Ramberg-Backlund-Reaction-Bromine', 'Elimination-Alkene-Lg-Iodine', 'Aza-Grob-Fragmentation-X-Chlorine', 'Decarboxylation-CG-Nitrite', 'Amadori-Rearrangement', 'Rupe-Rearrangement', 'Grob-Fragmentation-X-Chlorine', 'Elimination-Alkene-Lg-Sulfonate', 'Elimination-with-Alkyl-Shift-Lg-Hydroxyl', 'Semi-Pinacol-Rearrangement-Nu-Iodine', 'Grob-Fragmentation-X-Sulfonate', 'Oxazole-Synthesis-EWG-Carbonyl-EWG3-Nitrile', 'Oxazole-Synthesis', 'Fries-Rearrangement-para', 'Buchner-Ring-Expansion-O', 'Chan-Rearrangement', 'Irreversable-Azo-Cope-Rearrangement', 'Claisen-Rearrangement', 'Paal-Knorr-Furan-Synthesis', 'Chapman-Rearrangement', 'Ramberg-Backlund-Reaction-Chlorine', 'Overman-Rearrangement-Pt2', 'Hemi-Acetal-Degradation', 'Vinylcyclopropane-Rearrangement', 'Sulfanyl-anol-Degradation', 'Cyclopropanation-Part-2', 'Oxindole-Synthesis-X-Fluorine', 'Curtius-Rearrangement', 'Oxazole-Synthesis-EWG-Nitrite-EWG3-Nitrile', 'Elimination-Lg-Iodine', 'Aza-Vinylcyclopropane-Rearrangement', 'Elimination-Acyl-Chlorine', 'Imine-Tautomerization-EWG-Phosphonate-EWG3-Nitrile', 'Elimination-Lg-Chlorine', 'Semi-Pinacol-Rearrangement-Nu-Chlorine', 'Elimination-Lg-Hydroxyl', 'Aza-Grob-Fragmentation-X-Sulfonate', 'Elimination-Acyl-Iodine', 'Imine-Tautomerization-EWG-Nitrite-EWG3-Nitrile', 'Imine-Tautomerization-EWG-Carbonyl-EWG3-Nitrile', 'Elimination-Acyl-Sulfonate', 'Elimination-with-Hydride-Shift-Lg-Iodine', 'Elimination-Alkene-Lg-Chlorine', 'Semi-Pinacol-Rearrangement-Nu-Sulfonate', 'Thiocarbamate-Resonance', 'Elimination-with-Hydride-Shift-Lg-Chlorine', 'Meisenheimer-Rearrangement', 'Imine-Tautomerization-EWG-Carboxyl-EWG3-Nitrile', 'Mumm-Rearrangement', 'Claisen-Rearrangement-Aromatic', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Cl', '2-Sulfanyl-anol-Degradation', 'Meisenheimer-Rearrangement-Conjugated', 'Elimination-with-Hydride-Shift-Lg-Bromine', 'Azaindole-Synthesis', 'Oxy-Cope-Rearrangement', 'Beckmann-Rearrangement', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Br', 'Decarboxylation-CG-Carboxyl', 'Benzimidazolone-Synthesis-1-X-Bromine', 'Benzimidazolone-Synthesis-1-X-Iodine', 'Ramberg-Backlund-Reaction-Fluorine', 'Elimination-Acyl-Bromine', 'Oxazole-Synthesis-EWG-Phosphonate-EWG3-Nitrile', 'Decarboxylation-Carbamic-Acid', 'Grob-Fragmentation-X-Iodine', 'Imine-Tautomerization-EWG-Nitrile-EWG3-Nitrile', 'Grob-Fragmentation-X-Bromine', 'Elimination-To-Form-Cyclopropanone-Chlorine', 'Enolate-Claisen-Rearrangement', 'Elimination-with-Alkyl-Shift-Lg-Sulfonate', 'Petasis-Ferrier-Rearrangement', 'Buchner-Ring-Expansion-C', 'Madelung-Indole-Synthesis', 'Thio-Claisen-Rearrangement', 'Semi-Pinacol-Rearrangement-Alkene', 'Decarboxylation-CG-Carbonyl', 'Semi-Pinacol-Rearrangement-Nu-Bromine', 'Robinson-Gabriel-Synthesis', 'Newman-Kwart-Rearrangement', 'Azo-Vinylcyclopropane-Rearrangement', 'Buchner-Ring-Expansion-N', 'Elimination-Lg-Bromine', 'Lobry-de-Bruyn-Van-Ekenstein-Transformation', 'Oxindole-Synthesis-X-Bromine', 'Electrocyclic-Ring-Opening', 'Ester-Pyrolysis', 'Knorr-Quinoline-Synthesis', 'Lossen-Rearrangement', 'Pinacol-Rearrangement', 'Piancatelli-Rearrangement', 'Elimination-Water-Imine', 'Skraup-Quinoline-Synthesis', 'Wittig']#[]
failed=['Elimination-with-Hydride-Shift-Lg-Sulfonate', 'Fries-Rearrangement-ortho', 'Oxazole-Synthesis-EWG-Nitrile-EWG3-Nitrile', 'Indole-Synthesis-1', 'Elimination-To-Form-Cyclopropanone-Sulfonate', 'Oxindole-Synthesis-X-Iodine', 'Nazarov-Cyclization', 'Baker-Venkataraman-Rearrangement', 'Elimination-with-Alkyl-Shift-Lg-Iodine', 'Elimination-with-Alkyl-Shift-Lg-Bromine', 'Oxazole-Synthesis-EWG-Alkane-EWG3-Nitrile', 'Meyer-Schuster-Rearrangement', 'Ramberg-Backlund-Reaction-Iodine', 'Aza-Grob-Fragmentation-X-Iodine', 'Oxindole-Synthesis-X-Chlorine', 'Elimination-Amine-Imine', 'Camps-Quinoline-Synthesis', 'Oxazole-Synthesis-EWG-Carboxyl-EWG3-Nitrile', 'Elimination-with-Hydride-Shift-Lg-Hydroxyl', 'Aza-Grob-Fragmentation-X-Fluorine', 'Indole-Synthesis-Hemetsberger-Knittel', 'Bradsher-Cyclization-2', 'Elimination-To-Form-Cyclopropanone-Bromine', 'Bradsher-Cyclization-1', 'Elimination-To-Form-Cyclopropanone-Iodine', 'Bamford-Stevens-Reaction', '1-2-Amide-Phthalamide-Synthesis', 'Elimination-Lg-Sulfonate', 'Oxa-Vinylcyclopropane-Rearrangement', 'Bamberger-Rearrangement', 'Wittig_DFT'] #[]

# +
all_max_barriers = []
all_n_steps = []
peak_barriers = []
all_n_atoms = []



tol = 0.001*BOHR_TO_ANGSTROMS

all_ts_guesses = []

# for i, rn in enumerate(rns):
for i, rn in enumerate(succ):
    # try:
    cs = get_relevant_leaves(rn)
    # if all([x.get_maximum_grad_magnitude() <= tol for x in cs]):
    eAs = [get_eA_chain(c) for c in cs]
    ts_guesses = [c.get_ts_guess() for c in cs]
    for tsg in ts_guesses:
        tsg.tc_model_method='wb97xd3'
        tsg.tc_model_basis='def2-svp'
        # tsg.tc_model_method='gfn2xtb'
        # tsg.tc_model_basis='gfn2xtb'
        
        
    all_ts_guesses.extend(ts_guesses)
    
    
    c = Chain.from_list_of_chains(cs, ChainInputs())
    max_delta_en = (max(c.energies) - c.energies[0])*627.5
    
    
    maximum_barrier = max(eAs)
    peak_barrier = max_delta_en
    n_steps = len(eAs)
    n_atoms = c.n_atoms

    all_max_barriers.append((i, maximum_barrier))
    peak_barriers.append((i, peak_barrier))
    all_n_steps.append((i, n_steps))
    all_n_atoms.append((i, n_atoms))
    
        # succ.append(rn)
    # else:
    #     print(f"{rn} has not converged")
        # failed.append(rn)
            
    # except:
        # failed.append(rn)
# -

import pandas as pd





df = pd.DataFrame()
df['name'] = succ
df['max_barrier'] = [x[1] for x in all_max_barriers]
df['peak_barrier'] = [x[1] for x in peak_barriers]
df['n_atoms'] = [x[1] for x in all_n_atoms]
df['n_steps'] = [x[1] for x in all_n_steps]

all_ts_guesses_inputs = [td._prepare_input('freq') for td in all_ts_guesses]

# all_ts_guesses_results = [ client.compute(inp, engine='bigchem') for inp in all_ts_guesses_inputs]
all_ts_guesses_results = [ ]
ts_guesses_that_failed = []
for td in all_ts_guesses:
    try:
        all_ts_guesses_results.append(td.tc_freq_calculation() )
    except:
        ts_guesses_that_failed.append(td)
        continue

dist_vals = [(i, sum(np.array(val) < 0)) for (i, val) in enumerate(all_ts_guesses_results)]

delta_vals = [(i, np.abs(np.abs(val[0]) - np.abs(val[1]))) for (i, val) in enumerate(all_ts_guesses_results)]

# +
fs = 18
plt.ylabel("Frequency",fontsize=fs)
plt.xlabel("N negative frequencies\nat XTB level",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
xs, heights = np.unique([x[1] for x in dist_vals], return_counts=True)
plt.bar(xs, heights)
plt.savefig("/home/jdep/T3D_data/msmep_draft/figures/ts_guess_freq_dist_dft.svg")
# plt.savefig("/home/jdep/T3D_data/msmep_draft/figures/ts_guess_freq_dist_xtb.svg")

plt.show()

# +
plt.ylabel("Frequency",fontsize=fs)
plt.xlabel("abs(∆) between most negative and\n secondmost negative freq",fontsize=fs)
plt.hist([x[1] for x in delta_vals])
plt.xticks(fontsize=fs-5)
plt.yticks(fontsize=fs)
# xs, heights = np.unique([x[1] for x in delta_vals], return_counts=True)
# plt.bar(xs, heights)
# plt.plot(xs, heights, 'o')
plt.savefig("/home/jdep/T3D_data/msmep_draft/figures/ts_guess_freq_deltas_dft.svg")
# plt.savefig("/home/jdep/T3D_data/msmep_draft/figures/ts_guess_freq_deltas_xtb.svg")

plt.show()
# -

client = CCClient()

all_ts_guesses_opts_batch1 = client.compute(all_ts_guesses_inputs[:100], engine='bigchem')

all_ts_guesses_opts_batch1_results = all_ts_guesses_opts_batch1.get()

# +
# all_ts_guesses_opts_batch2 = client.compute(all_ts_guesses_inputs[100:], engine='bigchem')
# -

# look_at_me = rns[19]
look_at_me = succ[1]
look_at_me

# cs = get_relevant_leaves(look_at_me)
cs = get_relevant_leaves('Claisen-Rearrangement')

c = Chain.from_list_of_chains(cs,ChainInputs())

c.plot_chain()


def do_pseudo_irc(ts_structure: TDStructure):
    
    freqs = ts_structure.tc_freq_calculation()
    nma = ts_structure.tc_nma_calculation()
    assert sum(np.array(freqs) < 0) == 1, "Not a first order saddle point."
    
    direction = np.array(nma[0]).reshape(ts_structure.coords.shape)
    dr = .1
    
    td_disp_plus = ts_structure.update_coords(ts_structure.coords + dr*direction)
    td_disp_minus = ts_structure.update_coords(ts_structure.coords - dr*direction)
    
    inps = [td_disp_plus._prepare_input("opt"), td_disp_minus._prepare_input("opt")]
    out = ts_structure.tc_client.compute_procedure(inps, 'geometric')
    results = out.get()
    td_plus = TDStructure.from_cc_result(results[0])
    td_minus = TDStructure.from_cc_result(results[1])
    
    return Trajectory([td_minus, ts_structure, td_plus])


# +
def create_dft_rxn_profile_inputs(chain):
    client = CCClient()
    method = 'wb97xd3'
    basis = 'def2-svp'
    
    r = chain[0].tdstructure
    ts_guess = get_ts_guess(chain)
    p = chain[-1].tdstructure
    
    for td in [r, ts_guess, p]:
        td.tc_model_basis = basis
        td.tc_model_method = method
    
    inp_objs = [r._prepare_input(method='opt'), ts_guess._prepare_input(method='ts'), p._prepare_input(method='opt')]
    inp_opts = client.compute_procedure(inp_objs, procedure='geometric')
    return inp_objs
#     inp_results = inp_opts.get()
    
#     if all([r.success for r in inp_results]):
#         return Trajectory([td_from_cc_result(r) for r in inp_results])
#     else:
#         print("Opt failed... returning objects")
#         return inp_results


# -

client = CCClient()

all_failures = []
for rn in succ:
    cs = get_relevant_leaves(rn)
    data_dir = ca.out_folder / rn
    
    if len(list(data_dir.glob("dft_*"))) == 0:
    
        print(f"Doing {rn}")
        inps = []
        for x in cs:
            inps.extend(create_dft_rxn_profile_inputs(x))


        client = CCClient()
        inp_opts = client.compute_procedure(inps, procedure='geometric')

        results = inp_opts.get()

        start_ind = 0
        if len(results) > 3:
            for end in range(3, len(results), 3):
                try:
                    t = Trajectory([TDStructure.from_cc_result(r) for r in results[start_ind:end]])
                    t.write_trajectory(ca.out_folder / rn / f'dft_chain_{int((start_ind / 3))}.xyz')
                    start_ind+=3
                except:
                    print(f"---{rn} had a failure")
                    all_failures.append(rn)
        else:
            try:
                t = Trajectory([TDStructure.from_cc_result(r) for r in results])
                t.write_trajectory(ca.out_folder / rn / f'dft_chain_0.xyz')
            except:
                print(f"---{rn} had a failure")
                all_failures.append(rn)
                
                
    elif len(list(data_dir.glob("dft_*"))) < len(cs):
        print(f"Completing {rn}")
        inps = []
        
        inps.extend(create_dft_rxn_profile_inputs(cs[-1]))


        client = CCClient()
        inp_opts = client.compute_procedure(inps, procedure='geometric')

        results = inp_opts.get()
    
        start_ind = 0
        if len(results) > 3:
            for end in range(3, len(results), 3):
                try:
                    t = Trajectory([TDStructure.from_cc_result(r) for r in results[start_ind:end]])
                    t.write_trajectory(ca.out_folder / rn / f'dft_chain_{int((start_ind / 3))}.xyz')
                    start_ind+=3
                except:
                    print(f"---{rn} had a failure")
                    all_failures.append(rn)
        else:
            try:
                t = Trajectory([TDStructure.from_cc_result(r) for r in results])
                t.write_trajectory(ca.out_folder / rn / f'dft_chain_{len(cs)-1}.xyz')
            except:
                print(f"---{rn} had a failure")
                all_failures.append(rn)

all_failures

# ## Stats

# +
# N.B.: The -1 comes from the fact that Wittig_DFT is still included in the dataset for whatever reason. It should not be. 


print(f"Tot N reactions: {len(rns)-1}")
print(f"\tTot N converged: {len(succ)}")
print(f"\tTot N unconverged: {len(failed)-1}")
print(f"Convergence percentage: {round(len(succ) / (len(rns)-1), 3)*100}%")


# -

def get_error_message(rn):
    data_dir = ca.out_folder / rn
    out_fp = data_dir / 'out.txt'
    datum = open(out_fp).read().splitlines()
    return datum


# +
reasons = []
wtfs = []

for f in failed[:-1]: # ignoring 'Wittig_DFT'
    datum = get_error_message(f)
    # print(datum[-1])
    if 'AttributeError' in datum[-1] or "TypeError: 'NoneType'" in datum[-1]:
        reasons.append("Insufficient\nOptimization Steps")
    elif 'step' in datum[-1]:
        reasons.append("Insufficient\nTime")
    elif 'scf' in datum[-1]:
        reasons.append("Electronic\nStructure Error")
    else:
        reasons.append("Small bug\nto be fixed\nasap")
        wtfs.append(f)
# -

f, ax = plt.subplots()
plt.hist(reasons)
