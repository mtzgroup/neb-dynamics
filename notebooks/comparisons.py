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
# import os
# del os.environ['OE_LICENSE']
# -

from neb_dynamics.Janitor import Janitor

from neb_dynamics.MSMEP import MSMEP
from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

# # ASE

from ase import io
from ase.neb import NEB
from ase.optimize import LBFGS, FIRE, MDMin

from xtb.ase.calculator import XTB
from xtb.interface import Calculator, XTBException
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

init_guess_path = "/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess.xyz"

t = Trajectory.from_xyz(init_guess_path)

start = Node3D_gfn1xtb(t[0])

end = Node3D_gfn1xtb(t[-1])

start_opt = start.do_geometry_optimization()

start_opt.tdstructure.to_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/start_gfn1.xyz")

end_opt.tdstructure.to_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/end_gfn1.xyz")

t = Trajectory([start_opt.tdstructure, end_opt.tdstructure])

gi = t.run_geodesic(nimages=15)

gi.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/initial_guess_gfn1.xyz")

end_opt = end.do_geometry_optimization()

# Read initial and final states:
images = io.read(init_guess_path,index=':')
for image in images:
    image.set_positions(image.get_positions())

neb = NEB(images)

from xtb.ase.calculator import XTB

# +
chg = 0
spinmult = 1


# Set calculators:
for image in images[1:-1]:
    image.calc = XTB(method="GFN2-xTB")
# Optimize:
optimizer = LBFGS(neb, trajectory='A2B.traj',maxstep=.33)
optimizer.run(fmax=0.01,steps=100)
# -

# # DL-Find

# +
import functools

import numpy as np
from libdlfind import dl_find
from libdlfind.callback import (dlf_get_gradient_wrapper,
                                dlf_put_coords_wrapper, make_dlf_get_params)
from xtb.interface import Calculator
from xtb.utils import get_method

from wurlitzer import pipes
    


# Create function to calculate energies and gradients
@dlf_get_gradient_wrapper
def e_g_func(coordinates, iimage, kiter, calculator):
    calculator.update(coordinates)
    results = calculator.singlepoint()
    energy = results.get_energy()
    gradient = results.get_gradient()
    return energy, gradient

# Create function to store results from DL-FIND
@dlf_put_coords_wrapper
def store_results(switch, energy, coordinates, iam, traj_coords, traj_energies):
    traj_coords.append(np.array(coordinates))
    traj_energies.append(energy)
    return

def main():
    # Create hydrogen molecule
    numbers = np.array([1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])  # Coordinates in Bohr

    # Create XTB calculator
    calculator = Calculator(get_method("GFN2-xTB"), numbers, positions)

    # Two lists for storing results
    traj_energies = []
    traj_coordinates = []

    dlf_get_params = make_dlf_get_params(coords=positions)
    dlf_get_gradient = functools.partial(e_g_func, calculator=calculator)
    dlf_put_coords = functools.partial(
        store_results, traj_coords=traj_coordinates, traj_energies=traj_energies
    )
	
    # Run DL-FIND
    with pipes() as (stdout, stderr):
        dl_find(
            nvarin=len(numbers) * 3,
            dlf_get_gradient=dlf_get_gradient,
            dlf_get_params=dlf_get_params,
            dlf_put_coords=dlf_put_coords,
        )
	
    # Print results
    print(f"Number of iterations: {len(traj_energies)}")
    print(f"Finaly energy (a.u.): {traj_energies[-1]}")
    return traj_energies, traj_coordinates


# -

# # Variables

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
ind = 1

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
    k=0.001,
    node_class=node_to_use,
    delta_k=0,
    step_size=1,
    do_parallel=False,
    use_geodesic_interpolation=False,
    min_step_size=0.0001
)
# chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref)
chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref2)

n_ref_long = NEB(initial_chain=chain_ref_long,parameters=nbi)
n_ref_long.optimize_chain()
# -

#### get energies for countourplot
gridsize = 100
min_val = -4
max_val = 4
# min_val = -.05
# max_val = .05
x = np.linspace(start=min_val, stop=max_val, num=gridsize)
y = x.reshape(-1, 1)

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
plot_chain(n_ref.chain_trajectory[-1], c='skyblue',linestyle='-',label=f'NEB({nimages} nodes)')
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
plot_chain2d(n_ref.chain_trajectory[-1], c='skyblue',linestyle='-',label=f'neb({nimages} nodes)')
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

# # Corner Cutting Problem

# +
# nimgs =  [5, 7, 10, 15, 30]
nimgs =  [7, 8, 9, 10, 15]
# nimgs =  [8, 9, 10, 15]
# nimgs = [9]


outputs = []
for nimg in nimgs:

    coords = np.linspace(start_point, end_point, nimg)
    if do_noise:
        coords[1:-1] += the_noise # i.e. good initial guess
    gii = GIInputs(nimages=nimg)
    nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, node_freezing=False)
    chain_ref = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni_ref)

    n_ref = NEB(initial_chain=chain_ref,parameters=nbi )
    n_ref.optimize_chain()
    outputs.append(n_ref)

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, 2*fig),nrows=2)
# x = np.linspace(start=min_val, stop=max_val, num=1000)
# y = x.reshape(-1, 1)

cs = ax[0].contourf(x, x, h_ref,alpha=1)
# cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)


if ind == 0:
    ax[0].scatter([
            -2.59807447, 
            0
                ],  
                [
            1.49999904,
            2.99999996
                ], marker='*',c='yellow',label='minima',s=120)
if ind == 1:
    ax[0].scatter(-2.80511811,  3.13131252, marker='*',c='yellow',label='minima',s=120)



colors = ['skyblue','orange','red','lime', 'sandybrown']
# colors = ['skyblue','orange','skyblue','orange']

start_ind = 0
cut=5
plot_chain(outputs[0].initial_chain,label=f'initial guess',linestyle='--',color='white',marker='',ax=ax[0])
for i, (neb_obj, nimg) in enumerate(zip(outputs[start_ind:cut], nimgs[start_ind:cut])):
# for i, (neb_obj, nimg) in enumerate(zip(outputs, nimgs)):
    plot_chain(neb_obj.optimized,label=f'neb ({nimg} images)',linestyle='-',c=colors[i],marker='o',ax=ax[0])
    ax[1].plot(neb_obj.optimized.integrated_path_length, neb_obj.optimized.energies, 'o-',label=f'neb ({nimg} images)')


chains = [n_ref.initial_chain,
                          n_ref.chain_trajectory[-1],
                          out_chain,
                          n_ref_long.chain_trajectory[-1]]
labels = ['initial chain',
                           'NEB (15 nodes)',
                           'AS-NEB',
                           "NEB (45 nodes)"
                           
                          ]

cutoff = 0

# for i, (neb_obj, label) in enumerate(zip(chains[:cutoff],labels[:cutoff])):
#     plot_chain(neb_obj,label=label,linestyle='-',c=colors[i],marker='o',ax=ax[0])
#     ax[1].plot(neb_obj.integrated_path_length, neb_obj.energies, 'o-',label=label,c=colors[i])

ax[0].legend(fontsize=fs)
ax[1].legend(fontsize=fs)

ax[1].set_ylabel("Energy",fontsize=fs)
ax[1].set_xlabel("Integrated path length",fontsize=fs)

ax[0].tick_params(axis='both', which='major', labelsize=fs)
ax[1].tick_params(axis='both', which='major', labelsize=fs)





plt.show()


# f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
# plt.show()



# +

nopt_steps = [(len(neb_obj.chain_trajectory)-1)*nimg for i, (neb_obj, nimg) in enumerate(zip(outputs, nimgs))]
# -

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
plt.plot(nimgs[start_ind:], nopt_steps[start_ind:],'o-')
plt.ylabel("N gradient calls",fontsize=fs)
plt.xlabel("N nodes",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# # Visualize Wittig

# asneb2 = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/tighter_conv/looser_geom/initial_guess_msmep/"))
# asneb2 = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/tighter_conv/initial_guess_tight_endpoints_msmep/"))
asneb2 = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/fixed_result_looser_k"))

asneb2.output_chain.plot_chain()

# neb_short = NEB.read_from_disk(Path("./neb_short"))
# neb_short = NEB.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/nebd/Wittig/initial_guess_tight_endpoints"))
neb_short = asneb2.data

neb_short.plot_opt_history(do_3d=True)

# neb_long = NEB.read_from_disk(Path("./neb_long_45nodes"))
neb_long = NEB.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/initial_guess_long_aligned_neb.xyz"))

neb_long.plot_opt_history(do_3d=True)

# clean_chain = Chain.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/tighter_conv/initial_guess_tight_endpoints_msmep_clean.xyz/"), ChainInputs())
clean_chain = Chain.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/fixed_result_clean.xyz"), ChainInputs())

# asneb_history = TreeNode.read_from_disk(Path("./wittig_early_stop/"))
# asneb_history = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/initial_guess_msmep/"))
# asneb_history = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/tighter_conv/initial_guess_tight_endpoints_msmep/"))
asneb_history = asneb2

# +
# asneb_history.write_to_disk(Path("./wittig_early_stop/"))

# +
# cleanups = NEB.read_from_disk(Path("./cleanup_neb"))
cleanup_neb0 = NEB.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/tighter_conv/initial_guess_tight_endpoints_cleanups/cleanup_neb_0"))
cleanup_neb1 = NEB.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/tighter_conv/initial_guess_tight_endpoints_cleanups/cleanup_neb_1"))

cleanup_nebs = [cleanup_neb0, cleanup_neb1]
# -

cni = ChainInputs(k=0.1,delta_k=0.00, node_class=Node3D, step_size=1,friction_optimal_gi=True)
nbi = NEBInputs(tol=0.01, # tol means nothing in this case
        grad_thre=0.001,
        rms_grad_thre=0.0005,
        en_thre=0.001,
        v=True, 
        max_steps=4000,
        early_stop_chain_rms_thre=0.00,
        early_stop_force_thre=0.00, 
        early_stop_still_steps_thre=1111100,
        node_freezing=False,



        vv_force_thre=0.0)

# +
m = MSMEP(nbi, cni, GIInputs())

j = Janitor(history_object=asneb2,out_path=Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/tighter_conv/initial_guess_tight_endpoints_cleanups"),
           msmep_object=m)

clean_chain = j._merge_cleanups_and_leaves(cleanup_nebs)
# -

all_grads = [node._cached_gradient for node in neb_short.chain_trajectory[-2].nodes]

# +
RMS_CUT = 0.002
GRAD_CUT = 0.01

max_grads = np.array([np.amax(chain.gradients) for chain in neb_short.chain_trajectory])
# -

c = clean_chain
s = 5
fs = 18
f,ax = plt.subplots(figsize=(2.168*s, s))
plt.plot(c.integrated_path_length, (c.energies-c.energies[0])*627.5,'o-')
plt.ylabel("Energy (kcal/mols)",fontsize=fs)
plt.xlabel("Normalized Path Length",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

# +
chain_rms = np.array(neb_short._calculate_chain_distances())
chain_rms[0] = chain_rms[1]


stop_by_rms = np.argmax(chain_rms < RMS_CUT)
stop_by_grad = np.argmax(max_grads < GRAD_CUT)
# -

plt.plot(max_grads,'o-',label="max|F|",markersize=2)
plt.plot(chain_rms,'^-',label='chain RMS',markersize=2)
plt.axvline(stop_by_rms,c='orange',label=f"rms < {RMS_CUT}")
plt.axvline(stop_by_grad,c='blue', label=f"max|F| < {GRAD_CUT}")
plt.legend()

# +
fig = 5
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(2.62*fig,fig))

plt.plot(neb_long.optimized.integrated_path_length, (neb_long.optimized.energies-clean_chain.energies[0])*627.5,'o-',label=f"NEB ({len(neb_long.optimized)} nodes)")
plt.plot(neb_short.optimized.integrated_path_length, (neb_short.optimized.energies-clean_chain.energies[0])*627.5,'o-',label="NEB (15 nodes)")
plt.plot(clean_chain.integrated_path_length, (clean_chain.energies-clean_chain.energies[0])*627.5,'o-',label="AS-NEB")

plt.yticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.legend(fontsize=fs)
plt.xticks(fontsize=fs)
plt.xlabel("Normalized path length",fontsize=fs)
plt.savefig("/home/jdep/T3D_data/msmep_draft/wittig_comparison_paths_v2.svg")
plt.show()

# +
n_steps_msmep = sum([len(obj.chain_trajectory) -1 for obj in asneb_history.get_optimization_history()])
n_steps_long_neb = len(neb_long.chain_trajectory) -1 

n_grad_msmep = n_steps_msmep*(15-2)
n_grad_long_neb = n_steps_long_neb*(60-2)

# +
fig = 7
min_val = -5.3
max_val = 5.3
fs = 18
f,ax = plt.subplots(figsize=(1*fig,fig))

bars = ax.bar(x=["AS-NEB",f'NEB({len(neb_long.optimized)} nodes)'],
       height=[n_grad_msmep, n_grad_long_neb])

ax.bar_label(bars,fontsize=fs)


plt.yticks(fontsize=fs)
plt.text(.03,.95, f"{round(n_grad_long_neb / n_grad_msmep, 2 )}x improvement",transform=ax.transAxes,fontsize=fs,
        bbox={'visible':True,'fill':False})
plt.ylabel("Number of gradient calls",fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# +
n_steps_orig_neb = len(neb_short.chain_trajectory) - 1 # the last chain in trajectory is the optimized one!

n_steps_msmep = sum([len(obj.chain_trajectory) -1 for obj in asneb_history.get_optimization_history()])
# n_steps_msmep += len(cleanups.chain_trajectory)
n_steps_msmep += 104

n_steps_long_neb = len(neb_long.chain_trajectory) -1 


nimages = len(neb_short.initial_chain)
nimages_long = len(neb_long.initial_chain)


n_grad_orig_neb = n_steps_orig_neb*(nimages-2)
n_grad_msmep = n_steps_msmep*(nimages-2)
n_grad_long_neb = n_steps_long_neb*(nimages_long-2)

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f,ax = plt.subplots(figsize=(1.62*fig,fig))

bars = ax.bar(x=["AS-NEB",f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)'],
       height=[n_grad_msmep, n_grad_orig_neb, n_grad_long_neb])

ax.bar_label(bars,fontsize=fs)


plt.yticks(fontsize=fs)
# plt.ylabel("Number of optimization steps",fontsize=fs)
plt.text(.03,.95, f"% improvement: {round((1 - n_grad_msmep / n_grad_long_neb)* 100, 3)}",transform=ax.transAxes,fontsize=fs,
        bbox={'visible':True,'fill':False})
plt.ylabel("Number of gradient calls",fontsize=fs)
plt.xticks(fontsize=fs)
# plt.savefig("/home/jdep/T3D_data/msmep_draft/wittig_comparison_barplots.svg")
plt.show()
# -
# ### load the Wittig from DL-Find

out_short_dlf = Chain.from_xyz(Path('/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/orca_scr_gfn2/orca_input_gi_MEP_trj.xyz'), ChainInputs())

traj = out_short_dlf.to_trajectory()

td = traj[0]

from openbabel.pybel import Molecule

pb = Molecule(td.molecule_obmol)

pb.conformers

out_long_dlf = Chain.from_xyz(Path('/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/orca_scr_gfn2_long/orca_input_gi_MEP_trj.xyz'), ChainInputs())

out_long_dlf.plot_chain()

minimized_sp2_1 = out_long_dlf[32].do_geometry_optimization()
minimized_sp2_2 = out_long_dlf[34].do_geometry_optimization()

int_1 = out_long_dlf[29].do_geometry_optimization()

minimized_sp2_1.tdstructure

int_1.tdstructure

int_1.is_identical(minimized_sp2_1)

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(1.62*fig,fig))

# plt.plot(neb_long.optimized.integrated_path_length, (neb_long.optimized.energies-clean_chain.energies[0])*627.5,'o-',label="NEB (45 nodes)")
# plt.plot(neb_short.optimized.integrated_path_length, (neb_short.optimized.energies-clean_chain.energies[0])*627.5,'o-',label="NEB (15 nodes)")
plt.plot(clean_chain.integrated_path_length, (clean_chain.energies-clean_chain.energies[0])*627.5,'o-',label="AS-NEB")



plt.plot(out_long_dlf.integrated_path_length, (out_long_dlf.energies-clean_chain.energies[0])*627.5,'o-',label="NEB-ORCA (45 nodes)")
plt.plot(out_short_dlf.integrated_path_length, (out_short_dlf.energies-clean_chain.energies[0])*627.5,'o-',label="NEB-ORCA (15 nodes)")


plt.yticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.legend(fontsize=fs)
plt.xticks(fontsize=fs)
plt.xlabel("Normalized path length",fontsize=fs)
plt.savefig("/home/jdep/T3D_data/msmep_draft/orca_comparison_paths.svg")
plt.show()
# -

data = [
    [n_grad_orig_neb, n_grad_long_neb, n_grad_msmep, 'nebd'],
    [2000*13, 1134*43, None,'orca']
] # cols are neb15, neb45, as-neb

import pandas as pd
import seaborn as sns


df = pd.DataFrame(data, columns=["NEB-15","NEB-45","AS-NEB", 'type'])

melted = pd.melt(df,id_vars='type')

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
# f,ax = plt.subplots(figsize=(1.62*fig,fig))
out = sns.catplot(melted,x='variable',y='value',hue='type',kind='bar')
plt.yticks(fontsize=fs)
plt.text(.03,.95, f"% improvement: {round((1 - n_grad_msmep / (1134*43))* 100, 3)}",transform=out.ax.transAxes,fontsize=fs,
        bbox={'visible':True,'fill':False})
plt.ylabel("Number of gradient calls",fontsize=fs)
plt.xticks(fontsize=fs)
plt.tight_layout()
plt.savefig("/home/jdep/T3D_data/msmep_draft/orca_comparison_barplots.svg")

plt.show()
# -

t2 = Trajectory.from_xyz(Path('/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/orca_scr_gfn2_long/orca_input_MEP_trj.xyz'))

t2.draw()

# +
init = Trajectory.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/orca_scr_gfn2/orca_input_gi_MEP_trj.xyz"))
init2 = Trajectory.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/nebd/Wittig/orca_gfn2_comp/initial_guess_failed.xyz"))


init2_long = Trajectory.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/nebd/Wittig/initial_guess_long_neb.xyz"))
# -

n = NEB.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/nebd/Wittig/initial_guess_long_neb"))

len(n.chain_trajectory)

n.plot_opt_history(do_3d=True)

# +
c = Chain.from_traj(init, ChainInputs(k=0.1,delta_k=0.09, step_size=.33))
c2 = Chain.from_traj(init2, ChainInputs(k=0.1,delta_k=0.09, step_size=.33))

c2_long = Chain.from_traj(init2_long, ChainInputs(k=0.1,delta_k=0.09, step_size=.33))
# -

plt.plot(c.integrated_path_length, (c.energies-c.energies[0])*627.5,'o-',label='dl-find')
plt.plot(c2.integrated_path_length, (c2.energies-c.energies[0])*627.5,'o-',label='nebd')
plt.plot(c2_long.integrated_path_length, (c2_long.energies-c.energies[0])*627.5,'o-',label='nebd (long)')
plt.legend()

n = NEB(initial_chain=c, parameters=NEBInputs(v=True, grad_thre=0.001, rms_grad_thre=0.0005,en_thre=0.001, max_steps=2000))

n.optimize_chain()

plt.plot(t.energies_xtb())

fp = Path("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/orca_scr_gfn2/orca_input_MEP_trj.xyz
cni = ChainInputs()
chain = Chain.from_xyz(fp / 'nebpath.xyz',parameters=cni)

import pandas as pd

df = pd.read_csv(fp / 'nebinfo', sep='  ', skiprows=1, header=None)

df.columns = ['path_len','energies','work']
df["path_len_norm"] = df["path_len"].values / df["path_len"].max()
df["energies_kcal"] = df["energies"]*627.5

df.plot(x='path_len_norm',y='energies_kcal', style='o-')
# plt.plot(neb_long.optimized.integrated_path_length, 
#          (neb_long.optimized.energies-neb_long.optimized.energies[0])*627.5, 'o-')

end = t[-1]

start_opt = start.tc_local_geom_optimization()

end_opt = end.tc_local_geom_optimization()

tr = Trajectory([start_opt, end_opt]).run_geodesic(nimages=15)

tr = Trajectory.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/initial_guess_tc_xtb.xyz")

tr.write_trajectory("./inital_b3lyp_guess.xyz")

chain = Chain.from_traj(tr,parameters=ChainInputs(node_class=Node3D_TC_Local))

n = NEB(initial_chain=chain,parameters=NEBInputs(v=True))

n.optimize_chain()

# # Make Cross MSMEPS

hist = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/tighter_conv/initial_guess_tight_endpoints_msmep/"))

hist.data

inp = hist.data.initial_chain

out_leaves = hist.ordered_leaves

inp_start, inp_end = inp[0], inp[-1]


