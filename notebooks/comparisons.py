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
from itertools import product
import matplotlib.pyplot as plt

from neb_dynamics.MSMEP import MSMEP
from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

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
    do_parallel=False,
    use_geodesic_interpolation=False,
)
gii = GIInputs(nimages=nimages)
nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=0, node_freezing=False)
chain_ref = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni_ref)
# -

n_ref = NEB(initial_chain=chain_ref,parameters=nbi)
n_ref.optimize_chain()

gii = GIInputs(nimages=nimages)
nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_chain_rms_thre=0.002, node_freezing=False)
m = MSMEP(neb_inputs=nbi_msmep,chain_inputs=cni_ref, gi_inputs=gii, root_early_stopping=False)
history, out_chain = m.find_mep_multistep(chain_ref)

# +
nimages_long = len(out_chain)

coords_long = np.linspace(start_point, end_point, nimages_long)
if do_noise:
    coords_long[1:-1] += the_noise # i.e. good initial guess
gii = GIInputs(nimages=nimages_long)
nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=0, node_freezing=False)
# cni_ref2 = ChainInputs(
#     k=1,
#     node_class=node_to_use,
#     delta_k=0,
#     step_size=.1,
#     do_parallel=False,
#     use_geodesic_interpolation=False,
# )
chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref)
# chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref2)

n_ref_long = NEB(initial_chain=chain_ref_long,parameters=nbi)
n_ref_long.optimize_chain()
# -

#### get energies for countourplot
gridsize = 100
# min_val = -5.3
# max_val = 5.3
min_val = -4
max_val = 4
x = np.linspace(start=min_val, stop=max_val, num=gridsize)
y = x.reshape(-1, 1)

h_flat_ref = np.array([node_to_use.en_func_arr(pair) for pair in product(x,x)])
h_ref = h_flat_ref.reshape(gridsize,gridsize).T

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
# x = np.linspace(start=min_val, stop=max_val, num=1000)
# y = x.reshape(-1, 1)

# cs = ax[0].contourf(x, x, h_ref, cmap="Greys",alpha=.8)
cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

plot_chain(n_ref.initial_chain, c='orange',label='initial guess')
plot_chain(n_ref.chain_trajectory[-1], c='skyblue',linestyle='-',label=f'neb({nimages} nodes)')
plot_chain(out_chain, c='red',marker='o',linestyle='-',label='as-neb')
plot_chain(n_ref_long.chain_trajectory[-1], c='yellow',linestyle='-',label=f'neb({nimages_long} nodes)')
plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()
# -

n_ref_long.plot_grad_delta_mag_history()

# +
n_steps_orig_neb = len(n_ref.chain_trajectory)
n_steps_msmep = sum([len(obj.chain_trajectory) for obj in history.get_optimization_history()]) 
n_steps_long_neb = len(n_ref_long.chain_trajectory)

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(1.16*fig,fig))
plt.bar(x=["AS-NEB",f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)'],
       height=[n_steps_msmep, n_steps_orig_neb, n_steps_long_neb])
plt.yticks(fontsize=fs)
plt.ylabel("Number of optimization steps",fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()
# -

# # Corner Cutting Problem

# +
# nimgs =  [5, 7, 10, 15, 30]
# nimgs =  [7, 8, 9, 10, 15]
# nimgs =  [8, 9, 10, 15]
nimgs = [9]


outputs = []
for nimg in nimgs:

    coords = np.linspace(start_point, end_point, nimg)
    if do_noise:
        coords[1:-1] += the_noise # i.e. good initial guess
    gii = GIInputs(nimages=nimg)
    nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, stopping_threshold=0, node_freezing=False)
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
# plot_chain(outputs[0].initial_chain,label=f'initial guess',linestyle='--',color='white',marker='',ax=ax[0])
# for i, (neb_obj, nimg) in enumerate(zip(outputs[start_ind:cut], nimgs[start_ind:cut])):
# for i, (neb_obj, nimg) in enumerate(zip(outputs, nimgs)):
# plot_chain(neb_obj.optimized,label=f'neb ({nimg} images)',linestyle='-',c=colors[i],marker='o',ax=ax[0])
# ax[1].plot(neb_obj.optimized.integrated_path_length, neb_obj.optimized.energies, 'o-',label=f'neb ({nimg} images)')


chains = [n_ref.initial_chain,
                          n_ref.chain_trajectory[-1],
                          out_chain,
                          n_ref_long.chain_trajectory[-1]]
labels = ['initial chain',
                           'NEB (15 nodes)',
                           'AS-NEB',
                           "NEB (45 nodes)"
                           
                          ]

cutoff = 4

for i, (neb_obj, label) in enumerate(zip(chains[:cutoff],labels[:cutoff])):
    plot_chain(neb_obj,label=label,linestyle='-',c=colors[i],marker='o',ax=ax[0])
    ax[1].plot(neb_obj.integrated_path_length, neb_obj.energies, 'o-',label=label,c=colors[i])

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

nopt_steps = [len(neb_obj.chain_trajectory) for i, (neb_obj, nimg) in enumerate(zip(outputs, nimgs))]
# -

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
plt.plot(nimgs[start_ind:], nopt_steps[start_ind:],'o-')
plt.ylabel("N optimization steps",fontsize=fs)
plt.xlabel("N nodes",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()


