# +
from pathlib import Path
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
import numpy as np
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.Node2d import Node2D_Flower, Node2D, Node2D_Zero, Node2D_2
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Node3D import Node3D

from neb_dynamics.Node3D_gfn1xtb import Node3D_gfn1xtb
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS

from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.ChainBiaser import ChainBiaser

from itertools import product
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# +
def plot_chain(chain,linestyle='--',ax=None, marker='o',**kwds):
    if ax:
        ax.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)
    else:
        plt.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)
        
        
        

def plot_coordinates(coords,linestyle='--',ax=None, marker='o',**kwds):
    if ax:
        ax.plot(coords[:,0],coords[:,1],linestyle=linestyle,marker=marker,**kwds)
    else:
        plt.plot(coords[:,0],coords[:,1],linestyle=linestyle,marker=marker,**kwds)
        


# -

NIMAGES = 15

# +
ind = 1

the_noise = [-.5,.5]
# the_noise = [-1,1]

noises_bool = [
    True,
    False,
    False

]




start_points = [
     [-2.59807434, -1.499999  ],
    [-3.77931026, -3.283186  ],
    [-1, 1]#[-1.05565696,  1.01107738] #
]

end_points = [
    [2.5980755 , 1.49999912],
    [2.99999996, 1.99999999], # --> interesting other endpoint [ 3.58442836, -1.84812646]
    [1, 1]#[1.05565701, -1.01107741]#,

]
tols = [
    0.1,
    0.05,
    0.1

]

step_sizes = [
    1,
    1,
    .1
]


k_values = [
    1,#.05,
    50,
    1

]



nodes = [Node2D_Flower, Node2D, Node2D_2]
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
n_ref = NEB(initial_chain=chain_ref,parameters=nbi)
n_ref.optimize_chain()


# -

def path_work(chain_obj):
    grads = np.array([np.abs(n.gradient) for n in chain_obj[1:-1]])
    
    # tangents = np.array(chain_obj.unit_tangents)
    tangents = chain_obj.coordinates[1:-1] - chain_obj.coordinates[:-2] 
    
    work = sum(
        np.linalg.norm(g)*np.linalg.norm(t) for g,t in zip(grads, tangents)
    )
    # work = sum(
    #     np.dot(g, t) for g,t in zip(grads, tangents)
    # )

    return work


def path_work2(chain_obj):
    grads = np.array([np.abs(n.gradient) for n in chain_obj[1:-1]])
    
    # tangents = np.array(chain_obj.unit_tangents)
    tangents = chain_obj.coordinates[2:] - chain_obj.coordinates[1:-1] 
    
    work = sum(
        np.linalg.norm(g)*np.linalg.norm(t) for g,t in zip(grads, tangents)
    )
    # work = sum(
    #     np.dot(g, t) for g,t in zip(grads, tangents)
    # )

    return work


#### asneb
gii = GIInputs(nimages=NIMAGES)
nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_chain_rms_thre=0.002, early_stop_force_thre=1, node_freezing=False)
# nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=3, node_freezing=False)
m = MSMEP(neb_inputs=nbi_msmep,chain_inputs=cni_ref, gi_inputs=gii)
history_ref, out_chain_ref = m.find_mep_multistep(chain_ref)

# +
c_to_plot = chain_ref
# bias_chain = n_ref.optimized
bias_chains = [out_chain_ref]
# bias_chains = [out_chain_ref, out_chain]



cb = ChainBiaser(reference_chains=bias_chains,
                amplitude=1, distance_func='simp_frechet')
chain_grad = cb.grad_chain_bias(c_to_plot)
# -

foobar_node = Node2D_Zero([0,0])
tans = c_to_plot.unit_tangents
tans.insert(0, np.array([0,0]))
tans.append(np.array([0,0]))
grads = cb.grad_chain_bias(c_to_plot)
proj_grads = np.array([foobar_node.get_nudged_pe_grad(tan, grad) for tan,grad in zip(tans, grads)])

# +
#### get energies for countourplot
gridsize = 100
min_val = -4
max_val = 4
# min_val = -2
# max_val = 2
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


# cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

reference = bias_chains[0]



plot_chain(reference, color='white')

plot_chain(c_to_plot,color='yellow')
plot_chain(history_ref.data.optimized, color='blue')
# for ind in range(len(c_to_plot)):
#     locs = c_to_plot
#     dx,dy = chain_grad[ind] / np.linalg.norm(chain_grad[ind]) 
#     plt.arrow(locs[ind].coords[0], locs[ind].coords[1], dx=-1*dx, dy=-1*dy, width=.1)

plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

plt.show()
# -

# # Testing it out

# +
ss = 0.01
amp = 1000
sig = 1
distance_func = 'simp_frechet'

cb = ChainBiaser(reference_chains=bias_chains, amplitude=amp, sigma=sig, distance_func=distance_func)
cni = ChainInputs(step_size=.1,min_step_size=0.001, node_class=node_to_use, k=5, delta_k=0, do_parallel=False, 
                 do_chain_biasing=True, cb=cb)
 


init_chain = Chain(n_ref.chain_trajectory[0].nodes, parameters=cni)

for i, node in enumerate(init_chain):
    init_chain.nodes[i] = node_to_use(pair_of_coordinates=node.coords)


# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)


# cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

for reference in bias_chains:
    plot_chain(reference, color='white', label='reference')
plot_chain(init_chain)
plot_chain(n_ref.initial_chain,color='yellow', label='initial guess')

# plot_chain(n.chain_trajectory[-1], color='skyblue', label='biased')

plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.show()
# +
#### asneb
gii = GIInputs(nimages=nimages)
nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=True, early_stop_chain_rms_thre=0.0002, early_stop_force_thre=1, node_freezing=False, early_stop_still_steps_thre=100)

cni = ChainInputs(step_size=.1,min_step_size=0.001, node_class=node_to_use, k=5, delta_k=0, do_parallel=False, use_geodesic_interpolation=False )
cni.do_chain_biasing = True
cni.cb = cb 

init_chain = Chain(n_ref.chain_trajectory[0].nodes, parameters=cni)

for i, node in enumerate(init_chain):
    init_chain.nodes[i] = node_to_use(pair_of_coordinates=node.coords)
    


m = MSMEP(neb_inputs=nbi_msmep,chain_inputs=cni, gi_inputs=gii)
history, out_chain = m.find_mep_multistep(init_chain)
# -

relaxed_chains = []
for biased_out_leaf in history.ordered_leaves:
    biased_out_chain = biased_out_leaf.data.optimized
    init_copy = biased_out_chain.copy()
    init_copy.parameters = cni_ref
    
    n_relax = NEB(initial_chain=init_copy, parameters=nbi)
    n_relax.optimize_chain()
    relaxed_chains.append(n_relax.optimized)

relaxed_out_chain = Chain.from_list_of_chains(relaxed_chains, cni)

# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)


# cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

reference = bias_chains[0]



plot_chain(reference, color='white', label='reference AS-NEB')
# plot_chain(init_chain)
plot_chain(n_ref.initial_chain,color='yellow', label='initial guess')

# plot_chain(n_ref.chain_trajectory[-1], color='blue', label='SD NEB')
plot_chain(history.data.optimized, color='red')
# plot_chain(out_chain, color='red', label='biased AS-NEB')
plot_chain(relaxed_out_chain, color='orange', label='relaxed biased AS-NEB')


plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.legend(fontsize=fs, bbox_to_anchor=(1.8,1), shadow=True, facecolor='lightgray')
plt.show()

# +
n_dig = 1

plt.plot(out_chain.integrated_path_length, out_chain.energies, 'o-',label=f'work: {round(path_work2(out_chain),n_dig)}', color='red')
plt.plot(relaxed_out_chain.integrated_path_length, relaxed_out_chain.energies, 'o-',label=f'work: {round(path_work2(out_chain),n_dig)}', color='orange')


plt.plot(n_ref.initial_chain.integrated_path_length,  n_ref.initial_chain.energies,'o-', label=f'work: {round(path_work2(n_ref.initial_chain),n_dig)}', color='yellow')

plt.plot(out_chain_ref.integrated_path_length,  out_chain_ref.energies,'o-', label=f'work: {round(path_work2(out_chain_ref),n_dig)}', color='lightgray')


plt.plot(n_ref.chain_trajectory[-1].integrated_path_length,  n_ref.chain_trajectory[-1].energies,'o-', label=f'work: {round(path_work2(n_ref.chain_trajectory[-1]),n_dig)}', color='blue')

plt.legend(loc='upper right')

# -
# # Maybe a Molecular Example

# +
# vetoing this until I implement having multiple chains as bias!
# -


