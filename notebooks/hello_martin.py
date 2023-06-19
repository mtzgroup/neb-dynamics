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

chain1 = n_ref.optimized.coordinates
chain2 = n_ref.chain_trajectory[41].coordinates


def node_wise_distance(chain1: np.array, chain2: np.array):
    tot_dist = 0
    for n1, n2 in zip(chain1[1:-1], chain2[1:-1]):
        tot_dist += np.linalg.norm(abs(n1 - n2))
    return tot_dist / len(chain1)


def node_wise_distance_avg(chain1: np.array, chain2: np.array):
    tot_dist = 0
    for node1 in chain1[1:-1]:
        distances = []
        for node2 in chain2[1:-1]:
            distances.append(np.linalg.norm(abs(node1-node2)))
        tot_dist+=min(distances)
        
    return tot_dist / len(chain1)


# +
# c2 = n_ref.chain_trajectory[41].coordinates.copy()
# c2[7] += [1,-1]
# c3 = n_ref.initial_chain.coordinates

# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)


# cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

reference = n_ref.optimized

plot_chain(reference, color='white', label=node_wise_distance(reference.coordinates, chain1))
plot_chain(n_ref.initial_chain,color='yellow', label=node_wise_distance(reference.coordinates,n_ref.initial_chain.coordinates))
plot_chain(n_ref.chain_trajectory[41],color='red', label=node_wise_distance(reference.coordinates,n_ref.chain_trajectory[41].coordinates))

plot_chain(n_ref.chain_trajectory[2],color='purple', label=node_wise_distance(reference.coordinates,n_ref.chain_trajectory[2].coordinates))

# plot_coordinates(c2,color='gray', label=node_wise_distance(reference.coordinates,c2))


# plot_chain(n_ref.initial_chain, color='yellow', label=node_wise_distance(reference.coordinates, c3))

plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.legend()

plt.show()


# -

def path_bias(distance, sigma=1, amplitude=1):
    return amplitude*np.exp(-distance**2 / (2*sigma**2))


cs = n_ref.optimized.coordinates


def path_work(chain_obj):
    grads = np.array([np.abs(n.gradient) for n in chain_obj[1:-1]])
    # tangents = np.array(chain_obj.unit_tangents)
    tangents = chain_obj.coordinates[1:-1] - chain_obj.coordinates[:-2]
    
    # work = sum(
    #     np.linalg.norm(g)*np.linalg.norm(t) for g,t in zip(grads, tangents)
    # )
    work = sum(
        np.dot(g, t) for g,t in zip(grads, tangents)
    )

    return work


path_work(n_ref.initial_chain)

path_work(n_ref.optimized)

all_works = []
for chain in n_ref.chain_trajectory:
    all_works.append(path_work(chain))


plt.plot(all_works, 'o-')

# +
# fig = 8
# fs = 18
# f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)


# # cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
# cs = ax.contourf(x, x, h_ref,alpha=1)
# _ = f.colorbar(cs)

# reference = n_ref.optimized


# dist_to_ref0 = node_wise_distance(reference.coordinates, chain1)
# plot_chain(reference, color='white', label=path_bias(dist_to_ref0))


# dist_to_ref1 = node_wise_distance(reference.coordinates,n_ref.chain_trajectory[41].coordinates)
# plot_chain(n_ref.chain_trajectory[41],color='red', label=path_bias(dist_to_ref1))

# dist_to_ref1 = node_wise_distance(reference.coordinates,n_ref.chain_trajectory[0].coordinates)
# plot_chain(n_ref.chain_trajectory[0],color='yellow', label=path_bias(dist_to_ref1))



# plt.yticks(fontsize=fs)
# plt.xticks(fontsize=fs)
# plt.legend()

# plt.show()
# +
### get gradients
# -

chain = n_ref.initial_chain
ref = n_ref.optimized


def chain_bias(chain: Chain, ref_chain: Chain):
    dist_to_chain = node_wise_distance(chain.coordinates, ref_chain.coordinates)
    return path_bias(dist_to_chain)


def grad_chain_bias(chain: Chain, ref_chain: Chain, ind_node=1, dr=.1):

    all_grads = []
    for ind_node in range(len(chain)):


        node = chain[ind_node]


        node_disp_x = node.update_coords(node.coords+[dr, 0])
        fake_chain = chain.copy()
        fake_chain.nodes[ind_node] = node_disp_x 

        grad_x = chain_bias(fake_chain, ref_chain) - chain_bias(chain, ref_chain)

        node_disp_y = node.update_coords(node.coords+[0, dr])
        fake_chain = chain.copy()
        fake_chain.nodes[ind_node] = node_disp_y 

        grad_y = chain_bias(fake_chain, ref_chain) - chain_bias(chain, ref_chain)

        grad_node = np.array([grad_x, grad_y])
        all_grads.append(grad_node)
    return np.array(all_grads)

chain_grad = grad_chain_bias(chain, ref)

# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)


# cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

reference = n_ref.optimized


dist_to_ref0 = node_wise_distance(reference.coordinates, chain1)
plot_chain(reference, color='white', label=path_bias(dist_to_ref0))


dist_to_ref1 = node_wise_distance(reference.coordinates,n_ref.chain_trajectory[0].coordinates)
plot_chain(n_ref.chain_trajectory[0],color='yellow', label=path_bias(dist_to_ref1))

for ind in range(len(n_ref.initial_chain)):
    locs = n_ref.initial_chain
    dx,dy = chain_grad[ind] / np.linalg.norm(chain_grad[ind]) 
    plt.arrow(locs[ind].coords[0], locs[ind].coords[1], dx=-1*dx, dy=-1*dy, width=.1)

plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.legend()

plt.show()

# +
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neb_dynamics.Node import Node
from scipy.optimize import minimize


@dataclass
class Node2D_Bias(Node):
    pair_of_coordinates: np.array
    converged: bool = False
    do_climb: bool = False

    is_a_molecule = False
    
    _cached_energy: float | None = None
    _cached_gradient: np.array | None = None

    @property
    def coords(self):
        return self.pair_of_coordinates

    @staticmethod
    def en_func(node: Node2D_Bias):
        x, y = node.coords
       # TODO

    @staticmethod
    def en_func_arr(xy_vals):
        x, y = xy_vals
        # TODO

    def do_geometry_optimization(self) -> Node:
        out = minimize(self.en_func_arr, self.coords)
        out_node = self.copy()
        out_node.pair_of_coordinates = out.x
        return out_node

    def is_identical(self, other: Node):
        other_opt = other.do_geometry_optimization()
        self_opt = self.do_geometry_optimization()
        dist = np.linalg.norm(other_opt.coords - self_opt.coords)
        return abs(dist) < .1


    @staticmethod
    def grad_func(node: Node2D_Bias):
        x, y = node.coords
        # TODO 

    @property
    def energy(self) -> float:
        return self.en_func(self)

    @property
    def gradient(self) -> np.array:
        return self.grad_func(self)

    @property
    def hessian(self) -> np.array:
        return self.hess_func(self)

    @staticmethod
    def dot_function(self, other: Node2D_Bias) -> float:
        return np.dot(self, other)

    def copy(self):
        return Node2D_Bias(
            pair_of_coordinates=self.pair_of_coordinates,
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array):
        new_node = self.copy()
        new_node.pair_of_coordinates = coords
        return new_node

    def get_nudged_pe_grad(self, unit_tangent, gradient):
        """
        Returns the component of the gradient that acts perpendicular to the path tangent
        """
        pe_grad = gradient
        pe_grad_nudged_const = self.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
        return pe_grad_nudged

