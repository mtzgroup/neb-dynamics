# -*- coding: utf-8 -*-
# +
from dataclasses import dataclass
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from neb_dynamics.Node import Node
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
from itertools import product
import warnings
warnings.filterwarnings("ignore")



from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.Node2d import Node2D_Flower
from neb_dynamics.Node3D_TC import Node3D_TC


from matplotlib.animation import FuncAnimation
import IPython
from pathlib import Path


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
nimages = 5
np.random.seed(0)
ks = .1


start_point = [-2.59807434, -1.499999  ]
end_point = [2.5980755 , 1.49999912]


coords = np.linspace(start_point, end_point, nimages)
coords[1:-1] += [-1,1] # i.e. good initial guess
cni_ref = ChainInputs(
    k=ks,
    node_class=Node2D_Flower,
    delta_k=0,
    step_size=.1,
    do_parallel=False,
    use_geodesic_interpolation=False,
)
gii = GIInputs(nimages=nimages)
# chain = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni)
nbi = NEBInputs(tol=.1, v=1, max_steps=8000, climb=False, stopping_threshold=0)
chain_ref = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni_ref)
# -

n_ref = NEB(initial_chain=chain_ref,parameters=nbi )
n_ref.optimize_chain()

gii = GIInputs(nimages=5)
m = MSMEP(neb_inputs=nbi,chain_inputs=cni_ref, gi_inputs=gii,split_method='maxima',recycle_chain=True)
history, out_chain = m.find_mep_multistep(chain_ref)

#### get energies for countourplot
gridsize = 100
# min_val = -5.3
# max_val = 5.3
min_val = -4
max_val = 4
x = np.linspace(start=min_val, stop=max_val, num=gridsize)
y = x.reshape(-1, 1)

h_flat_ref = np.array([Node2D_Flower.en_func_arr(pair) for pair in product(x,x)])
h_ref = h_flat_ref.reshape(gridsize,gridsize).T

n_ref.optimized.plot_chain()

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

plot_chain(n_ref.initial_chain, c='orange')
plot_chain(n_ref.optimized, c='skyblue',linestyle='-')
plot_chain(out_chain, c='red')
plt.show()


# -

# # Other Stuff

def get_distance(x1,y1,x2,y2):
    dist = np.linalg.norm([(x2-x1)  ,(y2 - y1)])
    grad = np.array([(x1-x2), (y1 - y2)]) / dist
    return dist, grad


def animate_func(neb_obj: NEB, h, x, y):
    n_nodes = len(neb_obj.initial_chain.nodes)
    en_func = neb_obj.initial_chain[0].en_func
    chain_traj = neb_obj.chain_trajectory
    plt.style.use("seaborn-pastel")
    s = 5.3
    figsize = 5

    f, ax = plt.subplots(figsize=(1.618 * figsize, figsize))

    min_val = -s
    max_val = s

    
    cs = plt.contourf(x, x, h, cmap="Greys",alpha=.8)
    _ = f.colorbar(cs, ax=ax)
    arrows = [
        ax.arrow(0, 0, 0, 0, head_width=0.05, facecolor="black") for _ in range(n_nodes)
    ]
    (line,) = ax.plot([], [], "o--", lw=3)

    def animate(chain):

        x = chain.coordinates[:, 0]
        y = chain.coordinates[:, 1]

        for arrow, (x_i, y_i), (dx_i, dy_i) in zip(
            arrows, chain.coordinates, chain.gradients
        ):
            arrow.set_data(x=x_i, y=y_i, dx=-1 * dx_i, dy=-1 * dy_i)

        line.set_data(x, y)
        return (x for x in arrows)

    anim = FuncAnimation(
        fig=f,
        func=animate,
        frames=chain_traj,
        blit=True,
        repeat_delay=1000,
        interval=200,
    )
    anim.save(f'wtf.gif')
    return anim


@dataclass
class Node2D_Flower_Bias(Node):
    pair_of_coordinates: np.array
    # reference: np.array 
    converged: bool = False
    do_climb: bool = False
    
    # references = [node.pair_of_coordinates for node in n_ref.optimized[1:-1]]
    # references = [node.pair_of_coordinates for node in n_ref.optimized]
    references = [node.pair_of_coordinates for node in out_chain[1:-1]]

    # references = [np.array([3,-2])]
    strength = 5
    alpha = .8

    @property
    def coords(self):
        return self.pair_of_coordinates

    @staticmethod
    def en_func(node):
        x, y = node.coords
        en_orig =  (1./20.)*(( 1*(x**2 + y**2) - 6*np.sqrt(x**2 + y**2))**2 + 30 ) * -1*np.abs(.4 * np.cos(6  * np.arctan(x/y))+1) 
        grad_bias, en_bias = Node2D_Flower_Bias.mtd_grad_energy(point=node.coords, references=Node2D_Flower_Bias.references)
        return en_orig + en_bias
        
    @staticmethod
    def en_func_arr(xy_vals):
        x,y = xy_vals
        en_orig =  (1./20.)*(( 1*(x**2 + y**2) - 6*np.sqrt(x**2 + y**2))**2 + 30 ) * -1*np.abs(.4 * np.cos(6  * np.arctan(x/y))+1) 
        grad_bias, en_bias = Node2D_Flower_Bias.mtd_grad_energy(point=[x,y],references=Node2D_Flower_Bias.references)
        return en_orig + en_bias
    
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
    def grad_func(node):
        x, y = node.coords
        x2y2 = x**2 + y**2
        
        cos_term  = 0.4*np.cos(6*np.arctan(x/y)) + 1
        # d/dx
        Ax = 0.12*((-6*np.sqrt(x2y2) + x2y2)**2 + 30) 
        
        Bx = np.sin(6*np.arctan(x/y))*(cos_term)
        
        Cx = y*(x**2 / y**2 + 1)*np.abs(cos_term)
                                       
        Dx = (1/10)*(2*x - (6*x / np.sqrt(x2y2) ))*(-6*np.sqrt(x2y2) + x2y2)*np.abs(cos_term)
        
        dx = (Ax*Bx)/Cx - Dx
        
        # d/dy
        Ay = (-1/10)*(2*y - 6*y/(np.sqrt(x2y2)))*(-6*np.sqrt(x2y2) + x2y2)*(np.abs(cos_term))
        
        By = 0.12*x*((-6*np.sqrt(x2y2) + x2y2)**2 + 30)*np.sin(6*np.arctan(x/y))
        
        Cy = cos_term
        
        Dy = y**2 * (x**2 / y**2 + 1)*np.abs(cos_term)
        
        dy =   Ay - (By*Cy)/Dy
        
        grad_orig =  np.array([dx,dy])
        grad_bias, en_bias = Node2D_Flower_Bias.mtd_grad_energy(point=node.coords, references=Node2D_Flower_Bias.references)
        
        return grad_orig + grad_bias
        # return grad_orig - grad_bias
        
        
    @property
    def energy(self) -> float:
        return Node2D_Flower_Bias.en_func(self)

    @property
    def gradient(self) -> np.array:
        return Node2D_Flower_Bias.grad_func(self)

    @staticmethod
    def dot_function(self, other) -> float:
        return np.dot(self, other)

    def copy(self):
        return Node2D_Flower_Bias(
            pair_of_coordinates=self.pair_of_coordinates,
            converged=self.converged,
            do_climb=self.do_climb,
        )

    def update_coords(self, coords: np.array):
        new_node = self.copy()
        new_node.pair_of_coordinates = coords
        return new_node
    
    @staticmethod
    def mtd_grad_energy(point, references: list):
        x,y = point
        gradient = np.zeros(2)
        energy = 0


        for reference in references:
            rmsd, g_rmsd = get_distance(x,y, reference[0],reference[1])
            biaspot_i  = Node2D_Flower_Bias.strength*math.exp(-(Node2D_Flower_Bias.alpha* rmsd**2))
            biasgrad_i =  -2*Node2D_Flower_Bias.alpha*g_rmsd*biaspot_i * rmsd

            gradient += biasgrad_i
            energy += biaspot_i

        return gradient, energy

    def get_nudged_pe_grad(self, unit_tangent, gradient):
        """
        Returns the component of the gradient that acts perpendicular to the path tangent
        """
        pe_grad = gradient
        pe_grad_nudged_const = self.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
        return pe_grad_nudged


def get_grad_bias(point):
    grad_bias, en_bias = Node2D_Flower_Bias.mtd_grad_energy(point, Node2D_Flower_Bias.references)
    
    return grad_bias


def get_grad(point):
    node = Node2D_Flower(point)
    
    return node.gradient


from neb_dynamics.Node2d import Node2D

#### get energies for countourplot
gridsize = 100
# min_val = -5.3
# max_val = 5.3
min_val = -4
max_val = 4
x = np.linspace(start=min_val, stop=max_val, num=gridsize)
y = x.reshape(-1, 1)
h_flat = np.array([Node2D_Flower_Bias.en_func_arr(pair) for pair in product(x,x)])
# h_flat = np.array([Node2D.en_func_arr(pair) for pair in product(x,x)])
h = h_flat.reshape(gridsize,gridsize).T

# +
fig = 8

fs = 18
points = [(2, -1), (0,2)]

f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)


cs = plt.contourf(x, x, h, cmap="Greys",alpha=.8)
_ = f.colorbar(cs)

# for i, point in enumerate(points):
#     grad_bias = -1*get_grad_bias(point)
#     grad = -1*get_grad(point)
#     if i ==0:
#         plt.arrow(point[0], point[1],dx=grad_bias[0],dy=grad_bias[1], color='red',width=.1, label='force bias')
#         plt.arrow(point[0], point[1],dx=grad[0],dy=grad[1], color='blue',width=.1, label='force')
#         plt.arrow(point[0], point[1],dx=grad[0]+grad_bias[0],dy=grad[1]+grad_bias[1], color='green',width=.1,label='tot')
#     else:
#         plt.arrow(point[0], point[1],dx=grad_bias[0],dy=grad_bias[1], color='red',width=.1)
#         plt.arrow(point[0], point[1],dx=grad[0],dy=grad[1], color='blue',width=.1)
#         plt.arrow(point[0], point[1],dx=grad[0]+grad_bias[0],dy=grad[1]+grad_bias[1], color='green',width=.1)
# plt.legend()

plt.xlim(min_val, max_val)
plt.ylim(min_val,max_val)
# -

cni = ChainInputs(
    k=0.00,
    node_class=Node2D_Flower_Bias,
    delta_k=0,
    step_size=.15,
    do_parallel=False,
    use_geodesic_interpolation=False,
)
chain = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni)
nbi = NEBInputs(tol=1, v=1, max_steps=200, climb=False, stopping_threshold=0)

n = NEB(initial_chain=chain,parameters=nbi)
n.optimize_chain()

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18

f, ax = plt.subplots(figsize=(2.3 * fig, fig),ncols=2)
# x = np.linspace(start=min_val, stop=max_val, num=1000)
# y = x.reshape(-1, 1)

cs = ax[0].contourf(x, x, h, cmap="Greys",alpha=.8)
_ = f.colorbar(cs,ax=ax[0])

plot_neb(n,ax=ax[0], c='red')
plot_chain(out_chain,ax=ax[0],c='blue')

# -

history_final, out_final = m.find_mep_multistep(n.optimized)

h_flat_ref = np.array([Node2D_Flower.en_func_arr(pair) for pair in product(x,x)])
h_ref = h_flat_ref.reshape(gridsize,gridsize).T

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18

f, ax = plt.subplots(figsize=(2.3 * fig, fig),ncols=2)
# x = np.linspace(start=min_val, stop=max_val, num=1000)
# y = x.reshape(-1, 1)

cs = ax[0].contourf(x, x, h_ref, cmap="Greys",alpha=.8)
_ = f.colorbar(cs,ax=ax[0])

plot_chain(n.optimized,ax=ax[0], c='red')
plot_chain(out_chain,ax=ax[0],c='blue')
plot_chain(out_final,c='orange',ax=ax[0])

# +
# animate_func(n, h, x, x)

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18

f, ax = plt.subplots(figsize=(2.3 * fig, fig),ncols=2)
x = np.linspace(start=min_val, stop=max_val, num=1000)
y = x.reshape(-1, 1)

# h_flat = np.array([Node2D_Flower.en_func_arr(pair) for pair in product(x,x)])
# h = h_flat.reshape(1000,1000).T
cs = ax[0].contourf(x, x, h, cmap="Greys",alpha=.8)
_ = f.colorbar(cs,ax=ax[0])
# plot_neb(n,ax=ax[0], c='red')
plot_neb(n_ref,ax=ax[0],c='blue')
plot_chain(out_chain,ax=ax[0], c= 'skyblue')
# -

