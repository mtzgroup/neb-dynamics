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
ind = 2

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
    [-1.05565696,  1.01107738] #[-1, 1]
]

end_points = [
    [2.5980755 , 1.49999912],
    [2.99999996, 1.99999999], # --> interesting other endpoint [ 3.58442836, -1.84812646]
    [1.05565701, -1.01107741]#[1, -1],

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
# -

#### get energies for countourplot
gridsize = 100
# min_val = -4
# max_val = 4
min_val = -2
max_val = 2
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


#### asneb
gii = GIInputs(nimages=NIMAGES)
nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_chain_rms_thre=0.002, early_stop_force_thre=1, node_freezing=False)
# nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=3, node_freezing=False)
m = MSMEP(neb_inputs=nbi_msmep,chain_inputs=cni_ref, gi_inputs=gii)
history_ref, out_chain_ref = m.find_mep_multistep(chain_ref)

# +
c_to_plot = chain_ref
# bias_chain = n_ref.optimized
bias_chain = out_chain_ref



cb = ChainBiaser(reference_chain=bias_chain,
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
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)


# cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

reference = bias_chain



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

# # Chain2

# +
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from retropaths.abinitio.trajectory import Trajectory

from neb_dynamics.Node import Node
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.Inputs import ChainInputs
from neb_dynamics.helper_functions import RMSD, get_mass, _get_ind_minima, _get_ind_maxima, linear_distance, qRMSD_distance, pairwise


from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method
from pathlib import Path

import scipy


@dataclass
class Chain2:
    nodes: List[Node]
    parameters: ChainInputs
    
    def __post_init__(self):
        if not hasattr(self.parameters, "velocity"):
            self._zero_velocity()
            
            
    def _zero_velocity(self):
        if self[0].is_a_molecule:
            self.parameters.velocity = np.zeros(shape=(len(self.nodes), len(self.nodes[0].coords), 3))
        else:
            self.parameters.velocity = np.zeros(shape=(len(self.nodes), len(self.nodes[0].coords)))

    @property
    def n_atoms(self):
        return self.coordinates[0].shape[0]

    @classmethod
    def from_xyz(cls, fp: Path, parameters: ChainInputs):
        traj = Trajectory.from_xyz(fp)
        chain = cls.from_traj(traj, parameters=parameters)
        energies_fp = fp.parent / Path(str(fp.stem)+".energies")
        grad_path = fp.parent / Path(str(fp.stem)+".gradients")
        grad_shape_path = fp.parent / "grad_shapes.txt"\
        
        if energies_fp.exists() and grad_path.exists() and grad_shape_path.exists():
            energies = np.loadtxt(energies_fp)
            gradients_flat = np.loadtxt(grad_path)
            gradients_shape = np.loadtxt(grad_shape_path,dtype=int)
            
            gradients = gradients_flat.reshape(gradients_shape)
            
            for node,(ene, grad) in zip(chain.nodes, zip(energies, gradients)):
                node._cached_energy = ene
                node._cached_gradient = grad
        return chain
    
    @classmethod
    def from_list_of_chains(cls, list_of_chains, parameters):
        nodes = []
        for chain in list_of_chains:
            nodes.extend(chain.nodes)
        return cls(nodes=nodes, parameters=parameters)


    def _distance_to_chain(self, other_chain: Chain):
        chain1 = self
        chain2 = other_chain

        distances = []
        

        for node1, node2 in zip(chain1.nodes, chain2.nodes):
            if node1.coords.shape[0] > 2:
                dist,_ = RMSD(node1.coords, node2.coords)
            else:
                dist = np.linalg.norm(node1.coords - node2.coords)
            distances.append(dist)

        return sum(distances) / len(chain1)
    
    def _tangent_correlations(self, other_chain: Chain):
        chain1_vec = np.array(self.unit_tangents).flatten()
        chain2_vec = np.array(other_chain.unit_tangents).flatten()
        projector = np.dot(chain1_vec, chain2_vec)
        normalization = np.dot(chain1_vec, chain1_vec)
        
        return projector / normalization
    
    def _gradient_correlation(self, other_chain: Chain):
        
        chain1_vec = np.array(self.gradients).flatten()
        chain1_vec = chain1_vec / np.linalg.norm(chain1_vec)
        
        chain2_vec = np.array(other_chain.gradients).flatten()
        chain2_vec = chain2_vec / np.linalg.norm(chain2_vec)
        
        projector = np.dot(chain1_vec, chain2_vec)
        normalization = np.dot(chain1_vec, chain1_vec)
        
        return projector / normalization
    
    def _gradient_delta_mags(self, other_chain: Chain):
        
        chain1_vec = np.array(self.gradients).flatten()
        chain2_vec = np.array(other_chain.gradients ).flatten()
        diff = np.linalg.norm(chain2_vec - chain1_vec)
        normalization = self.n_atoms * len(self.nodes)
        
        return diff / normalization


    def _get_mass_weighed_coords(self):
        traj = self.to_trajectory()
        coords = traj.coords
        weights = np.array([np.sqrt(get_mass(s)) for s in traj.symbols]) 
        weights = weights  / sum(weights)
        mass_weighed_coords = coords  * weights.reshape(-1,1)
        return mass_weighed_coords


    @property
    def _path_len_coords(self):
        if self.nodes[0].is_a_molecule:
            coords = self._get_mass_weighed_coords()
        else:
            coords = self.coordinates
        return coords
    
    def _path_len_dist_func(self, coords1, coords2):
        if self.nodes[0].is_a_molecule:
            return qRMSD_distance(coords1, coords2)
        else:
            return linear_distance(coords1, coords2)

    @property
    def integrated_path_length(self):
        coords = self._path_len_coords
        cum_sums = [0]
        int_path_len = [0]
        for i, frame_coords in enumerate(coords):
            if i == len(coords) - 1:
                continue
            next_frame = coords[i + 1]
            distance = self._path_len_dist_func(frame_coords, next_frame)
            cum_sums.append(cum_sums[-1] + distance)

        cum_sums = np.array(cum_sums)
        int_path_len = cum_sums / cum_sums[-1]
        return np.array(int_path_len)

    def _k_between_nodes(
        self, node0: Node, node1: Node, e_ref: float, k_max: float, e_max: float
    ):
        e_i = max(node1.energy, node0.energy)
        if e_i > e_ref:
            new_k = k_max - self.parameters.delta_k * ((e_max - e_i) / (e_max - e_ref))
        elif e_i <= e_ref:
            new_k = k_max - self.parameters.delta_k
        return new_k

    def plot_chain(self):
        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))
        plt.plot(
            self.integrated_path_length,
            (self.energies - self.energies[0]) * 627.5,
            "o--",
            label="neb",
        )
        plt.ylabel("Energy (kcal/mol)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.show()

    def __getitem__(self, index):
        return self.nodes.__getitem__(index)

    def __len__(self):
        return len(self.nodes)

    def insert(self, index, node):
        self.nodes.insert(index, node)

    def append(self, node):
        self.nodes.append(node)

    def copy(self):
        list_of_nodes = [node.copy() for node in self.nodes]
        chain_copy = Chain2(nodes=list_of_nodes, parameters=self.parameters)
        return chain_copy

    def iter_triplets(self) -> list[list[Node]]:
        for i in range(1, len(self.nodes) - 1):
            yield self.nodes[i - 1 : i + 2]

    @classmethod
    def from_traj(cls, traj: Trajectory, parameters: ChainInputs):
        nodes = [parameters.node_class(s) for s in traj]
        return Chain2(nodes, parameters=parameters)

    @classmethod
    def from_list_of_coords(
        cls, list_of_coords: List, parameters: ChainInputs
    ) -> Chain2:
        nodes = [parameters.node_class(point) for point in list_of_coords]
        return cls(nodes=nodes, parameters=parameters)

    @property
    def path_distances(self):
        dist = []
        for i in range(len(self.nodes)):
            if i == 0:
                continue
            start = self.nodes[i - 1]
            end = self.nodes[i]

            dist.append(self.quaternionrmsd(start.coords, end.coords))

        return np.array(dist)

    @cached_property
    def work(self) -> float:
        ens = self.energies
        ens -= ens[0]

        works = np.abs(ens[1:] * self.path_distances)
        tot_work = works.sum()
        return tot_work

    @cached_property
    def energies(self) -> np.array:
        return np.array([node.energy for node in self.nodes])
    
    @property
    def energies_kcalmol(self) -> np.array:
        return (self.energies - self.energies[0])*627.5

    def neighs_grad_func(self, prev_node: Node, current_node: Node, next_node: Node):

        vec_tan_path = self._create_tangent_path(
            prev_node=prev_node, current_node=current_node, next_node=next_node
        )
        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        pe_grad = current_node.gradient

        if not current_node.do_climb:
            pe_grads_nudged = current_node.get_nudged_pe_grad(
                unit_tan_path, gradient=pe_grad
            )
            spring_forces_nudged = self.get_force_spring_nudged(
                prev_node=prev_node,
                current_node=current_node,
                next_node=next_node,
                unit_tan_path=unit_tan_path,
            )

        elif current_node.do_climb:

            pe_along_path_const = current_node.dot_function(pe_grad, unit_tan_path)
            pe_along_path = pe_along_path_const * unit_tan_path

            climbing_grad = 2 * pe_along_path

            pe_grads_nudged = pe_grad - climbing_grad

            zero = np.zeros_like(pe_grad)
            spring_forces_nudged = zero
        else:
            raise ValueError(
                f"current_node.do_climb is not a boolean: {current_node.do_climb=}"
            )

        return pe_grads_nudged, spring_forces_nudged  # , anti_kinking_grads

    def pe_grads_spring_forces_nudged(self):
        pe_grads_nudged = []
        spring_forces_nudged = []
        # anti_kinking_grads = []
        for prev_node, current_node, next_node in self.iter_triplets():
            pe_grad_nudged, spring_force_nudged = self.neighs_grad_func(
                prev_node=prev_node,
                current_node=current_node,
                next_node=next_node,
            )

            # anti_kinking_grads.append(anti_kinking_grad)
            if not current_node.converged:
                pe_grads_nudged.append(pe_grad_nudged)
                spring_forces_nudged.append(spring_force_nudged)
            else:
                zero = np.zeros_like(pe_grad_nudged)
                pe_grads_nudged.append(zero)
                spring_forces_nudged.append(zero)

        pe_grads_nudged = np.array(pe_grads_nudged)
        spring_forces_nudged = np.array(spring_forces_nudged)
        return pe_grads_nudged, spring_forces_nudged

    def get_maximum_grad_magnitude(self):
        
        return np.max([np.amax(np.abs(grad)) for grad in self.gradients])
    
    def get_maximum_gperp(self):
        gperp, gspring = self.pe_grads_spring_forces_nudged()
        max_gperps = []
        for gp, node in zip(gperp, self):
            if not node.converged:
                max_gperps.append(np.amax(np.abs(gp)))
        return np.max(max_gperps)
    
    def get_maximum_rms_grad(self):
        return np.max([np.sqrt(np.mean(np.square(grad.flatten())) / len(grad.flatten())) for grad in self.gradients])

    @staticmethod
    def calc_xtb_ene_grad_from_input_tuple(tuple):
        atomic_numbers, coords_bohr, charge, spinmult = tuple

        calc = Calculator(
            get_method("GFN2-xTB"),
            numbers=np.array(atomic_numbers),
            positions=coords_bohr,
            charge=charge,
            uhf=spinmult - 1,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()

        return res.get_energy(), res.get_gradient() * BOHR_TO_ANGSTROMS


    @cached_property
    def gradients(self) -> np.array:
        
        all_grads = [node._cached_gradient for node in self.nodes]
        
        if not np.all([g is not None for g in all_grads]):
            if self.parameters.do_parallel:
                energy_gradient_tuples = self.parameters.node_class.calculate_energy_and_gradients_parallel(chain=self)
            else:
                energies = [node.energy for node in self.nodes]
                gradients = [node.gradient for node in self.nodes]
                energy_gradient_tuples = list(zip(energies, gradients))

            for (ene, grad), node in zip(energy_gradient_tuples, self.nodes):
                node._cached_energy = ene
                node._cached_gradient = grad
        
        pe_grads_nudged, spring_forces_nudged = self.pe_grads_spring_forces_nudged()

        grads = (
            pe_grads_nudged - spring_forces_nudged
        )  # + self.parameters.k * anti_kinking_grads
        
        
        # add chain bias if relevant
        if self.parameters.do_chain_biasing:
            foobar_node = Node2D_Zero([0,0])
            tans = self.unit_tangents
            
            # cb = ChainBiaser(reference_chain=n_ref.optimized, amplitude=self.parameters.amp, sigma=self.parameters.sig, distance_func=self.parameters.distance_func)
            bias_grads = self.parameters.cb.grad_chain_bias(self)
            proj_grads = np.array([foobar_node.get_nudged_pe_grad(tan, grad) for tan,grad in zip(tans, bias_grads)])
            
                
            grads += proj_grads

            
        # add the zeros for the endpoints
        zero = np.zeros_like(grads[0])
        grads = np.insert(grads, 0, zero, axis=0)
        grads = np.insert(grads, len(grads), zero, axis=0)
            
            
        # remove rotations and translations
        if grads.shape[1] >= 3:  # if we have at least 3 atoms
            grads[:, 0, :] = 0  # this atom cannot move
            grads[:, 1, :2] = 0  # this atom can only move in a line
            grads[:, 2, :1] = 0  # this atom can only move in a plane


        # zero all nodes that have converged 
        for (i, grad), node in zip(enumerate(grads), self.nodes):
            if node.converged:
                grads[i] = grad*0
                
                

        return grads

    @property
    def unit_tangents(self):
        tan_list = []
        for prev_node, current_node, next_node in self.iter_triplets():
            tan_vec = self._create_tangent_path(
                prev_node=prev_node, current_node=current_node, next_node=next_node
            )
            unit_tan = tan_vec / np.linalg.norm(tan_vec)
            tan_list.append(unit_tan)

        return tan_list

    @property
    def coordinates(self) -> np.array:

        return np.array([node.coords for node in self.nodes])

    def _create_tangent_path(
        self, prev_node: Node, current_node: Node, next_node: Node
    ):
        en_2 = next_node.energy
        en_1 = current_node.energy
        en_0 = prev_node.energy
        if en_2 > en_1 and en_1 > en_0:
            return next_node.coords - current_node.coords
        elif en_2 < en_1 and en_1 < en_0:
            return current_node.coords - prev_node.coords

        else:
            deltaV_max = max(np.abs(en_2 - en_1), np.abs(en_0 - en_1))
            deltaV_min = min(np.abs(en_2 - en_1), np.abs(en_0 - en_1))

            tau_plus = next_node.coords - current_node.coords
            tau_minus = current_node.coords - prev_node.coords
            if en_2 > en_0:
                tan_vec = deltaV_max * tau_plus + deltaV_min * tau_minus
            elif en_2 < en_0:
                tan_vec = deltaV_min * tau_plus + deltaV_max * tau_minus

            else:
                return 0.5 * (tau_minus + tau_plus)
                # raise ValueError(
                #     f"Energies adjacent to current node are identical. {en_2=} {en_0=}"
                # )

            return tan_vec

    def _get_anti_kink_switch_func(self, prev_node, current_node, next_node):
        # ANTI-KINK FORCE
        vec_2_to_1 = next_node.coords - current_node.coords
        vec_1_to_0 = current_node.coords - prev_node.coords
        cos_phi = current_node.dot_function(vec_2_to_1, vec_1_to_0) / (
            np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0)
        )

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))
        return f_phi

    def get_force_spring_nudged(
        self,
        prev_node: Node,
        current_node: Node,
        next_node: Node,
        unit_tan_path: np.array,
    ):

        k_max = (
            max(self.parameters.k)
            if hasattr(self.parameters.k, "__iter__")
            else self.parameters.k
        )
        e_ref = max(self.nodes[0].energy, self.nodes[len(self.nodes)-1].energy)
        e_max = max(self.energies)

        k01 = self._k_between_nodes(
            node0=prev_node,
            node1=current_node,
            e_ref=e_ref,
            k_max=k_max,
            e_max=e_max,
        )

        k12 = self._k_between_nodes(
            node0=current_node,
            node1=next_node,
            e_ref=e_ref,
            k_max=k_max,
            e_max=e_max,
        )

        # print(f"***{k12=} // {k01=}")

        force_spring = k12 * np.linalg.norm(
            next_node.coords - current_node.coords
        ) - k01 * np.linalg.norm(current_node.coords - prev_node.coords)
        return force_spring * unit_tan_path

    def to_trajectory(self):
        t = Trajectory([n.tdstructure for n in self.nodes])
        return t
    
    def is_elem_step(self):
        chain = self.copy()
        if len(self) <= 1:
            return True

        conditions = {}
        is_concave = self._chain_is_concave()
        conditions['concavity'] = is_concave
        if not is_concave:
            return False, "minima"
        
        
        r,p = self._approx_irc()
        
        cases = [
            r.is_identical(chain[0]) and p.is_identical(chain[-1]), # best case
            r.is_identical(chain[0]) and p.is_identical(chain[0]), # both collapsed to reactants
            r.is_identical(chain[-1]) and p.is_identical(chain[-1]) # both collapsed to products
        ]
        
        minimizing_gives_endpoints = any(cases) 
        conditions['irc'] = minimizing_gives_endpoints

        split_method = self._select_split_method(conditions)
        elem_step = True if split_method is None else False
        return elem_step, split_method

    def _chain_is_concave(self):
        ind_minima = _get_ind_minima(self)
        minima_present =  len(ind_minima) != 0
        if minima_present:
            minimas_is_r_or_p = []
            for i in ind_minima:
                opt =  self[i].do_geometry_optimization()
                minimas_is_r_or_p.append(
                    opt.is_identical(self[0]) or opt.is_identical(self[-1]) 
                    )
            
            print(f"\n{minimas_is_r_or_p=}\n")
            return all(minimas_is_r_or_p)
            
        else:
            return True

    def _approx_irc(self, index=None):
        chain = self.copy()
        if index is None:
            arg_max = np.argmax(chain.energies)
        else:
            arg_max = index
            
        if arg_max == len(chain)-1 or arg_max == 0: # monotonically changing function, 
            return chain[0], chain[len(chain)-1]

        candidate_r = chain[arg_max - 1]
        candidate_p = chain[arg_max + 1]
        r = candidate_r.do_geometry_optimization()
        p = candidate_p.do_geometry_optimization()
        return r, p

    def _select_split_method(self, conditions: dict):
        all_conditions_met = all([val for key,val in conditions.items()])
        if all_conditions_met: 
            return None

        if conditions['concavity'] is False: # prioritize the minima condition
            return 'minima'
        elif conditions['irc'] is False:
            return 'maxima'
        
        
    def write_ene_info_to_disk(self, fp):
        ene_path = fp.parent / Path(str(fp.stem)+".energies")
        grad_path = fp.parent / Path(str(fp.stem)+".gradients")
        grad_shape_path = fp.parent / "grad_shapes.txt"
        
        np.savetxt(ene_path, self.energies)
        np.savetxt(grad_path, self.gradients.flatten())
        np.savetxt(grad_shape_path, self.gradients.shape)
        
    def write_to_disk(self, fp: Path):
        if self.nodes[0].is_a_molecule:
            traj = self.to_trajectory()
            traj.write_trajectory(fp)
            
            self.write_ene_info_to_disk(fp)
            
        else:
            raise NotImplementedError("Cannot write 2D chains yet.")


# -

# # NEB2

# +
from __future__ import annotations

import sys
from dataclasses import dataclass, field

# from hashlib import new
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema

from neb_dynamics.Chain import Chain
from neb_dynamics.helper_functions import pairwise
from neb_dynamics.Node import Node
from neb_dynamics import ALS
from neb_dynamics.Inputs import NEBInputs, ChainInputs
from kneed import KneeLocator

import matplotlib.pyplot as plt


VELOCITY_SCALING = .3

@dataclass
class NoneConvergedException(Exception):
    trajectory: list[Chain]
    msg: str
    obj: NEB


@dataclass
class NEB2:
    initial_chain: Chain2
    parameters: NEBInputs

    optimized: Chain2 = None
    chain_trajectory: list[Chain2] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)

    def __post_init__(self):
        self.n_steps_still_chain = 0
        
    def do_velvel(self, chain: Chain):
        max_grad_val = chain.get_maximum_grad_magnitude()
        return max_grad_val < self.parameters.vv_force_thre

    def _reset_node_convergence(self, chain):
        for node in chain:
            node.converged = False

    def set_climbing_nodes(self, chain: Chain2):
        # reset node convergence
        self._reset_node_convergence(chain=chain)

        inds_maxima = argrelextrema(chain.energies, np.greater, order=2)[0]
        if self.parameters.v > 1:
            print(f"----->Setting {len(inds_maxima)} nodes to climb")

        for ind in inds_maxima:
            chain[ind].do_climb = True


    def _check_early_stop(self, chain: Chain2):
        max_grad_val = chain.get_maximum_grad_magnitude()
        
        dist_to_prev_chain = chain._distance_to_chain(self.chain_trajectory[-2]) # the -1 is the chain im looking at
        if dist_to_prev_chain < self.parameters.early_stop_chain_rms_thre:
            self.n_steps_still_chain += 1
        else:
            self.n_steps_still_chain = 0
        
        
        correlation = self.chain_trajectory[-2]._gradient_correlation(chain)
        conditions = [ 
                      max_grad_val <= self.parameters.early_stop_force_thre,
                      dist_to_prev_chain <= self.parameters.early_stop_chain_rms_thre,
                      correlation >= self.parameters.early_stop_corr_thre,
                      self.n_steps_still_chain >= self.parameters.early_stop_still_steps_thre
        ]
        # if any(conditions):
        if (conditions[0] and conditions[1]) or conditions[3]: # if we've dipped below the force thre and chain rms is low
                                                                # or chain has stayed still for a long time
            is_elem_step, split_method = chain.is_elem_step()
            
            if not is_elem_step:
                print(f"\nStopped early because chain is not an elementary step.")
                print(f"Split chain based on: {split_method}")
                self.optimized = chain
                return True
            
            else:
                
                if (conditions[0] and conditions[1]): # dont reset them if you stopped due to stillness
                    # reset early stop checks
                    self.parameters.early_stop_force_thre = 0.0
                    self.parameters.early_stop_chain_rms_thre = 0.0
                    self.parameters.early_stop_corr_thre = 10.
                    self.parameters.early_stop_still_steps_thre = 100000
                    
                    self.set_climbing_nodes(chain=chain)
                    self.parameters.climb = False  # no need to set climbing nodes again
                else:
                    self.n_steps_still_chain = 0
                    
                
                return False

        else:
            return False
            

    def optimize_chain(self):
        nsteps = 1
        chain_previous = self.initial_chain.copy()
        chain_previous._zero_velocity()
        self.chain_trajectory.append(chain_previous)

        while nsteps < self.parameters.max_steps + 1:
            max_grad_val = chain_previous.get_maximum_grad_magnitude()
            max_rms_grad_val = chain_previous.get_maximum_rms_grad()
            if nsteps > 1:    
                stop_early = self._check_early_stop(chain_previous)
                if stop_early: 
                    return
                
            new_chain = self.update_chain(chain=chain_previous)
            n_nodes_frozen = 0
            for node in new_chain:
                if node.converged:
                    n_nodes_frozen+=1
                    
            if self.parameters.v:
                print(
                    f"step {nsteps} // max |gradient| {max_grad_val} // rms grad {max_rms_grad_val} // |velocity| {np.linalg.norm(new_chain.parameters.velocity)} // nodes_frozen {n_nodes_frozen}{' '*20}", end="\r"
                )
            sys.stdout.flush()

            self.chain_trajectory.append(new_chain)
            self.gradient_trajectory.append(new_chain.gradients)

            if self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
                if self.parameters.v:
                    print("\nChain converged!")

                self.optimized = new_chain
                return
            chain_previous = new_chain.copy()

            nsteps += 1

        new_chain = self.update_chain(chain=chain_previous)
        if not self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
            raise NoneConvergedException(
                trajectory=self.chain_trajectory,
                msg=f"\nChain did not converge at step {nsteps}",
                obj=self,
            )

    def get_chain_velocity(self, chain: Chain2) -> np.array:
        
        prev_velocity = chain.parameters.velocity
        als_max_steps = chain.parameters.als_max_steps
        
        beta = (chain.parameters.min_step_size / chain.parameters.step_size)**(1/als_max_steps)
        step = ALS.ArmijoLineSearch(
                chain=chain,
                t=chain.parameters.step_size,
                alpha=0.01,
                beta=beta,
                grad=chain.gradients,
                max_steps=als_max_steps
        )
        
        # step = chain.parameters.step_size
        # new_force = -(chain.gradients) * step        
        # directions = np.dot(prev_velocity.flatten(),new_force.flatten())
        
        # if directions < 0:
        #     total_force = new_force
        #     new_vel = np.zeros_like(chain.gradients)
        # else:
            
        #     new_velocity = directions*new_force # keep the velocity component in direcition of force
        #     total_force = new_velocity + new_force
        #     new_vel = total_force
        #     # print(f"\n\n keeping part of velocity! {np.linalg.norm(new_vel)}\n\n")
        
        # prev_velocity = chain.parameters.velocity
        # step = chain.parameters.step_size / 100

        new_force = -(chain.gradients) * step        
        new_vels_proj = []
        for vel_i, f_i in zip(prev_velocity, new_force):
            proj = np.dot(vel_i.flatten(), f_i.flatten()) / np.dot(f_i.flatten(), f_i.flatten())
            if proj > 0:
                vel_proj_flat = proj*f_i.flatten()
                vel_proj = vel_proj_flat.reshape(f_i.shape)
                new_vels_proj.append(vel_proj)
            else:
                new_vels_proj.append(0*f_i)
            
        new_vels_proj = np.array(new_vels_proj) + new_force
        # new_vel = new_vels_proj  + new_force
        # total_force = new_force + new_vel
        new_vel = new_vels_proj
        total_force = new_vel #+ new_force
        
        return new_vel, total_force

    def update_chain(self, chain: Chain) -> Chain:

        do_vv = self.do_velvel(chain=chain)

        if do_vv:
            new_vel, force = self.get_chain_velocity(chain=chain)
            new_chain_coordinates = chain.coordinates + force
            chain.parameters.velocity = new_vel

        else:
            als_max_steps = chain.parameters.als_max_steps
            beta = (chain.parameters.min_step_size / chain.parameters.step_size)**(1/als_max_steps)
            
            disp = ALS.ArmijoLineSearch(
                chain=chain,
                t=chain.parameters.step_size,
                alpha=0.01,
                beta=beta,
                grad=chain.gradients,
                max_steps=als_max_steps
            )
            new_chain_coordinates = chain.coordinates - chain.gradients * disp

        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = Chain2(new_nodes, parameters=chain.parameters)
        return new_chain

    def _update_node_convergence(self, chain: Chain, indices: np.array) -> None:
        for i, node in enumerate(chain):
            if i in indices:
                node.converged = True
            else:
                node.converged = False

    def _check_en_converged(self, chain_prev: Chain2, chain_new: Chain2) -> bool:
        differences = np.abs(chain_new.energies - chain_prev.energies)
        indices_converged = np.where(differences <= self.parameters.en_thre)

        return indices_converged[0], differences

    def _check_grad_converged(self, chain: Chain) -> bool:
        bools = []
        max_grad_components = []
        gradients = chain.gradients
        for grad in gradients:
            max_grad = np.amax(np.abs(grad))
            max_grad_components.append(max_grad)
            bools.append(max_grad < self.parameters.grad_thre)
        # bools = [True] # start node
        # max_grad_components = []
        # gradients = np.array([node.gradient for node in chain.nodes[1:-1]])
        # tans = chain.unit_tangents
        # for grad, tan in zip(gradients,tans):
        #     grad_perp = grad.flatten() - np.dot(grad.flatten(), tan.flatten())*tan.flatten()
        #     max_grad = np.amax(grad_perp)
        #     max_grad_components.append(max_grad)
        #     bools.append(max_grad <= self.parameters.grad_thre)
        
        # bools.append(True) # end node

        return np.where(bools), max_grad_components

    def _check_rms_grad_converged(self, chain: Chain2):
        bools = []
        rms_grads = []
        grads = chain.gradients
        for grad in grads:
            rms_gradient = np.sqrt(np.mean(np.square(grad.flatten())) / len(grad))
            rms_grads.append(rms_gradient)
            rms_grad_converged = rms_gradient <= self.parameters.rms_grad_thre
            bools.append(rms_grad_converged)

        return np.where(bools), rms_grads

    def _chain_converged(self, chain_prev: Chain2, chain_new: Chain2) -> bool:
        """
        https://chemshell.org/static_files/py-chemshell/manual/build/html/opt.html?highlight=nudged
        """

        rms_grad_conv_ind, max_rms_grads = self._check_rms_grad_converged(chain_new)
        en_converged_indices, en_deltas = self._check_en_converged(
            chain_prev=chain_prev, chain_new=chain_new
        )

        grad_conv_ind, max_grad_components = self._check_grad_converged(chain=chain_new)

        converged_nodes_indices = np.intersect1d(
            en_converged_indices, rms_grad_conv_ind
        )
        converged_nodes_indices = np.intersect1d(converged_nodes_indices, grad_conv_ind)

        if self.parameters.v > 1:
            [
                print(
                    f"\t\tnode{i} | ∆E : {en_deltas[i]} | Max(RMS Grad): {max_rms_grads[i]} | Max(Grad components): {max_grad_components[i]} | Converged? : {chain_new.nodes[i].converged}"
                )
                for i in range(len(chain_new))
            ]
        if self.parameters.v > 1:
            print(f"\t{len(converged_nodes_indices)} nodes have converged")
        if self.parameters.node_freezing:
            self._update_node_convergence(chain=chain_new, indices=converged_nodes_indices)
        return len(converged_nodes_indices) == len(chain_new)

    

    def _check_dot_product_converged(self, chain: Chain) -> bool:
        dps = []
        for prev_node, current_node, next_node in chain.iter_triplets():
            vec1 = current_node.coords - prev_node.coords
            vec2 = next_node.coords - current_node.coords
            dps.append(current_node.dot_function(vec1, vec2) > 0)

        return all(dps)


    def redistribution_helper(self, num, cum, chain: Chain) -> Node:
        """
        num: the distance from first node to return output point to
        cum: cumulative sums
        new_chain: chain that we are considering

        """

        for ii, ((cum_sum_init, node_start), (cum_sum_end, node_end)) in enumerate(
            pairwise(zip(cum, chain))
        ):

            if cum_sum_init <= num < cum_sum_end:
                direction = node_end.coords - node_start.coords
                percentage = (num - cum_sum_init) / (cum_sum_end - cum_sum_init)

                new_node = node_start.copy()
                new_coords = node_start.coords + (direction * percentage)
                new_node = new_node.update_coords(new_coords)

                return new_node

    def write_to_disk(self, fp: Path, write_history=False):
        out_traj = self.chain_trajectory[-1].to_trajectory()
        out_traj.write_trajectory(fp)

        if write_history:
            out_folder = fp.resolve().parent / (fp.stem + "_history")
            if not out_folder.exists():
                out_folder.mkdir()

            for i, chain in enumerate(self.chain_trajectory):
                fp = out_folder / f"traj_{i}.xyz"
                chain.write_to_disk(fp)
                
                
    def _calculate_chain_distances(self):
        chain_traj = self.chain_trajectory
        distances = [None] # None for the first chain
        for i,chain in enumerate(chain_traj):
            if i == 0 :
                continue
            
            prev_chain = chain_traj[i-1]
            dist = prev_chain._distance_to_chain(chain)
            distances.append(dist)
        return np.array(distances)
      
    def plot_chain_distances(self):
        distances = self._calculate_chain_distances()

        fs = 18
        s = 8
        kn = KneeLocator(x=list(range(len(distances)))[1:], y=distances[1:], curve='convex', direction='decreasing')


        f,ax = plt.subplots(figsize=(1.16*s, s))

        plt.text(.65,.9, s=f"elbow: {kn.elbow}\nelbow_yval: {round(kn.elbow_y,4)}", transform=ax.transAxes,fontsize=fs)

        plt.plot(distances,'o-')
        plt.yticks(fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.ylabel("Distance to previous chain",fontsize=fs)
        plt.xlabel("Chain id",fontsize=fs)

        plt.show()
      
    def plot_grad_delta_mag_history(self):
        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))
        projs = []
        
        for i, chain in enumerate(self.chain_trajectory):
            if i == 0: continue
            prev_chain = self.chain_trajectory[i-1]
            projs.append(prev_chain._gradient_delta_mags(chain))

        plt.plot(projs)
        plt.ylabel("NEB |∆gradient|",fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.xticks(fontsize=fs)
        # plt.ylim(0,1.1)
        plt.xlabel("Optimization step",fontsize=fs)
        plt.show()  
      
                
    def plot_projector_history(self, var='gradients'):
        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))
        projs = []
        
        for i, chain in enumerate(self.chain_trajectory):
            if i == 0: continue
            prev_chain = self.chain_trajectory[i-1]
            if var == 'gradients':
                projs.append(prev_chain._gradient_correlation(chain))
            elif var == 'tangents':
                projs.append(prev_chain._tangent_correlations(chain))
            else:
                raise ValueError(f"Unrecognized var: {var}")
        plt.plot(projs)
        plt.ylabel(f"NEB {var} correlation",fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.ylim(-1.1,1.1)
        plt.xlabel("Optimization step",fontsize=fs)
        plt.show()
        

    def plot_opt_history(self, do_3d=False):

        s = 8
        fs = 18
        
        if do_3d:
            all_chains = self.chain_trajectory


            ens = np.array([c.energies-c.energies[0] for c in all_chains])
            all_integrated_path_lengths = np.array([c.integrated_path_length for c in all_chains])
            opt_step = np.array(list(range(len(all_chains))))
            ax = plt.figure().add_subplot(projection='3d')

            # Plot a sin curve using the x and y axes.
            x = opt_step
            ys = all_integrated_path_lengths
            zs = ens
            for i, (xind, y) in enumerate(zip(x, ys)):
                if i < len(ys) -1:
                    ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='gray',markersize=3,alpha=.1)
                else:
                    ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='blue',markersize=3)
            ax.grid(False)

            ax.set_xlabel('optimization step')
            ax.set_ylabel('integrated path length')
            ax.set_zlabel('energy (hartrees)')

            # Customize the view angle so it's easier to see that the scatter points lie
            # on the plane y=0
            ax.view_init(elev=20., azim=-45, roll=0)
            plt.tight_layout()
            plt.show()
        
        else:
            f, ax = plt.subplots(figsize=(1.16 * s, s))

            
            for i, chain in enumerate(self.chain_trajectory):
                if i == len(self.chain_trajectory) - 1:
                    plt.plot(chain.integrated_path_length, chain.energies, "o-", alpha=1)
                else:
                    plt.plot(
                        chain.integrated_path_length,
                        chain.energies,
                        "o-",
                        alpha=0.1,
                        color="gray",
                    )

            plt.xlabel("Integrated path length", fontsize=fs)

            plt.ylabel("Energy (kcal/mol)", fontsize=fs)
            plt.xticks(fontsize=fs)
            plt.yticks(fontsize=fs)
            plt.show()


    def read_from_disk(fp: Path, history_folder: Path = None):
        if history_folder is None:
            history_folder = fp.parent / (str(fp.stem) + "_history")

        if not history_folder.exists():
            raise ValueError("No history exists for this. Cannot load object.")
        else:
            history_files = list(history_folder.glob("*.xyz"))
            history = [
                Chain.from_xyz(
                    history_folder / f"traj_{i}.xyz", parameters=ChainInputs()
                )
                for i, _ in enumerate(history_files)
            ]

        n = NEB2(
            initial_chain=history[0],
            parameters=NEBInputs(),
            optimized=history[-1],
            chain_trajectory=history,
        )
        return n

# -

# # MSMEP2

# +
from dataclasses import dataclass

from pathlib import Path

import numpy as np
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from retropaths.helper_functions import pairwise

from neb_dynamics.Chain import Chain
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Inputs import NEBInputs, ChainInputs, GIInputs
from neb_dynamics.helper_functions import _get_ind_minima, _get_ind_maxima, create_friction_optimal_gi
from neb_dynamics.TreeNode import TreeNode

@dataclass
class MSMEP2:

    neb_inputs: NEBInputs
    chain_inputs: ChainInputs 
    gi_inputs: GIInputs 

    # electronic structure params
    charge: int = 0
    spinmult: int = 1

    def create_endpoints_from_rxn_name(self, rxn_name, reactions_object):
        rxn = reactions_object[rxn_name]
        root = TDStructure.from_rxn_name(rxn_name, reactions_object)

        c3d_list = root.get_changes_in_3d(rxn)

        root = root.pseudoalign(c3d_list)
        root.gum_mm_optimization()
        root_opt = root.xtb_geom_optimization()

        target = root.copy()
        target.apply_changed3d_list(c3d_list)
        target_opt = target.xtb_geom_optimization()

        if not root_opt.molecule_rp.is_bond_isomorphic_to(root.molecule_rp):
            raise ValueError(
                "Pseudoaligned start molecule was not a minimum at this level of theory. Exiting."
            )

        if not target_opt.molecule_rp.is_bond_isomorphic_to(target.molecule_rp):
            raise ValueError(
                "Product molecule was not a minimum at this level of theory. Exiting."
            )

        return root_opt, target_opt
    
    def _input_chain_in_reference(self, input_chain):
        if input_chain[0].is_a_molecule:
            raise NotImplementedError
        else:
            cb = self.chain_inputs.cb
            ref = cb.reference_chain
            for input_node in input_chain.nodes[1:-1]:
                for reference_node in ref.nodes[1:-1]:
                    dist = np.linalg.norm(input_node.coords - reference_node.coords)
                    if dist < .1:
                        return True
                    

            return False

    def find_mep_multistep(self, input_chain):
        
        if input_chain[0].is_a_molecule:
            if input_chain[0]._is_connectivity_identical(input_chain[-1]):
                print("Endpoints are identical. Returning nothing")
                return None, None    
        
        if self.chain_inputs.do_chain_biasing:
            if self._input_chain_in_reference(input_chain):
                print("Found a minima in reference. Returning nothing.")
                return None, None
            
        
        else:
            if input_chain[0].is_identical(input_chain[-1]):
                print("Endpoints are identical. Returning nothing")
                return None, None
        
        root_neb_obj, chain = self.get_neb_chain(input_chain=input_chain)
        history = TreeNode(data=root_neb_obj, children=[])
        
        elem_step, split_method = chain.is_elem_step()
        
        if elem_step:
            return history, chain
       
        else:
            sequence_of_chains = self.make_sequence_of_chains(chain,split_method)
            print(f"Splitting chains based on: {split_method}")
            elem_steps = []

            for i, chain_frag in enumerate(sequence_of_chains):
                print(f"On chain {i+1} of {len(sequence_of_chains)}...")

                out_history, chain = self.find_mep_multistep(chain_frag)
                if chain:
                    elem_steps.append(chain)
                    history.children.append(out_history)

            stitched_elem_steps = self.stitch_elem_steps(elem_steps)
            return history, stitched_elem_steps

    def _create_interpolation(self, chain: Chain):

        if chain.parameters.use_geodesic_interpolation:
            traj = Trajectory(
                [node.tdstructure for node in chain],
                charge=self.charge,
                spinmult=self.spinmult,
            )
            if chain.parameters.friction_optimal_gi:
                gi = create_friction_optimal_gi(traj, self.gi_inputs)
            else:
                gi = traj.run_geodesic(
                    nimages=self.gi_inputs.nimages,
                    friction=self.gi_inputs.friction,
                    nudge=self.gi_inputs.nudge,
                    **self.gi_inputs.extra_kwds,
                )
            
            interpolation = Chain.from_traj(traj=gi, parameters=self.chain_inputs)
            interpolation._zero_velocity()

        else:  # do a linear interpolation using numpy
            start_point = chain[0].coords
            end_point = chain[-1].coords
            coords = np.linspace(start_point, end_point, self.gi_inputs.nimages)
            coords[1:-1] += np.random.normal(scale=0.00)

            interpolation = Chain.from_list_of_coords(
                list_of_coords=coords, parameters=self.chain_inputs
            )

        return interpolation

    def get_neb_chain(self, input_chain: Chain):
        
        if len(input_chain) < self.gi_inputs.nimages:
            interpolation = self._create_interpolation(input_chain)
        else:
            interpolation = input_chain

        n = NEB2(initial_chain=interpolation, parameters=self.neb_inputs)
        try:
            print("Running NEB calculation...")
            n.optimize_chain()
            out_chain = n.optimized

        except NoneConvergedException:
            print(
                "\nWarning! A chain did not converge. Returning an unoptimized chain..."
            )
            out_chain = n.chain_trajectory[-1]

        return n, out_chain
    

    def _make_chain_frag(self, chain: Chain, pair_of_inds):
        start, end = pair_of_inds
        chain_frag = chain.copy()
        chain_frag.nodes = chain[start : end + 1]
        opt_start = chain[start].do_geometry_optimization()
        opt_end = chain[end].do_geometry_optimization()

        chain_frag.insert(0, opt_start)
        chain_frag.append(opt_end)

        return chain_frag

    def _make_chain_pair(self, chain: Chain, pair_of_inds):
        start, end = pair_of_inds
        start_opt = chain[start].do_geometry_optimization()
        end_opt = chain[end].do_geometry_optimization()

        chain_frag = Chain2(nodes=[start_opt, end_opt], parameters=chain.parameters)

        return chain_frag


    def _do_minima_based_split(self, chain):
        all_inds = [0]
        ind_minima = _get_ind_minima(chain)
        all_inds.extend(ind_minima)
        all_inds.append(len(chain) - 1)

        pairs_inds = list(pairwise(all_inds))

        chains = []
        for ind_pair in pairs_inds:
            chains.append(self._make_chain_frag(chain, ind_pair))

        return chains

    def _do_maxima_based_split(self, chain: Chain):
        
        
        
        ind_maxima = _get_ind_maxima(chain)
        r, p = chain._approx_irc(index=ind_maxima)
        chains_list = []
        
        # add the input start, to R
        nodes = [chain[0], r]
        chain_frag = chain.copy()
        chain_frag.nodes = nodes
        chains_list.append(chain_frag)

        # add the r to p, passing through the maxima
        nodes2 = [r, chain[ind_maxima], p]
        chain_frag2 = chain.copy()
        chain_frag2.nodes = nodes2
        chains_list.append(chain_frag2)

        # add the p to final chain
        nodes3 = [p, chain[len(chain)-1]]
        chain_frag3 = chain.copy()
        chain_frag3.nodes = nodes3
        chains_list.append(chain_frag3)
        
        
        return chains_list

    def make_sequence_of_chains(self, chain, split_method):
        if split_method == 'minima':
            chains = self._do_minima_based_split(chain)

        elif split_method == 'maxima':
            chains = self._do_maxima_based_split(chain)

        return chains


    def stitch_elem_steps(self, list_of_chains):
        out_list_of_chains = [chain for chain in list_of_chains if chain is not None]
        return Chain.from_list_of_chains(
            out_list_of_chains, parameters=self.chain_inputs
        )
        

        
    
    
    




# -

# # Testing it out

# +
ss = 0.01
amp = 50
sig = .5
distance_func = 'simp_frechet'
# distance_func = 'per_node'
# distance_func = 'foobar'

cb = ChainBiaser(reference_chain=bias_chain, amplitude=amp, sigma=sig, distance_func=distance_func)

# cni = ChainInputs(step_size=1,min_step_size=0.001, node_class=Node2D_Zero, k=0.0, do_parallel=False)
cni = ChainInputs(step_size=.1,min_step_size=0.001, node_class=node_to_use, k=5, delta_k=0, do_parallel=False)
cni.do_chain_biasing = True
cni.cb = cb 

 


init_chain = Chain2(n_ref.chain_trajectory[0].nodes, parameters=cni)
# init_chain = Chain2(n_ref.chain_trajectory[30].nodes, parameters=cni)

for i, node in enumerate(init_chain):
    # init_chain.nodes[i] = Node2D_Zero(pair_of_coordinates=node.coords)
    init_chain.nodes[i] = node_to_use(pair_of_coordinates=node.coords)
    
# -

n = NEB2(initial_chain=init_chain,
       parameters=NEBInputs(v=True, max_steps=100, tol=tol))

n.optimize_chain()

# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)


# cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

reference = bias_chain



plot_chain(reference, color='white', label='reference')
plot_chain(init_chain)
plot_chain(n.initial_chain,color='yellow', label='initial guess')

plot_chain(n.chain_trajectory[-1], color='skyblue', label='biased')

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

init_chain = Chain2(n_ref.chain_trajectory[0].nodes, parameters=cni)
# init_chain = Chain2(n_ref.chain_trajectory[30].nodes, parameters=cni)

for i, node in enumerate(init_chain):
    # init_chain.nodes[i] = Node2D_Zero(pair_of_coordinates=node.coords)
    init_chain.nodes[i] = node_to_use(pair_of_coordinates=node.coords)
    


m = MSMEP2(neb_inputs=nbi_msmep,chain_inputs=cni, gi_inputs=gii)
history, out_chain = m.find_mep_multistep(init_chain)
# -

relaxed_chains = []
for biased_out_leaf in history.ordered_leaves:
    biased_out_chain = biased_out_leaf.data.optimized
    init_copy = biased_out_chain.copy()
    init_copy.parameters = cni
    
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

reference = bias_chain



plot_chain(reference, color='white', label='reference AS-NEB')
plot_chain(init_chain)
plot_chain(n.initial_chain,color='yellow', label='initial guess')

plot_chain(n_ref.chain_trajectory[-1], color='blue', label='SD NEB')
plot_chain(out_chain, color='red', label='biased AS-NEB')
plot_chain(relaxed_out_chain, color='orange', label='relaxed biased AS-NEB')


plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.legend(fontsize=fs, bbox_to_anchor=(1.8,1), shadow=True, facecolor='lightgray')
plt.show()

# +
plt.plot(out_chain.integrated_path_length, out_chain.energies, 'o-',label=f'work: {round(path_work(out_chain),3)}', color='red')
plt.plot(relaxed_out_chain.integrated_path_length, relaxed_out_chain.energies, 'o-',label=f'work: {round(path_work(out_chain),3)}', color='orange')


plt.plot(n_ref.initial_chain.integrated_path_length,  n_ref.initial_chain.energies,'o-', label=f'work: {round(path_work(n_ref.initial_chain),3)}', color='yellow')

plt.plot(out_chain_ref.integrated_path_length,  out_chain_ref.energies,'o-', label=f'work: {round(path_work(out_chain_ref),3)}', color='lightgray')


plt.plot(n_ref.chain_trajectory[-1].integrated_path_length,  n_ref.chain_trajectory[-1].energies,'o-', label=f'work: {path_work(n_ref.chain_trajectory[-1])}', color='blue')

plt.legend(loc='upper right')

# -



