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
class NEB:
    initial_chain: Chain
    parameters: NEBInputs

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)

    def __post_init__(self):
        self.n_steps_still_chain = 0
        
    def do_velvel(self, chain: Chain):
        max_grad_val = chain.get_maximum_grad_magnitude()
        return max_grad_val < self.parameters.vv_force_thre

    def _reset_node_convergence(self, chain):
        for node in chain:
            node.converged = False

    def set_climbing_nodes(self, chain: Chain):
        # reset node convergence
        self._reset_node_convergence(chain=chain)

        inds_maxima = argrelextrema(chain.energies, np.greater, order=2)[0]
        if self.parameters.v > 1:
            print(f"----->Setting {len(inds_maxima)} nodes to climb")

        for ind in inds_maxima:
            chain[ind].do_climb = True


    def _check_early_stop(self, chain: Chain):
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

    def get_chain_velocity(self, chain: Chain) -> np.array:
        
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

        new_chain = Chain(new_nodes, parameters=chain.parameters)
        return new_chain

    def _update_node_convergence(self, chain: Chain, indices: np.array) -> None:
        for i, node in enumerate(chain):
            if i in indices:
                node.converged = True
            else:
                node.converged = False

    def _check_en_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
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

    def _check_rms_grad_converged(self, chain: Chain):
        bools = []
        rms_grads = []
        grads = chain.gradients
        for grad in grads:
            rms_gradient = np.sqrt(np.mean(np.square(grad.flatten())) / len(grad))
            rms_grads.append(rms_gradient)
            rms_grad_converged = rms_gradient <= self.parameters.rms_grad_thre
            bools.append(rms_grad_converged)

        return np.where(bools), rms_grads

    def _chain_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:
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

    def remove_chain_folding(self, chain: Chain) -> Chain:
        not_converged = True
        count = 0
        points_removed = []
        while not_converged:
            if self.parameters.v > 1:
                print(f"anti-folding: on count {count}...")
            new_chain = []
            new_chain.append(chain[0])

            for prev_node, current_node, next_node in chain.iter_triplets():
                vec1 = current_node.coords - prev_node.coords
                vec2 = next_node.coords - current_node.coords

                if np.all(current_node.dot_function(vec1, vec2)) > 0:
                    new_chain.append(current_node)
                else:
                    points_removed.append(current_node)

            new_chain.append(chain[-1])
            new_chain = Chain(nodes=new_chain, parameters=chain.parameters)
            if self._check_dot_product_converged(new_chain):
                not_converged = False
            chain = new_chain.copy()
            count += 1

        return chain

    def _check_dot_product_converged(self, chain: Chain) -> bool:
        dps = []
        for prev_node, current_node, next_node in chain.iter_triplets():
            vec1 = current_node.coords - prev_node.coords
            vec2 = next_node.coords - current_node.coords
            dps.append(current_node.dot_function(vec1, vec2) > 0)

        return all(dps)

    def redistribute_chain(self, chain: Chain, requested_length_of_chain: int) -> Chain:
        # if len(chain) < requested_length_of_chain:
        #     fixed_chain = chain.copy()
        #     [
        #         fixed_chain.nodes.insert(1, fixed_chain[1])
        #         for _ in range(requested_length_of_chain - len(chain))
        #     ]
        #     chain = fixed_chain

        direction = np.array(
            [
                next_node.coords - current_node.coords
                for current_node, next_node in pairwise(chain)
            ]
        )
        distances = np.linalg.norm(direction, axis=1)
        tot_dist = np.sum(distances)
        cumsum = np.cumsum(distances)  # cumulative sum
        cumsum = np.insert(cumsum, 0, 0)

        distributed_chain = []
        for num in np.linspace(0, tot_dist, len(chain)):
            new_node = self.redistribution_helper(num=num, cum=cumsum, chain=chain)

            distributed_chain.append(new_node)

        distributed_chain[0] = chain[0]
        distributed_chain[-1] = chain[-1]

        return Chain(distributed_chain, parameters=chain)

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


    def read_from_disk(fp: Path, history_folder: Path = None, chain_parameters=ChainInputs(), neb_parameters=NEBInputs()):
        if history_folder is None:
            history_folder = fp.parent / (str(fp.stem) + "_history")

        if not history_folder.exists():
            raise ValueError("No history exists for this. Cannot load object.")
        else:
            history_files = list(history_folder.glob("*.xyz"))
            history = [
                Chain.from_xyz(
                    history_folder / f"traj_{i}.xyz", parameters=chain_parameters
                )
                for i, _ in enumerate(history_files)
            ]

        n = NEB(
            initial_chain=history[0],
            parameters=neb_parameters,
            optimized=history[-1],
            chain_trajectory=history,
        )
        return n
