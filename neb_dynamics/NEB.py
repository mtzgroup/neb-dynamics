from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from typing import Union, Tuple, List
# from hashlib import new
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openbabel import pybel

from neb_dynamics.Chain import Chain
from neb_dynamics.errors import (ElectronicStructureError,
                                 NoneConvergedException)
from neb_dynamics.helper_functions import pairwise
from neb_dynamics.Inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.Node import Node
from nodes.node3d import Node3D
from neb_dynamics.Optimizer import Optimizer
from neb_dynamics.optimizers import ALS
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)

VELOCITY_SCALING = .3
ACTIVATION_TOL = 100


@dataclass
class NEB:
    """
    Class for running, storing, and visualizing nudged elastic band minimizations.
    Main functions to use are:
    - self.optimize_chain()
    - self.plot_opt_history()


    """
    initial_chain: Chain
    parameters: NEBInputs
    optimizer: Optimizer

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)

    def __post_init__(self):
        self.n_steps_still_chain = 0
        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0

    def _reset_node_convergence(self, chain) -> None:
        for node in chain:
            node.converged = False

    def set_climbing_nodes(self, chain: Chain) -> None:
        if self.parameters.climb:
            inds_maxima = [chain.energies.argmax()]

            if self.parameters.v > 0:
                print(f"\n----->Setting {len(inds_maxima)} nodes to climb\n")

            for ind in inds_maxima:
                chain[ind].do_climb = True

    def _do_early_stop_check(self, chain: Chain):
        """
        this function calls geometry minimizations to verify if
        chain is an elementary step

        Args:
            chain (Chain): chain to check

        Returns:
            tuple(boolean, tuple(), int) : boolean of whether
                    to stop early, a tuple containing minimization
                    results, and an integer number of gradient
                    calls used in getting the minimization results
        """
        # TODO: change the output of is_elem_step to a \
        # dictionary or some other object

        elem_step_results = chain.is_elem_step()

        is_elem_step, split_method, minimization_results, \
            n_geom_opt_grad_calls = elem_step_results

        if not is_elem_step:
            print("\nStopped early because chain is not an elementary step.")
            print(f"Split chain based on: {split_method}")
            self.optimized = chain
            return True, elem_step_results, n_geom_opt_grad_calls

        else:
            self.n_steps_still_chain = 0
            return False, elem_step_results, n_geom_opt_grad_calls

    def _check_early_stop(self, chain: Chain):
        """
        this function computes chain distances and checks gradient
        values in order to decide whether the expensive minimization of
        the chain should be done.
        """
        # max_grad_val = chain.get_maximum_grad_magnitude()
        # max_grad_val = np.linalg.norm(chain.get_g_perps())
        ind_ts_guess = np.argmax(chain.energies)
        ts_guess_grad = np.amax(np.abs(chain.get_g_perps()[ind_ts_guess]))

        dist_to_prev_chain = chain._distance_to_chain(
            self.chain_trajectory[-2])  # the -1 is the chain im looking at
        if dist_to_prev_chain < self.parameters.early_stop_chain_rms_thre:
            self.n_steps_still_chain += 1
        else:
            self.n_steps_still_chain = 0

        correlation = self.chain_trajectory[-2]._gradient_correlation(chain)
        conditions = [
            ts_guess_grad <= self.parameters.early_stop_force_thre,
            dist_to_prev_chain <= self.parameters.early_stop_chain_rms_thre,
            correlation >= self.parameters.early_stop_corr_thre,
            self.n_steps_still_chain >= self.parameters.early_stop_still_steps_thre
        ]
        # if any(conditions):
        # if we've dipped below the force thre and chain rms is low
        if (conditions[0] and conditions[1]) or conditions[3]:
            # or chain has stayed still for a long time
            # dont reset them if you stopped due to stillness
            if (conditions[0] and conditions[1]):
                new_params = self.parameters.copy()

                # reset early stop checks
                new_params.early_stop_force_thre = 0.0
                new_params.early_stop_chain_rms_thre = 0.0
                new_params.early_stop_corr_thre = 10.
                new_params.early_stop_still_steps_thre = 100000

                self.parameters = new_params

            # going to set climbing nodes when checking early stop
            if self.parameters.climb:
                self.set_climbing_nodes(chain=chain)
                self.parameters.climb = False  # no need to set climbing nodes again

            stop_early, elem_step_results, geom_n_grad_calls = self._do_early_stop_check(
                chain)
            return stop_early, elem_step_results, geom_n_grad_calls

        else:
            return False, None, 0

    # @Jan: This should be a more general function so that the
    # lower level of theory can be whatever the user wants.
    def _do_xtb_preopt(self, chain) -> Chain:  #
        """
        This function will loosely minimize an input chain using the GFN2-XTB method,
        then return a new chain which can be used as an initial guess for a higher
        level of theory calculation
        """

        xtb_params = chain.parameters.copy()
        xtb_params.node_class = Node3D
        chain_traj = chain.to_trajectory()
        xtb_chain = Chain.from_traj(chain_traj, parameters=xtb_params)
        xtb_nbi = NEBInputs(tol=self.parameters.tol*10,
                            v=True, preopt_with_xtb=False, max_steps=1000)

        opt_xtb = VelocityProjectedOptimizer(timestep=1)
        n = NEB(initial_chain=xtb_chain, parameters=xtb_nbi, optimizer=opt_xtb)
        try:
            _ = n.optimize_chain()
            print(
                f"\nConverged an xtb chain in {len(n.chain_trajectory)} steps")
        except Exception:
            print(
                f"\nCompleted {len(n.chain_trajectory)} xtb steps. Did not converge.")

        xtb_seed_tr = n.chain_trajectory[-1].to_trajectory()
        xtb_seed_tr.update_tc_parameters(chain[0].tdstructure)

        xtb_seed = Chain.from_traj(
            xtb_seed_tr, parameters=chain.parameters.copy())
        xtb_seed.gradients  # calling it to cache the values

        return xtb_seed

    # this craziness output is an elem step results. @Jan: Refactor ASAP.
    def optimize_chain(self) -> Tuple(bool, str, Union[list, Chain], int):
        """
        Main function. After an NEB object has been created, running this function will
        minimize the chain and return the elementary step results from the final minimized chain.

        Running this function will populate the `.chain_trajectory` object variable, which
        contains the history of the chains minimized. Once it is completed, you can use
        `.plot_opt_history()` to view the optimization over time.

        Args:
            self: initialized NEB object
        Raises:
            NoneConvergedException: If chain did not converge in alloted steps.
        """

        nsteps = 1
        nsteps_negative_grad_corr = 0

        if self.parameters.preopt_with_xtb:
            chain_previous = self._do_xtb_preopt(self.initial_chain)
            self.chain_trajectory.append(chain_previous)

            stop_early, elem_step_results, geom_n_grad_calls = self._do_early_stop_check(
                chain_previous)
            self.geom_grad_calls_made += geom_n_grad_calls
            if stop_early:
                return elem_step_results
        else:
            chain_previous = self.initial_chain.copy()
            self.chain_trajectory.append(chain_previous)
        chain_previous._zero_velocity()

        while nsteps < self.parameters.max_steps + 1:
            if nsteps > 1:
                stop_early, elem_step_results, geom_n_grad_calls = self._check_early_stop(
                    chain_previous)
                self.geom_grad_calls_made += geom_n_grad_calls
                if stop_early:
                    return elem_step_results
                else:
                    if elem_step_results:
                        is_elem_step, split_method, minimization_results, geom_n_grad_calls = elem_step_results
                        self.geom_grad_calls_made += geom_n_grad_calls
                        if isinstance(minimization_results, Chain):
                            chain_previous = minimization_results

            new_chain = self.update_chain(chain=chain_previous)
            # max_grad_val = np.amax(np.abs(new_chain.gradients))
            max_rms_grad_val = np.amax(new_chain.rms_gperps)
            ind_ts_guess = np.argmax(new_chain.energies)
            ts_guess_grad = np.amax(
                np.abs(new_chain.get_g_perps()[ind_ts_guess]))
            chain_converged = self._chain_converged(
                chain_prev=chain_previous, chain_new=new_chain)

            # print([node._cached_energy for node in new_chain])
            # print([node.converged for node in new_chain])

            n_nodes_frozen = 0
            for node in new_chain:
                if node.converged:
                    n_nodes_frozen += 1

            grad_calls_made = len(new_chain) - n_nodes_frozen
            self.grad_calls_made += grad_calls_made

            grad_corr = new_chain._gradient_correlation(chain_previous)
            if grad_corr < 0:
                nsteps_negative_grad_corr += 1
            else:
                nsteps_negative_grad_corr = 0

            if nsteps_negative_grad_corr >= self.parameters.negative_steps_thre:
                print("\nstep size causing oscillations. decreasing by 50%")
                self.optimizer.timestep *= 0.5
                nsteps_negative_grad_corr = 0

            if self.parameters.v:

                print(
                    f"step {nsteps} // argmax(|TS gperp|) {np.amax(np.abs(ts_guess_grad))} // max rms grad {max_rms_grad_val} // armax(|TS_triplet_gsprings|) {new_chain.ts_triplet_gspring_infnorm} // nodes_frozen {n_nodes_frozen} // {grad_corr}{' '*20}", end="\r"
                )
                sys.stdout.flush()

            self.chain_trajectory.append(new_chain)
            self.gradient_trajectory.append(new_chain.gradients)

            if chain_converged:
                if self.parameters.v:
                    print("\nChain converged!")

                # N.B. One could skip this if you don't want to do minimization on converged chain.
                elem_step_results = new_chain.is_elem_step()
                is_elem_step, split_method, minimization_results, grad_calls_geom = elem_step_results
                self.geom_grad_calls_made += grad_calls_geom
                # self.optimized = new_chain
                if is_elem_step:
                    # print(f"\nUsing resampled chain as output. \n\tEns: {minimization_results.energies}\n\t Grads:{minimization_results.gradients}")

                    minimization_results.gradients  # making sure they're cached for writeup
                    # Controversial! Now the optimized chain is the 'resampled' chain from IRC check
                    self.optimized = minimization_results
                    self.chain_trajectory.append(minimization_results)
                else:
                    self.optimized = new_chain
                return elem_step_results

            chain_previous = new_chain  # previous

            nsteps += 1

        new_chain = self.update_chain(chain=chain_previous)
        if not self._chain_converged(chain_prev=chain_previous, chain_new=new_chain):
            raise NoneConvergedException(
                trajectory=self.chain_trajectory,
                msg=f"\nChain did not converge at step {nsteps}",
                obj=self,
            )

    def update_chain(self, chain: Chain) -> Chain:
        grad_step = chain.gradients
        new_chain = self.optimizer.optimize_step(
            chain=chain, chain_gradients=grad_step)
        return new_chain

    def _copy_node_information_to_converged(self, new_chain, old_chain):
        for new_node, old_node in zip(new_chain.nodes, old_chain.nodes):
            if old_node.converged:
                new_node._cached_energy = old_node._cached_energy
                new_node._cached_gradient = old_node._cached_gradient

    def _update_node_convergence(self, chain: Chain, indices: np.array, prev_chain: Chain) -> None:
        for i, (node, prev_node) in enumerate(zip(chain, prev_chain)):
            if i in indices:
                if prev_node._cached_energy is not None and prev_node._cached_gradient is not None:
                    node.converged = True
                    node._cached_energy = prev_node._cached_energy
                    node._cached_gradient = prev_node._cached_gradient
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

        return np.where(bools), max_grad_components

    def _check_barrier_height_conv(self, chain_prev: Chain, chain_new: Chain):
        prev_eA = chain_prev.get_eA_chain()
        new_eA = chain_new.get_eA_chain()

        delta_eA = np.abs(new_eA - prev_eA)
        return delta_eA <= self.parameters.barrier_thre

    def _check_rms_grad_converged(self, chain: Chain):

        bools = []
        rms_gperps = []

        for rms_gp, rms_gradient in zip(chain.rms_gperps, chain.rms_gradients):
            rms_gperps.append(rms_gp)

            # the boolens are used exclusively for deciding to freeze nodes or not
            # I want the spring forces to affect whether a node is frozen, but not
            # to affect the total chain's convergence.
            rms_grad_converged = rms_gradient <= self.parameters.rms_grad_thre
            bools.append(rms_grad_converged)

        return np.where(bools), rms_gperps

    def _chain_converged(self, chain_prev: Chain, chain_new: Chain) -> bool:

        rms_grad_conv_ind, rms_gperps = self._check_rms_grad_converged(
            chain_new)
        ts_triplet_gspring = chain_new.ts_triplet_gspring_infnorm
        en_converged_indices, en_deltas = self._check_en_converged(
            chain_prev=chain_prev, chain_new=chain_new
        )

        grad_conv_ind, max_grad_components = self._check_grad_converged(
            chain=chain_new)

        converged_nodes_indices = np.intersect1d(
            en_converged_indices, rms_grad_conv_ind
        )
        # converged_nodes_indices = np.intersect1d(converged_nodes_indices, grad_conv_ind)
        ind_ts_node = chain_new.energies.argmax()
        # never freeze TS node
        converged_nodes_indices = converged_nodes_indices[converged_nodes_indices != ind_ts_node]

        if chain_new.parameters.node_freezing:
            self._update_node_convergence(
                chain=chain_new, indices=converged_nodes_indices, prev_chain=chain_prev)
            self._copy_node_information_to_converged(
                new_chain=chain_new, old_chain=chain_prev)

        if self.parameters.v > 1:
            print("\n")
            [
                print(
                    f"\t\tnode{i} | ∆E : {en_deltas[i]} | RMS G_perp: {rms_gperps[i]} | Max(Grad components): {max_grad_components[i]} | Converged? : {chain_new.nodes[i].converged}"
                )
                for i in range(len(chain_new))
            ]
        if self.parameters.v > 1:
            print(f"\t{len(converged_nodes_indices)} nodes have converged")

        barrier_height_converged = self._check_barrier_height_conv(
            chain_prev=chain_prev, chain_new=chain_new)
        ind_ts_guess = np.argmax(chain_new.energies)
        ts_guess = chain_new[ind_ts_guess]
        ts_guess_grad = np.amax(np.abs(chain_new.get_g_perps()[ind_ts_guess]))

        criteria_converged = [
            max(rms_gperps) <= self.parameters.max_rms_grad_thre,
            # max(max_grad_components) <= self.parameters.grad_thre,
            sum(rms_gperps)/len(chain_new) <= self.parameters.rms_grad_thre,
            ts_guess_grad <= self.parameters.ts_grad_thre,
            ts_triplet_gspring <= self.parameters.ts_spring_thre,
            barrier_height_converged]
        # max(en_deltas) <= self.parameters.en_thre,
        # return len(converged_nodes_indices) == len(chain_new)
        criteria_names = ['Max(RMS_GPERP)', 'AVG_RMS_GPERP', 'TS_GRAD',
                          'TS_SPRINGS', 'BARRIER_THRE']  # , 'ENE_DELTAS']

        converged = sum(criteria_converged) >= 5
        if converged and self.parameters.v:
            print(
                f"\nConverged on conditions: {[criteria_names[ind] for ind, b in enumerate(criteria_converged) if b]}")

        return converged

    def _check_dot_product_converged(self, chain: Chain) -> bool:
        dps = []
        for prev_node, current_node, next_node in chain.iter_triplets():
            vec1 = current_node.coords - prev_node.coords
            vec2 = next_node.coords - current_node.coords
            dps.append(current_node.dot_function(vec1, vec2) > 0)

        return all(dps)

    def write_to_disk(self, fp: Path, write_history=True):
        # write output chain
        self.chain_trajectory[-1].write_to_disk(fp)

        if write_history:
            out_folder = fp.resolve().parent / (fp.stem + "_history")

            if out_folder.exists():
                shutil.rmtree(out_folder)

            if not out_folder.exists():
                out_folder.mkdir()

            for i, chain in enumerate(self.chain_trajectory):
                fp = out_folder / f"traj_{i}.xyz"
                chain.write_to_disk(fp)

    def _calculate_chain_distances(self):
        chain_traj = self.chain_trajectory
        distances = [None]  # None for the first chain
        for i, chain in enumerate(chain_traj):
            if i == 0:
                continue

            prev_chain = chain_traj[i-1]
            dist = prev_chain._distance_to_chain(chain)
            distances.append(dist)
        return np.array(distances)

    def plot_chain_distances(self):
        distances = self._calculate_chain_distances()

        fs = 18
        s = 8

        f, ax = plt.subplots(figsize=(1.16*s, s))

        plt.plot(distances, 'o-')
        plt.yticks(fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.ylabel("Distance to previous chain", fontsize=fs)
        plt.xlabel("Chain id", fontsize=fs)

        plt.show()

    def plot_grad_delta_mag_history(self):
        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))
        projs = []

        for i, chain in enumerate(self.chain_trajectory):
            if i == 0:
                continue
            prev_chain = self.chain_trajectory[i-1]
            projs.append(prev_chain._gradient_delta_mags(chain))

        plt.plot(projs)
        plt.ylabel("NEB |∆gradient|", fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.xticks(fontsize=fs)
        # plt.ylim(0,1.1)
        plt.xlabel("Optimization step", fontsize=fs)
        plt.show()

    def plot_projector_history(self, var='gradients'):
        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))
        projs = []

        for i, chain in enumerate(self.chain_trajectory):
            if i == 0:
                continue
            prev_chain = self.chain_trajectory[i-1]
            if var == 'gradients':
                projs.append(prev_chain._gradient_correlation(chain))
            elif var == 'tangents':
                projs.append(prev_chain._tangent_correlations(chain))
            else:
                raise ValueError(f"Unrecognized var: {var}")
        plt.plot(projs)
        plt.ylabel(f"NEB {var} correlation", fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.ylim(-1.1, 1.1)
        plt.xlabel("Optimization step", fontsize=fs)
        plt.show()

    def plot_opt_history(self, do_3d=False):

        s = 8
        fs = 18

        if do_3d:
            all_chains = self.chain_trajectory

            ens = np.array([c.energies-c.energies[0] for c in all_chains])
            all_integrated_path_lengths = np.array(
                [c.integrated_path_length for c in all_chains])
            opt_step = np.array(list(range(len(all_chains))))
            s = 7
            fs = 18
            ax = plt.figure(figsize=(1.16*s, s)).add_subplot(projection='3d')

            # Plot a sin curve using the x and y axes.
            x = opt_step
            ys = all_integrated_path_lengths
            zs = ens
            for i, (xind, y) in enumerate(zip(x, ys)):
                if i < len(ys) - 1:
                    ax.plot([xind]*len(y), y, 'o-', zs=zs[i],
                            color='gray', markersize=3, alpha=.1)
                else:
                    ax.plot([xind]*len(y), y, 'o-', zs=zs[i],
                            color='blue', markersize=3)
            ax.grid(False)

            ax.set_xlabel('optimization step', fontsize=fs)
            ax.set_ylabel('integrated path length', fontsize=fs)
            ax.set_zlabel('energy (hartrees)', fontsize=fs)

            # Customize the view angle so it's easier to see that the scatter points lie
            # on the plane y=0
            ax.view_init(elev=20., azim=-45)
            plt.tight_layout()
            plt.show()

        else:
            f, ax = plt.subplots(figsize=(1.16 * s, s))

            for i, chain in enumerate(self.chain_trajectory):
                if i == len(self.chain_trajectory) - 1:
                    plt.plot(chain.integrated_path_length,
                             chain.energies, "o-", alpha=1)
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

    def plot_convergence_metrics(self, do_indiv=False):
        ct = self.chain_trajectory

        avg_rms_gperp = []
        max_rms_gperp = []
        avg_rms_g = []
        barr_height = []
        ts_gperp = []
        inf_norm_g = []
        inf_norm_gperp = []

        for ind in range(1, len(ct)):
            avg_rms_g.append(
                sum(ct[ind].rms_gradients[1:-1]) / (len(ct[ind])-2))
            avg_rms_gperp.append(
                sum(ct[ind].rms_gperps[1:-1]) / (len(ct[ind])-2))
            max_rms_gperp.append(max(ct[ind].rms_gperps))
            barr_height.append(
                abs(ct[ind].get_eA_chain() - ct[ind-1].get_eA_chain()))
            ts_node_ind = ct[ind].energies.argmax()
            ts_node_gperp = np.max(ct[ind].get_g_perps()[ts_node_ind])
            ts_gperp.append(ts_node_gperp)
            inf_norm_val_g = inf_norm_g.append(np.max(ct[ind].gradients))
            inf_norm_val_gperp = inf_norm_gperp.append(
                np.max(ct[ind].get_g_perps()))

        if do_indiv:
            f, ax = plt.subplots()
            plt.plot(avg_rms_gperp, label='RMS Grad$_{\perp}$')
            plt.ylabel("Gradient data")
            xmin = ax.get_xlim()[0]
            xmax = ax.get_xlim()[1]
            ax.hlines(y=self.parameters.rms_grad_thre, xmin=xmin, xmax=xmax,
                      label='rms_grad_thre', linestyle='--', color='blue')
            f.legend()
            plt.show()

            f, ax = plt.subplots()
            plt.plot(max_rms_gperp, label='Max RMS Grad$_{\perp}$')
            plt.ylabel("Gradient data")
            xmin = ax.get_xlim()[0]
            xmax = ax.get_xlim()[1]
            ax.hlines(y=self.parameters.max_rms_grad_thre, xmin=xmin, xmax=xmax,
                      label='max_rms_grad_thre', linestyle='--', color='orange')
            f.legend()
            plt.show()

            f, ax = plt.subplots()
            plt.ylabel("Gradient data")
            plt.plot(ts_gperp, label='TS gperp')
            xmin = ax.get_xlim()[0]
            xmax = ax.get_xlim()[1]
            ax.hlines(y=self.parameters.ts_grad_thre, xmin=xmin, xmax=xmax,
                      label='ts_grad_thre', linestyle='--', color='green')
            f.legend()
            plt.show()

            f, ax = plt.subplots()
            plt.plot(barr_height, 'o--',
                     label='barr_height_delta', color='purple')
            plt.ylabel("Barrier height data")
            ax.hlines(y=self.parameters.barrier_thre, xmin=xmin, xmax=xmax,
                      label='barrier_thre', linestyle='--', color='purple')
            xmin = ax.get_xlim()[0]
            xmax = ax.get_xlim()[1]
            f.legend()
            plt.show()

            plt.show()

        else:

            f, ax = plt.subplots()
            plt.plot(avg_rms_gperp, label='RMS Grad$_{\perp}$')
            plt.plot(max_rms_gperp, label='Max RMS Grad$_{\perp}$')
            # plt.plot(avg_rms_g, label='RMS Grad')
            plt.plot(ts_gperp, label='TS gperp')
            # plt.plot(inf_norm_g,label='Inf norm G')
            # plt.plot(inf_norm_gperp,label='Inf norm Gperp')
            plt.ylabel("Gradient data")

            xmin = ax.get_xlim()[0]
            xmax = ax.get_xlim()[1]
            ax.hlines(y=self.parameters.rms_grad_thre, xmin=xmin, xmax=xmax,
                      label='rms_grad_thre', linestyle='--', color='blue')
            ax.hlines(y=self.parameters.max_rms_grad_thre, xmin=xmin, xmax=xmax,
                      label='max_rms_grad_thre', linestyle='--', color='orange')
            ax.hlines(y=self.parameters.ts_grad_thre, xmin=xmin, xmax=xmax,
                      label='ts_grad_thre', linestyle='--', color='green')

            ax2 = plt.twinx()
            plt.plot(barr_height, 'o--',
                     label='barr_height_delta', color='purple')
            plt.ylabel("Barrier height data")

            ax2.hlines(y=self.parameters.barrier_thre, xmin=xmin, xmax=xmax,
                       label='barrier_thre', linestyle='--', color='purple')

            f.legend()
            plt.show()

    def read_from_disk(fp: Path, history_folder: Path = None,
                       chain_parameters=ChainInputs(),
                       neb_parameters=NEBInputs(),
                       gi_parameters=GIInputs(),
                       optimizer=VelocityProjectedOptimizer()):
        if isinstance(fp, str):
            fp = Path(fp)

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
            optimizer=optimizer
        )
        return n
