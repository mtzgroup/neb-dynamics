from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from typing import Tuple, Union, Any, Dict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openbabel import pybel

from chain import Chain
from neb_dynamics.errors import NoneConvergedException
from neb_dynamics.inputs import ChainInputs, GIInputs, NEBInputs
from nodes.node import Node
from neb_dynamics.Optimizer import Optimizer
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.convergence_helpers import chain_converged
from neb_dynamics.helper_functions import _calculate_chain_distances
from neb_dynamics.elementarystep import ElemStepResults, check_if_elem_step
from neb_dynamics.engine import Engine, QCOPEngine
from qcio.models.inputs import ProgramInput

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

    !!! Note
        Colton said this rocks

    """
    initial_chain: Chain
    optimizer: Optimizer
    parameters: NEBInputs

    engine_type: str = 'qcop'
    engine_inputs: Dict = field(default_factory=dict)

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)

    def __post_init__(self):
        self.n_steps_still_chain = 0
        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0

        # input checks on engine
        if self.engine_type in ['qcop', 'chemcloud']:
            prog_inp_key_exists = "program_input" in self.engine_inputs.keys()
            prog_key_exists = "program" in self.engine_inputs.keys()

            err_msg1 = f"If using {self.engine_type} you need to specify a `program_input` in the engine_inputs"
            err_msg2 = f"If using {self.engine_type} you need to specify a `program` in the engine_inputs"
            if prog_inp_key_exists:
                assert self.engine_inputs["program_input"] is not None, err_msg1
            else:
                raise AssertionError(err_msg1)

            if prog_key_exists:
                assert self.engine_inputs["program"] is not None, err_msg2
            else:
                raise AssertionError(err_msg2)

    @property
    def engine(self) -> Engine:
        if self.engine_type == 'qcop':
            eng = QCOPEngine(**self.engine_inputs)
            return eng
        else:
            raise NotImplementedError

    def _reset_node_convergence(self, chain) -> None:
        for node in chain:
            node.converged = False

    def set_climbing_nodes(self, chain: Chain) -> None:
        """Iterates through chain and sets the nodes that should climb.

        Args:
            chain: chain to set inputs for
        """
        if self.parameters.climb:
            inds_maxima = [chain.energies.argmax()]

            if self.parameters.v > 0:
                print(f"\n----->Setting {len(inds_maxima)} nodes to climb\n")

            for ind in inds_maxima:
                chain[ind].do_climb = True

    def _do_early_stop_check(self, chain: Chain) -> Tuple[bool, ElemStepResults]:
        """
        this function calls geometry minimizations to verify if
        chain is an elementary step

        Args:
            chain (Chain): chain to check

        Returns:
            tuple(boolean, ElemStepResults) : boolean of whether
                    to stop early, and an ElemStepResults objects
        """

        elem_step_results = check_if_elem_step(chain)

        if not elem_step_results.is_elem_step:
            print("\nStopped early because chain is not an elementary step.")
            print(
                f"Split chain based on: {elem_step_results.splitting_criterion}")
            self.optimized = chain
            return True, elem_step_results

        else:
            self.n_steps_still_chain = 0
            return False, elem_step_results

    def _check_early_stop(self, chain: Chain):
        """
        this function computes chain distances and checks gradient
        values in order to decide whether the expensive minimization of
        the chain should be done.
        """
        import neb_dynamics.chainhelpers as ch
        ind_ts_guess = np.argmax(chain.energies)
        ts_guess_grad = np.amax(np.abs(ch.get_g_perps(chain)[ind_ts_guess]))

        if ts_guess_grad <= self.parameters.early_stop_force_thre:

            new_params = self.parameters.copy()
            new_params.early_stop_force_thre = 0.0
            self.parameters = new_params

            # going to set climbing nodes when checking early stop
            if self.parameters.climb:
                self.set_climbing_nodes(chain=chain)
                self.parameters.climb = False  # no need to set climbing nodes again

            stop_early, elem_step_results = self._do_early_stop_check(
                chain)
            return stop_early, elem_step_results

        else:
            return False, ElemStepResults(is_elem_step=None,
                                          is_concave=None,
                                          splitting_criterion=None,
                                          minimization_results=[],
                                          number_grad_calls=0)

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

    def optimize_chain(self) -> ElemStepResults:
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
        import neb_dynamics.chainhelpers as ch

        nsteps = 1
        nsteps_negative_grad_corr = 0

        if self.parameters.preopt_with_xtb:
            chain_previous = self._do_xtb_preopt(self.initial_chain)
            self.chain_trajectory.append(chain_previous)

            stop_early, elem_step_results = self._do_early_stop_check(
                chain_previous)
            self.geom_grad_calls_made += elem_step_results.number_grad_calls
            if stop_early:
                return elem_step_results
        else:
            chain_previous = self.initial_chain.copy()
            self.chain_trajectory.append(chain_previous)
        chain_previous._zero_velocity()

        while nsteps < self.parameters.max_steps + 1:
            if nsteps > 1:
                stop_early, elem_step_results = self._check_early_stop(
                    chain_previous)
                self.geom_grad_calls_made += elem_step_results.number_grad_calls
                if stop_early:
                    return elem_step_results

            new_chain = self.update_chain(chain=chain_previous)
            max_rms_grad_val = np.amax(new_chain.rms_gperps)
            ind_ts_guess = np.argmax(new_chain.energies)
            ts_guess_grad = np.amax(
                np.abs(ch.get_g_perps(new_chain)[ind_ts_guess]))
            converged = chain_converged(
                chain_prev=chain_previous, chain_new=new_chain,
                parameters=self.parameters)
            if converged and self.parameters.v:
                print("\nConverged!")

            n_nodes_frozen = 0
            for node in new_chain:
                if node.converged:
                    n_nodes_frozen += 1

            grad_calls_made = len(new_chain) - n_nodes_frozen
            self.grad_calls_made += grad_calls_made

            grad_corr = ch._gradient_correlation(new_chain, chain_previous)
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
                    f"step {nsteps} // argmax(|TS gperp|) {np.amax(np.abs(ts_guess_grad))} // \
                        max rms grad {max_rms_grad_val} // armax(|TS_triplet_gsprings|) \
                            {new_chain.ts_triplet_gspring_infnorm} // nodes_frozen\
                                  {n_nodes_frozen} // {grad_corr}{' '*20}", end="\r"
                )
                sys.stdout.flush()

            self.chain_trajectory.append(new_chain)
            self.gradient_trajectory.append(new_chain.gradients)

            if converged:
                if self.parameters.v:
                    print("\nChain converged!")

                elem_step_results = check_if_elem_step(new_chain)
                self.geom_grad_calls_made += elem_step_results.number_grad_calls
                self.optimized = new_chain
                return elem_step_results

            chain_previous = new_chain
            nsteps += 1

        new_chain = self.update_chain(chain=chain_previous)
        if not chain_converged(chain_prev=chain_previous, chain_new=new_chain, parameters=self.parameters):
            raise NoneConvergedException(
                trajectory=self.chain_trajectory,
                msg=f"\nChain did not converge at step {nsteps}",
                obj=self,
            )

    def update_chain(self, chain: Chain) -> Chain:
        self.engine.compute_gradients(chain)

        import neb_dynamics.chainhelpers as ch
        grad_step = ch.compute_NEB_gradient(chain)
        new_chain = self.optimizer.optimize_step(
            chain=chain, chain_gradients=grad_step)

        self.engine.compute_gradients(new_chain)
        # print(f"{chain.gradients=}")
        # print(f"{new_chain.gradients=}")
        return new_chain

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

    def plot_chain_distances(self):
        distances = _calculate_chain_distances(self.chain_trajectory)

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

        if do_indiv:
            def plot_with_hline(data, label, y_hline, hline_label, hline_color, ylabel):
                f, ax = plt.subplots()
                plt.plot(data, label=label)
                plt.ylabel(ylabel)
                xmin, xmax = ax.get_xlim()
                ax.hlines(y=y_hline, xmin=xmin, xmax=xmax,
                          label=hline_label, linestyle='--', color=hline_color)
                f.legend()
                plt.show()

            # Plot RMS Grad$_{\perp}$
            plot_with_hline(avg_rms_gperp, label='RMS Grad$_{\perp}$',
                            y_hline=self.parameters.rms_grad_thre,
                            hline_label='rms_grad_thre', hline_color='blue',
                            ylabel="Gradient data")

            # Plot Max RMS Grad$_{\perp}$
            plot_with_hline(max_rms_gperp, label='Max RMS Grad$_{\perp}$',
                            y_hline=self.parameters.max_rms_grad_thre,
                            hline_label='max_rms_grad_thre', hline_color='orange',
                            ylabel="Gradient data")

            # Plot TS gperp
            plot_with_hline(ts_gperp, label='TS gperp',
                            y_hline=self.parameters.ts_grad_thre,
                            hline_label='ts_grad_thre', hline_color='green',
                            ylabel="Gradient data")

            # Plot barrier height
            plot_with_hline(barr_height, label='barr_height_delta',
                            y_hline=self.parameters.barrier_thre,
                            hline_label='barrier_thre', hline_color='purple',
                            ylabel="Barrier height data")

        else:
            # Define the data and parameters
            data_list = [
                (avg_rms_gperp, 'RMS Grad$_{\perp}$',
                 self.parameters.rms_grad_thre, 'rms_grad_thre', 'blue'),
                (max_rms_gperp, 'Max RMS Grad$_{\perp}$',
                 self.parameters.max_rms_grad_thre, 'max_rms_grad_thre', 'orange'),
                (ts_gperp, 'TS gperp', self.parameters.ts_grad_thre,
                 'ts_grad_thre', 'green')
            ]

            # Create subplots
            f, ax = plt.subplots()

            # Plot the gradient data
            for data, label, hline, hline_label, color in data_list:
                ax.plot(data, label=label)
                xmin, xmax = ax.get_xlim()
                ax.hlines(y=hline, xmin=xmin, xmax=xmax,
                          label=hline_label, linestyle='--', color=color)

            # Set y-axis label for gradient data
            ax.set_ylabel("Gradient data")

            # Create a second y-axis for barrier height data
            ax2 = ax.twinx()
            ax2.plot(barr_height, 'o--',
                     label='barr_height_delta', color='purple')
            ax2.set_ylabel("Barrier height data")
            ax2.hlines(y=self.parameters.barrier_thre, xmin=xmin, xmax=xmax,
                       label='barrier_thre', linestyle='--', color='purple')

            # Show legends and plot
            f.legend(loc='upper left')
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
