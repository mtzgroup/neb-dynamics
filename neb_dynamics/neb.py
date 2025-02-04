from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import copy

import matplotlib.pyplot as plt
import numpy as np
from neb_dynamics.convergence_helpers import chain_converged
from numpy.typing import NDArray
from openbabel import pybel
from qcop.exceptions import ExternalProgramError

import neb_dynamics.chainhelpers as ch
from neb_dynamics.chain import Chain
from neb_dynamics.elementarystep import ElemStepResults, check_if_elem_step
from neb_dynamics.engines import Engine
from neb_dynamics.engines.ase import ASEEngine
from neb_dynamics.errors import ElectronicStructureError, NoneConvergedException
from neb_dynamics.gsm_helper import minimal_wrapper_de_gsm, gsm_to_ase_atoms
from neb_dynamics.inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.nodes.node import StructureNode, Node
from neb_dynamics.optimizers.optimizer import Optimizer
from neb_dynamics.pathminimizers.pathminimizer import PathMinimizer
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics.qcio_structure_helpers import (
    structure_to_ase_atoms,
    ase_atoms_to_structure,
)

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
IS_ELEM_STEP = ElemStepResults(
    is_elem_step=True,
    is_concave=True,
    splitting_criterion=None,
    minimization_results=None,
    number_grad_calls=0,)
VELOCITY_SCALING = 0.3
ACTIVATION_TOL = 100
MAX_NRETRIES = 3


@dataclass
class NEB(PathMinimizer):
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
    engine: Engine

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)

    def __post_init__(self):
        self.n_steps_still_chain = 0
        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0
        if self.parameters.frozen_atom_indices is not None:
            if isinstance(self.parameters.frozen_atom_indices, str):
                self.parameters.frozen_atom_indices = [
                    int(x) for x in self.parameters.frozen_atom_indices.split(",") if x]

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

        elem_step_results = check_if_elem_step(
            inp_chain=chain, engine=self.engine)

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
        springgrad_infnorm = np.amax(abs(chain.springgradients))

        # if ts_guess_grad < self.parameters.early_stop_force_thre:
        if springgrad_infnorm < self.parameters.early_stop_force_thre:

            new_params = copy.deepcopy(self.parameters)
            new_params.early_stop_force_thre = 0.0
            self.parameters = new_params

            # going to set climbing nodes when checking early stop
            if self.parameters.climb:
                self.set_climbing_nodes(chain=chain)
                self.parameters.climb = False  # no need to set climbing nodes again

            stop_early, elem_step_results = self._do_early_stop_check(chain)

            self.parameters.early_stop_force_thre = (
                0.0  # setting it to 0 so we don't check it over and over
            )
            return stop_early, elem_step_results

        else:
            return False, ElemStepResults(
                is_elem_step=None,
                is_concave=None,
                splitting_criterion=None,
                minimization_results=[],
                number_grad_calls=0,
            )

    # @Jan: This should be a more general function so that the
    # lower level of theory can be whatever the user wants.
    def _do_xtb_preopt(self, chain) -> Chain:  #
        """
        This function will loosely minimize an input chain using the GFN2-XTB method,
        then return a new chain which can be used as an initial guess for a higher
        level of theory calculation
        """

        xtb_params = chain.parameters.copy()
        xtb_params.node_class = Node
        chain_traj = chain.to_trajectory()
        xtb_chain = Chain.from_traj(chain_traj, parameters=xtb_params)
        xtb_nbi = NEBInputs(
            tol=self.parameters.tol * 10, v=True, preopt_with_xtb=False, max_steps=1000
        )

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

        # if self.parameters.preopt_with_xtb:
        #     chain_previous = self._do_xtb_preopt(self.initial_chain)
        #     self.chain_trajectory.append(chain_previous)

        #     stop_early, elem_step_results = self._do_early_stop_check(
        #         chain_previous)
        #     self.geom_grad_calls_made += elem_step_results.number_grad_calls
        #     if stop_early:
        #         return elem_step_results
        # else:
        chain_previous = self.initial_chain.copy()
        self.chain_trajectory.append(chain_previous)
        chain_previous._zero_velocity()
        self.optimizer.g_old = None

        while nsteps < self.parameters.max_steps + 1:
            if nsteps > 1:
                stop_early, elem_step_results = self._check_early_stop(
                    chain_previous)
                self.geom_grad_calls_made += elem_step_results.number_grad_calls
                if stop_early:
                    return elem_step_results
            try:
                new_chain = self.update_chain(chain=chain_previous)
            except ExternalProgramError:
                elem_step_results = check_if_elem_step(
                    inp_chain=chain_previous, engine=self.engine
                )
                raise ElectronicStructureError(msg="QCOP failed.")

            max_rms_grad_val = np.amax(new_chain.rms_gperps)
            ind_ts_guess = np.argmax(new_chain.energies)
            ts_guess_grad = np.amax(
                np.abs(ch.get_g_perps(new_chain)[ind_ts_guess]))
            converged = chain_converged(
                chain_prev=chain_previous,
                chain_new=new_chain,
                parameters=self.parameters,
            )
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
                                  {n_nodes_frozen} // {grad_corr}{' '*20}",
                    end="\r",
                )
                sys.stdout.flush()

            self.chain_trajectory.append(new_chain)
            self.gradient_trajectory.append(new_chain.gradients)

            if converged:
                if self.parameters.v:
                    print("\nChain converged!")
                if self.parameters.do_elem_step_checks:
                    elem_step_results = check_if_elem_step(
                        inp_chain=new_chain, engine=self.engine
                    )
                    self.geom_grad_calls_made += elem_step_results.number_grad_calls
                else:
                    elem_step_results = IS_ELEM_STEP
                self.optimized = new_chain
                return elem_step_results

            chain_previous = new_chain
            nsteps += 1

        new_chain = self.update_chain(chain=chain_previous)
        if not chain_converged(
            chain_prev=chain_previous, chain_new=new_chain, parameters=self.parameters
        ):
            raise NoneConvergedException(
                trajectory=self.chain_trajectory,
                msg=f"\nChain did not converge at step {nsteps}",
                obj=self,
            )

    def _update_cache(
        self, chain: Chain, gradients: NDArray, energies: NDArray
    ) -> None:
        """
        will update the `_cached_energy` and `_cached_gradient` attributes in the chain
        nodes based on the input `gradients` and `energies`
        """
        from neb_dynamics.fakeoutputs import FakeQCIOOutput, FakeQCIOResults

        for node, grad, ene in zip(chain, gradients, energies):
            if not hasattr(node, "_cached_result"):
                res = FakeQCIOResults(energy=ene, gradient=grad)
                outp = FakeQCIOOutput(results=res)
                node._cached_result = outp
                node._cached_energy = ene
                node._cached_gradient = grad

    def update_chain(self, chain: Chain) -> Chain:
        import neb_dynamics.chainhelpers as ch

        grads = self.engine.compute_gradients(chain)
        enes = self.engine.compute_energies(chain)
        self._update_cache(chain, grads, enes)

        grad_step = ch.compute_NEB_gradient(
            chain, geodesic_tangent=self.parameters.use_geodesic_tangent)
        if self.parameters.frozen_atom_indices:
            for index in self.parameters.frozen_atom_indices:
                grad_step[index] = np.array([0.0, 0.0, 0.0])

        alpha = 1.0
        ntries = 0
        grads_success = False
        while not grads_success and ntries < MAX_NRETRIES:
            try:
                new_chain = self.optimizer.optimize_step(
                    chain=chain, chain_gradients=grad_step*alpha)

                # need to copy the gradients from the converged nodes
                for new_node, old_node in zip(new_chain.nodes, chain.nodes):
                    if old_node.converged:
                        new_node._cached_result = old_node._cached_result
                        new_node._cached_energy = old_node._cached_energy
                        new_node._cached_gradient = old_node._cached_gradient

                self.engine.compute_gradients(new_chain)
                grads_success = True

            except ExternalProgramError:
                print("UH OH SPAGGHETIOooOOOO")
                self.optimizer.g_old = None
                alpha *= .8
                ntries += 1
        if not grads_success and ntries >= MAX_NRETRIES:
            print("!!!Electronic structure error! Smoothing current chain with GI")
            new_chain = ch.run_geodesic(
                chain, chain_inputs=chain.parameters, nimages=len(chain))
            self.engine.compute_gradients(new_chain)

        return new_chain

    def plot_chain_distances(self):
        import neb_dynamics.chainhelpers as ch

        distances = ch._calculate_chain_distances(self.chain_trajectory)

        fs = 18
        s = 8

        f, ax = plt.subplots(figsize=(1.16 * s, s))

        plt.plot(distances, "o-")
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
            prev_chain = self.chain_trajectory[i - 1]
            projs.append(prev_chain._gradient_delta_mags(chain))

        plt.plot(projs)
        plt.ylabel("NEB |∆gradient|", fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.xticks(fontsize=fs)
        # plt.ylim(0,1.1)
        plt.xlabel("Optimization step", fontsize=fs)
        plt.show()

    def plot_projector_history(self, var="gradients"):
        s = 8
        fs = 18
        f, ax = plt.subplots(figsize=(1.16 * s, s))
        projs = []

        for i, chain in enumerate(self.chain_trajectory):
            if i == 0:
                continue
            prev_chain = self.chain_trajectory[i - 1]
            if var == "gradients":
                projs.append(prev_chain._gradient_correlation(chain))
            elif var == "tangents":
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

    def plot_convergence_metrics(self, do_indiv=False):
        ct = self.chain_trajectory

        avg_rms_gperp = []
        max_rms_gperp = []
        avg_rms_g = []
        barr_height = []
        ts_gperp = []
        grad_infnorm = []

        for ind in range(1, len(ct)):
            avg_rms_g.append(
                sum(ct[ind].rms_gradients[1:-1]) / (len(ct[ind]) - 2))
            avg_rms_gperp.append(
                sum(ct[ind].rms_gperps[1:-1]) / (len(ct[ind]) - 2))
            max_rms_gperp.append(max(ct[ind].rms_gperps))
            barr_height.append(
                abs(ct[ind].get_eA_chain() - ct[ind - 1].get_eA_chain()))
            ts_node_ind = ct[ind].energies.argmax()
            ts_node_gperp = np.max(ch.get_g_perps(ct[ind])[ts_node_ind])
            ts_gperp.append(ts_node_gperp)
            grad_infnorm.append(np.amax(abs(ch.compute_NEB_gradient(ct[ind]))))

        if do_indiv:

            def plot_with_hline(data, label, y_hline, hline_label, hline_color, ylabel):
                f, ax = plt.subplots()
                plt.plot(data, label=label)
                plt.ylabel(ylabel)
                xmin, xmax = ax.get_xlim()
                ax.hlines(
                    y=y_hline,
                    xmin=xmin,
                    xmax=xmax,
                    label=hline_label,
                    linestyle="--",
                    color=hline_color,
                )
                f.legend()
                plt.show()

            # Plot RMS Grad$_{\perp}$
            plot_with_hline(
                avg_rms_gperp,
                label="RMS Grad$_{\perp}$",
                y_hline=self.parameters.rms_grad_thre,
                hline_label="rms_grad_thre",
                hline_color="blue",
                ylabel="Gradient data",
            )

            # Plot Max RMS Grad$_{\perp}$
            plot_with_hline(
                max_rms_gperp,
                label="Max RMS Grad$_{\perp}$",
                y_hline=self.parameters.max_rms_grad_thre,
                hline_label="max_rms_grad_thre",
                hline_color="orange",
                ylabel="Gradient data",
            )

            # Plot TS gperp
            plot_with_hline(
                ts_gperp,
                label="TS gperp",
                y_hline=self.parameters.ts_grad_thre,
                hline_label="ts_grad_thre",
                hline_color="green",
                ylabel="Gradient data",
            )

            # Plot barrier height
            plot_with_hline(
                barr_height,
                label="barr_height_delta",
                y_hline=self.parameters.barrier_thre,
                hline_label="barrier_thre",
                hline_color="purple",
                ylabel="Barrier height data",
            )

        else:
            # Define the data and parameters
            data_list = [
                (
                    avg_rms_gperp,
                    "RMS Grad$_{\perp}$",
                    self.parameters.rms_grad_thre,
                    "rms_grad_thre",
                    "blue",
                ),
                (
                    max_rms_gperp,
                    "Max RMS Grad$_{\perp}$",
                    self.parameters.max_rms_grad_thre,
                    "max_rms_grad_thre",
                    "orange",
                ),
                (
                    ts_gperp,
                    "TS gperp",
                    self.parameters.ts_grad_thre,
                    "ts_grad_thre",
                    "green",
                ),
                (grad_infnorm,
                 "Grad infnorm",
                 self.parameters.ts_grad_thre,
                 "grad_infnorm",
                 "gray")
            ]

            # Create subplots
            f, ax = plt.subplots()

            # Plot the gradient data
            for data, label, hline, hline_label, color in data_list:
                ax.plot(data, label=label)
                xmin, xmax = ax.get_xlim()
                ax.hlines(
                    y=hline,
                    xmin=xmin,
                    xmax=xmax,
                    label=hline_label,
                    linestyle="--",
                    color=color,
                )

            # Set y-axis label for gradient data
            ax.set_ylabel("Gradient data")

            # Create a second y-axis for barrier height data
            ax2 = ax.twinx()
            ax2.plot(barr_height, "o--",
                     label="barr_height_delta", color="purple")
            ax2.set_ylabel("Barrier height data")
            ax2.hlines(
                y=self.parameters.barrier_thre,
                xmin=xmin,
                xmax=xmax,
                label="barrier_thre",
                linestyle="--",
                color="purple",
            )

            # Show legends and plot
            f.legend(loc="upper left")
            plt.show()

    def read_from_disk(
        fp: Path,
        history_folder: Path = None,
        chain_parameters=ChainInputs(),
        neb_parameters=NEBInputs(),
        gi_parameters=GIInputs(),
        optimizer=VelocityProjectedOptimizer(),
        engine: Engine = None,
        charge: int = 0,
        multiplicity: int = 1,
    ):
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
                    history_folder / f"traj_{i}.xyz", parameters=chain_parameters,
                    charge=charge,
                    spinmult=multiplicity
                )
                for i, _ in enumerate(history_files)
            ]

        n = NEB(
            initial_chain=history[0],
            parameters=neb_parameters,
            optimized=history[-1],
            chain_trajectory=history,
            optimizer=optimizer,
            engine=engine,
        )
        return n


@dataclass
class PYGSM(PathMinimizer):
    initial_chain: Chain
    engine: ASEEngine  # incompatible with other Engines right now
    chain_trajectory: list[Chain] = field(default_factory=list)
    pygsm_kwds: dict = field(default_factory=dict)

    def optimize_chain(self) -> ElemStepResults:
        start = self.initial_chain[0]
        end = self.initial_chain[-1]
        start_ase = structure_to_ase_atoms(start.structure)
        end_ase = structure_to_ase_atoms(end.structure)

        gsm = minimal_wrapper_de_gsm(
            atoms_reactant=start_ase,
            atoms_product=end_ase,
            num_nodes=len(self.initial_chain),
            calculator=self.engine.calculator,
            **self.pygsm_kwds,
        )

        ase_frames, ase_ts = gsm_to_ase_atoms(gsm=gsm)
        charge = self.initial_chain[0].structure.charge
        multiplicity = self.initial_chain[0].structure.multiplicity
        frames = [
            ase_atoms_to_structure(
                atoms=frame, charge=charge, multiplicity=multiplicity
            )
            for frame in ase_frames
        ]
        nodes = [StructureNode(structure=struct) for struct in frames]
        out_chain = Chain(
            nodes=nodes, parameters=self.initial_chain.parameters.copy())
        self.engine.compute_energies(out_chain)
        self.chain_trajectory.append(out_chain)
        self.optimized = out_chain
        return check_if_elem_step(out_chain, engine=self.engine)
