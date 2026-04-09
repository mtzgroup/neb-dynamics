import traceback
import logging
import copy
import os
import concurrent.futures
from dataclasses import dataclass

import sys
import numpy as np
from neb_dynamics.helper_functions import pairwise
from typing import Tuple, List

from neb_dynamics.nodes.node import Node, StructureNode
from neb_dynamics.nodes.nodehelpers import _is_connectivity_identical
from neb_dynamics.elementarystep import check_if_elem_step

from neb_dynamics.chain import Chain
import neb_dynamics.chainhelpers as ch
from neb_dynamics.dynamics.chainbiaser import ChainBiaser
from neb_dynamics.elementarystep import ElemStepResults
from neb_dynamics.neb import NEB, NoneConvergedException
from neb_dynamics.nodes.nodehelpers import is_identical

from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.errors import ElectronicStructureError
from neb_dynamics.optimizers.cg import ConjugateGradient
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics.optimizers.lbfgs import LBFGS
from neb_dynamics.optimizers.adam import AdamOptimizer
from neb_dynamics.optimizers.amg import AdaptiveMomentumGradient
from neb_dynamics.optimizers.fire import FIREOptimizer
from qcio import Structure

from neb_dynamics.pathminimizers.fneb import FreezingNEB
from neb_dynamics.pathminimizers.nebdlf import DLFindNEB
from neb_dynamics.inputs import RunInputs
from neb_dynamics.scripts.progress import (
    print_neb_step,
    preserve_chain_snapshot,
    progress_monitor,
    start_status,
    update_status,
    stop_status,
)


def _get_verbose(inputs: RunInputs) -> bool:
    """Get verbosity from RunInputs."""
    return getattr(inputs.path_min_inputs, 'v', False)


PATH_METHODS = ["NEB", "FNEB", "MLPGI", "NEB-DLF"]


def _normalize_path_method(path_min_method: str) -> str:
    method = str(path_min_method or "").strip().upper().replace("_", "-")
    aliases = {
        "NEBDLF": "NEB-DLF",
        "DLFNEB": "NEB-DLF",
        "DLFIND": "NEB-DLF",
        "DL-FIND": "NEB-DLF",
    }
    return aliases.get(method, method)


@dataclass
class MSMEP:
    """Class for running autosplitting MEP minimizations."""

    inputs: RunInputs
    path_minimizer = None

    def __post_init__(self):
        assert (
            self.inputs.path_min_method or self.path_minimizer is not None
        ), "Need to input a path_min_method or path minimizer"
        if self.path_minimizer is None:
            normalized_method = _normalize_path_method(self.inputs.path_min_method)
            assert (
                normalized_method in PATH_METHODS
            ), f"Invalid path method: {self.inputs.path_min_method}. Allowed are: {PATH_METHODS}"

    def _build_neb_optimizer(self):
        kwds = dict(self.inputs.optimizer_kwds or {})
        optimizer_name = kwds.pop("name", "cg").lower()
        optimizer_map = {
            "cg": ConjugateGradient,
            "conjugate_gradient": ConjugateGradient,
            "vpo": VelocityProjectedOptimizer,
            "velocity_projected": VelocityProjectedOptimizer,
            "lbfgs": LBFGS,
            "adam": AdamOptimizer,
            "amg": AdaptiveMomentumGradient,
            "adaptive_momentum": AdaptiveMomentumGradient,
            "fire": FIREOptimizer,
        }
        if optimizer_name not in optimizer_map:
            available = ", ".join(sorted(set(optimizer_map.keys())))
            raise ValueError(
                f"Unsupported optimizer '{optimizer_name}'. Supported values: {available}"
            )
        return optimizer_map[optimizer_name](**kwds)

    def _should_disable_graphs(self) -> bool:
        return self.inputs.engine.__class__.__name__ == "QMMMEngine"

    def _disable_molecular_graphs(self, chain: Chain) -> None:
        if not self._should_disable_graphs():
            return
        for node in chain:
            if hasattr(node, "has_molecular_graph"):
                node.has_molecular_graph = False
            if hasattr(node, "graph"):
                node.graph = None

    def run_recursive_minimize(self, input_chain: Chain, tree_node_index=0) -> TreeNode:
        """Will take a chain as an input and run NEB minimizations until it exits out.
        NEB can exit due to the chain being converged, the chain needing to be split,
        or the maximum number of alloted steps being met.

        Args:
            input_chain (Chain): _description_
            tree_node_index (int, optional): index of node minimization. Root node is 0. Defaults to 0.

        Returns:
            TreeNode: Tree containing the history of NEB optimizations. First node is the initial
            chain given and its corresponding neb minimization. Children are chains into which
            the root chain was split.
        """
        import neb_dynamics.chainhelpers as ch
        if isinstance(input_chain, list):
            input_chain = Chain.model_validate(
                {"nodes": input_chain, "parameters": self.inputs.chain_inputs})
        self._disable_molecular_graphs(input_chain)

        if getattr(self.inputs.path_min_inputs, "skip_identical_graphs", True) and input_chain[0].has_molecular_graph:
            if not _get_verbose(self.inputs):
                update_status("Checking endpoint connectivity")
            if _is_connectivity_identical(
                input_chain[0],
                input_chain[-1],
                verbose=_get_verbose(self.inputs),
            ):
                msg = "Endpoints are identical. Returning nothing"
                if _get_verbose(self.inputs):
                    print(msg)
                else:
                    update_status(msg)
                return TreeNode(data=None, children=[], index=tree_node_index)

        ch._reset_node_convergence(input_chain)
        self.inputs.engine.compute_gradients(input_chain)

        if is_identical(
            self=input_chain[0],
            other=input_chain[-1],
            fragment_rmsd_cutoff=self.inputs.chain_inputs.node_rms_thre,
            kcal_mol_cutoff=self.inputs.chain_inputs.node_ene_thre,
            verbose=False,
        ):
            msg = "Endpoints are identical. Returning nothing"
            if _get_verbose(self.inputs):
                print(msg)
            else:
                update_status(msg)
            return TreeNode(data=None, children=[], index=tree_node_index)

        try:
            root_neb_obj, elem_step_results = self.run_minimize_chain(
                input_chain=input_chain
            )
            history = TreeNode(data=root_neb_obj,
                               children=[], index=tree_node_index)

            if elem_step_results.is_elem_step:
                return history

            else:
                msg = f"Splitting chains based on: {elem_step_results.splitting_criterion}"
                if _get_verbose(self.inputs):
                    print(msg)
                else:
                    preserve_chain_snapshot(note=msg)

                # the last chain in the minimization
                chain = root_neb_obj.chain_trajectory[-1]
                sequence_of_chains = self.make_sequence_of_chains(
                    chain=chain,
                    split_method=elem_step_results.splitting_criterion,
                    minimization_results=elem_step_results.minimization_results,
                )

                new_tree_node_index = tree_node_index + 1
                for i, chain_frag in enumerate(sequence_of_chains, start=1):
                    msg = f"On chain {i} of {len(sequence_of_chains)}..."
                    if _get_verbose(self.inputs):
                        print(msg)
                    else:
                        update_status(msg)
                    out_history = self.run_recursive_minimize(
                        chain_frag, tree_node_index=new_tree_node_index
                    )

                    history.children.append(out_history)

                    # increment the node indices
                    new_tree_node_index = out_history.max_index + 1
                return history

        except ElectronicStructureError as e:
            e.obj.save("/tmp/failed_output.qcio")
            return TreeNode(data=None, children=[], index=tree_node_index)

    def _run_recursive_step(self, input_chain: Chain, tree_node_index: int) -> tuple[TreeNode, list[Chain]]:
        """Run a single recursive minimization step and return child fragments to continue."""
        import neb_dynamics.chainhelpers as ch
        if isinstance(input_chain, list):
            input_chain = Chain.model_validate(
                {"nodes": input_chain, "parameters": self.inputs.chain_inputs})
        self._disable_molecular_graphs(input_chain)

        if getattr(self.inputs.path_min_inputs, "skip_identical_graphs", True) and input_chain[0].has_molecular_graph:
            if _is_connectivity_identical(
                input_chain[0],
                input_chain[-1],
                verbose=_get_verbose(self.inputs),
            ):
                return TreeNode(data=None, children=[], index=tree_node_index), []

        ch._reset_node_convergence(input_chain)
        self.inputs.engine.compute_gradients(input_chain)

        if is_identical(
            self=input_chain[0],
            other=input_chain[-1],
            fragment_rmsd_cutoff=self.inputs.chain_inputs.node_rms_thre,
            kcal_mol_cutoff=self.inputs.chain_inputs.node_ene_thre,
            verbose=False,
        ):
            return TreeNode(data=None, children=[], index=tree_node_index), []

        root_neb_obj, elem_step_results = self.run_minimize_chain(
            input_chain=input_chain
        )
        history_node = TreeNode(data=root_neb_obj, children=[], index=tree_node_index)

        if elem_step_results.is_elem_step:
            return history_node, []

        chain_trajectory = getattr(root_neb_obj, "chain_trajectory", None) or []
        if not chain_trajectory:
            return history_node, []
        split_chain = chain_trajectory[-1]
        sequence_of_chains = self.make_sequence_of_chains(
            chain=split_chain,
            split_method=elem_step_results.splitting_criterion,
            minimization_results=elem_step_results.minimization_results,
        )
        return history_node, sequence_of_chains

    def run_parallel_recursive_minimize(
        self,
        input_chain: Chain,
        tree_node_index: int = 0,
        max_workers: int | None = None,
    ) -> TreeNode:
        """Recursively autosplit NEBs, evaluating split branches in parallel."""
        cpu_cap = max(1, int(os.cpu_count() or 1))
        if max_workers is None:
            bounded_workers = min(4, cpu_cap)
        else:
            bounded_workers = max(1, min(int(max_workers), cpu_cap))

        with progress_monitor(f"branch-{int(tree_node_index)}"):
            root_history, root_children = self._run_recursive_step(
                input_chain=input_chain, tree_node_index=tree_node_index
            )
        if not root_children:
            return root_history

        next_tree_index = tree_node_index + 1
        pending: dict[concurrent.futures.Future, tuple[TreeNode, int, int]] = {}

        def _submit_children(
            executor: concurrent.futures.Executor,
            parent_node: TreeNode,
            child_fragments: list[Chain],
        ) -> None:
            nonlocal next_tree_index
            parent_node.children = [None] * len(child_fragments)
            for child_position, child_chain in enumerate(child_fragments):
                child_index = next_tree_index
                next_tree_index += 1
                future = executor.submit(
                    _parallel_recursive_step_worker,
                    self.inputs,
                    child_chain,
                    child_index,
                )
                pending[future] = (parent_node, child_position, child_index)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bounded_workers
        ) as executor:
            _submit_children(executor, root_history, root_children)
            while pending:
                done, _ = concurrent.futures.wait(
                    tuple(pending.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    parent_node, child_position, child_index = pending.pop(future)
                    try:
                        child_history, child_children = future.result()
                    except Exception:
                        child_history = TreeNode(
                            data=None, children=[], index=child_index
                        )
                        child_children = []
                    parent_node.children[child_position] = child_history
                    if child_children:
                        _submit_children(executor, child_history, child_children)

        return root_history

    def _create_interpolation(self, chain: Chain):
        import neb_dynamics.chainhelpers as ch
        logger = logging.getLogger(
            'neb_dynamics.geodesic_interpolation2.interpolation')
        logger.propagate = False
        # if chain.parameters.frozen_atom_indices:
        #     chain_original = chain.copy()
        #     all_indices = list(range(len(chain[0].coords)))

        #     inds_frozen = np.array(
        #         chain.parameters.frozen_atom_indices.split(), dtype=int
        #     )
        #     subsys_inds = np.setdiff1d(all_indices, inds_frozen)
        #     subsys_coords = [node.coords[subsys_inds] for node in chain]

        #     subsys_symbs = [chain[0].structure.symbols[i] for i in subsys_inds]
        #     subsys_structs = [Structure(geometry=c, symbols=subsys_symbs,
        #                                 charge=chain[0].structure.charge,
        #                                 multiplicity=chain[0].structure.multiplicity) for c in subsys_coords]

        #     subsys_nodes = [StructureNode(structure=s) for s in subsys_structs]

        #     print(f"{all_indices=} {subsys_inds=} {inds_frozen=}")
        #     chain = Chain.model_validate({
        #         "nodes": subsys_nodes, "parameters": copy.deepcopy(self.inputs.chain_inputs)})

        if self.inputs.chain_inputs.use_geodesic_interpolation:
            if chain.parameters.frozen_atom_indices:
                inds_frozen = chain.parameters.frozen_atom_indices
                if _get_verbose(self.inputs):
                    print("will be freezing inds:", inds_frozen,
                          ' during geodesic interpolation')
                    print(type(inds_frozen))
            else:
                inds_frozen = np.array([], dtype=int)

            smoother = ch.sample_shortest_geodesic(
                chain=chain,
                chain_inputs=copy.deepcopy(self.inputs.chain_inputs),
                nimages=self.inputs.gi_inputs.nimages,
                friction=self.inputs.gi_inputs.friction,
                nudge=self.inputs.gi_inputs.nudge,
                align=self.inputs.gi_inputs.align,
                ignore_atoms=inds_frozen,
                **self.inputs.gi_inputs.extra_kwds,

            )
            interpolation = ch.gi_path_to_nodes(xyz_coords=smoother.path,
                                                symbols=chain.symbols,
                                                parameters=copy.deepcopy(
                                                    self.inputs.chain_inputs),
                                                charge=chain[0].structure.charge,
                                                spinmult=chain[0].structure.multiplicity)
            interpolation = Chain.model_validate({
                "nodes": interpolation,
                "parameters": copy.deepcopy(self.inputs.chain_inputs)})

            interpolation._zero_velocity()

        else:  # do a linear interpolation using numpy
            start_point = chain[0].coords
            end_point = chain[-1].coords
            coords = np.linspace(
                start=start_point, stop=end_point, num=self.inputs.gi_inputs.nimages
            )
            nodes = [
                node.update_coords(c)
                for node, c in zip([chain.nodes[0]] * len(coords), coords)
            ]
            interpolation = Chain.model_validate({
                "nodes": nodes, "parameters": copy.deepcopy(self.inputs.chain_inputs)})

        # if chain.parameters.frozen_atom_indices:
        #     # need to reinsert the frozen atoms into the interpolation
        #     new_nodes = []
        #     for node in interpolation:
        #         new_geom = np.zeros_like(chain_original[0].coords)
        #         new_geom[subsys_inds] = node.coords
        #         new_geom[inds_frozen] = chain_original[0].coords[inds_frozen]
        #         new_node = chain_original[0].update_coords(new_geom)
        #         new_nodes.append(new_node)

        #     interpolation = Chain.model_validate({
        #         "nodes": new_nodes, "parameters": copy.deepcopy(self.inputs.chain_inputs)})
        #     interpolation._zero_velocity()
        return interpolation

    def _construct_path_minimizer(self, initial_chain: Chain):
        path_method = _normalize_path_method(self.inputs.path_min_method)
        if path_method == "NEB":

            msg = "Using in-house NEB optimizer"
            if _get_verbose(self.inputs):
                print(msg)
                sys.stdout.flush()
            else:
                update_status(msg)
            optimizer = self._build_neb_optimizer()

            n = NEB(
                initial_chain=initial_chain,
                parameters=self.inputs.path_min_inputs,
                optimizer=optimizer,
                engine=self.inputs.engine,
            )
        # elif self.inputs.path_min_method.upper() == "PYGSM":
        #     print("Using PYGSM optimizer")
        #     n = PYGSM(
        #         initial_chain=initial_chain,
        #         engine=self.inputs.engine,
        #         pygsm_kwds=self.inputs.path_min_inputs,
        #     )

        elif path_method == "FNEB":
            msg = "Using Freezing NEB optimizer"
            if _get_verbose(self.inputs):
                print(msg)
            else:
                update_status(msg)
            n = FreezingNEB(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
                parameters=self.inputs.path_min_inputs,
                optimizer=self.inputs.optimizer,
                gi_inputs=self.inputs.gi_inputs
            )

        elif path_method == "MLPGI":
            from neb_dynamics.pathminimizers.mlpgi import MLPGI
            msg = "Using MLP Geodesic Optimizer"
            if _get_verbose(self.inputs):
                print(msg)
            else:
                update_status(msg)
            n = MLPGI(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
                parameters=self.inputs.path_min_inputs,
            )
        elif path_method == "NEB-DLF":
            msg = "Using DL-Find NEB optimizer via TeraChem/QCOP"
            if _get_verbose(self.inputs):
                print(msg)
            else:
                update_status(msg)
            n = DLFindNEB(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
                parameters=self.inputs.path_min_inputs,
            )
        else:
            raise NotImplementedError(
                "Invalid path minimization method. Select from NEB, FNEB, MLPGI, or NEB-DLF.")

        return n

    def run_minimize_chain(self, input_chain: Chain) -> Tuple[NEB, ElemStepResults]:
        if isinstance(input_chain, list):
            input_chain = Chain.model_validate(
                {"nodes": input_chain, "parameters": self.inputs.chain_inputs})
        self._disable_molecular_graphs(input_chain)

        # make sure the chain parameters are reset
        # if they come from a converged chain
        if len(input_chain) != self.inputs.gi_inputs.nimages:
            interpolation = self._create_interpolation(
                input_chain,
            )
            assert (
                len(interpolation) == self.inputs.gi_inputs.nimages
            ), f"Geodesic interpolation wrong length.\
                 Requested: {self.inputs.gi_inputs.nimages}. Given: {len(interpolation)}"

        else:
            interpolation = input_chain
        self._disable_molecular_graphs(interpolation)

        # Use spinner when v=0, print when v=1
        verbose = _get_verbose(self.inputs)
        if verbose:
            print("Running path minimization...")
        else:
            start_status("Minimizing path...")

        try:
            n = self._construct_path_minimizer(initial_chain=interpolation)
            elem_step_results = n.optimize_chain()
            out_chain = n.optimized

        except NoneConvergedException:
            print(traceback.format_exc())

            print(
                "\nWarning! A chain did not converge.\
                        Returning an unoptimized chain..."
            )
            out_chain = n.chain_trajectory[-1]
            if self.inputs.path_min_inputs.do_elem_step_checks:
                elem_step_results = check_if_elem_step(
                    out_chain, engine=self.inputs.engine, verbose=_get_verbose(self.inputs))
            else:
                elem_step_results = ElemStepResults(
                    is_elem_step=True,
                    is_concave=None,
                    splitting_criterion=None,
                    minimization_results=None,
                    number_grad_calls=0,
                )

        except ElectronicStructureError as e:
            # print(traceback.format_exc())

            print(
                "\nWarning! A chain has electronic structure errors. \
                    Returning an unoptimized chain..."
            )
            if getattr(e, "msg", None):
                print(f"ElectronicStructureError message: {e.msg}")
            if e.__cause__ is not None:
                print(f"ElectronicStructureError cause: {type(e.__cause__).__name__}: {e.__cause__}")
            if e.obj is not None:
                if isinstance(e.obj, (list, tuple)):
                    print(
                        f"ElectronicStructureError includes {len(e.obj)} result objects."
                    )
                else:
                    print(
                        "ElectronicStructureError includes a result object "
                        f"of type {type(e.obj).__name__}."
                    )
            out_chain = n.chain_trajectory[-1]
            elem_step_results = ElemStepResults(
                is_elem_step=True,
                is_concave=None,
                splitting_criterion=None,
                minimization_results=None,
                number_grad_calls=0,
            )

        finally:
            if not verbose:
                stop_status()

        return n, elem_step_results

    def _update_chain_dynamics(
        self,
        chain: Chain,
        engine,
        dt,
        temperature: float,
        mass=1.0,
        biaser: ChainBiaser = None,
        freeze_ends: bool = True
    ) -> Chain:
        import neb_dynamics.chainhelpers as ch

        # R = 0.008314 / 627.5  # Hartree/mol.K
        R = 1  # i'm not even sure what R means in this world.
        _ = engine.compute_gradients(chain)
        grads = ch.compute_NEB_gradient(chain)

        new_vel = chain.velocity.copy()
        current_temperature = (
            np.dot(new_vel.flatten(), new_vel.flatten()) * mass / (3 * R)
        )
        thermostating_scaling = np.sqrt(temperature / current_temperature)
        new_vel *= thermostating_scaling

        new_vel += 0.5 * -1 * grads * dt / mass
        print("/n", thermostating_scaling, current_temperature)

        position = chain.coordinates
        ref_start = position[0]
        ref_end = position[-1]
        new_position = position + new_vel * dt
        # print(new_vel)
        if biaser:
            grad_bias = biaser.grad_chain_bias(chain)
            # energy_weights = chain.energies_kcalmol[1:-1]
            if freeze_ends:
                new_position[1:-1] -= grad_bias
        if freeze_ends:
            new_position[0] = ref_start
            new_position[-1] = ref_end
        new_chain = chain.copy()
        new_nodes = []
        for coords, node in zip(new_position, new_chain):
            new_nodes.append(node.update_coords(coords))
        new_chain.nodes = new_nodes

        grads = engine.compute_gradients(new_chain)

        if freeze_ends:
            grads[0] = np.zeros_like(grads[0])
            grads[-1] = np.zeros_like(grads[0])
        new_vel += 0.5 * -1 * grads * dt / mass
        new_chain.velocity = new_vel

        return new_chain

    def run_dynamics(
        self,
        chain: Chain,
        max_steps: int,
        dt=0.1,
        biaser: ChainBiaser = None,
        temperature: float = 300.0,  # K
        freeze_ends: bool = True
    ):
        """
        Runs dynamics on the chain.
        """
        if np.all(chain.velocity == 0):
            rand_vel = np.random.random(
                size=len(chain.velocity.flatten())).reshape(chain.velocity.shape)
            rand_vel /= np.linalg.norm(rand_vel)
            rand_vel *= temperature
            chain.velocity = rand_vel
        chain_trajectory = [chain]
        nsteps = 1
        chain_previous = chain
        img_weight = sum(ch._get_mass_weights(chain, False))
        chain_weight = img_weight * len(chain)
        while nsteps < max_steps + 1:
            try:
                new_chain = self._update_chain_dynamics(
                    chain=chain_previous,
                    engine=self.inputs.engine,
                    dt=dt,
                    biaser=biaser,
                    mass=chain_weight,
                    temperature=temperature,
                    freeze_ends=freeze_ends
                )
            except Exception:
                print(traceback.format_exc())
                print("Electronic structure error")
                return chain_trajectory

            max_rms_grad_val = np.amax(new_chain.rms_gperps)
            ind_ts_guess = np.argmax(new_chain.energies)
            ts_guess_grad = np.amax(
                np.abs(ch.get_g_perps(new_chain)[ind_ts_guess]))

            print_neb_step(
                step=nsteps,
                ts_grad=np.amax(np.abs(ts_guess_grad)),
                max_rms_grad=max_rms_grad_val,
                ts_triplet_gspring=new_chain.ts_triplet_gspring_infnorm,
                nodes_frozen=0,
                timestep=np.linalg.norm(new_chain.velocity),
                grad_corr="",
                force_update=True
            )
            chain_trajectory.append(new_chain)
            nsteps += 1
            chain_previous = new_chain
        return chain_trajectory

    # def _make_chain_frag(self, chain: Chain, pair_of_inds):
    def _make_chain_frag(self, chain: Chain, geom_pair, ind_pair):
        start_ind, end_ind = ind_pair
        opt_start, opt_end = geom_pair
        # chain_frag_nodes = chain.nodes[start_ind: end_ind + 1]
        # chain_frag = Chain(
        #     nodes=[opt_start] + chain_frag_nodes + [opt_end],
        #     parameters=self.inputs.chain_inputs,
        # )

        # JDEP 01132025: Going to not recycle fragment nodes. Want a fresh
        # interpolation
        chain_frag = chain.model_copy(update={
            "nodes": [opt_start, opt_end],
            "parameters": self.inputs.chain_inputs})
        # opt_start = chain[start].do_geometry_optimization()
        # opt_end = chain[end].do_geometry_optimization()

        # chain_frag.insert(0, opt_start)
        # chain_frag.append(opt_end)
        # print(f"using a frag of {len(chain_frag)} nodes")
        return chain_frag

    def _make_chain_pair(self, chain: Chain, pair_of_inds):
        start, end = pair_of_inds
        start_opt = chain[start].do_geometry_optimization()
        end_opt = chain[end].do_geometry_optimization()

        chain_frag = Chain(
            nodes=[start_opt, end_opt], parameters=self.inputs.chain_inputs
        )

        return chain_frag

    def _do_minima_based_split(self, chain: Chain, minimization_results: List[Node]):
        import neb_dynamics.chainhelpers as ch

        all_geometries = [chain[0]]
        all_geometries.extend(minimization_results)
        all_geometries.append(chain[-1])

        all_inds = [0]
        ind_minima = ch._get_ind_minima(chain)
        all_inds.extend(ind_minima)
        all_inds.append(len(chain) - 1)

        # pairs_inds = list(pairwise(all_inds))
        pairs_geoms = list(pairwise(all_geometries))

        chains = []
        # for geom_pair, ind_pair in zip(pairs_geoms, pairs_inds):
        for geom_pair in pairs_geoms:
            chains.append(self._make_chain_frag(
                chain=chain, geom_pair=geom_pair, ind_pair=(None, None)))

        return chains

    def _do_maxima_based_split(
        self, chain: Chain, minimization_results: List[Node]
    ) -> List[Chain]:
        """
        Will take a chain that needs to be split based on 'maxima' criterion and outputs
        a list of Chains to be minimized.
        Args:
            chain (Chain): _description_
            minimization_results (List[Node]): a length-2 list containing the Reactant and Product
            geometries found through the `elemetnarystep.pseudo_irc()`

        Returns:
            List[Chain]: list of chains to be minimized
        """
        r, p = minimization_results
        chains_list = []

        # add the input_R->R
        nodes = [chain[0], r]
        chain_frag = chain.copy()
        chain_frag.nodes = nodes
        chains_list.append(chain_frag)

        # add the R->P transition
        nodes = [r, p]
        chain_frag2 = chain.copy()
        chain_frag2.nodes = nodes
        chains_list.append(chain_frag2)

        # add the P -> input_P
        nodes3 = [p, chain[len(chain) - 1]]
        chain_frag3 = chain.copy()
        chain_frag3.nodes = nodes3
        chains_list.append(chain_frag3)

        return chains_list

    def make_sequence_of_chains(
        self, chain: Chain, split_method: str, minimization_results: List[Node]
    ) -> List[Chain]:
        """
        Takes an input chain, `chain`, that needs to be split accordint to `split_method` ("minima", "maxima")
        Returns a list of Chain objects to be minimzed.
        """
        if split_method == "minima":
            chains = self._do_minima_based_split(chain, minimization_results)

        elif split_method == "maxima":
            chains = self._do_maxima_based_split(chain, minimization_results)

        return chains


def _parallel_recursive_step_worker(
    run_inputs: RunInputs,
    input_chain: Chain,
    tree_node_index: int,
) -> tuple[TreeNode, list[Chain]]:
    try:
        local_inputs = copy.deepcopy(run_inputs)
    except Exception:
        local_inputs = run_inputs

    try:
        local_inputs.path_min_inputs.v = True
    except Exception:
        pass

    with progress_monitor(f"branch-{int(tree_node_index)}"):
        runner = MSMEP(inputs=local_inputs)
        return runner._run_recursive_step(
            input_chain=input_chain,
            tree_node_index=tree_node_index,
        )
