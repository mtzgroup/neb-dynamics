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
from qcio import Structure

from neb_dynamics.pathminimizers.fneb import FreezingNEB
from neb_dynamics.inputs import RunInputs

import traceback
import copy
import logging

PATH_METHODS = ["NEB", "FNEB", "MLPGI"]


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
            assert (
                self.inputs.path_min_method.upper() in PATH_METHODS
            ), f"Invalid path method: {self.inputs.path_min_method}. Allowed are: {PATH_METHODS}"

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

        if self.inputs.path_min_inputs.skip_identical_graphs and input_chain[0].has_molecular_graph:
            if _is_connectivity_identical(input_chain[0], input_chain[-1]):
                print("Endpoints are identical. Returning nothing")
                return TreeNode(data=None, children=[], index=tree_node_index)

        ch._reset_node_convergence(input_chain)
        self.inputs.engine.compute_gradients(input_chain)

        if is_identical(
            self=input_chain[0],
            other=input_chain[-1],
            fragment_rmsd_cutoff=self.inputs.chain_inputs.node_rms_thre,
            kcal_mol_cutoff=self.inputs.chain_inputs.node_ene_thre,
        ):
            print("Endpoints are identical. Returning nothing")
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
                # the last chain in the minimization
                chain = root_neb_obj.chain_trajectory[-1]
                sequence_of_chains = self.make_sequence_of_chains(
                    chain=chain,
                    split_method=elem_step_results.splitting_criterion,
                    minimization_results=elem_step_results.minimization_results,
                )
                print(
                    f"Splitting chains based on: {elem_step_results.splitting_criterion}"
                )
                new_tree_node_index = tree_node_index + 1
                for i, chain_frag in enumerate(sequence_of_chains, start=1):
                    print(f"On chain {i} of {len(sequence_of_chains)}...")
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

    def _create_interpolation(self, chain: Chain):
        import neb_dynamics.chainhelpers as ch
        logger = logging.getLogger('neb_dynamics.geodesic_interpolation2.interpolation')
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
                print("will be freezing inds:", inds_frozen, ' during geodesic interpolation')
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
        if self.inputs.path_min_method.upper() == "NEB":

            print("Using in-house NEB optimizer")
            sys.stdout.flush()
            optimizer = ConjugateGradient(**self.inputs.optimizer_kwds)

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

        elif self.inputs.path_min_method.upper() == "FNEB":
            print("Using Freezing NEB optimizer")
            n = FreezingNEB(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
                parameters=self.inputs.path_min_inputs,
                optimizer=self.inputs.optimizer,
                gi_inputs=self.inputs.gi_inputs
            )


        elif self.inputs.path_min_method.upper() == "MLPGI":
            from neb_dynamics.pathminimizers.mlpgi import MLPGI
            print("Using MLP Geodesic Optimizer")
            n = MLPGI(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
            )
        else:
            raise NotImplementedError("Invalid path minimization method. Select from NEB, FNEB, or MLPGI.")

        return n

    def run_minimize_chain(self, input_chain: Chain) -> Tuple[NEB, ElemStepResults]:
        if isinstance(input_chain, list):
            input_chain = Chain.model_validate(
                {"nodes": input_chain, "parameters": self.inputs.chain_inputs})

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

        print("Running path minimization...")

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
                    out_chain, engine=self.inputs.engine)
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
            # print(e)
            if e.obj is not None:
                for i, po in enumerate(e.obj):
                    print(f"Traceback {i}:")
                    po.ptraceback
            out_chain = n.chain_trajectory[-1]
            elem_step_results = ElemStepResults(
                is_elem_step=True,
                is_concave=None,
                splitting_criterion=None,
                minimization_results=None,
                number_grad_calls=0,
            )

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

            print(
                f"step {nsteps} // argmax(|TS gperp|) {np.amax(np.abs(ts_guess_grad))} // \
                    max rms grad {max_rms_grad_val} // armax(|TS_triplet_gsprings|) \
                        {new_chain.ts_triplet_gspring_infnorm} // \
                            temperature={np.linalg.norm(new_chain.velocity)}//{' '*20}",
                end="\r",
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
        print(f"using a frag of {len(chain_frag)} nodes")
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
