from dataclasses import dataclass

import sys
import numpy as np
from neb_dynamics.helper_functions import pairwise
from typing import Tuple, List

from neb_dynamics.nodes.node import Node
from neb_dynamics.nodes.nodehelpers import _is_connectivity_identical
from neb_dynamics.elementarystep import check_if_elem_step

from neb_dynamics.chain import Chain
from neb_dynamics.elementarystep import ElemStepResults
from neb_dynamics.neb import NEB, PYGSM, NoneConvergedException
from neb_dynamics.engines import Engine
from neb_dynamics.inputs import NEBInputs, ChainInputs, GIInputs

from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.optimizers.optimizer import Optimizer
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics.errors import ElectronicStructureError

PATH_METHODS = ['NEB', 'PYGSM']


@dataclass
class MSMEP:
    """Class for running autosplitting MEP minimizations."""

    engine: Engine
    neb_inputs: NEBInputs = NEBInputs()
    chain_inputs: ChainInputs = ChainInputs()
    gi_inputs: GIInputs = GIInputs()

    optimizer: Optimizer = VelocityProjectedOptimizer()
    path_min_method: str = 'NEB'

    def find_mep_multistep(self, input_chain: Chain, tree_node_index=0) -> TreeNode:
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

        if self.neb_inputs.skip_identical_graphs and input_chain[0].has_molecular_graph:
            if _is_connectivity_identical(input_chain[0], input_chain[-1]):
                print("Endpoints are identical. Returning nothing")
                return TreeNode(data=None, children=[], index=tree_node_index)

        ch._reset_node_convergence(input_chain)
        self.engine.compute_gradients(input_chain)

        if input_chain[0] == input_chain[-1]:
            print("Endpoints are identical. Returning nothing")
            return TreeNode(data=None, children=[], index=tree_node_index)

        try:
            root_neb_obj, elem_step_results = self.minimize_chain(
                input_chain=input_chain
            )
            history = TreeNode(data=root_neb_obj, children=[], index=tree_node_index)

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
                    out_history = self.find_mep_multistep(
                        chain_frag, tree_node_index=new_tree_node_index
                    )

                    history.children.append(out_history)

                    # increment the node indices
                    new_tree_node_index = out_history.max_index + 1
                return history

        except ElectronicStructureError:
            return TreeNode(data=None, children=[], index=tree_node_index)

    def _create_interpolation(self, chain: Chain):
        import neb_dynamics.chainhelpers as ch

        if chain.parameters.use_geodesic_interpolation:
            if chain.parameters.friction_optimal_gi:
                interpolation = ch.create_friction_optimal_gi(
                    chain=chain, gi_inputs=self.gi_inputs.copy()
                )
            else:
                interpolation = ch.run_geodesic(
                    chain=chain,
                    nimages=self.gi_inputs.nimages,
                    friction=self.gi_inputs.friction,
                    nudge=self.gi_inputs.nudge,
                    **self.gi_inputs.extra_kwds,
                )

            interpolation._zero_velocity()

        else:  # do a linear interpolation using numpy
            start_point = chain[0].coords
            end_point = chain[-1].coords
            coords = np.linspace(
                start=start_point, stop=end_point, num=self.gi_inputs.nimages
            )
            nodes = [
                node.update_coords(c)
                for node, c in zip([chain.nodes[0]] * len(coords), coords)
            ]
            interpolation = Chain(nodes=nodes, parameters=self.chain_inputs)

        return interpolation

    def minimize_chain(self, input_chain: Chain) -> Tuple[NEB, ElemStepResults]:
        assert self.path_min_method.upper() in PATH_METHODS, f"Invalid path method: {self.path_min_method}. Allowed are: {PATH_METHODS}"

        # make sure the chain parameters are reset
        # if they come from a converged chain
        input_chain.parameters = self.chain_inputs
        if len(input_chain) != self.gi_inputs.nimages:
            interpolation = self._create_interpolation(input_chain)
            assert (
                len(interpolation) == self.gi_inputs.nimages
            ), f"Geodesic interpolation wrong length.\
                 Requested: {self.gi_inputs.nimages}. Given: {len(interpolation)}"

        else:
            interpolation = input_chain

        print("Running path minimization...")
        if self.path_min_method.upper() == 'NEB':
            print("Using in-house NEB optimizer")
            sys.stdout.flush()

            n = NEB(
                initial_chain=interpolation,
                parameters=self.neb_inputs.copy(),
                optimizer=self.optimizer.copy(),
                engine=self.engine,
            )
        elif self.path_min_method.upper() == "PYGSM":
            print("Using PYGSM optimizer")
            n = PYGSM(
                initial_chain=interpolation,
                engine=self.engine,
                pygsm_kwds=self.neb_inputs.pygsm_kwds
            )

        try:
            elem_step_results = n.optimize_chain()
            out_chain = n.optimized

        except NoneConvergedException:
            print(
                "\nWarning! A chain did not converge.\
                        Returning an unoptimized chain..."
            )
            out_chain = n.chain_trajectory[-1]
            elem_step_results = check_if_elem_step(out_chain, engine=self.engine)

        except ElectronicStructureError as e:
            print(
                "\nWarning! A chain has electronic structure errors. \
                    Returning an unoptimized chain..."
            )
            print(e)
            out_chain = n.chain_trajectory[-1]
            elem_step_results = ElemStepResults(
                is_elem_step=True,
                is_concave=None,
                splitting_criterion=None,
                minimization_results=None,
                number_grad_calls=0,
            )

        return n, elem_step_results

    # def _make_chain_frag(self, chain: Chain, pair_of_inds):
    def _make_chain_frag(self, chain: Chain, geom_pair, ind_pair):
        start_ind, end_ind = ind_pair
        opt_start, opt_end = geom_pair
        chain_frag_nodes = chain.nodes[start_ind: end_ind + 1]
        chain_frag = Chain(
            nodes=[opt_start] + chain_frag_nodes + [opt_end],
            parameters=chain.parameters,
        )
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

        chain_frag = Chain(nodes=[start_opt, end_opt], parameters=chain.parameters)

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

        pairs_inds = list(pairwise(all_inds))
        pairs_geoms = list(pairwise(all_geometries))

        chains = []
        for geom_pair, ind_pair in zip(pairs_geoms, pairs_inds):
            chains.append(self._make_chain_frag(chain, geom_pair, ind_pair))

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
