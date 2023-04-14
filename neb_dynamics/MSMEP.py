from dataclasses import dataclass

import numpy as np
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from retropaths.helper_functions import pairwise

from neb_dynamics.Chain import Chain
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Inputs import NEBInputs, ChainInputs, GIInputs
from neb_dynamics.helper_functions import _get_ind_minima, _get_ind_maxima
from neb_dynamics.TreeNode import TreeNode

@dataclass
class MSMEP:

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
        target.gum_mm_optimization()
        target_opt = target.xtb_geom_optimization()

        if not root_opt.molecule_rp == root.molecule_rp:
            raise ValueError(
                "Pseudoaligned start molecule was not a minimum at this level of theory. Exiting."
            )

        if not root_opt.molecule_rp == root.molecule_rp:
            raise ValueError(
                "Product molecule was not a minimum at this level of theory. Exiting."
            )

        return root_opt, target_opt

    def find_mep_multistep(self, input_chain):
        
        if input_chain[0].is_a_molecule:
            if input_chain[0]._is_connectivity_identical(input_chain[-1]):
                print("Endpoints are identical. Returning nothing")
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

            gi = traj.run_geodesic(
                nimages=self.gi_inputs.nimages,
                friction=self.gi_inputs.friction,
                nudge=self.gi_inputs.nudge,
                **self.gi_inputs.extra_kwds,
            )
            interpolation = Chain.from_traj(traj=gi, parameters=self.chain_inputs)

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

        n = NEB(initial_chain=interpolation, parameters=self.neb_inputs)
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
    
    def _merge_cleanups_and_leaves(self, list_of_cleanup_nebs, history):
        start_chain = history.data.initial_chain
        insertion_points = self._get_insertion_points_leaves(history.ordered_leaves,original_start=start_chain[0])
        list_of_cleanup_nodes = [TreeNode(data=neb_i, children=[]) for neb_i in list_of_cleanup_nebs]
        new_leaves = history.ordered_leaves
        print('before:',len(new_leaves))
        for insertion_ind, tree_node in zip(insertion_points, list_of_cleanup_nodes):
            new_leaves.insert(insertion_ind, tree_node)
        print('after:',len(new_leaves))
        new_chains = [leaf.data.optimized for leaf in new_leaves]
        clean_out_chain = Chain.from_list_of_chains(new_chains,parameters=start_chain.parameters)
        return clean_out_chain
    
    def create_clean_msmep(self, history):
        start_chain = history.data.initial_chain
        list_of_cleanup_nebs = self.cleanup_nebs(starting_chain=start_chain, history_obj=history)
        
        if len(list_of_cleanup_nebs) == 0:
            return None
        
        clean_out_chain = self._merge_cleanups_and_leaves(list_of_cleanup_nebs=list_of_cleanup_nebs, history=history)
        return clean_out_chain

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

        chain_frag = Chain(nodes=[start_opt, end_opt], parameters=chain.parameters)

        return chain_frag


    def _do_minima_based_split(self, chain, chains_list):
        all_inds = [0]
        ind_minima = _get_ind_minima(chain)
        all_inds.extend(ind_minima)
        all_inds.append(len(chain) - 1)

        pairs_inds = list(pairwise(all_inds))

        chains = []
        for ind_pair in pairs_inds:
            chains.append(self._make_chain_frag(chain, ind_pair))

        return chains

    def _do_maxima_based_split(self, chain: Chain, chains_list):
        all_inds = []
        ind_maxima = _get_ind_maxima(chain)
        all_inds.extend(ind_maxima)

        if all_inds:
            for ind_maxima in all_inds:
                r, p = chain._approx_irc(index=ind_maxima)
                if not chains_list:
                    # add the start point
                    nodes = [chain[0], r]
                    chain_frag = chain.copy()
                    chain_frag.nodes = nodes
                    chains_list.append(chain_frag)

                nodes = [r, chain[ind_maxima], p]
                chain_frag = chain.copy()
                chain_frag.nodes = nodes
                chains_list.append(chain_frag)

            # add the end point
            nodes = [p, chain[len(chain)-1]]
            chain_frag = chain.copy()
            chain_frag.nodes = nodes
            chains_list.append(chain_frag)
            return chains_list
        else:
            print("There were no maxima found for some reason?")
            print(f"{ind_maxima=}\n{chain.energies=}")

    def make_sequence_of_chains(self, chain, split_method):
        chains_list = []
        if split_method == 'minima':
            chains = self._do_minima_based_split(chain, chains_list)

        elif split_method == 'maxima':
            chains = self._do_maxima_based_split(chain, chains_list)

        return chains


    def stitch_elem_steps(self, list_of_chains):
        out_list_of_chains = [chain for chain in list_of_chains if chain is not None]
        return Chain.from_list_of_chains(
            out_list_of_chains, parameters=self.chain_inputs
        )
        

        
    def cleanup_nebs(self, starting_chain, history_obj):
        leaves = history_obj.ordered_leaves
        cleanup_results = []
        original_start = starting_chain[0]
        insertion_indices = self._get_insertion_points_leaves(leaves=leaves,original_start=original_start)
        for index in insertion_indices:
            if index == 0:
                prev_end = original_start
            else:
                prev_end = leaves[index-1].data.optimized[-1]

            curr_start = leaves[index].data.optimized[0]
            chain_pair = Chain(nodes=[prev_end, curr_start], parameters=starting_chain.parameters)
            neb_obj, _ = self.get_neb_chain(chain_pair)
            cleanup_results.append(neb_obj)

        return cleanup_results
    
    def _get_insertion_points_leaves(self, leaves, original_start):
        """
        returns a list of indices 
        """
        insertion_indices = []
        for i, leaf in enumerate(leaves):
            if i == 0:
                prev_end = original_start
            else:
                prev_end = leaves[i-1].data.optimized[-1]
            
            leaf_chain = leaf.data.optimized
            curr_start = leaf_chain[0]
            if not prev_end._is_conformer_identical(curr_start):
                insertion_indices.append(i)
        return insertion_indices



