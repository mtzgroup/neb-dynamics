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
        insertion_points = self._get_insertion_points_leaves(history.ordered_leaves,start_chain=start_chain)
        list_of_cleanup_nodes = [TreeNode(data=neb_i, children=[]) for neb_i in list_of_cleanup_nebs]
        new_leaves = history.ordered_leaves
        print('before:',len(new_leaves))
        for insertion_ind, tree_node in zip(insertion_points, list_of_cleanup_nodes):
            if insertion_ind == -1:
                new_leaves.append(tree_node)
            else:
                new_leaves.insert(insertion_ind, tree_node)
        print('after:',len(new_leaves))
        new_chains = [leaf.data.optimized for leaf in new_leaves]
        clean_out_chain = Chain.from_list_of_chains(new_chains,parameters=start_chain.parameters)
        return clean_out_chain
    
    def create_clean_msmep(self, history, out_path=None):
        start_chain = history.data.initial_chain
        list_of_cleanup_nebs = self.cleanup_nebs(starting_chain=start_chain, history_obj=history, out_path=out_path)
        
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


    def _do_minima_based_split(self, chain):
        chains_list = []
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
        

        
    def cleanup_nebs(self, starting_chain, history_obj, out_path=None):
        leaves = history_obj.ordered_leaves
        cleanup_results = []
        original_start = starting_chain[0]
        insertion_indices = self._get_insertion_points_leaves(leaves=leaves,start_chain=starting_chain)
        for index in insertion_indices:
            if index == 0:
                prev_end = original_start
                curr_start = leaves[index].data.optimized[0]
                
            elif index == -1: # need to do a conformer rearrangement from product conformer to input product conformer
                prev_end = leaves[index].data.optimized[-1]
                curr_start = starting_chain[index]
                
            else:
                prev_end = leaves[index-1].data.optimized[-1]
                curr_start = leaves[index].data.optimized[0]
            
            chain_pair = Chain(nodes=[prev_end, curr_start], parameters=starting_chain.parameters)
            neb_obj, _ = self.get_neb_chain(chain_pair)
            if out_path:
                fp = out_path / f"cleanup_neb_{index}.xyz"
                if not out_path.exists():
                    out_path.mkdir()
                    
                neb_obj.write_to_disk(fp, write_history=True)
            
            cleanup_results.append(neb_obj)

        return cleanup_results
    
    def _get_insertion_points_leaves(self, leaves, start_chain):
        """
        returns a list of indices 
        """
        original_start = start_chain[0]
        original_end = start_chain[-1]
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
        
        # check if the final added structure         
        last_end = leaves[-1].data.optimized[-1]
        if not last_end._is_conformer_identical(original_end):
            insertion_indices.append(-1)
        return insertion_indices



