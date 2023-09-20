from dataclasses import dataclass
from neb_dynamics.Chain import Chain
from neb_dynamics.TreeNode import TreeNode
from pathlib import Path
from functools import cached_property
from neb_dynamics.MSMEP import MSMEP


@dataclass
class Janitor:
    history_object: TreeNode
    msmep_object: MSMEP
    out_path: Path = None
    
    
    @property
    def starting_chain(self):
        return self.history_object.data.initial_chain
    
    @cached_property
    def insertion_points(self):
        """
        returns a list of indices 
        """
        original_start = self.starting_chain[0]
        original_end = self.starting_chain[-1]
        leaves = [l for l in self.history_object.ordered_leaves if l.data]
        insertion_indices = []
        for i, leaf in enumerate(leaves):
            if leaf.data:
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
    
    
    def cleanup_nebs(self):
        leaves = self.history_object.ordered_leaves
        cleanup_results = []
        original_start = self.starting_chain[0]
        
        for index in self.insertion_points:
            if index == 0:
                prev_end = original_start
                curr_start = leaves[index].data.optimized[0]
                
            elif index == -1: # need to do a conformer rearrangement from product conformer to input product conformer
                prev_end = leaves[index].data.optimized[-1]
                curr_start = self.starting_chain[index]
                
            else:
                prev_end = leaves[index-1].data.optimized[-1]
                curr_start = leaves[index].data.optimized[0]
            
            chain_pair = Chain(nodes=[prev_end, curr_start], parameters=self.starting_chain.parameters)
            neb_obj, _ = self.msmep_object.get_neb_chain(chain_pair)
            if self.out_path:
                fp = self.out_path / f"cleanup_neb_{index}.xyz"
                if not self.out_path.exists():
                    self.out_path.mkdir()
                    
                neb_obj.write_to_disk(fp, write_history=True)
            
            cleanup_results.append(neb_obj) 

        return cleanup_results
    
    
    
    def merge_by_indices(
        self,
        insertions_inds, 
        insertions_vals,
        orig_inds, 
        orig_values
    ):
    
        insertions_inds = insertions_inds.copy()
        insertions_vals = insertions_vals.copy()
        orig_inds = orig_inds.copy()
        orig_values = orig_values.copy()
        out = []
        while len(insertions_inds) > 0:
            if 0 <= insertions_inds[0] <= orig_inds[0]:
                out.append(insertions_vals[0])
                insertions_inds.pop(0)
                insertions_vals.pop(0)
            elif insertions_inds[0] == -1 and len(orig_values) == 0:
                out.append(insertions_vals[0])
                insertions_inds.pop(0)
                insertions_vals.pop(0)
            else:
                out.append(orig_values[0])
                orig_inds.pop(0)
                orig_values.pop(0)

        if len(orig_values) > 0:
            out.extend(orig_values)
            
        return out
    
    
    def _merge_cleanups_and_leaves(self, list_of_cleanup_nebs):
        list_of_cleanup_nodes = [TreeNode(data=neb_i, children=[], index=99) for neb_i in list_of_cleanup_nebs]
        orig_leaves = self.history_object.ordered_leaves
        print('before:',len(orig_leaves))
        new_leaves = self.merge_by_indices(
            insertions_inds=self.insertion_points,
            insertions_vals=list_of_cleanup_nodes,
            orig_inds=list(range(len(orig_leaves))),
            orig_values=orig_leaves)
        print('after:',len(new_leaves))
        new_chains = [leaf.data.optimized for leaf in new_leaves]
        clean_out_chain = Chain.from_list_of_chains(new_chains,parameters=self.starting_chain.parameters)
        return clean_out_chain
    
    def create_clean_msmep(self):        
        list_of_cleanup_nebs = self.cleanup_nebs()
        if len(list_of_cleanup_nebs) == 0:
            return None
        
        clean_out_chain = self._merge_cleanups_and_leaves(list_of_cleanup_nebs)
        return clean_out_chain
    