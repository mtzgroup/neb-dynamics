from dataclasses import dataclass, field
from chain import Chain
from neb_dynamics.treenode import TreeNode
from neb_dynamics.neb import NEB
from pathlib import Path
from functools import cached_property
from neb_dynamics.msmep import MSMEP


@dataclass
class Janitor:
    history_object: TreeNode
    msmep_object: MSMEP
    reaction_leaves: list = None
    out_path: Path = None
    cleanup_trees: list[TreeNode] = field(default_factory=list)

    def get_n_grad_calls(self):
        return sum([tree.get_num_grad_calls() for tree in self.cleanup_trees])

    def __post_init__(self):
        if self.reaction_leaves is None:
            assert isinstance(
                self.history_object, TreeNode
            ), "Need to either input a TreeNode for history or give a list of reaction chains to cleanup"
            self.reaction_leaves = [
                obj.data for obj in self.history_object.ordered_leaves
            ]
        if self.history_object is None:
            raise NotImplementedError(
                "Need to give a history object as input. In the future thiss will be able to be a chain but not yet."
            )

    @property
    def starting_chain(self):
        if isinstance(self.history_object, TreeNode):
            return self.history_object.data.initial_chain
        elif isinstance(self.history_object, Chain):
            return self.history_object

    @cached_property
    def insertion_points(self):
        """
        returns a list of indices
        """
        original_start = self.starting_chain[0]
        original_end = self.starting_chain[-1]
        leaves = [leaf for leaf in self.reaction_leaves]
        insertion_indices = []
        for i, leaf in enumerate(leaves):
            if i == 0:
                prev_end = original_start
            else:
                prev_end = leaves[i - 1].optimized[-1]

            curr_start = leaf.optimized[0]
            if not prev_end._is_conformer_identical(curr_start):
                insertion_indices.append(i)

        # check if the final added structure
        last_end = leaves[-1].optimized[-1]
        if not last_end._is_conformer_identical(original_end):
            insertion_indices.append(-1)
        return insertion_indices

    def cleanup_nebs(self):
        leaves = [leaf for leaf in self.reaction_leaves]
        original_start = self.starting_chain[0]

        for index in self.insertion_points:
            if index == 0:
                prev_end = original_start
                curr_start = leaves[index].optimized[0]

            elif (
                index == -1
            ):  # need to do a conformer rearrangement from product conformer to input product conformer
                prev_end = leaves[index].optimized[-1]
                curr_start = self.starting_chain[index]

            else:
                prev_end = leaves[index - 1].optimized[-1]
                curr_start = leaves[index].optimized[0]

            chain_pair = Chain(
                nodes=[prev_end, curr_start],
                parameters=self.msmep_object.chain_inputs.copy(),
            )
            h, output_chain = self.msmep_object.find_mep_multistep(chain_pair)

            self.cleanup_trees.append(h)
            # cleanup_results.append(h.ordered_leaves)

        # return cleanup_results

    def write_to_disk(self, out_path: Path):
        out_path.mkdir(exist_ok=True)
        for index, h in enumerate(self.cleanup_trees):
            fp = out_path / f"cleanup_neb_{index}.xyz"

            h.write_to_disk(fp)

    def merge_by_indices(
        self, insertions_inds, insertions_vals, orig_inds, orig_values
    ):
        # print(f"{insertions_inds}")
        # print(f"{insertions_vals}")
        insertions_inds = insertions_inds.copy()
        insertions_vals = insertions_vals.copy()
        orig_inds = orig_inds.copy()
        orig_values = orig_values.copy()
        out = []
        while len(insertions_inds) > 0:
            if 0 <= insertions_inds[0] <= orig_inds[0]:
                out.append(insertions_vals[0].optimized)
                insertions_inds.pop(0)
                insertions_vals.pop(0)
            elif insertions_inds[0] == -1 and len(orig_values) == 0:
                out.append(insertions_vals[0].optimized)
                insertions_inds.pop(0)
                insertions_vals.pop(0)
            else:
                out.append(orig_values[0].optimized)
                orig_inds.pop(0)
                orig_values.pop(0)

        if len(orig_values) > 0:
            out.extend([n.optimized for n in orig_values])

        return out

    def _merge_cleanups_and_leaves(self, list_of_cleanup_nebs: list[NEB]):
        # list_of_cleanup_nodes = [TreeNode(data=neb_i, children=[], index=99) for neb_i in list_of_cleanup_nebs]
        orig_leaves = self.reaction_leaves
        print("before:", len(orig_leaves))
        new_leaves = self.merge_by_indices(
            insertions_inds=self.insertion_points,
            insertions_vals=list_of_cleanup_nebs,
            orig_inds=list(range(len(orig_leaves))),
            orig_values=orig_leaves,
        )
        print("after:", len(new_leaves))
        # new_chains = [leaf.data.optimized for leaf in new_leaves]
        clean_out_chain = Chain.from_list_of_chains(
            new_leaves, parameters=self.starting_chain.parameters
        )
        return clean_out_chain

    def create_clean_msmep(self):
        if len(self.cleanup_trees) == 0:
            self.cleanup_nebs()

        if len(self.cleanup_trees) == 0:
            return None

        list_of_cleanup_nebs = []
        for tree in self.cleanup_trees:
            list_of_cleanup_nebs.extend([leaf.data for leaf in tree.ordered_leaves])

        clean_out_chain = self._merge_cleanups_and_leaves(list_of_cleanup_nebs)
        return clean_out_chain
