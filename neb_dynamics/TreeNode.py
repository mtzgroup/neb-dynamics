from dataclasses import dataclass
from functools import cached_property
from neb_dynamics.NEB import NEB
from pathlib import Path
import numpy as np
import networkx as nx
import shutil
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer


@dataclass
class TreeNode:
    data: NEB
    children: list
    index: int

    _force_tree_recalc: bool = False

    @property
    def max_index(self):
        max_found = 0
        for node in self.depth_first_ordered_nodes:
            if node.index > max_found:
                max_found = node.index
        return max_found

    @property
    def n_children(self):
        return len(self.children)

    @property
    def depth_first_ordered_nodes(self) -> list:
        if self.is_leaf and len(self.children) == 0:
            return [self]
        else:
            nodes = [self]
            for child in self.children:
                out_nodes = child.depth_first_ordered_nodes
                nodes.extend(out_nodes)
        return nodes

    @property
    def ordered_leaves(self):
        leaves = []
        for node in self.depth_first_ordered_nodes:
            if node.is_leaf and bool(node.data):
                leaves.append(node)
        return leaves

    @classmethod
    def max_depth(cls, node, depth=0):
        if node.is_leaf:
            return depth
        else:
            max_depths = []
            for child in node.children:
                max_depths.append(cls.max_depth(child, depth + 1))

            return max(max_depths)

    @property
    def total_nodes(self):
        return len(self.depth_first_ordered_nodes)

    def get_num_opt_steps(self):
        return sum(
            [
                len(leaf.chain_trajectory)
                for leaf in self.get_optimization_history()
                if leaf
            ]
        )

    def get_num_grad_calls(self):
        return sum(
            [leaf.grad_calls_made for leaf in self.get_optimization_history() if leaf]
        )

    def get_nodes_at_depth(self, depth):
        curr_depth = 0
        nodes_to_iter_through = [self]
        while curr_depth < depth:
            new_nodes_to_iter_through = []
            for node in nodes_to_iter_through:
                new_nodes_to_iter_through.extend(node.children)
            curr_depth += 1
            nodes_to_iter_through = new_nodes_to_iter_through

        return nodes_to_iter_through

    def write_to_disk(self, folder_name: Path):

        if folder_name.exists():
            shutil.rmtree(folder_name)
        folder_name.mkdir()

        np.savetxt(fname=folder_name / "adj_matrix.txt", X=self.adj_matrix)

        for node in self.depth_first_ordered_nodes:
            i = node.index
            if node.data:
                node.data.write_to_disk(
                    fp=folder_name / f"node_{i}.xyz", write_history=True
                )

    def draw(self):
        foo = self.adj_matrix - np.identity(len(self.adj_matrix))
        g = nx.from_numpy_array(foo)
        nx.draw_networkx(g)

    def _update_adj_matrix(self, matrix, node):
        matrix_copy = matrix.copy()
        if node.data:
            matrix_copy[node.index, node.index] = 1
        if node.is_leaf:
            return matrix_copy
        else:
            for child in node.children:
                matrix_copy[node.index, child.index] = 1
                matrix_copy = self._update_adj_matrix(matrix=matrix_copy, node=child)

        return matrix_copy

    @property
    def adj_matrix(self):
        # mat = np.zeros((self.total_nodes, self.total_nodes))
        mat = np.zeros((self.max_index + 1, self.max_index + 1))
        mat = self._update_adj_matrix(matrix=mat, node=self)

        return mat

    @classmethod
    def read_from_disk(
        cls,
        folder_name,
        neb_parameters=NEBInputs(),
        chain_parameters=ChainInputs(),
        gi_parameters=GIInputs(),
        optimizer=VelocityProjectedOptimizer(),
    ):
        if isinstance(folder_name, str):
            folder_name = Path(folder_name)
        adj_mat = np.loadtxt(folder_name / "adj_matrix.txt")
        if len(adj_mat.shape) > 0:

            nodes = list(folder_name.glob("node*.xyz"))
            true_node_indices = [int(p.stem.split("_")[1]) for p in nodes]
            node_list_indices = list(range(len(true_node_indices)))

            translator = {}
            for true_ind, local_ind in zip(true_node_indices, node_list_indices):
                translator[true_ind] = local_ind

            neb_nodes = [
                NEB.read_from_disk(
                    nodes[i],
                    chain_parameters=chain_parameters,
                    neb_parameters=neb_parameters,
                    gi_parameters=gi_parameters,
                    optimizer=optimizer,
                )
                for i in range(len(nodes))
            ]
            root = cls._get_node_helper(
                true_node_index=0,
                matrix=adj_mat,
                list_of_nodes=neb_nodes,
                indices_translator=translator,
            )
        else:
            neb_nodes = [
                NEB.read_from_disk(
                    folder_name / "node_0.xyz",
                    chain_parameters=chain_parameters,
                    neb_parameters=neb_parameters,
                    optimizer=optimizer,
                )
            ]
            root = cls(data=neb_nodes[0], children=[], index=0)

        return root

    @classmethod
    def _get_node_helper(
        cls, true_node_index, matrix, list_of_nodes, indices_translator
    ):

        node = list_of_nodes[indices_translator[true_node_index]]
        row = matrix[true_node_index, true_node_index:]
        ind_nonzero_nodes = row.nonzero()[0] + true_node_index
        ind_children = ind_nonzero_nodes[1:]
        if len(ind_children):
            children = [
                cls._get_node_helper(
                    true_node_index=true_child_index,
                    matrix=matrix,
                    list_of_nodes=list_of_nodes,
                    indices_translator=indices_translator,
                )
                for true_child_index in ind_children
                if matrix[true_child_index]
                .nonzero()[0]
                .any()  # i.e. if it was not a 'None' Node
            ]
            return cls(data=node, children=children, index=true_node_index)
        else:
            return cls(data=node, children=[], index=true_node_index)

    # @property
    @cached_property
    def is_leaf(self):
        if self._force_tree_recalc:  # this should never be used but alas
            return self.data.chain_trajectory[-1].is_elem_step()[0]
        else:
            return len(self.children) == 0

    def get_optimization_history(self, node=None):
        if node:
            opt_history = [node.data]
            for child in node.children:
                if child.is_leaf and len(child.children) == 0:
                    opt_history.extend([child.data])
                else:
                    child_opt_history = self.get_optimization_history(child)
                    opt_history.extend(child_opt_history)
            return opt_history
        else:
            return self.get_optimization_history(node=self)

    def get_adj_mat_leaves_indices(self):
        matrix = self.adj_matrix
        inds = []
        for i, row in enumerate(matrix):
            if len(row.nonzero()[0]) == 1:
                inds.append(i)
        return inds

    @property
    def output_chain(self):
        inds = self.get_adj_mat_leaves_indices()
        chains = [
            node.data.chain_trajectory[-1]
            for node in self.depth_first_ordered_nodes
            if node.index in inds
        ]
        out = Chain.from_list_of_chains(chains, parameters=chains[0].parameters)
        return out
