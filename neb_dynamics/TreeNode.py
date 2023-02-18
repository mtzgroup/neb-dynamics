from dataclasses import dataclass
from neb_dynamics.NEB import NEB
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class TreeNode:
    data: NEB
    children: list

    @classmethod
    def from_node_list(cls, recursive_list):
        fixed_list = [val for val in recursive_list if val is not None]
        if len(fixed_list) == 1:
            if isinstance(fixed_list[0], NEB):
                return cls(data=fixed_list[0], children=[])
            elif isinstance(fixed_list[0], list):
                list_of_nodes = [
                    cls.from_node_list(recursive_list=l) for l in fixed_list[0]
                ]
                return cls(data=list_of_nodes[0], children=list_of_nodes[1:])
        else:
            children = [cls.from_node_list(recursive_list=l) for l in fixed_list[1:]]

            return cls(data=fixed_list[0], children=children)

    
    @property
    def depth_first_ordered_nodes(self) -> list:
        nodes = []
        for d in range(0, self.max_depth + 1):
            n = self.get_nodes_at_depth(d)
            nodes.extend(n)

        return nodes

    @property
    def max_depth(self):
        d = 0
        n_nodes = len(self.get_nodes_at_depth(d))
        while n_nodes > 0:
            d += 1
            n_nodes = len(self.get_nodes_at_depth(d))
        return d - 1

    @property
    def total_nodes(self):
        return len(self.depth_first_ordered_nodes)

    def get_nodes_at_depth(self, depth):
        curr_depth = 0
        nodes_to_iter_through = self.children
        while curr_depth < depth:
            new_nodes_to_iter_through = []
            for node in nodes_to_iter_through:
                new_nodes_to_iter_through.extend(node.children)
            curr_depth += 1
            nodes_to_iter_through = new_nodes_to_iter_through

        return nodes_to_iter_through

    def write_to_disk(self, folder_name: Path):
        if not folder_name.exists():
            folder_name.mkdir()

        for i, node in enumerate(self.depth_first_ordered_nodes):
            node.data.write_to_disk(
                fp=folder_name / f"node_{i}.xyz", write_history=True
            )

        np.savetxt(fname=folder_name / "adj_matrix.txt", X=self.adj_matrix)

    @property
    def adj_matrix(self):
        mat = np.identity(self.total_nodes)
        all_nodes = self.depth_first_ordered_nodes
        for i, node in enumerate(all_nodes):
            mat = self._update_adj_matrix(row_ind=i, matrix=mat, node=node)
        return mat
    
    def _update_adj_matrix(self, row_ind, matrix, node):
        matrix_copy = matrix.copy()
        children = node.children
        if len(children) > 0:
            start_col = row_ind + 1
            end_col = start_col + len(children)
            matrix_copy[row_ind, start_col:end_col] = 1

        return matrix_copy

    @classmethod
    def read_from_disk(cls, folder_name):
        adj_mat = np.loadtxt(folder_name / "adj_matrix.txt")

        nodes = list(folder_name.glob("node*.xyz"))
        n_nodes = len(nodes)
        neb_nodes = [NEB.read_from_disk(nodes[i]) for i in range(n_nodes)]
        root = TreeNode._get_node_helper(
            ind_parent=0, matrix=adj_mat, list_of_nodes=neb_nodes
        )

        return root

    def draw(self):
        foo = self.adj_matrix - np.identity(len(self.adj_matrix))
        g = nx.from_numpy_matrix(foo)
        plt.figure()
        nx.draw_networkx(g)
        plt.show()


    def draw(self):
        foo = self.adj_matrix - np.identity(len(self.adj_matrix))
        g = nx.from_numpy_matrix(foo)
        nx.draw_networkx(g)

    @classmethod
    def _get_node_helper(cls, ind_parent, matrix, list_of_nodes):
        node = list_of_nodes[ind_parent]
        row = matrix[ind_parent, ind_parent:]
        ind_nonzero_nodes = row.nonzero()[0] + ind_parent
        ind_children = ind_nonzero_nodes[1:]
        if len(ind_children):
            children = [
                TreeNode._get_node_helper(
                    ind_parent=j, matrix=matrix, list_of_nodes=list_of_nodes
                )
                for j in ind_children
            ]
            return TreeNode(data=node, children=children)
        else:
            return TreeNode(data=node, children=[])


    @property
    def is_leaf(self):
        return len(self.children) == 0

    def get_optimization_history(self, node=None):
        if node:
            opt_history = [node.data]
            for child in node.children:
                if child.is_leaf:
                    opt_history.extend([child.data])
                else:
                    child_opt_history = self.get_optimization_history(child)
                    opt_history.extend(child_opt_history)
            return opt_history
        else:
            return self.get_optimization_history(node=self)