from dataclasses import dataclass
from functools import cached_property
from neb_dynamics.NEB import NEB
from pathlib import Path
import numpy as np
import networkx as nx
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs

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
            if node.is_leaf: 
                leaves.append(node)
        return leaves
    
    @classmethod
    def max_depth(cls, node, depth=0):
        if node.is_leaf:
            return depth
        else:
            max_depths = []
            for child in node.children:
                max_depths.append(cls.max_depth(child, depth+1))

            return max(max_depths)

    @property
    def total_nodes(self):
        return len(self.depth_first_ordered_nodes)

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
        if not folder_name.exists():
            folder_name.mkdir()

        np.savetxt(fname=folder_name / "adj_matrix.txt", X=self.adj_matrix)
        
        for node in self.depth_first_ordered_nodes:
            i = node.index
            node.data.write_to_disk(
                fp=folder_name / f"node_{i}.xyz", write_history=True
            )

        


    def draw(self):
        foo = self.adj_matrix - np.identity(len(self.adj_matrix))
        g = nx.from_numpy_matrix(foo)
        nx.draw_networkx(g)



    def _update_adj_matrix(self, ind, matrix, node, free_inds):
            matrix_copy = matrix.copy()
            node.index = free_inds[0]
            free_inds.pop(0)
            
            if node.is_leaf:
                return matrix_copy
            else:
                for i, child in enumerate(node.children,start=1):
                    child_ind = free_inds[0]
                    matrix_copy[ind, child_ind] = 1
                    matrix_copy = self._update_adj_matrix(ind=ind+i, matrix=matrix_copy, node=child, free_inds=free_inds)

            return matrix_copy
        
    @property
    def adj_matrix(self):
        mat = np.identity(self.total_nodes)
        free_inds = list(range(self.total_nodes))

        mat = self._update_adj_matrix(ind=0, matrix=mat, node=self, free_inds=free_inds)
        
        return mat

    @classmethod
    def read_from_disk(cls, folder_name, neb_parameters=NEBInputs(), chain_parameters=ChainInputs()):
        adj_mat = np.loadtxt(folder_name / "adj_matrix.txt")

        nodes = list(folder_name.glob("node*.xyz"))
        n_nodes = len(nodes)
        neb_nodes = [NEB.read_from_disk(nodes[i], chain_parameters=chain_parameters, neb_parameters=neb_parameters) for i in range(n_nodes)]
        root = TreeNode._get_node_helper(
            ind_parent=0, matrix=adj_mat, list_of_nodes=neb_nodes
        )

        return root


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


    # @property
    @cached_property
    def is_leaf(self):
        # return len(self.children) == 0
        # TODO: fix file labelling bug
        return self.data.chain_trajectory[-1].is_elem_step()[0]

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
    
    @property
    def output_chain(self):
        leaves = self.ordered_leaves
        chains = []
        for leaf in leaves:
            c = leaf.data.chain_trajectory[-1]
            chains.append(c)
        out = Chain.from_list_of_chains(chains, parameters=chains[0].parameters)
        return out