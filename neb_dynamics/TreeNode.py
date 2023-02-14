from dataclasses import dataclass
from neb_dynamics.NEB import NEB


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
