from __future__ import annotations

import itertools
import json
import os
from ast import literal_eval
from enum import Enum, auto
from pathlib import Path
from time import time

import networkx as nx
from IPython.core.display import HTML
from pydantic import BaseModel
from timeout_timer import TimeoutInterrupt, timeout

import neb_dynamics.helper_functions as hf
from neb_dynamics.helper_functions import pairwise

from neb_dynamics.d3_tools import draw_d3, forward_to_d3json
from neb_dynamics.neb import NEB

# from retropaths.molecules.molecular_formula import MolecularFormula
from neb_dynamics.molecule import Molecule

# from retropaths.molecules.utilities import give_me_free_index, naturals
# from retropaths.reactions.conditions import Conditions
# from retropaths.reactions.library import Library


class TimeoutPot(TimeoutInterrupt):
    pass


class PotStatus(Enum):
    EMPTY = auto()
    FINISHED = auto()
    FATHER = auto()
    ITERATION = auto()
    TIMEOUT = auto()
    OTHER = auto()


class PotMoleculeSummary(BaseModel):
    """
    NtimesLeaf -> number of times this molecule is seen in the graph in a LEAF node
    NtimesTot -> number of times this molecule is seen in the graph
    TotLeaves -> number of leaves
    TotNodes -> number of total nodes
    Multiplier -> number of copied of the root molecules
    it is important to note that this is done by multiplying the counting
    based on the number of fathers each node has.
    """

    NtimesLeaf: int
    NtimesTot: int
    TotLeaves: int
    TotNodes: int
    Multiplier: int

    def calculate_leaf_based_yield(self):
        """
        This method calculates the yield based on the leaf
        """
        return (self.NtimesLeaf / self.Multiplier) / self.TotLeaves

    def calculate_intermediate_yield(self):
        """
        This calculated the yield disregarding leaves logic
        """
        return (self.NtimesTot / self.Multiplier) / self.TotNodes

    def to_tuple_string(self):
        """a quick serializer just for the numbers, order matters here"""
        # TODO, there must be a better way to do this.
        tupleZ = (
            self.NtimesLeaf,
            self.NtimesTot,
            self.TotLeaves,
            self.TotNodes,
            self.Multiplier,
        )
        return str(tupleZ)

    @classmethod
    def from_tuple_string(cls, string):
        """from the tuple written above, to the object back"""
        a, b, c, d, e = literal_eval(string)
        pms = PotMoleculeSummary(
            NtimesLeaf=a, NtimesTot=b, TotLeaves=c, TotNodes=d, Multiplier=e
        )
        return pms


def serialize_single_pot(pot: Pot, file_name: str, folder: Path):
    """This serializes a list of pots into jsons"""
    name = folder / f"{file_name}.json"
    pot.to_json(name)


def serialize_list_of_pots(pots, root_name, folder):
    """This serializes a list of pots into jsons"""
    for pot in pots:
        file_name = f"{root_name}-M{pot.multiplier}.json"
        name = os.path.join(folder, file_name)
        pot.to_json(name)


def pot_graph_serializer_from_molecules_to_smiles(graph):
    """
    In the process of serialization, this function is used to convert a pot graph containing molecules
    into one containing smiles
    """
    graph_copy = graph.copy()
    for ind in graph_copy.nodes:
        node = graph_copy.nodes[ind]
        node["molecule"] = node["molecule"].force_smiles()

    return graph_copy


def pot_graph_serializer_from_smiles_to_molecules(graph):
    """
    In the process of serialization, this function is used to convert a pot graph containing smiles
    into one containing molecules.
    """
    graph_copy = graph.copy()
    for ind in graph_copy.nodes:
        node = graph_copy.nodes[ind]
        node["molecule"] = Molecule.from_smiles(node["molecule"])

    return graph_copy


class FatherError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class TooManyIterationError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Pot:
    """
    The pot object is the chemical reactor.
    """

    def __init__(self, root, target=Molecule(), multiplier=1, rxn_name=None):
        multiplied_root = root * multiplier
        self.target = target
        # this needs to be here because it is needed when we calculates yield.
        self.multiplier = multiplier
        self.root = multiplied_root
        self.graph = nx.DiGraph()
        self.run_time = None
        self.rxn_name = rxn_name
        self.graph.add_node(
            0, molecule=multiplied_root, converged=False, root=True
        )  # root=True is for drawing

    def create_name_solvent_multiplier(self, root_string):
        """
        creates an unique file name
        """
        name = f"{root_string}-M{self.multiplier}"
        return name

    @property
    def average_node_degree(self):
        """
        Average number of neighbors a node has
        """
        G = self.graph
        degrees = G.degree()
        sum_of_edges = sum([v for k, v in degrees])
        avg_node_degree = sum_of_edges / len(degrees)
        return avg_node_degree

    @property
    def number_of_nodes(self):
        """returns the number of nodes"""
        return len(self.graph.nodes)

    @property
    def clustering_coefficient(self):
        """
        Something about clusters in the graph
        """
        return nx.average_clustering(self.graph)

    def __str__(self):
        return f"POT {self.root.smiles}"

    def __repr__(self):
        return str(self)

    def draw_reaction_graph(self, size=(800, 800), percentage=0.6, string_mode=False):
        nodes, links = forward_to_d3json(self.graph, self.leaves)
        totalstring = draw_d3(
            nodes, links, string_mode=True, percentage=percentage, size=size
        )

        if string_mode:
            return totalstring
        else:
            return HTML(totalstring)

    def draw(
        self,
        size=(400, 400),
        size_graph=(600, 600),
        string_mode=False,
        percentage=1,
        node_index=False,
        leaves=False,
        just=None,
        mode="rdkit",
    ):
        """
        Draw the POT
        """
        reactantZ = self.root
        if self.target.is_empty():
            target_string = ""
        else:
            target_string = f"""<div style="width: 20%; display: table-cell; border: 1px solid black;">
<p style="text-align: center; font-weight: bold;">Target Molecule</p>
{self.target.draw(string_mode=True, percentage=percentage, size=size, mode=mode)}
</div>"""

        runtime_string = f"- Time: {float(self.run_time):.2} s" if self.run_time else ""

        # totalstring = f'''<h2>POT</h2><h3>Score {self.score:.2f} | Status: {self.status.name}</h3><p>Conditions: {self.conditions.draw_reaction_arrow()}</p>
        totalstring = f"""<h3>Pot name: {self.rxn_name}  </h3>
<div style="width: 70%; display: table;"> <div style="display: table-row;">
<div style="width: 20%; display: table-cell; border: 1px solid black;">
<p style="text-align: center; font-weight: bold;">Pot root</p>
{reactantZ.draw(string_mode=True, percentage=percentage, size=size, mode=mode)}
</div>
<div style="width: 20%; display: table-cell; border: 1px solid black;">
<p style="text-align: center; font-weight: bold;">Pot reaction graph</p>
{self.draw_reaction_graph(size=size_graph, percentage=percentage, string_mode=True)}
</div>
{target_string}
</div></div>"""
        totalstring += (
            self.draw_leaves(string_mode=True, mode=mode,
                             just=just) if leaves else ""
        )
        if string_mode:
            return totalstring
        else:
            return HTML(totalstring)

    @property
    def leaves(self):
        """
        Returns a list of leaf nodes, defined as nodes with 'in degree' equal to zero
        """
        return [x[0] for x in self.graph.in_degree() if x[1] == 0]

    def is_node_converged(self, n: int) -> bool:
        """is this particular node converged?"""
        return self.graph.nodes[n]["converged"]

    def max_depth_of_a_node(self, n: int):
        """gives you the maximum depth of a node"""
        return max(len(x) for x in self.paths_from(n))

    def any_leaves_growable(self) -> bool:
        """Are all leaves in the graph converged?"""
        return any([not self.is_node_converged(x) for x in self.leaves])

    def check_for_number_of_nodes(self, maximum_number_of_nodes: int, t1: float):
        """checks for graph size"""
        if len(self.graph.nodes) > maximum_number_of_nodes:
            self.status = PotStatus.ITERATION
            self.run_time = time() - t1
            raise TooManyIterationError(
                (self.root.smiles),
                f"This pot exceeded the number of max nodes {maximum_number_of_nodes}.",
            )

    def draw_molecules_in_nodes(self, width=80):
        """
        a quick draw to visualize every molecule in the pot graph
        """
        molsz = [self.graph.nodes[x]["molecule"] for x in self.graph.nodes]
        namesz = [
            f'{x}   {self.graph.nodes[x]["molecule"].force_smiles()}'
            for x in self.graph.nodes
        ]
        return Molecule.draw_list(molsz, names=namesz, width=width)

    def unique_smiles(self):
        """
        returns the unique molecules smiles found in the pot
        """
        indexes = [x for x in self.graph.nodes if x != 0]
        a = set()
        for ind in indexes:
            graph_mols = self.graph.nodes[ind]["molecule"]
            for piece in graph_mols.separate_graph_in_pieces():
                a.add(piece.force_smiles())
        return a

    @property
    def reactions_in_the_pot(self):
        """it returs a list of the unique reactions that happened in the pot"""
        return sorted(
            list(set([self.graph.edges[x]["reaction"]
                 for x in self.graph.edges]))
        )

    def find_all_fathers(self, index):
        """
        When you have a node in a pot tree, it will return a list of immediate fathers
        watch out this is the reverse direction as the same identical method in bipartite
        """
        edges = self.graph.edges.data()
        return [x[1] for x in edges if x[0] == index]

    def unique_smiles_in_leaves(self):
        """
        This will give back how many different molecules ar in the final leaves of the pot
        """
        leaveZ = [self.graph.nodes[x]["molecule"] for x in self.leaves]
        smiles_unique = set()
        for x in leaveZ:
            for y in x.separate_graph_in_pieces():
                smiles_unique.add(y.smiles)
        return smiles_unique

    def paths_from(self, node_ind):
        simple_path = nx.all_simple_paths(
            self.graph, source=node_ind, target=0)
        return simple_path

    def subgraph_from(self, source, target=0):
        """
        Returns all the simple subgraphs from source to target
        """
        if source == 0:
            return self.graph.subgraph([0])
        all_path_nodes = set(
            itertools.chain(
                *list(nx.all_simple_paths(self.graph, source=source, target=target))
            )
        )
        return self.graph.subgraph(all_path_nodes)

    def draw_reaction_subgraphs(
        self, source, target=0, size=(800, 800), percentage=0.6, string_mode=False
    ):
        """
        Draws the simple subgraphs from source to target
        """
        H = self.subgraph_from(source, target)
        nodes, links = forward_to_d3json(H, self.leaves)
        totalstring = draw_d3(
            nodes, links, string_mode=True, percentage=percentage, size=size
        )

        if string_mode:
            return totalstring
        else:
            return HTML(totalstring)

    def is_this_molecule_in_pot(self, molecule: Molecule) -> bool:
        """
        we use this function to check if a molecule is in the pot graph
        """
        this_pot_booleans = []
        for node in self.graph.nodes:
            content = self.graph.nodes[node]["molecule"]
            boo = molecule.is_subgraph_isomorphic_to(content)
            this_pot_booleans.append(boo)
        return any(this_pot_booleans)

    def in_which_node_is_this_molecule(self, molecule: Molecule) -> list[int]:
        """
        we use this function to get number of node of where a mol is
        """
        where = []
        for node in self.graph.nodes:
            content = self.graph.nodes[node]["molecule"]
            boo = molecule.is_subgraph_isomorphic_to(content)
            if boo:
                where.append(node)
        return where

    def draw_from_target(self, just=None, mode="rdkit", string_mode=False, width=100):
        how_many_paths = 0
        for ti in self.target_indexes:
            how_many_paths = len(list(self.paths_from(ti)))
            print(
                f'I see {how_many_paths} different path{"" if how_many_paths == 1 else "s"} from node {ti}.'
            )
        return self.draw_from_nodes(
            self.target_indexes,
            just=just,
            mode=mode,
            width=width,
            string_mode=string_mode,
        )

    @property
    def target_indexes(self) -> list[int]:
        if not self.target.is_empty():
            list_of_nodes = self.in_which_node_is_this_molecule(self.target)
            if len(list_of_nodes) == 0:
                raise ValueError(
                    "Target is not present in the reaction network.")
            return list_of_nodes

        else:
            raise ValueError("This pot has been created without a target.")

    def draw_reactions(self, Reactions, string_mode=False):
        """
        it dras the reactions templates that are in the pot
        """
        if string_mode:
            return " ".join(
                [
                    Reactions[x].draw(string_mode=True, size=(400, 400))
                    for x in self.reactions_in_the_pot
                ]
            )
        else:
            return HTML(
                " ".join(
                    [
                        Reactions[x].draw(string_mode=True, size=(400, 400))
                        for x in self.reactions_in_the_pot
                    ]
                )
            )

    def draw_from_nodes(
        self,
        node_list,
        just=None,
        string_mode=False,
        charges=False,
        env=False,
        mode="rdkit",
        columns=5,
        width=100,
        size=(650, 650),
    ):
        """
        This function draws the paths from a list of nodes.
        """
        s = ""
        for node in node_list:
            s += f"<h2>from node -> {node}</h2>"
            s += self.draw_from_node(
                node,
                just=just,
                string_mode=True,
                charges=charges,
                env=env,
                mode=mode,
                columns=columns,
                width=width,
                size=size,
            )
        if string_mode:
            return s
        else:
            return HTML(s)

    def draw_from_node(
        self,
        node,
        just=None,
        string_mode=False,
        charges=False,
        env=False,
        mode="rdkit",
        columns=5,
        width=100,
        size=(650, 650),
        arrows=False,
    ):
        """
        This function draws the paths from nodes.
        """
        # A generator
        paths = []
        if just is not None:
            for counter, path in enumerate(self.paths_from(node)):
                paths.append(path)
                if counter > just:
                    break
        else:
            paths = sorted(list(self.paths_from(node)), key=lambda x: len(x))

        return self.draw_from_node_list_of_lists(
            paths,
            string_mode=string_mode,
            charges=charges,
            env=env,
            mode=mode,
            columns=columns,
            width=width,
            size=size,
            arrows=arrows,
        )

    def draw_shortest_to_node(
        self,
        node,
        just=None,
        string_mode=False,
        charges=False,
        env=False,
        mode="rdkit",
        columns=5,
        width=100,
        size=(650, 650),
        arrows=False,
        weight=None
    ):
        """
        This function draws shortest path to nodes as per networkx `shortest_path` function.
        Specify the distance variable through `weight`
        """
        # A generator
        g = self.graph
        path = nx.shortest_path(g, source=0, target=node, weight=weight)
        path.reverse()
        return self.draw_from_single_path(path=path,
                                          string_mode=string_mode,
                                          charges=charges,
                                          env=env,
                                          mode=mode,
                                          columns=columns,
                                          width=width,
                                          size=size)

    def draw_neighbors_of_node(self, n: int, string_mode=False, width=100):
        graph = self.graph
        parent = [(x, y) for (x, y) in graph.edges if x == n]
        children = [(x, y) for (x, y) in graph.edges if y == n]
        string = ""
        for i, (x, y) in enumerate(parent):
            if i == 0:
                string += f"<h1>Parents of node {n}</h1>"
            reaction = graph.edges[x, y]["reaction"]
            string += Molecule.draw_list(
                [graph.nodes[y]["molecule"], graph.nodes[x]["molecule"]],
                names=[f"{y} -> {x}", reaction],
                string_mode=True,
                width=width,
            )

        for i, (x, y) in enumerate(children):
            if i == 0:
                string += f"<h1>Children of node {n}</h1>"
            reaction = graph.edges[x, y]["reaction"]
            string += Molecule.draw_list(
                [graph.nodes[y]["molecule"], graph.nodes[x]["molecule"]],
                names=[f"{y} -> {x}", reaction],
                string_mode=True,
                width=width,
            )
        if string_mode:
            return string
        else:
            return HTML(string)

    def draw_from_node_list_of_lists(
        self,
        paths_node_list_of_list: list[list[int]],
        string_mode=False,
        charges=False,
        env=False,
        mode="rdkit",
        columns=5,
        width=100,
        size=(650, 650),
        arrows=False,
    ):

        env_molecules = Molecule()

        if paths_node_list_of_list == []:
            path = [0]
            names = ["Root Molecule"]
            molecules_list = list(
                reversed(
                    [self.graph.nodes[x]["molecule"] + env_molecules for x in path]
                )
            )
            stringZ = "<h3>Empty Path</h3>"
            stringZ += Molecule.draw_list(
                molecules_list,
                names=names,
                string_mode=True,
                charges=charges,
                mode=mode,
                size=size,
                width=width,
                columns=columns,
            )
        else:
            stringZ = ""
            for ii, path in enumerate(paths_node_list_of_list):
                # AV: I have no idea what I did here
                pairswiZ = list(pairwise(path))
                names = list(
                    reversed(
                        [
                            f'{self.graph.edges[x]["reaction"]} - {x[0]}'
                            for x in pairswiZ
                        ]
                        + ["Initial_molecule - 0"]
                    )
                )
                molecules_list = list(
                    reversed(
                        [self.graph.nodes[x]["molecule"] +
                            env_molecules for x in path]
                    )
                )
                stringZ += f"<h3>Path to {path[0]} n. {ii}:</h3>"
                stringZ += Molecule.draw_list(
                    molecules_list,
                    names=names,
                    string_mode=True,
                    charges=charges,
                    mode=mode,
                    size=size,
                    width=width,
                    columns=columns,
                    arrows=arrows,
                )
        if string_mode:
            return stringZ
        else:
            return HTML(stringZ)

    def draw_from_single_path(
        self,
        path: list[int],
        string_mode=False,
        charges=False,
        env=False,
        mode="rdkit",
        columns=5,
        width=100,
        size=(650, 650),
    ):
        return self.draw_from_node_list_of_lists(
            [path],
            string_mode=string_mode,
            charges=charges,
            env=env,
            mode=mode,
            columns=columns,
            width=width,
            size=size,
        )

    def draw_neighborhood(self, n, size=(800, 800), percentage=0.6, string_mode=False):
        """
        Draw a neighborhood of the pot graph given a node index
        """

        h = nx.DiGraph()
        neighbors = nx.neighbors(self.graph.to_undirected(), n)

        for neighbor in neighbors:
            h.add_node(neighbor)
            h.add_edge(n, neighbor)

        nodes, links = forward_to_d3json(h, [])
        totalstring = draw_d3(
            nodes, links, string_mode=True, percentage=percentage, size=size
        )

        if string_mode:
            return totalstring
        else:
            return HTML(totalstring)

    def draw_mol_and_subgraph_from_node(self, N, string_mode=False):
        html = f"List of paths: {list(self.paths_from(N))}"
        html += f"""<div style="width: 40%; display: table;"> <div style="display: table-row;"><div style="width: 50%; display: table-cell;">
        <p style="text-align: center; font-weight: bold;">Molecule</p>{self.graph.nodes[N]['molecule'].draw(string_mode=True)}</div>
        <div style="width: 50%; display: table-cell;">
        <p style="text-align: center; font-weight: bold;">Subgraph</p>{self.draw_reaction_subgraphs(N, size=(400,400), percentage=1, string_mode=True)}</div>"""
        if string_mode:
            return html
        else:
            return HTML(html)

    def how_many_paths(self):
        """
        Given a pot, returns how many unique paths it finds
        """
        return sum(len(list(self.paths_from(node))) for node in self.leaves)

    def draw_leaves(
        self,
        mode="d3",
        string_mode=False,
        just=None,
        charges=False,
        columns=8,
        width=50,
        size=(650, 650),
    ):
        """
        This function will draw every possible path going from root to every leaf
        """

        leaves = self.leaves
        string_list = [
            self.draw_from_node(
                x,
                string_mode=True,
                charges=charges,
                mode=mode,
                size=size,
                width=width,
                columns=columns,
                just=just,
            )
            for x in leaves
        ]

        if string_mode:
            return " ".join(string_list)
        else:
            return HTML(" ".join(string_list))

    def edge_view(self):
        return [hf.Edge(y, x) for x, y in self.graph.edges]

    @property
    def score(self):
        """
        this tries to calculate some scoring
        """
        denominator = len(self.leaves) if len(self.leaves) > 0 else 1
        how_many_times = sum(
            self.target.is_subgraph_isomorphic_to(
                self.graph.nodes[x]["molecule"])
            for x in self.leaves
        )
        score = how_many_times / denominator
        return score

    def to_dict(self):
        d = {}
        d["root"] = self.root.smiles

        d["target"] = self.target.smiles
        d["graph"] = nx.node_link_data(
            pot_graph_serializer_from_molecules_to_smiles(self.graph)
        )
        d["status"] = self.status.name
        d["encountered_species"] = self.encountered_species
        d["average_node_degree"] = self.average_node_degree
        d["clustering_coefficient"] = self.clustering_coefficient
        d["multiplier"] = self.multiplier
        d["run_time"] = self.run_time
        d["rxn_name"] = self.rxn_name
        return d

    def to_json(self, fn):
        """
        json serializer
        """
        d = self.to_dict()
        json.dump(d, open(fn, "w"))

    @classmethod
    def from_json(cls, fn):
        read = json.load(open(fn, "r"))
        pot = cls.from_dict(read)
        return pot

    @classmethod
    def from_dict(cls, read):
        pot = cls(Molecule.from_smiles(read["root"]))
        pot.target = Molecule.from_smiles(read["target"])

        try:
            pot.encountered_species = read["encountered_species"]
        except KeyError:
            # pass
            print(
                "ALESSIO PLEASE FIX THIS"
            )  # TODO Take out this try/except when you do not use anymore USPTO jsons without this key
        pot.graph = pot_graph_serializer_from_smiles_to_molecules(
            nx.node_link_graph(read["graph"])
        )
        pot.status = PotStatus[read["status"]]
        pot.multiplier = read["multiplier"]
        pot.run_time = read["run_time"]
        pot.rxn_name = read["rxn_name"]
        return pot
