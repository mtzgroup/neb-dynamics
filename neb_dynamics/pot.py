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

import retropaths.helper_functions as hf
from retropaths.helper_functions import pairwise
from retropaths.molecules.d3_tools import draw_d3, forward_to_d3json
from retropaths.molecules.molecular_formula import MolecularFormula
from retropaths.molecules.molecule import Molecule
from retropaths.molecules.utilities import give_me_free_index, naturals
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.library import Library


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
        tupleZ = (self.NtimesLeaf, self.NtimesTot, self.TotLeaves, self.TotNodes, self.Multiplier)
        return str(tupleZ)

    @classmethod
    def from_tuple_string(cls, string):
        """from the tuple written above, to the object back"""
        a, b, c, d, e = literal_eval(string)
        pms = PotMoleculeSummary(NtimesLeaf=a, NtimesTot=b, TotLeaves=c, TotNodes=d, Multiplier=e)
        return pms


def serialize_single_pot(pot: Pot, file_name: str, folder: Path):
    """This serializes a list of pots into jsons"""
    name = folder / f"{file_name}.json"
    pot.to_json(name)


def serialize_list_of_pots(pots, root_name, folder):
    """This serializes a list of pots into jsons"""
    for pot in pots:
        file_name = f"{root_name}-{pot.conditions.solvent.name}-M{pot.multiplier}.json"
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
        if "environment" in node.keys():
            node["environment"] = node["environment"].force_smiles()
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
        if "environment" in node.keys():
            node["environment"] = Molecule.from_smiles(node["environment"])
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
    def __init__(self, root, target=Molecule(), conditions=Conditions(), environment=Molecule(), multiplier=1, rxn_name=None):
        multiplied_root = root * multiplier
        self.conditions = conditions
        self.environment = environment
        self.target = target
        self.encountered_species = {root.smiles: 0}
        self.status = PotStatus.EMPTY
        self.multiplier = multiplier  # this needs to be here because it is needed when we calculates yield.
        self.root = multiplied_root
        self.graph = nx.DiGraph()
        self.run_time = None
        self.rxn_name = rxn_name
        self.graph.add_node(0, molecule=multiplied_root, environment=environment, converged=False, root=True)  # root=True is for drawing

    def create_name_solvent_multiplier(self, root_string):
        """
        creates an unique file name depending on some of the pot conditions
        """
        name = f"{root_string}-{self.conditions.solvent.name}-M{self.multiplier}"
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

    @classmethod
    def from_components(cls, mol_names, conditions=Conditions(), environment=Molecule(), spectators=Molecule()):
        """
        Creates a pot from components
        """

        mol = Molecule.add_list_of_molecules([Molecule.from_smiles(x) for x in mol_names])

        # Can do this better later
        solv = conditions.solvent.name
        if solv == "water":
            solvent = Molecule.from_smiles("O")
        else:
            solvent = Molecule()
            print("I don't know what solvent to add")

        return cls(mol, conditions=conditions, environment=environment + spectators + solvent)

    def __str__(self):
        return f"POT {self.root.smiles} -> {self.conditions}"

    def __repr__(self):
        return str(self)

    def draw_reaction_graph(self, size=(800, 800), percentage=0.6, string_mode=False):
        nodes, links = forward_to_d3json(self.graph, self.leaves)
        totalstring = draw_d3(nodes, links, string_mode=True, percentage=percentage, size=size)

        if string_mode:
            return totalstring
        else:
            return HTML(totalstring)

    def draw(self, size=(400, 400), size_graph=(600, 600), string_mode=False, percentage=1, node_index=False, leaves=False, just=None, mode="rdkit"):
        """
        Draw the POT
        """
        reactantZ = self.root
        environment_mols = self.environment
        if self.target.is_empty():
            target_string = ""
        else:
            target_string = f"""<div style="width: 20%; display: table-cell; border: 1px solid black;">
<p style="text-align: center; font-weight: bold;">Target Molecule</p>
{self.target.draw(string_mode=True, percentage=percentage, size=size, mode=mode)}
</div>"""

        runtime_string = f"- Time: {float(self.run_time):.2} s" if self.run_time else ""

        # totalstring = f'''<h2>POT</h2><h3>Score {self.score:.2f} | Status: {self.status.name}</h3><p>Conditions: {self.conditions.draw_reaction_arrow()}</p>
        totalstring = f"""<h3>Pot name: {self.rxn_name} | Status: {self.status.name} {runtime_string} </h3><p>{self.conditions}</p>
<div style="width: 70%; display: table;"> <div style="display: table-row;">
<div style="width: 20%; display: table-cell; border: 1px solid black;">
<p style="text-align: center; font-weight: bold;">Pot root</p>
{reactantZ.draw(string_mode=True, percentage=percentage, size=size, mode=mode)}
</div>
<div style="width: 20%; display: table-cell; border: 1px solid black;">
<p style="text-align: center; font-weight: bold;">Pot environment</p>
{environment_mols.draw(string_mode=True, percentage=percentage, size=size, mode=mode)}
</div>
<div style="width: 20%; display: table-cell; border: 1px solid black;">
<p style="text-align: center; font-weight: bold;">Pot reaction graph</p>
{self.draw_reaction_graph(size=size_graph, percentage=percentage, string_mode=True)}
</div>
{target_string}
</div></div>"""
        totalstring += self.draw_leaves(string_mode=True, mode=mode, just=just) if leaves else ""
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
        '''is this particular node converged?'''
        return self.graph.nodes[n]["converged"]

    def max_depth_of_a_node(self, n: int):
        ''' gives you the maximum depth of a node '''
        return max(len(x) for x in self.paths_from(n))

    def any_leaves_growable(self) -> bool:
        '''Are all leaves in the graph converged?'''
        return any([not self.is_node_converged(x) for x in self.leaves])

    def free_index(self):
        '''It gives back the smallest index that is not used in graph'''
        counter = give_me_free_index(naturals(0), self.graph)
        return next(counter)

    def add_node_to_graph(self,
                          leaf: int,
                          name: str,
                          this_smiles: str,
                          result: Molecule,
                          verbosity: int = 0):
        '''
        add a new node connected to leaf with the result molcule
        also updates the encountered_species dictionary
        '''
        free_index = self.free_index()
        self.graph.add_node(free_index, molecule=result, converged=False)
        self.graph.add_edge(free_index, leaf, reaction=name)
        self.encountered_species[this_smiles] = free_index
        if verbosity > 0:
            print(f"\nadding node {free_index} attached with {leaf} with {name}.\n\n\n")

    def grow_this_node(self,
                       leaf: int,
                       library: Library,
                       filter_minor_products: bool = True,
                       use_father_error: bool = False,
                       skip_rules: bool = False,
                       verbosity: int = 0):
        '''This method just grows one node'''
        initial_molecules = self.graph.nodes[leaf]["molecule"]
        initial_graph = initial_molecules + self.environment

        was_this_node_converged = True  # this stays True if no template applies

        for name, reaction in library.items():
            results = reaction.apply_forward(initial_graph, filter_minor_products=filter_minor_products, skip_rules=skip_rules, verbosity=verbosity)
            for result in results:
                for this_minor_reactant in self.environment.separate_graph_in_pieces():
                    if this_minor_reactant.is_subgraph_isomorphic_to(result):
                        result = result.graph_difference(this_minor_reactant)

                if initial_molecules.is_subgraph_isomorphic_to(result):
                    if verbosity > 0:
                        print(f"Careful, the initial molecule graph {result.smiles} was not touched.")
                else:
                    this_smiles = result.smiles_from_multiple_molecules()
                    if this_smiles in self.encountered_species:
                        link_to = self.encountered_species[this_smiles]
                        if verbosity > 1:
                            print(f"Molecule graph {this_smiles}, coming from {leaf} on {name} is giving something already present in {link_to}")
                        is_in_father = any([link_to in x for x in self.paths_from(leaf)])
                        if is_in_father:
                            if use_father_error:
                                # AV Jan 2022: when we have a father link an we do not want the error to raise
                                # (now default behavior) we DO NOT make te edge back to the father.
                                # this is why there is no else branch here.
                                self.status = PotStatus.FATHER
                                raise FatherError((leaf, link_to), f"A node is trying to link to a father node (infinite loop). {leaf} -> {link_to}, reaction -> {name}")
                        else:
                            # if it is not a father, just link it
                            self.graph.add_edge(self.encountered_species[this_smiles], leaf, reaction=name)
                    else:
                        # first time we see this guy in the graph, add node.
                        self.add_node_to_graph(leaf, name, this_smiles, result, verbosity=verbosity)
                        was_this_node_converged = False

        # this needs to be at same indentation as library.items() loop
        self.graph.nodes[leaf]["converged"] = was_this_node_converged

    def run(self,
            unfiltered_library: Library,
            verbosity: int = 0,
            filter_minor_products: bool = True,
            maximum_number_of_nodes: int = 100,
            use_father_error: bool = False,
            max_depth: int = 100000,
            skip_condition_filter: bool = False):
        """
        This function takes care of the pot going forward logic.
        """
        if not skip_condition_filter:
            library = unfiltered_library.filter_compatible(self.conditions)
        else:
            library = unfiltered_library

        t1 = time()
        if verbosity > 2:
            print(f"{library=}")
        layer_counter = 0
        while self.any_leaves_growable() and layer_counter < max_depth:
            for leaf in self.leaves:
                if not self.is_node_converged(leaf):
                    self.grow_this_node(leaf, library, filter_minor_products, use_father_error, verbosity=verbosity)
                self.check_for_number_of_nodes(maximum_number_of_nodes, t1)
            layer_counter += 1

        self.status = PotStatus.FINISHED
        self.run_time = time() - t1

    def check_for_number_of_nodes(self, maximum_number_of_nodes: int, t1: float):
        ''' checks for graph size'''
        if len(self.graph.nodes) > maximum_number_of_nodes:
            self.status = PotStatus.ITERATION
            self.run_time = time() - t1
            raise TooManyIterationError((self.root.smiles), f"This pot exceeded the number of max nodes {maximum_number_of_nodes}.")

    def draw_molecules_in_nodes(self, width=80):
        '''
        a quick draw to visualize every molecule in the pot graph
        '''
        molsz = [self.graph.nodes[x]['molecule'] for x in self.graph.nodes]
        namesz = [f'{x}   {self.graph.nodes[x]["molecule"].force_smiles()}' for x in self.graph.nodes]
        return Molecule.draw_list(molsz, names=namesz, width=width)

    # STOICHIOMETRIC CODE ---- needs to be better implemented.
    def add_environment_to_root(self, how_many_times: int = 1):
        '''
        used to modify the root node and add the environment
        '''
        self.graph.nodes[0]['molecule'] += self.environment * how_many_times
        return self

    def run_stoichiometric(self,
                           library: Library,
                           verbosity: int = 0,
                           maximum_number_of_nodes: int = 100,
                           max_depth: int = 100000,
                           ):
        """
        This function takes care of the pot going forward logic.
        """
        self.add_environment_to_root()
        t1 = time()
        layer_counter = 0
        while self.any_leaves_growable() and layer_counter < max_depth:
            for leaf in self.leaves:
                if not self.is_node_converged(leaf):
                    self.grow_this_node_stoichiometric(leaf, library, verbosity=verbosity)
                self.check_for_number_of_nodes(maximum_number_of_nodes, t1)
            layer_counter += 1

        self.status = PotStatus.FINISHED
        self.run_time = time() - t1

    def grow_this_node_stoichiometric(self,
                                      leaf: int,
                                      library: Library,
                                      verbosity: int = 0,
                                      mf: MolecularFormula = MolecularFormula()
                                      ):
        '''This method just grows one node'''
        initial_graph = self.graph.nodes[leaf]["molecule"]

        was_this_node_converged = True  # this stays True if no template applies

        for name, reaction in library.items():
            results = reaction.apply_forward(initial_graph, filter_minor_products=False, skip_rules=False, verbosity=verbosity)
            for result in results:
                if initial_graph.is_subgraph_isomorphic_to(result):
                    pass  # AV: August 2022 this is actually something we do not need to worry about
                    # raise InitialGraphNotTouchedError(f'\n{self.rxn_name}\n{self.root}\n -> Error in Grow_this_node leaf #{leaf}.\nNothing has been applied.\n')
                else:
                    this_smiles = result.smiles_from_multiple_molecules()
                    if this_smiles in self.encountered_species:
                        link_to = self.encountered_species[this_smiles]
                        if verbosity > 1:
                            print(f"Molecule graph {this_smiles}, coming from {leaf} on {name} is giving something already present in {link_to}")
                        self.graph.add_edge(self.encountered_species[this_smiles], leaf, reaction=name)
                    else:
                        # first time we see this guy in the graph, add node.
                        self.add_node_to_graph(leaf, name, this_smiles, result, verbosity=verbosity)
                        was_this_node_converged = False

        # this needs to be at same indentation as library.items() loop
        self.graph.nodes[leaf]["converged"] = was_this_node_converged
    # END OF STOICHIOMETRIC CODE --------

    def run_with_timeout_and_error_catching(self,
                                            timeout_seconds_pot: int,
                                            library: Library,
                                            name: str = '',
                                            max_depth: int = 10000,
                                            maximum_number_of_nodes: int = 100,
                                            skip_condition_filter: bool = False,
                                            stoichiometric: bool = False
                                            ):
        try:
            with timeout(timeout_seconds_pot, exception=TimeoutPot):
                try:
                    if stoichiometric:

                        self.run_stoichiometric(library,
                                                maximum_number_of_nodes=maximum_number_of_nodes,
                                                max_depth=max_depth,
                                                )
                    else:
                        self.run(library,
                                 maximum_number_of_nodes=maximum_number_of_nodes,
                                 max_depth=max_depth,
                                 skip_condition_filter=skip_condition_filter
                                 )
                except FatherError as e:
                    print(f"Father Error {self.root.smiles} -> {name} \n\n{e.expression} {e.message}")
                except TooManyIterationError as e:
                    print(f"Too many iterations Error {self.root.smiles} -> {name}\n\n{e.expression} {e.message}")
                except Exception:  # noqa E722 I want a bare exception here, to collect all other cases.
                    print(f"another error happened in POT {self.root.smiles} -> {name}")
        except TimeoutPot:
            print(f"{self.root.smiles} -> {name} did not finish in time with {timeout_seconds_pot} seconds")

    def unique_smiles(self):
        '''
        returns the unique molecules smiles found in the pot
        '''
        indexes = [x for x in self.graph.nodes if x != 0]
        a = set()
        for ind in indexes:
            graph_mols = self.graph.nodes[ind]["molecule"]
            for piece in graph_mols.separate_graph_in_pieces():
                a.add(piece.force_smiles())
        return a

    def unique_smiles_in_leaves_with_counting(self):
        """
        This is taking a pot graph and it is creating a list of smiles with a counting tuple.
        [(str, (int,int,int,int))]
        it is basically a smile:
        'CCC' along with a PotMoleculeSummary.
        """
        indexes = [x for x in self.graph.nodes if x != 0]  # we do not want root molecule
        leaves = [x for x in self.leaves if x != 0]  # we do not want root molecule
        how_many_times = {}
        how_many_times_in_leaves = {}
        totnodes = 0
        totleaves = 0

        for ind in indexes:
            graph_mols = self.graph.nodes[ind]["molecule"]
            how_many_fathers = len(self.find_all_fathers(ind))

            # Because of linking the number of nodes in the graph
            # does not equal how many times the molecule was actually produced
            # To calculate yield we need to know how many times a product was actually produced
            totnodes += how_many_fathers
            if ind in leaves:
                totleaves += how_many_fathers

            for piece in graph_mols.separate_graph_in_pieces():
                # this is where neutralization goes
                neutralized_y = piece.neutralize_anions_with_H().smiles
                if neutralized_y in how_many_times:
                    how_many_times[neutralized_y] += how_many_fathers
                else:
                    how_many_times[neutralized_y] = how_many_fathers

                if ind in leaves:
                    if neutralized_y in how_many_times_in_leaves:
                        how_many_times_in_leaves[neutralized_y] += how_many_fathers
                    else:
                        how_many_times_in_leaves[neutralized_y] = how_many_fathers

        results = [(x, PotMoleculeSummary(NtimesLeaf=how_many_times_in_leaves.get(x, 0), NtimesTot=how_many_times[x], TotLeaves=totleaves, TotNodes=totnodes, Multiplier=self.multiplier)) for x in how_many_times.keys()]
        return results

    @property
    def reactions_in_the_pot(self):
        """it returs a list of the unique reactions that happened in the pot"""
        return sorted(list(set([self.graph.edges[x]["reaction"] for x in self.graph.edges])))

    def find_all_fathers(self, index):
        """
        When you have a node in a pot tree, it will return a list of immediate fathers
        watch out this is the reverse direction as the same identical method in bipartite
        """
        edges = self.graph.edges.data()
        return [x[1] for x in edges if x[0] == index]

    @property
    def reaction_molecules_smiles(self):
        """
        returns the smiles that are in the pot environment
        """
        react_mol = ".".join([x.smiles for x in self.environment.separate_graph_in_pieces()])
        return react_mol if react_mol else "NA"

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
        simple_path = nx.all_simple_paths(self.graph, source=node_ind, target=0)
        return simple_path

    def subgraph_from(self, source, target=0):
        """
        Returns all the simple subgraphs from source to target
        """
        if source == 0:
            return self.graph.subgraph([0])
        all_path_nodes = set(itertools.chain(*list(nx.all_simple_paths(self.graph, source=source, target=target))))
        return self.graph.subgraph(all_path_nodes)

    def draw_reaction_subgraphs(self, source, target=0, size=(800, 800), percentage=0.6, string_mode=False):
        """
        Draws the simple subgraphs from source to target
        """
        H = self.subgraph_from(source, target)
        nodes, links = forward_to_d3json(H, self.leaves)
        totalstring = draw_d3(nodes, links, string_mode=True, percentage=percentage, size=size)

        if string_mode:
            return totalstring
        else:
            return HTML(totalstring)

    def is_this_molecule_in_pot(self, molecule: Molecule) -> bool:
        '''
        we use this function to check if a molecule is in the pot graph
        '''
        this_pot_booleans = []
        for node in self.graph.nodes:
            content = self.graph.nodes[node]["molecule"]
            boo = molecule.is_subgraph_isomorphic_to(content)
            this_pot_booleans.append(boo)
        return any(this_pot_booleans)

    def in_which_node_is_this_molecule(self, molecule: Molecule) -> list[int]:
        '''
        we use this function to get number of node of where a mol is
        '''
        where = []
        for node in self.graph.nodes:
            content = self.graph.nodes[node]["molecule"]
            boo = molecule.is_subgraph_isomorphic_to(content)
            if boo:
                where.append(node)
        return where

    def draw_from_target(self, just=None, mode='rdkit', string_mode=False, width=100):
        how_many_paths = 0
        for ti in self.target_indexes:
            how_many_paths = len(list(self.paths_from(ti)))
            print(f'I see {how_many_paths} different path{"" if how_many_paths == 1 else "s"} from node {ti}.')
        return self.draw_from_nodes(self.target_indexes, just=just, mode=mode, width=width, string_mode=string_mode)

    @property
    def target_indexes(self) -> list[int]:
        if not self.target.is_empty():
            list_of_nodes = self.in_which_node_is_this_molecule(self.target)
            if len(list_of_nodes) == 0:
                raise ValueError('Target is not present in the reaction network.')
            return list_of_nodes

        else:
            raise ValueError('This pot has been created without a target.')

    def draw_reactions(self, Reactions, string_mode=False):
        """
        it dras the reactions templates that are in the pot
        """
        if string_mode:
            return " ".join([Reactions[x].draw(string_mode=True, size=(400, 400)) for x in self.reactions_in_the_pot])
        else:
            return HTML(" ".join([Reactions[x].draw(string_mode=True, size=(400, 400)) for x in self.reactions_in_the_pot]))

    def draw_from_nodes(self,
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
        '''
        This function draws the paths from a list of nodes.
        '''
        s = ''
        for node in node_list:
            s += f'<h2>from node -> {node}</h2>'
            s += self.draw_from_node(node,
                                     just=just,
                                     string_mode=True,
                                     charges=charges,
                                     env=env,
                                     mode=mode,
                                     columns=columns,
                                     width=width,
                                     size=size)
        if string_mode:
            return s
        else:
            return HTML(s)

    def draw_from_node(self,
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

        return self.draw_from_node_list_of_lists(paths,
                                                 string_mode=string_mode,
                                                 charges=charges,
                                                 env=env,
                                                 mode=mode,
                                                 columns=columns,
                                                 width=width,
                                                 size=size,
                                                 arrows=arrows,
                                                 )

    def draw_neighbors_of_node(self, n: int, string_mode=False, width=100):
        graph = self.graph
        parent = [(x, y) for (x, y) in graph.edges if x == n]
        children = [(x, y) for (x, y) in graph.edges if y == n]
        string = ''
        for i, (x, y) in enumerate(parent):
            if i == 0:
                string += f'<h1>Parents of node {n}</h1>'
            reaction = graph.edges[x, y]['reaction']
            string += Molecule.draw_list([graph.nodes[y]['molecule'], graph.nodes[x]['molecule']],
                                         names=[f'{y} -> {x}', reaction],
                                         string_mode=True,
                                         width=width)

        for i, (x, y) in enumerate(children):
            if i == 0:
                string += f'<h1>Children of node {n}</h1>'
            reaction = graph.edges[x, y]['reaction']
            string += Molecule.draw_list([graph.nodes[y]['molecule'], graph.nodes[x]['molecule']],
                                         names=[f'{y} -> {x}', reaction],
                                         string_mode=True,
                                         width=width)
        if string_mode:
            return string
        else:
            return HTML(string)

    def draw_from_node_list_of_lists(self,
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

        if env:
            env_molecules = self.environment
        else:
            env_molecules = Molecule()

        if paths_node_list_of_list == []:
            path = [0]
            names = ["Root Molecule"]
            molecules_list = list(reversed([self.graph.nodes[x]["molecule"] + env_molecules for x in path]))
            stringZ = '<h3>Empty Path</h3>'
            stringZ += Molecule.draw_list(molecules_list, names=names, string_mode=True, charges=charges, mode=mode, size=size, width=width, columns=columns)
        else:
            stringZ = ""
            for ii, path in enumerate(paths_node_list_of_list):
                pairswiZ = list(pairwise(path))  # AV: I have no idea what I did here
                names = list(reversed([f'{self.graph.edges[x]["reaction"]} - {x[0]}' for x in pairswiZ] + ["Initial_molecule - 0"]))
                molecules_list = list(reversed([self.graph.nodes[x]["molecule"] + env_molecules for x in path]))
                stringZ += f'<h3>Path to {path[0]} n. {ii}:</h3>'
                stringZ += Molecule.draw_list(molecules_list, names=names, string_mode=True, charges=charges, mode=mode, size=size, width=width, columns=columns, arrows=arrows)
        if string_mode:
            return stringZ
        else:
            return HTML(stringZ)

    def draw_from_single_path(self, path: list[int],
                              string_mode=False,
                              charges=False,
                              env=False,
                              mode="rdkit",
                              columns=5,
                              width=100,
                              size=(650, 650)):
        return self.draw_from_node_list_of_lists([path],
                                                 string_mode=string_mode,
                                                 charges=charges,
                                                 env=env,
                                                 mode=mode,
                                                 columns=columns,
                                                 width=width,
                                                 size=size)

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
        totalstring = draw_d3(nodes, links, string_mode=True, percentage=percentage, size=size)

        if string_mode:
            return totalstring
        else:
            return HTML(totalstring)

    def draw_mol_and_subgraph_from_node(self, N, string_mode=False):
        html = f'List of paths: {list(self.paths_from(N))}'
        html += f'''<div style="width: 40%; display: table;"> <div style="display: table-row;"><div style="width: 50%; display: table-cell;">
        <p style="text-align: center; font-weight: bold;">Molecule</p>{self.graph.nodes[N]['molecule'].draw(string_mode=True)}</div>
        <div style="width: 50%; display: table-cell;">
        <p style="text-align: center; font-weight: bold;">Subgraph</p>{self.draw_reaction_subgraphs(N, size=(400,400), percentage=1, string_mode=True)}</div>'''
        if string_mode:
            return html
        else:
            return HTML(html)

    def how_many_paths(self):
        """
        Given a pot, returns how many unique paths it finds
        """
        return sum(len(list(self.paths_from(node))) for node in self.leaves)

    def draw_leaves(self, mode="d3", string_mode=False, just=None, charges=False, columns=8, width=50, size=(650, 650)):
        """
        This function will draw every possible path going from root to every leaf
        """

        leaves = self.leaves
        string_list = [self.draw_from_node(x, string_mode=True, charges=charges, mode=mode, size=size, width=width, columns=columns, just=just) for x in leaves]

        if string_mode:
            return " ".join(string_list)
        else:
            return HTML(" ".join(string_list))

    def edge_view(self):
        return [hf.Edge(y, x) for x, y in self.graph.edges]

    def draw_step(self, edge: hf.Edge, library, string_mode=False, stoichiometric=False):
        if edge is None:
            print('This seems an empty pot. No edges here...')
            return ''
        n1 = edge.n1
        n2 = edge.n2
        try:
            rxn_name = self.graph.edges[n2, n1]['reaction']
        except KeyError:
            html = f'<h2>No edge from {n1} to {n2}.</h2><h3>You need to give me a correct edge</h3>'
            return html if string_mode else HTML(html)

        mol1 = self.graph.nodes[n1]['molecule']
        mol2 = self.graph.nodes[n2]['molecule']
        rxn = library[rxn_name]
        env = self.environment

        if not stoichiometric:
            mol3 = self.environment.graph_difference(rxn.minor_reactants) + rxn.minor_products
            me_string = f'''<div style="width: 16.5%; display: table-cell; border: 1px solid black;">
        <p style="text-align: center; font-weight: bold;">Molecule + Environment</p>{(mol1 + env).draw(string_mode=True)}</div>'''

            bc_string = f'''<div style="width: 16.5%; display: table-cell; border: 1px solid black;">
        <p style="text-align: center; font-weight: bold;">Before Cleaning</p>{(mol2 + mol3).draw(string_mode=True)}</div>'''
        else:
            me_string = ''
            bc_string = ''

        html = f'<h2>{rxn_name}</h2>'
        html += f'''<div style="width: 100%; display: table;"> <div style="display: table-row;">
        <div style="width: 16.5%; display: table-cell; border: 1px solid black;">
        <p style="text-align: center; font-weight: bold;">Node {n1}</p>{mol1.draw(string_mode=True)}</div>
        {me_string}
        <div style="width: 16.5%; display: table-cell; border: 1px solid black;">
        <p style="text-align: center; font-weight: bold;">Template Reactant</p>{rxn.reactants.draw(string_mode=True)}</div>
        <div style="width: 16.5%; display: table-cell; border: 1px solid black;">
        <p style="text-align: center; font-weight: bold;">Template Product</p>{rxn.products.draw(string_mode=True)}</div>
        {bc_string}
        <div style="width: 16.5%; display: table-cell; border: 1px solid black;">
        <p style="text-align: center; font-weight: bold;">Node {n2}</p>{mol2.draw(string_mode=True)}</div>
        '''
        return html if string_mode else HTML(html)

    @property
    def score(self):
        """
        this tries to calculate some scoring
        """
        denominator = len(self.leaves) if len(self.leaves) > 0 else 1
        how_many_times = sum(self.target.is_subgraph_isomorphic_to(self.graph.nodes[x]["molecule"]) for x in self.leaves)
        score = how_many_times / denominator
        return score

    def to_dict(self):
        d = {}
        d["root"] = self.root.smiles
        d["conditions"] = self.conditions.json()
        d["environment"] = self.environment.smiles
        d["target"] = self.target.smiles
        d["graph"] = nx.node_link_data(pot_graph_serializer_from_molecules_to_smiles(self.graph))
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
        pot.conditions = Conditions().parse_raw(read["conditions"])
        pot.environment = Molecule.from_smiles(read["environment"])
        try:
            pot.encountered_species = read['encountered_species']
        except KeyError:
            # pass
            print('ALESSIO PLEASE FIX THIS')  # TODO Take out this try/except when you do not use anymore USPTO jsons without this key
        pot.graph = pot_graph_serializer_from_smiles_to_molecules(nx.node_link_graph(read["graph"]))
        pot.status = PotStatus[read["status"]]
        pot.multiplier = read["multiplier"]
        pot.run_time = read["run_time"]
        pot.rxn_name = read["rxn_name"]
        return pot
