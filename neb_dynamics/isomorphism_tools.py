import networkx as nx
from timeout_timer import timeout, TimeoutInterrupt


class TimeoutIsomorphism(TimeoutInterrupt):
    pass


class SubGraphMatcher:
    """
    This class is the normal matcher, used by molecules that do not have R groups.
    This is also used by templates with R removed.
    Neighbors and charges are used towards isomorphism
    ISMAGS ON
    """

    GM = nx.isomorphism.ISMAGS

    def __init__(self, mol, verbosity=0, timeout_seconds=10):
        """
        Initialise this graph matcher with a particular mol
        """
        self.mol = mol
        self.verbosity = verbosity
        self.timeout_seconds = timeout_seconds

    def _node_matcher(self, n1, n2):
        nodes_equiv = (
            n1["element"] == n2["element"]
            and n1["neighbors"] == n2["neighbors"]
            and n1["charge"] == n2["charge"]
        )
        return nodes_equiv

    def _node_matcher_no_charge(self, n1, n2):
        nodes_equiv = (
            n1["element"] == n2["element"] and n1["neighbors"] == n2["neighbors"]
        )
        return nodes_equiv

    def _edge_matcher(self, e1, e2):
        edge_equiv = e1["bond_order"] == e2["bond_order"]
        return edge_equiv

    def _edge_match_by_existence(self, e1, e2):
        """
        will only check whether a bond exists in both cases.
        Adding it cause aromatics might be a double or a single
        but there is still a bond present
        """
        edge_equiv = bool(e1["bond_order"] and e2["bond_order"])
        return edge_equiv

    def is_isomorphic(self, g):
        """
        returns a boolean to see if it's isomorphic.
        """
        try:
            with timeout(self.timeout_seconds, exception=TimeoutIsomorphism):
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher,
                    edge_match=self._edge_matcher,
                )
                result = GM.is_isomorphic()
        except TimeoutIsomorphism:
            print(
                f"A SubGraphMatcher timeout error occurred in is_isomorphic {self.mol.force_smiles()} -> {g.force_smiles()}."
            )
            result = False
        return result

    def is_bond_isomorphic(self, g):
        """
        returns a boolean to see if it's isomorphic in connectivity
        """
        try:
            with timeout(self.timeout_seconds, exception=TimeoutIsomorphism):
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher_no_charge,
                    edge_match=self._edge_match_by_existence,
                )
                result = GM.is_isomorphic()
        except TimeoutIsomorphism:
            print(
                f"A SubGraphMatcher timeout error occurred in is_isomorphic {self.mol.force_smiles()} -> {g.force_smiles()}."
            )
            result = False
        return result

    def get_isomorphisms(self, g):
        """
        Returns a list of dictionaries of node mappings.
        The keys of each dictionary are nodes in self.mol, while values are nodes in g.
        """
        try:
            with timeout(self.timeout_seconds, exception=TimeoutIsomorphism):
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher,
                    edge_match=self._edge_matcher,
                )
                isos = [a for a in GM.isomorphisms_iter()]
        except TimeoutIsomorphism:
            print(
                f"A SubGraphMatcher timeout error occurred in get_isomorphisms {self.mol.force_smiles()} -> {g.force_smiles()}."
            )
            isos = []
        return isos

    def is_subgraph_isomorphic(self, g):
        """
        Returns a boolean if self is subgraph isomorphic of g
        """
        try:
            with timeout(self.timeout_seconds, exception=TimeoutIsomorphism):
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher,
                    edge_match=self._edge_matcher,
                )
                result = GM.subgraph_is_isomorphic()
        except TimeoutIsomorphism:
            print(
                f"A SubGraphMatcher timeout error occurred in is_subgraph_isomorphic {self.mol.force_smiles()} -> {g.force_smiles()}."
            )
            result = False
        return result

    def get_subgraph_isomorphisms(self, g):
        """
        Returns a list of dictionaries of node mappings.
        The keys of each dictionary are nodes in self.mol, while values are nodes in g.
        """
        try:
            with timeout(self.timeout_seconds, exception=TimeoutIsomorphism):
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher,
                    edge_match=self._edge_matcher,
                )
                isos = [a for a in GM.subgraph_isomorphisms_iter()]
        except TimeoutIsomorphism:
            print(
                f"A SubGraphMatcher timeout error occurred in get_subgraph_isomorphisms {self.mol.force_smiles()} -> {g.force_smiles()}."
            )
            isos = []
        return isos

    def get_bond_subgraph_isomorphisms(self, g):
        """
        Returns a list of dictionaries of node mappings.
        The keys of each dictionary are nodes in self.mol, while values are nodes in g.
        """
        try:
            with timeout(self.timeout_seconds, exception=TimeoutIsomorphism):
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher_no_charge,
                    edge_match=self._edge_matcher,
                )
                isos = [a for a in GM.subgraph_isomorphisms_iter()]
        except TimeoutIsomorphism:
            print(
                f"A SubGraphMatcher timeout error occurred in get_subgraph_isomorphisms {self.mol.force_smiles()} -> {g.force_smiles()}."
            )
            isos = []
        return isos

    def largest_common_subgraph(self, g):
        """
        This returns the ISMAGS largest common subgraph.
        """
        GM = self.GM(
            self.mol, g, node_match=self._node_matcher, edge_match=self._edge_matcher
        )
        return list(GM.largest_common_subgraph())


class SubGraphMatcherApplyPermute(SubGraphMatcher):
    """
    This matcher is used for apply_permute
    NO ismags
    NO neighbors
    R are wildcards
    """

    GM = nx.isomorphism.GraphMatcher

    def _node_matcher(self, n1, n2):
        nodes_equiv = n1["element"] == n2["element"]

        if n1["element"][0] == "R" or n2["element"][0] == "R":
            nodes_equiv = True

        return nodes_equiv


class SubGraphMatcherRsAreEqualToEachOther(SubGraphMatcher):
    """
    This matcher is used for template to template comparison
    NO ismags
    NO neighbors
    R match true only on other Rs
    """

    GM = nx.isomorphism.GraphMatcher

    def _node_matcher(self, n1, n2):
        nodes_equiv = n1["element"] == n2["element"]

        if n1["element"][0] == "R" and n2["element"][0] == "R":
            nodes_equiv = True

        return nodes_equiv


class SubGraphMatcherRules(SubGraphMatcher):
    """
    This matcher is used by the Rules
    ISMAGS OFF
    Neighbors are not used.
    """

    GM = nx.isomorphism.GraphMatcher

    def _node_matcher(self, n1, n2):
        if n2["element"] == "A":
            nodes_equiv = n1["element"] in n2["element_matching"]
        #             print(f'is {n1["element"]} in {n2["element_matching"]}? {nodes_equiv}')
        else:
            nodes_equiv = n1["element"] == n2["element"]
        #             print(f'is {n1["element"]} equal to {n2["element"]}? {nodes_equiv}')
        return nodes_equiv
