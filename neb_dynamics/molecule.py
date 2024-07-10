"""This is the main file of the molecule object"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from cairosvg import svg2png
from IPython.core.display import HTML
from IPython.display import SVG

# from openeye import oechem, oedepict
from neb_dynamics.isomorphism_tools import SubGraphMatcher
from neb_dynamics.rdkit_draw import moldrawsvg

from neb_dynamics.d3_tools import draw_d3, molecule_to_d3json
from neb_dynamics.helper_functions import (
    graph_to_smiles, from_number_to_element, bond_ord_number_to_string, give_me_free_index, naturals)
from neb_dynamics.isomorphism_mapping import IsomorphismMappings

from rdkit import Chem


class Molecule(nx.Graph):
    """
    Basic class for manipulating molecules as graphs.
    Molecules are undirected graphs with node attribute 'element', being the atom type and sometimes a unique integer id.
    """

    def __init__(self, name: str = "", smi: str = ""):
        """
        Constructs a molecule as a subclass of networkx Graph type

        Args:
            name (list): a name for the molecule
            smi (list): stores the smiles so it does not need to be recalculated

        Returns:
            molecule (Molecule): The Molecule Object
        """

        self.chemical_name = name
        self._smiles = smi
        super(Molecule, self).__init__()

    def __str__(self):
        """
        This just prints out the debug string for a networkx graph
        """
        edge_data = self.edges.data()
        return f"nodes = {self.nodes.data()}\nedges = {edge_data}\n"

    def __repr__(self):
        rep = None
        if self.chemical_name != "":
            rep = self.chemical_name
        elif self._smiles != "":
            rep = self.smiles
        else:
            rep = self.__str__()
        return rep

    def indices_are_identical(self, b: Molecule):
        res = self.remove_Hs().get_subgraph_isomorphisms_of(b.remove_Hs())
        if len(res) > 1:
            # print("Warning! Multiple isomorphisms found.")
            return True
        elif len(res) == 0:
            # print("No isomorphisms found")
            return False

        for k, v in res[0].reverse_mapping.items():
            if k != v:
                # print(f"{k} in predicted is different to {v}")
                return False
        return True

    def copy(self, reindex=False, start=0) -> Molecule:
        """Overload the nx.Graph.copy() method to bring the non-networkx attributes across

        Args:
            reindex (bool, optional): convert the node labels to integers starting with start. Defaults to False.
            start (int, optional): initial value for the reindexing. Defaults to 0.

        Returns:
            Molecule: retropaths molecule object
        """
        if reindex:
            # if reindex is true, convert the node labels to integers starting with start
            new_mol = nx.convert_node_labels_to_integers(
                self, first_label=start)
        else:
            new_mol = super().copy()
        new_mol._smiles = self._smiles
        new_mol.chemical_name = self.chemical_name

        return new_mol

    def graph_difference(self, subtracted_molecule: Molecule) -> Molecule:
        """This will remove all subgraphs from self matching subtracted molecule

        Args:
            subtracted_molecule (Molecule): a molecule graph

        Returns:
            Molecule: the self molecule graph without subtracted_molecule
        """
        target = self.copy()
        isomorphisms = subtracted_molecule.get_subgraph_isomorphisms_of(target)

        assert (
            len(isomorphisms) > 0
        ), f"The graph difference between {self.force_smiles()} and {subtracted_molecule.force_smiles()} needs a second look."

        this_isomorphism = isomorphisms[0]
        this_iso_remaining_fragments = target.copy()

        # remove the nodes
        for _, value in this_isomorphism.reverse_mapping.items():
            this_iso_remaining_fragments.remove_node(value)

        return this_iso_remaining_fragments

    def graph_difference_with_list_and_duplicates(
        self, mols: list[Molecule]
    ) -> Molecule:
        """
        it is a graph difference between a graph and a list of molecules
        Every instance of each molecule in the list is removed from the self
        molecule.
        """
        new_mol = self.copy()
        for mol in mols:
            while mol.is_subgraph_isomorphic_to(new_mol):
                new_mol = new_mol.graph_difference(mol)
        return new_mol

    def __add__(self, mol2: Molecule) -> Molecule:
        """when you do: mol1 + mol2
        it creates a single graph with both molecules

        **NOTE**: atoms will be renumbered by this operation !

        Args:
            mol2 (Molecule): a molecule

        Returns:
            Molecule: a molecule that is the sum of self and mol2
        """

        mol1 = self

        n1 = len(mol1)
        n2 = len(mol2)
        # create a mapping between nodes in mol1 and nodes in mol2,
        mapping1 = dict(zip(mol1.nodes(), range(0, n1)))
        mapping2 = dict(zip(mol2.nodes(), range(n1, n1 + n2)))

        # relabel nodes to match the mapping dictionary
        mol1 = nx.relabel_nodes(mol1, mapping1)
        mol2 = nx.relabel_nodes(mol2, mapping2)
        new_mol = nx.compose(mol1, mol2)

        return new_mol

    def __sub__(self, mol2: Molecule) -> Molecule:
        return self.graph_difference(mol2)

    def __eq__(self, other_molecule) -> bool:
        """Equality without hydrogens is waaay faster. Also, hydrogens are always a terminal.
        But the problem comes when H-H or H+ get to do the isomorphism. In this case I do NOT want to collapse them.

        Args:
            other_molecule: a molecule object

        Returns:
            bool: returns if mols are equals (isomorphic).
        """
        if not isinstance(other_molecule, Molecule):
            raise NotImplementedError

        if len(self) < 3 or len(other_molecule) < 3:
            first = self
            second = other_molecule
        else:
            first = self.remove_Hs()
            second = other_molecule.remove_Hs()
        GM = SubGraphMatcher(first)
        is_this_equal = GM.is_isomorphic(second)
        return is_this_equal

    def __mul__(self, other: int) -> Molecule:
        """In case one wants more of the same molecule

        Args:
            other (int): how many molecules

        Returns:
            Molecule: a retropaths molecule
        """
        assert isinstance(other, int)
        final = self.__class__()
        for _ in range(other):
            final += self
        return final

    def draw(
        self,
        mode="rdkit",
        size=None,
        string_mode=False,
        node_index=True,
        percentage=None,
        force=None,
        fixed_bond_length=None,
        fixedScale=None,
        fontSize=None,
        lineWidth=None,
        charges=True,
        neighbors=False,
    ):
        """
        Draw the graph, mode can be 'd3' for interactive force directed graphs or 'rdkit' or 'oe' for chemdraw style images
        """

        if mode == "d3":
            size = size or (500, 500)
            G = self.copy()
            # G = nx.convert_node_labels_to_integers(G)
            nodes, links = molecule_to_d3json(
                G, node_index, charges=charges, neighbors=neighbors
            )
            return draw_d3(
                nodes,
                links,
                size=size,
                string_mode=string_mode,
                percentage=percentage,
                force_layout_charge=force,
            )

        elif mode == "rdkit":
            size = size or (300, 300)
            d = {
                "Sg": "R1",
                "Rf": "R2",
                "Ne": "R3",
                "Ar": "R4",
                "Kr": "R5",
                "Ru": "R6",
                "Rn": "R7",
                "Og": "R8",
                "Fr": "R9",
                "At": "R10",
                "Db": "R11",
                "Hs": "R12",
                "Bh": "R13",
                "Mt": "R14",
                "Rg": "R15",
                "Cn": "R16",
                "Hf": "R17",
                "U": "R18",
                "W": "R19",
                "Pu": "R20",
                "Am": "R21",
                "Cm": "R22",
            }
            smiles = self.force_smiles()
            svg_str = moldrawsvg(
                smiles,
                d,
                molSize=size,
                fixed_bond_length=fixed_bond_length,
                fixedScale=fixedScale,
                fontSize=fontSize,
                lineWidth=lineWidth,
            )
            if string_mode:
                return svg_str
            else:
                return SVG(svg_str)

            # elif mode == "oe":
            #     width, height = 400, 400

            #     mol = oechem.OEGraphMol()
            #     oechem.OESmilesToMol(mol, self.smiles)
            #     oedepict.OEPrepareDepiction(mol)

            #     opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)
            #     opts.SetMargins(10)
            #     disp = oedepict.OE2DMolDisplay(mol, opts)

            #     font = oedepict.OEFont(oedepict.OEFontFamily_Default, oedepict.OEFontStyle_Default, 12,
            #                            oedepict.OEAlignment_Center, oechem.OEDarkRed)

            #     for adisp in disp.GetAtomDisplays():
            #         atom = adisp.GetAtom()
            #         toggletext = f"{atom.GetIdx()}"
            #         oedepict.OEDrawSVGToggleText(disp, adisp, toggletext, font)

            #     ofs = oechem.oeosstream()
            #     oedepict.OERenderMolecule(ofs, "svg", disp)
            #     string = ofs.str()

            #     sss = f'<div style="width: {100}%; display: table;"> <div style="display: table-row;">'
            #     sss += f'{string.decode()}</div></div>'
            if string_mode:
                return sss
            else:
                return HTML(sss)
        else:
            raise ValueError(
                f'mode must be one of "oe", "d3" or "rdkit", received {mode}'
            )

    def to_svg(self, folder=Path("."), file_name=None):
        if file_name is None:
            file_name = f"{self.force_smiles()}.svg"
        full_path = folder / file_name
        with open(full_path, "w") as f:
            f.write(self.draw(mode="rdkit", string_mode=True))

    def to_png(self, folder=Path("."), file_name=None):
        if file_name is None:
            file_name = f"{self.force_smiles()}.png"
        full_path = folder / file_name
        full_path = str(full_path)
        svg_code = self.draw(mode="rdkit", string_mode=True)
        svg2png(bytestring=svg_code, write_to=full_path)

    @classmethod
    def from_rdmol(cls, rdmol, smi, name=None):
        new_mol = cls(name=name, smi=smi)
        assert isinstance(
            rdmol, Chem.rdchem.Mol), "rdmol must be Rdkit molecule"

        # atom_list = [(x.GetTotalNumHs(), x.GetAtomicNum()) for x in rdmol.GetAtoms()]
        atom_list = [(atom.GetAtomicNum(), atom.GetFormalCharge(),
                      atom.GetTotalNumHs()) for atom in rdmol.GetAtoms()]
        edge_list = [(x.GetEndAtomIdx(), x.GetBeginAtomIdx(),
                      x.GetBondTypeAsDouble()) for x in rdmol.GetBonds()]
        [new_mol.add_node(i, neighbors=0, element=from_number_to_element(
            x), charge=y) for i, (x, y, _) in enumerate(atom_list)]
        [new_mol.add_edge(i, j, bond_order=bond_ord_number_to_string(k))
         for i, j, k in edge_list]

        # # now adding the hydrogens
        non_hydrogen_atoms = len(new_mol)
        indexes_of_hydrogens = non_hydrogen_atoms

        for i in range(non_hydrogen_atoms):
            _, _, n_hs = atom_list[i]
            j = 0
            while j < n_hs:
                new_mol.add_node(indexes_of_hydrogens,
                                 neighbors=0, element='H', charge=0)
                new_mol.add_edge(indexes_of_hydrogens, i, bond_order='single')
                indexes_of_hydrogens += 1
                j += 1
        # the neighbors is a number set to have a better isomorphism.
        new_mol.set_neighbors()

        # Need to create smiles to canonicalize
        new_mol._smiles = new_mol.create_smiles()

        return new_mol

    @staticmethod
    def draw_list_smiles(smis, **kwargs):
        mols = [Molecule.from_smiles(smi) for smi in smis]
        return Molecule.draw_list(mols, names=smis, **kwargs)

    @staticmethod
    def draw_list(
        molecule_list,
        names=[],
        mode="rdkit",
        title="",
        charges=False,
        size=(650, 650),
        width=100,
        columns=5,
        string_mode=False,
        node_index=True,
        neighbors=False,
        arrows=False,
        borders=False,
    ):
        """
        Draws a list of molecules
        """
        if len(molecule_list) == 0:
            print("This list is empty")
            molecule_list.append(Molecule())
            names.append("")

        true_mol_len = len(molecule_list)
        while len(molecule_list) < columns:
            molecule_list.append(Molecule())
            names.append("")

        if columns <= len(molecule_list):
            how_many_columns = columns
        else:
            how_many_columns = len(molecule_list)

        cell_width = 100.0 / how_many_columns

        borders_string = "border: 1px solid black;" if borders else ""

        sstring = f'<h2>{title}</h2><div style="width: {width}%; display: table;"> <div style="display: table-row;">'

        for i, mol in enumerate(molecule_list):
            if i % how_many_columns == 0:
                sstring += '</div><div style="display: table-row;">'

            try:
                name = f'<p style="text-align: center;">{names[i]}</p>'
            except IndexError:
                name = ""
            this_border_string = (
                borders_string if not mol.is_empty() else ""
            )  # I do not want to draw border on empty molecules.
            sstring += f'<div style="width: {cell_width}%; display: table-cell;{this_border_string}"> \
                {mol.draw(mode=mode, string_mode=True, size=size, charges=charges, neighbors=neighbors, node_index=node_index,percentage=0.8)} {name} </div>'
            if arrows and i < true_mol_len - 1:
                sstring += '<div style="width: 0%; display: table-cell; vertical-align: middle;"><font size="+2">⟶</font></div>'

        sstring += "</div></div>"
        if string_mode:
            return sstring
        else:
            return HTML(sstring)

    def renumber_indexes(self, swaps):
        """
        Takes a molecule and a mapping {2:3, 3:2, 5:4, 4:6, 6:5}
        and returns the new molecule that has the VALUES of the mapping where the KEYS were
        """
        mol2 = nx.relabel_nodes(self, swaps)
        return mol2

    def change_element_name(self, lab1, lab2):
        """
        label1 and label2 -> str
        this change the 'element' name
        """
        mol2 = self.copy()
        for node in mol2.nodes():
            label = mol2.nodes[node]["element"]
            if label == lab1:
                mol2.nodes[node]["element"] = lab2
        return mol2

    def get_bond_order(self, first_atom, second_atom):
        """get the bond orde between two indexes"""
        return self.edges[first_atom, second_atom]["bond_order"]

    def get_element(self, index_atom):
        """returns the element at one index"""
        return self.nodes[index_atom]["element"]

    def is_isomorphic_to(self, mol):
        GM = SubGraphMatcher(self)
        return GM.is_isomorphic(mol)

    def is_bond_isomorphic_to(self, mol):
        GM = SubGraphMatcher(self)
        return GM.is_bond_isomorphic(mol)

    def largest_common_subgraph(self, mol):
        GM = SubGraphMatcher(self)
        return GM.largest_common_subgraph(mol)

    def is_subgraph_isomorphic_to(self, mol, timeout_seconds=10):
        """
        Compares self to g and returns True if self is isomorphic to a subgraph of g
        a.is_subgraph_isomorphic_to(b)
        means that a is a subgraph of b
        """
        GM = SubGraphMatcher(mol, timeout_seconds=timeout_seconds)
        boolean = GM.is_subgraph_isomorphic(self.remove_r_groups())
        return boolean

    def get_subgraph_isomorphisms_of(
        self, target, verbosity=0
    ) -> list[IsomorphismMappings]:
        """
        a.get_subgraph_isomorphisms_of(b)
        gives the isomorphic map of A being a subgraph of B
        self.get_subgraph_isomorphisms_of(target)
        tells if the template molecule SELF is a subgraph of TARGET molecule
        """
        GM = SubGraphMatcher(target, verbosity=verbosity)
        isoms = GM.get_subgraph_isomorphisms(self.remove_r_groups())
        return [IsomorphismMappings(x) for x in isoms]

    def get_bond_subgraph_isomorphisms_of(
        self, target, verbosity=0
    ) -> list[IsomorphismMappings]:
        """
        a.get_subgraph_isomorphisms_of(b)
        gives the isomorphic map of A being a subgraph of B
        self.get_subgraph_isomorphisms_of(target)
        tells if the template molecule SELF is a subgraph of TARGET molecule
        """
        GM = SubGraphMatcher(target, verbosity=verbosity)
        isoms = GM.get_bond_subgraph_isomorphisms(self.remove_r_groups())
        return [IsomorphismMappings(x) for x in isoms]

    def create_smiles(self):
        smiles = ".".join(
            sorted([graph_to_smiles(x)
                   for x in self.separate_graph_in_pieces()])
        )
        return smiles

    def which_atoms_are_in(self):
        """returns a list of unique elements"""
        return {self.nodes[x]["element"] for x in self.nodes}

    def list_of_elements(self):
        """returns a list of elements"""
        return [self.nodes[x]["element"] for x in self.nodes]

    def is_empty(self):
        """method to check if the molecule graph is empty or not"""
        return len(self.nodes) == 0

    def separate_graph_in_pieces(self) -> list[Molecule]:
        """
        this function returns a list of single connected graphs.
        """
        return list((self.subgraph(x).copy() for x in nx.connected_components(self)))

    def remove_Hs(self):
        """
        Remove Hydrogens
        """
        # make a copy of this molecule graph
        mol = self.copy()
        r_groups = {k: v for k, v in self.atom_types.items() if "H" in v}
        for k in r_groups.keys():
            mol.remove_node(k)
        return mol

    def add_Hs(self):
        """
        add Hydrogens back after removing them
        """
        mol = self.copy()
        atoms_to_fix = []
        all_hs_to_add = []
        for node_ind in mol.nodes:
            hs_to_add = mol.nodes[node_ind]["neighbors"] - len(
                mol.get_neighbors_of_node(node_ind)
            )
            if hs_to_add > 0:
                atoms_to_fix.append(node_ind)
                all_hs_to_add.append(hs_to_add)

        fixed_mol = mol.add_hydrogens(atoms_to_fix, all_hs_to_add)
        return fixed_mol

    def save_smiles(self, path=None):
        if path is None:
            raise ValueError("Must provide a path")

        with open(path, "w") as f:
            f.write(self.smiles)

    def smiles_from_multiple_molecules(mol):
        """
        this method is used to have a unique smile for each graph
        even when the graph contains multiple molecules.
        It is used in the pot to have uniqueness.
        """
        list_smiles = [x.force_smiles()
                       for x in mol.separate_graph_in_pieces()]
        return ".".join(sorted(list_smiles))

    def force_smiles(self):
        """
        in case I really wat to recalculate the smiles
        """
        return self.create_smiles()

    def get_neighbors_of_node(self, ind):
        """
        get neighbors of molecule atom with index ind
        """
        return list(nx.neighbors(self, ind))

    @property
    def atom_types(self):
        return nx.get_node_attributes(self, "element")

    @property
    def neighbors_number(self):
        return nx.get_node_attributes(self, "neighbors")

    @property
    def num_heavy_atoms(self):
        heavy = [x for x in self.atom_types.values() if x != "H"]
        return len(heavy)

    @property
    def bond_order(self):
        return nx.get_edge_attributes(self, "bond_order")

    @property
    def smiles(self):
        """
        Return a SMILES representation, tries to avoid recomputing the smiles string
        """
        if self._smiles == "":
            try:
                self._smiles = self.create_smiles()
            except Exception as e:
                print("Error making smiles:")
                print(e)
                self._smiles = ""
        return self._smiles

    def set_neighbors(self):
        """
        given a molecule, this function will set the neighbors attribute
        """
        for node in self.nodes:
            self.nodes[node]["neighbors"] = len(list(self.neighbors(node)))
        return self

    @property
    def charge(self):
        return sum(self.nodes[x]["charge"] for x in self.nodes)

    def add_hydrogen(self, index):
        new_mol = self.copy()
        counter = new_mol.give_me_free_index()
        new_index = next(counter)
        new_mol.add_node(new_index, neighbors=0, element="H", charge=0)
        new_mol.add_edge(index, new_index, bond_order="single")
        return new_mol

    def add_hydrogens(self, indexes, how_many):
        new_mol = self.copy()
        for index, num in zip(indexes, how_many):
            for _ in range(num):
                new_mol = new_mol.add_hydrogen(index)
        return new_mol

    def give_me_free_index(self):
        return give_me_free_index(naturals(0), self)

    @classmethod
    def get_ind_mapped_smi(cls, smi: str):
        """
        mapped smiles are 1-indexed, but RP is 0-indexed, so
        the value (i.e. 'v') for the indices needs to be 1 less
        than what the string says
        """
        raw = smi.split("]")[:-1]
        if len(raw) == 1:
            raw = smi
        inds = [int(r.split(":")[-1])-1 for r in raw]
        return inds

    @classmethod
    def get_smi_mapping(cls, smi, mol):

        inds = cls.get_ind_mapped_smi(smi)
        m = {}
        mol_inds = [n for n in mol.nodes]
        mapping_list = zip(mol_inds, inds)
        for k, v in mapping_list:
            m[k] = v
        return m

    @classmethod
    def from_mapped_smiles(cls, smi, name=None):
        """
        creates a molecule object from a smiles string, X, or if X is a path, the smiles string it points to
        """
        rdmol = Chem.MolFromSmiles(smi)
        new_mol = Molecule.from_rdmol(rdmol, smi, name)
        new_mol = new_mol.remove_Hs()
        m = cls.get_smi_mapping(smi, new_mol)
        atomn = len(new_mol.atom_types.values())
        assert len(
            m) == atomn, f'some atoms do not have a new index. Inds: {len(m)}. Atoms: {atomn}'
        new_mol = new_mol.renumber_indexes(m)
        new_mol = new_mol.add_Hs()
        new_mol.set_neighbors()

        return new_mol


if __name__ == "__main__":
    pass
