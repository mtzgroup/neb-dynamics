from __future__ import annotations

from dataclasses import dataclass
from qcio.models.structure import Structure
from qcio.models.outputs import ProgramOutput
import numpy as np
from neb_dynamics.molecule import Molecule
from neb_dynamics.qcio_structure_helpers import structure_to_molecule
from neb_dynamics.errors import GradientsNotComputedError, EnergiesNotComputedError



@dataclass
class Node:
    structure: Structure
    graph: Molecule = None
    converged: bool = False
    has_molecular_graph: bool = True

    do_climb: bool = False
    _cached_result: ProgramOutput = None

    def __eq__(self, other: Node) -> bool:
        from neb_dynamics.nodes.nodehelpers import is_identical
        return is_identical(self, other)

    def __post_init__(self):
        self.graph = structure_to_molecule(self.structure)


    @property
    def coords(self) -> np.array:
        """
        shortcut to coordinates stored in Structure.geometry
        """
        return self.structure.geometry

    def update_coords(self, new_coords: np.array) -> Node:
        """
        returns a new Node with coordinates 'new_coords'
        """
        copy_node = self.copy()
        copy_node._cached_result = None
        new_struct_dict = copy_node.structure.__dict__.copy()
        new_struct_dict['geometry'] = new_coords
        copy_node.structure = Structure(**new_struct_dict)
        return copy_node

    def copy(self) -> Node:
        return Node(**self.__dict__)

    @property
    def energy(self):
        if self._cached_result is not None:
            return self._cached_result.results.energy
        else:
            raise EnergiesNotComputedError

    @property
    def gradient(self):
        if self._cached_result is not None:
            return self._cached_result.results.gradient
        else:
            raise GradientsNotComputedError

    @property
    def symbols(self):
        return self.structure.symbols
