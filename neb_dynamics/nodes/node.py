from __future__ import annotations

from dataclasses import dataclass
from qcio.models.structure import Structure
from qcio.models.outputs import ProgramOutput
from neb_dynamics.helper_functions import _change_prog_input_property
import numpy as np


@dataclass
class Node:
    structure: Structure

    converged: bool = False
    has_molecular_graph: bool = False

    do_climb: bool = False
    _cached_result: ProgramOutput = None

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
            return None

    @property
    def gradient(self):
        if self._cached_result is not None:
            return self._cached_result.results.gradient
        else:
            return None
