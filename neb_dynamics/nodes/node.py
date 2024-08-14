from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray
from qcio.models.outputs import ProgramOutput
from qcio.models.structure import Structure

from neb_dynamics.errors import (EnergiesNotComputedError,
                                 GradientsNotComputedError)
from neb_dynamics.fakeoutputs import FakeQCIOOutput
from neb_dynamics.molecule import Molecule
from neb_dynamics.qcio_structure_helpers import structure_to_molecule


@dataclass
class Node(ABC):

    do_climb: bool = False

    @property
    @abstractmethod
    def has_molecular_graph(self): ...

    @property
    @abstractmethod
    def converged(self): ...

    @property
    @abstractmethod
    def structure(self): ...

    @property
    @abstractmethod
    def _cached_energy(self): ...

    @property
    @abstractmethod
    def _cached_gradient(self): ...

    @abstractmethod
    def update_coords(coords): ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def coords(self):
        ...

    @property
    def energy(self):
        if self._cached_energy is not None:
            return self._cached_energy
        else:
            raise EnergiesNotComputedError()

    @property
    def gradient(self):
        if self._cached_gradient is not None:
            return self._cached_gradient
        else:
            raise GradientsNotComputedError()


@dataclass
class XYNode(Node):
    structure: np.array = None
    converged: bool = False
    has_molecular_graph = False
    _cached_result: FakeQCIOOutput = None
    _cached_energy: float = None
    _cached_gradient: NDArray = None

    def update_coords(self, new_coords: np.array) -> XYNode:
        """
        returns a new XYNode with coordinates 'new_coords'
        """
        copy_node = self.copy()
        copy_node.structure = new_coords
        return copy_node

    @property
    def coords(self) -> np.array:
        """
        shortcut to coordinates stored in Structure.geometry
        """
        return self.structure

    def copy(self) -> XYNode:
        return XYNode(**self.__dict__)

    def __eq__(self, other: Node) -> bool:
        DIST_THRESH = 0.001
        return np.linalg.norm(self.coords - other.coords) <= DIST_THRESH


@dataclass
class StructureNode(Node):
    structure: Structure = None
    has_molecular_graph: bool = True
    converged: bool = False
    _cached_energy: float = None
    _cached_gradient: NDArray = None

    _cached_result: Union[ProgramOutput, FakeQCIOOutput] = None
    graph: Molecule = None

    def __eq__(self, other: Node) -> bool:
        from neb_dynamics.nodes.nodehelpers import is_identical

        return is_identical(self, other)

    def __post_init__(self):
        self.graph = structure_to_molecule(self.structure)
        if self._cached_result is not None:
            self._cached_energy = self._cached_result.results.energy
            self._cached_gradient = self._cached_result.results.gradient

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
        copy_node._cached_gradient = None
        copy_node._cached_energy = None

        new_struct_dict = copy_node.structure.__dict__.copy()
        new_struct_dict["geometry"] = new_coords
        copy_node.structure = Structure(**new_struct_dict)
        return copy_node

    @property
    def symbols(self):
        return self.structure.symbols

    def copy(self) -> StructureNode:
        return StructureNode(**self.__dict__)
