from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from qcio import ProgramOutput, Structure

from neb_dynamics.errors import (EnergiesNotComputedError,
                                 GradientsNotComputedError)
from neb_dynamics.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from neb_dynamics.molecule import Molecule
from neb_dynamics.qcio_structure_helpers import structure_to_molecule


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


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
    # has_molecular_graph: bool = True
    has_molecular_graph: bool = True  # temp changing it for protein stuff
    converged: bool = False
    _cached_energy: float = None
    _cached_gradient: list = None
    comparison_atom_indices: list[int] | None = None
    disable_smiles: bool = False

    _cached_result: Union[ProgramOutput, FakeQCIOOutput] = None
    graph: Molecule = None

    def __post_init__(self):
        if self.comparison_atom_indices is not None:
            self.comparison_atom_indices = [int(i) for i in self.comparison_atom_indices]
        self.disable_smiles = bool(self.disable_smiles)
        if self.has_molecular_graph and self.graph is None:
            self.graph = structure_to_molecule(self.structure)
        if not self.has_molecular_graph:
            self.graph = None
        if self._cached_result is not None:
            self._cached_energy = self._cached_result.results.energy
            self._cached_gradient = self._cached_result.results.gradient

    @classmethod
    def from_serializable(cls, data):
        if data.get('graph') is not None:
            data['graph'] = Molecule.from_serializable(data['graph'])
        data['structure'] = Structure(**data['structure'])
        if '_cached_result' in data and data['_cached_result'] is not None:
            if len(data['_cached_result'].keys()) > 2:
                data['_cached_result'] = ProgramOutput(
                    **data['_cached_result'])
            else:
                d = data['_cached_result']
                res = FakeQCIOOutput.model_validate({
                    "results": FakeQCIOResults.model_validate(
                        {"energy": d['results']['energy'],
                         "gradient": d['results']['gradient']})})
                data['_cached_result'] = res
        return cls(**data)

    def to_serializable(self):
        data = self.__dict__.copy()
        data['structure'] = data['structure'].model_dump()
        data["_cached_energy"] = _json_safe(data.get("_cached_energy"))
        data["_cached_gradient"] = _json_safe(data.get("_cached_gradient"))
        if data["_cached_result"] is not None:
            data["_cached_result"] = _json_safe(data["_cached_result"].model_dump())
        if hasattr(self, 'graph') and data['graph'] is not None:
            data['graph'] = data['graph'].to_serializable()
        return data

    @property
    def coords(self) -> np.array:
        """
        shortcut to coordinates stored in Structure.geometry
        """
        return self.structure.geometry.copy()

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
        new_struct_dict["charge"] = copy_node.structure.charge
        new_struct_dict["multiplicity"] = copy_node.structure.multiplicity

        copy_node.structure = Structure(**new_struct_dict)
        return copy_node

    @property
    def symbols(self):
        return self.structure.symbols

    def copy(self) -> StructureNode:
        return StructureNode(**self.__dict__.copy())
