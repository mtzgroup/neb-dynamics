from __future__ import annotations

from dataclasses import dataclass
from qcio.models.structure import Structure
from qcio.models.outputs import ProgramOutput
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


