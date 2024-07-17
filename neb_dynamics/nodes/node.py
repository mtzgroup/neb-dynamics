from __future__ import annotations

from dataclasses import dataclass
from qcio.models.structure import Structure
import numpy as np


@dataclass
class Node:
    structure: Structure

    converged: bool = False
    has_molecular_graph: bool = False
    energy: float = False
    gradient: np.array = False
    do_climb: bool = False

    @property
    def coords(self) -> np.array:
        """
        shortcut to coordinates stored in Structure.geometry
        """
        return self.structure.geometry
