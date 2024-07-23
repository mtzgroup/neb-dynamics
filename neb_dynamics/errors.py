from dataclasses import dataclass
from typing import Any

@dataclass
class NoneConvergedException(Exception):

    trajectory: list
    msg: str
    obj: Any = None

@dataclass
class EnergiesNotComputedError(Exception):
    msg: str = "Energies not computed."

@dataclass
class GradientsNotComputedError(Exception):
    msg: str = "Gradients not computed."

@dataclass
class ElectronicStructureError(Exception):

    msg: str
    obj: Any = None
