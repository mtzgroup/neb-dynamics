from dataclasses import dataclass
from typing import Any
from neb_dynamics.Chain import Chain


@dataclass
class NoneConvergedException(Exception):

    trajectory: list[Chain]
    msg: str
    obj: Any = None


@dataclass
class ElectronicStructureError(Exception):

    trajectory: list[Chain]
    msg: str
    obj: Any = None
