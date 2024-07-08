from dataclasses import dataclass
from typing import Any

@dataclass
class NoneConvergedException(Exception):

    trajectory: list
    msg: str
    obj: Any = None


@dataclass
class ElectronicStructureError(Exception):

    trajectory: list
    msg: str
    obj: Any = None
