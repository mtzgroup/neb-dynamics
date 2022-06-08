from dataclasses import dataclass

@dataclass
class NoneConvergedException(Exception):
    trajectory: list[Chain]
    message: str

@dataclass
class AlessioError(Exception):
    message: str