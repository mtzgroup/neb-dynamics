from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AlessioError(Exception):
    message: str


class Node(ABC):

    @property
    @abstractmethod
    def converged(self):
        ...

    @abstractmethod
    def en_func(coords):
        ...

    @abstractmethod
    def grad_func(coords):
        ...

    @property
    @abstractmethod
    def energy(self):
        ...

    @abstractmethod
    def copy(self):
        ...

    @property
    @abstractmethod
    def gradient(self):
        ...

    @property
    @abstractmethod
    def coords(self):
        ...

    @property
    @abstractmethod
    def do_climb(self):
        ...

    @abstractmethod
    def dot_function(self, other):
        ...

    @abstractmethod
    def update_coords(self, coords):
        ...

    @abstractmethod
    def do_geometry_optimization(self) -> Node:
        ... 
        
    @abstractmethod
    def is_identical(self, other):
        ...