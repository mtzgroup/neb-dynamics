from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Optimizer(ABC):
    @abstractmethod
    def optimize_step(self):
        ...
