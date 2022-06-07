from dataclasses import dataclass
import numpy as np

@dataclass
class Chain:
    nodes: np.array

    def __iter__(self):
        return np.nditer(self.nodes)

@dataclass
class Node:
    coords: np.array

