from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List
from neb_dynamics.chain import Chain
import numpy as np


from neb_dynamics.nodes.node import Node
from neb_dynamics.fakeoutputs import FakeQCIOOutput
from qcio.models.outputs import ProgramOutput


@dataclass
class Engine(ABC):

    @abstractmethod
    def compute_gradients(self, chain: Union[Chain, List]) -> Union[FakeQCIOOutput, ProgramOutput]:
        """
        returns the gradients for each node in the chain as
        specified by the object inputs
        """
        ...

    @abstractmethod
    def compute_energies(self, chain: Union[Chain, List]) -> Union[FakeQCIOOutput, ProgramOutput]:
        """
        returns the gradients for each node in the chain as
        specified by the object inputs
        """
        ...

    def steepest_descent(self, node: Node, ss=1, max_steps=10,
                         ene_thre: float = 1e-6, grad_thre: float = 1e-4) -> list[Node]:
        # print("************\n\n\n\nRUNNING STEEPEST DESCENT\n\n\n\nn***********")
        history = []
        last_node = node.copy()
        # make sure the node isn't frozen so it returns a gradient
        last_node.converged = False

        curr_step = 0
        converged = False
        while curr_step < max_steps and not converged:
            grad = last_node.gradient
            new_coords = last_node.coords - 1*ss*grad
            node_new = last_node.update_coords(new_coords)
            grads = self.compute_gradients([node_new])
            ene = self.compute_energies([node_new])
            node_new._cached_gradient = grads[0]
            node_new._cached_energy = ene[0]

            history.append(node_new)

            delta_en = node_new.energy - last_node.energy
            grad_inf_norm = np.amax(np.abs(node_new.gradient))
            converged = delta_en <= ene_thre and grad_inf_norm <= grad_thre

            last_node = node_new.copy()
            curr_step += 1

        return history
