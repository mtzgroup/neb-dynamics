from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union

from chain import Chain
from numpy.typing import NDArray
from qcio.models.inputs import ProgramInput
from qcio.models.structure import Structure
import qcop

from neb_dynamics.chain import Chain
from neb_dynamics.nodes.node import Node


@dataclass
class Engine(ABC):

    @abstractmethod
    def compute_gradients(self, chain: Chain) -> NDArray:
        """
        returns the gradients for each node in the chain as
        specified by the object inputs
        """
        ...

    @abstractmethod
    def compute_energies(self, chain: Chain) -> NDArray:
        """
        returns the gradients for each node in the chain as
        specified by the object inputs
        """
        ...


@dataclass
class QCOPEngine(Engine):
    program_input: ProgramInput
    program: str

    def compute_gradients(self, chain: Chain) -> NDArray:
        assert isinstance(
            chain.nodes[0], Node), "input Chain has nodes incompatible with QCOPEngine."

        # first make sure the program input has calctype set to gradients
        prog_inp = self._change_prog_input_property(
            prog_inp=self.program_input, key='calctype', value='gradient')

        # now create program inputs for each geometry
        all_prog_inps = [self._change_prog_input_property(
            prog_inp=prog_inp, key='structure', value=node.structure
        ) for node in chain]

        all_results = [qcop.compute(self.program, pi) for pi in all_prog_inps]
        for node, result in zip(chain.nodes, all_results):
            node._cached_result = result

        return chain.gradients

    def compute_energies(self, chain: Chain) -> NDArray:
        assert isinstance(
            chain.nodes[0], Node), "input Chain has nodes incompatible with QCOPEngine."

        # first make sure the program input has calctype set to gradients
        prog_inp = self._change_prog_input_property(
            prog_inp=self.program_input, key='calctype', value='energy')

        # now create program inputs for each geometry
        all_prog_inps = [self._change_prog_input_property(
            prog_inp=prog_inp, key='structure', value=node.structure
        ) for node in chain]

        all_results = [qcop.compute(self.program, pi) for pi in all_prog_inps]
        for node, result in zip(chain.nodes, all_results):
            node._cached_result = result

        return chain.energies

    def _change_prog_input_property(self, prog_inp: ProgramInput,
                                    key: str, value: Union[str, Structure]):
        prog_dict = prog_inp.__dict__.copy()
        if prog_dict[key] is not value:
            prog_dict['calctype'] = 'gradient'
            new_prog_inp = ProgramInput(**prog_dict)
        else:
            new_prog_inp = prog_inp

        return new_prog_inp

