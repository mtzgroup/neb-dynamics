from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List
from functools import partial
from neb_dynamics.chain import Chain
from numpy.typing import NDArray
import numpy as np
from qcio.models.inputs import ProgramInput, DualProgramInput, Structure
import qcop
import concurrent

from neb_dynamics.nodes.node import Node
from neb_dynamics.qcio_structure_helpers import _change_prog_input_property


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

    def steepest_descent(self, node: Node, ss=1, max_steps=10) -> list[Node]:
        history = []
        last_node = node.copy()
        # make sure the node isn't frozen so it returns a gradient
        last_node.converged = False
        for i in range(max_steps):
            grad = last_node.gradient
            new_coords = last_node.coords - 1*ss*grad
            node_new = last_node.update_coords(new_coords)
            self.compute_gradients([node_new])
            history.append(node_new)
            last_node = node_new.copy()
        return history


@dataclass
class QCOPEngine(Engine):
    program_input: ProgramInput
    program: str
    geometry_optimizer: str = 'geometric'

    def _run_calc(self, calctype: str, chain: Union[Chain, List]) -> List[Node]:
        if isinstance(chain, Chain):
            assert isinstance(
                chain.nodes[0], Node), "input Chain has nodes incompatible with QCOPEngine."
            node_list = chain.nodes
        elif isinstance(chain, list):
            assert isinstance(
                chain[0], Node), "input list has nodes incompatible with QCOPEngine."
            node_list = chain
        else:
            raise ValueError(
                f"Input needs to be a Chain or a List. You input a: {type(chain)}")

        # first make sure the program input has calctype set to gradients
        prog_inp = _change_prog_input_property(
            prog_inp=self.program_input, key='calctype', value=calctype)

        # now create program inputs for each geometry that is not frozen
        inds_frozen = [i for i, node in enumerate(
            node_list) if node.converged]
        all_prog_inps = [_change_prog_input_property(
            prog_inp=prog_inp, key='structure', value=node.structure
        ) for node in node_list]
        non_frozen_prog_inps = [pi for i, pi in enumerate(
            all_prog_inps) if i not in inds_frozen]

        # only compute the nonfrozen structures
        if self.program == 'xtb':
            def helper(inp):
                prog, prog_inp = inp
                return qcop.compute(prog, prog_inp)

            iterables = [(self.program, inp) for inp in non_frozen_prog_inps]
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                non_frozen_results = list(executor.map(helper, iterables))
        else:
            non_frozen_results = [qcop.compute(
                self.program, pi) for pi in non_frozen_prog_inps]

        # merge the results
        all_results = []
        for i, node in enumerate(node_list):
            if i in inds_frozen:
                all_results.append(node_list[i]._cached_result)
            else:
                all_results.append(non_frozen_results.pop(0))

        for node, result in zip(node_list, all_results):
            node._cached_result = result

        return node_list

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        node_list = self._run_calc(chain=chain, calctype='gradient')
        return np.array([node.gradient for node in node_list])

    def compute_energies(self, chain: Chain) -> NDArray:
        node_list = self._run_calc(chain=chain, calctype='energy')
        return np.array([node.energy for node in node_list])

    def _run_geom_opt_calc(self, structure: Structure):
        """
        this will return a ProgramOutput from qcio geom opt call.
        """
        dpi = DualProgramInput(
            calctype="optimization",  # type: ignore
            structure=structure,
            subprogram=self.program,
            subprogram_args={'model': self.program_input.model,
                             'keywords': self.program_input.keywords},
            keywords={}
        )

        output = qcop.compute(
            self.geometry_optimizer, dpi, propagate_wfn=False, rm_scratch_dir=False)
        return output

    def compute_geometry_optimization(self, node: Node) -> list[Node]:
        """
        will run a geometry optimization call and parse the output into
        a list of Node objects
        """
        output = self._run_geom_opt_calc(structure=node.structure)
        all_outputs = output.results.trajectory
        structures = [output.input_data.structure for output in all_outputs]
        return [Node(structure=struct, _cached_result=result) for struct, result in zip(structures, all_outputs)]
