from __future__ import annotations

import concurrent
from dataclasses import dataclass
from typing import List, Union
from numpy.typing import NDArray
import numpy as np

import qcop
from qcio.models.inputs import DualProgramInput, ProgramInput, Structure

from chemcloud import CCClient

from neb_dynamics.chain import Chain
from neb_dynamics.engines.engine import Engine
from neb_dynamics.errors import EnergiesNotComputedError, GradientsNotComputedError
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.qcio_structure_helpers import _change_prog_input_property

AVAIL_PROGRAMS = ["qcop", "chemcloud"]


@dataclass
class QCOPEngine(Engine):
    program_input: ProgramInput = ProgramInput(
        structure=Structure.from_smiles("C"),
        model={"method": "GFN2xTB"},
        calctype="energy",
    )
    program: str = "xtb"
    geometry_optimizer: str = "geometric"
    compute_program: str = "qcop"

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        try:
            return np.array([node.gradient for node in chain])
        except GradientsNotComputedError:
            node_list = self._run_calc(chain=chain, calctype="gradient")
            return np.array([node.gradient for node in node_list])

    def compute_energies(self, chain: Chain) -> NDArray:
        try:
            return np.array([node.energy for node in chain])
        except EnergiesNotComputedError:
            node_list = self._run_calc(chain=chain, calctype="energy")
            return np.array([node.energy for node in node_list])

    def compute_func(self, *args):
        if self.compute_program == "qcop":
            return qcop.compute(*args)
        elif self.compute_program == "chemcloud":
            client = CCClient()
            return client.compute(*args).get()
        else:
            raise ValueError(
                f"Invalid compute program: {self.compute_program}. Must be one of: {AVAIL_PROGRAMS}"
            )

    def _run_calc(
        self, calctype: str, chain: Union[Chain, List]
    ) -> List[StructureNode]:
        if isinstance(chain, Chain):
            assert isinstance(
                chain.nodes[0], StructureNode
            ), "input Chain has nodes incompatible with QCOPEngine."
            node_list = chain.nodes
        elif isinstance(chain, list):
            assert isinstance(
                chain[0], StructureNode
            ), "input list has nodes incompatible with QCOPEngine."
            node_list = chain
        else:
            raise ValueError(
                f"Input needs to be a Chain or a List. You input a: {type(chain)}"
            )

        # first make sure the program input has calctype set to input calctype
        prog_inp = _change_prog_input_property(
            prog_inp=self.program_input, key="calctype", value=calctype
        )

        # now create program inputs for each geometry that is not frozen
        inds_frozen = [i for i, node in enumerate(node_list) if node.converged]
        all_prog_inps = [
            _change_prog_input_property(
                prog_inp=prog_inp, key="structure", value=node.structure
            )
            for node in node_list
        ]
        non_frozen_prog_inps = [
            pi for i, pi in enumerate(all_prog_inps) if i not in inds_frozen
        ]

        # only compute the nonfrozen structures
        if self.program == "xtb" and self.compute_program == "qcop":

            def helper(inp):
                prog, prog_inp = inp
                return self.compute_func(prog, prog_inp)

            iterables = [(self.program, inp) for inp in non_frozen_prog_inps]
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                non_frozen_results = list(executor.map(helper, iterables))
        else:
            non_frozen_results = [
                self.compute_func(self.program, pi) for pi in non_frozen_prog_inps
            ]

        # merge the results
        all_results = []
        for i, node in enumerate(node_list):
            if i in inds_frozen:
                all_results.append(node_list[i]._cached_result)
            else:
                all_results.append(non_frozen_results.pop(0))

        for node, result in zip(node_list, all_results):
            node._cached_result = result
            node._cached_energy = result.results.energy
            node._cached_gradient = result.results.gradient

        return node_list

    def _run_geom_opt_calc(self, structure: Structure):
        """
        this will return a ProgramOutput from qcio geom opt call.
        """
        dpi = DualProgramInput(
            calctype="optimization",  # type: ignore
            structure=structure,
            subprogram=self.program,
            subprogram_args={
                "model": self.program_input.model,
                "keywords": self.program_input.keywords,
            },
            keywords={},
        )

        output = self.compute_func(self.geometry_optimizer, dpi)
        return output

    def compute_geometry_optimization(self, node: StructureNode) -> list[StructureNode]:
        """
        will run a geometry optimization call and parse the output into
        a list of Node objects
        """
        output = self._run_geom_opt_calc(structure=node.structure)
        all_outputs = output.results.trajectory
        structures = [output.input_data.structure for output in all_outputs]
        return [
            StructureNode(structure=struct, _cached_result=result)
            for struct, result in zip(structures, all_outputs)
        ]
