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
from neb_dynamics.nodes.nodehelpers import update_node_cache
from neb_dynamics.qcio_structure_helpers import _change_prog_input_property
from neb_dynamics.dynamics.chainbiaser import ChainBiaser

AVAIL_PROGRAMS = ["qcop", "chemcloud"]


@dataclass
class QCOPEngine(Engine):
    program_input: ProgramInput = ProgramInput(
        structure=Structure.from_smiles("C"),
        model={"method": "GFN2xTB"},
        calctype="gradient",
    )
    program: str = "xtb"
    geometry_optimizer: str = "geometric"
    compute_program: str = "qcop"
    biaser: ChainBiaser = None

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        try:
            grads = np.array([node.gradient for node in chain])

        except GradientsNotComputedError:
            node_list = self._run_calc(chain=chain, calctype="gradient")
            grads = np.array([node.gradient for node in node_list])

        if self.biaser:
            new_grads = grads.copy()
            for i, (node, grad) in enumerate(zip(chain, grads)):
                for ref_chain in self.biaser.reference_chains:
                    g_bias = self.biaser.gradient_node_bias(node=node)
                    new_grads[i] += g_bias
            grads = new_grads
        return grads

    def compute_energies(self, chain: Chain) -> NDArray:
        try:
            enes = np.array([node.energy for node in chain])
        except EnergiesNotComputedError:
            node_list = self._run_calc(
                chain=chain, calctype="gradient"
            )  # need the gradients to be cached anyway
            enes = np.array([node.energy for node in node_list])

        except AttributeError:
            node_list = self._run_calc(
                chain=chain, calctype="gradient"
            )  # need the gradients to be cached anyway
            enes = np.array([node.energy for node in node_list])

        if self.biaser:
            new_enes = enes.copy()
            for i, (node, ene) in enumerate(zip(chain, enes)):
                for ref_chain in self.biaser.reference_chains:
                    dist = self.biaser.compute_min_dist_to_ref(
                        node=node,
                        dist_func=self.biaser.compute_euclidean_distance,
                        reference=ref_chain
                    )
                    new_enes[i] += self.biaser.energy_gaussian_bias(distance=dist)
            enes = new_enes
        return enes

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
            ), f"input list has nodes incompatible with QCOPEngine: {chain[0]}"
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

        if self.compute_program == "chemcloud":
            # the following hack is implemented until chemcloud can accept single-length lists.
            if len(non_frozen_prog_inps) == 1:
                non_frozen_prog_inps = non_frozen_prog_inps[0]
                non_frozen_results = self.compute_func(
                    self.program, non_frozen_prog_inps
                )
                non_frozen_results = [non_frozen_results]
            else:
                non_frozen_results = self.compute_func(
                    self.program, non_frozen_prog_inps
                )

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

        update_node_cache(node_list=node_list, results=all_results)
        return node_list

    def _run_geom_opt_calc(self, structure: Structure):
        """
        this will return a ProgramOutput from qcio geom opt call.
        """
        if "terachem" not in self.program:
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

        else:
            prog_input = ProgramInput(
                structure=structure,
                # Can be "energy", "gradient", "hessian", "optimization", "transition_state"
                calctype="optimization",  # type: ignore
                model=self.program_input.model,
                keywords={
                    "purify": "no",
                    "new_minimizer": "yes",
                },  # new_minimizer yes is required
            )
            output = self.compute_func("terachem", prog_input)

        return output

    def _compute_hessian_result(self, node: StructureNode):
        from chemcloud import CCClient

        prog = self.program
        if "terachem" in self.program:
            prog = "terachem"

        dpi = DualProgramInput(
            calctype="hessian",  # type: ignore
            structure=node.structure,
            subprogram=prog,
            subprogram_args={
                "model": self.program_input.model,
                "keywords": self.program_input.keywords,
            },
            keywords={},
        )
        client = CCClient()
        fres = client.compute("bigchem", dpi)
        output = fres.get()
        return output

    def _compute_ts_result(self, node: StructureNode, keywords={'maxiter': 200}):
        dpi = DualProgramInput(
            structure=node.structure,
            calctype="transition_state",
            subprogram=self.program,
            subprogram_args={
                "model": self.program_input.model,
                "keywords": self.program_input.keywords,
            })
        return self.compute_func('geometric', dpi)

    def compute_hessian(self, node: StructureNode):
        output = self._compute_hessian_result(node)
        return output.return_result

    def compute_transition_state(self, node: StructureNode, keywords={'maxiter': 200}):
        output = self._compute_ts_result(node=node)
        if output.success:
            return StructureNode(structure=output.return_result)
        else:
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
