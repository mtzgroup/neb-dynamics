from dataclasses import dataclass
from typing import List, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Hartree
from numpy.typing import NDArray
from qcio.constants import ANGSTROM_TO_BOHR
from qcio_structure_helpers import structure_to_ase_atoms

from neb_dynamics.chain import Chain
from neb_dynamics.engines import Engine
from neb_dynamics.errors import EnergiesNotComputedError, GradientsNotComputedError
from neb_dynamics.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from neb_dynamics.nodes.node import StructureNode


@dataclass
class ASEEngine(Engine):
    """
    !!! Warning:
    ASE uses the following standard units:
        - energy (eV)
        - positions (Angstroms)

        Neb-dynamics uses Hartree and Bohr for coordinates.
        Appropriate conversions must me made.
    """

    calculator: Calculator

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

        # now create program inputs for each geometry that is not frozen
        inds_frozen = [i for i, node in enumerate(node_list) if node.converged]
        all_ase_atoms = [structure_to_ase_atoms(node.structure) for node in node_list]
        non_frozen_ase_atoms = [
            atoms for i, atoms in enumerate(all_ase_atoms) if i not in inds_frozen
        ]
        non_frozen_results = [
            self.compute_func(atoms=atoms) for atoms in non_frozen_ase_atoms
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

    def compute_func(self, atoms: Atoms):
        ene_ev = self.calculator.get_potential_energy(atoms=atoms)  # eV
        ene = ene_ev / Hartree  # Hartree

        # ASE outputs the negative gradient
        grad_ev_ang = self.calculator.get_forces(atoms=atoms) * (-1)  # eV / Angstroms
        grad = (grad_ev_ang / ANGSTROM_TO_BOHR) / Hartree  # Hartree / Bohr

        res = FakeQCIOResults(energy=ene, gradient=grad)
        return FakeQCIOOutput(results=res)
