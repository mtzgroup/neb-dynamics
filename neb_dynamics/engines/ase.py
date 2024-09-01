from dataclasses import dataclass
from typing import List, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Hartree
from numpy.typing import NDArray
from qcio.constants import ANGSTROM_TO_BOHR
from neb_dynamics.qcio_structure_helpers import structure_to_ase_atoms, ase_atoms_to_structure

from neb_dynamics.chain import Chain
from neb_dynamics.engines.engine import Engine
from neb_dynamics.errors import EnergiesNotComputedError, GradientsNotComputedError, ElectronicStructureError
from neb_dynamics.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.nodes.nodehelpers import update_node_cache

from ase.optimize.optimize import Optimizer
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.bfgs import BFGS
from ase.optimize.fire import FIRE
from ase.optimize.mdmin import MDMin


from ase.io import Trajectory

from pathlib import Path
import tempfile

AVAIL_OPTS = {
    'LBFGS': LBFGS,
    "BFGS": BFGS,
    "FIRE": FIRE,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
}

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
    ase_optimizer: Optimizer = None
    ase_opt_str: str = "LBFGS"

    def __post_init__(self):
        if self.ase_optimizer is None:
            assert self.ase_opt_str is not None and self.ase_opt_str in AVAIL_OPTS.keys(), f"Must input either an ase optimizer or a string name for an\
             available optimizer: {AVAIL_OPTS.keys()}"

            self.ase_optimizer = AVAIL_OPTS[self.ase_opt_str]

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
        update_node_cache(node_list=node_list, results=all_results)
        return node_list


    def compute_func(self, atoms: Atoms):
        try:
            ene_ev = self.calculator.get_potential_energy(atoms=atoms)  # eV
            ene = ene_ev / Hartree  # Hartree

            # ASE outputs the negative gradient
            grad_ev_ang = self.calculator.get_forces(atoms=atoms) * (-1)  # eV / Angstroms
            grad = (grad_ev_ang / ANGSTROM_TO_BOHR) / Hartree  # Hartree / Bohr

            res = FakeQCIOResults(energy=ene, gradient=grad)
            return FakeQCIOOutput(results=res)
        except Exception:
            raise ElectronicStructureError(msg='Electronic structure failed.')

    def compute_geometry_optimization(self, node: StructureNode) -> list[StructureNode]:
        """
        Computes a geometry optimization using ASE calculation and optimizer
        """
        atoms = structure_to_ase_atoms(node.structure)
        atoms.set_calculator(self.calculator)
        tmp = tempfile.NamedTemporaryFile(suffix=".traj", mode="w+", delete=False)

        optimizer = self.ase_optimizer(atoms=atoms, logfile=None, trajectory=tmp.name)  # ASE doing geometry optimizations inplace
        try:
            optimizer.run(fmax=0.01)
        except Exception:
            raise ElectronicStructureError(msg='Electronic structure failed.')

        # 'atoms' variable is now updated
        charge = node.structure.charge
        multiplicity = node.structure.multiplicity

        aT = Trajectory(tmp.name)
        traj_list = []
        for i, _ in enumerate(aT):
            traj_list.append(
                ase_atoms_to_structure(atoms=aT[i], charge=charge, multiplicity=multiplicity)
            )

        energies = [obj.get_potential_energy()/ Hartree  for obj in aT]
        gradients = [(-1*obj.get_forces()/ANGSTROM_TO_BOHR) for obj in aT]
        all_results = []
        for ene, grad in zip(energies, gradients):
            res = FakeQCIOResults(energy=ene, gradient=grad)
            out = FakeQCIOOutput(results=res)
            all_results.append(out)
        Path(tmp.name).unlink()
        node_list = [StructureNode(structure=struct) for struct in traj_list]
        update_node_cache(node_list=node_list, results=all_results)
        return node_list
