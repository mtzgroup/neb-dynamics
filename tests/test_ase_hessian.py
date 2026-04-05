import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree
from qcconst.constants import ANGSTROM_TO_BOHR
from qcio import Structure

from neb_dynamics.engines.ase import ASEEngine
from neb_dynamics.engines.engine import Engine
from neb_dynamics.nodes.node import StructureNode


class _CountingHarmonicCalculator(Calculator):
    """Simple harmonic potential in ASE units with call accounting."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k: float = 2.0):
        super().__init__()
        self.k = float(k)
        self.energy_calls = 0
        self.force_calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, list(properties), system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)

        if "energy" in properties:
            self.energy_calls += 1
            self.results["energy"] = 0.5 * self.k * float(np.sum(positions**2))
        if "forces" in properties:
            self.force_calls += 1
            self.results["forces"] = -self.k * positions


def _single_atom_node() -> StructureNode:
    structure = Structure(
        symbols=["H"],
        geometry=np.array([[0.0, 0.0, 0.0]], dtype=float),
        charge=0,
        multiplicity=1,
    )
    return StructureNode(structure=structure)


def test_ase_hessian_gradient_fd_is_correct_and_avoids_energy_calls():
    calc = _CountingHarmonicCalculator(k=2.5)
    eng = ASEEngine(calculator=calc)
    node = _single_atom_node()

    hessian = eng.compute_hessian(node=node, step_size=1e-3)

    ndof = int(np.asarray(node.coords).size)
    expected_diag = float(calc.k) / (Hartree * (ANGSTROM_TO_BOHR**2))
    expected = np.eye(ndof, dtype=float) * expected_diag

    assert np.allclose(hessian, expected, atol=1e-6)
    assert calc.energy_calls == 0
    assert calc.force_calls == 2 * ndof


def test_ase_hessian_is_much_cheaper_than_default_engine_fallback():
    step_size = 1e-3
    node = _single_atom_node()
    ndof = int(np.asarray(node.coords).size)

    fast_calc = _CountingHarmonicCalculator(k=1.7)
    fast_eng = ASEEngine(calculator=fast_calc)
    h_fast = fast_eng.compute_hessian(node=node, step_size=step_size)

    slow_calc = _CountingHarmonicCalculator(k=1.7)
    slow_eng = ASEEngine(calculator=slow_calc)
    with pytest.warns(RuntimeWarning, match="finite-difference Hessian fallback"):
        h_slow = Engine.compute_hessian(slow_eng, node=node, step_size=step_size)

    expected_fallback_evals = 1 + 2 * ndof * ndof

    assert np.allclose(h_fast, h_slow, atol=1e-6)
    assert fast_calc.force_calls == 2 * ndof
    assert fast_calc.energy_calls == 0
    assert slow_calc.energy_calls == expected_fallback_evals
    assert slow_calc.force_calls == expected_fallback_evals
