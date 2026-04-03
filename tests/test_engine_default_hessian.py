from dataclasses import dataclass, field

import numpy as np
import pytest

from neb_dynamics.engines.engine import Engine
from neb_dynamics.nodes.node import XYNode


@dataclass
class _QuadraticEngine(Engine):
    k: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [3.0, 0.4],
                [0.4, 1.5],
            ],
            dtype=float,
        )
    )

    def compute_energies(self, chain):
        energies = []
        for node in chain:
            x = np.asarray(node.coords, dtype=float).reshape(-1)
            energies.append(0.5 * float(x @ self.k @ x))
        return np.asarray(energies, dtype=float)

    def compute_gradients(self, chain):
        gradients = []
        for node in chain:
            x = np.asarray(node.coords, dtype=float).reshape(-1)
            gradients.append((self.k @ x).reshape(np.asarray(node.coords).shape))
        return np.asarray(gradients, dtype=float)


def test_default_compute_hessian_uses_finite_difference_and_warns():
    eng = _QuadraticEngine()
    node = XYNode(structure=np.array([0.35, -0.15], dtype=float))

    with pytest.warns(RuntimeWarning, match="finite-difference Hessian fallback"):
        hessian = eng.compute_hessian(node=node, step_size=1e-4)

    assert np.allclose(hessian, eng.k, atol=1e-5)


def test_default_compute_hessian_result_contains_modes_and_hessian():
    eng = _QuadraticEngine()
    node = XYNode(structure=np.array([0.2, 0.1], dtype=float))

    hessres = eng._compute_hessian_result(node=node, step_size=1e-4)

    assert hessres.success is True
    assert np.allclose(hessres.results.hessian, eng.k, atol=1e-5)
    assert hessres.return_result.shape == (2, 2)
    assert len(hessres.results.normal_modes_cartesian) == 2
    assert all(np.asarray(mode).shape == (2,) for mode in hessres.results.normal_modes_cartesian)
    assert len(hessres.results.freqs_wavenumber) == 2
