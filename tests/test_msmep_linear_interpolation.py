from types import SimpleNamespace

import numpy as np
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.msmep import MSMEP
from neb_dynamics.nodes.node import StructureNode


def _structure(coords: np.ndarray) -> Structure:
    return Structure(
        geometry=np.array(coords, dtype=float),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def test_linear_interpolation_preserves_endpoint_geometries_and_clears_cache():
    inputs = SimpleNamespace(
        engine=SimpleNamespace(__class__=SimpleNamespace(__name__="FakeEngine")),
        path_min_method="NEB",
        chain_inputs=ChainInputs(use_geodesic_interpolation=False),
        gi_inputs=SimpleNamespace(nimages=5),
    )
    m = MSMEP(inputs=inputs)

    start = StructureNode(structure=_structure([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]))
    end = StructureNode(structure=_structure([[1.0, 0.0, 0.0], [1.0, 0.0, 1.7]]))
    start._cached_energy = -1.0
    start._cached_gradient = np.ones((2, 3))
    start._cached_result = SimpleNamespace(results=SimpleNamespace(energy=-1.0, gradient=start._cached_gradient))
    end._cached_energy = -2.0
    end._cached_gradient = np.full((2, 3), 2.0)
    end._cached_result = SimpleNamespace(results=SimpleNamespace(energy=-2.0, gradient=end._cached_gradient))

    chain = Chain.model_validate({"nodes": [start, end], "parameters": inputs.chain_inputs})

    interpolation = m._create_interpolation(chain)

    assert len(interpolation) == 5
    np.testing.assert_allclose(interpolation[0].coords, start.coords)
    np.testing.assert_allclose(interpolation[-1].coords, end.coords)
    np.testing.assert_allclose(interpolation[2].coords, np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 1.2]]))

    assert interpolation[0] is not start
    assert interpolation[-1] is not end
    assert interpolation[0]._cached_energy is None
    assert interpolation[0]._cached_gradient is None
    assert interpolation[-1]._cached_energy is None
    assert interpolation[-1]._cached_gradient is None
