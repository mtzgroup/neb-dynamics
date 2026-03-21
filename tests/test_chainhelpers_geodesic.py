import numpy as np
import pytest
from qcio import Structure

from neb_dynamics.chainhelpers import run_geodesic
from neb_dynamics.nodes.node import StructureNode


def _structure(symbols, geometry):
    return Structure(
        geometry=np.array(geometry, dtype=float),
        symbols=list(symbols),
        charge=0,
        multiplicity=1,
    )


def test_run_geodesic_reorders_endpoint_coordinates_to_reference_symbols(monkeypatch):
    start = StructureNode(
        structure=_structure(
            ["C", "Br", "Cl"],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
        )
    )
    end = StructureNode(
        structure=_structure(
            ["C", "Cl", "Br"],
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [7.5, 0.0, 0.0],
            ],
        )
    )

    captured = {}

    class _FakeSmoother:
        def __init__(self, path):
            self.path = np.array(path, dtype=float)

    def _fake_run_geodesic_get_smoother(input_object, **kwargs):
        symbols, coords = input_object
        captured["symbols"] = list(symbols)
        captured["coords"] = np.array(coords, copy=True)
        return _FakeSmoother(coords)

    monkeypatch.setattr(
        "neb_dynamics.chainhelpers.run_geodesic_get_smoother",
        _fake_run_geodesic_get_smoother,
    )

    with pytest.warns(RuntimeWarning, match="not provided with the same atomic symbol ordering"):
        gi = run_geodesic([start, end], nimages=5, align=True)

    assert captured["symbols"] == ["C", "Br", "Cl"]
    assert np.allclose(captured["coords"][0], start.coords)
    assert np.allclose(
        captured["coords"][-1],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [7.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        ),
    )
    assert list(gi[-1].symbols) == ["C", "Br", "Cl"]
    assert np.allclose(gi[-1].coords, captured["coords"][-1])
