from types import SimpleNamespace

import numpy as np
from qcio import Structure

from neb_dynamics.elementarystep import _run_geom_opt
from neb_dynamics.nodes.node import StructureNode


def _node() -> StructureNode:
    n = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            symbols=["H", "H"],
            charge=0,
            multiplicity=1,
        )
    )
    n.has_molecular_graph = False
    n.graph = None
    return n


def test_run_geom_opt_engine_without_geometry_optimizer_attribute():
    captured = {}

    class _Engine:
        def compute_geometry_optimization(self, node, keywords=None):
            captured["keywords"] = keywords
            return [node]

    traj = _run_geom_opt(_node(), _Engine())
    assert len(traj) == 1
    assert captured["keywords"] == {}


def test_run_geom_opt_geometric_engine_uses_geometric_defaults():
    captured = {}

    class _Engine(SimpleNamespace):
        geometry_optimizer = "geometric"

        def compute_geometry_optimization(self, node, keywords=None):
            captured["keywords"] = keywords
            return [node]

    traj = _run_geom_opt(_node(), _Engine())
    assert len(traj) == 1
    assert captured["keywords"] == {"coord_sys": "cart", "maxiter": 1000}
