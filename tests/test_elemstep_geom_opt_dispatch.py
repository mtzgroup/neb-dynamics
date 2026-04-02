from neb_dynamics.elementarystep import _run_geom_opt


class _EngineWithoutGeometryOptimizer:
    def __init__(self):
        self.kwargs_seen = None

    def compute_geometry_optimization(self, node, keywords=None):
        self.kwargs_seen = keywords
        return ["ok"]


class _EngineWithGeometric:
    geometry_optimizer = "geometric"

    def __init__(self):
        self.kwargs_seen = None

    def compute_geometry_optimization(self, node, keywords=None):
        self.kwargs_seen = keywords
        return ["ok"]


def test_run_geom_opt_does_not_require_geometry_optimizer_attr():
    engine = _EngineWithoutGeometryOptimizer()
    out = _run_geom_opt(node=None, engine=engine)
    assert out == ["ok"]
    assert engine.kwargs_seen == {}


def test_run_geom_opt_passes_geometric_keywords_when_requested():
    engine = _EngineWithGeometric()
    out = _run_geom_opt(node=None, engine=engine)
    assert out == ["ok"]
    assert engine.kwargs_seen == {"coord_sys": "cart", "maxiter": 1000}
