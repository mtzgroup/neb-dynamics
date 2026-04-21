from types import SimpleNamespace

import numpy as np
from qcio import Structure
from qcio.models.inputs import ProgramArgs

from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.nodes.node import StructureNode


def _node_at_x(x: float) -> StructureNode:
    struct = Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]], dtype=float),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )
    return StructureNode(structure=struct)


def test_terachem_single_geomopt_preserves_program_keywords():
    engine = QCOPEngine(
        program="terachem",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 7, "precision": "mixed"},
        ),
    )
    captured = {}

    def _fake_compute_func(program, inp_obj, **kwargs):
        captured["program"] = program
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    assert captured["program"] == "terachem"
    kw = captured["input"].keywords
    assert kw["threads"] == 7
    assert kw["precision"] == "mixed"
    assert kw["purify"] == "no"
    assert kw["new_minimizer"] == "yes"


def test_terachem_batch_geomopt_preserves_program_keywords():
    engine = QCOPEngine(
        program="terachem",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 3, "gpus": 1},
        ),
    )
    captured = {}

    def _fake_compute_func(program, inp_obj, **kwargs):
        captured["program"] = program
        captured["inputs"] = inp_obj
        return [SimpleNamespace(), SimpleNamespace()]

    engine.compute_func = _fake_compute_func
    _ = engine.compute_geometry_optimizations([_node_at_x(1.0), _node_at_x(1.2)])

    assert captured["program"] == "terachem"
    assert isinstance(captured["inputs"], list)
    assert len(captured["inputs"]) == 2
    for prog_input in captured["inputs"]:
        kw = prog_input.keywords
        assert kw["threads"] == 3
        assert kw["gpus"] == 1
        assert kw["purify"] == "no"
        assert kw["new_minimizer"] == "yes"
