from types import SimpleNamespace

import pytest
from qcio import ProgramArgs, Structure

from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.nodes.node import StructureNode


def _single_atom_node() -> StructureNode:
    structure = Structure(
        symbols=["H"],
        geometry=[[0.0, 0.0, 0.0]],
        charge=0,
        multiplicity=1,
    )
    return StructureNode(structure=structure)


def test_qcop_engine_runs_crest_msreact_nanoreactor(monkeypatch):
    engine = QCOPEngine(
        program_args=ProgramArgs(model={"method": "gfn2", "basis": "gfn2"}, keywords={}),
        program="crest",
        compute_program="chemcloud",
    )
    calls = {}
    products_xyz = """1
0.0
H 0.0 0.0 0.0
1
-1.0
H 0.0 0.0 0.5
"""

    def _fake_compute(program, inp, collect_files=False):
        calls["program"] = program
        calls["input"] = inp
        calls["collect_files"] = collect_files
        return SimpleNamespace(data=SimpleNamespace(files={"crest_msreact_products.xyz": products_xyz}))

    monkeypatch.setattr(engine, "compute_func", _fake_compute)

    candidates = engine.compute_nanoreactor_candidates(_single_atom_node(), nanoreactor_inputs={"max_candidates": 1})

    assert calls["program"] == "crest"
    assert "-msreact" in calls["input"].cmdline_args
    assert calls["collect_files"] is True
    assert len(candidates) == 1


def test_qcop_engine_runs_terachem_md_nanoreactor(monkeypatch):
    engine = QCOPEngine(
        program_args=ProgramArgs(model={"method": "ub3lyp", "basis": "3-21g"}, keywords={}),
        program="terachem",
        compute_program="chemcloud",
    )
    calls = {}
    trajectory_xyz = """1
frame0
H 0.0 0.0 0.0
1
frame1
H 0.0 0.0 0.2
1
frame2
H 0.0 0.0 0.4
"""
    log_text = """0\t0.0
1\t-1.0
2\t0.0
"""

    def _fake_compute(program, inp, collect_files=False):
        calls["program"] = program
        calls["input"] = inp
        return SimpleNamespace(
            data=SimpleNamespace(
                files={
                    "scr/coors.xyz": trajectory_xyz,
                    "scr/log.xls": log_text,
                }
            )
        )

    monkeypatch.setattr(engine, "compute_func", _fake_compute)

    candidates = engine.compute_nanoreactor_candidates(
        _single_atom_node(),
        nanoreactor_inputs={"max_candidates": 3, "terachem_frame_stride": 1},
    )

    assert calls["program"] == "terachem"
    assert calls["input"].cmdline_args == ["tc.in"]
    assert len(candidates) == 1
    tcin = calls["input"].files["tc.in"]
    assert "nstep                2000" in tcin
    assert "mdbc_t1" in tcin and "750" in tcin
    assert "mdbc_t2" in tcin and "250" in tcin


def test_qcop_engine_applies_fast_oscillating_terachem_preset(monkeypatch):
    engine = QCOPEngine(
        program_args=ProgramArgs(model={"method": "ub3lyp", "basis": "3-21g"}, keywords={}),
        program="terachem",
        compute_program="chemcloud",
    )
    calls = {}
    trajectory_xyz = """1
frame0
H 0.0 0.0 0.0
"""

    def _fake_compute(program, inp, collect_files=False):
        calls["program"] = program
        calls["input"] = inp
        return SimpleNamespace(data=SimpleNamespace(files={"scr/coors.xyz": trajectory_xyz}))

    monkeypatch.setattr(engine, "compute_func", _fake_compute)

    engine.compute_nanoreactor_candidates(
        _single_atom_node(),
        nanoreactor_inputs={"preset": "fast-oscillating"},
    )

    tcin = calls["input"].files["tc.in"]
    assert "nstep                1600" in tcin
    assert "md_r1                5.8" in tcin
    assert "md_r2                3.6" in tcin
    assert "md_k1                3.5" in tcin
    assert "md_k2                6.0" in tcin
    assert "mdbc_t1" in tcin and "300" in tcin
    assert "mdbc_t2" in tcin and "100" in tcin


def test_qcop_engine_rejects_unknown_terachem_nanoreactor_preset():
    engine = QCOPEngine(
        program_args=ProgramArgs(model={"method": "ub3lyp", "basis": "3-21g"}, keywords={}),
        program="terachem",
        compute_program="chemcloud",
    )

    with pytest.raises(ValueError, match="Unsupported nanoreactor preset"):
        engine.compute_nanoreactor_candidates(
            _single_atom_node(),
            nanoreactor_inputs={"preset": "not-a-real-preset"},
        )


def test_qcop_engine_geometry_optimization_falls_back_to_optim_xyz_files(monkeypatch):
    engine = QCOPEngine(
        program_args=ProgramArgs(model={"method": "ub3lyp", "basis": "3-21g"}, keywords={}),
        program="terachem",
        compute_program="chemcloud",
    )
    optim_xyz = """1
frame0
H 0.0 0.0 0.0
1
frame1
H 0.0 0.0 0.2
"""

    class _Files:
        def model_dump(self):
            return {"optim.xyz": optim_xyz}

    def _fake_geom_opt_result(node, keywords=None):
        return SimpleNamespace(
            data=SimpleNamespace(files=_Files()),
            results=SimpleNamespace(files=_Files()),
        )

    monkeypatch.setattr(engine, "_compute_geom_opt_result", _fake_geom_opt_result)

    trajectory = engine.compute_geometry_optimization(_single_atom_node())

    assert len(trajectory) == 2
    assert float(trajectory[-1].structure.geometry[0][2]) != float(trajectory[0].structure.geometry[0][2])
