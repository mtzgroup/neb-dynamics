import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qcio import Structure

from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.scripts import main_cli


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]], dtype=float),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def _node_at_x(x: float) -> StructureNode:
    node = StructureNode(structure=_structure_at_x(x))
    node._cached_energy = float(x)
    node._cached_gradient = np.zeros_like(node.coords)
    return node


class _FakeHessianResult:
    def __init__(self, modes, freqs):
        self.results = SimpleNamespace(
            normal_modes_cartesian=modes,
            freqs_wavenumber=freqs,
        )

    def save(self, fp):
        Path(fp).write_text("fake hessian")


def test_hessian_sample_clips_and_dedupes(monkeypatch, tmp_path):
    class _FakeEngine:
        def __init__(self):
            self.opt_calls = 0

        def _compute_hessian_result(self, node):
            modes = [
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
                np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            ]
            freqs = [-100.0, 50.0, 75.0]
            return _FakeHessianResult(modes=modes, freqs=freqs)

        def compute_geometry_optimization(self, node, keywords=None):
            outputs = [0.5, 0.5, 1.5, 1.5]
            out = _node_at_x(outputs[self.opt_calls])
            self.opt_calls += 1
            return [out]

    fake_engine = _FakeEngine()
    fake_run_inputs = SimpleNamespace(
        engine=fake_engine,
        chain_inputs=ChainInputs(node_rms_thre=5.0, node_ene_thre=5.0),
        write_qcio=False,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda _fp: fake_run_inputs))
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli.Structure, "open", staticmethod(lambda _fp: _structure_at_x(0.0)))
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    main_cli.hessian_sample(
        geometry="seed.xyz",
        inputs="dummy.toml",
        name="sample",
        dr=0.2,
        max_candidates=4,
    )

    summary_fp = tmp_path / "sample_hessian_sample_summary.json"
    unique_fp = tmp_path / "sample_hessian_sample_unique.xyz"
    assert summary_fp.exists()
    assert unique_fp.exists()
    assert (tmp_path / "sample_hessian_sample_displaced.xyz").exists()
    assert (tmp_path / "sample_hessian_sample_optimized.xyz").exists()
    assert (tmp_path / "sample_hessian.qcio").exists()

    payload = json.loads(summary_fp.read_text())
    assert payload["normal_modes_total"] == 3
    assert payload["displaced_candidates"] == 4
    assert payload["optimized_candidates"] == 4
    assert payload["failed_candidates"] == 0
    assert payload["unique_minima"] == 2
    assert fake_engine.opt_calls == 4

    unique_structures = main_cli.read_multiple_structure_from_file(
        unique_fp, charge=0, spinmult=1
    )
    assert len(unique_structures) == 2


def test_hessian_sample_uses_parse_nma_fallback(monkeypatch, tmp_path):
    called = {"parse": 0}

    class _FakeEngine:
        def __init__(self):
            self.opt_calls = 0

        def _compute_hessian_result(self, node):
            return _FakeHessianResult(modes=[], freqs=[])

        def compute_geometry_optimization(self, node, keywords=None):
            outputs = [0.2, 0.8]
            out = _node_at_x(outputs[self.opt_calls])
            self.opt_calls += 1
            return [out]

    def _fake_parse_nma_freq_data(_hessres):
        called["parse"] += 1
        mode = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        return [mode], [-123.4]

    fake_engine = _FakeEngine()
    fake_run_inputs = SimpleNamespace(
        engine=fake_engine,
        chain_inputs=ChainInputs(node_rms_thre=5.0, node_ene_thre=5.0),
        write_qcio=False,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda _fp: fake_run_inputs))
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli.Structure, "open", staticmethod(lambda _fp: _structure_at_x(0.0)))
    monkeypatch.setattr(main_cli, "parse_nma_freq_data", _fake_parse_nma_freq_data)
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    main_cli.hessian_sample(
        geometry="seed.xyz",
        inputs="dummy.toml",
        name="fallback",
        max_candidates=2,
    )

    payload = json.loads((tmp_path / "fallback_hessian_sample_summary.json").read_text())
    assert called["parse"] == 1
    assert payload["normal_modes_total"] == 1
    assert payload["displaced_candidates"] == 2
    assert payload["optimized_candidates"] == 2


def test_hessian_sample_uses_chemcloud_batch_not_serial(monkeypatch, tmp_path):
    calls = {"batch": 0, "serial": 0}

    class _FakeEngine:
        compute_program = "chemcloud"

        def _compute_hessian_result(self, node):
            mode = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
            return _FakeHessianResult(modes=[mode], freqs=[-99.0])

        def compute_geometry_optimizations(self, nodes, keywords=None):
            calls["batch"] += 1
            assert len(nodes) == 2
            return [[_node_at_x(0.3)], [_node_at_x(0.7)]]

        def compute_geometry_optimization(self, node, keywords=None):
            calls["serial"] += 1
            raise AssertionError("Serial optimizer should not be used for ChemCloud.")

    fake_run_inputs = SimpleNamespace(
        engine=_FakeEngine(),
        engine_name="chemcloud",
        chain_inputs=ChainInputs(node_rms_thre=5.0, node_ene_thre=5.0),
        write_qcio=False,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda _fp: fake_run_inputs))
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli.Structure, "open", staticmethod(lambda _fp: _structure_at_x(0.0)))
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    main_cli.hessian_sample(
        geometry="seed.xyz",
        inputs="dummy.toml",
        name="chemcloud",
        max_candidates=2,
    )

    payload = json.loads((tmp_path / "chemcloud_hessian_sample_summary.json").read_text())
    assert payload["displaced_candidates"] == 2
    assert payload["optimized_candidates"] == 2
    assert payload["optimization_submission_mode"] == "chemcloud_batch"
    assert calls["batch"] == 1
    assert calls["serial"] == 0


def test_hessian_sample_refuses_chemcloud_engine_name_with_non_chemcloud_compute_program(
    monkeypatch, tmp_path
):
    class _FakeEngine:
        compute_program = "qcop"

        def _compute_hessian_result(self, node):
            mode = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
            return _FakeHessianResult(modes=[mode], freqs=[-99.0])

        def compute_geometry_optimizations(self, nodes, keywords=None):
            raise AssertionError("Should fail before batch submission when compute_program is mismatched.")

    fake_run_inputs = SimpleNamespace(
        engine=_FakeEngine(),
        engine_name="chemcloud",
        chain_inputs=ChainInputs(node_rms_thre=5.0, node_ene_thre=5.0),
        write_qcio=False,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda _fp: fake_run_inputs))
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli.Structure, "open", staticmethod(lambda _fp: _structure_at_x(0.0)))
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    import pytest

    with pytest.raises(main_cli.typer.Exit):
        main_cli.hessian_sample(
            geometry="seed.xyz",
            inputs="dummy.toml",
            name="mismatch",
            max_candidates=2,
        )


def test_hessian_sample_supports_ase_omol25_with_compute_hessian_fallback(monkeypatch, tmp_path):
    class _FakeASEEngine:
        def compute_hessian(self, node):
            ndof = int(np.asarray(node.coords).size)
            return np.eye(ndof, dtype=float)

        def compute_geometry_optimization(self, node, keywords=None):
            return [_node_at_x(0.4)]

    fake_run_inputs = SimpleNamespace(
        engine=_FakeASEEngine(),
        engine_name="ase",
        program="omol25",
        chain_inputs=ChainInputs(node_rms_thre=5.0, node_ene_thre=5.0),
        write_qcio=False,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda _fp: fake_run_inputs))
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli.Structure, "open", staticmethod(lambda _fp: _structure_at_x(0.0)))
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    main_cli.hessian_sample(
        geometry="seed.xyz",
        inputs="dummy.toml",
        name="ase_omol25",
        max_candidates=2,
    )

    payload = json.loads((tmp_path / "ase_omol25_hessian_sample_summary.json").read_text())
    assert payload["normal_modes_total"] == 1
    assert payload["displaced_candidates"] == 2
    assert payload["optimized_candidates"] == 2
