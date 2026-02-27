import os
from pathlib import Path

import neb_dynamics.engines.qmmm as qmmm_module
from neb_dynamics.engines.qmmm import QMMMEngine
from neb_dynamics.inputs import RunInputs


def _write_minimal_qmmm_files(tmp_path: Path):
    (tmp_path / "qmindices.dat").write_text("0\n")
    (tmp_path / "ref.prmtop").write_text("%FLAG ATOMIC_NUMBER\n%FORMAT(10I8)\n       1\n%FLAG END\n")
    (tmp_path / "ref.rst7").write_text("TITLE\n1\n 0.0000000  0.0000000  0.0000000\n")


def test_qmmm_engine_chemcloud_queue_from_env(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    monkeypatch.setenv("CHEMCLOUD_QUEUE", "env-queue")
    captured = {}

    def _fake_ccompute(*args, **kwargs):
        captured["queue"] = kwargs.get("queue")
        return [type("Out", (), {"stdout": ""})()]

    monkeypatch.setattr(qmmm_module, "ccompute", _fake_ccompute)
    eng = QMMMEngine(
        tcin_text="run gradient\n",
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="chemcloud",
    )
    try:
        eng._compute_enegrad([eng.ref_rst7_react])
    except Exception:
        # Queue assertion is the point of this test; parsing is mocked.
        pass
    assert captured["queue"] == "env-queue"


def test_qmmm_engine_print_stdout_forwarded_to_qcop(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    captured = {}

    def _fake_compute(*args, **kwargs):
        captured["print_stdout"] = kwargs.get("print_stdout")
        raise RuntimeError("stop after capturing print_stdout")

    monkeypatch.setattr(qmmm_module, "compute", _fake_compute)
    eng = QMMMEngine(
        tcin_text="run gradient\n",
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="qcop",
        print_stdout=True,
    )
    try:
        eng._compute_enegrad([eng.ref_rst7_react])
    except Exception:
        pass
    assert captured["print_stdout"] is True


def test_runinputs_qmmm_toml_queue_overrides_env(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    monkeypatch.setenv("CHEMCLOUD_QUEUE", "env-queue")
    inputs_fp = tmp_path / "qmmm_inputs.toml"
    inputs_fp.write_text(
        "\n".join(
            [
                'engine_name = "qmmm"',
                'program = "terachem"',
                'chemcloud_queue = "toml-queue"',
                "",
                "[qmmm_inputs]",
                'qminds_fp = "qmindices.dat"',
                'prmtop_fp = "ref.prmtop"',
                'rst7_fp_react = "ref.rst7"',
                'compute_program = "chemcloud"',
            ]
        )
    )
    run_inputs = RunInputs.open(inputs_fp)
    assert run_inputs.engine.chemcloud_queue == "toml-queue"


def test_runinputs_qmmm_toml_print_stdout_passthrough(tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    inputs_fp = tmp_path / "qmmm_inputs.toml"
    inputs_fp.write_text(
        "\n".join(
            [
                'engine_name = "qmmm"',
                'program = "terachem"',
                "",
                "[qmmm_inputs]",
                'qminds_fp = "qmindices.dat"',
                'prmtop_fp = "ref.prmtop"',
                'rst7_fp_react = "ref.rst7"',
                'compute_program = "qcop"',
                "print_stdout = true",
            ]
        )
    )
    run_inputs = RunInputs.open(inputs_fp)
    assert run_inputs.engine.print_stdout is True
