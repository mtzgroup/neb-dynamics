from pathlib import Path

from neb_dynamics.engines import qcop as qcop_module
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.inputs import RunInputs


def test_qcop_engine_chemcloud_queue_precedence_toml_over_env(monkeypatch):
    monkeypatch.setenv("CHEMCLOUD_QUEUE", "env-queue")
    captured = {}

    def _fake_cc_compute(*args, **kwargs):
        captured["queue"] = kwargs.get("queue")
        return None

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    eng = QCOPEngine(
        compute_program="chemcloud",
        chemcloud_queue="toml-queue",
    )
    eng.compute_func("xtb", object())
    assert captured["queue"] == "toml-queue"


def test_qcop_engine_chemcloud_queue_from_env(monkeypatch):
    monkeypatch.setenv("CHEMCLOUD_QUEUE", "env-queue")
    captured = {}

    def _fake_cc_compute(*args, **kwargs):
        captured["queue"] = kwargs.get("queue")
        return None

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    eng = QCOPEngine(compute_program="chemcloud")
    eng.compute_func("xtb", object())
    assert captured["queue"] == "env-queue"


def test_qcop_engine_chemcloud_queue_defaults_to_celery(monkeypatch):
    monkeypatch.delenv("MEPD_CHEMCLOUD_QUEUE", raising=False)
    monkeypatch.delenv("CHEMCLOUD_QUEUE", raising=False)
    monkeypatch.delenv("CCQUEUE", raising=False)
    captured = {}

    def _fake_cc_compute(*args, **kwargs):
        captured["queue"] = kwargs.get("queue")
        return None

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    eng = QCOPEngine(compute_program="chemcloud")
    eng.compute_func("xtb", object())
    assert captured["queue"] == "celery"


def test_runinputs_toml_chemcloud_queue_propagates(tmp_path: Path):
    inputs_fp = tmp_path / "inputs.toml"
    inputs_fp.write_text(
        "\n".join(
            [
                'engine_name = "chemcloud"',
                'program = "xtb"',
                'chemcloud_queue = "toml-queue"',
            ]
        )
    )
    run_inputs = RunInputs.open(inputs_fp)
    assert run_inputs.engine.chemcloud_queue == "toml-queue"
