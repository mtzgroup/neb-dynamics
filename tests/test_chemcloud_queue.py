from pathlib import Path

import numpy as np
from qcio import Structure

from neb_dynamics.engines import qcop as qcop_module
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.nodes.node import StructureNode
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


def test_qcop_engine_chemcloud_batches_geometry_optimizations(monkeypatch):
    captured = {"calls": 0, "input_len": None}

    class _Result:
        def __init__(self, structure):
            self.input_data = type("In", (), {"structure": structure})()
            self.results = type("ResData", (), {"energy": 0.0, "gradient": [[0.0, 0.0, 0.0] for _ in structure.symbols]})()

    class _Output:
        def __init__(self, structure):
            self.results = type("Res", (), {"trajectory": [_Result(structure)]})()

    def _fake_cc_compute(*args, **kwargs):
        captured["calls"] += 1
        batch = args[1]
        captured["input_len"] = len(batch) if isinstance(batch, list) else 1
        if isinstance(batch, list):
            return [_Output(inp.structure) for inp in batch]
        return _Output(batch.structure)

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    eng = QCOPEngine(compute_program="chemcloud")
    nodes = [
        StructureNode(
            structure=Structure(
                geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
                symbols=["H", "H"],
                charge=0,
                multiplicity=1,
            )
        )
        for x in (0.8, 1.0, 1.2)
    ]

    out = eng.compute_geometry_optimizations(nodes)

    assert captured["calls"] == 1
    assert captured["input_len"] == 3
    assert len(out) == 3
