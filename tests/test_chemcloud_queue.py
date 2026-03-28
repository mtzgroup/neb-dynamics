from pathlib import Path

import numpy as np
import pytest

from neb_dynamics.chain import Chain
from neb_dynamics.engines import qcop as qcop_module
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.inputs import RunInputs
from neb_dynamics.nodes.node import StructureNode
from qcio import ProgramOutput
from qcio import Structure
from qcio.models.inputs import ProgramArgs
from qcop.exceptions import ExternalProgramError


def _configured_queue(configured: dict) -> str | None:
    return configured.get("queue", configured.get("chemcloud_queue"))


def test_qcop_engine_chemcloud_queue_precedence_toml_over_env(monkeypatch):
    monkeypatch.setenv("CHEMCLOUD_QUEUE", "env-queue")
    captured = {}

    def _fake_cc_compute(*args, **kwargs):
        captured["queue"] = kwargs.get("queue")
        return None

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    monkeypatch.setattr(
        qcop_module,
        "cc_configure_client",
        lambda **kwargs: captured.setdefault("configured", kwargs),
    )
    eng = QCOPEngine(
        compute_program="chemcloud",
        chemcloud_queue="toml-queue",
    )
    eng.compute_func("xtb", object())
    assert captured["queue"] == "toml-queue"
    assert _configured_queue(captured["configured"]) == "toml-queue"


def test_qcop_engine_chemcloud_queue_from_env(monkeypatch):
    monkeypatch.setenv("CHEMCLOUD_QUEUE", "env-queue")
    captured = {}

    def _fake_cc_compute(*args, **kwargs):
        captured["queue"] = kwargs.get("queue")
        return None

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    monkeypatch.setattr(
        qcop_module,
        "cc_configure_client",
        lambda **kwargs: captured.setdefault("configured", kwargs),
    )
    eng = QCOPEngine(compute_program="chemcloud")
    eng.compute_func("xtb", object())
    assert captured["queue"] == "env-queue"
    assert _configured_queue(captured["configured"]) == "env-queue"


def test_qcop_engine_chemcloud_queue_defaults_to_celery(monkeypatch):
    monkeypatch.delenv("MEPD_CHEMCLOUD_QUEUE", raising=False)
    monkeypatch.delenv("CHEMCLOUD_QUEUE", raising=False)
    monkeypatch.delenv("CCQUEUE", raising=False)
    captured = {}

    def _fake_cc_compute(*args, **kwargs):
        captured["queue"] = kwargs.get("queue")
        return None

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    monkeypatch.setattr(
        qcop_module,
        "cc_configure_client",
        lambda **kwargs: captured.setdefault("configured", kwargs),
    )
    eng = QCOPEngine(compute_program="chemcloud")
    eng.compute_func("xtb", object())
    assert captured["queue"] == "celery"
    assert _configured_queue(captured["configured"]) == "celery"


def test_qcop_engine_chemcloud_schema_mismatch_raises_external_program_error(
    monkeypatch,
):
    def _sample_validation_error():
        try:
            ProgramOutput(input_data=None, success=False, results={})
        except Exception as exc:
            return exc
        raise AssertionError("Expected ProgramOutput construction to fail")

    validation_error = _sample_validation_error()

    def _fake_cc_compute(*args, **kwargs):
        raise validation_error

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    eng = QCOPEngine(compute_program="chemcloud")

    with pytest.raises(ExternalProgramError, match="ProgramOutput schema"):
        eng.compute_func("xtb", object())


def test_qcop_engine_chemcloud_retries_transient_502(monkeypatch):
    calls = {"n": 0, "slept": 0}

    class _HTTP502(Exception):
        def __init__(self):
            self.response = type("Resp", (), {"status_code": 502})()
            super().__init__("502 Bad Gateway")

    def _fake_cc_compute(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _HTTP502()
        return "ok"

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    monkeypatch.setattr(
        qcop_module.time,
        "sleep",
        lambda _: calls.__setitem__("slept", calls["slept"] + 1),
    )
    eng = QCOPEngine(compute_program="chemcloud")

    out = eng._chemcloud_compute_with_retries("xtb", object())
    assert out == "ok"
    assert calls["n"] == 2
    assert calls["slept"] == 1


def test_qcop_engine_chemcloud_no_retry_non_retryable(monkeypatch):
    calls = {"n": 0}

    class _HTTP400(Exception):
        def __init__(self):
            self.response = type("Resp", (), {"status_code": 400})()
            super().__init__("400 Bad Request")

    def _fake_cc_compute(*args, **kwargs):
        calls["n"] += 1
        raise _HTTP400()

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    monkeypatch.setattr(qcop_module.time, "sleep", lambda _: None)
    eng = QCOPEngine(compute_program="chemcloud")

    with pytest.raises(Exception, match="400"):
        eng._chemcloud_compute_with_retries("xtb", object())
    assert calls["n"] == 1


def test_qcop_engine_chemcloud_retries_connect_error_message(monkeypatch):
    calls = {"n": 0, "slept": 0}

    class _ConnectError(Exception):
        pass

    def _fake_cc_compute(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _ConnectError("All connection attempts failed")
        return "ok"

    monkeypatch.setattr(qcop_module, "cc_compute", _fake_cc_compute)
    monkeypatch.setattr(
        qcop_module.time,
        "sleep",
        lambda _: calls.__setitem__("slept", calls["slept"] + 1),
    )
    eng = QCOPEngine(compute_program="chemcloud")

    out = eng._chemcloud_compute_with_retries("xtb", object())
    assert out == "ok"
    assert calls["n"] == 2
    assert calls["slept"] == 1


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
            self.results = type(
                "ResData",
                (),
                {
                    "energy": 0.0,
                    "gradient": [[0.0, 0.0, 0.0] for _ in structure.symbols],
                },
            )()

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


def test_qcop_engine_chemcloud_batches_geometry_optimizations_with_file_only_trajectory_entries(monkeypatch):
    class _Files:
        def model_dump(self):
            return {"optim.xyz": "unused"}

    class _Result:
        def __init__(self, structure):
            self.input_data = type("In", (), {"structure": structure})()
            self.results = _Files()

    class _Output:
        def __init__(self, structure):
            self.results = type("Res", (), {"trajectory": [_Result(structure)]})()

    def _fake_cc_compute(*args, **kwargs):
        batch = args[1]
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
        for x in (0.8, 1.0)
    ]

    out = eng.compute_geometry_optimizations(nodes)

    assert len(out) == 2
    assert out[0][-1]._cached_energy is None
    assert out[0][-1]._cached_gradient is None


def test_runinputs_toml_write_qcio_propagates(tmp_path: Path):
    inputs_fp = tmp_path / "inputs.toml"
    inputs_fp.write_text(
        "\n".join(
            [
                'engine_name = "chemcloud"',
                'program = "xtb"',
                "write_qcio = true",
            ]
        )
    )
    run_inputs = RunInputs.open(inputs_fp)
    assert run_inputs.write_qcio is True
    assert run_inputs.engine.write_qcio is True


def test_qcop_engine_warns_when_write_qcio_enabled(monkeypatch):
    calls = []

    monkeypatch.setattr(
        qcop_module.logging,
        "warning",
        lambda message, *args, **kwargs: calls.append(
            message % args if args else message
        ),
    )

    eng = QCOPEngine(write_qcio=True)

    assert eng.write_qcio is True
    assert any("write_qcio=True" in message for message in calls)


def test_qcop_engine_chemcloud_run_calc_submits_gradient_batch(
    monkeypatch,
):
    calls = []

    def _fake_update_node_cache(node_list, results):
        return None

    def _fake_compute_func(program, prog_inp, collect_files=False):
        calls.append(prog_inp)
        if isinstance(prog_inp, list):
            return [object() for _ in prog_inp]
        return object()

    monkeypatch.setattr(qcop_module, "update_node_cache", _fake_update_node_cache)

    eng = QCOPEngine(
        compute_program="chemcloud",
        program="crest",
        program_args=ProgramArgs(
            model={"method": "gfn2", "basis": "gfn2"},
            keywords={"threads": 1},
        ),
    )
    monkeypatch.setattr(eng, "compute_func", _fake_compute_func)

    def _node(x):
        node = StructureNode(
            structure=Structure(
                geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
                symbols=["H", "H"],
                charge=0,
                multiplicity=1,
            )
        )
        node.has_molecular_graph = False
        node.graph = None
        return node

    chain = Chain.model_validate({"nodes": [_node(0.8), _node(1.0), _node(1.2)]})
    eng._run_calc(chain=chain, calctype="gradient")

    assert len(calls) == 1
    assert isinstance(calls[0], list)
    assert len(calls[0]) == 3
