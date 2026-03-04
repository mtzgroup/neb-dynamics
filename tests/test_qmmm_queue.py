import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qcio import Structure

import neb_dynamics.engines.qmmm as qmmm_module
from neb_dynamics.engines.qmmm import QMMMEngine
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import RunInputs
from neb_dynamics.nodes.node import StructureNode


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


def test_qmmm_engine_chemcloud_submits_batch_once(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    captured = {"calls": 0, "input_type": None, "input_len": None}

    def _fake_ccompute(*args, **kwargs):
        captured["calls"] += 1
        batch = args[1]
        captured["input_type"] = type(batch).__name__
        captured["input_len"] = len(batch) if isinstance(batch, list) else 1
        return [type("Out", (), {"stdout": ""})() for _ in range(captured["input_len"])]

    monkeypatch.setattr(qmmm_module, "ccompute", _fake_ccompute)
    eng = QMMMEngine(
        tcin_text="run gradient\n",
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="chemcloud",
    )
    try:
        eng._compute_enegrad([eng.ref_rst7_react, eng.ref_rst7_react, eng.ref_rst7_react])
    except Exception:
        # Submission shape assertion is the point of this test; parsing is mocked.
        pass

    assert captured["calls"] == 1
    assert captured["input_type"] == "list"
    assert captured["input_len"] == 3


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


def test_qmmm_engine_debug_dump_inputs_writes_selected_nodes(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)

    def _fake_compute_enegrad(self, rst7_strings):
        self._dump_debug_inputs(rst7_strings)
        n_atoms = 1
        n_nodes = len(rst7_strings)
        energies = [float(i) for i in range(n_nodes)]
        gradients = [np.zeros((n_atoms, 3)) for _ in range(n_nodes)]
        outputs = [SimpleNamespace(stdout="") for _ in range(n_nodes)]
        self._next_debug_dump_counter()
        return energies, gradients, outputs

    monkeypatch.setattr(QMMMEngine, "_compute_enegrad", _fake_compute_enegrad)

    eng = QMMMEngine(
        tcin_text="run gradient\nmethod b3lyp\n",
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="qcop",
        debug_dump_inputs=True,
        debug_dump_dir=tmp_path / "debug",
    )
    chain = Chain.model_validate(
        {
            "nodes": [
                StructureNode(
                    structure=Structure(
                        geometry=np.array([[float(i), 0.0, 0.0]]),
                        symbols=["H"],
                        charge=0,
                        multiplicity=1,
                    )
                )
                for i in range(5)
            ]
        }
    )

    eng.compute_gradients(chain)

    call_dir = tmp_path / "debug" / "call_0000"
    assert call_dir.exists()
    assert (call_dir / "README.txt").exists()
    assert (call_dir / "node_000_first.tc.in").read_text() == eng.inp_file
    assert (call_dir / "node_000_first.rst7").exists()
    assert (call_dir / "node_002_middle.tc.in").read_text() == eng.inp_file
    assert (call_dir / "node_002_middle.rst7").exists()
    assert (call_dir / "node_004_last.tc.in").read_text() == eng.inp_file
    assert (call_dir / "node_004_last.rst7").exists()


def test_runinputs_qmmm_toml_debug_dump_passthrough(tmp_path):
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
                "debug_dump_inputs = true",
                'debug_dump_dir = "debug_out"',
            ]
        )
    )
    run_inputs = RunInputs.open(inputs_fp)
    assert run_inputs.engine.debug_dump_inputs is True
    assert run_inputs.engine.debug_dump_dir == (tmp_path / "debug_out").resolve()
