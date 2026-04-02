import os
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import logging

import numpy as np
from qcio import Structure

import neb_dynamics.engines.qmmm as qmmm_module
from neb_dynamics.constants import ANGSTROM_TO_BOHR
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


def test_qmmm_engine_chemcloud_retries_transient_502(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    calls = {"n": 0, "slept": 0}

    class _HTTP502(Exception):
        def __init__(self):
            self.response = SimpleNamespace(status_code=502)
            super().__init__("502 Bad Gateway")

    def _fake_ccompute(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _HTTP502()
        return [type("Out", (), {"stdout": ""})()]

    monkeypatch.setattr(qmmm_module, "ccompute", _fake_ccompute)
    monkeypatch.setattr(qmmm_module.time, "sleep", lambda _: calls.__setitem__("slept", calls["slept"] + 1))

    eng = QMMMEngine(
        tcin_text="run gradient\n",
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="chemcloud",
    )

    out = eng._chemcloud_compute_with_retries(inp="dummy", collect_files=False)
    assert calls["n"] == 2
    assert calls["slept"] == 1
    assert isinstance(out, list)


def test_qmmm_engine_chemcloud_no_retry_non_retryable(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    calls = {"n": 0}

    class _HTTP400(Exception):
        def __init__(self):
            self.response = SimpleNamespace(status_code=400)
            super().__init__("400 Bad Request")

    def _fake_ccompute(*args, **kwargs):
        calls["n"] += 1
        raise _HTTP400()

    monkeypatch.setattr(qmmm_module, "ccompute", _fake_ccompute)
    monkeypatch.setattr(qmmm_module.time, "sleep", lambda _: None)

    eng = QMMMEngine(
        tcin_text="run gradient\n",
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="chemcloud",
    )

    try:
        eng._chemcloud_compute_with_retries(inp="dummy", collect_files=False)
        assert False, "Expected non-retryable exception to be raised."
    except Exception as exc:
        assert "400" in str(exc)
    assert calls["n"] == 1


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


def test_runinputs_qmmm_uses_chain_frozen_indices_by_default(tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    inputs_fp = tmp_path / "qmmm_inputs.toml"
    inputs_fp.write_text(
        "\n".join(
            [
                'engine_name = "qmmm"',
                'program = "terachem"',
                "",
                "[chain_inputs]",
                'frozen_atom_indices = "0 1 2"',
                "",
                "[qmmm_inputs]",
                'qminds_fp = "qmindices.dat"',
                'prmtop_fp = "ref.prmtop"',
                'rst7_fp_react = "ref.rst7"',
                'compute_program = "qcop"',
            ]
        )
    )
    run_inputs = RunInputs.open(inputs_fp)
    assert run_inputs.engine.frozen_atom_indices == [0, 1, 2]


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


def test_qmmm_geometry_optimization_uses_minimize_and_parses_trajectory(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    captured = {}

    class _FakeOut:
        def __init__(self, files):
            self.files = files
            self.stdout = "ok"

    def _fake_compute(*args, **kwargs):
        fi = args[1]
        captured["tcin"] = fi.files["tc.in"]
        captured["kwargs"] = kwargs
        optim_xyz = (
            "1\n"
            "frame0\n"
            "H 0.000000 0.000000 0.000000\n"
            "1\n"
            "frame1\n"
            "H 0.100000 0.000000 0.000000\n"
        )
        return _FakeOut(files={"optim.xyz": optim_xyz})

    monkeypatch.setattr(qmmm_module, "compute", _fake_compute)
    eng = QMMMEngine(
        tcin_text="\n".join(
            [
                "# Inputs",
                "prmtop ref.prmtop",
                "coordinates ref.rst7",
                "qmindices qmindices.dat",
                "charge 0",
                "spinmult 1",
                "",
                "# Runtype",
                "run gradient",
                "",
                "# Method",
                "basis 6-31g**",
                "method b3lyp",
                "",
            ]
        ),
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="qcop",
    )
    node = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0]]),
            symbols=["H"],
            charge=0,
            multiplicity=1,
        )
    )

    traj = eng.compute_geometry_optimization(node=node, keywords={"maxit": 25})

    assert len(traj) == 2
    assert "run minimize" in captured["tcin"]
    assert "basis 6-31g**" in captured["tcin"]
    assert "method b3lyp" in captured["tcin"]
    assert "maxit 25" in captured["tcin"]
    assert captured["kwargs"]["collect_files"] is True


def test_qmmm_geometry_optimization_normalizes_keyword_aliases(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    captured = {}

    class _FakeOut:
        def __init__(self, files):
            self.files = files
            self.stdout = "ok"

    def _fake_compute(*args, **kwargs):
        fi = args[1]
        captured["tcin"] = fi.files["tc.in"]
        optim_xyz = (
            "1\n"
            "frame0\n"
            "H 0.000000 0.000000 0.000000\n"
            "1\n"
            "frame1\n"
            "H 0.100000 0.000000 0.000000\n"
        )
        return _FakeOut(files={"optim.xyz": optim_xyz})

    monkeypatch.setattr(qmmm_module, "compute", _fake_compute)
    eng = QMMMEngine(
        tcin_text="\n".join(
            [
                "# Inputs",
                "prmtop ref.prmtop",
                "coordinates ref.rst7",
                "qmindices qmindices.dat",
                "charge 0",
                "spinmult 1",
                "min_coordinates cartesian",
                "",
                "# Runtype",
                "run gradient",
                "",
            ]
        ),
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="qcop",
    )
    node = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0]]),
            symbols=["H"],
            charge=0,
            multiplicity=1,
        )
    )

    _ = eng.compute_geometry_optimization(node=node, keywords={"coordsys": "cart", "maxiter": 75})

    assert "run minimize" in captured["tcin"]
    assert "min_coordinates cartesian" in captured["tcin"]
    assert "maxit 75" in captured["tcin"]
    assert "coordsys cart" not in captured["tcin"]
    assert "maxiter 75" not in captured["tcin"]


def test_qmmm_geometry_optimization_forces_new_minimizer_no(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    captured = {}

    class _FakeOut:
        def __init__(self, files):
            self.files = files
            self.stdout = "ok"

    def _fake_compute(*args, **kwargs):
        fi = args[1]
        captured["tcin"] = fi.files["tc.in"]
        optim_xyz = (
            "1\n"
            "frame0\n"
            "H 0.000000 0.000000 0.000000\n"
            "1\n"
            "frame1\n"
            "H 0.100000 0.000000 0.000000\n"
        )
        return _FakeOut(files={"optim.xyz": optim_xyz})

    monkeypatch.setattr(qmmm_module, "compute", _fake_compute)
    eng = QMMMEngine(
        tcin_text="\n".join(
            [
                "# Inputs",
                "prmtop ref.prmtop",
                "coordinates ref.rst7",
                "qmindices qmindices.dat",
                "charge 0",
                "spinmult 1",
                "",
                "# Runtype",
                "run gradient",
                "new_minimizer yes",
                "",
            ]
        ),
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="qcop",
    )
    node = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0]]),
            symbols=["H"],
            charge=0,
            multiplicity=1,
        )
    )

    _ = eng.compute_geometry_optimization(node=node, keywords={"new_minimizer": "yes"})

    assert "run minimize" in captured["tcin"]
    assert "new_minimizer no" in captured["tcin"]
    assert "new_minimizer yes" not in captured["tcin"]


def test_qmmm_transition_state_returns_final_node_with_energy(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    captured = {}

    class _FakeTSOut:
        xyzs = [
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.3, 0.1, -0.2]]),
        ]

    def _fake_run_ts(self, node, keywords=None):
        captured["keywords"] = keywords
        return _FakeTSOut()

    def _fake_compute_energies(self, chain):
        captured["final_coords"] = chain[0].coords.copy()
        return np.array([-123.456])

    monkeypatch.setattr(QMMMEngine, "_run_geometric_transition_state", _fake_run_ts)
    monkeypatch.setattr(QMMMEngine, "compute_energies", _fake_compute_energies)

    eng = QMMMEngine(
        tcin_text="run gradient\n",
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="qcop",
    )
    node = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0]]),
            symbols=["H"],
            charge=0,
            multiplicity=1,
        )
    )

    ts_node = eng.compute_transition_state(node=node, keywords={"maxiter": 123})

    assert np.allclose(captured["final_coords"], np.array([[0.3, 0.1, -0.2]]) * ANGSTROM_TO_BOHR)
    assert np.allclose(ts_node.coords, np.array([[0.3, 0.1, -0.2]]) * ANGSTROM_TO_BOHR)
    assert ts_node.energy == -123.456
    assert ts_node.has_molecular_graph is False
    assert ts_node.graph is None
    assert captured["keywords"] == {"maxiter": 123}


def test_qmmm_transition_state_print_stdout_enables_geometric_logs(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    captured = {}

    geometric_mod = types.ModuleType("geometric")
    geometric_engine_mod = types.ModuleType("geometric.engine")
    geometric_molecule_mod = types.ModuleType("geometric.molecule")
    geometric_optimize_mod = types.ModuleType("geometric.optimize")

    class _FakeBaseEngine:
        def __init__(self, molecule):
            self.molecule = molecule

    class _FakeMolecule:
        def __init__(self):
            self.elem = []
            self.xyzs = []

    def _fake_run_optimizer(*args, **kwargs):
        logger = logging.getLogger("geometric.optimize")
        captured["disabled_during_run"] = logger.disabled
        captured["handlers_during_run"] = len(logger.handlers)
        return SimpleNamespace(xyzs=[np.array([[0.0, 0.0, 0.0]])])

    geometric_engine_mod.Engine = _FakeBaseEngine
    geometric_molecule_mod.Molecule = _FakeMolecule
    geometric_optimize_mod.run_optimizer = _fake_run_optimizer

    geometric_mod.engine = geometric_engine_mod
    geometric_mod.molecule = geometric_molecule_mod
    geometric_mod.optimize = geometric_optimize_mod

    monkeypatch.setitem(sys.modules, "geometric", geometric_mod)
    monkeypatch.setitem(sys.modules, "geometric.engine", geometric_engine_mod)
    monkeypatch.setitem(sys.modules, "geometric.molecule", geometric_molecule_mod)
    monkeypatch.setitem(sys.modules, "geometric.optimize", geometric_optimize_mod)

    logger = logging.getLogger("geometric.optimize")
    orig_disabled = logger.disabled
    logger.disabled = True

    eng = QMMMEngine(
        tcin_text="run gradient\n",
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="qcop",
        print_stdout=True,
    )
    node = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0]]),
            symbols=["H"],
            charge=0,
            multiplicity=1,
        )
    )

    eng._run_geometric_transition_state(node=node, keywords={"maxiter": 1})

    assert captured["disabled_during_run"] is False
    assert captured["handlers_during_run"] >= 1
    assert logging.getLogger("geometric.optimize").disabled is True
    logger.disabled = orig_disabled


def test_qmmm_transition_state_respects_frozen_atom_indices(monkeypatch, tmp_path):
    _write_minimal_qmmm_files(tmp_path)
    captured = {}

    class _FakeTSOut:
        # One active atom trajectory frame (active index will be atom 1)
        xyzs = [np.array([[0.4, 0.0, 0.0]])]
        _active_atom_indices = [1]

    def _fake_run_ts(self, node, keywords=None):
        return _FakeTSOut()

    def _fake_compute_energies(self, chain):
        captured["final_coords"] = chain[0].coords.copy()
        return np.array([-1.0])

    monkeypatch.setattr(QMMMEngine, "_run_geometric_transition_state", _fake_run_ts)
    monkeypatch.setattr(QMMMEngine, "compute_energies", _fake_compute_energies)

    eng = QMMMEngine(
        tcin_text="run gradient\n",
        qminds_fp=tmp_path / "qmindices.dat",
        prmtop_fp=tmp_path / "ref.prmtop",
        rst7_fp_react=tmp_path / "ref.rst7",
        compute_program="qcop",
        frozen_atom_indices=[0],
    )
    node = StructureNode(
        structure=Structure(
            geometry=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            symbols=["H", "H"],
            charge=0,
            multiplicity=1,
        )
    )

    ts_node = eng.compute_transition_state(node=node, keywords=None)

    # atom 0 frozen, atom 1 updated from active trajectory frame
    expected = np.array(node.coords, dtype=float).copy()
    expected[1] = np.array([0.4, 0.0, 0.0]) * ANGSTROM_TO_BOHR
    assert np.allclose(ts_node.coords, expected)
    assert np.allclose(captured["final_coords"], expected)
