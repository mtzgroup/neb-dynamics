import sys
import types
from pathlib import Path

from neb_dynamics.inputs import RunInputs


def test_qmmm_runinputs_open_resolves_relative_paths(tmp_path, monkeypatch):
    for name in ["qmindices.dat", "ref.prmtop", "ref.rst7"]:
        (tmp_path / name).write_text("dummy")

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
                'compute_program = "chemcloud"',
                "print_stdout = true",
                "charge = 0",
                "spinmult = 1",
                "",
                "[program_kwds]",
                "cmdline_args = []",
                "",
                "[program_kwds.keywords]",
                "gpus = 1",
                "sphericalbasis = false",
                "",
                "[program_kwds.model]",
                'method = "b3lyp"',
                'basis = "6-31g**"',
            ]
        )
    )

    class FakeQMMMEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(QMMMEngine=FakeQMMMEngine)
    monkeypatch.setitem(sys.modules, "neb_dynamics.engines.qmmm", fake_module)

    run_inputs = RunInputs.open(inputs_fp)

    assert isinstance(run_inputs.engine, FakeQMMMEngine)
    assert run_inputs.engine.kwargs["qminds_fp"] == Path(tmp_path / "qmindices.dat")
    assert run_inputs.engine.kwargs["prmtop_fp"] == Path(tmp_path / "ref.prmtop")
    assert run_inputs.engine.kwargs["rst7_fp_react"] == Path(tmp_path / "ref.rst7")
    assert run_inputs.engine.kwargs["compute_program"] == "chemcloud"
    assert run_inputs.engine.kwargs["print_stdout"] is True
    tcin_text = run_inputs.engine.kwargs["tcin_text"]
    assert "method b3lyp" in tcin_text
    assert "basis 6-31g**" in tcin_text
    assert "qmindices qmindices.dat" in tcin_text
    assert "prmtop ref.prmtop" in tcin_text
    assert "sphericalbasis no" in tcin_text


def test_qmmm_runinputs_legacy_tcin_fp_and_rst7_prod_passthrough(tmp_path, monkeypatch):
    for name in ["tc.in", "qmindices.dat", "ref.prmtop", "ref.rst7", "optim_product.rst7"]:
        (tmp_path / name).write_text("dummy")

    inputs_fp = tmp_path / "qmmm_legacy_inputs.toml"
    inputs_fp.write_text(
        "\n".join(
            [
                'engine_name = "qmmm"',
                'program = "terachem"',
                "",
                "[qmmm_inputs]",
                'tcin_fp = "tc.in"',
                'qminds_fp = "qmindices.dat"',
                'prmtop_fp = "ref.prmtop"',
                'rst7_fp_react = "ref.rst7"',
                'rst7_fp_prod = "optim_product.rst7"',
                'compute_program = "chemcloud"',
            ]
        )
    )

    class FakeQMMMEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(QMMMEngine=FakeQMMMEngine)
    monkeypatch.setitem(sys.modules, "neb_dynamics.engines.qmmm", fake_module)

    run_inputs = RunInputs.open(inputs_fp)

    assert isinstance(run_inputs.engine, FakeQMMMEngine)
    assert run_inputs.engine.kwargs["tcin_fp"] == Path(tmp_path / "tc.in")
    assert run_inputs.engine.kwargs["rst7_fp_prod"] == Path(tmp_path / "optim_product.rst7")
    assert "tcin_text" not in run_inputs.engine.kwargs
