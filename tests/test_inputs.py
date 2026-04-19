from neb_dynamics.inputs import RunInputs
from neb_dynamics.pathminimizers.mlpgi import _resolve_optimizer_config_values
import pytest


def test_runinputs_fsm_uses_empty_path_min_inputs():
    inputs = RunInputs(path_min_method="fsm")

    assert vars(inputs.path_min_inputs) == {}


def test_runinputs_fsm_can_save_defaults(tmp_path):
    inputs = RunInputs(path_min_method="fsm")
    out_fp = tmp_path / "default_inputs.toml"

    inputs.save(out_fp)

    text = out_fp.read_text()
    assert 'path_min_method = "fsm"' in text


def test_runinputs_neb_dlf_has_expected_defaults():
    inputs = RunInputs(
        engine_name="qcop",
        program="terachem",
        path_min_method="neb-dlf",
    )

    defaults = vars(inputs.path_min_inputs)
    assert defaults["nstep"] == 200
    assert defaults["min_nebk"] == 0.01
    assert defaults["do_elem_step_checks"] is True
    assert defaults["skip_identical_graphs"] is True
    assert defaults["collect_files"] is True
    assert isinstance(defaults["dlfind_keywords"], dict)


def test_runinputs_mlpgi_has_expected_defaults():
    inputs = RunInputs(path_min_method="mlpgi")

    defaults = vars(inputs.path_min_inputs)
    assert defaults["backend"] == "fairchem"
    assert defaults["fire_stage1_iter"] == 200
    assert defaults["fire_stage2_iter"] == 500
    assert defaults["variance_penalty_weight"] == pytest.approx(0.0433641)
    assert defaults["fire_conv_geolen_tol"] == pytest.approx(0.25)
    assert defaults["fire_conv_erelpeak_tol"] == pytest.approx(0.25)
    assert defaults["refinement_step_interval"] == 10
    assert defaults["refinement_dynamic_threshold_fraction"] == pytest.approx(0.1)
    assert defaults["do_elem_step_checks"] is True
    assert defaults["skip_identical_graphs"] is True


def test_mlpgi_optimizer_fire_conv_tolerances_use_kcal_input_units():
    cfg = _resolve_optimizer_config_values(
        {
            "fire_conv_geolen_tol": 0.25,
            "fire_conv_erelpeak_tol": 0.25,
        }
    )

    assert cfg["fire_conv_geolen_tol"] == pytest.approx(0.010841025)
    assert cfg["fire_conv_erelpeak_tol"] == pytest.approx(0.010841025)


def test_mlpgi_optimizer_aliases_map_to_config_values():
    cfg = _resolve_optimizer_config_values(
        {
            "beta": 1.0,
            "tau_refine": 8,
            "cutoff": 10,
            "convergence_window": 12,
            "path_length_tolerance": 0.25,
            "barrier_height_tolerance": 0.25,
        }
    )

    assert cfg["variance_penalty_weight"] == pytest.approx(0.0433641)
    assert cfg["refinement_step_interval"] == 8
    assert cfg["refinement_dynamic_threshold_fraction"] == pytest.approx(0.1)
    assert cfg["fire_conv_window"] == 12
    assert cfg["fire_conv_geolen_tol"] == pytest.approx(0.010841025)
    assert cfg["fire_conv_erelpeak_tol"] == pytest.approx(0.010841025)


def test_runinputs_ase_omol25_reports_missing_fairchem(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "fairchem.core":
            raise ModuleNotFoundError("No module named 'fairchem'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    try:
        RunInputs(
            engine_name="ase",
            program="omol25",
            program_kwds={},
            path_min_method="fsm",
        )
    except ModuleNotFoundError as exc:
        msg = str(exc)
        assert "fairchem-core" in msg
        assert "Python 3.14" in msg
    else:
        raise AssertionError("Expected ModuleNotFoundError when fairchem.core is unavailable")


def test_runinputs_ase_omol25_uses_configured_model_path_and_device(monkeypatch):
    import sys
    import types

    calls = {}

    def _fake_load_predict_unit(model_path, device):
        calls["model_path"] = model_path
        calls["device"] = device
        return "predictor"

    class _FakeFAIRChemCalculator:
        def __init__(self, predictor, task_name):
            calls["predictor"] = predictor
            calls["task_name"] = task_name

    fairchem_core = types.ModuleType("fairchem.core")
    fairchem_core.pretrained_mlip = types.SimpleNamespace(
        load_predict_unit=_fake_load_predict_unit
    )
    fairchem_core.FAIRChemCalculator = _FakeFAIRChemCalculator

    fairchem_root = types.ModuleType("fairchem")
    fairchem_root.core = fairchem_core

    monkeypatch.setitem(sys.modules, "fairchem", fairchem_root)
    monkeypatch.setitem(sys.modules, "fairchem.core", fairchem_core)

    run_inputs = RunInputs(
        engine_name="ase",
        program="omol25",
        program_kwds={},
        path_min_method="fsm",
        path_min_inputs={
            "model_path": "/tmp/custom_omol25_checkpoint.pt",
            "device": "cpu",
        },
    )

    assert run_inputs.engine.__class__.__name__ == "ASEEngine"
    assert calls["model_path"] == "/tmp/custom_omol25_checkpoint.pt"
    assert calls["device"] == "cpu"
    assert calls["predictor"] == "predictor"
    assert calls["task_name"] == "omol"


def test_runinputs_ase_omol25_raises_with_model_path_context(monkeypatch):
    import sys
    import types

    def _fake_load_predict_unit(model_path, device):
        raise RuntimeError("load failed")

    class _FakeFAIRChemCalculator:
        def __init__(self, predictor, task_name):
            pass

    fairchem_core = types.ModuleType("fairchem.core")
    fairchem_core.pretrained_mlip = types.SimpleNamespace(
        load_predict_unit=_fake_load_predict_unit
    )
    fairchem_core.FAIRChemCalculator = _FakeFAIRChemCalculator

    fairchem_root = types.ModuleType("fairchem")
    fairchem_root.core = fairchem_core

    monkeypatch.setitem(sys.modules, "fairchem", fairchem_root)
    monkeypatch.setitem(sys.modules, "fairchem.core", fairchem_core)

    with pytest.raises(RuntimeError) as excinfo:
        RunInputs(
            engine_name="ase",
            program="omol25",
            program_kwds={},
            path_min_method="fsm",
            path_min_inputs={"model_path": "/tmp/missing.pt", "device": "cpu"},
        )

    msg = str(excinfo.value)
    assert "Failed to load OMol25 model for ASE engine" in msg
    assert "model_path='/tmp/missing.pt'" in msg
    assert "device='cpu'" in msg
