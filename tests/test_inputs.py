from neb_dynamics.inputs import RunInputs


def test_runinputs_fsm_uses_empty_path_min_inputs():
    inputs = RunInputs(path_min_method="fsm")

    assert vars(inputs.path_min_inputs) == {}


def test_runinputs_fsm_can_save_defaults(tmp_path):
    inputs = RunInputs(path_min_method="fsm")
    out_fp = tmp_path / "default_inputs.toml"

    inputs.save(out_fp)

    text = out_fp.read_text()
    assert 'path_min_method = "fsm"' in text


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
