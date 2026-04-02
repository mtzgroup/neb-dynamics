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
