from pathlib import Path

import pytest
import tomli

from neb_dynamics.scripts import main_cli


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomli.load(handle)


@pytest.mark.parametrize(
    "method,expected_method,expected_key,expected_value",
    [
        ("neb", "neb", "max_steps", 500),
        ("fneb", "fneb", "max_min_iter", 100),
        ("fsm", "fneb", "max_min_iter", 100),
        ("mlpgi", "mlpgi", "fire_conv_geolen_tol", 0.25),
    ],
)
def test_make_default_inputs_path_min_method_defaults(
    tmp_path, method, expected_method, expected_key, expected_value
):
    out_fp = tmp_path / "default_inputs.toml"

    main_cli.make_default_inputs(name=str(out_fp), path_min_method=method)

    payload = _load_toml(out_fp)
    assert payload["path_min_method"] == expected_method
    assert payload["path_min_inputs"][expected_key] == expected_value
    if expected_method == "mlpgi":
        assert payload["path_min_inputs"]["fire_conv_erelpeak_tol"] == 0.25


def test_make_default_inputs_rejects_unknown_method(tmp_path):
    out_fp = tmp_path / "default_inputs.toml"
    with pytest.raises(main_cli.typer.BadParameter):
        main_cli.make_default_inputs(name=str(out_fp), path_min_method="unknown-method")
