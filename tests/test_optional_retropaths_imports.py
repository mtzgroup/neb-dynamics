import subprocess
import sys


def _run_inline(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_neb_dynamics_does_not_eagerly_import_retropaths_modules() -> None:
    result = _run_inline(
        "import sys; import neb_dynamics; "
        "assert 'neb_dynamics.retropaths_queue' not in sys.modules; "
        "assert 'neb_dynamics.retropaths_compat' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr


def test_retropaths_exports_load_lazily_from_package_namespace() -> None:
    result = _run_inline(
        "import sys; import neb_dynamics; "
        "assert 'neb_dynamics.retropaths_queue' not in sys.modules; "
        "_ = neb_dynamics.build_retropaths_neb_queue; "
        "assert 'neb_dynamics.retropaths_queue' in sys.modules"
    )
    assert result.returncode == 0, result.stderr
