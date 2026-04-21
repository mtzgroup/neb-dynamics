#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

SYSTEMS: dict[str, dict[str, Path]] = {
    "dakin": {
        "start": ROOT / "dakin" / "start_dakin_oh.xyz",
        "end": ROOT / "dakin" / "end_dakin_oh.xyz",
    },
    "oxycope": {
        "start": ROOT / "oxycope" / "start_oxycope.xyz",
        "end": ROOT / "oxycope" / "end_oxycope.xyz",
    },
    "wittig": {
        "start": ROOT / "wittig" / "start_wittig_ph.xyz",
        "end": ROOT / "wittig" / "end_wittig_ph.xyz",
    },
}

# Progressively looser than example_inputs.toml defaults.
PARAMETER_SETS: list[dict[str, Any]] = [
    {
        "label": "baseline",
        "en_thre": 0.0001,
        "rms_grad_thre": 0.001,
        "max_rms_grad_thre": 0.01,
        "ts_grad_thre": 0.01,
        "ts_spring_thre": 0.01,
        "early_stop_force_thre": 0.01,
        "negative_steps_thre": 3,
        "positive_steps_thre": 50,
        "max_steps": 500,
    },
    {
        "label": "loose_2",
        "en_thre": 0.0003,
        "rms_grad_thre": 0.002,
        "max_rms_grad_thre": 0.02,
        "ts_grad_thre": 0.015,
        "ts_spring_thre": 0.015,
        "early_stop_force_thre": 0.015,
        "negative_steps_thre": 3,
        "positive_steps_thre": 40,
        "max_steps": 400,
    },
    {
        "label": "loose_4",
        "en_thre": 0.001,
        "rms_grad_thre": 0.004,
        "max_rms_grad_thre": 0.04,
        "ts_grad_thre": 0.03,
        "ts_spring_thre": 0.03,
        "early_stop_force_thre": 0.03,
        "negative_steps_thre": 2,
        "positive_steps_thre": 30,
        "max_steps": 300,
    },
]

PATH_MIN_KEYS = {
    "en_thre",
    "rms_grad_thre",
    "max_rms_grad_thre",
    "ts_grad_thre",
    "ts_spring_thre",
    "early_stop_force_thre",
    "negative_steps_thre",
    "positive_steps_thre",
    "max_steps",
}

DEFAULTS = {
    "en_thre": 0.0001,
    "rms_grad_thre": 0.001,
    "max_rms_grad_thre": 0.01,
    "ts_grad_thre": 0.01,
    "ts_spring_thre": 0.01,
    "early_stop_force_thre": 0.01,
    "max_steps": 500,
}


@dataclass
class RunOutcome:
    param_label: str
    system: str
    run_dir: Path
    neb_returncode: int | None = None
    ts_returncode: int | None = None
    hess_returncode: int | None = None
    neb_converged: bool = False
    ts_converged: bool = False
    highest_energy_index: int = -1
    n_images: int = 0
    n_negative_frequencies: int = -1
    valid_first_order_ts: bool = False
    elapsed_seconds: float = 0.0
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.neb_converged and self.ts_converged and self.valid_first_order_ts


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _write_parametrized_input(base_toml: Path, output_toml: Path, params: dict[str, Any]) -> None:
    text = base_toml.read_text()
    lines = text.splitlines()
    in_path_min = False
    seen: set[str] = set()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_path_min = stripped == "[path_min_inputs]"
            continue

        if not in_path_min or stripped.startswith("#") or "=" not in stripped:
            continue

        key = stripped.split("=", 1)[0].strip()
        if key not in PATH_MIN_KEYS or key not in params:
            continue

        indent = line[: len(line) - len(line.lstrip())]
        lines[i] = f"{indent}{key} = {_format_toml_value(params[key])}"
        seen.add(key)

    missing = PATH_MIN_KEYS.difference(seen)
    if missing:
        raise RuntimeError(f"Missing expected keys in [path_min_inputs]: {sorted(missing)}")

    output_toml.write_text("\n".join(lines) + "\n")


def _run_command(cmd: list[str], cwd: Path, log_fp: Path, timeout_s: int) -> int:
    with log_fp.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    return proc.returncode


def _read_status_state(status_fp: Path) -> str:
    if not status_fp.exists():
        return "missing"
    data = json.loads(status_fp.read_text())
    return str(data.get("run_state", "unknown"))


def _parse_energies(energies_fp: Path) -> list[float]:
    vals = []
    for raw in energies_fp.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        vals.append(float(raw))
    return vals


def _split_xyz_frames(xyz_fp: Path) -> list[list[str]]:
    lines = xyz_fp.read_text().splitlines()
    frames: list[list[str]] = []
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].strip())
        end = i + nat + 2
        if end > n:
            raise ValueError(f"Malformed XYZ at line {i + 1} in {xyz_fp}")
        frames.append(lines[i:end])
        i = end
    return frames


def _extract_ts_guess(chain_xyz: Path, energies_fp: Path, ts_guess_fp: Path) -> tuple[int, int]:
    energies = _parse_energies(energies_fp)
    frames = _split_xyz_frames(chain_xyz)
    if len(energies) != len(frames):
        raise RuntimeError(
            f"Energy/frame mismatch for {chain_xyz.name}: {len(energies)} energies vs {len(frames)} frames"
        )

    idx = max(range(len(energies)), key=lambda i: energies[i])
    ts_guess_fp.write_text("\n".join(frames[idx]) + "\n")
    return idx, len(frames)


def _find_key_recursive(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


def _count_negative_freqs(hessian_qcio: Path) -> int:
    payload = json.loads(hessian_qcio.read_text())
    freqs = _find_key_recursive(payload, "frequencies_wavenumber")
    if not isinstance(freqs, list):
        raise RuntimeError(f"Could not locate frequencies_wavenumber in {hessian_qcio}")
    numeric = [float(x) for x in freqs]
    # Ignore near-zero numerical noise around translational/rotational modes.
    return sum(1 for f in numeric if f < -10.0)


def _looseness_score(params: dict[str, Any]) -> tuple[Any, ...]:
    return (
        params["rms_grad_thre"],
        params["max_rms_grad_thre"],
        params["ts_grad_thre"],
        params["ts_spring_thre"],
        params["en_thre"],
        params["early_stop_force_thre"],
        -params["max_steps"],
    )


def _run_one(
    run_root: Path,
    param_cfg: dict[str, Any],
    system_name: str,
    paths: dict[str, Path],
    inputs_fp: Path,
    timeout_s: int,
) -> RunOutcome:
    outcome = RunOutcome(param_label=str(param_cfg["label"]), system=system_name, run_dir=run_root)
    start_t = time.time()

    run_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{system_name}_{param_cfg['label']}"

    neb_log = run_root / "neb.log"
    ts_log = run_root / "ts.log"
    hess_log = run_root / "hessian.log"

    try:
        neb_cmd = [
            "uv", "run", "mepd", "run",
            "--start", str(paths["start"]),
            "--end", str(paths["end"]),
            "-i", str(inputs_fp),
            "--name", run_name,
        ]
        outcome.neb_returncode = _run_command(neb_cmd, cwd=run_root, log_fp=neb_log, timeout_s=timeout_s)

        status_fp = run_root / f"{run_name}_status.json"
        status = _read_status_state(status_fp)
        chain_xyz = run_root / f"{run_name}.xyz"
        energies_fp = run_root / f"{run_name}.energies"

        if outcome.neb_returncode != 0:
            outcome.error = f"NEB command failed (exit {outcome.neb_returncode})"
            return outcome
        if status != "completed":
            outcome.error = f"NEB status not completed (run_state={status})"
            return outcome
        if not chain_xyz.exists() or not energies_fp.exists():
            outcome.error = "NEB output chain/energies missing"
            return outcome

        ts_guess_fp = run_root / f"{run_name}_ts_guess.xyz"
        ts_idx, n_images = _extract_ts_guess(chain_xyz, energies_fp, ts_guess_fp)
        outcome.highest_energy_index = ts_idx
        outcome.n_images = n_images

        if ts_idx <= 0 or ts_idx >= n_images - 1:
            outcome.error = f"Highest-energy image was an endpoint (idx={ts_idx}, n_images={n_images})"
            return outcome

        outcome.neb_converged = True

        ts_name = f"{run_name}_ts"
        ts_cmd = [
            "uv", "run", "mepd", "ts",
            str(ts_guess_fp),
            "-i", str(inputs_fp),
            "--name", ts_name,
        ]
        outcome.ts_returncode = _run_command(ts_cmd, cwd=run_root, log_fp=ts_log, timeout_s=timeout_s)
        ts_xyz = run_root / f"{ts_name}.xyz"

        if outcome.ts_returncode != 0 or not ts_xyz.exists():
            outcome.error = f"TS optimization failed (exit {outcome.ts_returncode})"
            return outcome

        outcome.ts_converged = True

        hess_name = f"{run_name}_tscheck"
        hess_cmd = [
            "uv", "run", "mepd", "pseuirc",
            str(ts_xyz),
            "-i", str(inputs_fp),
            "--name", hess_name,
            "--dr", "0.05",
        ]
        outcome.hess_returncode = _run_command(hess_cmd, cwd=run_root, log_fp=hess_log, timeout_s=timeout_s)

        hess_qcio = run_root / f"{hess_name}_hessian.qcio"
        if outcome.hess_returncode != 0 or not hess_qcio.exists():
            outcome.error = f"Hessian computation failed (exit {outcome.hess_returncode})"
            return outcome

        n_neg = _count_negative_freqs(hess_qcio)
        outcome.n_negative_frequencies = n_neg
        outcome.valid_first_order_ts = n_neg == 1
        if not outcome.valid_first_order_ts:
            outcome.error = f"TS has {n_neg} negative frequencies"

    except subprocess.TimeoutExpired as exc:
        outcome.error = f"Timeout after {int(exc.timeout)} s"
    except Exception as exc:  # keep sweep running even if one case breaks
        outcome.error = f"{type(exc).__name__}: {exc}"
    finally:
        outcome.elapsed_seconds = time.time() - start_t

    return outcome


def main() -> None:
    base_inputs = ROOT / "example_inputs.toml"
    if not base_inputs.exists():
        raise FileNotFoundError(base_inputs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "sweep_results" / f"neb_convergence_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    timeout_s = 20 * 60
    results: list[RunOutcome] = []

    for idx, param_cfg in enumerate(PARAMETER_SETS, start=1):
        label = str(param_cfg["label"])
        param_root = out_root / f"set_{idx:02d}_{label}"
        param_root.mkdir(parents=True, exist_ok=True)

        input_fp = param_root / f"inputs_{label}.toml"
        _write_parametrized_input(base_inputs, input_fp, param_cfg)

        for system_name, paths in SYSTEMS.items():
            run_dir = param_root / system_name
            print(f"[{datetime.now().isoformat(timespec='seconds')}] set={label} system={system_name} ...", flush=True)
            outcome = _run_one(
                run_root=run_dir,
                param_cfg=param_cfg,
                system_name=system_name,
                paths=paths,
                inputs_fp=input_fp,
                timeout_s=timeout_s,
            )
            results.append(outcome)
            print(
                f"  -> neb={outcome.neb_converged} ts={outcome.ts_converged} "
                f"nneg={outcome.n_negative_frequencies} pass={outcome.passed} "
                f"elapsed={outcome.elapsed_seconds:.1f}s"
                + (f" error={outcome.error}" if outcome.error else ""),
                flush=True,
            )

    summary_rows: list[dict[str, Any]] = []
    for param_cfg in PARAMETER_SETS:
        label = str(param_cfg["label"])
        subset = [r for r in results if r.param_label == label]
        pass_count = sum(1 for r in subset if r.passed)
        neb_count = sum(1 for r in subset if r.neb_converged)
        ts_count = sum(1 for r in subset if r.ts_converged)
        summary_rows.append(
            {
                "label": label,
                "systems_tested": len(subset),
                "neb_converged": neb_count,
                "ts_converged": ts_count,
                "passed_first_order_ts": pass_count,
                "pass_rate": pass_count / len(subset) if subset else 0.0,
                "mean_elapsed_seconds": (
                    sum(r.elapsed_seconds for r in subset) / len(subset) if subset else 0.0
                ),
                "params": {k: v for k, v in param_cfg.items() if k in PATH_MIN_KEYS},
                "looseness_score": _looseness_score(param_cfg),
            }
        )

    best = None
    if summary_rows:
        best = sorted(
            summary_rows,
            key=lambda row: (
                row["passed_first_order_ts"],
                row["looseness_score"],
                -row["mean_elapsed_seconds"],
            ),
            reverse=True,
        )[0]

    details_csv = out_root / "details.csv"
    with details_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "param_label",
                "system",
                "neb_returncode",
                "ts_returncode",
                "hess_returncode",
                "neb_converged",
                "ts_converged",
                "highest_energy_index",
                "n_images",
                "n_negative_frequencies",
                "valid_first_order_ts",
                "passed",
                "elapsed_seconds",
                "error",
                "run_dir",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "param_label": r.param_label,
                    "system": r.system,
                    "neb_returncode": r.neb_returncode,
                    "ts_returncode": r.ts_returncode,
                    "hess_returncode": r.hess_returncode,
                    "neb_converged": r.neb_converged,
                    "ts_converged": r.ts_converged,
                    "highest_energy_index": r.highest_energy_index,
                    "n_images": r.n_images,
                    "n_negative_frequencies": r.n_negative_frequencies,
                    "valid_first_order_ts": r.valid_first_order_ts,
                    "passed": r.passed,
                    "elapsed_seconds": f"{r.elapsed_seconds:.3f}",
                    "error": r.error,
                    "run_dir": str(r.run_dir),
                }
            )

    summary_json = out_root / "summary.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "output_root": str(out_root),
        "details_csv": str(details_csv),
        "summary_rows": summary_rows,
        "best": best,
    }
    summary_json.write_text(json.dumps(payload, indent=2))

    print("\n=== Sweep Complete ===")
    print(f"Output root: {out_root}")
    print(f"Details CSV: {details_csv}")
    print(f"Summary JSON: {summary_json}")
    if best is None:
        print("No best parameter set could be determined.")
    else:
        print("Best candidate:")
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
