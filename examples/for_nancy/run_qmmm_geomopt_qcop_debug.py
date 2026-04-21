#!/usr/bin/env python3
"""Run a single QMMMEngine geometry optimization on prod.xyz via QCOP.

This script is designed for failure diagnosis. It uses the for_nancy inputs by
default, forces `compute_program=qcop`, and writes detailed artifacts so the
failure stage is clear.
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from qcio import Structure

from neb_dynamics.engines.qmmm import QMMMEngine
from neb_dynamics.errors import ElectronicStructureError
from neb_dynamics.inputs import RunInputs
from neb_dynamics.nodes.node import StructureNode


def _resolve(base: Path, value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _safe_output_path(base: Path, key: str) -> Path:
    parts = [part for part in Path(str(key)).parts if part not in {"", ".", ".."}]
    if not parts:
        return base / "unnamed_file"
    return base.joinpath(*parts)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _dump_output(output: Any, engine: QMMMEngine, out_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if output is None:
        return summary

    for field in ("success", "traceback"):
        if hasattr(output, field):
            summary[field] = getattr(output, field)

    stdout = getattr(output, "stdout", None)
    stderr = getattr(output, "stderr", None)
    if stdout is not None:
        _write_text(out_dir / "stdout.log", str(stdout))
        summary["stdout_len"] = len(str(stdout))
    if stderr is not None:
        _write_text(out_dir / "stderr.log", str(stderr))
        summary["stderr_len"] = len(str(stderr))

    files = engine._extract_output_files(output) or {}
    summary["n_output_files"] = len(files)
    for key, content in files.items():
        fp = _safe_output_path(out_dir / "output_files", str(key))
        if isinstance(content, bytes):
            _write_bytes(fp, content)
        else:
            _write_text(fp, str(content))

    return summary


def _set_charge_and_multiplicity(
    structure: Structure,
    charge: int,
    multiplicity: int,
) -> Structure:
    if structure.charge == charge and structure.multiplicity == multiplicity:
        return structure
    data = structure.model_dump()
    data["charge"] = charge
    data["multiplicity"] = multiplicity
    return Structure(**data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug QMMMEngine geometry optimization (for_nancy)."
    )
    parser.add_argument(
        "--inputs",
        default="qmmm_inputs_s0min_frozen.toml",
        help="QMMM RunInputs TOML (default: qmmm_inputs_s0min_frozen.toml).",
    )
    parser.add_argument(
        "--geometry",
        default="prod.xyz",
        help="Geometry file for optimization start (default: prod.xyz).",
    )
    parser.add_argument(
        "--compute-program",
        choices=("qcop", "chemcloud"),
        default="qcop",
        help="Backend for QMMMEngine (default: qcop).",
    )
    parser.add_argument(
        "--coordsys",
        default="cart",
        help="Geometry optimization coord system alias passed to QMMMEngine.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="Max geometry optimization iterations (default: 500).",
    )
    parser.add_argument(
        "--artifacts-root",
        default="stdout_probe",
        help="Where to write logs/debug artifacts (default: stdout_probe).",
    )
    parser.add_argument(
        "--print-stdout",
        action="store_true",
        help="Stream QCOP/TeraChem stdout while the job runs.",
    )
    parser.add_argument(
        "--skip-graph-check",
        action="store_true",
        help="Skip StructureNode graph-construction probe on optimized structures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    engine: QMMMEngine | None = None

    inputs_fp = _resolve(base_dir, args.inputs)
    geometry_fp = _resolve(base_dir, args.geometry)
    artifacts_root = _resolve(base_dir, args.artifacts_root)
    run_dir = artifacts_root / f"qmmm_geomopt_{args.compute_program}_{timestamp}"
    debug_dump_dir = run_dir / "qmmm_debug_inputs"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not inputs_fp.exists():
            raise FileNotFoundError(f"Inputs TOML not found: {inputs_fp}")
        if not geometry_fp.exists():
            raise FileNotFoundError(f"Geometry file not found: {geometry_fp}")

        run_inputs = RunInputs.open(inputs_fp)
        if run_inputs.engine_name != "qmmm" or not isinstance(run_inputs.engine, QMMMEngine):
            raise RuntimeError(
                f"Expected a QMMM RunInputs file, got engine_name={run_inputs.engine_name!r}"
            )

        engine = run_inputs.engine
        engine.compute_program = args.compute_program
        engine.print_stdout = bool(args.print_stdout)
        engine.debug_dump_inputs = True
        engine.debug_dump_dir = debug_dump_dir

        qmmm_cfg = dict(run_inputs.qmmm_inputs or {})
        print(f"inputs: {inputs_fp}")
        print(f"geometry: {geometry_fp}")
        print(f"compute_program: {engine.compute_program}")
        print(f"qmindices: {engine.qminds_fp}")
        print(f"prmtop: {engine.prmtop_fp}")
        print(f"rst7_ref: {engine.rst7_fp_react}")
        print(f"qm index count: {len([x for x in engine.qmindices.splitlines() if x.strip()])}")
        print(f"artifacts: {run_dir}")

        structure = Structure.open(geometry_fp)
        raw_charge = qmmm_cfg.get("charge", structure.charge if structure.charge is not None else 0)
        raw_mult = qmmm_cfg.get(
            "spinmult", structure.multiplicity if structure.multiplicity is not None else 1
        )
        charge = int(raw_charge)
        multiplicity = int(raw_mult)
        structure = _set_charge_and_multiplicity(
            structure=structure,
            charge=charge,
            multiplicity=multiplicity,
        )
        _write_text(
            run_dir / "input_structure.xyz",
            structure.to_xyz(),
        )

        minimize_kwds = {"coordsys": args.coordsys, "maxiter": args.maxiter}
        preview_kwds = engine._normalize_minimize_keywords(minimize_kwds)
        preview_frozen = engine._coerce_frozen_atom_indices(
            preview_kwds.pop("frozen_atom_indices", None)
        )
        if len(preview_frozen) == 0:
            preview_frozen = list(engine.frozen_atom_indices or [])
        preview_kwds["new_minimizer"] = "no"
        resolved_tcin = engine._with_run_type(engine.inp_file, "minimize")
        resolved_tcin = engine._apply_tcin_overrides(resolved_tcin, preview_kwds)
        resolved_tcin = engine._with_constraints(
            tcin_text=resolved_tcin,
            frozen_atom_indices=preview_frozen,
        )
        _write_text(run_dir / "resolved_minimize.tc.in", resolved_tcin)
        print(f"frozen_atom_indices_applied: {len(set(int(i) for i in preview_frozen if int(i) >= 0))}")
        print(f"resolved_tcin: {run_dir / 'resolved_minimize.tc.in'}")

        _write_text(
            run_dir / "run_settings.txt",
            "\n".join(
                [
                    f"inputs={inputs_fp}",
                    f"geometry={geometry_fp}",
                    f"compute_program={engine.compute_program}",
                    f"coordsys={args.coordsys}",
                    f"maxiter={args.maxiter}",
                    f"charge={charge}",
                    f"multiplicity={multiplicity}",
                    f"debug_dump_dir={debug_dump_dir}",
                ]
            )
            + "\n",
        )

        # Use direct engine methods to isolate backend failures from unrelated path logic.
        rst7_payload = engine.structure_to_rst7(structure)
        output = engine._compute_minimize_output(
            rst7_string=rst7_payload,
            keywords=minimize_kwds,
        )
        summary = _dump_output(output=output, engine=engine, out_dir=run_dir)

        structures = engine._extract_optimization_structures(
            output=output,
            reference_structure=structure,
        )
        if not structures:
            raise ElectronicStructureError(
                msg="Optimization returned no structures (optim.xyz missing or empty).",
                obj=output,
            )

        traj_text = "".join(s.to_xyz().rstrip() + "\n" for s in structures)
        _write_text(run_dir / "optimized_trajectory.xyz", traj_text)
        structures[-1].save(run_dir / "optimized_final.xyz")

        if not args.skip_graph_check:
            probe_structures = [structures[0], structures[-1]] if len(structures) > 1 else [structures[0]]
            try:
                _ = [StructureNode(structure=s) for s in probe_structures]
                _write_text(run_dir / "graph_check.txt", "ok\n")
            except Exception as graph_exc:
                _write_text(
                    run_dir / "graph_check.txt",
                    f"failed: {type(graph_exc).__name__}: {graph_exc}\n",
                )

        print(f"status: success ({len(structures)} optimization frames)")
        if summary:
            print(f"output_summary: {summary}")
        print(f"trajectory: {run_dir / 'optimized_trajectory.xyz'}")
        print(f"final_structure: {run_dir / 'optimized_final.xyz'}")
        print(f"debug_inputs: {debug_dump_dir}")
        return 0

    except Exception as exc:
        tb = traceback.format_exc()
        _write_text(run_dir / "traceback.txt", tb)

        if (
            isinstance(exc, ElectronicStructureError)
            and exc.obj is not None
            and engine is not None
        ):
            _ = _dump_output(output=exc.obj, engine=engine, out_dir=run_dir / "error_obj")

        print(f"status: failure ({type(exc).__name__}: {exc})")
        print(f"traceback: {run_dir / 'traceback.txt'}")
        print(f"artifacts: {run_dir}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
