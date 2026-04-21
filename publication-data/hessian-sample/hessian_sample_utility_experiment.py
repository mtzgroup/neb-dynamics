#!/usr/bin/env python3
"""Run a Hessian-sampling utility experiment over a DR sweep.

Flow:
1) Load input structure (SMILES or geometry file path).
2) Minimize once using the provided RunInputs.
3) Compute Hessian normal modes on the minimized seed.
4) For each DR value, displace along all modes (+/-), minimize all candidates,
   and classify converged minima relative to the seed structure.
5) Write summary logs and a stacked-bar plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import RunInputs
from neb_dynamics.molecule import Molecule
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.nodes.nodehelpers import _is_connectivity_identical, displace_by_dr, is_identical
from neb_dynamics.qcio_structure_helpers import molecule_to_structure
from neb_dynamics.retropaths_workflow import (
    _compute_hessian_result_for_sampling,
    _extract_normal_modes_from_hessian_result,
    _resolve_hessian_use_bigchem,
)

DEFAULT_DR_VALUES = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


@dataclass
class CandidateOutcome:
    mode_index: int
    direction: str
    dr: float
    converged: bool
    classification: str
    error: str = ""


def _parse_dr_values(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError(f"DR values must be positive. Got {value}.")
        values.append(value)
    if not values:
        raise ValueError("No DR values were provided.")
    return values


def _load_structure(start: str, charge: int, multiplicity: int) -> Structure:
    start_path = Path(start)
    if start_path.exists():
        structure = Structure.open(start_path)
    else:
        structure = molecule_to_structure(
            Molecule.from_smiles(start),
            charge=int(charge),
            spinmult=int(multiplicity),
        )
    payload = structure.model_dump()
    payload["charge"] = int(charge)
    payload["multiplicity"] = int(multiplicity)
    return Structure(**payload)


def _final_optimized_node(result: Any) -> StructureNode | None:
    if result is None:
        return None
    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            return None
        candidate = result[-1]
        return candidate if isinstance(candidate, StructureNode) else None
    return result if isinstance(result, StructureNode) else None


def _optimize_single_candidate(
    engine: Any,
    candidate: StructureNode,
    maxiter: int,
) -> tuple[StructureNode | None, str]:
    compute_single = getattr(engine, "compute_geometry_optimization", None)
    if compute_single is None:
        return None, "Engine has no compute_geometry_optimization method."
    try:
        try:
            result = compute_single(candidate, keywords={"coordsys": "cart", "maxiter": int(maxiter)})
        except TypeError:
            result = compute_single(candidate)
    except Exception:
        return None, traceback.format_exc().strip()
    final_node = _final_optimized_node(result)
    if final_node is None:
        return None, "Optimization returned no final optimized node."
    return final_node, ""


def _optimize_candidates(
    run_inputs: RunInputs,
    candidates: list[StructureNode],
    *,
    maxiter: int,
) -> tuple[list[StructureNode | None], list[str]]:
    engine = run_inputs.engine
    optimized: list[StructureNode | None] = [None for _ in candidates]
    errors: list[str] = ["" for _ in candidates]

    engine_name = str(getattr(run_inputs, "engine_name", "") or "").strip().lower()
    compute_program = str(getattr(engine, "compute_program", "") or "").strip().lower()
    compute_many = getattr(engine, "compute_geometry_optimizations", None)
    use_chemcloud_batch = engine_name == "chemcloud" or compute_program == "chemcloud"

    if use_chemcloud_batch and callable(compute_many):
        try:
            try:
                histories = compute_many(
                    candidates,
                    keywords={"coordsys": "cart", "maxiter": int(maxiter)},
                )
            except TypeError:
                histories = compute_many(candidates)
            histories = list(histories)
        except Exception:
            batch_error = traceback.format_exc().strip()
            for i in range(len(candidates)):
                errors[i] = f"ChemCloud batch optimization failed: {batch_error}"
            return optimized, errors

        if len(histories) < len(candidates):
            histories.extend([[] for _ in range(len(candidates) - len(histories))])
        if len(histories) > len(candidates):
            histories = histories[: len(candidates)]

        for i, hist in enumerate(histories):
            final_node = _final_optimized_node(hist)
            if final_node is None:
                errors[i] = "Batch optimization returned an empty/non-node trajectory."
            else:
                optimized[i] = final_node
        return optimized, errors

    for i, candidate in enumerate(candidates):
        final_node, error = _optimize_single_candidate(engine, candidate, maxiter=maxiter)
        optimized[i] = final_node
        errors[i] = error
    return optimized, errors


def _classify_minimum(
    *,
    seed: StructureNode,
    optimized: StructureNode,
    rmsd_cutoff: float,
    kcal_cutoff: float,
) -> str:
    same_connectivity = _is_connectivity_identical(
        seed,
        optimized,
        verbose=False,
        collect_comparison=False,
    )
    if not same_connectivity:
        return "new_connectivity"
    same_structure = is_identical(
        seed,
        optimized,
        fragment_rmsd_cutoff=float(rmsd_cutoff),
        kcal_mol_cutoff=float(kcal_cutoff),
        verbose=False,
        collect_comparison=False,
    )
    return "same_seed" if same_structure else "new_conformer"


def _write_candidate_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_nodes_xyz(path: Path, nodes: list[StructureNode], run_inputs: RunInputs) -> None:
    if len(nodes) == 0:
        return
    chain = Chain.model_validate(
        {
            "nodes": [node.copy() for node in nodes],
            "parameters": run_inputs.chain_inputs,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    chain.write_to_disk(path, write_qcio=False)


def _dedupe_nodes_by_identity(
    nodes: list[StructureNode],
    *,
    rmsd_cutoff: float,
    kcal_cutoff: float,
) -> list[StructureNode]:
    unique: list[StructureNode] = []
    for node in nodes:
        duplicate = False
        for existing in unique:
            if is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=float(rmsd_cutoff),
                kcal_mol_cutoff=float(kcal_cutoff),
                verbose=False,
                collect_comparison=False,
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(node.copy())
    return unique


def _dedupe_nodes_by_connectivity(nodes: list[StructureNode]) -> list[StructureNode]:
    unique: list[StructureNode] = []
    for node in nodes:
        duplicate = False
        for existing in unique:
            if _is_connectivity_identical(
                node,
                existing,
                verbose=False,
                collect_comparison=False,
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(node.copy())
    return unique


def _save_plot(
    *,
    summary_rows: list[dict[str, Any]],
    normal_modes_total: int,
    output_path: Path,
) -> None:
    dr_labels = [f"{row['dr']:.1f}" for row in summary_rows]
    x = np.arange(len(summary_rows))
    new_conn = np.array([int(row["new_connectivity"]) for row in summary_rows], dtype=int)
    new_conf = np.array([int(row["new_conformer"]) for row in summary_rows], dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, new_conn, color="#1f77b4", label="New connectivity")
    ax.bar(x, new_conf, bottom=new_conn, color="#ff7f0e", label="New conformer")

    total_displacements = int(normal_modes_total) * 2
    ax.axhline(
        y=float(total_displacements),
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"2 x normal modes ({total_displacements})",
    )

    for i, row in enumerate(summary_rows):
        ax.text(
            float(i),
            float(new_conn[i] + new_conf[i]) + 0.2,
            f"errors={int(row['errors'])}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dr_labels)
    ax.set_xlabel("dr")
    ax.set_ylabel("Count")
    ax.set_title("Hessian Sample Utility Experiment")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hessian sample utility experiment.")
    parser.add_argument(
        "--inputs",
        required=True,
        help="Path to RunInputs TOML.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start structure: file path (xyz/qcio/etc.) or a SMILES string.",
    )
    parser.add_argument("--charge", type=int, default=0, help="Total charge (default: 0).")
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity (default: 1).",
    )
    parser.add_argument(
        "--dr-values",
        default=",".join(str(v) for v in DEFAULT_DR_VALUES),
        help="Comma-separated DR values (default: 0.1,0.5,1.0,2.0,3.0,4.0,5.0).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="Maximum geometry-optimization steps (default: 500).",
    )
    parser.add_argument(
        "--max-normal-modes",
        type=int,
        default=100,
        help="Only run if total normal modes is strictly less than this value (default: 100).",
    )
    parser.add_argument(
        "--use-bigchem",
        action="store_true",
        help="Request use_bigchem=True for Hessian computation when supported.",
    )
    parser.add_argument(
        "--outdir",
        default="hessian_sample_utility_runs",
        help="Output root directory (default: hessian_sample_utility_runs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dr_values = _parse_dr_values(args.dr_values)
    if args.maxiter < 1:
        raise ValueError("--maxiter must be >= 1.")
    if args.max_normal_modes < 1:
        raise ValueError("--max-normal-modes must be >= 1.")

    run_inputs = RunInputs.open(args.inputs)
    structure = _load_structure(args.start, charge=args.charge, multiplicity=args.multiplicity)
    seed_node = StructureNode(structure=structure)

    out_root = Path(args.outdir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"hessian_sample_experiment_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"RunInputs: {Path(args.inputs).resolve()}")
    print(f"Start: {args.start}")
    print(f"DR sweep: {dr_values}")
    print(f"Output: {run_dir}")

    print("Step 1/4: Minimizing start structure...")
    try:
        try:
            start_hist = run_inputs.engine.compute_geometry_optimization(
                seed_node,
                keywords={"coordsys": "cart", "maxiter": int(args.maxiter)},
            )
        except TypeError:
            start_hist = run_inputs.engine.compute_geometry_optimization(seed_node)
    except Exception as exc:
        raise RuntimeError(f"Initial minimization failed: {type(exc).__name__}: {exc}") from exc
    minimized_seed = _final_optimized_node(start_hist)
    if minimized_seed is None:
        raise RuntimeError("Initial minimization produced no final optimized structure.")

    minimized_seed.structure.save(run_dir / "seed_minimized.xyz")

    print("Step 2/4: Computing Hessian and normal modes...")
    requested_use_bigchem = True if args.use_bigchem else None
    try:
        hessres = _compute_hessian_result_for_sampling(
            run_inputs.engine,
            minimized_seed,
            use_bigchem=_resolve_hessian_use_bigchem(
                run_inputs=run_inputs,
                requested_use_bigchem=requested_use_bigchem,
            ),
        )
    except Exception as exc:
        hessres = getattr(exc, "program_output", None)
        if hessres is None:
            raise RuntimeError(f"Hessian computation failed: {type(exc).__name__}: {exc}") from exc

    if hasattr(hessres, "save"):
        hessres.save(run_dir / "seed_hessian.qcio")
    normal_modes, frequencies = _extract_normal_modes_from_hessian_result(hessres)
    normal_modes_total = len(normal_modes)
    if normal_modes_total == 0:
        raise RuntimeError("No normal modes were found in Hessian result.")

    if normal_modes_total >= int(args.max_normal_modes):
        payload = {
            "skipped": True,
            "reason": (
                f"Total normal modes ({normal_modes_total}) is not < max-normal-modes "
                f"({int(args.max_normal_modes)})."
            ),
            "normal_modes_total": int(normal_modes_total),
            "max_normal_modes": int(args.max_normal_modes),
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        print(payload["reason"])
        print(f"Wrote: {run_dir / 'summary.json'}")
        return 0

    run_summary_rows: list[dict[str, Any]] = []
    candidate_logs_dir = run_dir / "candidate_logs"
    candidate_logs_dir.mkdir(parents=True, exist_ok=True)
    candidate_xyz_dir = run_dir / "candidate_xyz"
    candidate_xyz_dir.mkdir(parents=True, exist_ok=True)
    minimized_xyz_dir = run_dir / "minimized_xyz"
    minimized_xyz_dir.mkdir(parents=True, exist_ok=True)

    rms_cutoff = float(run_inputs.chain_inputs.node_rms_thre)
    ene_cutoff = float(run_inputs.chain_inputs.node_ene_thre)

    print("Step 3/4: Running DR sweep and minimizing all Hessian-displaced candidates...")
    for dr in dr_values:
        candidates: list[StructureNode] = []
        candidate_meta: list[tuple[int, str]] = []
        for mode_index, mode in enumerate(normal_modes):
            for direction, signed_dr in (("+", float(dr)), ("-", -float(dr))):
                displaced = displace_by_dr(
                    node=minimized_seed,
                    displacement=np.array(mode),
                    dr=signed_dr,
                )
                candidates.append(displaced)
                candidate_meta.append((int(mode_index), direction))

        _write_nodes_xyz(
            candidate_xyz_dir / f"dr_{dr:.3f}_candidates.xyz",
            candidates,
            run_inputs,
        )

        optimized_nodes, optimization_errors = _optimize_candidates(
            run_inputs,
            candidates,
            maxiter=int(args.maxiter),
        )

        outcomes: list[CandidateOutcome] = []
        for idx, optimized in enumerate(optimized_nodes):
            mode_index, direction = candidate_meta[idx]
            if optimized is None:
                outcomes.append(
                    CandidateOutcome(
                        mode_index=mode_index,
                        direction=direction,
                        dr=float(dr),
                        converged=False,
                        classification="error",
                        error=optimization_errors[idx] or "Unknown optimization failure.",
                    )
                )
                continue
            cls = _classify_minimum(
                seed=minimized_seed,
                optimized=optimized,
                rmsd_cutoff=rms_cutoff,
                kcal_cutoff=ene_cutoff,
            )
            outcomes.append(
                CandidateOutcome(
                    mode_index=mode_index,
                    direction=direction,
                    dr=float(dr),
                    converged=True,
                    classification=cls,
                )
            )

        as_rows = [
            {
                "mode_index": int(out.mode_index),
                "direction": out.direction,
                "dr": float(out.dr),
                "converged": bool(out.converged),
                "classification": out.classification,
                "error": out.error,
            }
            for out in outcomes
        ]
        _write_candidate_log(candidate_logs_dir / f"dr_{dr:.3f}.jsonl", as_rows)

        failed_candidate_nodes = [
            candidates[idx]
            for idx, out in enumerate(outcomes)
            if out.classification == "error"
        ]
        _write_nodes_xyz(
            candidate_xyz_dir / f"dr_{dr:.3f}_failed_candidates.xyz",
            failed_candidate_nodes,
            run_inputs,
        )

        converged_nodes = [
            optimized_nodes[idx]
            for idx, out in enumerate(outcomes)
            if out.converged and optimized_nodes[idx] is not None
        ]
        new_connectivity_hits = [
            optimized_nodes[idx]
            for idx, out in enumerate(outcomes)
            if out.classification == "new_connectivity" and optimized_nodes[idx] is not None
        ]
        new_conformer_hits = [
            optimized_nodes[idx]
            for idx, out in enumerate(outcomes)
            if out.classification == "new_conformer" and optimized_nodes[idx] is not None
        ]

        unique_new_connectivity_nodes = _dedupe_nodes_by_connectivity(new_connectivity_hits)
        unique_new_conformer_nodes = _dedupe_nodes_by_identity(
            new_conformer_hits,
            rmsd_cutoff=rms_cutoff,
            kcal_cutoff=ene_cutoff,
        )

        converged_xyz_fp = minimized_xyz_dir / f"dr_{dr:.3f}_converged_optimized.xyz"
        new_connectivity_xyz_fp = minimized_xyz_dir / f"dr_{dr:.3f}_new_connectivity.xyz"
        new_conformer_xyz_fp = minimized_xyz_dir / f"dr_{dr:.3f}_new_conformer.xyz"
        new_connectivity_hits_xyz_fp = minimized_xyz_dir / f"dr_{dr:.3f}_new_connectivity_hits.xyz"
        new_conformer_hits_xyz_fp = minimized_xyz_dir / f"dr_{dr:.3f}_new_conformer_hits.xyz"

        _write_nodes_xyz(converged_xyz_fp, converged_nodes, run_inputs)
        _write_nodes_xyz(new_connectivity_xyz_fp, unique_new_connectivity_nodes, run_inputs)
        _write_nodes_xyz(new_conformer_xyz_fp, unique_new_conformer_nodes, run_inputs)
        _write_nodes_xyz(new_connectivity_hits_xyz_fp, new_connectivity_hits, run_inputs)
        _write_nodes_xyz(new_conformer_hits_xyz_fp, new_conformer_hits, run_inputs)

        errors = sum(1 for out in outcomes if out.classification == "error")
        converged = sum(1 for out in outcomes if out.converged)
        raw_new_connectivity_hits = len(new_connectivity_hits)
        raw_new_conformer_hits = len(new_conformer_hits)
        new_connectivity = len(unique_new_connectivity_nodes)
        new_conformer = len(unique_new_conformer_nodes)
        same_seed = sum(1 for out in outcomes if out.classification == "same_seed")

        row = {
            "dr": float(dr),
            "normal_modes_total": int(normal_modes_total),
            "total_candidates": int(len(outcomes)),
            "converged": int(converged),
            "errors": int(errors),
            "new_connectivity": int(new_connectivity),
            "new_conformer": int(new_conformer),
            "raw_new_connectivity_hits": int(raw_new_connectivity_hits),
            "raw_new_conformer_hits": int(raw_new_conformer_hits),
            "same_seed": int(same_seed),
            "candidates_xyz": str((candidate_xyz_dir / f"dr_{dr:.3f}_candidates.xyz").resolve()),
            "failed_candidates_xyz": str(
                (candidate_xyz_dir / f"dr_{dr:.3f}_failed_candidates.xyz").resolve()
            )
            if errors > 0
            else "",
            "converged_optimized_xyz": str(converged_xyz_fp.resolve()) if converged > 0 else "",
            "new_connectivity_xyz": str(new_connectivity_xyz_fp.resolve()) if new_connectivity > 0 else "",
            "new_conformer_xyz": str(new_conformer_xyz_fp.resolve()) if new_conformer > 0 else "",
            "new_connectivity_hits_xyz": str(new_connectivity_hits_xyz_fp.resolve())
            if raw_new_connectivity_hits > 0
            else "",
            "new_conformer_hits_xyz": str(new_conformer_hits_xyz_fp.resolve())
            if raw_new_conformer_hits > 0
            else "",
        }
        run_summary_rows.append(row)
        print(
            f"dr={dr:.3f} | total={len(outcomes)} converged={converged} errors={errors} "
            f"new_connectivity(unique={new_connectivity}, raw={raw_new_connectivity_hits}) "
            f"new_conformer(unique={new_conformer}, raw={raw_new_conformer_hits})"
        )

    print("Step 4/4: Writing summary + plot...")
    summary_json = {
        "run_inputs": str(Path(args.inputs).resolve()),
        "start": args.start,
        "start_charge": int(args.charge),
        "start_multiplicity": int(args.multiplicity),
        "dr_values": [float(v) for v in dr_values],
        "maxiter": int(args.maxiter),
        "normal_modes_total": int(normal_modes_total),
        "frequencies_wavenumber": [float(v) for v in frequencies],
        "selection_rule": f"normal_modes_total < {int(args.max_normal_modes)}",
        "chain_inputs_thresholds": {
            "node_rms_thre": float(rms_cutoff),
            "node_ene_thre": float(ene_cutoff),
        },
        "per_dr": run_summary_rows,
        "paths": {
            "seed_minimized_xyz": str((run_dir / "seed_minimized.xyz").resolve()),
            "candidate_logs_dir": str(candidate_logs_dir.resolve()),
            "candidate_xyz_dir": str(candidate_xyz_dir.resolve()),
            "minimized_xyz_dir": str(minimized_xyz_dir.resolve()),
            "plot_png": str((run_dir / "hessian_sample_utility_summary.png").resolve()),
            "summary_csv": str((run_dir / "summary.csv").resolve()),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dr",
            "normal_modes_total",
            "total_candidates",
            "converged",
            "errors",
            "new_connectivity",
            "new_conformer",
            "raw_new_connectivity_hits",
            "raw_new_conformer_hits",
            "same_seed",
            "candidates_xyz",
            "failed_candidates_xyz",
            "converged_optimized_xyz",
            "new_connectivity_xyz",
            "new_conformer_xyz",
            "new_connectivity_hits_xyz",
            "new_conformer_hits_xyz",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_summary_rows)

    _save_plot(
        summary_rows=run_summary_rows,
        normal_modes_total=normal_modes_total,
        output_path=run_dir / "hessian_sample_utility_summary.png",
    )

    print(f"Wrote: {run_dir / 'summary.json'}")
    print(f"Wrote: {run_dir / 'summary.csv'}")
    print(f"Wrote: {run_dir / 'hessian_sample_utility_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
