from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.geodesic_interpolation2.fileio import write_xyz
from neb_dynamics.pot import Pot


def _write_chain_with_nan_fallback(chain, fp: Path) -> None:
    """Write chain xyz plus energy/gradient sidecars; missing data is serialized as NaN."""
    fp = Path(fp)
    xyz_arr = chain.coordinates * BOHR_TO_ANGSTROMS
    write_xyz(filename=fp, atoms=chain.symbols, coords=xyz_arr)

    n_nodes = len(chain.nodes)
    n_atoms = chain.coordinates.shape[1] if n_nodes > 0 else 0

    ene_path = fp.parent / Path(f"{fp.stem}.energies")
    grad_path = fp.parent / Path(f"{fp.stem}.gradients")
    grad_shape_path = fp.parent / Path(f"{fp.stem}_grad_shapes.txt")

    energies = []
    missing_energies = False
    for node in chain.nodes:
        try:
            energies.append(float(node.energy))
        except Exception:
            missing_energies = True
            break
    if missing_energies:
        np.savetxt(ene_path, np.full(n_nodes, np.nan))
    else:
        np.savetxt(ene_path, np.array(energies))

    gradients = []
    missing_gradients = False
    for node in chain.nodes:
        try:
            gradients.append(np.array(node.gradient, dtype=float))
        except Exception:
            missing_gradients = True
            break
    if missing_gradients:
        np.savetxt(grad_path, np.full(n_nodes * n_atoms * 3, np.nan))
        np.savetxt(grad_shape_path, np.array([n_nodes, n_atoms, 3]))
    else:
        grad_arr = np.array(gradients, dtype=float)
        np.savetxt(grad_path, grad_arr.flatten())
        np.savetxt(grad_shape_path, np.array(grad_arr.shape))


def _write_chain_history_with_nan_fallback(chain_trajectory: list, fp: Path) -> None:
    out_folder = fp.resolve().parent / f"{fp.stem}_history"
    if out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for i, chain in enumerate(chain_trajectory):
        _write_chain_with_nan_fallback(chain, out_folder / f"traj_{i}.xyz")


def _write_neb_results_with_history(neb_result, fp: Path, console=None) -> bool:
    """Write final chain plus history for non-recursive NEB runs."""
    if hasattr(neb_result, "write_to_disk"):
        try:
            neb_result.write_to_disk(fp, write_history=True)
            return True
        except Exception as exc:
            if console is not None:
                console.print(
                    f"[yellow]⚠ Could not write full NEB history via write_to_disk: {exc}. "
                    "Falling back to NaN-safe writer.[/yellow]"
                )

    chain_for_profile = None
    if getattr(neb_result, "chain_trajectory", None):
        chain_for_profile = neb_result.chain_trajectory[-1]
    elif getattr(neb_result, "optimized", None) is not None:
        chain_for_profile = neb_result.optimized

    if chain_for_profile is None:
        return False

    _write_chain_with_nan_fallback(chain_for_profile, fp)
    if getattr(neb_result, "chain_trajectory", None):
        _write_chain_history_with_nan_fallback(neb_result.chain_trajectory, fp)
    return True


def _write_json_atomic(fp: Path, payload: dict) -> None:
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = fp.with_name(fp.name + ".tmp")
    tmp_fp.write_text(json.dumps(payload, indent=2))
    tmp_fp.replace(fp)


def _recursive_split_manifest_path(output_dir: Path, base_name: str) -> Path:
    return Path(output_dir) / f"{base_name}_request_manifest.json"


def _run_status_path(output_dir: Path, base_name: str) -> Path:
    return Path(output_dir) / f"{base_name}_status.json"


def _request_record_summary(request_records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in request_records:
        status = record.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _create_recursive_request_record(
    request_id: int,
    parent_request_id: int | None,
    start_index: int,
    end_index: int,
    status: str,
    **extra,
) -> dict:
    record = {
        "request_id": request_id,
        "parent_request_id": parent_request_id,
        "start_index": start_index,
        "end_index": end_index,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }
    record.update(extra)
    return record


def _upsert_request_record(request_records: list[dict], new_record: dict) -> None:
    for i, record in enumerate(request_records):
        if record.get("request_id") == new_record.get("request_id"):
            request_records[i] = new_record
            return
    request_records.append(new_record)
    request_records.sort(key=lambda row: row.get("request_id", -1))


def _summarize_network_file(network_fp: Path | None) -> dict | None:
    if network_fp is None or not Path(network_fp).exists():
        return None
    try:
        pot = Pot.read_from_disk(network_fp)
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": str(exc),
        }

    nodes = sorted(str(node) for node in pot.graph.nodes)
    edges = sorted([list(map(str, edge)) for edge in pot.graph.edges])
    return {
        "status": "available",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _write_recursive_split_manifest(
    output_dir: Path,
    base_name: str,
    request_records: list[dict],
    run_state: str,
    current_request_id: int | None = None,
    network_fp: Path | None = None,
) -> Path:
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "base_name": base_name,
        "run_state": run_state,
        "current_request_id": current_request_id,
        "total_requests": len(request_records),
        "counts": _request_record_summary(request_records),
        "requests": request_records,
    }
    if network_fp is not None:
        manifest["network_path"] = str(network_fp)
        manifest["network_summary"] = _summarize_network_file(network_fp)
    manifest_fp = _recursive_split_manifest_path(output_dir=output_dir, base_name=base_name)
    _write_json_atomic(manifest_fp, manifest)
    return manifest_fp


def _write_run_status(
    status_fp: Path,
    *,
    base_name: str,
    run_state: str,
    phase: str,
    recursive: bool | None = None,
    network_splits: bool | None = None,
    path_min_method: str | None = None,
    output_chain_path: Path | None = None,
    tree_path: Path | None = None,
    network_splits_dir: Path | None = None,
    manifest_fp: Path | None = None,
    network_fp: Path | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "base_name": base_name,
        "run_state": run_state,
        "phase": phase,
    }
    if recursive is not None:
        payload["recursive"] = recursive
    if network_splits is not None:
        payload["network_splits"] = network_splits
    if path_min_method is not None:
        payload["path_min_method"] = path_min_method
    if output_chain_path is not None:
        payload["output_chain_path"] = str(output_chain_path)
    if tree_path is not None:
        payload["tree_path"] = str(tree_path)
    if network_splits_dir is not None:
        payload["network_splits_dir"] = str(network_splits_dir)
    if manifest_fp is not None:
        payload["manifest_path"] = str(manifest_fp)
    if network_fp is not None:
        payload["network_path"] = str(network_fp)
        payload["network_summary"] = _summarize_network_file(network_fp)
    if error is not None:
        payload["error"] = error
    _write_json_atomic(status_fp, payload)


def _resolve_status_artifact(path_text: str) -> tuple[Path, str]:
    src = Path(path_text).resolve()
    candidates: list[tuple[Path, str]] = []

    if src.is_file():
        if src.name.endswith("_status.json"):
            candidates.append((src, "run_status"))
        if src.name.endswith("_request_manifest.json"):
            candidates.append((src, "request_manifest"))
        if src.name.endswith("_network.json"):
            manifest = src.with_name(src.name.replace("_network.json", "_request_manifest.json"))
            if manifest.exists():
                candidates.append((manifest, "request_manifest"))
        if src.suffix == ".xyz":
            stem = src.stem
            candidates.append((src.parent / f"{stem}_status.json", "run_status"))
            candidates.append((src.parent / f"{stem}_network_splits" / f"{stem}_request_manifest.json", "request_manifest"))
    elif src.is_dir():
        manifests = sorted(src.glob("*_request_manifest.json"))
        statuses = sorted(src.glob("*_status.json"))
        if statuses:
            candidates.append((statuses[0], "run_status"))
        if manifests:
            candidates.append((manifests[0], "request_manifest"))
        if src.name.endswith("_network_splits"):
            stem = src.name[: -len("_network_splits")]
            candidates.append((src / f"{stem}_request_manifest.json", "request_manifest"))
        else:
            candidates.append((src.parent / f"{src.name}_status.json", "run_status"))
            candidates.append((src.parent / f"{src.name}_network_splits" / f"{src.name}_request_manifest.json", "request_manifest"))
    else:
        stem = src.stem if src.suffix else src.name
        candidates.append((src.parent / f"{stem}_status.json", "run_status"))
        candidates.append((src.parent / f"{stem}_network_splits" / f"{stem}_request_manifest.json", "request_manifest"))

    for candidate, kind in candidates:
        if candidate.exists():
            return candidate, kind
    raise ValueError(f"Could not find a status artifact for: {src}")


def _load_status_snapshot(path_text: str) -> dict:
    artifact_fp, artifact_kind = _resolve_status_artifact(path_text)
    payload = json.loads(artifact_fp.read_text())
    if artifact_kind == "request_manifest":
        run_status = None
        status_fp = artifact_fp.with_name(artifact_fp.name.replace("_request_manifest.json", "_status.json"))
        if status_fp.exists():
            run_status = json.loads(status_fp.read_text())
        return {
            "artifact_kind": artifact_kind,
            "artifact_path": str(artifact_fp),
            "manifest": payload,
            "run_status": run_status,
        }

    manifest = None
    manifest_path = payload.get("manifest_path")
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text())
    return {
        "artifact_kind": artifact_kind,
        "artifact_path": str(artifact_fp),
        "run_status": payload,
        "manifest": manifest,
    }
