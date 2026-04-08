from __future__ import annotations

from dataclasses import dataclass, field
import contextlib
import re
from typing import Any

import numpy as np
from qcio import FileInput, Structure

from neb_dynamics.chain import Chain
from neb_dynamics.constants import ANGSTROM_TO_BOHR
from neb_dynamics.elementarystep import ElemStepResults, check_if_elem_step
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.errors import ElectronicStructureError
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.pathminimizers.pathminimizer import PathMinimizer
from neb_dynamics.scripts.progress import print_persistent, update_status


IS_ELEM_STEP = ElemStepResults(
    is_elem_step=True,
    is_concave=True,
    splitting_criterion=None,
    minimization_results=None,
    number_grad_calls=0,
)


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _format_tc_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _coerce_int(value: Any, *, default: int | None, field_name: str) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return int(default)
        raise ValueError(f"Missing required integer value for `{field_name}`.")
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(
            f"Invalid integer value for `{field_name}`: {value!r}"
        ) from exc


def _iter_xyz_frames(chain: Chain) -> str:
    return "".join(
        node.structure.to_xyz().rstrip() + "\n"
        for node in chain
    )


def _parse_xyz_structures(
    xyz_text: str,
    *,
    charge: int,
    multiplicity: int,
) -> list[Structure]:
    lines = xyz_text.splitlines()
    idx = 0
    frames: list[str] = []

    while idx < len(lines):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            break
        nat_line = lines[idx].strip()
        nat = int(nat_line)
        if idx + 1 >= len(lines):
            raise ValueError("Malformed XYZ: missing comment line.")
        comment_line = lines[idx + 1]
        atom_start = idx + 2
        atom_end = atom_start + nat
        if atom_end > len(lines):
            raise ValueError("Malformed XYZ: not enough atom lines for declared frame size.")
        atom_lines = lines[atom_start:atom_end]
        frame_text = "\n".join([nat_line, comment_line, *atom_lines]).strip() + "\n"
        frames.append(frame_text)
        idx = atom_end

    structures: list[Structure] = []
    for frame_text in frames:
        try:
            structures.append(
                Structure.from_xyz(
                    frame_text,
                    charge=charge,
                    multiplicity=multiplicity,
                )
            )
        except ValueError as exc:
            message = str(exc).lower()
            if "charge cannot be set in the file and as an argument" not in message:
                raise
            structures.append(Structure.from_xyz(frame_text))
    return structures


def _extract_output_files(output: Any) -> dict[str, Any]:
    for files_obj in (
        getattr(output, "files", None),
        getattr(getattr(output, "results", None), "files", None),
        getattr(getattr(output, "data", None), "files", None),
    ):
        if not files_obj:
            continue
        if isinstance(files_obj, dict):
            return dict(files_obj)
        if hasattr(files_obj, "items"):
            return dict(files_obj.items())
        if hasattr(files_obj, "model_dump"):
            dumped = files_obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        with contextlib.suppress(Exception):
            return dict(files_obj)
    return {}


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _find_file_text(files: dict[str, Any], *suffixes: str) -> str | None:
    norm_suffixes = tuple(s.lower() for s in suffixes)
    for name, contents in files.items():
        key = str(name).lower()
        if key.endswith(norm_suffixes):
            return _decode_text(contents)
    return None


def _parse_last_neb_energies(nebinfo_text: str, nimages: int) -> list[float] | None:
    float_pattern = re.compile(r"[-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?")
    rows: list[list[float]] = []
    for raw_line in nebinfo_text.splitlines():
        matches = float_pattern.findall(raw_line)
        if len(matches) < 2:
            continue
        with contextlib.suppress(Exception):
            rows.append([float(token) for token in matches])
    if not rows:
        return None

    for idx_col in (1, 0):
        for base in (0, 1):
            energies_by_index: dict[int, float] = {}
            for row in reversed(rows):
                if idx_col >= len(row):
                    continue
                idx_raw = row[idx_col]
                idx_round = int(round(idx_raw))
                if not np.isclose(idx_raw, float(idx_round), atol=1e-8):
                    continue
                idx = idx_round - base
                if idx < 0 or idx >= nimages:
                    continue
                if idx not in energies_by_index:
                    energies_by_index[idx] = float(row[-1])
                if len(energies_by_index) == nimages:
                    return [energies_by_index[i] for i in range(nimages)]
    return None


def _extract_dlf_image_trajectory(
    files: dict[str, Any],
    *,
    nimages: int,
    charge: int,
    multiplicity: int,
    chain_parameters,
) -> list[Chain]:
    image_frames: dict[int, list[Structure]] = {}
    image_pattern = re.compile(r"(?:^|/|\\)neb_(\d+)\.xyz$", re.IGNORECASE)
    for file_name, contents in files.items():
        match = image_pattern.search(str(file_name))
        if match is None:
            continue
        img_idx = int(match.group(1))
        xyz_text = _decode_text(contents)
        with contextlib.suppress(Exception):
            image_frames[img_idx] = _parse_xyz_structures(
                xyz_text,
                charge=charge,
                multiplicity=multiplicity,
            )

    if len(image_frames) == 0:
        return []

    sorted_indices = sorted(image_frames)
    if len(sorted_indices) != nimages:
        return []
    frame_count = min(len(image_frames[idx]) for idx in sorted_indices)
    if frame_count <= 0:
        return []

    chain_trajectory: list[Chain] = []
    for step_idx in range(frame_count):
        nodes = [
            StructureNode(structure=image_frames[img_idx][step_idx])
            for img_idx in sorted_indices
        ]
        chain_trajectory.append(
            Chain.model_validate({"nodes": nodes, "parameters": chain_parameters})
        )
    return chain_trajectory


def _parse_path_geometry_structures(
    geometry_text: str,
    *,
    charge: int,
    multiplicity: int,
) -> list[Structure]:
    lines = geometry_text.splitlines()
    structures: list[Structure] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        if "Molecular Geometry" not in line:
            idx += 1
            continue
        idx += 1
        while idx < len(lines) and ("Type" in lines[idx] or not lines[idx].strip()):
            idx += 1

        symbols: list[str] = []
        coords_ang: list[list[float]] = []
        while idx < len(lines):
            row = lines[idx].strip()
            if not row:
                break
            parts = row.split()
            if len(parts) < 4:
                break
            symbol = str(parts[0])
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except Exception:
                break
            symbols.append(symbol)
            coords_ang.append([x, y, z])
            idx += 1

        if symbols:
            geometry_bohr = np.array(coords_ang, dtype=float) * ANGSTROM_TO_BOHR
            structures.append(
                Structure(
                    geometry=geometry_bohr,
                    symbols=symbols,
                    charge=charge,
                    multiplicity=multiplicity,
                )
            )
        idx += 1
    return structures


def _parse_path_from_neb_image_files(
    files: dict[str, Any],
    *,
    charge: int,
    multiplicity: int,
    expected_nimages: int | None = None,
    start_structure: Structure | None = None,
) -> list[Structure]:
    image_pattern = re.compile(r"(?:^|/|\\)neb_(\d+)\.xyz$", re.IGNORECASE)
    image_structures: dict[int, Structure] = {}

    for file_name, contents in files.items():
        match = image_pattern.search(str(file_name))
        if match is None:
            continue
        image_idx = int(match.group(1))
        xyz_text = _decode_text(contents)
        with contextlib.suppress(Exception):
            frames = _parse_xyz_structures(
                xyz_text,
                charge=charge,
                multiplicity=multiplicity,
            )
            if frames:
                image_structures[image_idx] = frames[-1]

    if len(image_structures) == 0:
        return []

    ordered = [image_structures[idx] for idx in sorted(image_structures)]
    optim_text = _find_file_text(files, "scr/optim.xyz", "scr.path/optim.xyz", "optim.xyz")
    if optim_text:
        with contextlib.suppress(Exception):
            optim_frames = _parse_xyz_structures(
                optim_text,
                charge=charge,
                multiplicity=multiplicity,
            )
            if optim_frames:
                ordered.append(optim_frames[-1])

    if expected_nimages is not None and start_structure is not None:
        if len(ordered) == expected_nimages - 1:
            ordered = [start_structure] + ordered
        elif len(ordered) > expected_nimages:
            ordered = ordered[:expected_nimages]

    if len(ordered) < 2:
        return []
    return ordered


@dataclass
class DLFindNEB(PathMinimizer):
    initial_chain: Chain
    engine: QCOPEngine
    parameters: object | None = None

    optimized: Chain | None = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    grad_calls_made: int = 0
    geom_grad_calls_made: int = 0

    def __post_init__(self):
        if not isinstance(self.engine, QCOPEngine):
            raise ValueError(
                "DLFindNEB requires QCOPEngine so TeraChem can be invoked via QCOP/ChemCloud."
            )
        if "terachem" not in str(self.engine.program).lower():
            raise ValueError(
                f"DLFindNEB requires a TeraChem-backed engine. Got program={self.engine.program!r}."
            )
        self._params = _as_dict(self.parameters)
        self._verbose = bool(self._params.get("v", False))

    def _status(self, message: str, *, persistent: bool = False) -> None:
        if self._verbose:
            print(message)
            return
        if persistent:
            print_persistent(message=message)
        else:
            update_status(message)

    def _build_terachem_input(self, chain: Chain) -> tuple[str, dict[str, str]]:
        if len(chain) < 2:
            raise ValueError("DLFindNEB requires at least two images.")

        model = dict(getattr(self.engine.program_args, "model", {}) or {})
        user_keywords = dict(getattr(self.engine.program_args, "keywords", {}) or {})
        dlf_keywords = dict(
            self._params.get("dlfind_keywords")
            or self._params.get("dlf_keywords")
            or {}
        )

        nstep = _coerce_int(
            self._params.get("nstep", self._params.get("max_steps", 200)),
            default=200,
            field_name="nstep",
        )
        min_image = _coerce_int(
            self._params.get("min_image", len(chain)),
            default=len(chain),
            field_name="min_image",
        )
        min_nebk = self._params.get("min_nebk", 0.01)
        max_nebk = self._params.get("max_nebk", None)
        new_minimizer = self._params.get("new_minimizer", "no")

        merged_keywords: dict[str, Any] = {}
        merged_keywords.update(user_keywords)
        merged_keywords.update(dlf_keywords)
        merged_keywords.update(
            {
                "nstep": nstep,
                "min_image": min_image,
                "min_nebk": min_nebk,
                "new_minimizer": new_minimizer,
            }
        )
        merged_keywords.setdefault(
            "ts_method",
            self._params.get("ts_method", "neb_frozen"),
        )
        if max_nebk is not None:
            merged_keywords["max_nebk"] = max_nebk

        protected = {
            "basis",
            "method",
            "coordinates",
            "charge",
            "spinmult",
            "run",
            "end",
        }
        lines = [
            f"{'coordinates':<20} path.xyz",
            f"{'charge':<20} {int(chain[0].structure.charge)}",
            f"{'spinmult':<20} {int(chain[0].structure.multiplicity)}",
            f"{'basis':<20} {model.get('basis', '3-21g')}",
            f"{'method':<20} {model.get('method', 'ub3lyp')}",
            f"{'run':<20} ts",
        ]
        for key in sorted(merged_keywords):
            key_l = str(key).strip().lower()
            if key_l in protected:
                continue
            value = merged_keywords[key]
            if value is None:
                continue
            lines.append(f"{key:<20} {_format_tc_value(value)}")
        lines.append("end")

        constraints_text = self._params.get("constraints_text")
        if constraints_text:
            lines.extend(["", str(constraints_text).strip(), ""])

        files = {"tc.in": "\n".join(lines) + "\n", "path.xyz": _iter_xyz_frames(chain)}
        input_files = dict(self._params.get("input_files") or {})
        for file_name, contents in input_files.items():
            files[str(file_name)] = str(contents)

        return files["tc.in"], files

    def _run_dlfind(self, chain: Chain):
        _tcin_text, files = self._build_terachem_input(chain)
        collect_files = bool(self._params.get("collect_files", True))
        file_input = FileInput(
            files=files,
            cmdline_args=["tc.in"],
        )
        return self.engine.compute_func(
            "terachem",
            file_input,
            collect_files=collect_files,
        )

    def _chain_from_output(self, output: Any, chain: Chain) -> tuple[Chain, list[Chain]]:
        files = _extract_output_files(output)
        path_xyz = _find_file_text(
            files,
            "scr/nebpath.xyz",
            "scr.path/nebpath.xyz",
            "nebpath.xyz",
        )
        if path_xyz is None:
            path_xyz = _find_file_text(files, "scr/optim.xyz", "scr.path/optim.xyz", "optim.xyz")
        structures: list[Structure] = []
        if path_xyz:
            structures = _parse_xyz_structures(
                path_xyz,
                charge=int(chain[0].structure.charge),
                multiplicity=int(chain[0].structure.multiplicity),
            )
        if len(structures) < 2:
            path_geometry = _find_file_text(
                files,
                "scr.path/path.geometry",
                "scr/path.geometry",
                "path.geometry",
            )
            if path_geometry:
                structures = _parse_path_geometry_structures(
                    path_geometry,
                    charge=int(chain[0].structure.charge),
                    multiplicity=int(chain[0].structure.multiplicity),
                )
        if len(structures) < 2:
            image_series_structures = _parse_path_from_neb_image_files(
                files,
                charge=int(chain[0].structure.charge),
                multiplicity=int(chain[0].structure.multiplicity),
                expected_nimages=len(chain),
                start_structure=chain[0].structure,
            )
            if image_series_structures:
                structures = image_series_structures
        if len(structures) == 1 and len(chain) >= 2:
            self._status(
                "DL-Find NEB: only one optimized structure returned; building a minimal path from endpoints + optimized point.",
                persistent=True,
            )
            structures = [chain[0].structure, structures[0], chain[-1].structure]
        if len(structures) < 2:
            available = ", ".join(sorted(str(name) for name in files.keys()))
            raise ElectronicStructureError(
                msg=(
                    "DL-Find output path has fewer than two structures. "
                    f"Returned files: {available}"
                ),
                obj=output,
            )

        final_chain = Chain.model_validate(
            {
                "nodes": [StructureNode(structure=struct) for struct in structures],
                "parameters": chain.parameters.copy(),
            }
        )

        nebinfo_text = _find_file_text(files, "scr/nebinfo", "nebinfo")
        energies = None
        if nebinfo_text:
            energies = _parse_last_neb_energies(nebinfo_text, len(final_chain))
        if energies is not None:
            for node, ene in zip(final_chain, energies):
                node._cached_energy = float(ene)
        else:
            self._status(
                "DL-Find NEB: no parseable energies in nebinfo, computing endpoint energies via engine.",
                persistent=True,
            )
            self.engine.compute_energies(final_chain)
            self.grad_calls_made += len(final_chain)

        chain_trajectory = _extract_dlf_image_trajectory(
            files,
            nimages=len(final_chain),
            charge=int(chain[0].structure.charge),
            multiplicity=int(chain[0].structure.multiplicity),
            chain_parameters=chain.parameters.copy(),
        )
        return final_chain, chain_trajectory

    def optimize_chain(self) -> ElemStepResults:
        work_chain = self.initial_chain.copy()
        self.chain_trajectory = [work_chain.copy()]
        self._status("DL-Find NEB: submitting TeraChem `run ts` through QCOP.")

        try:
            output = self._run_dlfind(work_chain)
            final_chain, parsed_history = self._chain_from_output(output, work_chain)
        except ElectronicStructureError:
            raise
        except Exception as exc:
            context = (
                f"path_min_inputs(nstep={self._params.get('nstep')!r}, "
                f"min_image={self._params.get('min_image')!r}, "
                f"min_nebk={self._params.get('min_nebk')!r})"
            )
            raise ElectronicStructureError(
                msg=f"DL-Find NEB execution failed ({type(exc).__name__}): {exc}. {context}",
                obj=None,
            ) from exc

        if parsed_history:
            self.chain_trajectory.extend(parsed_history)
        if not self.chain_trajectory or self.chain_trajectory[-1] is not final_chain:
            self.chain_trajectory.append(final_chain)
        self.optimized = final_chain

        if bool(self._params.get("do_elem_step_checks", True)):
            self._status("DL-Find NEB: running elementary-step checks.")
            elem_step_results = check_if_elem_step(inp_chain=final_chain, engine=self.engine)
            self.geom_grad_calls_made += int(elem_step_results.number_grad_calls)
            return elem_step_results
        return IS_ELEM_STEP
