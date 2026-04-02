from neb_dynamics.engines.engine import Engine
from neb_dynamics.errors import GradientsNotComputedError, ElectronicStructureError
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from neb_dynamics.TreeNode import TreeNode
import neb_dynamics.chainhelpers as ch
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs, RunInputs
from neb_dynamics.neb import NEB
from neb_dynamics.msmep import MSMEP
from neb_dynamics import StructureNode
from neb_dynamics.elementarystep import check_if_elem_step
from qccodec.parsers.terachem import parse_energy, parse_gradient
from qcio import FileInput, view
from chemcloud import compute as ccompute

from qcop import compute
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
import numpy as np
from neb_dynamics.elements import atomic_number_to_symbol
from neb_dynamics.geodesic_interpolation2.coord_utils import align_geom
import numpy as np
import os

from qcio import Structure, ProgramOutput
from neb_dynamics.helper_functions import parse_symbols_from_prmtop, rst7_to_coords_and_indices, parse_qmmm_gradients

from pathlib import Path

import matplotlib.pyplot as plt
import re
import tempfile
import logging
import time


def _resolve_chemcloud_queue(explicit_queue: str | None) -> str:
    if explicit_queue:
        return explicit_queue
    for env_key in ("MEPD_CHEMCLOUD_QUEUE", "CHEMCLOUD_QUEUE", "CCQUEUE"):
        env_val = os.getenv(env_key)
        if env_val:
            return env_val
    return "celery"


@dataclass
class QMMMEngine(Engine):
    """"
    This is what initialization looks like:

    >>> eng = QMMMEngine(tcin_fp=Path("./tc.in"),
                qminds_fp=Path("./qmindices.dat"),
                prmtop_fp=Path("./ref.prmtop"),
                rst7_fp_prod=Path("./optim_product.rst7"),
                rst7_fp_react=Path("./ref.rst7"))
    """
    tcin_fp: Path = None
    tcin_text: str = None
    qminds_fp: Path = None
    prmtop_fp: Path = None
    rst7_fp_prod: Path = None
    rst7_fp_react: Path = None
    compute_program: str = "qcop"
    chemcloud_queue: str | None = None
    print_stdout: bool = False
    debug_dump_inputs: bool = False
    debug_dump_dir: Path | None = None
    frozen_atom_indices: list[int] | str | None = None

    def __post_init__(self):
        self.chemcloud_queue = _resolve_chemcloud_queue(self.chemcloud_queue)
        self._debug_dump_counter = 0
        if isinstance(self.frozen_atom_indices, str):
            self.frozen_atom_indices = [
                int(tok)
                for tok in self.frozen_atom_indices.replace(",", " ").split()
                if tok.strip()
            ]
        elif self.frozen_atom_indices is None:
            self.frozen_atom_indices = []
        else:
            self.frozen_atom_indices = [int(v) for v in self.frozen_atom_indices]
        if self.tcin_text is not None:
            self.inp_file = self.tcin_text
        elif self.tcin_fp is not None:
            self.inp_file = self.tcin_fp.read_text()
        else:
            raise ValueError("QMMMEngine needs either tcin_text or tcin_fp.")
        self.qmindices = self.qminds_fp.read_text()
        self.prmtop = self.prmtop_fp.read_text()
        self.ref_rst7_react = self.rst7_fp_react.read_text()
        self.ref_rst7_prod = self.rst7_fp_prod.read_text() if self.rst7_fp_prod else None
        _, indices_coordinates = rst7_to_coords_and_indices(
            self.ref_rst7_react)
        self.indices_coordinates = indices_coordinates

    def _dump_debug_inputs(self, rst7_strings: list[str]) -> None:
        if not self.debug_dump_inputs or len(rst7_strings) == 0:
            return

        dump_dir = Path(self.debug_dump_dir) if self.debug_dump_dir else Path.cwd(
        ) / "qmmm_debug_inputs"
        dump_dir.mkdir(parents=True, exist_ok=True)

        selected_indices = sorted(
            {0, len(rst7_strings) // 2, len(rst7_strings) - 1})
        labels = {
            0: "first",
            len(rst7_strings) // 2: "middle",
            len(rst7_strings) - 1: "last",
        }

        call_dir = dump_dir / f"call_{self._debug_dump_counter:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        (call_dir / "README.txt").write_text(
            "\n".join(
                [
                    f"call_index: {self._debug_dump_counter}",
                    f"nimages: {len(rst7_strings)}",
                    f"selected_indices: {' '.join(str(i) for i in selected_indices)}",
                    "",
                    "Each node dump contains the exact tc.in and ref.rst7 payload bundled for submission.",
                ]
            )
            + "\n"
        )

        for idx in selected_indices:
            label = labels[idx]
            prefix = call_dir / f"node_{idx:03d}_{label}"
            (prefix.with_suffix(".tc.in")).write_text(self.inp_file)
            (prefix.with_suffix(".rst7")).write_text(rst7_strings[idx])

    def _next_debug_dump_counter(self) -> None:
        self._debug_dump_counter += 1

    def _construct_input(self, rst7_string, tcin_text: str | None = None):
        # Create a FileInput object for TeraChem
        file_inp = FileInput(
            files={
                "tc.in": self.inp_file if tcin_text is None else tcin_text,
                "ref.rst7": rst7_string,
                "qmindices.dat": self.qmindices,
                "ref.prmtop": self.prmtop,
            },
            cmdline_args=["tc.in"],
        )
        return file_inp

    def _with_run_type(self, tcin_text: str, run_type: str) -> str:
        pattern = re.compile(r"(?im)^\s*run\s+\S+\s*$")
        replacement = f"run {run_type}"
        if pattern.search(tcin_text):
            return pattern.sub(replacement, tcin_text, count=1)
        return tcin_text.rstrip() + f"\n\n# Runtype\n{replacement}\n"

    def _apply_tcin_overrides(self, tcin_text: str, overrides: dict | None) -> str:
        if not overrides:
            return tcin_text
        out = tcin_text
        for key, value in overrides.items():
            line = f"{key} {value}"
            pattern = re.compile(rf"(?im)^\s*{re.escape(str(key))}\s+.*$")
            if pattern.search(out):
                out = pattern.sub(line, out, count=1)
            else:
                out = out.rstrip() + f"\n{line}\n"
        return out

    @staticmethod
    def _extract_output_files(output) -> dict | None:
        for files_obj in (
            getattr(output, "files", None),
            getattr(getattr(output, "results", None), "files", None),
            getattr(getattr(output, "data", None), "files", None),
        ):
            if files_obj:
                return files_obj
        return None

    def _extract_optimization_structures(self, output, reference_structure: Structure) -> list[Structure]:
        trajectory = None
        for tr in (
            getattr(getattr(output, "results", None), "trajectory", None),
            getattr(getattr(output, "data", None), "trajectory", None),
            getattr(output, "trajectory", None),
        ):
            if tr:
                trajectory = tr
                break

        if trajectory:
            structures = []
            for entry in trajectory:
                struct = getattr(getattr(entry, "input_data", None), "structure", None)
                if struct is not None:
                    structures.append(struct)
            if len(structures) > 0:
                return structures

        files = self._extract_output_files(output)
        if files:
            for key, contents in files.items():
                if str(key).endswith("optim.xyz") or str(key) == "optim.xyz":
                    with tempfile.TemporaryDirectory() as td:
                        xyz_fp = Path(td) / "optim.xyz"
                        if isinstance(contents, bytes):
                            xyz_fp.write_bytes(contents)
                        else:
                            xyz_fp.write_text(str(contents))
                        return Structure.open_multi(
                            xyz_fp,
                            charge=reference_structure.charge,
                            multiplicity=reference_structure.multiplicity,
                        )

        return []

    def _compute_minimize_output(self, rst7_string: str, keywords: dict | None = None):
        tcin_text = self._with_run_type(self.inp_file, "minimize")
        min_kwds = self._normalize_minimize_keywords(keywords)
        # TeraChem's new minimizer is unstable for this QMMM workflow; force legacy minimizer.
        min_kwds["new_minimizer"] = "no"
        tcin_text = self._apply_tcin_overrides(
            tcin_text, min_kwds
        )
        inp = self._construct_input(rst7_string, tcin_text=tcin_text)
        self._dump_debug_inputs([rst7_string])
        try:
            if self.compute_program.lower() != "chemcloud":
                return compute(
                    "terachem",
                    inp,
                    print_stdout=self.print_stdout,
                    collect_files=True,
                )
            return self._chemcloud_compute_with_retries(
                inp,
                collect_files=True,
            )
        except Exception as exc:
            raise ElectronicStructureError(
                msg=f"QMMM minimize submission failed ({self.compute_program}): {exc}",
                obj=None,
            ) from exc
        finally:
            self._next_debug_dump_counter()

    def _compute_enegrad(self, rst7_strings):

        inputs = [self._construct_input(string) for string in rst7_strings]
        qminds = [int(x) for x in self.qmindices.strip().split("\n")]
        self._dump_debug_inputs(rst7_strings)

        try:
            try:
                if self.compute_program.lower() != 'chemcloud':
                    outputs = []
                    for string in inputs:
                        outputs.append(
                            compute(
                                "terachem",
                                string,
                                print_stdout=self.print_stdout,
                            )
                        )
                else:
                    batch_input = inputs[0] if len(inputs) == 1 else inputs
                    outputs = self._chemcloud_compute_with_retries(
                        batch_input,
                        collect_files=False,
                    )
                    if len(inputs) == 1 and not isinstance(outputs, list):
                        outputs = [outputs]
            except Exception as exc:
                raise ElectronicStructureError(
                    msg=f"QMMM compute submission failed ({self.compute_program}): {exc}",
                    obj=None,
                ) from exc

            if len(outputs) != len(rst7_strings):
                raise ElectronicStructureError(
                    msg=(
                        "QMMM returned an unexpected number of results: "
                        f"expected {len(rst7_strings)}, got {len(outputs)}"
                    ),
                    obj=outputs,
                )

            gradients = []
            energies = []
            for output in outputs:
                qm_grad, mm_grad = parse_qmmm_gradients(output.stdout)
                if len(qm_grad) == 0:
                    raise ElectronicStructureError(
                        msg="QMMM gradient parsing failed (no QM gradients found in output).",
                        obj=output,
                    )
                lines = output.stdout.split("\n")
                nlink_atom = int(
                    [l for l in lines if "link" in l][0].split()[6])
                if nlink_atom > 0:
                    qm_grad = qm_grad[:-nlink_atom]  # Get rid of linked-atoms

                gradient = np.zeros((len(qm_grad)+len(mm_grad), 3))
                # print("len qmgrad:", len(qm_grad))
                # print("len mmgrad:", len(mm_grad))
                allinds = list(range(len(gradient)))
                mminds = np.delete(allinds, qminds)
                gradient[qminds] = qm_grad
                if len(mminds) > 0:
                    gradient[mminds] = mm_grad

                # Parses FINAL ENERGY: line
                energy = parse_energy(output.stdout)

                energies.append(energy)
                gradients.append(gradient)

            return energies, gradients, outputs
        finally:
            self._next_debug_dump_counter()

    @staticmethod
    def _is_retryable_chemcloud_error(exc: Exception) -> bool:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code is not None:
            return int(status_code) >= 500
        msg = str(exc).lower()
        retry_markers = (
            "502",
            "503",
            "504",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "temporarily unavailable",
            "connection reset",
            "timed out",
        )
        return any(token in msg for token in retry_markers)

    def _chemcloud_compute_with_retries(self, inp, collect_files: bool):
        max_attempts = 3
        delay_sec = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                out = ccompute(
                    "terachem",
                    inp,
                    queue=self.chemcloud_queue,
                    collect_files=collect_files,
                )
                if hasattr(out, "get"):
                    out = out.get()
                return out
            except Exception as exc:
                retryable = self._is_retryable_chemcloud_error(exc)
                if attempt >= max_attempts or not retryable:
                    raise
                logging.warning(
                    "QMMM ChemCloud call failed on attempt %d/%d (%s). Retrying in %.1fs...",
                    attempt,
                    max_attempts,
                    exc,
                    delay_sec,
                )
                time.sleep(delay_sec)
                delay_sec *= 2.0

    def compute_energies(self, chain: Chain):
        self.compute_gradients(chain)
        enes = np.array([node.energy for node in chain])
        return enes

    def compute_gradients(self, chain):
        try:
            grads = np.array([node.gradient for node in chain])

        except GradientsNotComputedError:
            qminds = [int(v) for v in self.qmindices.strip().split("\n")]

            rst7strings = [self.structure_to_rst7(
                qmstructure=node.structure) for node in chain]

            # print(rst7strings[0])
            energies, gradients, outputs = self._compute_enegrad(rst7strings)
            for node, energy, gradient in zip(chain, energies, gradients):
                # print("\nEnergy:", energy, "\n")
                node._cached_energy = energy
                node._cached_gradient = gradient
                # node._cached_result = enegrad_output[2]
                node._cached_result = None

            if not all([node._cached_gradient is not None for node in chain]):
                failed_results = []
                for node in chain:
                    if node._cached_result is not None and node._cached_gradient is None:
                        failed_results.append(node._cached_result)
                raise ElectronicStructureError(
                    msg="Gradient calculation failed.", obj=failed_results)
            grads = np.array([node.gradient for node in chain])

        return grads

    def compute_geometry_optimization(
        self,
        node: StructureNode,
        keywords: dict | None = None,
    ) -> list[StructureNode]:
        if keywords is None:
            keywords = {"coordsys": "cart", "maxit": 500}
        rst7_string = self.structure_to_rst7(node.structure)
        output = self._compute_minimize_output(rst7_string=rst7_string, keywords=keywords)
        structures = self._extract_optimization_structures(output, reference_structure=node.structure)
        if len(structures) == 0:
            raise ElectronicStructureError(
                msg="QMMM minimize completed but no optimization trajectory was found (optim.xyz missing).",
                obj=output,
            )
        return [StructureNode(structure=struct) for struct in structures]

    @staticmethod
    def _normalize_minimize_keywords(keywords: dict | None) -> dict:
        if not keywords:
            return {}

        normalized = {}
        for key, value in keywords.items():
            k = str(key).strip().lower()
            if k in {"maxiter", "max_iter"}:
                normalized["maxit"] = value
            elif k in {"coordsys", "coord_sys", "coordinates", "coordinate_system"}:
                coord_val = str(value).strip().lower()
                normalized["min_coordinates"] = "cartesian" if coord_val == "cart" else value
            else:
                normalized[str(key)] = value
        return normalized

    def _run_geometric_transition_state(
        self,
        node: StructureNode,
        keywords: dict | None = None,
    ):
        try:
            import geometric
            import geometric.engine
            import geometric.molecule
            import geometric.optimize
        except ImportError as exc:
            raise ElectronicStructureError(
                msg=(
                    "QMMM transition-state optimization requires the "
                    "`geometric` package to be installed."
                ),
                obj=None,
            ) from exc

        @contextmanager
        def _geometric_log_context():
            if not self.print_stdout:
                yield
                return

            logger_names = ("geometric", "geometric.nifty", "geometric.optimize")
            snapshots = []
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(logging.Formatter("%(message)s"))

            for logger_name in logger_names:
                logger = logging.getLogger(logger_name)
                snapshots.append(
                    (logger, logger.disabled, logger.level, logger.propagate, list(logger.handlers))
                )
                logger.disabled = False
                logger.setLevel(logging.INFO)
                logger.propagate = False
                logger.addHandler(stream_handler)

            try:
                yield
            finally:
                for logger, disabled, level, propagate, handlers in snapshots:
                    logger.handlers = handlers
                    logger.disabled = disabled
                    logger.setLevel(level)
                    logger.propagate = propagate

        class _GeometricQMMMEngine(geometric.engine.Engine):
            def __init__(self, molecule, qmmm_engine, ref_node, active_atom_indices):
                super().__init__(molecule)
                self.qmmm_engine = qmmm_engine
                self.ref_node = ref_node
                self.active_atom_indices = np.array(active_atom_indices, dtype=int)

            def calc_new(self, coords, dirname):
                curr_coords = np.array(coords, dtype=float).reshape((-1, 3))
                full_coords = np.array(self.ref_node.coords, dtype=float).copy()
                full_coords[self.active_atom_indices] = curr_coords
                curr_node = self.ref_node.copy().update_coords(full_coords)
                curr_node.has_molecular_graph = False
                curr_node.graph = None
                energy = self.qmmm_engine.compute_energies([curr_node])[0]
                gradient = self.qmmm_engine.compute_gradients([curr_node])[0]
                gradient = np.array(gradient)[self.active_atom_indices]
                # geomeTRIC expects gradient in Hartree/Angstrom.
                return {
                    "energy": energy,
                    "gradient": np.array(gradient).reshape(-1) * BOHR_TO_ANGSTROMS,
                }

        ts_kwds = dict(keywords or {})
        kwd_frozen = ts_kwds.pop("frozen_atom_indices", None)
        if kwd_frozen is not None:
            if isinstance(kwd_frozen, str):
                frozen_atom_indices = [
                    int(tok)
                    for tok in kwd_frozen.replace(",", " ").split()
                    if tok.strip()
                ]
            else:
                frozen_atom_indices = [int(v) for v in kwd_frozen]
        else:
            frozen_atom_indices = list(self.frozen_atom_indices or [])

        n_atoms = len(node.structure.symbols)
        frozen_set = {i for i in frozen_atom_indices if 0 <= i < n_atoms}
        active_atom_indices = [i for i in range(n_atoms) if i not in frozen_set]
        if len(active_atom_indices) == 0:
            raise ElectronicStructureError(
                msg="All atoms are frozen for TS optimization; need at least one active atom.",
                obj=None,
            )

        molecule = geometric.molecule.Molecule()
        molecule.elem = [node.structure.symbols[i] for i in active_atom_indices]
        molecule.xyzs = [node.structure.geometry[active_atom_indices] * ANGSTROM_TO_BOHR]
        ref_node = node.copy()
        ref_node.has_molecular_graph = False
        ref_node.graph = None
        custom_engine = _GeometricQMMMEngine(
            molecule=molecule,
            qmmm_engine=self,
            ref_node=ref_node,
            active_atom_indices=active_atom_indices,
        )

        ts_keywords = {
            "check": 1,
            "transition": True,
            "converge": ["gmax", "1.0e-5"],
            "trust": 0.1,
            "tmax": 0.3,
            "maxiter": 800,
        }
        if ts_kwds:
            ts_keywords.update(ts_kwds)
        ts_keywords["transition"] = True

        with _geometric_log_context():
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmpf:
                output = geometric.optimize.run_optimizer(
                    customengine=custom_engine,
                    input=tmpf.name,
                    **ts_keywords,
                )
                setattr(output, "_active_atom_indices", active_atom_indices)
                return output

    def compute_transition_state(
        self,
        node: StructureNode,
        keywords: dict | None = None,
    ) -> StructureNode:
        output = self._run_geometric_transition_state(node=node, keywords=keywords)
        xyzs = getattr(output, "xyzs", None)
        if not xyzs:
            raise ElectronicStructureError(
                msg=(
                    "QMMM transition-state optimization completed but no trajectory "
                    "coordinates were returned."
                ),
                obj=output,
            )

        active_atom_indices = getattr(output, "_active_atom_indices", None)
        if active_atom_indices is None:
            active_atom_indices = list(range(len(node.structure.symbols)))
        final_active_bohr = np.array(xyzs[-1]) * ANGSTROM_TO_BOHR
        final_coords_bohr = np.array(node.coords, dtype=float).copy()
        final_coords_bohr[np.array(active_atom_indices, dtype=int)] = final_active_bohr
        ts_node = node.copy().update_coords(final_coords_bohr)
        ts_node.has_molecular_graph = False
        ts_node.graph = None
        ts_node._cached_energy = self.compute_energies([ts_node])[0]
        return ts_node

    def structure_to_rst7(self, qmstructure):
        """
        takes a reference string of the rst7 file 'reference_string',
        indices of QM section of rst7 file, 'qmindices',
        indices of coordinates section of rst7 file, 'coordinate_indices',
        and a Structure object whose coordinates will be swapping in, 'qmstructure'
        """
        # reference_coords, _ = rst7_to_coords_and_indices(self.ref_rst7_react)

        # LMAO , removing alignment. Comment out next line if u want alignment
        # aligned_geom = (np.array(qmstructure.geometry)*(1/ANGSTROM_TO_BOHR))[qmindices]
        aligned_geom = (np.array(qmstructure.geometry)*(1/ANGSTROM_TO_BOHR))
        # reference_coords[qmindices] = aligned_geom
        reference_coords = aligned_geom

        arr = np.array([46.4274321, 82.5659739, 37.1465461,
                       9.8415857, 51.2254588, 44.5854156])

        # Determine the maximum width of the integer part
        max_int_width = 0
        for x in arr:
            int_part_len = len(str(int(abs(x))))  # Handle negative numbers
            if x < 0:
                int_part_len += 1  # Account for the negative sign
            if int_part_len > max_int_width:
                max_int_width = int_part_len

        # Define a custom formatter to align numbers by decimal point and specify precision
        def format_float_with_precision(x):
            return f"{x:{max_int_width + 8}.7f}"

        string = ""
        i = 0
        for j, _ in enumerate(reference_coords):
            atom = reference_coords[j]
            if i == 0:
                string += "  "+np.array2string(atom, separator='  ', prefix=' ', max_line_width=1e7, formatter={
                                               'float': format_float_with_precision})[1:-1]+" "
                i += 1
            elif i == 1:
                string += np.array2string(atom, separator='  ', prefix=' ', max_line_width=1e7, formatter={
                                          'float': format_float_with_precision})[1:-1]+"\n"
                i = 0

        reference_string = self.ref_rst7_react

        headers = reference_string.split("\n")[:2]
        # headers+=velocities

        newfile = "\n".join(headers)
        newfile += '\n'
        newfile += string
        # print(newfile)

        return newfile


def rst7prmtop_to_structure(rst7_str, prmtopdata_str):
    xyz_coords, coord_indices = rst7_to_coords_and_indices(rst7_str)
    symbols = np.array(parse_symbols_from_prmtop(prmtopdata_str))
    structure = Structure(geometry=xyz_coords *
                          ANGSTROM_TO_BOHR, symbols=symbols)
    return structure
