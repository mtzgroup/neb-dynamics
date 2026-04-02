from neb_dynamics.engines.engine import Engine
from neb_dynamics.errors import GradientsNotComputedError, ElectronicStructureError
from dataclasses import dataclass
from pathlib import Path
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
from time import sleep


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
    geometry_optimizer: str = "native"
    chemcloud_queue: str | None = None
    print_stdout: bool = False
    debug_dump_inputs: bool = False
    debug_dump_dir: Path | None = None
    frozen_atom_indices: list[int] | None = None
    chemcloud_retry_attempts: int = 3
    chemcloud_retry_delay_seconds: float = 2.0

    def __post_init__(self):
        self.chemcloud_queue = _resolve_chemcloud_queue(self.chemcloud_queue)
        self._debug_dump_counter = 0
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
        if isinstance(self.frozen_atom_indices, tuple):
            self.frozen_atom_indices = list(self.frozen_atom_indices)
        if isinstance(self.frozen_atom_indices, list):
            self.frozen_atom_indices = sorted(
                {
                    int(ind)
                    for ind in self.frozen_atom_indices
                    if int(ind) >= 0
                }
            )
        else:
            self.frozen_atom_indices = None

    def _dump_debug_inputs(self, rst7_strings: list[str], tcin_text: str | None = None) -> None:
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
            (prefix.with_suffix(".tc.in")).write_text(self.inp_file if tcin_text is None else tcin_text)
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
    def _coerce_files_mapping(files_obj) -> dict | None:
        if files_obj is None:
            return None
        if isinstance(files_obj, dict):
            return files_obj
        direct = getattr(files_obj, "files", None)
        if isinstance(direct, dict):
            return direct
        if hasattr(files_obj, "items"):
            try:
                return dict(files_obj.items())
            except Exception:
                return None
        return None

    @classmethod
    def _extract_output_files(cls, output) -> dict | None:
        for files_obj in (
            getattr(output, "files", None),
            getattr(getattr(output, "results", None), "files", None),
            getattr(getattr(output, "data", None), "files", None),
        ):
            files_map = cls._coerce_files_mapping(files_obj)
            if files_map:
                return files_map
        return None

    @staticmethod
    def _decode_file_contents(contents) -> str:
        if isinstance(contents, bytes):
            return contents.decode()
        return str(contents)

    @staticmethod
    def _coords_from_rst7_text(rst7_text: str) -> np.ndarray:
        lines = [line for line in rst7_text.splitlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError("RST7 content is too short.")
        natom = int(lines[1].split()[0])
        if natom <= 0:
            raise ValueError("RST7 content has no atoms.")

        numbers: list[float] = []
        for token in " ".join(lines[2:]).split():
            try:
                numbers.append(float(token))
            except ValueError:
                continue
            if len(numbers) >= 3 * natom:
                break
        if len(numbers) < 3 * natom:
            raise ValueError("RST7 content does not contain enough coordinate values.")
        return np.array(numbers[: 3 * natom], dtype=float).reshape(natom, 3)

    def _structure_from_rst7_contents(self, contents, reference_structure: Structure) -> Structure | None:
        try:
            rst7_text = self._decode_file_contents(contents)
            coords_angstrom = self._coords_from_rst7_text(rst7_text)
            if len(coords_angstrom) != len(reference_structure.symbols):
                return None
            return Structure(
                geometry=np.array(coords_angstrom) * ANGSTROM_TO_BOHR,
                symbols=reference_structure.symbols,
                charge=reference_structure.charge,
                multiplicity=reference_structure.multiplicity,
            )
        except Exception:
            return None

    def _extract_optimization_structures(self, output, reference_structure: Structure) -> list[Structure]:
        for struct in (
            getattr(getattr(output, "results", None), "structure", None),
            getattr(getattr(output, "data", None), "structure", None),
            getattr(getattr(output, "data", None), "final_structure", None),
            getattr(output, "structure", None),
        ):
            if struct is not None:
                return [struct]

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
            rst7_items = [
                (str(key), contents)
                for key, contents in files.items()
                if ".rst7" in str(key).lower()
            ]
            rst7_items.sort(
                key=lambda kv: (
                    0 if Path(kv[0]).name.lower() == "optim.rst7" else 1,
                    0 if "optim" in kv[0].lower() else 1,
                    kv[0].lower(),
                )
            )
            for _, contents in rst7_items:
                struct = self._structure_from_rst7_contents(
                    contents, reference_structure=reference_structure
                )
                if struct is not None:
                    return [struct]

            xyz_items = [
                (str(key), contents)
                for key, contents in files.items()
                if str(key).lower().endswith(".xyz")
            ]
            xyz_items.sort(
                key=lambda kv: (
                    0 if any(token in kv[0].lower() for token in ("optim", "opt", "min")) else 1,
                    kv[0].lower(),
                )
            )
            for key, contents in xyz_items:
                with tempfile.TemporaryDirectory() as td:
                    xyz_fp = Path(td) / Path(key).name
                    if isinstance(contents, bytes):
                        xyz_fp.write_bytes(contents)
                    else:
                        xyz_fp.write_text(str(contents))
                    try:
                        structures = Structure.open_multi(
                            xyz_fp,
                            charge=reference_structure.charge,
                            multiplicity=reference_structure.multiplicity,
                        )
                    except Exception:
                        continue
                    if len(structures) > 0:
                        return structures

        return []

    @staticmethod
    def _is_retryable_chemcloud_exception(exc: Exception) -> bool:
        msg = str(exc).lower()
        retryable_markers = (
            "500 internal server error",
            "error collecting task",
            "validation errors for programoutput",
            "extra inputs are not permitted",
            "field required",
        )
        return any(marker in msg for marker in retryable_markers)

    @staticmethod
    def _is_retryable_chemcloud_output(output) -> bool:
        if output is None or getattr(output, "success", True):
            return False
        payload = "\n".join(
            str(getattr(output, attr, "") or "")
            for attr in ("logs", "stdout", "traceback")
        ).lower()
        return "500 internal server error" in payload or "error collecting task" in payload

    @staticmethod
    def _normalize_single_output(output):
        if isinstance(output, list):
            if len(output) != 1:
                raise ElectronicStructureError(
                    msg=f"QMMM minimize expected one output, got {len(output)}.",
                    obj=output,
                )
            return output[0]
        return output

    def _with_frozen_atom_constraints(self, tcin_text: str) -> str:
        if not self.frozen_atom_indices:
            return tcin_text
        if re.search(r"(?im)^\s*\$constraints\b", tcin_text):
            return tcin_text

        # TeraChem constraint atoms are 1-indexed.
        constraint_lines = ["$constraints"]
        constraint_lines.extend(f"atom {atom_index + 1}" for atom_index in self.frozen_atom_indices)
        constraint_lines.append("$end")
        return tcin_text.rstrip() + "\n\n" + "\n".join(constraint_lines) + "\n"

    def _compute_minimize_output(self, rst7_string: str, keywords: dict | None = None):
        tcin_text = self._with_run_type(self.inp_file, "minimize")
        tcin_text = self._apply_tcin_overrides(tcin_text, keywords or {})
        tcin_text = self._with_frozen_atom_constraints(tcin_text)
        inp = self._construct_input(rst7_string, tcin_text=tcin_text)
        self._dump_debug_inputs([rst7_string], tcin_text=tcin_text)
        try:
            if self.compute_program.lower() != "chemcloud":
                return compute(
                    "terachem",
                    inp,
                    print_stdout=self.print_stdout,
                    collect_files=True,
                )

            attempts = max(1, int(self.chemcloud_retry_attempts))
            for attempt in range(1, attempts + 1):
                try:
                    out = ccompute(
                        "terachem",
                        inp,
                        queue=self.chemcloud_queue,
                        collect_files=True,
                    )
                    if hasattr(out, "get"):
                        out = out.get()
                    out = self._normalize_single_output(out)

                    if self._is_retryable_chemcloud_output(out) and attempt < attempts:
                        sleep(self.chemcloud_retry_delay_seconds * attempt)
                        continue

                    if getattr(out, "success", True) is False:
                        raise ElectronicStructureError(
                            msg=(
                                "QMMM minimize failed on ChemCloud. "
                                "See output logs/traceback on attached object."
                            ),
                            obj=out,
                        )
                    return out
                except ElectronicStructureError:
                    raise
                except Exception as exc:
                    if self._is_retryable_chemcloud_exception(exc) and attempt < attempts:
                        sleep(self.chemcloud_retry_delay_seconds * attempt)
                        continue
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
                    outputs = ccompute(
                        "terachem", batch_input, queue=self.chemcloud_queue)
                    if hasattr(outputs, "get"):
                        outputs = outputs.get()
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

    def compute_geometry_optimization(self, node: StructureNode, keywords={'coordsys': 'cart', 'maxit': 500}) -> list[StructureNode]:
        rst7_string = self.structure_to_rst7(node.structure)
        output = self._compute_minimize_output(rst7_string=rst7_string, keywords=keywords)
        structures = self._extract_optimization_structures(output, reference_structure=node.structure)
        if len(structures) == 0:
            raise ElectronicStructureError(
                msg="QMMM minimize completed but no parseable optimized structures were found in output.",
                obj=output,
            )
        return [StructureNode(structure=struct) for struct in structures]

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

        class _GeometricQMMMEngine(geometric.engine.Engine):
            def __init__(self, molecule, qmmm_engine, ref_node):
                super().__init__(molecule)
                self.qmmm_engine = qmmm_engine
                self.ref_node = ref_node

            def calc_new(self, coords, dirname):
                curr_node = self.ref_node.copy().update_coords(np.array(coords))
                energy = self.qmmm_engine.compute_energies([curr_node])[0]
                gradient = self.qmmm_engine.compute_gradients([curr_node])[0]
                # geomeTRIC expects gradient in Hartree/Angstrom.
                return {
                    "energy": energy,
                    "gradient": np.array(gradient).reshape(-1) * BOHR_TO_ANGSTROMS,
                }

        molecule = geometric.molecule.Molecule()
        molecule.elem = list(node.structure.symbols)
        molecule.xyzs = [node.structure.geometry * ANGSTROM_TO_BOHR]
        custom_engine = _GeometricQMMMEngine(
            molecule=molecule,
            qmmm_engine=self,
            ref_node=node,
        )

        ts_keywords = {
            "check": 1,
            "transition": True,
            "converge": ["gmax", "1.0e-5"],
            "trust": 0.1,
            "tmax": 0.3,
            "maxiter": 800,
        }
        if keywords:
            ts_keywords.update(keywords)
        ts_keywords["transition"] = True

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmpf:
            return geometric.optimize.run_optimizer(
                customengine=custom_engine,
                input=tmpf.name,
                **ts_keywords,
            )

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

        final_coords_bohr = np.array(xyzs[-1]) * ANGSTROM_TO_BOHR
        ts_node = node.copy().update_coords(final_coords_bohr)
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
