from __future__ import annotations

import concurrent
import inspect
import contextlib
from dataclasses import dataclass
from typing import Any, List, Union
import os
import threading
from pathlib import Path
import tempfile
import time
import logging
from numpy.typing import NDArray
import numpy as np
from pydantic import ValidationError

import qcop
from qcop.exceptions import ExternalProgramError
from qcio.models.inputs import DualProgramInput, ProgramInput, ProgramArgs, FileInput
from qcio import ProgramOutput, Structure
import shutil

from chemcloud import CCClient
from chemcloud import configure_client as cc_configure_client
from chemcloud.config import Settings as ChemCloudSettings

from neb_dynamics.chain import Chain
from neb_dynamics.engines.engine import Engine
from neb_dynamics.errors import GradientsNotComputedError, ElectronicStructureError
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.nodes.nodehelpers import update_node_cache, displace_by_dr
from neb_dynamics.qcio_structure_helpers import _change_prog_input_property
from neb_dynamics.dynamics.chainbiaser import ChainBiaser
import copy

AVAIL_PROGRAMS = ["qcop", "chemcloud"]


TERACHEM_NANOREACTOR_PRESETS: dict[str, dict[str, object]] = {
    "conservative": {
        "description": (
            "Baseline spherical-piston MD: nstep=2000, timestep=0.5, "
            "langevin thermostat, md_r1/r2=6.0/4.0, md_k1/k2=3.0/5.0, "
            "mdbc_t1/t2=750/250."
        ),
        "values": {},
    },
    "fast-oscillating": {
        "description": (
            "Shorter piston cycle without extending runtime: nstep=1600, "
            "md_r1/r2=5.8/3.6, md_k1/k2=3.5/6.0, mdbc_t1/t2=300/100, "
            "frame_stride=5."
        ),
        "values": {
            "terachem_nstep": 1600,
            "terachem_md_r1": 5.8,
            "terachem_md_r2": 3.6,
            "terachem_md_k1": 3.5,
            "terachem_md_k2": 6.0,
            "terachem_mdbc_t1": 300,
            "terachem_mdbc_t2": 100,
            "terachem_frame_stride": 5,
        },
    },
    "hot-fast-oscillating": {
        "description": (
            "Shorter, hotter piston cycling: nstep=1500, tinit/t0=1600/1900, "
            "lnvtime=120, md_r1/r2=5.6/3.4, md_k1/k2=4.0/7.0, mdbc_t1/t2=250/75, "
            "frame_stride=5."
        ),
        "values": {
            "terachem_nstep": 1500,
            "terachem_tinit": 1600.0,
            "terachem_t0": 1900.0,
            "terachem_lnvtime": 120.0,
            "terachem_md_r1": 5.6,
            "terachem_md_r2": 3.4,
            "terachem_md_k1": 4.0,
            "terachem_md_k2": 7.0,
            "terachem_mdbc_t1": 250,
            "terachem_mdbc_t2": 75,
            "terachem_frame_stride": 5,
        },
    },
}


def _resolve_chemcloud_queue(explicit_queue: str | None) -> str:
    """Queue precedence: explicit (TOML) > env var > default."""
    if explicit_queue:
        return explicit_queue
    for env_key in ("MEPD_CHEMCLOUD_QUEUE", "CHEMCLOUD_QUEUE", "CCQUEUE"):
        env_val = os.getenv(env_key)
        if env_val:
            return env_val
    return "celery"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _configure_chemcloud_client(queue: str) -> None:
    """Handle ChemCloud client API drift across installed versions."""
    cc_configure_client(**_chemcloud_client_kwargs(queue))


def _chemcloud_client_kwargs(queue: str) -> dict[str, Any]:
    """Build kwargs compatible with the installed CCClient constructor."""
    client_params = {}
    settings_kwargs = {
        "chemcloud_queue": queue,
        "chemcloud_concurrency": _env_int("MEPD_CHEMCLOUD_CONCURRENCY", 1),
        "chemcloud_connect_timeout": _env_int("MEPD_CHEMCLOUD_CONNECT_TIMEOUT", 120),
        "chemcloud_read_timeout": _env_int("MEPD_CHEMCLOUD_READ_TIMEOUT", 900),
        "chemcloud_write_timeout": _env_int("MEPD_CHEMCLOUD_WRITE_TIMEOUT", 120),
        "chemcloud_pool_timeout": _env_int("MEPD_CHEMCLOUD_POOL_TIMEOUT", 120),
    }

    try:
        ccclient_params = inspect.signature(CCClient).parameters
    except Exception:
        ccclient_params = {}

    if "queue" in ccclient_params:
        client_params["queue"] = queue
    else:
        client_params["chemcloud_queue"] = queue

    direct_setting_keys = {
        key for key in settings_kwargs if key in ccclient_params and key != "chemcloud_queue"
    }
    for key in direct_setting_keys:
        client_params[key] = settings_kwargs.pop(key)

    if "settings" in ccclient_params and settings_kwargs:
        client_params["settings"] = ChemCloudSettings(**settings_kwargs)
    else:
        client_params.update(settings_kwargs)

    return client_params


def _resolve_nanoreactor_inputs(nanoreactor_inputs: dict[str, object] | None) -> dict[str, object]:
    resolved = dict(nanoreactor_inputs or {})
    preset_name = str(resolved.get("preset", "conservative")
                      or "conservative").strip().lower()
    preset = TERACHEM_NANOREACTOR_PRESETS.get(preset_name)
    if preset is None:
        supported = ", ".join(sorted(TERACHEM_NANOREACTOR_PRESETS))
        raise ValueError(
            f"Unsupported nanoreactor preset `{preset_name}`. Available presets: {supported}.")
    resolved = {**dict(preset.get("values", {})), **resolved}
    resolved["preset"] = preset_name
    resolved.setdefault("preset_description", str(
        preset.get("description", "")))
    return resolved


@dataclass
class QCOPEngine(Engine):
    program_args: ProgramArgs = ProgramArgs(
        model={"method": "gfn2", "basis": "gfn2"},)
    program: str = "crest"
    geometry_optimizer: str = "geometric"
    compute_program: str = "qcop"
    chemcloud_queue: str | None = None
    biaser: ChainBiaser = None
    collect_files: bool = False
    write_qcio: bool = False
    print_stdout: bool = False

    def __post_init__(self):
        self.chemcloud_queue = _resolve_chemcloud_queue(self.chemcloud_queue)
        self._chemcloud_client_lock = threading.Lock()
        self._chemcloud_clients_by_thread: dict[int, CCClient] = {}
        self._chemcloud_client_params: dict[str, Any] = _chemcloud_client_kwargs(self.chemcloud_queue)
        if self.write_qcio:
            logging.warning(
                "QCOPEngine write_qcio=True: cached qcio.ProgramOutput objects will be "
                "written when results are saved to disk. This can consume substantial disk space."
            )
        if self.compute_program == "chemcloud":
            _configure_chemcloud_client(self.chemcloud_queue)

    def _get_thread_chemcloud_client(self) -> CCClient:
        thread_id = int(threading.get_ident())
        with self._chemcloud_client_lock:
            client = self._chemcloud_clients_by_thread.get(thread_id)
            if client is None:
                client = CCClient(**dict(self._chemcloud_client_params))
                self._chemcloud_clients_by_thread[thread_id] = client
        return client

    def _drop_thread_chemcloud_client(self) -> None:
        thread_id = int(threading.get_ident())
        with self._chemcloud_client_lock:
            self._chemcloud_clients_by_thread.pop(thread_id, None)

    @staticmethod
    def _is_retryable_chemcloud_error(exc: Exception) -> bool:
        status_code = getattr(
            getattr(exc, "response", None), "status_code", None)
        if status_code is not None:
            return int(status_code) >= 500
        msg = str(exc).lower()
        retry_markers = (
            "502",
            "503",
            "504",
            "bad gateway",
            "all connection attempts failed",
            "connecterror",
            "service unavailable",
            "gateway timeout",
            "temporarily unavailable",
            "connection reset",
            "timed out",
            "connecttimeout",
            "different event loop",
            "client has been closed",
        )
        return any(token in msg for token in retry_markers)

    @staticmethod
    def _is_chemcloud_output_fetch_error(exc: Exception) -> bool:
        request = getattr(exc, "request", None)
        request_url = str(getattr(request, "url", "") or "")
        if "/compute/output/" in request_url:
            return True
        msg = str(exc).lower()
        return "compute/output/" in msg

    def _chemcloud_compute_with_retries(self, *args, **kwargs):
        max_attempts = 5
        delay_sec = 5.0
        for attempt in range(1, max_attempts + 1):
            try:
                return self._get_thread_chemcloud_client().compute(*args, **kwargs)
            except Exception as exc:
                if self._is_chemcloud_output_fetch_error(exc):
                    # A fetch failure means the job was already submitted; retrying here
                    # resubmits duplicate work instead of retrying the fetch.
                    raise
                lowered = str(exc).lower()
                if "different event loop" in lowered or "client has been closed" in lowered:
                    # The underlying async client got tied to another loop/thread.
                    self._drop_thread_chemcloud_client()
                retryable = self._is_retryable_chemcloud_error(exc)
                if attempt >= max_attempts or not retryable:
                    raise
                logging.warning(
                    "QCOP ChemCloud call failed on attempt %d/%d (%s). Retrying in %.1fs...",
                    attempt,
                    max_attempts,
                    exc,
                    delay_sec,
                )
                time.sleep(delay_sec)

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        try:
            grads = np.array([node.gradient for node in chain])

        except GradientsNotComputedError:
            node_list = self._run_calc(chain=chain, calctype="gradient")
            if not all([node._cached_gradient is not None for node in node_list]):
                failed_results = []
                for node in node_list:
                    if node._cached_result is not None and node._cached_gradient is None:
                        failed_results.append(node._cached_result)
                raise ElectronicStructureError(
                    msg="Gradient calculation failed.", obj=failed_results)
            grads = np.array([node.gradient for node in node_list])

        if self.biaser:
            new_grads = grads.copy()
            for i, node in enumerate(chain):
                new_grads[i] += self.biaser.gradient_node_bias(node=node)
            grads = new_grads
        return grads

    def compute_energies(self, chain: Chain) -> NDArray:
        self.compute_gradients(chain)
        enes = np.array([node.energy for node in chain])

        if self.biaser:
            new_enes = enes.copy()
            for i, node in enumerate(chain):
                new_enes[i] += self.biaser.energy_node_bias(node=node)
            enes = new_enes
        return enes

    def compute_func(self, *args, **kwargs):
        if self.compute_program == "qcop":
            call_kwargs = dict(kwargs)
            if self.print_stdout:
                call_kwargs.setdefault("print_stdout", True)
            return qcop.compute(*args, **call_kwargs)
        elif self.compute_program == "chemcloud":
            try:
                call_kwargs = dict(kwargs)
                call_kwargs.setdefault("queue", self.chemcloud_queue)
                return self._chemcloud_compute_with_retries(*args, **call_kwargs)
            except ValidationError as exc:
                message = str(exc)
                if "ProgramOutput" not in message:
                    raise
                program = str(args[0]) if args else self.program
                raise ExternalProgramError(
                    program=program,
                    message=(
                        "ChemCloud returned a response that is incompatible with the installed "
                        "qcio/chemcloud ProgramOutput schema. This usually happens when ChemCloud "
                        "tries to materialize a failed task using an older ProgramOutput layout. "
                        "Check the remote task failure details and align the chemcloud/qcio versions."
                    ),
                    original_exception=exc,
                ) from exc
            except Exception as exc:
                if not self._is_chemcloud_output_fetch_error(exc):
                    raise
                program = str(args[0]) if args else self.program
                request = getattr(exc, "request", None)
                request_url = str(getattr(request, "url", "") or "")
                task_id = ""
                if "/compute/output/" in request_url:
                    task_id = request_url.rsplit("/", 1)[-1].strip()
                status_code = getattr(
                    getattr(exc, "response", None), "status_code", None)
                message = (
                    "ChemCloud accepted the submission, but output collection failed while "
                    "polling `/compute/output/<task_id>`."
                )
                if status_code is not None:
                    message = f"{message} HTTP status: {status_code}."
                if task_id:
                    message = f"{message} Task ID: {task_id}."
                message = (
                    f"{message} This error is on result retrieval, so automatic retry is "
                    "disabled to avoid duplicate task submissions."
                )
                raise ExternalProgramError(
                    program=program,
                    message=message,
                    original_exception=exc,
                ) from exc
        else:
            raise ValueError(
                f"Invalid compute program: {self.compute_program}. Must be one of: {AVAIL_PROGRAMS}"
            )

    def _run_calc(
        self, calctype: str, chain: Union[Chain, List]
    ) -> List[StructureNode]:
        if isinstance(chain, Chain):
            assert isinstance(
                chain.nodes[0], StructureNode
            ), "input Chain has nodes incompatible with QCOPEngine."
            node_list = chain.nodes
        elif isinstance(chain, list):
            assert isinstance(
                chain[0], StructureNode
            ), f"input list has nodes incompatible with QCOPEngine: {chain[0]}"
            node_list = chain
        else:
            raise ValueError(
                f"Input needs to be a Chain or a List. You input a: {type(chain)}")

        # first make sure the program input has calctype set to input calctype
        prog_inp = ProgramInput(
            structure=node_list[0].structure, calctype=calctype, **self.program_args.__dict__)

        # now create program inputs for each geometry that is not frozen
        inds_frozen = [i for i, node in enumerate(node_list) if node.converged]
        all_prog_inps = [
            _change_prog_input_property(
                prog_inp=prog_inp, key="structure", value=node.structure
            )
            for node in node_list
        ]
        non_frozen_prog_inps = [
            pi for i, pi in enumerate(all_prog_inps) if i not in inds_frozen
        ]

        # only compute the nonfrozen structures
        if any([self.program == 'xtb']) and self.compute_program == "qcop":

            def helper(inp):
                prog, prog_inp = inp
                return self.compute_func(prog, prog_inp, collect_files=self.collect_files)

            iterables = [(self.program, inp) for inp in non_frozen_prog_inps]
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                non_frozen_results = list(executor.map(helper, iterables))

        elif self.compute_program == "chemcloud":
            batch_or_single = (
                non_frozen_prog_inps
                if len(non_frozen_prog_inps) > 1
                else non_frozen_prog_inps[0]
            )
            non_frozen_results = self.compute_func(
                self.program,
                batch_or_single,
                collect_files=self.collect_files,
            )
            if len(non_frozen_prog_inps) == 1:
                non_frozen_results = [non_frozen_results]

        else:
            non_frozen_results = [
                self.compute_func(self.program, pi, collect_files=self.collect_files) for pi in non_frozen_prog_inps
            ]

        # merge the results
        all_results = []
        for i, node in enumerate(node_list):
            if i in inds_frozen:
                all_results.append(copy.deepcopy(node_list[i]._cached_result))
            else:
                all_results.append(copy.deepcopy(non_frozen_results.pop(0)))

        update_node_cache(node_list=node_list, results=all_results)
        return node_list

    def _compute_geom_opt_result(self, node: StructureNode, keywords={}):
        """
        this will return a ProgramOutput from qcio geom opt call.
        """
        if "terachem" not in self.program:

            dpi = DualProgramInput(
                calctype="optimization",  # type: ignore
                structure=node.structure,
                subprogram=self.program,
                subprogram_args={
                    "model": self.program_args.model,
                    "keywords": self.program_args.keywords,
                },
                keywords=keywords,
            )

            output = self.compute_func(
                self.geometry_optimizer, dpi, collect_files=self.collect_files)

        else:  # DEC162025: Trying again... # OCT062025: bug where terachem optimizations werent being passed.
            tc_keywords = dict(getattr(self.program_args, "keywords", {}) or {})
            tc_keywords.update(
                {
                    "purify": "no",
                    "new_minimizer": "yes",
                }
            )

            prog_input = ProgramInput(
                structure=node.structure,
                # Can be "energy", "gradient", "hessian", "optimization", "transition_state"
                calctype="optimization",  # type: ignore
                model=self.program_args.model,
                # Preserve user-provided TeraChem keywords while enforcing required flags.
                keywords=tc_keywords,
            )

            output = self.compute_func(
                "terachem", prog_input, collect_files=self.collect_files)

        return output

    def _compute_hessian_result(self, node: StructureNode, use_bigchem=True):
        def _coerce_files_mapping(files_obj: Any) -> dict[str, Any]:
            if files_obj is None:
                return {}
            if isinstance(files_obj, dict):
                return files_obj
            model_dump = getattr(files_obj, "model_dump", None)
            if callable(model_dump):
                dumped = model_dump()
                if isinstance(dumped, dict):
                    return dumped
            return {}

        def _has_square_hessian(candidate: Any) -> bool:
            if candidate is None:
                return False
            with contextlib.suppress(Exception):
                arr = np.asarray(candidate, dtype=float)
                return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size > 0
            return False

        def _has_modes_file(files_map: dict[str, Any]) -> bool:
            for key in files_map:
                normalized_key = str(key).replace("\\", "/").lower()
                if normalized_key == "mass.weighted.modes.dat" or normalized_key.endswith(
                    "/mass.weighted.modes.dat"
                ):
                    return True
            return False

        def _hessian_payload_is_usable(output_obj: Any) -> bool:
            results = getattr(output_obj, "results", None)
            if results is None:
                # Non-ProgramOutput-like objects are used in tests and some legacy
                # integrations; do not force a fallback for unknown objects.
                return True

            modes = getattr(results, "normal_modes_cartesian", None)
            if modes is not None:
                with contextlib.suppress(Exception):
                    if len(modes) > 0:
                        return True

            if _has_square_hessian(getattr(results, "hessian", None)):
                return True
            if _has_square_hessian(getattr(output_obj, "return_result", None)):
                return True

            files_map = _coerce_files_mapping(getattr(results, "files", None))
            if _has_modes_file(files_map):
                return True

            trajectory = getattr(results, "trajectory", None)
            if isinstance(trajectory, list):
                for entry in trajectory:
                    entry_results = getattr(entry, "results", None)
                    entry_modes = getattr(entry_results, "normal_modes_cartesian", None)
                    if entry_modes is not None:
                        with contextlib.suppress(Exception):
                            if len(entry_modes) > 0:
                                return True
                    if _has_square_hessian(getattr(entry_results, "hessian", None)):
                        return True
                    entry_files = _coerce_files_mapping(getattr(entry_results, "files", None))
                    if _has_modes_file(entry_files):
                        return True

            return False

        prog = self.program
        collect_files = self.collect_files
        if "terachem" in self.program:
            prog = "terachem"
            collect_files = True

        def _run_standard_hessian_call():
            proginp = ProgramInput(
                structure=node.structure,
                calctype='hessian', **self.program_args.__dict__)
            return self.compute_func(
                self.program, proginp, collect_files=collect_files)

        if use_bigchem:
            dpi = DualProgramInput(
                calctype="hessian",  # type: ignore
                structure=node.structure,
                subprogram=prog,
                subprogram_args={
                    "model": self.program_args.model,
                    "keywords": self.program_args.keywords,
                },
                keywords={},
            )
            try:
                fres = self._chemcloud_compute_with_retries("bigchem", dpi)
                # ChemCloud client return type can be future-like (with .get()) or a
                # direct ProgramOutput, depending on installed client/version.
                if hasattr(fres, "get"):
                    timeout_seconds = _env_float(
                        "MEPD_BIGCHEM_HESSIAN_TIMEOUT_SECONDS", 1800.0)
                    if timeout_seconds > 0:
                        get_method = fres.get
                        supports_timeout = False
                        with contextlib.suppress(Exception):
                            supports_timeout = "timeout" in inspect.signature(
                                get_method).parameters
                        if supports_timeout:
                            output = get_method(timeout=timeout_seconds)
                        else:
                            captured: dict[str, object] = {}

                            def _get_result_without_timeout() -> None:
                                try:
                                    captured["result"] = get_method()
                                except Exception as exc:
                                    captured["error"] = exc

                            waiter = threading.Thread(
                                target=_get_result_without_timeout,
                                name="qcop-bigchem-hessian-get",
                                daemon=True,
                            )
                            waiter.start()
                            waiter.join(timeout_seconds)
                            if waiter.is_alive():
                                raise TimeoutError(
                                    "Timed out waiting for BigChem Hessian result "
                                    f"after {timeout_seconds:.1f}s. Increase "
                                    "MEPD_BIGCHEM_HESSIAN_TIMEOUT_SECONDS if needed."
                                )
                            if "error" in captured:
                                raise captured["error"]  # type: ignore[misc]
                            output = captured.get("result")
                    else:
                        output = fres.get()
                else:
                    output = fres
            except Exception as exc:
                if "terachem" in str(prog).lower():
                    logging.warning(
                        "BigChem Hessian call for TeraChem failed (%s); retrying via "
                        "standard ChemCloud Hessian call.",
                        exc,
                    )
                    return _run_standard_hessian_call()
                raise

            # Some BigChem TeraChem Hessian payloads can come back with only input
            # files (`geometry.xyz`, `tc.in`) and no usable Hessian/mode data.
            # In that case, transparently retry through the standard ChemCloud
            # Hessian pathway, which reliably returns parseable Hessian results.
            if (
                use_bigchem
                and "terachem" in str(prog).lower()
                and not _hessian_payload_is_usable(output)
            ):
                logging.warning(
                    "BigChem Hessian output for TeraChem was missing normal-mode/Hessian "
                    "data; retrying via standard ChemCloud Hessian call."
                )
                output = _run_standard_hessian_call()
        else:
            output = _run_standard_hessian_call()
        return output

    def _compute_conf_result(self, node: StructureNode):
        assert shutil.which(
            "crest") is not None, "crest not found in path. this currently only works with CREST"

        pi = ProgramInput(
            calctype="conformer_search",  # type: ignore
            structure=node.structure,
            model=self.program_args.model,
            keywords=self.program_args.keywords,
        )
        output = self.compute_func(
            'crest', pi, collect_files=self.collect_files)
        return output

    def _compute_ts_result(self, node: StructureNode, keywords={'maxiter': 1000}, use_bigchem=False,
                           hessres: ProgramOutput = None):
        if hessres is not None:
            np.savetxt("/tmp/hess.txt", hessres.results.hessian)
            kwds = keywords.copy()
            kwds["hessian"] = "file:hessian.txt"
            files = {'hessian.txt': open("/tmp/hess.txt").read()}
        elif hessres is None and use_bigchem:
            hess = self.compute_hessian(node=node)
            np.savetxt("/tmp/hess.txt", hess)
            kwds = keywords.copy()
            kwds["hessian"] = "file:hessian.txt"
            files = {'hessian.txt': open("/tmp/hess.txt").read()}

        else:
            kwds = keywords.copy()
            files = {}

        dpi = DualProgramInput(keywords=kwds,
                               structure=node.structure,
                               calctype="transition_state",
                               subprogram=self.program,
                               subprogram_args={
                                   "model": self.program_args.model,
                                   "keywords": self.program_args.keywords,
                               },
                               files=files)

        return self.compute_func('geometric', dpi, collect_files=self.collect_files)

    def compute_sd_irc(self, ts: StructureNode, hessres: ProgramOutput = None, dr=0.1, max_steps=500,
                       use_bigchem=False, ss=1.0, **kwargs) -> List[List[StructureNode], List[StructureNode]]:
        """
        steepest descent IRC.
        """
        self.compute_gradients([ts])

        if hessres is None:
            hessres = self._compute_hessian_result(
                node=ts, use_bigchem=use_bigchem)
        if self.program == 'terachem':
            from neb_dynamics.helper_functions import parse_nma_freq_data
            normal_modes, freqs = parse_nma_freq_data(hessres)
        else:
            normal_modes = hessres.results.normal_modes_cartesian
            freqs = hessres.results.freqs_wavenumber

        nimaginary = 0
        for freq in freqs:
            if freq < 0:
                nimaginary += 1

        if nimaginary > 1:
            logging.getLogger(__name__).warning(
                "More than one imaginary frequency detected during IRC seed check; frequencies=%s",
                freqs,
            )

        node_plus = displace_by_dr(
            node=ts, dr=dr, displacement=normal_modes[0])

        node_minus = displace_by_dr(
            node=ts, dr=-1*dr, displacement=normal_modes[0])

        self.compute_gradients([ts, node_plus, node_minus])
        sd_plus = self.steepest_descent(
            node_plus, max_steps=max_steps, ss=ss, grad_thre=1e-7, **kwargs)
        sd_minus = self.steepest_descent(
            node_minus, max_steps=max_steps, ss=ss, grad_thre=1e-7, **kwargs)
        return [sd_minus, sd_plus]

    def compute_hessian(self, node: StructureNode):
        output = self._compute_hessian_result(node)
        return output.return_result

    def compute_conformers(self, node: StructureNode):
        output = self._compute_conf_result(node)
        return output.results.conformers

    def compute_transition_state(self, node: StructureNode, keywords={'maxiter': 500}):
        output = self._compute_ts_result(node=node)
        if output.success:
            return StructureNode(structure=output.return_result)
        else:
            return output

    def compute_geometry_optimization(self, node: StructureNode, keywords={'coordsys': 'cart', 'maxit': 500}) -> list[StructureNode]:
        """
        will run a geometry optimization call and parse the output into
        a list of Node objects
        """
        output = self._compute_geom_opt_result(node=node, keywords=keywords)
        trajectory = self._extract_optimization_trajectory(
            output,
            reference_structure=node.structure,
        )
        if trajectory:
            return trajectory
        raise ElectronicStructureError(
            msg="Geometry optimization completed but no optimization trajectory was found.",
            obj=output,
        )

    def compute_geometry_optimizations(
        self,
        nodes: list[StructureNode],
        keywords={'coordsys': 'cart', 'maxit': 500},
    ) -> list[list[StructureNode]]:
        """
        Batch geometry optimizations when the backend can accept a list of inputs.
        ChemCloud supports list submission, so use that path there; otherwise fall
        back to sequential single-node optimizations.

        For ChemCloud batch calls, individual optimization failures are represented
        as empty trajectories so successful optimizations can still be consumed by
        callers in a single batch submission.
        """
        if len(nodes) == 0:
            return []

        if self.compute_program != "chemcloud":
            return [self.compute_geometry_optimization(node=node, keywords=keywords) for node in nodes]

        program_inputs = []
        for node in nodes:
            if "terachem" not in self.program:
                program_inputs.append(
                    DualProgramInput(
                        calctype="optimization",  # type: ignore
                        structure=node.structure,
                        subprogram=self.program,
                        subprogram_args={
                            "model": self.program_args.model,
                            "keywords": self.program_args.keywords,
                        },
                        keywords=keywords,
                    )
                )
            else:
                tc_keywords = dict(getattr(self.program_args, "keywords", {}) or {})
                tc_keywords.update(
                    {
                        "purify": "no",
                        "new_minimizer": "yes",
                    }
                )
                program_inputs.append(
                    ProgramInput(
                        structure=node.structure,
                        calctype="optimization",  # type: ignore
                        model=self.program_args.model,
                        # Preserve user-provided TeraChem keywords while enforcing required flags.
                        keywords=tc_keywords,
                    )
                )

        outputs = self.compute_func(
            self.geometry_optimizer if "terachem" not in self.program else "terachem",
            program_inputs if len(program_inputs) > 1 else program_inputs[0],
            collect_files=self.collect_files,
        )
        if len(program_inputs) == 1:
            outputs = [outputs]

        trajectories: list[list[StructureNode]] = []
        for output, node in zip(outputs, nodes):
            trajectory = self._extract_optimization_trajectory(
                output,
                reference_structure=node.structure,
            )
            if not trajectory:
                if self.compute_program == "chemcloud":
                    logging.warning(
                        "ChemCloud batch optimization returned no trajectory for one candidate; "
                        "keeping an empty trajectory placeholder and continuing."
                    )
                    trajectories.append([])
                    continue
                raise ElectronicStructureError(
                    msg="Geometry optimization completed but no optimization trajectory was found.",
                    obj=output,
                )
            trajectories.append(trajectory)
        return trajectories

    @staticmethod
    def _extract_output_files(output: ProgramOutput) -> dict[str, object]:
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

    @classmethod
    def _file_text_from_output(cls, output: ProgramOutput, suffixes: tuple[str, ...]) -> str | None:
        files = cls._extract_output_files(output)
        normalized_suffixes = tuple(str(suffix).lower() for suffix in suffixes)
        for file_name, data in files.items():
            if not str(file_name).lower().endswith(normalized_suffixes):
                continue
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            return str(data)
        return None

    @classmethod
    def _extract_optimization_trajectory(
        cls,
        output: ProgramOutput,
        *,
        reference_structure: Structure,
    ) -> list[StructureNode]:
        for trajectory in (
            getattr(getattr(output, "results", None), "trajectory", None),
            getattr(getattr(output, "data", None), "trajectory", None),
            getattr(output, "trajectory", None),
        ):
            if trajectory:
                nodes: list[StructureNode] = []
                for entry in trajectory:
                    struct = getattr(
                        getattr(entry, "input_data", None), "structure", None)
                    if struct is not None:
                        node = StructureNode(structure=struct)
                        results = getattr(entry, "results", None)
                        with contextlib.suppress(Exception):
                            node._cached_energy = results.energy
                        with contextlib.suppress(Exception):
                            node._cached_gradient = results.gradient
                        if node._cached_energy is not None or node._cached_gradient is not None:
                            node._cached_result = entry
                        nodes.append(node)
                if nodes:
                    return nodes

        files = cls._extract_output_files(output)
        for file_name, contents in files.items():
            file_name = str(file_name)
            if not (file_name.endswith("optim.xyz") or file_name == "optim.xyz"):
                continue
            with tempfile.TemporaryDirectory() as td:
                xyz_fp = Path(td) / "optim.xyz"
                if isinstance(contents, bytes):
                    xyz_fp.write_bytes(contents)
                else:
                    xyz_fp.write_text(str(contents))
                structures = Structure.open_multi(
                    xyz_fp,
                    charge=reference_structure.charge,
                    multiplicity=reference_structure.multiplicity,
                )
            if structures:
                return [
                    StructureNode(structure=struct)
                    for struct in structures
                ]

        return []

    @staticmethod
    def _trajectory_minimum_indices(energies: list[float], *, max_candidates: int, frame_stride: int) -> list[int]:
        if len(energies) == 0:
            return []
        indices = [
            idx
            for idx in range(1, len(energies) - 1)
            if energies[idx] <= energies[idx - 1] and energies[idx] <= energies[idx + 1]
        ]
        if frame_stride > 1:
            indices = [idx for idx in indices if idx % frame_stride == 0]
        if not indices:
            indices = list(range(0, len(energies), max(frame_stride, 1)))
        ranked = sorted(indices, key=lambda idx: energies[idx])
        seen: set[int] = set()
        ordered: list[int] = []
        for idx in ranked:
            if idx in seen:
                continue
            seen.add(idx)
            ordered.append(int(idx))
            if len(ordered) >= max_candidates:
                break
        return ordered

    @staticmethod
    def _structure_nodes_from_structures(structures: list[Structure]) -> list[StructureNode]:
        nodes: list[StructureNode] = []
        for structure in structures:
            graph = None
            try:
                from neb_dynamics.qcio_structure_helpers import structure_to_molecule
                graph = structure_to_molecule(structure)
            except Exception:
                graph = None
            node = StructureNode(structure=structure, graph=graph)
            node.has_molecular_graph = graph is not None
            nodes.append(node)
        return nodes

    def _compute_crest_nanoreactor_candidates(
        self,
        node: StructureNode,
        *,
        nanoreactor_inputs: dict[str, object],
    ) -> list[StructureNode]:
        charge = int(node.structure.charge)
        multiplicity = int(node.structure.multiplicity)
        max_candidates = int(nanoreactor_inputs.get("max_candidates", 12))
        cmdline_args = ["structure.xyz", "-msreact"]
        msreact_input = str(nanoreactor_inputs.get(
            "crest_msinput", "") or "").strip()
        if msreact_input:
            cmdline_args.extend(["--msinput", "msreact.inp"])
        if nanoreactor_inputs.get("crest_msmolbar"):
            cmdline_args.append("--msmolbar")
        if nanoreactor_inputs.get("crest_msinchi"):
            cmdline_args.append("--msinchi")
        if nanoreactor_inputs.get("crest_msiso"):
            cmdline_args.append("--msiso")
        if nanoreactor_inputs.get("crest_msnoiso"):
            cmdline_args.append("--msnoiso")
        if "crest_msnbonds" in nanoreactor_inputs:
            cmdline_args.extend(
                ["--msnbonds", str(int(nanoreactor_inputs["crest_msnbonds"]))])
        if "crest_msnshifts" in nanoreactor_inputs:
            cmdline_args.extend(
                ["--msnshifts", str(int(nanoreactor_inputs["crest_msnshifts"]))])
        if charge:
            cmdline_args.extend(["--chrg", str(charge)])
        if multiplicity > 1:
            cmdline_args.extend(["--uhf", str(max(multiplicity - 1, 0))])

        files: dict[str, str] = {"structure.xyz": node.structure.to_xyz()}
        if msreact_input:
            files["msreact.inp"] = msreact_input + \
                ("\n" if not msreact_input.endswith("\n") else "")

        output = self.compute_func(
            "crest",
            FileInput(files=files, cmdline_args=cmdline_args),
            collect_files=True,
        )
        products_xyz = self._file_text_from_output(
            output, ("crest_msreact_products.xyz",))
        if not products_xyz:
            raise ExternalProgramError(
                program="crest",
                message="CREST nanoreactor run completed but no `crest_msreact_products.xyz` output was returned.",
            )
        structures = Structure.from_xyz_multi(
            products_xyz, charge=charge, multiplicity=multiplicity)
        return self._structure_nodes_from_structures(structures[:max_candidates])

    def _compute_terachem_nanoreactor_candidates(
        self,
        node: StructureNode,
        *,
        nanoreactor_inputs: dict[str, object],
    ) -> list[StructureNode]:
        nanoreactor_inputs = _resolve_nanoreactor_inputs(nanoreactor_inputs)
        model = dict(getattr(self.program_args, "model", {}) or {})
        method = str(nanoreactor_inputs.get(
            "terachem_method", model.get("method", "uhf")))
        basis = str(nanoreactor_inputs.get(
            "terachem_basis", model.get("basis", "3-21g")))
        max_candidates = int(nanoreactor_inputs.get("max_candidates", 12))
        frame_stride = int(nanoreactor_inputs.get("terachem_frame_stride", 10))
        tcin_lines = [
            f"{'coordinates':<20} geometry.xyz",
            f"{'charge':<20} {int(node.structure.charge)}",
            f"{'spinmult':<20} {int(node.structure.multiplicity)}",
            f"{'basis':<20} {basis}",
            f"{'method':<20} {method}",
            f"{'run':<20} md",
            f"{'nstep':<20} {int(nanoreactor_inputs.get('terachem_nstep', 2000))}",
            f"{'timestep':<20} {float(nanoreactor_inputs.get('terachem_timestep', 0.5))}",
            f"{'tinit':<20} {float(nanoreactor_inputs.get('terachem_tinit', 1200.0))}",
            f"{'thermostat':<20} {str(nanoreactor_inputs.get('terachem_thermostat', 'langevin'))}",
            f"{'t0':<20} {float(nanoreactor_inputs.get('terachem_t0', 1500.0))}",
            f"{'lnvtime':<20} {float(nanoreactor_inputs.get('terachem_lnvtime', 200.0))}",
            f"{'convthre':<20} {float(nanoreactor_inputs.get('terachem_convthre', 0.005))}",
            f"{'levelshift':<20} {'yes' if bool(nanoreactor_inputs.get('terachem_levelshift', True)) else 'no'}",
            f"{'levelshiftvala':<20} {float(nanoreactor_inputs.get('terachem_levelshiftvala', 0.3))}",
            f"{'levelshiftvalb':<20} {float(nanoreactor_inputs.get('terachem_levelshiftvalb', 0.1))}",
            f"{'scf':<20} {str(nanoreactor_inputs.get('terachem_scf', 'diis+a'))}",
            f"{'maxit':<20} {int(nanoreactor_inputs.get('terachem_maxit', 300))}",
            f"{'mdbc':<20} spherical",
            f"{'md_r1':<20} {float(nanoreactor_inputs.get('terachem_md_r1', 6.0))}",
            f"{'md_k1':<20} {float(nanoreactor_inputs.get('terachem_md_k1', 3.0))}",
            f"{'md_r2':<20} {float(nanoreactor_inputs.get('terachem_md_r2', 4.0))}",
            f"{'md_k2':<20} {float(nanoreactor_inputs.get('terachem_md_k2', 5.0))}",
            f"{'mdbc_hydrogen':<20} {'yes' if bool(nanoreactor_inputs.get('terachem_mdbc_hydrogen', True)) else 'no'}",
            f"{'mdbc_mass_scaled':<20} {'yes' if bool(nanoreactor_inputs.get('terachem_mdbc_mass_scaled', True)) else 'no'}",
            f"{'mdbc_t1':<20} {int(nanoreactor_inputs.get('terachem_mdbc_t1', 750))}",
            f"{'mdbc_t2':<20} {int(nanoreactor_inputs.get('terachem_mdbc_t2', 250))}",
            "end",
        ]
        output = self.compute_func(
            "terachem",
            FileInput(
                files={
                    "tc.in": "\n".join(tcin_lines) + "\n",
                    "geometry.xyz": node.structure.to_xyz(),
                },
                cmdline_args=["tc.in"],
            ),
            collect_files=True,
        )
        trajectory_xyz = self._file_text_from_output(
            output, ("scr/coors.xyz", "coors.xyz"))
        if not trajectory_xyz:
            raise ExternalProgramError(
                program="terachem",
                message="TeraChem nanoreactor run completed but no `scr/coors.xyz` trajectory was returned.",
            )
        log_text = self._file_text_from_output(
            output, ("scr/log.xls", "log.xls"))
        frames = Structure.from_xyz_multi(
            trajectory_xyz,
            charge=int(node.structure.charge),
            multiplicity=int(node.structure.multiplicity),
        )
        if not frames:
            return []

        energies: list[float] = []
        if log_text:
            for raw_line in str(log_text).splitlines():
                parts = raw_line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    energies.append(float(parts[1]))
                except ValueError:
                    continue
        frame_count = len(frames)
        if energies:
            usable = min(frame_count, len(energies))
            frames = frames[:usable]
            energies = energies[:usable]
            selected_indices = self._trajectory_minimum_indices(
                energies,
                max_candidates=max_candidates,
                frame_stride=frame_stride,
            )
            selected_frames = [frames[idx]
                               for idx in selected_indices if 0 <= idx < len(frames)]
        else:
            selected_frames = frames[:: max(frame_stride, 1)][:max_candidates]
        return self._structure_nodes_from_structures(selected_frames)

    def compute_nanoreactor_candidates(
        self,
        node: StructureNode,
        *,
        nanoreactor_inputs: dict[str, object] | None = None,
    ) -> list[StructureNode]:
        nanoreactor_inputs = dict(nanoreactor_inputs or {})
        program = str(self.program or "").strip().lower()
        if "terachem" in program:
            return self._compute_terachem_nanoreactor_candidates(
                node,
                nanoreactor_inputs=nanoreactor_inputs,
            )
        if "crest" in program:
            return self._compute_crest_nanoreactor_candidates(
                node,
                nanoreactor_inputs=nanoreactor_inputs,
            )
        raise ValueError(
            f"Nanoreactor sampling is not implemented for program `{self.program}`. "
            "Use a CREST or TeraChem-backed engine."
        )
