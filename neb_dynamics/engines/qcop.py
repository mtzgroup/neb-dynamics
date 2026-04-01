from __future__ import annotations

import concurrent
import inspect
from dataclasses import dataclass
from typing import List, Union
import os
import time
import logging
from numpy.typing import NDArray
import numpy as np
from pydantic import ValidationError

import qcop
from qcop.exceptions import ExternalProgramError
from qcio.models.inputs import DualProgramInput, ProgramInput, ProgramArgs
from qcio import ProgramOutput
import shutil

from chemcloud import CCClient
from chemcloud import compute as cc_compute
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


def _configure_chemcloud_client(queue: str) -> None:
    """Handle ChemCloud client API drift across installed versions."""
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

    cc_configure_client(**client_params)


@dataclass
class QCOPEngine(Engine):
    program_args: ProgramArgs = ProgramArgs(
        model={"method": "GFN2xTB", "basis": "GFN2xTB"},)
    program: str = "xtb"
    geometry_optimizer: str = "geometric"
    compute_program: str = "qcop"
    chemcloud_queue: str | None = None
    biaser: ChainBiaser = None
    collect_files: bool = False
    write_qcio: bool = False

    def __post_init__(self):
        self.chemcloud_queue = _resolve_chemcloud_queue(self.chemcloud_queue)
        if self.write_qcio:
            logging.warning(
                "QCOPEngine write_qcio=True: cached qcio.ProgramOutput objects will be "
                "written when results are saved to disk. This can consume substantial disk space."
            )
        if self.compute_program == "chemcloud":
            _configure_chemcloud_client(self.chemcloud_queue)

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
            "all connection attempts failed",
            "connecterror",
            "service unavailable",
            "gateway timeout",
            "temporarily unavailable",
            "connection reset",
            "timed out",
            "connecttimeout",
        )
        return any(token in msg for token in retry_markers)

    def _chemcloud_compute_with_retries(self, *args, **kwargs):
        max_attempts = 5
        delay_sec = 5.0
        for attempt in range(1, max_attempts + 1):
            try:
                return cc_compute(*args, queue=self.chemcloud_queue, **kwargs)
            except Exception as exc:
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
            return qcop.compute(*args, **kwargs)
        elif self.compute_program == "chemcloud":
            try:
                return self._chemcloud_compute_with_retries(*args, **kwargs)
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

        if self.compute_program == "chemcloud":
            # Submit ChemCloud jobs one geometry at a time. Batched NEB-image submissions
            # have been timing out on crest/xtb requests for this workflow.
            non_frozen_results = [
                self.compute_func(
                    self.program,
                    prog_inp,
                    collect_files=self.collect_files,
                )
                for prog_inp in non_frozen_prog_inps
            ]

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

            prog_input = ProgramInput(
                structure=node.structure,
                # Can be "energy", "gradient", "hessian", "optimization", "transition_state"
                calctype="optimization",  # type: ignore
                model=self.program_args.model,
                keywords={
                    "purify": "no",
                    "new_minimizer": "yes",
                },  # new_minimizer yes is required
            )

            output = self.compute_func(
                "terachem", prog_input, collect_files=self.collect_files)

        return output

    def _compute_hessian_result(self, node: StructureNode, use_bigchem=True):

        prog = self.program
        collect_files = self.collect_files
        if "terachem" in self.program:
            prog = "terachem"
            collect_files = True

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
            fres = cc_compute("bigchem", dpi)
            # ChemCloud client return type can be future-like (with .get()) or a
            # direct ProgramOutput, depending on installed client/version.
            output = fres.get() if hasattr(fres, "get") else fres
        else:
            proginp = ProgramInput(
                structure=node.structure,
                calctype='hessian', **self.program_args.__dict__)
            output = self.compute_func(
                self.program, proginp, collect_files=collect_files)
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
            print(
                "WARNING: More than one imaginary frequency detected. This is not a TS.")
            print(f"frequencies: {freqs}")

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
        all_outputs = output.results.trajectory
        structures = [output.input_data.structure for output in all_outputs]
        return [
            StructureNode(structure=struct, _cached_result=result)
            for struct, result in zip(structures, all_outputs)

        ]
