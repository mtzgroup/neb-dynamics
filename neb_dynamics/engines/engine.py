from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Union, List
from neb_dynamics.chain import Chain
import numpy as np


from neb_dynamics.nodes.node import Node
from neb_dynamics.fakeoutputs import FakeQCIOOutput
from neb_dynamics.helper_functions import get_mass
from qcio import ProgramOutput


@dataclass
class FiniteDifferenceHessianResults:
    hessian: np.ndarray
    normal_modes_cartesian: list[np.ndarray]
    freqs_wavenumber: list[float]


@dataclass
class FiniteDifferenceHessianOutput:
    input_data: Any
    results: FiniteDifferenceHessianResults
    success: bool = True

    @property
    def return_result(self) -> np.ndarray:
        return self.results.hessian

    def save(self, filename: str | Path) -> None:
        payload = {
            "success": bool(self.success),
            "input_data": {
                "structure": (
                    self.input_data.structure.model_dump()
                    if getattr(self.input_data, "structure", None) is not None
                    and hasattr(self.input_data.structure, "model_dump")
                    else None
                ),
            },
            "results": {
                "hessian": np.asarray(self.results.hessian, dtype=float).tolist(),
                "normal_modes_cartesian": [
                    np.asarray(mode, dtype=float).tolist()
                    for mode in self.results.normal_modes_cartesian
                ],
                "freqs_wavenumber": [float(freq) for freq in self.results.freqs_wavenumber],
            },
        }
        Path(filename).write_text(json.dumps(payload, indent=2))


def build_hessian_result_from_matrix(node: Node, hessian: np.ndarray) -> FiniteDifferenceHessianOutput:
    hessian_arr = np.asarray(hessian, dtype=float)
    if hessian_arr.ndim != 2 or hessian_arr.shape[0] != hessian_arr.shape[1]:
        raise ValueError("Hessian must be a square 2D array.")

    # Numerical finite differences are not exactly symmetric; enforce symmetry.
    hessian_arr = 0.5 * (hessian_arr + hessian_arr.T)
    eigvals, eigvecs = np.linalg.eigh(hessian_arr)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    refshape = np.asarray(node.coords).shape
    modes = [eigvecs[:, i].reshape(refshape) for i in range(eigvecs.shape[1])]
    # Keep sign information while producing compact "frequency-like" values.
    freqs = [float(np.sign(v) * np.sqrt(abs(v))) for v in eigvals]

    return FiniteDifferenceHessianOutput(
        input_data=SimpleNamespace(structure=getattr(node, "structure", None)),
        results=FiniteDifferenceHessianResults(
            hessian=hessian_arr,
            normal_modes_cartesian=modes,
            freqs_wavenumber=freqs,
        ),
        success=True,
    )


@dataclass
class Engine(ABC):
    finite_difference_hessian_step_size: ClassVar[float] = 1e-3

    @abstractmethod
    def compute_gradients(
        self, chain: Union[Chain, List]
    ) -> Union[FakeQCIOOutput, ProgramOutput]:
        """
        returns the gradients for each node in the chain as
        specified by the object inputs
        """
        ...

    @abstractmethod
    def compute_energies(
        self, chain: Union[Chain, List]
    ) -> Union[FakeQCIOOutput, ProgramOutput]:
        """
        returns the energies for each node in the chain as
        specified by the object inputs
        """
        ...

    def _warn_finite_difference_hessian_fallback(
        self,
        *,
        node: Node,
        step_size: float,
        expected_energy_evaluations: int,
    ) -> None:
        message = (
            "Using default finite-difference Hessian fallback "
            f"for `{self.__class__.__name__}` (step={step_size:g} Bohr; "
            f"~{expected_energy_evaluations} energy evaluations)."
        )
        logging.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def compute_hessian(
        self,
        node: Node,
        step_size: float | None = None,
    ) -> np.ndarray:
        h = float(step_size if step_size is not None else self.finite_difference_hessian_step_size)
        if h <= 0:
            raise ValueError("finite-difference Hessian step size must be positive.")

        coords = np.asarray(node.coords, dtype=float)
        refshape = coords.shape
        x0 = coords.reshape(-1)
        ndof = x0.size
        if ndof == 0:
            raise ValueError("Cannot compute Hessian for a node with zero coordinates.")

        expected_energy_evaluations = 1 + 2 * ndof * ndof
        self._warn_finite_difference_hessian_fallback(
            node=node,
            step_size=h,
            expected_energy_evaluations=expected_energy_evaluations,
        )

        def _energy_at(displacement: np.ndarray) -> float:
            displaced = node.update_coords((x0 + displacement).reshape(refshape))
            return float(self.compute_energies([displaced])[0])

        h2 = h * h
        e0 = _energy_at(np.zeros(ndof, dtype=float))
        hessian = np.zeros((ndof, ndof), dtype=float)

        e_plus = np.zeros(ndof, dtype=float)
        e_minus = np.zeros(ndof, dtype=float)
        for i in range(ndof):
            disp = np.zeros(ndof, dtype=float)
            disp[i] = h
            e_plus[i] = _energy_at(disp)
            disp[i] = -h
            e_minus[i] = _energy_at(disp)
            hessian[i, i] = (e_plus[i] - 2.0 * e0 + e_minus[i]) / h2

        for i in range(ndof):
            for j in range(i + 1, ndof):
                disp = np.zeros(ndof, dtype=float)
                disp[i], disp[j] = h, h
                e_pp = _energy_at(disp)
                disp[i], disp[j] = h, -h
                e_pm = _energy_at(disp)
                disp[i], disp[j] = -h, h
                e_mp = _energy_at(disp)
                disp[i], disp[j] = -h, -h
                e_mm = _energy_at(disp)
                value = (e_pp - e_pm - e_mp + e_mm) / (4.0 * h2)
                hessian[i, j] = value
                hessian[j, i] = value

        return hessian

    def _compute_hessian_result(
        self,
        node: Node,
        **kwargs,
    ) -> FiniteDifferenceHessianOutput:
        step_size = kwargs.pop("step_size", None)
        hessian = self.compute_hessian(node=node, step_size=step_size)
        return build_hessian_result_from_matrix(node=node, hessian=hessian)

    def steepest_descent(
        self,
        node: Node,
        ss=1.0,
        max_steps=500,
        ene_thre: float = 1e-6,
        grad_thre: float = 1e-4,
        mass_weighted: bool = False,
    ) -> list[Node]:
        # print("************\n\n\n\nRUNNING STEEPEST DESCENT\n\n\n\nn***********")
        history = []
        last_node = node.copy()
        # make sure the node isn't frozen so it returns a gradient
        last_node.converged = False

        curr_step = 0
        converged = False
        natom = node.coords.shape[0]
        while curr_step < max_steps and not converged:
            grad = np.array(last_node.gradient)
            if mass_weighted:
                masses = [get_mass(s) for s in node.structure.symbols]
                grad = np.array([atom*np.sqrt(mass)
                                for atom, mass in zip(grad, masses)])

            grad_mag = np.linalg.norm(grad) / np.sqrt(natom)
            # print(f"Step {curr_step}: Gradient magnitude {grad_mag:.4e}")
            if grad_mag > ss:
                logging.getLogger(__name__).debug(
                    "Step %s: gradient magnitude %.4e greater than step size %.4e; scaling step.",
                    curr_step,
                    grad_mag,
                    ss,
                )
                # normalize the gradient

                grad = grad / np.linalg.norm(grad)
                grad = grad / np.sqrt(natom)
                grad = grad*ss

            new_coords = last_node.coords - ((1.0 * ss) * grad)
            node_new = last_node.update_coords(new_coords)
            grads = self.compute_gradients([node_new])
            ene = self.compute_energies([node_new])
            node_new._cached_gradient = grads[0]
            node_new._cached_energy = ene[0]

            history.append(node_new)

            delta_en = node_new.energy - last_node.energy
            grad_inf_norm = np.amax(np.abs(node_new.gradient))
            converged = delta_en <= ene_thre and grad_inf_norm <= grad_thre

            last_node = node_new.copy()
            curr_step += 1

        return history
