from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from neb_dynamics.NEB import NoneConvergedException
from neb_dynamics.nodes.Node import Node


@dataclass
class TS_PRFO:
    initial_node: Node
    max_nsteps: int = 2000
    dr: float = 0.1
    grad_thre: float = 1e-6

    max_step_size: float = 1.0

    traj = []

    @property
    def ts(self):
        ts, _ = self.ts_and_traj
        return ts

    @cached_property
    def ts_and_traj(self):
        return self.find_ts()

    def get_lambda_n(self, all_eigenvalues, f_vector, tol=10e-3, break_limit=1e6):
        """
        f_vector and all_eigenvalues need to be ordered
        from lowest to highest based on eigenvalue value
        """
        break_ind = 0
        lamb_guess = 0
        hist = [lamb_guess]
        while break_ind < break_limit:
            tot_lamb = 0
            for eigval_i, f_i in zip(all_eigenvalues, f_vector):
                tot_lamb += (f_i**2) / (lamb_guess - eigval_i)
            if np.abs(tot_lamb - lamb_guess) <= tol:
                return tot_lamb, hist
            else:
                lamb_guess = tot_lamb
            break_ind += 1
            hist.append(lamb_guess)
        return None, hist

    def get_lambda_p(self, eigval, f_i):
        lambda_p = 0.5 * eigval + 0.5 * (np.sqrt(eigval**2 + 4 * (f_i) ** 2))
        return lambda_p

    def prfo_step(self, node):
        grad = node.gradient
        H = node.hessian
        H_evals, H_evecs = np.linalg.eigh(H)
        orig_dim = grad.shape
        grad_reshaped = grad.reshape(-1, 1)
        F_vec = np.dot(H_evecs.T, grad_reshaped)

        lambda_p = self.get_lambda_p(H_evals[0], F_vec[0])
        lambda_n, _ = self.get_lambda_n(
            all_eigenvalues=H_evals, f_vector=F_vec, break_limit=1e6
        )
        if lambda_n is None:
            raise NoneConvergedException(
                "lambda_n calculation failed. maybe start again with a different initial guess"
            )

        h0 = (-1 * F_vec[0] * H_evecs[:, 0]) / (H_evals[0] - lambda_p)
        hrest = sum(
            [
                (-1 * F_vec[i] * H_evecs[:, i]) / (H_evals[i] - lambda_n)
                for i in range(1, len(F_vec))
            ]
        )
        step = h0 + hrest
        step_reshaped = step.reshape(orig_dim)

        length_step = np.linalg.norm(step_reshaped)

        if length_step > self.max_step_size:
            step_rescaled = step_reshaped / length_step
            step_reshaped = step_rescaled * self.max_step_size

        return step_reshaped

    def find_ts(self):
        self.traj.append(self.initial_node)

        steps_taken = 0
        converged = False

        start_node = self.initial_node

        grad_mag = np.linalg.norm(start_node.gradient)
        converged = grad_mag <= self.grad_thre
        while steps_taken < self.max_nsteps and not converged:
            steps_taken += 1
            print(
                f"StepsTaken:{steps_taken}||grad_mag:{grad_mag}                 \r",
                end="",
            )
            direction = self.prfo_step(start_node)

            new_point = start_node.coords + direction * self.dr
            new_node = start_node.copy()
            new_node = new_node.update_coords(new_point)

            grad_mag = np.linalg.norm(new_node.gradient)
            converged = grad_mag <= self.grad_thre

            if converged:
                print(f"Converged in {steps_taken} steps!")
                new_node.converged = True

            self.traj.append(new_node)
            start_node = new_node

        if converged:
            return new_node, self.traj

        if not converged:
            print(f"Did not converge in {steps_taken} steps.")
            return new_node, self.traj

    def plot_path(self):
        traj = self.traj
        node = traj[0]

        s = 4
        fs = 18

        en_func = node.en_func_arr

        min_val = -s
        max_val = s

        fig = 10
        f, _ = plt.subplots(figsize=(1.18 * fig, fig))
        x = np.linspace(start=min_val, stop=max_val, num=10)
        y = x.reshape(-1, 1)

        h = en_func([x, y])
        cs = plt.contourf(x, x, h, levels=10)
        # _ = f.colorbar(cs)

        plt.plot(
            [x.coords[0] for x in traj],
            [x.coords[1] for x in traj],
            "*--",
            c="white",
            label="path",
            ms=15,
        )
        # for inp, (evec0, evec1) in zip(traj, eigvecs):
        #     plt.arrow(inp[0], inp[1], evec0[0], evec0[1], color='red', width=.05)
        #     plt.arrow(inp[0], inp[1], evec1[0], evec1[1], color='green', width=.05)

        plt.yticks(fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.show()
