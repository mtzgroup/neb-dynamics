from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class LBFGS:
    func: Callable[[np.array], float]
    func_args: dict
    grad_func: Callable[[np.array], float]
    m: int = 20
    # weighting_func: Callable
    weighting: bool = False

    e_thresh: float = 1.0e-3
    g_thresh_2: float = 1.0e-3
    g_thresh_inf: float = 1.0e-3
    max_iter: int = 500

    def __post_init__(self):
        self.x_0 = self.func_args["x_0"]
        self.num_variables = len(self.x_0)
        self.s = np.zeros((self.m, self.num_variables))
        self.y = np.zeros((self.m, self.num_variables))
        self.p = np.zeros(self.m)
        self.g = np.zeros((self.m, self.num_variables))

        self.x_traj = []

    def optimize(self):
        iter = 0
        E = self.func(self.x_0)
        grad = self.grad_func(self.x_0)
        print(f"f(x_0) = {E}")
        print(f"x_0 = {self.x_0}")
        print(f"f'(x_0) = {grad}")
        converged = self.check_convergence(0, grad)
        x = self.x_0
        self.x_traj.append(x)
        while not converged:
            if abs(
                np.dot(self.s[(iter - 1) % self.m], self.y[(iter - 1) % self.m]) > 0.001
            ):
                if iter == self.max_iter:
                    break
                q = grad
                hist_len = min(iter, self.m)
                a_list = np.zeros(hist_len)
                for i in list(reversed(range(iter)))[:hist_len]:
                    p_i = 1.0 / np.dot(self.s[i % self.m], self.y[i % self.m])
                    a_list[i % self.m] = p_i * np.dot(self.s[i % self.m], q)
                    q -= a_list[i % self.m] * self.y[i % self.m]
                if iter == 0:
                    gamma = 1.0
                else:
                    gamma = np.dot(
                        self.s[(iter - 1) % self.m], self.y[(iter - 1) % self.m]
                    ) / np.dot(self.y[(iter - 1) % self.m], self.y[(iter - 1) % self.m])
                H_0_k = gamma * np.eye(self.num_variables)
                z = np.matmul(H_0_k, q)
                for i in range(iter):
                    p_i = 1.0 / np.dot(self.s[i % self.m], self.y[i % self.m])
                    B_i = p_i * np.dot(self.y[i % self.m], z)
                    z += self.s[i % self.m] * (a_list[i % self.m] - B_i)
                z = -z
            else:
                z = -grad
            x_new, E_new, grad_new = self.linesearch(x, z)
            delta_E = abs(E - E_new)
            converged = self.check_convergence(delta_E, grad_new)
            self.s[iter % self.m] = x_new - x
            self.y[iter % self.m] = grad_new - grad
            E = E_new
            x = x_new
            self.x_traj.append(x)
            grad = grad_new
            iter += 1
            print(f"Iteration: {iter}")
            print(f"\tE = {E_new}, grad_norm_2 = {np.dot(grad_new, grad_new)}")
        return x, E_new, grad

    def linesearch(self, x_0, z):
        c1 = 10e-4
        c2 = 0.1
        beta = 0.5
        en0 = self.func(x_0)
        grad = self.grad_func(x_0)
        step = 0
        step_size = 1.0
        converged = False
        max_steps = 10
        while not converged:
            x_prime = x_0 + (step_size * z)
            en_prime = self.func(x_prime)
            g = self.grad_func(x_prime)
            armijo = en_prime <= en0 + c1 * step_size * np.dot(z, grad)
            wolfe = -np.dot(z, g) <= -c2 * np.dot(z, grad)
            if armijo and wolfe:
                converged = True
            else:
                step_size *= beta
            step += 1
            if step == max_steps:
                break
        return x_prime, en_prime, np.asarray(g)

    def check_convergence(self, delta_E, grad):
        norm_2 = np.dot(grad, grad)
        norm_inf = max(abs(grad))
        if (
            (delta_E < self.e_thresh)
            and (norm_2 < self.g_thresh_2)
            and (norm_inf < self.g_thresh_inf)
        ):
            return True
        else:
            return False
