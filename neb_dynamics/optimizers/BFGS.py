from __future__ import annotations


from dataclasses import dataclass, field
from typing import Callable
from neb_dynamics.optimizer import Optimizer
from neb_dynamics.optimizers import ALS
from chain import Chain
import numpy as np


@dataclass
class BFGS(Optimizer):
    bfgs_flush_steps: int = 1000
    bfgs_flush_thre: float = 0.98
    als_max_steps: int = 3
    step_size: float = 1.0
    min_step_size: float = 0.33
    alpha: float = 0.01
    beta: float = None
    activation_tol: float = 100.0
    update_using_gperp: bool = False

    steps_since_last_flush: int = 1

    en_func: Callable[[np.array], float] = None
    grad_func: Callable[[np.array], np.array] = None
    x0: np.array = None
    use_linesearch: bool = True

    hess_history: list = field(default_factory=list)

    def __post_init__(self):
        if self.beta is None:
            beta = (self.min_step_size / self.step_size) ** (1 / self.als_max_steps)
            self.beta = beta

        if not hasattr(self, "bfgs_hess"):
            if self.x0 is not None:
                bfgs_hess = np.eye(self.x0.flatten().shape[0])
                # self.hess_history.append(bfgs_hess)
                # self.bfgs_hess = bfgs_hess
                self.hess_history.append(bfgs_hess)
            else:
                pass
                # raise NotImplementedError

    def flush_chain_hess(self, chain):
        chain.bfgs_hess = np.eye(chain.gradients.flatten().shape[0])

    def flush_hess(self):
        """
        will flush internal hessian
        """
        # self.bfgs_hess = np.eye(self.x0.flatten().shape[0])
        self.hess_history[-1] = np.eye(self.x0.flatten().shape[0])

    def linesearch(self, x_0, z):
        c1 = 10e-4
        c2 = 0.1
        en0 = self.en_func(x_0)
        grad = self.grad_func(x_0)
        step = 0
        converged = False
        t = self.step_size
        while not converged:
            x_prime = x_0 + (t * z)
            en_prime = self.en_func(x_prime)
            g = self.grad_func(x_prime)
            armijo = en_prime <= en0 + c1 * t * np.dot(z, grad)
            wolfe = -np.dot(z, g) <= -c2 * np.dot(z, grad)
            if armijo and wolfe:
                converged = True
            else:
                t *= self.beta
            step += 1
            if step == self.als_max_steps:
                break
        # return x_prime, en_prime, np.asarray(g)

        scaling = 1
        if self._overshot_max_step(gradients=z, disp=t):
            scaling = (1 / (np.linalg.norm(z * t))) * self.step_size

        return t * scaling

    def _update_hessian(self, orig_gradients, disp, grad_prime, gradients, hess_prev):
        # hessian update from wikipedia
        sk = (-gradients * disp).flatten().reshape(-1, 1)
        yk = (grad_prime - orig_gradients).flatten().reshape(-1, 1)

        # print(f"yk={yk.shape}\nsk={sk.shape}")

        ## this updates the hessian that would have to be inverted
        alpha = 1.0 / (np.dot(yk.T, sk))
        beta = -1.0 / (sk.T @ hess_prev @ sk)

        print(f"{alpha=} // {beta=}")

        u_vec = yk
        v_vec = hess_prev @ sk

        # A = ((yk@yk.T) / (yk.T@sk))
        # B = (hess_prev@sk@sk.T@(hess_prev.T)) / (sk.T@hess_prev@sk)

        ## this updates an approximate inverted hessian
        # A = ((sk.T*yk + yk.T*hess_prev*yk)*(sk*sk.T)) / ((sk.T*yk)**2)

        # B = (hess_prev*yk*sk.T + sk*yk.T*hess_prev) / (sk.T*yk)
        # new_hess = hess_prev + A - B
        new_hess = hess_prev + alpha * u_vec * u_vec.T + beta * v_vec * v_vec.T

        # hess_upt = A - B

        # if np.amax(gradients) < self.ACTIVATION_TOL:
        # self.bfgs_hess = new_hess
        self.hess_history.append(new_hess)
        # else:
        # self.bfgs_hess = hess_prev

    def _overshot_max_step(self, gradients, disp):
        return np.linalg.norm(gradients * disp) > self.step_size

    def _compute_and_matmul_inv_hess(self, hess_prev, orig_grad):

        orig_shape = orig_grad.shape

        grad_step_flat = orig_grad.flatten()
        grad_step_flat = np.linalg.inv(hess_prev) @ grad_step_flat
        grad_step = grad_step_flat.reshape(orig_shape)

        return grad_step

    def bfgs_step(self, x_vector):
        if np.mod(self.steps_since_last_flush, self.bfgs_flush_steps) == 0:
            self.flush_hess()
            self.steps_since_last_flush = 0

        # hess_prev = self.bfgs_hess.copy()
        hess_prev = self.hess_history[-1].copy()

        orig_grad = self.grad_func(x_vector)
        gradients = self._compute_and_matmul_inv_hess(
            hess_prev=hess_prev, orig_grad=orig_grad
        )
        disp = self.linesearch(x_0=x_vector, z=-1 * gradients)

        xprime = x_vector - gradients * disp
        grad_prime = self.grad_func(xprime)

        self._update_hessian(
            orig_gradients=orig_grad,
            disp=disp,
            grad_prime=grad_prime,
            gradients=gradients,
            hess_prev=hess_prev,
        )

        if self._gradient_correlation(grad_prime, gradients) < self.bfgs_flush_thre:
            self.flush_hess()
            self.steps_since_last_flush = 0

        self.steps_since_last_flush += 1
        # self.hess_history.append(self.bfgs_hess)

        return xprime

    def _gradient_correlation(self, vec1, vec2):
        chain1_vec = np.array(vec1).flatten()
        chain1_vec = chain1_vec / np.linalg.norm(chain1_vec)

        chain2_vec = np.array(vec2).flatten()
        chain2_vec = chain2_vec / np.linalg.norm(chain2_vec)

        projector = np.dot(chain1_vec, chain2_vec)
        normalization = np.dot(chain1_vec, chain1_vec)
        # print(projector / normalization)

        return projector / normalization

    ##############################

    def optimize_step(self, chain, chain_gradients):
        max_disp = self.step_size
        atomn = chain[0].coords.shape[0]
        scaling = 1

        if np.amax(np.abs(chain.gradients)) <= self.activation_tol:

            if np.mod(self.steps_since_last_flush, self.bfgs_flush_steps) == 0:
                self.flush_chain_hess(chain=chain)
                self.steps_since_last_flush = 0

            hess_prev = chain.bfgs_hess
            orig_shape = chain_gradients.shape

            grad_step_flat = chain_gradients.flatten()
            grad_step_flat = np.linalg.inv(hess_prev) @ grad_step_flat
            grad_step = grad_step_flat.reshape(orig_shape)

            chain_gradients = grad_step

            if self.use_linesearch:

                disp = ALS.ArmijoLineSearch(
                    chain=chain,
                    t=self.step_size,
                    alpha=self.alpha,
                    beta=self.beta,
                    grad=chain_gradients,
                    max_steps=self.als_max_steps,
                )

            else:
                disp = self.min_step_size
        else:
            disp = self.min_step_size
            grad_step = chain_gradients

        new_chain_gradients_fails = True
        retry_count_max = 5
        retry_count = 0
        while new_chain_gradients_fails and retry_count < retry_count_max:
            try:
                if (
                    np.linalg.norm(chain_gradients * disp) > max_disp
                ):  # if step size is too large
                    scaling = (1 / (np.linalg.norm(chain_gradients))) * max_disp

                new_chain_coordinates = (
                    chain.coordinates - chain_gradients * disp * scaling
                )
                new_nodes = []
                for node, new_coords in zip(chain.nodes, new_chain_coordinates):

                    new_nodes.append(node.update_coords(new_coords))

                # need to copy the gradients from the converged nodes
                new_chain = Chain(new_nodes, parameters=chain.parameters)
                for new_node, old_node in zip(new_chain.nodes, chain.nodes):
                    if old_node.converged:
                        new_node._cached_energy = old_node._cached_energy
                        new_node._cached_gradient = old_node._cached_gradient

                if np.amax(np.abs(chain.gradients)) <= self.activation_tol:
                    # hessian update from wikipedia
                    if self.update_using_gperp:
                        sk = (
                            ((-chain.get_g_perps()) * disp).flatten().reshape(-1, 1)
                        )  # warning
                        yk = (
                            (new_chain.get_g_perps() - chain.get_g_perps())
                            .flatten()
                            .reshape(-1, 1)
                        )  # warning
                    else:
                        sk = (-grad_step * disp).flatten().reshape(-1, 1)
                        yk = (
                            (new_chain.gradients - chain.gradients)
                            .flatten()
                            .reshape(-1, 1)
                        )

                    # print(f"yk={yk.shape}\nsk={sk.shape}")

                    ## this updates the hessian that would have to be inverted
                    A = (yk @ yk.T) / (yk.T @ sk)
                    B = (hess_prev @ sk @ sk.T @ (hess_prev.T)) / (
                        sk.T @ hess_prev @ sk
                    )

                    ## this updates an approximate inverted hessian
                    # A = ((sk.T*yk + yk.T*hess_prev*yk)*(sk*sk.T)) / ((sk.T*yk)**2)

                    # B = (hess_prev*yk*sk.T + sk*yk.T*hess_prev) / (sk.T*yk)
                    new_hess = hess_prev + A - B

                    # hess_upt = A - B

                    new_chain.bfgs_hess = new_hess

                    # print(np.linalg.norm(new_hess))

                else:
                    new_chain.bfgs_hess = np.eye(chain.gradients.flatten().shape[0])

                if new_chain._gradient_correlation(chain) < self.bfgs_flush_thre:
                    self.flush_chain_hess(new_chain)
                    self.steps_since_last_flush = 0

                self.steps_since_last_flush += 1

                new_chain_gradients_fails = False
            except:
                print("BFGS: step size was too big. decreasing and flushing")
                disp *= 0.5
                retry_count += 1

        return new_chain
