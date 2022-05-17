from dataclasses import dataclass
from retropaths.abinitio.tdstructure import TDStructure
import numpy as np
from ALS_xtb import ArmijoLineSearch


@dataclass
class neb:
    def optimize_chain(
        self, chain, grad_func, en_func, k, en_thre=0.01, grad_thre=0.01, max_steps=1000
    ):

        chain_traj = []
        nsteps = 0
        ideal_dist = np.linalg.norm(np.array(chain[-1].coords) - np.array(chain[0].coords)) / len(
            chain
        )
        chain_previous = chain.copy()

        while nsteps < max_steps:
            new_chain = self.update_chain(
                chain=chain_previous,
                k=k,
                en_func=en_func,
                grad_func=grad_func,
                ideal_dist=ideal_dist
            )

            chain_traj.append(new_chain)

            if self._chain_converged(
                chain_previous=chain_previous,
                new_chain=new_chain,
                en_func=en_func,
                en_thre=en_thre,
                grad_func=grad_func,
                grad_thre=grad_thre,
            ):
                print("Chain converged!")
                return new_chain, chain_traj

            chain_previous = new_chain.copy()
            nsteps += 1

        print("Chain did not converge...")
        return new_chain, chain_traj

    def update_chain(self, chain, k, en_func, grad_func, ideal_dist):

        chain_copy = np.zeros_like(chain)
        chain_copy[0] = chain[0]
        chain_copy[-1] = chain[-1]

        

        for i in range(1, len(chain) - 1):
            print(f"updating node {i}...")
            view = chain[i - 1 : i + 2]

            grad = self.spring_grad_neb(
                view, k=k, ideal_distance=ideal_dist, grad_func=grad_func
            )

            dr = 0.01

            # dr, _ = ArmijoLineSearch(
            #     f=en_func,
            #     xk=chain[i],
            #     gfk=grad,
            #     phi0=en_func(chain[i]),
            #     alpha0=1,
            #     pk=-1 * grad,
            # )

            coords_new = chain[i].coords - grad * dr
            p_new = TDStructure.from_coords_symbs(coords=coords_new, 
                        symbs=chain[i].symbols, tot_charge=chain[i].charge,
                        tot_spinmult=chain[i].spinmult)

            chain_copy[i] = p_new

        return chain_copy

    def spring_grad_neb(self, view, grad_func, k, ideal_distance):

        neighs = view[[0, 2]]
        # neighs = [view[2]]

        vec_tan_path = neighs[1].coords - neighs[0].coords
        # vec_tan_path = view[2] - view[1]

        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        print(f"{unit_tan_path=}")

        pe_grad = grad_func(view[1])
        
        pe_grad_nudged_const = np.sum(pe_grad * unit_tan_path, axis=1).reshape(-1,1)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path

        grads_neighs = []
        force_springs = []

        for neigh in neighs:
            dist = np.abs(neigh.coords - view[1].coords)
            force_spring = -k*(dist - ideal_distance)


            # anywhere that the coordinate is past the current point
            # that force must be reversed (i.e. we calculated it would
            # be attractive, but because of the relative position it would
            # be repulsive)
            bools = neigh.coords > view[1].coords
            force_spring[bools] *= -1

            force_springs.append(force_spring)

            force_spring_nudged_const = np.sum(force_spring*unit_tan_path, axis=1).reshape(-1,1)
            force_spring_nudged = (
                force_spring - force_spring_nudged_const * unit_tan_path
            )

            grads_neighs.append(force_spring_nudged)

        tot_grads_neighs = np.sum(grads_neighs, axis=0)

        ### ANTI-KINK FORCE
        force_springs = np.sum(force_springs, axis=0)

        vec_2_to_1 = view[2].coords - view[1].coords
        vec_1_to_0 = view[1].coords - view[0].coords
        cos_phi = np.sum(vec_2_to_1*vec_1_to_0, axis=1).reshape(-1,1) / (
            np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0)
        )

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))

        proj_force_springs = (
            force_springs - np.sum(force_springs*unit_tan_path, axis=1).reshape(-1,1) * unit_tan_path
        )

        return (pe_grad_nudged - tot_grads_neighs) + f_phi * (proj_force_springs)

    def _check_en_converged(self, chain_prev, chain_new, en_func, en_thre):
        for i in range(1, len(chain_prev) - 1):
            node_prev = chain_prev[i]
            node_new = chain_new[i]

            delta_e = np.abs(en_func(node_new) - en_func(node_prev))
            if delta_e > en_thre:
                return False
        return True

    def _check_grad_converged(self, chain_prev, chain_new, grad_func, grad_thre):
        for i in range(
            1, len(chain_prev) - 1
        ):  
            node_prev = chain_prev[i]
            node_new = chain_new[i]

            delta_grad_x, delta_grad_y = np.abs(
                grad_func(node_new) - grad_func(node_prev)
            )
            

            if (delta_grad_x > grad_thre) or (delta_grad_y > grad_thre):
                return False
        return True

    def _chain_converged(
        self, chain_previous, new_chain, en_func, en_thre, grad_func, grad_thre
    ):

        en_converged = self._check_en_converged(
            chain_prev=chain_previous,
            chain_new=new_chain,
            en_func=en_func,
            en_thre=en_thre,
        )
        grad_converged = self._check_grad_converged(
            chain_prev=chain_previous,
            chain_new=new_chain,
            grad_func=grad_func,
            grad_thre=grad_thre,
        )

        return en_converged and grad_converged
