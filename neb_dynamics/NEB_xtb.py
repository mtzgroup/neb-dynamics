from dataclasses import dataclass

import numpy as np
from tdstructure import TDStructure
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from ALS_xtb import ArmijoLineSearch

ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROMS = 1 / ANGSTROM_TO_BOHR


@dataclass
class neb:
    def optimize_chain(self, chain, grad_func, en_func, k, en_thre=0.0005, grad_thre=0.0005, max_steps=1000):
        chain_traj = []
        nsteps = 0
        # ideal_dist = np.linalg.norm(np.array(chain[-1].coords) - np.array(chain[0].coords)) / len(
        #     chain
        # )
        chain_previous = chain.copy()
        nodes_converged = np.zeros(len(chain))

        while nsteps < max_steps:
            # print(f"--->On step {nsteps}")
            new_chain = self.update_chain(
                chain=chain_previous,
                k=k,
                en_func=en_func,
                grad_func=grad_func,
                # ideal_dist=ideal_dist,
                nodes_converged=nodes_converged,
            )

            chain_traj.append(new_chain)
            nodes_converged = self._chain_converged(
                chain_previous=chain_previous,
                new_chain=new_chain,
                en_func=en_func,
                en_thre=en_thre,
                grad_func=grad_func,
                grad_thre=grad_thre,
            )
            if False not in nodes_converged:
                print("Chain converged!")
                return new_chain, chain_traj

            chain_previous = new_chain.copy()
            nsteps += 1

        print("Chain did not converge...")
        return new_chain, chain_traj

    def update_chain(self, chain, k, en_func, grad_func, nodes_converged):

        chain_copy = np.zeros_like(chain)
        chain_copy[0] = chain[0]
        chain_copy[-1] = chain[-1]

        for i in range(1, len(chain) - 1):
            if nodes_converged[i]:
                chain_copy[i] = chain[i]
                continue

            print(f"updating node {i}...")
            view = chain[i - 1: i + 2]
            # print(f"{view=}")
            grad = self.spring_grad_neb(
                view,
                k=k,
                # ideal_distance=ideal_dist,
                grad_func=grad_func,
                en_func=en_func,
            )
            # dr = 0.01

            dr = ArmijoLineSearch(struct=chain[i], grad=grad, t=1, alpha=0.3, beta=0.8, f=en_func)

            coords_new_bohr = chain[i].coords_bohr - grad * dr
            coords_new = coords_new_bohr * BOHR_TO_ANGSTROMS

            p_new = TDStructure.from_coords_symbs(coords=coords_new, symbs=chain[i].symbols, tot_charge=chain[i].charge, tot_spinmult=chain[i].spinmult)

            chain_copy[i] = p_new

        return chain_copy

    def en_func(self, tdstruct):
        coords = tdstruct.coords_bohr
        atomic_numbers = tdstruct.atomic_numbers

        calc = Calculator(get_method("GFN2-xTB"), numbers=np.array(atomic_numbers), positions=coords, charge=tdstruct.charge, uhf=tdstruct.spinmult - 1)

        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()

        return res.get_energy()

    def grad_func(self, tdstruct):

        coords = tdstruct.coords_bohr
        atomic_numbers = tdstruct.atomic_numbers

        # blockPrint()
        calc = Calculator(get_method("GFN2-xTB"), numbers=np.array(atomic_numbers), positions=coords, charge=tdstruct.charge, uhf=tdstruct.spinmult - 1)
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()

        return res.get_gradient()

    def opt_func(self, tdstruct, en_func, grad_func, en_thre=0.0001, grad_thre=0.0001, maxsteps=5000):
        # coords = tdstruct.coords_bohr

        # atoms = Atoms(
        #         symbols = tdstruct.symbols.tolist(),
        #         positions = coords,
        #     )

        # atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
        # opt = LBFGS(atoms)
        # opt.run(fmax=0.1)

        # opt_struct = TDStructure.from_coords_symbs(
        #     coords=atoms.positions*0.529177,
        #     symbs=tdstruct.symbols,
        #     tot_charge=tdstruct.charge,
        #     tot_spinmult=tdstruct.spinmult)

        # return opt_struct

        e0 = en_func(tdstruct)
        g0 = grad_func(tdstruct)
        dr = ArmijoLineSearch(struct=tdstruct, grad=g0, t=1, alpha=0.3, beta=0.8, f=en_func)
        print(f"DR -->{dr}")
        count = 0

        coords1 = tdstruct.coords_bohr - dr * g0
        tdstruct_prime = TDStructure.from_coords_symbs(coords=coords1 * BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols, tot_charge=tdstruct.charge, tot_spinmult=tdstruct.spinmult)

        e1 = en_func(tdstruct_prime)
        g1 = grad_func(tdstruct_prime)

        struct_conv = (np.abs(e1 - e0) < en_thre) and False not in (np.abs(g1 - g0) < grad_thre).flatten()

        while not struct_conv and count < maxsteps:
            count += 1

            e0 = e1
            g0 = g1

            dr = ArmijoLineSearch(struct=tdstruct, grad=g0, t=1, alpha=0.3, beta=0.8, f=en_func)
            coords1 = tdstruct.coords_bohr - dr * g0
            tdstruct_prime = TDStructure.from_coords_symbs(coords=coords1 * BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols, tot_charge=tdstruct.charge, tot_spinmult=tdstruct.spinmult)

            e1 = en_func(tdstruct_prime)
            g1 = grad_func(tdstruct_prime)

            struct_conv = (np.abs(e1 - e0) < en_thre) and False not in (np.abs(g1 - g0) < grad_thre).flatten()

        print(f"Converged --> {struct_conv} in {count} steps")
        return tdstruct_prime

    def _create_tangent_path(self, view, en_func):
        en_2 = en_func(view[2])
        en_1 = en_func(view[1])
        en_0 = en_func(view[0])

        if en_2 > en_1 and en_1 > en_0:
            return view[2].coords - view[1].coords
        elif en_2 < en_1 and en_1 < en_2:
            return view[1].coords - view[0].coords

        else:
            deltaV_max = max(np.abs(en_2 - en_1), np.abs(en_0 - en_1))
            deltaV_min = min(np.abs(en_2 - en_1), np.abs(en_0 - en_1))

            if en_2 > en_0:
                tan_vec = (view[2].coords - view[1].coords) * deltaV_max + (view[1].coords - view[0].coords) * deltaV_min
            elif en_2 < en_0:
                tan_vec = (view[2].coords - view[1].coords) * deltaV_min + (view[1].coords - view[0].coords) * deltaV_max

            return tan_vec

    def spring_grad_neb(self, view, grad_func, k, en_func):

        # neighs = view[[0, 2]]

        vec_tan_path = self._create_tangent_path(view, en_func=en_func)
        unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

        pe_grad = grad_func(view[1])

        pe_grad_nudged_const = np.sum(pe_grad * unit_tan_path, axis=1).reshape(-1, 1)  # Nx1 matrix
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path

        grads_neighs = []
        force_springs = []

        # for neigh in neighs:
        # dist = np.abs(neigh.coords - view[1].coords)
        # force_spring = -k*(dist - ideal_distance) # magnitude tells me if it's attractive or repulsive
        force_spring = -k * (np.abs(view[2].coords - view[1].coords) - np.abs(view[1].coords - view[0].coords))  # magnitude tells me if it's attractive or repulsive
        # check if force vector is pointing towards neighbor
        direction = np.sum((view[2].coords - view[1].coords) * force_spring, axis=1)

        # flip vectors that weren't pointing towards neighbor
        force_spring[direction < 0] *= -1

        force_springs.append(force_spring)

        force_spring_nudged_const = np.sum(force_spring * unit_tan_path, axis=1).reshape(-1, 1)
        force_spring_nudged = force_spring - force_spring_nudged_const * unit_tan_path

        grads_neighs.append(force_spring_nudged)

        tot_grads_neighs = np.sum(grads_neighs, axis=0)

        # ANTI-KINK FORCE
        # print(f"{force_springs=}")
        force_springs = np.sum(force_springs, axis=0)
        # print(f"{force_springs=}")

        vec_2_to_1 = view[2].coords - view[1].coords
        vec_1_to_0 = view[1].coords - view[0].coords
        cos_phi = np.sum(vec_2_to_1 * vec_1_to_0, axis=1).reshape(-1, 1) / (np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0))

        f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))

        proj_force_springs = force_springs - np.sum(force_springs * unit_tan_path, axis=1).reshape(-1, 1) * unit_tan_path

        # print(f"{pe_grad_nudged=} {tot_grads_neighs=} {f_phi=} {proj_force_springs=}")

        return (pe_grad_nudged - tot_grads_neighs) + f_phi * (proj_force_springs)

    def _check_en_converged(self, chain_prev, chain_new, en_func, en_thre):
        bools = np.ones(len(chain_new))
        for i in range(1, len(chain_prev) - 1):
            node_prev = chain_prev[i]
            node_new = chain_new[i]

            delta_e = np.abs(en_func(node_new) - en_func(node_prev))
            if delta_e > en_thre:
                bools[i] = 0

        return bools

    def _check_grad_converged(self, chain_prev, chain_new, grad_func, grad_thre):
        bools = np.ones(len(chain_new))
        for i in range(1, len(chain_prev) - 1):
            node_prev = chain_prev[i]
            node_new = chain_new[i]

            delta_grad = np.abs(grad_func(node_new) - grad_func(node_prev))

            if True in delta_grad > grad_thre:
                bools[i] = 0
        return bools

    def _chain_converged(self, chain_previous, new_chain, en_func, en_thre, grad_func, grad_thre):
        """
        returns list of bools saying whether a node has converged
        """

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

        return en_converged * grad_converged
