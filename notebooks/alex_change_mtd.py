from neb_dynamics.TreeNode import TreeNode

h = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Robinson-Gabriel-Synthesis/wb97xd3_def2svp_2024/"))

h.output_chain.to_trajectory().write_trajectory("/home/jdep/hey_todd.xyz")

h_xtb = TreeNode.read_from_disk(Path('/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Robinson-Gabriel-Synthesis/explicit_solvent_xtb/'))

h_xtb.output_chain.plot_chain()

h_xtb.output_chain[0].tdstructure.molecule_rp.draw(mode='d3')

from retropaths.molecules.molecule import Molecule
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.reactions.changes import Changes3D, Changes3DList, ChargeChanges
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
from retropaths.reactions.template import ReactionTemplate

mol = Molecule.from_smiles('C=C=O.O.O')

td = TDStructure.from_RP(mol)

# +
single = [(2,9), (0, 10), # new bonds
          (0,1),(2, 1) # double bonds which are now single
         ]

double = [(1,4)]

delete = [(4,9), (4,10), (1,0)]

c3d_single = [Changes3D(start=s, end=e, bond_order=1) for s, e in single]
c3d_double = [Changes3D(start=s, end=e, bond_order=2) for s, e in double]
c3d_delete = [Changes3D(start=s, end=e, bond_order=1) for s, e in delete]
# -

d = {'charges': [],
 'delete':delete,
'double':double,
 'single':single}

conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='alex', reactants=mol, changes_react_to_prod_dict=d,
           conditions=conds, rules=rules)

temp.reactants.draw()

c3d_list = Changes3DList(deleted=c3d_delete, forming=c3d_single+c3d_double, charges=[])

root = TDStructure.from_RP(mol)

target = root.copy()

target.delete_bonds(c3d_delete)

target.add_bonds(c3d_single)

target.add_bonds(c3d_double)

target.gum_mm_optimization()

# +
from neb_dynamics.NEB import NoneConvergedException
from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.Linesearch import Linesearch
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.Inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from nodes.node3d import Node3D

from neb_dynamics.Chain import Chain
# -

tr = Trajectory([root, target]).run_geodesic(nimages=12)

nbi = NEBInputs(tol=0.001*BOHR_TO_ANGSTROMS,early_stop_force_thre=0.01,
                early_stop_chain_rms_thre=1, v=True)
# cni = ChainInputs(k=0.1, delta_k=0.09,node_class=Node3D_TC,node_freezing=True)
cni = ChainInputs(k=0.1, delta_k=0.09,node_class=Node3D,node_freezing=True)
gii = GIInputs(nimages=12)
optimizer = BFGS(bfgs_flush_steps=20, bfgs_flush_thre=0.80, use_linesearch=False,
                 step_size=3,
                 min_step_size= 0.5,
                 activation_tol=0.1
            )

from neb_dynamics.MSMEP import MSMEP


initial_chain = Chain.from_traj(tr, parameters=cni)

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=optimizer)

from pathlib import Path

data_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/")

all_paths = list(data_dir.iterdir())

subset = []
subset_tags = []
for path in all_paths:
    tag = "".join(path.stem.split("-")[:2])
    new_tag = tag not in subset_tags


    if new_tag:
        subset.append(path)
        subset_tags.append(tag)




subset

smaller_subset = []
sizes = []
for p in subset:
    try:
        tr_path = p / 'initial_guess.xyz'
        tr = Trajectory.from_xyz(tr_path)
        natoms = len(tr[0].coords)
        if natoms < 40:
            smaller_subset.append(p)
    except:
        continue

with open("/home/jdep/T3D_data/msmep_draft/comparisons_dft/reactions_todo.txt","w+") as f:
    for p in smaller_subset:
        f.write(str(p.resolve())+"\n")

len(smaller_subset)

h, out = m.find_mep_multistep(initial_chain)

out.to_trajectory()

# # root.pseudoalign(c3d_list)

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.gum_mm_optimization()

target.molecule_rp.draw(mode='d3')





target

# # Node3D shit

# +
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from retropaths.abinitio.tdstructure import TDStructure
from xtb.ase.calculator import XTB
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.Node import Node
import multiprocessing as mp


@dataclass
class Node3D_MTD(Node):
    tdstructure: TDStructure
    reference_trajs: list
    strength: float
    alpha: float
    converged: bool = False
    do_climb: bool = False


    _cached_energy: float | None = None
    _cached_gradient: np.array | None = None

    RMSD_CUTOFF = 0.5
    KCAL_MOL_CUTOFF = 1


    # reference_trajs = REFERENCE_OPT_TRAJ
    # reference_trajs = [n_orig.optimized.to_trajectory()]
    # reference_trajs = [out_chain.to_trajectory()]

    @property
    def coords(self):
        return self.tdstructure.coords

    @property
    def coords_bohr(self):
        return self.tdstructure.coords * ANGSTROM_TO_BOHR

    @staticmethod
    def en_func(node: Node3D_MTD):
        res = Node3D_MTD.run_xtb_calc(node.tdstructure)
        energy =  res.get_energy()
        gradient_mtd, energy_mtd = node.mtd_grad_energy(structure=node.tdstructure.coords)
        return energy + energy_mtd

    @staticmethod
    def grad_func(node: Node3D_MTD):
        res = Node3D_MTD.run_xtb_calc(node.tdstructure)
        gradient =  res.get_gradient() * BOHR_TO_ANGSTROMS
        gradient_mtd, energy_mtd = node.mtd_grad_energy(node.tdstructure.coords)


        return gradient + gradient_mtd

    def mtd_grad_energy(self, structure: np.array):

        n_atoms = structure.shape[0]
        gradient = np.zeros((n_atoms,3))
        energy = 0

        # for reference_chain in REFERENCE_OPT_TRAJ:
        for reference_chain in self.reference_trajs:
            for reference in reference_chain:
                # reference = REFERENCE.coords

                rmsd, g_rmsd = RMSD(structure=structure, reference=reference)
                biaspot_i  = self.strength*n_atoms*math.exp(-(self.alpha* rmsd**2))
                biasgrad_i =  -2*self.alpha*g_rmsd*biaspot_i * rmsd

                gradient += biasgrad_i
                energy += biaspot_i
        weights = np.array([np.sqrt(get_mass(s)) for s in self.tdstructure.symbols])
        weights = weights  / sum(weights)
        gradient = gradient * weights.reshape(-1,1)

        return gradient, energy

    @cached_property
    def energy(self):
        if self._cached_energy is not None:
            return self._cached_energy
        else:
            return Node3D_MTD.en_func(self)

    def do_geometry_optimization(self) -> Node3D_MTD:

        max_steps=2500
        ss=1
        tol=0.001
        nsteps = 0
        traj = []

        node = self.copy()
        while nsteps < max_steps:
            traj.append(node)
            grad, _ = node.mtd_grad_energy(node.tdstructure.coords)
            grad_energy = node.tdstructure.gradient_xtb()
            grad+=grad_energy

            if np.linalg.norm(grad) < tol:
                break
            new_coords = node.coords - ss*grad
            node = node.update_coords(new_coords)
            print(f"|grad|={np.linalg.norm(grad)}",end='\r')
            nsteps+=1

        if np.linalg.norm(grad) < tol:
            print(f"\nConverged in {nsteps} steps!")
        else:
            print(f"\nDid not converge in {nsteps} steps.")

        return node

    def is_identical(self, other) -> bool:
        return self.tdstructure.molecule_rp.is_bond_isomorphic_to(
            other.tdstructure.molecule_rp
        )

    @cached_property
    def gradient(self):
        if self._cached_gradient is not None:
            return self._cached_gradient
        else:
            return Node3D_MTD.grad_func(self)

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
        # return np.sum(first * second, axis=1).reshape(-1, 1)
        return np.tensordot(first, second)

    # i want to cache the result of this but idk how caching works
    def run_xtb_calc(tdstruct: TDStructure):
        atomic_numbers = tdstruct.atomic_numbers
        calc = Calculator(
            get_method("GFN2-xTB"),
            numbers=np.array(atomic_numbers),
            positions=tdstruct.coords_bohr,
            charge=tdstruct.charge,
            uhf=tdstruct.spinmult - 1,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()
        return res

    def get_nudged_pe_grad(self, unit_tangent, gradient):
        """
        Returns the component of the gradient that acts perpendicular to the path tangent
        """
        pe_grad = gradient

        pe_grad_nudged_const = self.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
        return pe_grad_nudged

    def copy(self):
        return Node3D_MTD(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
            reference_trajs=self.reference_trajs,
            strength=self.strength,
            alpha=self.alpha
        )

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()

        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        return Node3D_MTD(
            tdstructure=copy_tdstruct, converged=self.converged, do_climb=self.do_climb, reference_trajs=self.reference_trajs,\
            strength=self.strength,
            alpha=self.alpha
        )

    def opt_func(self, v=True):
        atoms = Atoms(
            symbols=self.tdstructure.symbols.tolist(),
            positions=self.coords,  # ASE works in angstroms
        )

        atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
        if not v:
            opt = LBFGS(atoms, logfile=None)
        else:
            opt = LBFGS(atoms)
        opt.run(fmax=0.1)

        opt_struct = TDStructure.from_coords_symbols(
            coords=atoms.positions,
            symbols=self.tdstructure.symbols,
            tot_charge=self.tdstructure.charge,
            tot_spinmult=self.tdstructure.spinmult,
        )  # ASE works in agnstroms

        return opt_struct

    def is_a_molecule(self):
        return True

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @property
    def hessian(self: Node3D_MTD):
        dr = 0.01  # displacement vector, Bohr
        numatoms = self.tdstructure.atomn
        approx_hess = []
        for n in range(numatoms):
            # for n in range(2):
            grad_n = []
            for coord_ind, coord_name in enumerate(["dx", "dy", "dz"]):
                # print(f"doing atom #{n} | {coord_name}")

                coords = np.array(self.coords, dtype="float64")

                # print(coord_ind, coords[n, coord_ind])
                coords[n, coord_ind] = coords[n, coord_ind] + dr
                # print(coord_ind, coords[n, coord_ind])

                node2 = self.copy()
                node2 = node2.update_coords(coords)

                delta_grad = (node2.gradient - self.gradient) / dr
                # print(delta_grad)
                grad_n.append(delta_grad.flatten())

            approx_hess.extend(grad_n)
        approx_hess = np.array(approx_hess)
        # raise AlessioError(f"{approx_hess.shape}")

        approx_hess_sym = 0.5 * (approx_hess + approx_hess.T)
        assert self.check_symmetric(
            approx_hess_sym, rtol=1e-3, atol=1e-3
        ), "Hessian not symmetric for some reason"

        return approx_hess_sym

    @property
    def input_tuple(self):
        return (
            self.tdstructure.atomic_numbers,
            self.tdstructure.coords_bohr,
            self.tdstructure.charge,
            self.tdstructure.spinmult,
        )


    @staticmethod
    def calc_xtb_ene_grad_from_input_tuple(tuple):
        atomic_numbers, coords_bohr, charge, spinmult = tuple

        calc = Calculator(
            get_method("GFN2-xTB"),
            numbers=np.array(atomic_numbers),
            positions=coords_bohr,
            charge=charge,
            uhf=spinmult - 1,
        )
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()

        return res.get_energy(), res.get_gradient() * BOHR_TO_ANGSTROMS


    @staticmethod
    def calc_mtd_ene_grad_from_input_tuple(inp):

        structure, reference_opt_traj, strength, alpha = inp
        n_atoms = structure.shape[0]
        gradient = np.zeros((n_atoms,3))
        energy = 0

        for reference_chain in reference_opt_traj:
            for reference in reference_chain:



                c1 = np.array(structure)
                c2 = np.array(reference)
                bary1 = np.mean(c1, axis = 0) #barycenters
                bary2 = np.mean(c2, axis = 0)

                c1 = c1 - bary1 #shift origins to barycenter
                c2 = c2 - bary2

                N = len(c1)
                R = np.dot(np.transpose(c1), c2) #correlation matrix

                F = np.array([[(R[0, 0] + R[1, 1] + R[2, 2]), (R[1, 2] - R[2, 1]), (R[2, 0] - R[0, 2]), (R[0, 1] - R[1, 0])],
                            [(R[1, 2] - R[2, 1]), (R[0, 0] - R[1, 1] - R[2, 2]), (R[1, 0] + R[0, 1]), (R[2, 0] + R[0, 2])],
                            [(R[2, 0] - R[0, 2]), (R[1, 0] + R[0, 1]), (-R[0, 0] + R[1, 1] - R[2, 2]), (R[1, 2] + R[2, 1])],
                            [(R[0, 1] - R[1, 0]), (R[2, 0] + R[0, 2]), (R[1, 2] + R[2, 1]), (-R[0, 0] - R[1, 1] + R[2, 2])]]) #Eq. 10 in Dill Quaternion RMSD paper (DOI:10.1002/jcc.20110)

                eigen = scipy.sparse.linalg.eigs(F, k = 1, which = 'LR') #find max eigenvalue and eigenvector
                lmax = float(eigen[0][0])
                qmax = np.array(eigen[1][0:4])
                qmax = np.float_(qmax)
                qmax = np.ndarray.flatten(qmax)
                rmsd = math.sqrt(abs((np.sum(np.square(c1)) + np.sum(np.square(c2)) - 2 * lmax)/N))  #square root of the minimum residual

                rot = np.array([[(qmax[0]**2 + qmax[1]**2 - qmax[2]**2 - qmax[3]**2), 2*(qmax[1]*qmax[2] - qmax[0]*qmax[3]), 2*(qmax[1]*qmax[3] + qmax[0]*qmax[2])],
                                [2*(qmax[1]*qmax[2] + qmax[0]*qmax[3]), (qmax[0]**2 - qmax[1]**2 + qmax[2]**2 - qmax[3]**2), 2*(qmax[2]*qmax[3] - qmax[0]*qmax[1])],
                                [2*(qmax[1]*qmax[3] - qmax[0]*qmax[2]), 2*(qmax[2]*qmax[3] + qmax[0]*qmax[1]), (qmax[0]**2 - qmax[1]**2 - qmax[2]**2 + qmax[3]**2)]]) #rotation matrix based on eigenvector corresponding $
                g_rmsd = (c1 - np.matmul(c2, rot))/(N*rmsd) #gradient of the rmsd


                biaspot_i  = strength*n_atoms*math.exp(-(alpha* rmsd**2))
                biasgrad_i =  -2*alpha*g_rmsd*biaspot_i * rmsd

                gradient += biasgrad_i
                energy += biaspot_i



        return energy, gradient * BOHR_TO_ANGSTROMS

    def is_identical(self, other) -> bool:
        return all([self._is_connectivity_identical(other), self._is_conformer_identical(other)])

    def _is_connectivity_identical(self, other) -> bool:
        connectivity_identical =  self.tdstructure.molecule_rp.is_bond_isomorphic_to(
            other.tdstructure.molecule_rp
        )
        return connectivity_identical

    def _is_conformer_identical(self, other) -> bool:
        if self._is_connectivity_identical(other):
            aligned_self = self.tdstructure.align_to_td(other.tdstructure)
            dist = RMSD(aligned_self.coords, other.tdstructure.coords)[0]
            en_delta = np.abs((self.energy - other.energy)*627.5)


            rmsd_identical = dist < self.RMSD_CUTOFF
            energies_identical = en_delta < self.KCAL_MOL_CUTOFF
            if rmsd_identical and energies_identical:
                conformer_identical = True

            if not rmsd_identical and energies_identical:
                # going to assume this is a rotation issue. Need To address.
                conformer_identical = False

            if not rmsd_identical and not energies_identical:
                conformer_identical = False

            if rmsd_identical and not energies_identical:
                conformer_identical = False
            print(f"\nRMSD : {dist} // |∆en| : {en_delta}\n")
            aligned_self.to_xyz(Path(f"/tmp/{round(dist,3)}_{round(en_delta, 3)}_self.xyz"))
            other.tdstructure.to_xyz(Path(f"/tmp/{round(dist,3)}_{round(en_delta, 3)}_other.xyz"))
            return conformer_identical
        else:
            return False




    @classmethod
    def calculate_energy_and_gradients_parallel(cls, chain):
        iterator = (
            (
                n.tdstructure.atomic_numbers,
                n.tdstructure.coords_bohr,
                n.tdstructure.charge,
                n.tdstructure.spinmult,
            )
            for n in chain.nodes
        )

        with mp.Pool() as p:
            ene_gradients = p.map(cls.calc_xtb_ene_grad_from_input_tuple, iterator)


        iterator2 = (
            (
                n.tdstructure.coords_bohr,
                [traj_coords*ANGSTROM_TO_BOHR for traj_coords in chain.nodes[0].reference_trajs],
                chain.parameters.mtd_strength,
                chain.parameters.mtd_alpha
            )
            for n in chain.nodes
        )

        with mp.Pool() as p2:
            ene_gradients_mtd = p2.map(cls.calc_mtd_ene_grad_from_input_tuple, iterator2)


        out_ene_gradients = []
        for i, (ene_bias, grad_bias) in enumerate(ene_gradients_mtd):
            orig_ene, orig_grad = ene_gradients[i]
            out_ene = orig_ene + ene_bias
            out_grad = orig_grad + grad_bias
            out_ene_gradients.append((out_ene, out_grad))
        return out_ene_gradients



# -

# # MTD NO energy

# +
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from retropaths.abinitio.tdstructure import TDStructure
from xtb.ase.calculator import XTB
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.Node import Node
import multiprocessing as mp


@dataclass
class Node3D_MTD_NoEne(Node):
    tdstructure: TDStructure
    reference_trajs: list
    strength: float
    alpha: float

    converged: bool = False
    do_climb: bool = False


    _cached_energy: float | None = None
    _cached_gradient: np.array | None = None


    # reference_trajs = REFERENCE_OPT_TRAJ
    # reference_trajs = [n_orig.optimized.to_trajectory()]
    # reference_trajs = [out_chain.to_trajectory()]

    @property
    def coords(self):
        return self.tdstructure.coords

    @property
    def coords_bohr(self):
        return self.tdstructure.coords * ANGSTROM_TO_BOHR

    @staticmethod
    def en_func(node: Node3D_MTD_NoEne):
        # res = Node3D_MTD.run_xtb_calc(node.tdstructure)
        # energy =  res.get_energy()
        gradient_mtd, energy_mtd = node.mtd_grad_energy(structure=node.tdstructure.coords)
        return energy_mtd

    @staticmethod
    def grad_func(node: Node3D_MTD_NoEne):
        # res = Node3D_MTD.run_xtb_calc(node.tdstructure)
        # gradient =  res.get_gradient() * BOHR_TO_ANGSTROMS
        gradient_mtd, energy_mtd = node.mtd_grad_energy(node.tdstructure.coords)
        return gradient_mtd

    def mtd_grad_energy(self, structure: np.array):

        n_atoms = structure.shape[0]
        gradient = np.zeros((n_atoms,3))
        energy = 0

        # for reference_chain in REFERENCE_OPT_TRAJ:
        for reference_chain in self.reference_trajs:
            for reference in reference_chain:
                # reference = REFERENCE.coords

                rmsd, g_rmsd = RMSD(structure=structure, reference=reference)
                biaspot_i  = self.strength*n_atoms*math.exp(-(self.alpha* rmsd**2))
                biasgrad_i =  -2*self.alpha*g_rmsd*biaspot_i * rmsd

                gradient += biasgrad_i
                energy += biaspot_i


        weights = np.array([np.sqrt(get_mass(s)) for s in self.tdstructure.symbols])
        weights = weights  / sum(weights)
        gradient = gradient * weights.reshape(-1,1)
        return gradient, energy

    @cached_property
    def energy(self):
        if self._cached_energy is not None:
            return self._cached_energy
        else:
            return Node3D_MTD_NoEne.en_func(self)

    def do_geometry_optimization(self) -> Node3D_MTD_NoEne:
        td_opt = self.tdstructure.xtb_geom_optimization()
        return Node3D_MTD_NoEne(tdstructure=td_opt)

    def is_identical(self, other) -> bool:
        return self.tdstructure.molecule_rp.is_bond_isomorphic_to(
            other.tdstructure.molecule_rp
        )

    @cached_property
    def gradient(self):
        if self._cached_gradient is not None:
            return self._cached_gradient
        else:
            return Node3D_MTD_NoEne.grad_func(self)

    @staticmethod
    def dot_function(first: np.array, second: np.array) -> float:
        # return np.sum(first * second, axis=1).reshape(-1, 1)
        return np.tensordot(first, second)

    def get_nudged_pe_grad(self, unit_tangent, gradient):
        """
        Returns the component of the gradient that acts perpendicular to the path tangent
        """
        pe_grad = gradient

        pe_grad_nudged_const = self.dot_function(pe_grad, unit_tangent)
        pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
        return pe_grad_nudged

    def copy(self):
        return Node3D_MTD_NoEne(
            tdstructure=self.tdstructure.copy(),
            converged=self.converged,
            do_climb=self.do_climb,
            reference_trajs=self.reference_trajs,
            strength=self.strength,
            alpha=self.alpha
        )

    def update_coords(self, coords: np.array) -> None:

        copy_tdstruct = self.tdstructure.copy()
        copy_tdstruct = copy_tdstruct.update_coords(coords=coords)
        return Node3D_MTD_NoEne(
            tdstructure=copy_tdstruct, converged=self.converged, do_climb=self.do_climb, reference_trajs=self.reference_trajs,
            strength=self.strength, alpha=self.alpha
        )

    def opt_func(self, v=True):
        atoms = Atoms(
            symbols=self.tdstructure.symbols.tolist(),
            positions=self.coords,  # ASE works in angstroms
        )

        atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
        if not v:
            opt = LBFGS(atoms, logfile=None)
        else:
            opt = LBFGS(atoms)
        opt.run(fmax=0.1)

        opt_struct = TDStructure.from_coords_symbols(
            coords=atoms.positions,
            symbols=self.tdstructure.symbols,
            tot_charge=self.tdstructure.charge,
            tot_spinmult=self.tdstructure.spinmult,
        )  # ASE works in agnstroms

        return opt_struct

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    @property
    def input_tuple(self):
        return (
            self.tdstructure.atomic_numbers,
            self.tdstructure.coords_bohr,
            self.tdstructure.charge,
            self.tdstructure.spinmult,
        )



    @staticmethod
    def calc_mtd_ene_grad_from_input_tuple(inp):

        structure, reference_opt_traj, strength, alpha = inp
        n_atoms = structure.shape[0]
        gradient = np.zeros((n_atoms,3))
        energy = 0


        for reference_chain in reference_opt_traj:
            for reference in reference_chain:



                c1 = np.array(structure)
                c2 = np.array(reference)
                bary1 = np.mean(c1, axis = 0) #barycenters
                bary2 = np.mean(c2, axis = 0)

                c1 = c1 - bary1 #shift origins to barycenter
                c2 = c2 - bary2

                N = len(c1)
                R = np.dot(np.transpose(c1), c2) #correlation matrix

                F = np.array([[(R[0, 0] + R[1, 1] + R[2, 2]), (R[1, 2] - R[2, 1]), (R[2, 0] - R[0, 2]), (R[0, 1] - R[1, 0])],
                            [(R[1, 2] - R[2, 1]), (R[0, 0] - R[1, 1] - R[2, 2]), (R[1, 0] + R[0, 1]), (R[2, 0] + R[0, 2])],
                            [(R[2, 0] - R[0, 2]), (R[1, 0] + R[0, 1]), (-R[0, 0] + R[1, 1] - R[2, 2]), (R[1, 2] + R[2, 1])],
                            [(R[0, 1] - R[1, 0]), (R[2, 0] + R[0, 2]), (R[1, 2] + R[2, 1]), (-R[0, 0] - R[1, 1] + R[2, 2])]]) #Eq. 10 in Dill Quaternion RMSD paper (DOI:10.1002/jcc.20110)

                eigen = scipy.sparse.linalg.eigs(F, k = 1, which = 'LR') #find max eigenvalue and eigenvector
                lmax = float(eigen[0][0])
                qmax = np.array(eigen[1][0:4])
                qmax = np.float_(qmax)
                qmax = np.ndarray.flatten(qmax)
                rmsd = math.sqrt(abs((np.sum(np.square(c1)) + np.sum(np.square(c2)) - 2 * lmax)/N))  #square root of the minimum residual

                rot = np.array([[(qmax[0]**2 + qmax[1]**2 - qmax[2]**2 - qmax[3]**2), 2*(qmax[1]*qmax[2] - qmax[0]*qmax[3]), 2*(qmax[1]*qmax[3] + qmax[0]*qmax[2])],
                                [2*(qmax[1]*qmax[2] + qmax[0]*qmax[3]), (qmax[0]**2 - qmax[1]**2 + qmax[2]**2 - qmax[3]**2), 2*(qmax[2]*qmax[3] - qmax[0]*qmax[1])],
                                [2*(qmax[1]*qmax[3] - qmax[0]*qmax[2]), 2*(qmax[2]*qmax[3] + qmax[0]*qmax[1]), (qmax[0]**2 - qmax[1]**2 - qmax[2]**2 + qmax[3]**2)]]) #rotation matrix based on eigenvector corresponding $
                g_rmsd = (c1 - np.matmul(c2, rot))/(N*rmsd) #gradient of the rmsd


                biaspot_i  = strength*n_atoms*math.exp(-(alpha* rmsd**2))
                biasgrad_i =  -2*alpha*g_rmsd*biaspot_i * rmsd

                gradient += biasgrad_i
                energy += biaspot_i


        return energy, gradient * BOHR_TO_ANGSTROMS

    @classmethod
    def calculate_energy_and_gradients_parallel(cls, chain):


        iterator2 = (
            (
                n.tdstructure.coords_bohr,
                [traj_coords*ANGSTROM_TO_BOHR for traj_coords in chain.nodes[0].reference_trajs],
                chain.parameters.mtd_strength,
                chain.parameters.mtd_alpha
            )
            for n in chain.nodes
        )

        with mp.Pool() as p2:
            ene_gradients_mtd = p2.map(cls.calc_mtd_ene_grad_from_input_tuple, iterator2)


        out_ene_gradients = []
        for i, (ene_bias, grad_bias) in enumerate(ene_gradients_mtd):
            out_ene = ene_bias
            out_grad = grad_bias
            out_ene_gradients.append((out_ene, out_grad))
        return out_ene_gradients



# -

# # Playground (Geomopt)

def chang2(structure: np.array,
          reference: np.array,
          strength=1,
          alpha=1, # basically defines the fwhm - alex chang 2023

         ):

    n_atoms = structure.shape[0]
    gradient = np.zeros((n_atoms,3))
    # for ref_chain in REFERENCE_OPT_TRAJ:
    for ref_chain in reference:
        for ref in ref_chain:

            rmsd, g_rmsd = RMSD(structure=structure, reference=ref)
            biaspot_i  = strength*n_atoms*math.exp(-(alpha* rmsd**2))
            biasgrad_i =  -2*alpha*g_rmsd*biaspot_i * rmsd
            # print(biaspot_i)
            # print(biasgrad_i)


            gradient = gradient + biasgrad_i

    return gradient


def geom_opt_mtd(node, max_steps=1000, ss=1, tol=0.001):
    nsteps = 0
    traj = []
    while nsteps < max_steps:
        traj.append(node)
        grad = node.gradient
        grad_energy = node.tdstructure.gradient_xtb()
        grad+=grad_energy

        if np.linalg.norm(grad) < tol:
            break
        new_coords = node.coords - ss*grad
        node = node.update_coords(new_coords)
        print(f"|grad|={np.linalg.norm(grad)}",end='\r')
        nsteps+=1

    if np.linalg.norm(grad) < tol:
        print(f"\nConverged in {nsteps} steps!")
    else:
        print(f"\nDid not converge in {nsteps} steps.")

    return node


# # Playground

import retropaths.helper_functions  as hf
reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")
m = MSMEP(neb_inputs=NEBInputs(v=True),chain_inputs=ChainInputs(step_size=3, delta_k=0.09), gi_inputs=GIInputs())

# rxn = "Diels-Alder-4+2"
rxn = "Claisen-Rearrangement"
start, end = m.create_endpoints_from_rxn_name(rxn,reactions)
start = start.xtb_geom_optimization()
end = end.xtb_geom_optimization()

# n_orig = NEB.read_from_disk(Path("../example_cases/alex_chang/attempt0/neb_original"))
gi = Trajectory([start, end]).run_geodesic(nimages=15)
nbi = NEBInputs(tol=0.001,v=True, vv_force_thre=0, climb=False, early_stop_force_thre=0.01, early_stop_chain_rms_thre=0.002)
cni_orig = ChainInputs(k=0.1, delta_k=0.09, step_size=3, node_class=Node3D,do_parallel=True)
chain_orig = Chain.from_traj(gi,parameters=cni_orig)
# n_orig = NEB(initial_chain=chain_orig,parameters=nbi)
# n_orig.optimize_chain()

h, out = m.find_mep_multistep(chain_orig)


def prepare_biased_neb(chain_inputs,
                   collective_variable_list,
                   initial_chain,
                   neb_inputs=NEBInputs(v=1,
                                        vv_force_thre=0, tol=0.1, max_steps=500)
                  ):
    cni = chain_inputs
    nc = cni.node_class
    assert hasattr(cni,'mtd_strength'), "You need to add the mtd_strength attribute to chain"
    assert hasattr(cni,'mtd_alpha'), "You need to add the mtd_alpha attribute to chain"

    rfs = collective_variable_list


    guess_to_use = initial_chain.nodes
    guess_chain = Chain(nodes=[nc(node.tdstructure,
                                      reference_trajs=rfs,
                                         strength=cni.mtd_strength,
                                         alpha=cni.mtd_alpha)
                                for node in guess_to_use],parameters=cni)

    nbi = neb_inputs
    n = NEB(initial_chain=guess_chain,parameters=nbi)
    return n


def prepare_clean_neb(chain_inputs,
                   initial_chain,
                   neb_inputs=NEBInputs(v=1,
                                        vv_force_thre=0, tol=0.1, max_steps=500)
                  ):
    cni = chain_inputs
    nc = cni.node_class

    guess_to_use = initial_chain.nodes
    guess_chain = Chain(nodes=[nc(node.tdstructure) for node in guess_to_use],parameters=cni)

    nbi = neb_inputs
    n = NEB(initial_chain=guess_chain,parameters=nbi)
    return n


n_orig = h.ordered_leaves[0].data

# +
nc = Node3D_MTD
cni.mtd_strength = .005
cni.mtd_alpha = 5
rfs = [
        n_orig.optimized.coordinates[1:-1]
      ]

cni = ChainInputs(k=0.1, delta_k=0.09, step_size=3, node_class=Node3D,do_parallel=True)
cni.mtd_strength = .005
cni.mtd_alpha = 5
guess_to_use = n_orig.initial_chain.nodes
guess_chain = Chain(nodes=[nc(node.tdstructure,
                                  reference_trajs=rfs,
                                     strength=cni.mtd_strength,
                                     alpha=cni.mtd_alpha)
                            for node in guess_to_use],parameters=cni)


n = NEB(initial_chain=guess_chain,parameters=nbi)
cv = [n_orig.optimized.coordinates[1:-1]]
cni = ChainInputs(k=0.00,step_size=3,node_class=Node3D_MTD)
cni.mtd_strength = .005
cni.mtd_alpha = 5
cv = [n_orig.optimized.coordinates[1:-1]]
nbi = NEBInputs(v=1, vv_force_thre=0, tol=0.001, max_steps=500)
# -

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs())

r = guess_chain[0].do_geometry_optimization()

p = guess_chain[-1].do_geometry_optimization()

mtd_guess_gi = Trajectory([r.tdstructure, p.tdstructure]).run_geodesic(nimages=15)

mtd_guess = Chain(nodes=[nc(td, reference_trajs=rfs,
                                     strength=cni.mtd_strength,
                                     alpha=cni.mtd_alpha)
                            for td in mtd_guess_gi],parameters=cni)

neb = NEB(initial_chain=mtd_guess, parameters=nbi)

neb.optimize_chain()

neb.optimized.plot_chain()

neb.optimized.to_trajectory()

# +
# rfs = [
#         n_orig.optimized.coordinates[1:-1]
#       ]
# guess_to_use = n_orig.initial_chain.nodes
# guess_chain = Chain(nodes=[Node3D_MTD(node.tdstructure,
#                               reference_trajs=rfs,
#                                  strength=cni.mtd_strength,
#                                  alpha=cni.mtd_alpha)
#                                 for node in guess_to_use],parameters=cni)


# h, out = m.find_mep_multistep(guess_chain)

# +
nbi_final = NEBInputs(v=1, vv_force_thre=0, tol=0.01, max_steps=500)

out_n = prepare_biased_neb(chain_inputs=cni,collective_variable_list=cv,initial_chain=n_orig.initial_chain,neb_inputs=nbi)
# -

out_n.optimize_chain()

cni_final = ChainInputs(k=0.01, step_size=1, node_class=Node3D,do_parallel=True)
out_n_final = prepare_clean_neb(chain_inputs=cni_final,initial_chain=out_n.optimized,neb_inputs=nbi_final)

out_n_final.optimize_chain()

out_n_final.optimized.plot_chain()

m = MSMEP(neb_inputs=nbi_final, recycle_chain=True)
h_1, out_chain_1 = m.find_mep_multistep(out_n_final.optimized)

# cv = [n_orig.optimized.coordinates[1:-1], out_n_final.optimized.coordinates[1:-1]]
cv = [n_orig.optimized.coordinates[1:-1], out_chain_1.coordinates[1:-1]]
out_n2 = prepare_biased_neb(chain_inputs=cni,collective_variable_list=cv,initial_chain=n_orig.initial_chain,neb_inputs=nbi)

out_n2.optimize_chain()

out_n2.optimized.plot_chain()

# cni_final_loose = ChainInputs(k=0.0, step_size=1, node_class=Node3D,do_parallel=True)
# out_n2_final = prepare_clean_neb(chain_inputs=cni_final_loose,initial_chain=out_n2.optimized,neb_inputs=nbi_final)
out_n2_final = prepare_clean_neb(chain_inputs=cni_final,initial_chain=out_n2.optimized,neb_inputs=nbi_final)

out_n2_final.optimize_chain()

out_n2_final.chain_trajectory[-1].plot_chain()


cv = [n_orig.optimized.coordinates[1:-1], out_n_final.optimized.coordinates[1:-1],out_n2_final.optimized.coordinates[1:-1]]
cni_3 = ChainInputs(k=0,step_size=2,node_class=Node3D_MTD)
cni_3.mtd_strength = .1
cni_3.mtd_alpha = 5
out_n3 = prepare_biased_neb(chain_inputs=cni_3,collective_variable_list=cv,initial_chain=n_orig.initial_chain,neb_inputs=nbi)

out_n3.optimize_chain()

out_n3_final = prepare_clean_neb(chain_inputs=cni_final,initial_chain=out_n3.optimized,neb_inputs=nbi_final)

out_n3_final.optimize_chain()

# +
# n_orig.write_to_disk(Path("../example_cases/alex_chang/attempt1/neb_original/"),write_history=True)

# out_n.write_to_disk(Path("../example_cases/alex_chang/attempt1/neb_bias1_raw/"),write_history=True)

# out_n_final.write_to_disk(Path("../example_cases/alex_chang/attempt1/neb_bias1/"),write_history=True)

# out_n2.write_to_disk(Path("../example_cases/alex_chang/attempt1/neb_bias2_raw/"),write_history=True)

# out_n2_final.write_to_disk(Path("../example_cases/alex_chang/attempt1/neb_bias2/"),write_history=True)
# -

out_n_final = NEB.read_from_disk(Path("../example_cases/alex_chang/attempt1/neb_bias1"))
out_n2_final = NEB.read_from_disk(Path("../example_cases/alex_chang/attempt1/neb_bias2"))
# out_n3_final = NEB.read_from_disk(Path("../example_cases/alex_chang/attempt1/neb_bias3"))

# +
s=8
fs=18
f, ax = plt.subplots(figsize=(1.61*s,s))


plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

ens_orig = n_orig.optimized.energies
ens_b1 = out_n_final.optimized.energies
ens_b2 = out_n2_final.optimized.energies

plt.plot(n_orig.optimized.integrated_path_length, (ens_orig-ens_orig[0])*627.5,'o-', label='original neb')
plt.plot(out_n_final.optimized.integrated_path_length, (ens_b1-ens_orig[0])*627.5,'o-', label='neb after bias1')
plt.plot(out_n2_final.optimized.integrated_path_length, (ens_b2-ens_orig[0])*627.5,'o-', label='neb after bias2')
# plt.plot(out_n3_final.optimized.integrated_path_length, out_n3_final.optimized.energies,'o-', label='neb after bias3')



plt.ylabel("Energy (Hartrees)",fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.legend(fontsize=fs)
# -


# # MSMESP

m = MSMEP(neb_inputs=nbi_final, recycle_chain=True)
m_orig = m.find_mep_multistep(n_orig.optimized)

m_1 = m.find_mep_multistep(out_n_final.optimized)

m_2 = m.find_mep_multistep(out_n2_final.optimized)

m_orig[1].to_trajectory()

m_1[1].to_trajectory()

m_2[1].to_trajectory()

# # TS search

from TS.TS_PRFO import TS_PRFO

# ts = TS_PRFO(initial_node=n_final.optimized[10],max_step_size=1, max_nsteps=2000, grad_thre=1e-4)
ts = TS_PRFO(initial_node=n_orig.optimized[7],max_step_size=1, max_nsteps=2000, grad_thre=1e-4)

ts.ts.tdstructure.to_xyz("ts1.xyz")

ts2.ts.tdstructure.to_xyz("ts2.xyz")

ts2 = TS_PRFO(initial_node=out_n3_final.optimized[7],max_step_size=1, max_nsteps=200)
ts2.ts.tdstructure


