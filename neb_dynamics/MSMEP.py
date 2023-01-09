from dataclasses import dataclass

import numpy as np
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from retropaths.helper_functions import pairwise
from retropaths.reactions.changes import Changes3D, Changes3DList
from scipy.signal import argrelextrema

from neb_dynamics.Chain import Chain
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Node import Node
from neb_dynamics.remapping_helpers import create_correct_product


@dataclass
class MSMEP:

    # electronic structure params
    charge: int = 0
    spinmult: int = 1

    # chain params
    k: float = 0.1
    delta_k: float = 0.0
    step_size: float = 1.0

    # neb params
    tol: float = 0.01
    max_steps: int = 500
    en_thre: float = None
    rms_grad_thre: float = None
    grad_thre: float = None
    v: bool = False
    node_class: Node = Node3D

    # geodesic interpolation params
    nimages: int = 15
    friction: float = 0.1
    nudge: float = 0.001

    # msmep params
    optimize_hydrogen: bool = False

    def create_endpoints_from_rxn_name(self, rxn_name, reactions_object):
        rxn = reactions_object[rxn_name]
        root = TDStructure.from_rxn_name(rxn_name, reactions_object)
        c3d_list = root.get_changes_in_3d(rxn)

        root = root.pseudoalign(c3d_list)
        root = root.xtb_geom_optimization()

        target = root.copy()
        target.apply_changed3d_list(c3d_list)
        target.mm_optimization("gaff", steps=5000)
        target.mm_optimization("uff", steps=5000)
        target = target.xtb_geom_optimization()

        return root, target

    def optimize_hydrogen_label(self, chain):
        start = chain[0].tdstructure
        end = chain[-1].tdstructure
        correct_endpoint = create_correct_product(start, end, kcal_window=10)[
            0
        ]  # currently only selecting the best, need to fix so that you do some more sampling
        if not np.all(correct_endpoint.coords != end.coords):
            print("Making chain with optimal hydrogen")
            _, new_chain = self.get_neb_chain(
                start=start, end=correct_endpoint, do_alignment=False
            )
            return new_chain
        else:
            return chain

    def find_mep_multistep(self, input_chain, do_alignment):
        n, chain = self.get_neb_chain(
            input_chain=input_chain, do_alignment=do_alignment
        )
        if not chain:
            return None, None
        if self.is_elem_step(chain):
            if self.optimize_hydrogen:
                chain_opt = self.optimize_hydrogen_label(chain)
                return None, chain_opt
            else:
                return None, chain
        else:
            sequence_of_chains = self.make_sequence_of_chains(chain)
            elem_steps = []
            for i, chain_frag in enumerate(sequence_of_chains):
                print(f"On chain {i+1} of {len(sequence_of_chains)}...")
                if i <= len(sequence_of_chains) - 2:
                    next_chain_frag = sequence_of_chains[i + 1]
                    do_alignment = (
                        chain_frag[-1].tdstructure.molecule_rp != next_chain_frag[0].tdstructure.molecule_rp
                    )  # i.e. if minima found is not just a conformer rearrangment
                    print(f"\t{do_alignment=}")
                    neb_obj, chain = self.find_mep_multistep(
                        chain_frag, do_alignment=False
                    )
                    elem_steps.append(chain)

                else:  # i.e. the final pair
                    neb_obj, chain = self.find_mep_multistep(chain_frag, do_alignment=False)
                    elem_steps.append(chain)

            stitched_elem_steps = self.stitch_elem_steps(elem_steps)
            return (
                None,
                stitched_elem_steps,
            )  # the first 'None' will hold the DataTree that holds all NEB objects

    def get_neb_chain(self, input_chain, do_alignment):

        if do_alignment:
            start, end = input_chain[0].tdstructure, input_chain[-1].tdstructure
            start, end = self._align_endpoints(start, end)
            traj = Trajectory([start, end], charge=self.charge, spinmult=self.spinmult)
        else:
            traj = Trajectory([node.tdstructure for node in input_chain], charge=self.charge, spinmult=self.spinmult)
 
        gi = traj.run_geodesic(
            nimages=self.nimages, friction=self.friction, nudge=self.nudge
        )

        if input_chain[0].tdstructure.molecule_rp == input_chain[-1].tdstructure.molecule_rp:
            print("Endpoints are identical. Returning nothing")
            return None, None
        else:
            chain = Chain.from_traj(
                gi,
                k=self.k,
                delta_k=self.delta_k,
                step_size=self.step_size,
                node_class=self.node_class,
            )

            max_steps = self.max_steps
            en_thre = self.en_thre if self.en_thre else self.tol / 450
            rms_grad_thre = (
                self.rms_grad_thre if self.rms_grad_thre else self.tol * (2 / 3)
            )
            grad_thre = self.grad_thre if self.grad_thre else self.tol
            
            n = NEB(
                initial_chain=chain,
                max_steps=max_steps,
                en_thre=en_thre,
                rms_grad_thre=rms_grad_thre,
                grad_thre=grad_thre,
                v=self.v,
            )

            try:
                print("Running NEB calculation...")
                n.optimize_chain()
                out_chain = n.optimized

            except NoneConvergedException:
                print(
                    "\nWarning! A chain did not converge. Returning an unoptimized chain..."
                )
                out_chain = n.chain_trajectory[-1]

            return n, out_chain

    def is_elem_step(self, chain):
        if len(chain) > 1:
            ind_minima = self._get_ind_minima(chain)

            return len(ind_minima) == 0
        else:
            return True

    def _get_ind_minima(self, chain):
        ind_minima = argrelextrema(chain.energies, np.less, order=1)[0]
        return ind_minima

    def _make_chain_frag(self, chain, pair_of_inds):
        start, end = pair_of_inds
        chain_frag = chain.copy()
        chain_frag.nodes = chain[start : end + 1]
        opt_start = chain[start].tdstructure.xtb_geom_optimization()
        opt_end = chain[end].tdstructure.xtb_geom_optimization()

        chain_frag.insert(0, Node3D(opt_start))
        chain_frag.insert(-1, Node3D(opt_end))

        return chain_frag

    def make_sequence_of_chains(self, chain):
        all_inds = [0]
        ind_minima = self._get_ind_minima(chain)
        all_inds.extend(ind_minima)
        all_inds.append(len(chain) - 1)

        pairs_inds = list(pairwise(all_inds))

        chains = []
        for ind_pair in pairs_inds:
            chains.append(self._make_chain_frag(chain, ind_pair))

        return chains

    def stitch_elem_steps(self, list_of_chains):
        list_of_tds = []
        for chain in list_of_chains:
            if (
                chain
            ):  # list of chains will contain None values for whenever an interpolation between identical structures was given
                [list_of_tds.append(n.tdstructure) for n in chain]
        t = Trajectory(list_of_tds, charge=self.charge, spinmult=self.spinmult)
        return Chain.from_traj(
            t,
            k=self.k,
            delta_k=self.delta_k,
            step_size=self.step_size,
            node_class=self.node_class,
        )

    def _align_endpoints(self, start: TDStructure, end: TDStructure):
        """
        this function gets the bond changes between the start and end structure,
        then applies these changes to the start structure in order to generate a
        modified end structure. the idea is to generate the closest structure to the
        starting structure.

        this function can (and often does!) lead to a change in conformers of the endpoints,
        so it should not be used if you want to conserve the initial and final input conformers
        """
        bc = start.molecule_rp.get_bond_changes(end.molecule_rp)
        c3d_list = self.from_bonds_changes(bc)

        if len(c3d_list.deleted + c3d_list.forming + c3d_list.charges) == 0:
            return start, end
        start = start.pseudoalign(c3d_list)
        start.mm_optimization("uff", steps=2000)
        start.mm_optimization("gaff", steps=2000)
        start.mm_optimization("mmff94", steps=2000)
        start = start.xtb_geom_optimization()

        end_mod = start.copy()
        end_mod.add_bonds(c3d_list.forming)
        end_mod.delete_bonds(c3d_list.deleted)
        end_mod.mm_optimization("uff", steps=2000)
        end_mod.mm_optimization("gaff", steps=2000)
        end_mod.mm_optimization("mmff94", steps=2000)
        end_mod = end_mod.xtb_geom_optimization()
        return start, end_mod

    def actual_reaction_happened_based_on_gi(self, traj: Trajectory):
        ens = traj.energies_xtb()
        delta_e = max(ens) - min(ens)
        print(f"{delta_e=}")
        if (
            delta_e <= 1
        ):  # if the difference between the highest energy point in Geodesic traj and the lowest energy point is less than 1kcal/mol
            return False
        else:
            return True

    def from_bonds_changes(self, bc):
        forming_list = []
        deleted_list = []

        for s, e in bc.forming:
            forming_list.append(Changes3D(start=s, end=e, bond_order=1))

        for s, e in bc.breaking:
            deleted_list.append(Changes3D(start=s, end=e, bond_order=-1))
        return Changes3DList(forming=forming_list, deleted=deleted_list, charges=[])
