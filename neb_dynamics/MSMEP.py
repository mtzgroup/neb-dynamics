from dataclasses import dataclass

import numpy as np
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from retropaths.helper_functions import pairwise
from retropaths.molecules.molecule import Molecule
from retropaths.reactions.changes import Changes3D, Changes3DList
from scipy.signal import argrelextrema

from neb_dynamics.NEB import NEB, Chain, Node3D, NoneConvergedException


@dataclass
class MSMEP:
    k: float =0.1
    delta_k: float = 0.0 
    step_size: float = 1.0

    tol: float = 0.03
    max_steps: int =500
    en_thre: float = None
    rms_grad_thre: float = None 
    grad_thre: float = None
    v: bool = False

    def find_mep_multistep(self, inp, do_alignment):
        start, end = inp
        chain = self.get_neb_chain(start=start, end=end, do_alignment=do_alignment)
        if not chain: return None
        if self.is_elem_step(chain):
            return chain
        else:
            pairs_of_minima = self.make_pairs_of_minima(chain)
            elem_steps = []
            for i, pair in enumerate(pairs_of_minima):
                print(f"On pair {i+1} of {len(pairs_of_minima)}...")
                if i <= len(pairs_of_minima)-2:
                    next_pair = pairs_of_minima[i+1]
                    do_alignment = pair[1].molecule_rp != next_pair[0].molecule_rp # i.e. if minima found is not just a conformer rearrangment
                    print(f"\t{do_alignment=}")
                    
                    elem_steps.append(self.find_mep_multistep(pair, do_alignment=do_alignment))
                    
                else: # i.e. the final pair
                    elem_steps.append(self.find_mep_multistep(pair, do_alignment=False))
                    
            
            stitched_elem_steps = self.stitch_elem_steps(elem_steps)
            return stitched_elem_steps
    

    def get_neb_chain(self, start,end, do_alignment):
    
        if do_alignment:
            start, end = self._align_endpoints(start, end)
        
        traj = Trajectory([start, end])
        gi = traj.run_geodesic(nimages=15)
        
        if self.actual_reaction_happened_based_on_gi(gi):

            chain = Chain.from_traj(gi,k=0.1,delta_k=0, step_size=1,node_class=Node3D)


            max_steps = self.max_steps
            en_thre = self.en_thre if self.en_thre else self.tol / 450 
            rms_grad_thre = self.rms_grad_thre if self.rms_grad_thre else self.tol*(2/3)
            grad_thre = self.grad_thre if self.grad_thre else self.tol

            n = NEB(initial_chain=chain,max_steps=max_steps,en_thre=en_thre, rms_grad_thre=rms_grad_thre, grad_thre=grad_thre, v=self.v)
            try:
                print("Running NEB calculation...")
                n.optimize_chain()
                out_chain = n.optimized

            except NoneConvergedException:
                print("Warning! A chain did not converge. Returning an unoptimized chain...")
                out_chain = n.chain_trajectory[-1]

            return out_chain   
        else:
            print("Endpoints are identical. Returning nothing")
            # return Chain(nodes=[Node3D(start)], k=0.1, delta_k=0,step_size=1,node_class=Node3D)
            return None


    def is_elem_step(self, chain):
        if len(chain) > 1:
            ind_minima = self._get_ind_minima(chain)

            return len(ind_minima) == 0
        else: 
            return True


    def _get_ind_minima(self, chain):
        ind_minima = argrelextrema(chain.energies, np.less, order=1)[0]
        return ind_minima

    def make_pairs_of_minima(self, chain):
        all_inds = [0]
        ind_minima = self._get_ind_minima(chain)
        all_inds.extend(ind_minima)
        all_inds.append(len(chain)-1)
        
        
        pairs_inds = list(pairwise(all_inds))
        
        structs = []
        for start_ind, end_ind in pairs_inds:
            structs.append((chain[start_ind].tdstructure.xtb_geom_optimization(), chain[end_ind].tdstructure.xtb_geom_optimization()))
        
        return structs

    def stitch_elem_steps(self, list_of_chains):
        list_of_tds = []
        for chain in list_of_chains:
            if chain: # list of chains will contain None values for whenever an interpolation between identical structures was given
                [list_of_tds.append(n.tdstructure) for n in chain]
        t = Trajectory(list_of_tds)
        return Chain.from_traj(t,k=0.1,delta_k=0,step_size=1,node_class=Node3D)


    def _align_endpoints(self, start: Molecule, end: Molecule):
        bc = start.molecule_rp.get_bond_changes(end.molecule_rp)
        c3d_list = self.from_bonds_changes(bc)

        if len(c3d_list.deleted+c3d_list.forming+c3d_list.charges)==0: return start, end
        start = start.pseudoalign(c3d_list)
        start.mm_optimization('uff')
        start = start.xtb_geom_optimization()
        
        end_mod = start.copy()
        end_mod.add_bonds(c3d_list.forming)
        end_mod.delete_bonds(c3d_list.deleted)
        end_mod.mm_optimization('uff')
        end_mod = end_mod.xtb_geom_optimization()
        return start, end_mod
    

    def actual_reaction_happened_based_on_gi(self, traj: Trajectory):
        ens = traj.energies_xtb()
        delta_e = (max(ens) - min(ens)) 
        print(f"{delta_e=}")
        if  delta_e <= 1 : # if the difference between the highest energy point in Geodesic traj and the lowest energy point is less than 1kcal/mol
            return False
        else:
            return True

    def from_bonds_changes(self, bc):
        forming_list = []
        deleted_list = []
        
        for s,e in bc.forming:
            forming_list.append(Changes3D(start=s,end=e,bond_order=1))
        
        for s,e in bc.breaking:
            deleted_list.append(Changes3D(start=s, end=e, bond_order=-1))
        return Changes3DList(forming=forming_list,deleted=deleted_list,charges=[])
    