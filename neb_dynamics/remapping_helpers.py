import numpy as np
from neb_dynamics.NEB import Chain, Node3D
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.geodesic_input import GeodesicInput
from retropaths.molecules.isomorphism_tools import SubGraphMatcherApplyPermute
from neb_dynamics.trajectory import Trajectory

def get_atom_xyz(struct, atom_ind):
    atom_r = struct.molecule_obmol.GetAtom(atom_ind+1)
    return np.array((atom_r.x(), atom_r.y(), atom_r.z()))


def get_neighs(struct, ind):

    neighs = [] # indices of the single bonds 
    for neigh in struct.molecule_rp[ind]:
        if struct.molecule_rp[ind][neigh]['bond_order']=="double": continue
        else: neighs.append(neigh)
        
    return neighs


def create_isomorphic_structure(struct, iso):

    orig_coords = struct.coords
    new_coords = orig_coords.copy()

    for orig_ind,remap_ind in iso.items():
        new_coords[orig_ind] = orig_coords[remap_ind]


    new_struct = struct.copy()
    new_struct.update_coords(new_coords*ANGSTROM_TO_BOHR) # <--- warning, this was left in because TDstructures from RP were still in angstroms...
    return new_struct

def get_all_product_isomorphisms(end_struct):
    

    mol = end_struct.molecule_rp
    sgmap = SubGraphMatcherApplyPermute(mol)
    isoms = sgmap.get_isomorphisms(mol)


    new_structs = []
    for isom in isoms:
        new_structs.append(create_isomorphic_structure(struct=end_struct, iso=isom))
        
    return np.array(new_structs)


def get_gi_info(new_structs, start_struct):
    max_gi_vals = []
    works = []
    trajs = []
    # out_dir = Path("./GI_filter")
    for i, end_point in enumerate(new_structs):
        gi = GeodesicInput.from_endpoints(initial=start_struct, final=end_point)
        traj = gi.run(nimages=15, friction=0.01, nudge=0.001)
        # traj.write_trajectory(out_dir/f"traj_{i}.xyz")
        trajs.append(traj)

        chain = Chain.from_traj(traj, k=99, delta_k=99, step_size=99, node_class=Node3D)
        chain_energies_hartree = chain.energies
        chain_energies_hartree -= chain_energies_hartree[0]
        chain_energies_kcal = chain_energies_hartree*627.5
        max_gi_vals.append(max(chain_energies_kcal))
        works.append(chain.work)
    return np.array(max_gi_vals), np.array(works), np.array(trajs)

def get_correct_product_structure(new_structs, gi_info, kcal_window=10):
    max_gi_vals, works, trajs = gi_info
    
    sorted_inds = np.argsort(max_gi_vals) # indices that would sort array
    sorted_arr = max_gi_vals[sorted_inds]
    sorted_arr-= sorted_arr[0]
    ints_to_do = sorted_inds[sorted_arr <= kcal_window]
    print(ints_to_do)
    return new_structs[ints_to_do], trajs[ints_to_do]
    # return new_structs[np.argmin(max_gi_vals)], trajs[np.argmin(max_gi_vals)]
    # return new_structs[np.argmin(works)], trajs[np.argmin(works)]

def create_correct_interpolation(start_ind, end_ind, root_conformers, transformed_conformers):
    start_struct = root_conformers[start_ind]
    start_struct_coords = start_struct.coords
    start_struct.update_coords(start_struct_coords*ANGSTROM_TO_BOHR)
    
    end_struct = transformed_conformers[end_ind]
    
    new_structs = get_all_product_isomorphisms(end_struct)
    gi_info = get_gi_info(new_structs=new_structs, start_struct=start_struct)
    correct_end_struct, correct_gi_traj = get_correct_product_structure(new_structs=new_structs, gi_info=gi_info)
    correct_gi_traj = np.array([Trajectory(t, tot_charge=0, tot_spinmult=1) for t in correct_gi_traj])

    return correct_gi_traj
