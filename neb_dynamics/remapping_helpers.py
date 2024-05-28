import numpy as np
from retropaths.abinitio.trajectory import Trajectory
from retropaths.molecules.isomorphism_tools import SubGraphMatcherApplyPermute

from neb_dynamics.Chain import Chain
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Node3D import Node3D


def get_atom_xyz(struct, atom_ind):
    atom_r = struct.molecule_obmol.GetAtom(atom_ind + 1)
    return np.array((atom_r.x(), atom_r.y(), atom_r.z()))


def get_neighs(struct, ind):

    neighs = []  # indices of the single bonds
    for neigh in struct.molecule_rp[ind]:
        if struct.molecule_rp[ind][neigh]["bond_order"] == "double":
            continue
        else:
            neighs.append(neigh)

    return neighs


def create_isomorphic_structure(struct, iso):

    orig_coords = struct.coords
    new_coords = orig_coords.copy()

    for orig_ind, remap_ind in iso.items():
        new_coords[orig_ind] = orig_coords[remap_ind]

    new_struct = struct.copy()
    new_struct = new_struct.update_coords(new_coords)
    return new_struct


def get_all_product_isomorphisms(end_struct, timeout=100):

    mol = end_struct.molecule_rp
    sgmap = SubGraphMatcherApplyPermute(mol, timeout_seconds=timeout)
    isoms = sgmap.get_isomorphisms(mol)

    if len(isoms) > 100:
        print(
            f"There are {len(isoms)} candidate structures. Too many. Returning only one struct"
        )
        return np.array([end_struct])

    new_structs = []
    for isom in isoms:
        new_structs.append(create_isomorphic_structure(struct=end_struct, iso=isom))

    return np.array(new_structs)


def get_gi_info(new_structs, start_struct):
    max_gi_vals = []
    # works = []
    trajs = []
    for i, end_point in enumerate(new_structs):

        traj = Trajectory(
            [start_struct, end_point]
        )  # WARNING <--- this NEEDS to be changed if dealing with charged species
        traj = traj.run_geodesic(nimages=10, friction=0.001)
        trajs.append(traj)

        ens = traj.energies_xtb()
        if None not in ens:
            max_gi_vals.append(max(ens))
        else:
            max_gi_vals.append(np.inf)

    # return np.array(max_gi_vals), np.array(works), np.array(trajs)
    return np.array(max_gi_vals), [], np.array(trajs)


def get_correct_product_structure(new_structs, gi_info, kcal_window):
    max_gi_vals, works, trajs = gi_info
    # print(f"{max_gi_vals=}")

    sorted_inds = np.argsort(max_gi_vals)  # indices that would sort array
    # print(f"{sorted_inds=}")
    sorted_arr = max_gi_vals[sorted_inds]
    sorted_arr -= sorted_arr[0]
    ints_to_do = sorted_inds[sorted_arr <= kcal_window]
    # print(ints_to_do)
    return new_structs[ints_to_do], trajs[ints_to_do]


def decide_with_neb(list_of_trajs: list):
    neb_results = []
    for i, traj in enumerate(list_of_trajs):
        print(f"Doing mapping {i}...")

        c = Chain.from_traj(traj, k=0.1, delta_k=0, step_size=2, node_class=Node3D)

        tol = 0.01
        n = NEB(
            initial_chain=c,
            en_thre=tol / 450,
            rms_grad_thre=tol * (2 / 3),
            grad_thre=tol,
            vv_force_thre=0,
            v=False,
        )
        try:
            n.optimize_chain()
            neb_results.append(n)
        except NoneConvergedException:
            print(f"warning, NEB {i} did not converge...")
            neb_results.append(n)
    else:
        neb_results.append(None)

    return neb_results


def create_correct_interpolation(start_struct, end_struct, kcal_window=10):

    new_structs = get_all_product_isomorphisms(end_struct)
    gi_info = get_gi_info(new_structs=new_structs, start_struct=start_struct)
    correct_end_struct, correct_gi_traj = get_correct_product_structure(
        new_structs=new_structs, gi_info=gi_info, kcal_window=kcal_window
    )
    correct_gi_traj = [Trajectory(t, charge=0, spinmult=1) for t in correct_gi_traj]
    # list_of_neb_trajs = decide_with_neb(correct_gi_traj)
    return correct_gi_traj
    # return list_of_neb_trajs


def create_correct_product(start_struct, end_struct, kcal_window=10):
    new_structs = get_all_product_isomorphisms(end_struct)
    if len(new_structs) == 1:
        return [new_structs[0]]
    gi_info = get_gi_info(new_structs=new_structs, start_struct=start_struct)
    correct_end_struct, _ = get_correct_product_structure(
        new_structs=new_structs, gi_info=gi_info, kcal_window=kcal_window
    )
    return correct_end_struct
