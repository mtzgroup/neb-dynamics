# +
from pathlib import Path

from retropaths.abinitio.inputs import Inputs
from retropaths.abinitio.tdstructure import TDStructure
# from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.rootstructure import RootStructure
from retropaths.abinitio.sampler import Sampler

from neb_dynamics.NEB import Chain, Node3D 
from neb_dynamics.trajectory import Trajectory

import numpy as np

import matplotlib.pyplot as plt

# +
rxn_name = 'Claisen-Rearrangement'
reaction_path = Path("/Users/janestrada/Retropaths/retropaths/data/reactions.p")

out_path=Path("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement")
inp = Inputs(rxn_name=rxn_name, reaction_file=reaction_path)
root_structure = RootStructure(
    root=TDStructure.from_rxn_name(rxn_name, data_folder=reaction_path.parent),
    master_path=out_path,
    rxn_args=inp,
    trajectory=Trajectory([]),
)
# -

root_conformers, root_energies = root_structure.crest(
    tdstruct=root_structure.root, id="root"
).conf_and_energy
transformed_conformers, transformed_energies = root_structure.crest(
    tdstruct=root_structure.transformed,
    id="transformed"
).conf_and_energy

# +
sampler = Sampler(mode="distance")
sub_root_conformers, _  = sampler.run(
conformers_to_subsample=root_conformers,
bonds_between_frags=root_structure._get_bonds_between_fragments(),
energies=root_energies,
cutoff=7
) 
sub_trans_conformers, _ = sampler.run(
conformers_to_subsample=transformed_conformers,
bonds_between_frags=root_structure._get_bonds_between_fragments(),
energies=transformed_energies,
cutoff=7
)

subselected_conf_pairs = sampler._get_conf_pairs(
start_conformers=sub_root_conformers,
end_conformers=sub_trans_conformers
)

# -

root_conformers = sub_root_conformers
transformed_conformers = sub_trans_conformers

# # Let's try to figure this shit out

# +
r_ind = 0
p_ind = 3


r = root_conformers[r_ind]
p = transformed_conformers[p_ind]

# +
traj = Trajectory.from_xyz(Path(f"/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement/traj_{r_ind}-{p_ind}_neb.xyz"))
chain = Chain.from_traj(traj, k=1,delta_k=1,step_size=0,node_class=Node3D)

plt.plot(chain.energies, 'o--')


# +
def get_atom_xyz(struct, atom_ind):
    
    atom_r = struct.molecule_obmol.GetAtom(atom_ind+1)
    return np.array((atom_r.x(), atom_r.y(), atom_r.z()))

get_atom_xyz(r, 5)


# +
def get_neighs(struct, ind):

    neighs = [] # indices of the single bonds 
    for neigh in r.molecule_rp[ind]:
        if r.molecule_rp[ind][neigh]['bond_order']=="double": continue
        else: neighs.append(neigh)
        
    return neighs

get_neighs(r, 5)
# -

get_neighs(r, 0)

get_neighs(r, 5)

# +
r_0 = get_atom_xyz(r, 0)
r_6 = get_atom_xyz(r, 6)
r_7 = get_atom_xyz(r, 7)


r_5 = get_atom_xyz(r,5)
r_12 = get_atom_xyz(r,12)
r_13 = get_atom_xyz(r, 13)

# +
p_0 = get_atom_xyz(p, 0)
p_6 = get_atom_xyz(p, 6)
p_7 = get_atom_xyz(p, 7)


p_5 = get_atom_xyz(p,5)
p_12 = get_atom_xyz(p,12)
p_13 = get_atom_xyz(p, 13)

# +
side_1 = np.linalg.norm(r_12 - r_6)
cross_1 = np.linalg.norm(r_13 - r_6)

side_2 = np.linalg.norm(r_13 - r_7)
cross_2 = np.linalg.norm(r_12 - r_7)

print((side_1, side_2))
print((cross_1, cross_2))

# +
side_1_p = np.linalg.norm(p_12 - p_6)
cross_1_p = np.linalg.norm(p_13 - p_6)

side_2_p = np.linalg.norm(p_13 - p_7)
cross_2_p = np.linalg.norm(p_12 - p_7)

print((side_1_p, side_2_p))
print((cross_1_p, cross_2_p))
# -

# # Checking if two lines cross

# +
p1 = np.array([0.0,0.0])
p2 = np.array([1.0, 0.0])

p3 = np.array([1,0.0])
p4 = np.array([1,1.0])


# +
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def get_intersect(a1, a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1


# -

def _check_point_in_both_lines(point,line1, line2):
    poi_x, poi_y = point

    
    p1, p2 = line1
    p3, p4 = line2
    
    
    min_x1 = min(p1[0], p2[0])
    min_y1 = min(p1[1], p2[1])
    
    max_x1 = max(p1[0], p2[0])
    max_y1 = max(p1[1], p2[1])
    
    min_x2 = min(p3[0], p4[0])
    min_y2 = min(p3[1], p4[1])
    
    max_x2 = max(p3[0], p4[0])
    max_y2 = max(p3[1], p4[1])
    
    in_line1 =  min_x1 <= poi_x <= max_x1 and min_y1 <= poi_y <= max_y1
    # print(in_line1)
    # print(f"\t{p1[0]}//{p2[0]}")
    in_line2 =  min_x2 <= poi_x <= max_x2 and min_y2 <= poi_y <= max_y2
    # print(in_line2)


    return in_line1 and in_line2
poi = get_intersect(p1,p2,p3,p4)
_check_point_in_both_lines(poi, (p1,p2), (p3,p4))


def check_if_lines_intersect(p1_var, p2_var, p3_var, p4_var):
    poi = get_intersect(p1_var,p2_var,p3_var,p4_var) # point of intersection (POI)
    # print(poi)
    return _check_point_in_both_lines(poi, line1=(p1_var,p2_var), line2=(p3_var,p4_var))


def get_points(struct, indices):
    a1, a2, b1, b2 = indices
    p_a1 = get_atom_xyz(struct, a1)
    p_a2 = get_atom_xyz(struct, a2)

    p_b1 = get_atom_xyz(struct, b1)
    p_b2 = get_atom_xyz(struct, b2)
    
    return p_a1, p_a2, p_b1, p_b2


def get_neigh_inds(struct, ref_atoms_inds):
    
    ref_atom_1, ref_atom_2 = ref_atoms_inds
    
    a1, a2 = get_neighs(struct, ref_atom_1)

    b1, b2 = get_neighs(struct, ref_atom_2)
    
    return a1, a2, b1, b2


def flip_coordinates(struct, a1, a2):
    struct_copy = struct.copy()
    
    original_a1 = get_atom_xyz(struct_copy, a1)
    original_a2 = get_atom_xyz(struct_copy, a2)
    all_coords = struct_copy.coords
    new_coords = all_coords.copy()
    new_coords[a1] = original_a2
    new_coords[a2] = original_a1

    struct_copy.update_coords(new_coords)
    return struct_copy


# +
def check_if_structure_needs_to_be_flipped(react_struc, product_struc, rs):
    a1, a2, b1, b2 = get_neigh_inds(product_struc, rs._get_bonds_between_fragments()['single'][0])
    p_a1, p_a2, p_b1, p_b2 = get_points(product_struc, [a1,a2,b1,b2])

    # print((a1, a2, b1, b2))

    # make sure that the correct mapping is a1 --> b1 and a2 --> b2
    if check_if_lines_intersect(p1_var=p_a1[[0,2]], p2_var=p_b1[[0,2]], p3_var=p_a2[[0,2]], p4_var=p_b2[[0,2]]):
        b1_copy = b1
        b2_copy = b2

        b1 = b2_copy
        b2 = b1_copy


    # the mappings have been corrected at this point with respect to the final structure
    a1, a2, b1, b2
    
    # now, check if the initial structure needs to be flipped

    p_a1, p_a2, p_b1, p_b2 = get_points(react_struc, [a1,a2,b1,b2])
    if check_if_lines_intersect(p1_var=p_a1[[0,2]], p2_var=p_b1[[0,2]], p3_var=p_a2[[0,2]], p4_var=p_b2[[0,2]]):
        return True
        print("react_struct_had_to_be_flipped")
        react_struct = flip_coordinates(react_struc, a1, a2)

        p_a1, p_a2, p_b1, p_b2 = get_points(react_struc, [a1,a2,b1,b2])
        assert check_if_lines_intersect(p1_var=p_a1[[0,2]], p2_var=p_b1[[0,2]], p3_var=p_a2[[0,2]], p4_var=p_b2[[0,2]])
    else: 
        return False
    

        
        
check_if_structure_needs_to_be_flipped(react_struc=root_conformers[0], product_struc=transformed_conformers[1], rs=root_structure)

# +
subselected_conf_pairs = sampler._get_conf_pairs(
    start_conformers=sub_root_conformers,
    end_conformers=sub_trans_conformers
)

count=0
pairs_to_fix = []
print(f"I now have: {len(subselected_conf_pairs)} pairs")
for conf_pair in subselected_conf_pairs:
    # print(conf_pair)
    start_conf, end_conf = conf_pair
    if check_if_structure_needs_to_be_flipped(react_struc=root_conformers[start_conf], product_struc=transformed_conformers[end_conf], rs=root_structure):
        count+=1
        pairs_to_fix.append((start_conf, end_conf))
# -

count

pairs_to_fix

f = open("./pairs_to_fix.txt", "w+")
for r, p in pairs_to_fix:
    f.write(f"{r} {p}\n")
f.close()

# # Use GI to filter
#
# Plan: 
# 1. Get all isomorphisms of product conformer
# 2. Interpolate between each, getting max barrier from GI
# 3. Select isomorphism with lowest barrier

from neb_dynamics.geodesic_input import GeodesicInput
from neb_dynamics.NEB import Chain, Node3D, NEB
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS

# +
root_conformers, root_energies = root_structure.crest(
    tdstruct=root_structure.root, id="root"
).conf_and_energy
transformed_conformers, transformed_energies = root_structure.crest(
    tdstruct=root_structure.transformed,
    id="transformed"
).conf_and_energy

sampler = Sampler(mode="distance")
sub_root_conformers, _  = sampler.run(
conformers_to_subsample=root_conformers,
bonds_between_frags=root_structure._get_bonds_between_fragments(),
energies=root_energies,
cutoff=7
) 
sub_trans_conformers, _ = sampler.run(
conformers_to_subsample=transformed_conformers,
bonds_between_frags=root_structure._get_bonds_between_fragments(),
energies=transformed_energies,
cutoff=7
)

root_conformers = sub_root_conformers
transformed_conformers = sub_trans_conformers


# +
# start_ind = 15
# end_ind = 20

start_ind = 28
end_ind = 3

# +
from retropaths.molecules.molecule import Molecule
from retropaths.molecules.isomorphism_tools import SubGraphMatcherApplyPermute

mol = root_conformers[start_ind].molecule_rp
sgmap = SubGraphMatcherApplyPermute(mol)
isoms = sgmap.get_isomorphisms(mol)

# -

def create_isomorphic_structure(struct, iso):

    orig_coords = struct.coords
    new_coords = orig_coords.copy()

    for orig_ind,remap_ind in iso.items():
        new_coords[orig_ind] = orig_coords[remap_ind]


    new_struct = struct.copy()
    new_struct.update_coords(new_coords*ANGSTROM_TO_BOHR)
    return new_struct


# +
start_struct = root_conformers[start_ind]
start_struct_coords = start_struct.coords
start_struct.update_coords(start_struct_coords*ANGSTROM_TO_BOHR)


end_struct = transformed_conformers[end_ind]
new_structs = []
for isom in isoms:
    new_structs.append(create_isomorphic_structure(struct=end_struct, iso=isom))
# -

max_gi_vals = []
works = []
trajs = []
out_dir = Path("./GI_filter")
for i, end_point in enumerate(new_structs):
    gi = GeodesicInput.from_endpoints(initial=start_struct, final=end_point)
    # traj = gi.run(nimages=15, friction=0.01, nudge=0.001)
    traj = gi.run(nimages=40, friction=0.01, nudge=0.001)
    # traj.write_trajectory(out_dir/f"traj_{i}.xyz")
    trajs.append(traj)
    
    chain = Chain.from_traj(traj, k=99, delta_k=99, step_size=99, node_class=Node3D)
    max_gi_vals.append(max(chain.energies))
    works.append(chain.work)

np.argmin(max_gi_vals)

np.argmin(works)

# ## Trial by fire. NEB on each of the GIs

# gis = []
# for fp in out_dir.iterdir():
#     gis.append(Trajectory.from_xyz(Path(fp)))


# gis_chains = [Chain.from_traj(gi, k=1, delta_k=0, node_class=Node3D, step_size=9) for gi in gis]
gis_chains = [Chain.from_traj(t, k=.1, delta_k=0, node_class=Node3D, step_size=1) for t in trajs]

for i, gi in enumerate(gis_chains):
    print(i, gi.work)

# +
f, ax = plt.subplots()
for i, c in enumerate(gis_chains):
    plt.plot(c.energies, 'o--', label=f'gi_{i}')

plt.legend()
# -

nebs = []
for gi in gis_chains:
    n = NEB(initial_chain=gi, grad_thre_per_atom=0.0016, vv_force_thre=0, climb=False)
    try:
        n.optimize_chain()
        nebs.append(n)
    except:
        nebs.append(None)

for i, n in enumerate(nebs):
    try:
        c = n.optimized
        print(i, c.work)
        # n.write_to_disk(Path(f'./neb_{i}.xyz'))
        
        # if i == 7 or i==5 : plt.plot(c.energies, 'o-', label=f'neb_{i}', linewidth=4)
        # else: plt.plot(c.energies, '--', label=f'neb_{i}')
    except:
        continue


# +
f, ax = plt.subplots()
for i, n in enumerate(nebs):
    try:
        c = n.optimized
        plt.plot(c.energies, 'o-', label=f'neb_{i}')
        # if i == 7 or i == 4: plt.plot(c.energies, 'o-', label=f'neb_{i}', linewidth=4)
        # else: plt.plot(c.energies, '--', label=f'neb_{i}')
    except:
        continue

plt.legend()

# +
long_gis = []

for i, end_point in enumerate(new_structs):
    if i==4 or i==7:
        gi = GeodesicInput.from_endpoints(initial=start_struct, final=end_point)
        traj = gi.run(nimages=40, friction=0.01, nudge=0.001)
        # traj.write_trajectory(out_dir/f"traj_{i}.xyz")
        long_gis.append(traj)

        # chain = Chain.from_traj(traj, k=99, delta_k=99, step_size=99, node_class=Node3D)
        # max_gi_vals.append(max(chain.energies))
        # works.append(chain.work)
# -

gi_4 = long_gis[0]
gi_4_chain = Chain.from_traj(gi_4, k=0.1, delta_k=0, step_size=1, node_class=Node3D)
print(max(gi_4_chain.energies))
plt.plot(gi_4_chain.energies)

gi_7 = long_gis[1]
gi_7_chain = Chain.from_traj(gi_7, k=0.1, delta_k=0, step_size=1, node_class=Node3D)
print(max(gi_7_chain.energies))
plt.plot(gi_7_chain.energies)

n_4 = NEB(gi_4_chain, grad_thre_per_atom=0.0016, vv_force_thre=0)
n_4.optimize_chain()

n_7 = NEB(gi_7_chain, grad_thre_per_atom=0.0016, vv_force_thre=0)
n_7.optimize_chain()

plt.plot((n_4.optimized.energies-n_4.optimized.energies[0])*627.5,'o--', label='neb_4')
plt.plot((n_7.optimized.energies-n_4.optimized.energies[0])*627.5,'o--', label='neb_7')
plt.legend()

print(n_7.optimized.work)
print(n_4.optimized.work)

n_7.write_to_disk(Path('./n7.xyz'))
n_4.write_to_disk(Path('./n4.xyz'))


# +
# for i in range(len(nebs)):
#     try:
#         nebs[i].write_to_disk(out_dir/f'neb_{i}.xyz')
#     except:
#         continue
# -

# ## functions

def get_all_product_isomorphisms(end_struct):
    new_structs = []
    for isom in isoms:
        new_structs.append(create_isomorphic_structure(struct=end_struct, iso=isom))
        
    return new_structs


def get_correct_product_structure(new_structs):
    max_gi_vals = []
    trajs = []
    # out_dir = Path("./GI_filter")
    for i, end_point in enumerate(new_structs):
        gi = GeodesicInput.from_endpoints(initial=start_struct, final=end_point)
        traj = gi.run(nimages=15, friction=0.01, nudge=0.001)
        # traj.write_trajectory(out_dir/f"traj_{i}.xyz")
        trajs.append(traj)

        chain = Chain.from_traj(traj, k=99, delta_k=99, step_size=99, node_class=Node3D)
        max_gi_vals.append(max(chain.energies))

    return new_structs[np.argmin(max_gi_vals)], trajs[np.argmin(max_gi_vals)]


def create_correct_interpolation(start_ind, end_ind):
    start_struct = root_conformers[start_ind]
    start_struct_coords = start_struct.coords
    start_struct.update_coords(start_struct_coords*ANGSTROM_TO_BOHR)
    
    end_struct = transformed_conformers[end_ind]
    
    new_structs = get_all_product_isomorphisms(end_struct)
    correct_end_struct, correct_gi_traj = get_correct_product_structure(new_structs)
    
    return correct_gi_traj


out = Path('./GI_filter_data/')
for r, p in pairs_to_fix:
    # print(f"{r=} {p=}")
    traj = create_correct_interpolation(r, p)
    traj.write_trajectory(out/f'traj_{r}-{p}.xyz')

print("done")


