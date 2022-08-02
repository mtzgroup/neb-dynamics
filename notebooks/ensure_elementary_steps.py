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
# -

rxn_name = 'Claisen-Rearrangement'
reaction_path = Path("/Users/janestrada/Documents/Stanford/Retropaths/retropaths/data/reactions.p")

out_path=Path("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement")
inp = Inputs(rxn_name=rxn_name, reaction_file=reaction_path)
root_structure = RootStructure(
    root=TDStructure.from_rxn_name(rxn_name, data_folder=reaction_path.parent),
    master_path=out_path,
    rxn_args=inp,
    trajectory=Trajectory([]),
)

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
traj = Trajectory.from_xyz(Path(f"./Claisen-Rearrangement/traj_{r_ind}-{p_ind}_neb.xyz"))
chain = Chain.from_traj(traj, k=1,delta_k=1,step_size=0,node_class=Node3D)

plt.plot(chain.energies, 'o--')
# -

root_structure._get_bonds_between_fragments()

r.coords[5]


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

# +
origin_a = get_atom_xyz(r, 5) 
r1_a = get_atom_xyz(r, 12) - origin_a
r2_a = get_atom_xyz(r, 13) - origin_a

origin_p_a = get_atom_xyz(p, 5)
p1_a = get_atom_xyz(p, 12) - origin_p_a
p2_a =  get_atom_xyz(p, 13) - origin_p_a
# -

cr_1 = np.cross(r1_a, r2_a)

cr_2 = np.cross(p1_a, p2_a)

np.sign(np.dot(cr_1, cr_2))

# # Attempt 2, The square bullshit

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

# # Attempt 3: Checking if two lines cross

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


# +
# def get_intersect(a1, a2, b1, b2):
#     """ 
#     Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
#     a1: [x, y] a point on the first line
#     a2: [x, y] another point on the first line
#     b1: [x, y] a point on the second line
#     b2: [x, y] another point on the second line
#     """
#     s = np.vstack([a1,a2,b1,b2])        # s for stacked
#     h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
#     l1 = np.cross(h[0], h[1])           # get first line
#     l2 = np.cross(h[2], h[3])           # get second line
#     x, y, z = np.cross(l1, l2)          # point of intersection
#     if z == 0:                          # lines are parallel
#         return (float('inf'), float('inf'))
#     return (x/z, y/z)
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


check_if_lines_intersect(
    p1_var=np.array((0,0)),
    p2_var=np.array((1,1)),
    p3_var=np.array((0,0)),
    p4_var=np.array((0,1))
)

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
# -

check_if_lines_intersect(p1_var=r_6[[0,2]], p2_var=r_13[[0,2]], p3_var=r_7[[0,2]], p4_var=r_12[[0,2]])

_check_point_in_both_lines([ 0.01729678,-0.10795274], line1=(r_6[[0,2]],r_13[[0,2]]), line2=(r_7[[0,2]],r_12[[0,2]]))



# +
x1 = r_6[[0,2]]
x2 = r_13[[0,2]]
x3 = r_7[[0,2]]
x4 = r_12[[0,2]]


plt.plot([x1[0], x2[0]], [x1[1],x2[1]], label='6-13')
plt.plot([x3[0], x4[0]], [x3[1],x4[1]], label='7-12')
plt.legend()
# -

check_if_lines_intersect(p1_var=p_6[[0,2]], p2_var=p_13[[0,2]], p3_var=p_7[[0,2]], p4_var=p_12[[0,2]])


def get_points(struct, indices):
    a1, a2, b1, b2 = indices
    p_a1 = get_atom_xyz(struct, a1)
    p_a2 = get_atom_xyz(struct, a2)

    p_b1 = get_atom_xyz(struct, b1)
    p_b2 = get_atom_xyz(struct, b2)
    
    return p_a1, p_a2, p_b1, p_b2


def get_neigh_inds(struct, ref_atoms_inds):
    
    ref_atom_1, ref_atom_2 = ref_atoms_inds
    
    a1, a2 = get_neighs(product_struc, ref_atom_1)

    b1, b2 = get_neighs(product_struc, ref_atom_2)
    
    return a1, a2, b1, b2


# +
a1, a2, b1, b2 = get_neigh_inds(product_struc, rs._get_bonds_between_fragments()['single'][0])
p_a1, p_a2, p_b1, p_b2 = get_points(product_struc, [a1,a2,b1,b2])


x1 = p_a1[[0,2]]
x2 = p_b1[[0,2]]
x3 = p_a2[[0,2]]
x4 = p_b2[[0,2]]


plt.plot([x1[0], x2[0]], [x1[1],x2[1]])
plt.plot([x3[0], x4[0]], [x3[1],x4[1]])


# +
# react_struc.write_to_disk("DEBUG_22.xyz")
# product_struc.write_to_disk("DEBUG_11.xyz")
# -

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


# this will be the function for checking if indices need to be flipped
product_struc = p
react_struc = r
rs = root_structure


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

root_conformers[0].write_to_disk("DEBUG_0.xyz")

transformed_conformers[3].write_to_disk("DEBUG_3.xyz")


