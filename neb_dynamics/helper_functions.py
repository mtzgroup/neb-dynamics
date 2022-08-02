"""
this module contains helper general functions
"""
import cProfile
import json
import multiprocessing as mp
import pickle
import signal
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.core.display import HTML


def print_matrix_2D(mat: np.array, precision: float = 6, threshold: float = 0.0):
    """
    given a numpy 2d array, returns the pandas matrix (making it beautiful in jupyter)
    """
    pd.set_option("precision", precision)
    pd.set_option("chop_threshold", threshold)
    (siza, _) = mat.shape
    indexes = np.arange(siza) + 1
    out = pd.DataFrame(mat, index=indexes, columns=indexes)
    return out


def pairwise(iterable):
    """
    from a list [a,b,c,d] to [(a,b),(b,c),(c,d)]
    """
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)
        a = b


def read_json(fn: Path):
    with open(fn, "r") as f:
        data = json.load(f)
    return data


def psave(thing, fn):
    """this quickle saves an object to disc"""
    pickle.dump(thing, open(fn, "wb"))


def pload(fn):
    """this quickle loads an object from disc"""
    return pickle.load(open(fn, "rb"))


def draw_starting_node(tree, name, size=(500, 500)):
    """
    This function is used to make presentations
    """
    molecule = tree.nodes[0]["molecule_list"][0]
    string = f"""<h2>{name.replace('_',' ').capitalize()}</h2><h3>Smiles: {molecule.smiles}</h3>
    <div style="width: 100%; display: table;"> <div style="display: table-row;">
    <div style="width: 30%; display: table-cell;">
    <p style="text-align: left;"><b>Rdkit visualization</b></p>
    {molecule.draw(mode='rdkit', string_mode=True)}
    </div>
    <div style="width: 10%; display: table-cell; vertical-align: middle;">
    <b>---></b>
    </div>
    <div style="width: 50%; display: table-cell;">
    <p style="text-align: left;"><b>Internal Graph visualization</b></p>
    {molecule.draw(mode='d3', string_mode=True, size=size, node_index=False, percentage=0.5)}
    </div>
    </div></div>"""
    return HTML(string)


def load_pickle(fn):
    """
    tedious to remember protocol flag and stuffs
    fn :: FilePath
    """
    return pickle.load(open(fn, "rb"))


def save_pickle(thing, fn):
    """
    tedious part 2
    fn :: FilePath
    thing :: Structure to save
    """
    with open(fn, "wb") as pickle_file:
        pickle.dump(thing, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def wrapper(fn, tup1, tup2):
    fn(*tup1, *tup2)


def execute_parallel(function, iterator, repeated_arguments):
    with mp.Pool() as pool:
        pool.starmap(wrapper, zip(repeat(function), iterator, repeat(repeated_arguments)))


def execute_serial(function, iterator, repeated_arguments):
    for iter in iterator:
        print(f"\n\nI am executing {function} in {iter} with {len(repeated_arguments)} repeated arguments")
        function(*iter, *repeated_arguments)


def profile_this_function(func):
    """
    a decorator for activating the profiler
    """

    def inner1(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        func(*args, **kwargs)
        pr.disable()
        pr.print_stats()

    return inner1


# --------------

def get_atom_xyz(struct, atom_ind):
    
    atom_r = struct.molecule_obmol.GetAtom(atom_ind+1)
    return np.array((atom_r.x(), atom_r.y(), atom_r.z()))


def get_neighs(struct, ind):

    neighs = [] # indices of the single bonds 
    for neigh in struct.molecule_rp[ind]:
        if struct.molecule_rp[ind][neigh]['bond_order']=="double": continue
        else: neighs.append(neigh)
        
    return neighs[:2]


def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def get_intersect(a1, a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1


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
        
    else: 
        return False
    

def flip_structure_if_necessary(react_struc, product_struc, rs):
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
        
        react_struct = flip_coordinates(react_struc, a1, a2)
        
        p_a1, p_a2, p_b1, p_b2 = get_points(react_struct, [a1,a2,b1,b2])
        assert not check_if_lines_intersect(p1_var=p_a1[[0,2]], p2_var=p_b1[[0,2]], p3_var=p_a2[[0,2]], p4_var=p_b2[[0,2]])
    


        return react_struct
    return False
        

# -------

# @author: alexchang
import numpy as np
import scipy.sparse.linalg

def quaternionrmsd(c1, c2):
    N = len(c1)
    if len(c2) != N:
        raise "Dimensions not equal!"
    bary1 = np.mean(c1, axis = 0)
    bary2 = np.mean(c2, axis = 0)

    c1 = c1 - bary1
    c2 = c2 - bary2

    R = np.dot(np.transpose(c1), c2)

    F = np.array([[(R[0, 0] + R[1, 1] + R[2, 2]), (R[1, 2] - R[2, 1]), (R[2, 0] - R[0, 2]), (R[0, 1] - R[1, 0])], 
            [(R[1, 2] - R[2, 1]), (R[0, 0] - R[1, 1] - R[2, 2]), (R[1, 0] + R[0, 1]), (R[2, 0] + R[0, 2])],
            [(R[2, 0] - R[0, 2]), (R[1, 0] + R[0, 1]), (-R[0, 0] + R[1, 1] - R[2, 2]), (R[1, 2] + R[2, 1])],
            [(R[0, 1] - R[1, 0]), (R[2, 0] + R[0, 2]), (R[1, 2] + R[2, 1]), (-R[0, 0] - R[1, 1] + R[2, 2])]])
    eigen = scipy.sparse.linalg.eigs(F, k = 1, which = 'LR')
    lmax = float(eigen[0][0])
    qmax = np.array(eigen[1][0:4])
    qmax = np.float_(qmax)
    qmax = np.ndarray.flatten(qmax)
    rmsd = ((np.sum(np.square(c1)) + np.sum(np.square(c2)) - 2 * lmax)/N)** 0.5
    rot = np.array([[(qmax[0]**2 + qmax[1]**2 - qmax[2]**2 - qmax[3]**2), 2*(qmax[1]*qmax[2] - qmax[0]*qmax[3]), 2*(qmax[1]*qmax[3] + qmax[0]*qmax[2])],
               [2*(qmax[1]*qmax[2] + qmax[0]*qmax[3]), (qmax[0]**2 - qmax[1]**2 + qmax[2]**2 - qmax[3]**2), 2*(qmax[2]*qmax[3] - qmax[0]*qmax[1])],
               [2*(qmax[1]*qmax[3] - qmax[0]*qmax[2]), 2*(qmax[2]*qmax[3] + qmax[0]*qmax[1]), (qmax[0]**2 - qmax[1]**2 - qmax[2]**2 + qmax[3]**2)]])
    g_rmsd = (c1 - np.matmul(c2, rot))/(N*rmsd)
    
    # return rmsd
    return g_rmsd