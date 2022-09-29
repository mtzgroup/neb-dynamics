# +
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from ase.optimize.bfgs import BFGS
from ase.optimize.lbfgs import LBFGS
from pytest import approx
from neb_dynamics.geodesic_input import GeodesicInput
from retropaths.abinitio.inputs import Inputs
from retropaths.abinitio.rootstructure import RootStructure
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from xtb.ase.calculator import XTB
from xtb.interface import Calculator
from xtb.utils import get_method

from itertools import combinations_with_replacement


from retropaths.molecules.molecule import Molecule

import multiprocessing as mp

from retropaths.reactions.changes import Changes3DList, Changes3D

from tqdm.notebook import tqdm
from multiprocessing.dummy import Pool
from itertools import product


from itertools import permutations
from neb_dynamics.remapping_helpers import get_all_product_isomorphisms

from neb_dynamics.helper_functions import pload

from neb_dynamics.NEB import Node3D, Chain, NEB
from neb_dynamics.constants import BOHR_TO_ANGSTROMS, ANGSTROM_TO_BOHR

from retropaths.reactions.changes import Changes

import pandas as pd
from itertools import product
from IPython.core.display import HTML
from scipy.optimize import linear_sum_assignment
import networkx as nx


HTML('<script src="//d3js.org/d3.v3.min.js"></script>')


# +
def change_to_bohr(td):
    td = td.copy()
    coords_b = td.coords_bohr
    td.update_coords(coords_b)
    return td


def change_to_ang(td):
    td = td.copy()
    coords_a = td.coords*BOHR_TO_ANGSTROMS
    td.update_coords(coords_a)
    return td


# -

df = pd.read_excel("/Users/janestrada/Documents/gi_atomic_mapper/nature_paper/41467_2019_9440_MOESM3_ESM.xlsx",header=1)
df = df.dropna().reset_index(drop=True)
df = df.drop([100, 201, 403, 429, 455, 481, 507]).reset_index(drop=True) # drop rows that are just text, not actual data


df.iloc[0]["SMILES"]


# +
def reactants(row):
    return row['SMILES'].split(">>")[0]

def products(row):
    return row['SMILES'].split(">>")[1]

df["reactant"] = df.apply(reactants,axis=1)
df["product"] = df.apply(products,axis=1)


# -

# # Workflow for generating mapping

def get_overlap_dict(td_react, td_prod):
    mol1 = td_react.molecule_rp.remove_Hs()
    mol2 = td_prod.molecule_rp.remove_Hs()
    
    d = mol1.largest_common_subgraph(mol2)[0] ## warning, this will only take the first mapping
    return d


def invert_dict(dict1):
    dict2 = {}

    for k, v in dict1.items():
        dict2[v] = dict2.get(v, []) + [k]

    return dict2
# invert_dict(reacting_nodes_init.atom_types)


# +
def get_reacting_node_dicts(td_react, td_prod):

    mol1 = td_react.molecule_rp.remove_Hs()
    mol2 = td_prod.molecule_rp.remove_Hs()


    ds = mol1.largest_common_subgraph(mol2)
    d = ds[0]

    mol1_prime = mol1.renumber_indexes(d)
    assert mol1_prime==mol1, "Reindexing has caused a connectivity issue."

    reacting_nodes_init = mol1.copy()
    [reacting_nodes_init.remove_node(x) for x in a]

    reacting_nodes_end = mol2.copy()
    [reacting_nodes_end.remove_node(d[x]) for x in a]

    
    return reacting_nodes_init, reacting_nodes_end

# reacting_nodes_init, reacting_nodes_end = get_reacting_node_dicts(start, end)
# reacting_nodes_init, reacting_nodes_end

# +
def generate_possible_perms(d1, d2, key):
    all_pairs = []
    for i, val_i in enumerate(d1[key]):
        pairs_for_i = []
        for j, val_j in enumerate(d2[key]):
            pairs_for_i.append((val_i, val_j))
        all_pairs.append(pairs_for_i)
        
    print(f"{all_pairs=}")

    
    inds = [x for x in permutations(range(len(all_pairs)))]
    maps = []
    for ind_choice in inds:
        mapping = []
        for i, arr in zip(ind_choice, all_pairs):
            # print(i, arr[i])
            mapping.append(arr[i])
        maps.append(mapping)
    # print(f"{maps=}")
            
            
    return maps
    # return [x for x in product(d1[key], d2[key])]
        
# generate_possible_perms(d1, d2, 'C')

# +
def remapping_possibilities(d1,d2):
    remap = {}

    
    alternatives = [] # list of tuple pairs that corresponds to alternative possible mappings
    
    for key, vals in d1.items():
        assert len(d2[key])==len(d1[key]), f"Something is wrong with key: {key}"
        if len(d2[key]) == 1 and len(d1[key])==1:
            remap[vals[0]] = d2[key][0]
            
        else:
            
            res = generate_possible_perms(d1=d1, d2=d2, key=key)
            [alternatives.append(alt) for alt in res]

            
    if len(alternatives) > 0:
        list_of_remaps = []
        for alt in alternatives:

            rm = remap.copy()
            for r,p in alt:
                rm[r] = p
            list_of_remaps.append(rm)



        return list_of_remaps
    else: 
        return [remap]


# d1 = invert_dict(reacting_nodes_init.atom_types)
# d2 = invert_dict(reacting_nodes_end.atom_types)
# result = remapping_possibilities(d1, d2)
# result

# +
def pseuodoalign(mol1, mol2):
    mol1_3d = TDStructure.from_RP(mol1)
    
    changes = mol1.remove_Hs().get_bond_changes(mol2.remove_Hs())
    c3d = [Changes3D(start=s, end=e, bond_order=1) for s,e in changes.forming]
    changes_list = Changes3DList(forming=c3d, deleted=[])

    mol1_3d.add_bonds_from_changes3d(c3d=changes_list)
    mol1_3d.mm_optimization('uff')
    mol1_3d.mm_optimization('gaff')
    
    mol1_3d.delete_formed_from_changes3d(c3d=changes_list)
    
    mol1_3d.mm_optimization("uff")
    mol1_3d.mm_optimization("gaff")
    
    return mol1_3d

# td = pseuodoalign(dio1, dio2)


# -

def get_eA(arr):
    arr -= arr[0]
    delta_hartrees = max(arr) - arr[0]
    delta_kcal_mol = delta_hartrees*627.5
    return delta_kcal_mol


def run_geo(inp):
    start, end = inp
    
    start_node = Node3D(change_to_bohr(start))
    end_node = Node3D(change_to_bohr(end))
    
    
    
    
    start = TDStructure(start_node.opt_func(v=False).molecule_obmol)
    end = TDStructure(end_node.opt_func(v=False).molecule_obmol)
    # start.remove_Hs_3d()
    # end.remove_Hs_3d()

    
    
    
    gi = GeodesicInput.from_endpoints(initial=start, final=end)
    traj = gi.run(nimages=10)
    chain = Chain.from_traj(traj,k=0.1, delta_k=0,step_size=2,node_class=Node3D)

    try:
        ens = chain.energies
    except:
        ens = None
    return chain, ens


def get_chem_formula(mol):
    inv_d = invert_dict(mol.atom_types)
    
    form = {}
    
    for k,v in inv_d.items():
        form[k] = len(v)
    return form
# print(get_chem_formula(mol1))
# print(get_chem_formula(mol2))


# +
def fix_stoich(mol1, mol2):
    form1 ,form2 = get_chem_formula(mol1), get_chem_formula(mol2)
    
    smi1,smi2 = mol1.smiles, mol2.smiles
    
    for k in form1.keys():
        # print(f"k={k} // form1: {form1[k]} // form2: {form2[k]}")
        v1, v2 = form1[k], form2[k]
        if v1!=v2:
            print(f"Stoich mismatch. Uneven number of {k}. Fixing...")
            # print(f"k={k} // form1: {form1[k]} // form2: {form2[k]}")
            mol_ind_to_fix = np.argmin([v1,v2])
            if mol_ind_to_fix==0:
                # print(f"\tfixing mol1.")
                how_much_to_add = form2[k] - form1[k]
                # print(f"\tadding {how_much_to_add} of {k}")
                smi1+=f".[{k}]"*how_much_to_add
                
            elif mol_ind_to_fix==1:
                # print(f"\tfixing mol2.")
                how_much_to_add = form1[k] - form2[k]
                print(f"\tadding {how_much_to_add} of {k}")
                smi2+=f".[{k}]"*how_much_to_add
    
    # print(f"new smi1: {smi1}\nnew smi2: {smi2}")
    new_mol1, new_mol2 = Molecule.from_smiles(smi1), Molecule.from_smiles(smi2)
    return new_mol1, new_mol2
            


# mol1_prime, mol2_prime = fix_stoich(mol1,mol2)

# +
def get_candidate_mappings(mol1, mol2, do_all=False):

    all_n_edits = []
    
    all_flips = get_all_flips(mol1, mol2)
    
    mappings = make_all_mappings(all_flips)

    for m in mappings:
        changes_d ={'delete':[],
                        'single':[],
                        'double':[],
                      'aromatic':[],
                        'triple':[],
                       'charges':[]}


        mol_copy = mol1.renumber_indexes(m).copy().remove_Hs()
        mol_copy.set_neighbors()

        mol2_copy = mol2.copy().remove_Hs()
        mol2_copy.set_neighbors()




        edge_d = make_changes_d(mol2_copy)

        bonds_to_make = []

        for s,e in mol2_copy.edges:

            if (s,e) in mol_copy.edges:
                # print(f'\tremoving {s} {e}')
                mol_copy.remove_edge(s,e)
            b = mol2_copy.edges[(s,e)]
            changes_d[b['bond_order']].append((s,e))

        changes = Changes.from_dict(edge_d)
        mol_copy = mol_copy.update_edges_and_charges(changes)

        n_edits = 0
        for s,e in mol_copy.edges:
            if (s,e) not in mol2_copy.edges:
                # print(f"--{s},{e} not in mol2_copy, deleting")
                mol_copy.remove_edge(s,e)
                n_edits+=1
        mol_copy.set_neighbors()


        all_n_edits.append(n_edits)

        
        
    # now get the best
    n_edits_sorted = sorted(list(enumerate(all_n_edits)), key=lambda x: x[1])
    best_val = n_edits_sorted[0][1]
 
    if do_all:
        candidates = [mappings[x[0]] for x in n_edits_sorted]
    else:
        candidates = [mappings[x[0]] for x in n_edits_sorted if x[1]<=best_val]
    
    
    
    
    
    
    print(f"{len(candidates)} candidates.")
    return candidates


        
# get_candidate_mappings(mol1, mol2)
# -

def get_all_flips(mol1, mol2):

    all_flips = {}
    
    buck = bucket_atoms(mol1)
    
    count=1
    for key in buck.keys():
        # if key1.split('-')[1]==key2.split('-')[1] and "H" not in key1:
        if "H" !=key:
            vals = buck[key]
            flips = list(permutations(vals))

            n_flips = len(flips)
            all_flips[key] = {'flips':flips, 'indices':vals}

            count*=n_flips
    print(f"{count} total flips are possible")
    return all_flips


# +
def make_perm_dict(all_flips):
    perm_dict = {}
    for x in range(len(all_flips)):
        for i, (key, val) in enumerate(all_flips[x]):
            if key in perm_dict.keys():
                perm_dict[key].append(val)
            else:
                perm_dict[key] = [val]

    return perm_dict

# pd = make_perm_dict(af)
# count=1
# for key in pd.keys():
#     count*=len(pd[key])
    
# count
    


# -

def bucket_atoms(mol):
    buckets = {}
    for n in mol.nodes:
        node = mol.nodes[n]
        # label = f"{node['neighbors']}-{node['element']}-{node['charge']}"
        label = f"{node['element']}"
        
        if label == 'H': continue # skipping hydrogens for now
        
        if label in buckets.keys():
            buckets[label].append(n)

        else:
            buckets[label]=[n]
    return buckets
# bucket_atoms(mol1)


def make_all_mappings(all_flips):

    mappings = []
    
    all_perms_list = list(product(*[all_flips[k]['flips'] for k in all_flips.keys()]))
    for val in all_perms_list:
        m = {}
        for i, key in enumerate(all_flips.keys()):
            orig_indices = all_flips[key]['indices']
            for ix_orig, ix_new in zip(orig_indices, val[i]):
                m[ix_orig] = ix_new
        mappings.append(m)
        
    return mappings


def make_changes_d(mol):
    changes_d ={'delete':[],
                'single':[],
                'double':[],
              'aromatic':[],
                'triple':[],
               'charges':[]}


    for i in range(len(mol.edges)):
        s,e = list(mol.edges)[i]
        b = mol.edges[(s,e)]
        changes_d[b['bond_order']].append((s,e))

    return changes_d
# make_changes_d(mol2)

def get_correct_3d_aligned_structures(inp):
    r_mol, p_mol = inp
    r_3d = pseuodoalign(r_mol, p_mol)
    p_3d = TDStructure.from_RP(p_mol)
    return (r_3d,p_3d)


def get_r_mols(inp):
    remap, td_start = inp
    mol1 = td_start.molecule_rp.copy()
    mol1 = mol1.renumber_indexes(remap)

    return mol1


def get_ints_to_do(all_ds, td_start, td_end, parallel=False):
    # get list of remapped starting structures
    
    if not parallel:
        all_r_mols = []
        for remap in all_ds:
            mol1 = td_start.molecule_rp.copy()
            mol1 = mol1.renumber_indexes(remap)

            all_r_mols.append(mol1)
    else:
        remaps_and_start_mol = list(zip(all_ds, [td_start]*len(all_ds)))
        with Pool() as p:
            all_r_mols = list(tqdm(p.imap(get_r_mols, remaps_and_start_mol,chunksize=10), total=len(remaps_and_start_mol)))


    end_mol = td_end.molecule_rp.copy()
    ints_mols = list(product(all_r_mols, [end_mol]))
    
    if not parallel:
        ints_to_do = []
        for i, (r_mol, p_mol) in enumerate(ints_mols):
            print(f'3DMolGen:{"*"*int((i/len(ints_mols))*10)}|{round((i/len(ints_mols))*100,1)}%                 \r', end="")
            # print(f"{int((i/len(ints_mols))*100)} % done")
            r_3d = pseuodoalign(r_mol, p_mol)
            p_3d = TDStructure.from_RP(p_mol)
            ints_to_do.append((r_3d,p_3d))
    else:
        with Pool() as p:
            ints_to_do = list(tqdm(p.imap(get_correct_3d_aligned_structures, ints_mols,chunksize=10), total=len(ints_mols)))
        
    return ints_to_do


def correctly_map_two_smiles(smi1, smi2, parallel=False):
    # make sure they have same stoic
    mol1, mol2 = Molecule.from_smiles(smi1), Molecule.from_smiles(smi2)
    mol1, mol2 = fix_stoich(mol1, mol2)
    
    # make 3d structures
    start = TDStructure.from_RP(mol1)
    end = TDStructure.from_RP(mol2)
    
    all_candidates = get_candidate_mappings(mol1, mol2)
        
    print(f"There are {len(all_candidates)} solutions to this pair") # all_ds is a dictionary that says what each key in the R is in the P molecule
    
    if len(all_candidates) > 1:
        print("Deciding between them using GI")
        
        ints_to_do = get_ints_to_do(all_ds=all_candidates, td_start=start , td_end=end, parallel=parallel)
        
        
        if not parallel:
            print("Running GI in series")
            gi_results = []
            for i, inp in enumerate(ints_to_do):
                print(f'GI:{"*"*int((i/len(ints_to_do))*10)}|{round((i/len(ints_to_do))*100,1)}%\r', end="")
                # print(f'GI: doing interpolation {i}')
                gi_results.append(run_geo(inp))
        else:
            print("Running GI in parallel")
            with Pool() as p:
                gi_results = list(tqdm(p.imap(run_geo, ints_to_do), total=len(ints_to_do)))
            
        gi_eA = [get_eA(en) for _, en in gi_results if type(en)!=type(None)] # energy returned in kcals!!
        ea_thre = 300 #kcal/mol
        inds = np.where((gi_eA - min(gi_eA)) <= ea_thre)[0]
        return np.array(all_candidates), gi_results, inds
    else:
        return np.array(all_candidates), None

# # Test Run

# +
smi_r, smi_p = 'C=C.C=CC=C','C1C=CCCC1'
# smi_r, smi_p = 'C=CC(CC=C)C', 'C(C=CC)CC=C'
# smi_r, smi_p = df.iloc[0]['reactant'], df.iloc[0]['product']



# -

# correct answer for mappings in Cope Rearrangement
correct = {6:3, 2:2, 1:1, 5:4, 4:5, 3:6, # C's
           13:16,
          18:13, 17:11, 16:12, # H on C3
          11:18, 12:17}

mol1 = Molecule.from_smiles(smi_r)
mol2 = Molecule.from_smiles(smi_p)


output  = correctly_map_two_smiles(smi_r, smi_p, parallel=False)

gi_eAs = [get_eA(en ) for c, en in output[1] if type(en)!=type(None)] 
ea_thre = 100 #kcal/mol
good_candidates = np.where(gi_eAs-min(gi_eAs) < ea_thre)[0]

# +
for i, (c, en) in enumerate(output[1]):
    if i in good_candidates:
        plt.plot(en, 'o--', label=i)
    
plt.legend()
# -

cm = get_candidate_mappings(mol1, mol2)

correct = {
    5:0,4:1, 3:2, 2:3, 1:4, 0:5
}

for i, val in enumerate(cm):
    if val==correct:
        print(i)

len(cm)

Molecule.draw_list([mol1, mol2], mode='d3')

good_mols = [mol1.renumber_indexes(cm[i]) for i in good_candidates]
good_mols.append(mol2)
Molecule.draw_list(good_mols, mode='d3', size=(300,300),names=[i for i in good_candidates])






