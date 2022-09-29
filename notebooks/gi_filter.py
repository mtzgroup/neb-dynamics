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


import multiprocessing as mp

from tqdm.notebook import tqdm
from multiprocessing.dummy import Pool
from itertools import product


from itertools import permutations
from neb_dynamics.remapping_helpers import get_all_product_isomorphisms

from neb_dynamics.helper_functions import pload

from neb_dynamics.NEB import Node3D, Chain, NEB
from neb_dynamics.constants import BOHR_TO_ANGSTROMS, ANGSTROM_TO_BOHR

out_dir = Path("/Users/janestrada/neb_dynamics/example_cases")
rxn_file = Path("/Users/janestrada/Retropaths/retropaths/data/reactions.p")


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

rxns = pload(rxn_file)

rn = "Diels-Alder-4+2"
# rn = "Hantzsch-Thiazole-Synthesis-X-Chlorine"
rxns[rn].draw()

# +
inps = Inputs(rxn_name=rn, reaction_file=rxn_file)


struct = TDStructure.from_rxn_name(rn, data_folder=rxn_file.parent)
rs = RootStructure(root=struct, master_path=out_dir, rxn_args=inps, trajectory=Trajectory(traj=[]))
# -

rs.root.mm_optimization('uff')

rs.root

rs.transformed.mm_optimization('uff')

rs.transformed

# # Generate all atomic mappings

start = TDStructure.from_smiles("C=C.C=CC(=C)")
end = TDStructure.from_smiles("C1=CCCCC1")
# start = rs.root
# end = rs.transformed
start_opt = Node3D(change_to_bohr(start)).opt_func()
end_opt = Node3D(change_to_bohr(end)).opt_func()


# +
# end = TDStructure.from_smiles("C1=CCCCC1")

# +
all_end_points = get_all_product_isomorphisms(end)

all_start_points = get_all_product_isomorphisms(start)

# +
print(len(all_start_points))

print(len(all_end_points))
# -

# # Parallelize the generation of GIs

# +


# ints_to_do = [(all_start_points[0],all_end_points[0]), (all_start_points[0], all_end_points[1])]
ints_to_do = list(product(all_start_points, all_end_points))
ints_to_do = list(product([all_start_points[0]], all_end_points))

def run_geo(inp):
    start, end = inp
    gi = GeodesicInput.from_endpoints(initial=start, final=end)
    traj = gi.run(nimages=10)
    chain = Chain.from_traj(traj,k=0.1, delta_k=0,step_size=2,node_class=Node3D)

    try:
        ens = chain.energies
    except:
        ens = None
    return chain, ens

# with Pool() as p:
#     results = list(tqdm(p.imap(run_geo, ints_to_do), total=len(ints_to_do)))


# +
# for i, (c, _) in enumerate(results):
#     traj = Trajectory([node.tdstructure for node in c])
#     traj.write_trajectory(f"./GI_filter/DielsAlder/interpolation_{i}.xyz")
# -

for i, (c, en) in enumerate(results):
    if type(en)!=type(None):
        # print(i, en, type(en))
        plt.plot(c.integrated_path_length, en, 'o-', label=f"gi_{i}")
# plt.legend()

def get_eA(arr):
    arr -= arr[0]
    delta_hartrees = max(arr) - arr[0]
    delta_kcal_mol = delta_hartrees*627.5
    return delta_kcal_mol
get_eA(results[0][1])

gi_eA = [get_eA(en) for _, en in results if type(en)!=type(None)]


# +
# (gi_eA - min(gi_eA))*627.5 
# -

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
# len(make_all_mappings(pd))


ea_thre = 1000000
inds = np.where((gi_eA - min(gi_eA))*627.5 <= ea_thre)[0]
print(inds)
print(len(inds))
for i in inds:
    c, en = results[i]
    if type(en) != type(None):
        print(f"\t{i}")
        plt.plot(c.integrated_path_length, en*627.5, 'o-', label=f"gi_{i}")
plt.legend()

t = Trajectory([change_to_ang(x.tdstructure) for x in results[3][0].nodes])
t.write_trajectory("best_DA_gi.xyz")

from neb_dynamics.NEB import NoneConvergedException
def run_neb_parallel(chain):
    try:
        n = NEB(initial_chain=chain, v=False, max_steps=2000)
        obj = n.optimize_chain()
    except NoneConvergedException:
        return n
        
    return obj


chains_to_do = [results[i][0] for i in inds]

with Pool() as p:
    results_neb = list(tqdm(p.imap(run_neb_parallel, chains_to_do), total=len(chains_to_do)))

results_neb[0].chain_trajectory[-1].plot_chain()

# +
# results_neb[1].chain_trajectory[-1].plot_chain()

# +
# results_neb[2].chain_trajectory[-1].plot_chain()
# -

# # Nature dataset

import pandas as pd
from itertools import product

df = pd.read_excel("/Users/janestrada/Documents/gi_atomic_mapper/nature_paper/41467_2019_9440_MOESM3_ESM.xlsx",header=1)
df = df.dropna().reset_index(drop=True)
df = df.drop([100, 201, 403, 429, 455, 481, 507]).reset_index(drop=True) # drop rows that are just text, not actual data


df


# +
def reactants(row):
    return row['SMILES'].split(">>")[0]

def products(row):
    return row['SMILES'].split(">>")[1]

df["reactant"] = df.apply(reactants,axis=1)
df["product"] = df.apply(products,axis=1)
# -

import matplotlib.pyplot as plt

lens = [len(val) for val in df['reactant'].values]
df['react_len'] = lens

df.iloc[1]['SMILES']

i = 0
r, p = df.iloc[i]['SMILES'].split(">>")

# r_td = TDStructure.from_smiles(r)
r_td = TDStructure.from_smiles("CCCl")
r_td

# p_td =TDStructure.from_smiles(p)
p_td =TDStructure.from_smiles("C")
p_td

# +
start = r_td
end = p_td

# start_opt = Node3D(change_to_bohr(start)).opt_func(v=False)
# end_opt = Node3D(change_to_bohr(end)).opt_func(v=False)

# -

r_mol = start.molecule_rp
p_mol = end.molecule_rp

r_mol.draw()

p_mol.draw()

# +
ismags = nx.isomorphism.ISMAGS(r_mol.to_directed(), p_mol.to_directed())
isomorphisms = list(ismags.isomorphisms_iter(symmetry=False))
print(len(isomorphisms))

largest_common_subgraph = list(ismags.largest_common_subgraph())

print(len(largest_common_subgraph))

# +
ismags = nx.isomorphism.ISMAGS(p_mol.to_directed(), r_mol.to_directed())
isomorphisms = list(ismags.isomorphisms_iter(symmetry=False))
print(len(isomorphisms))

largest_common_subgraph = list(ismags.largest_common_subgraph())

print(len(largest_common_subgraph))
# -

r_mol.subgraph(r_mol.nodes[10])

r_subs = r_mol.subgraph(r_mol.nodes)
r_subs[0]

p_subs = [p_mol.subgraph(i) for i in p_mol.nodes]
p_subs[1]

True in [r_mol.is_subgraph_isomorphic_to(p_subs[i]) for i in p_subs]

nx.is_isomorphic(r_mol.to_undirected(), p_mol.to_undirected())

import networkx as nx
from retropaths.molecules.molecule import Molecule
R = nx.intersection(p_mol.to_undirected(), r_mol.to_undirected())

all_end_points = get_all_product_isomorphisms(end, timeout=200)

len(all_end_points)

all_start_points = get_all_product_isomorphisms(start, timeout=200)

# +
print(len(all_start_points))

print(len(all_end_points))
# -

ints_to_do = list(product([all_start_points[0]], all_end_points))
with Pool() as p:
    results = list(tqdm(p.imap(run_geo, ints_to_do), total=len(ints_to_do)))

for i, (c, en) in enumerate(results):
    if type(en)!=type(None):
        # print(i, en, type(en))
        plt.plot(c.integrated_path_length, en, 'o-', label=f"gi_{i}")
plt.legend()

gi_eA = [get_eA(en) for _, en in results if type(en)!=type(None)]
ea_thre = 100
inds = np.where((gi_eA - min(gi_eA))*627.5 <= ea_thre)[0]
print(inds)
print(len(inds))
for i in inds:
    c, en = results[i]
    if type(en) != type(None):
        print(f"\t{i}")
        plt.plot(c.integrated_path_length, en*627.5, 'o-', label=f"gi_{i}")
plt.legend()

change_to_ang(all_start_points[0])

# +
from IPython.core.display import HTML
import json 

mol = all_end_points[4].molecule_rp
# -

list(zip(all_end_points[0].symbols, all_end_points[0].coords))

i=2
list(zip(all_end_points[i].symbols, all_end_points[i].coords))

chain, en = results[4]
traj = Trajectory([x.tdstructure for x in chain.nodes])
traj

# ? traj


