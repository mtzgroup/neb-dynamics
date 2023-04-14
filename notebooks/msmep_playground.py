# +
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.Node2d import Node2D_Flower



import retropaths.helper_functions as hf
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.molecules.molecule import Molecule

from IPython.core.display import HTML

import matplotlib.pyplot as plt
from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
from pathlib import Path

from retropaths.reactions.template_utilities import (
    give_me_molecule_with_random_replacement_from_rules,
)


from dataclasses import dataclass 
import numpy as np
from neb_dynamics.msmep_example import plot_2D, plot_func, plot_ethan, animate_func
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from retropaths.molecules.molecule import Molecule
from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory


from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Inputs import NEBInputs, ChainInputs
from neb_dynamics.TreeNode import TreeNode

from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules


import networkx as nx

reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

# # Wittig

# +
# mol1 =  Molecule.from_smiles('[P](=CC)(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3')
mol1 =  Molecule.from_smiles('[P](=CC)(C)(C)C')
mol2 =  Molecule.from_smiles('C(=O)C')

mol = Molecule.from_smiles('[P](=CC)(C)(C)C.C(=O)C')
# -


td = TDStructure.from_RP(mol)

tds = TDStructure.from_RP(mol1)
tds2 = TDStructure.from_RP(mol2)

from retropaths.abinitio.solvator import Solvator
solv = Solvator(sphere_rad=4)
td_list = [tds, tds2]
adj_td_list = solv.adjust_td_list_coords(td_list)
td_joined = solv.join_td_list(adj_td_list)

td_joined.molecule_rp.draw(mode='d3', size=(700,700))

td.molecule_rp.draw(mode='d3',size=(400,400))

# +
p_ind = 0
cp_ind = 1

me1 = 5
me2 = 3
me3 = 4

o_ind = 7
co_ind = 6

ind=0
settings = [

    (
        td.molecule_rp,
        {'charges': [], 'delete':[(co_ind, o_ind), (cp_ind, p_ind)], 'double':[(o_ind, p_ind), (cp_ind, co_ind)]},
        [
            (me1, p_ind, 'Me'),
            (me2, p_ind, 'Me'),
            (me3, p_ind, 'Me'),
  
        ],
        [Changes3D(start=s, end=e, bond_order=1) for s, e in [(co_ind, o_ind), (cp_ind, p_ind)]],
        [Changes3D(start=s, end=e, bond_order=2) for s, e in [(o_ind, p_ind), (cp_ind, co_ind)]]

    )]

mol, d, cg, deleting_list, forming_list = settings[ind]

# +
# ind=0
# settings = [

#     (
#         td_joined.molecule_rp,
#         {'charges': [], 'delete':[(40, 41), (1, 0)], 'double':[(41, 0), (1, 40)]},
#         [
#             (3, 0, 'Me'),
#             (15, 0, 'Me'),
#             (9, 0, 'Me'),
  
#         ],
#         [Changes3D(start=s, end=e, bond_order=1) for s, e in [(40, 41), (1, 0)]],
#         [Changes3D(start=s, end=e, bond_order=2) for s, e in [(41, 0), (1, 40)]]

#     )]

# mol, d, cg, deleting_list, forming_list = settings[ind]

# +
# ind=0
# settings = [

#     (
#         td_joined.molecule_rp,
#         {'charges': [], 'delete':[(38, 39), (1, 0)], 'double':[(39, 0), (1, 38)]},
#         [
#             (2, 0, 'Me'),
#             (14, 0, 'Me'),
#             (8, 0, 'Me'),
  
#         ],
#         [Changes3D(start=s, end=e, bond_order=1) for s, e in [(38, 39), (1, 0)]],
#         [Changes3D(start=s, end=e, bond_order=2) for s, e in [(39, 0), (1, 38)]]

#     )]

# mol, d, cg, deleting_list, forming_list = settings[ind]
# -

conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

# + endofcell="--"
c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
# -

root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root.gum_mm_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.gum_mm_optimization()


root = root.xtb_geom_optimization()
target = target.xtb_geom_optimization()

# --

root

target

tr = Trajectory([root, target]).run_geodesic(nimages=15, sweep=False)

# +
# cni = ChainInputs()
# start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess.xyz", parameters=cni)
# tr.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_phe.xyz")

# +
# history.write_to_disk(Path("wittig_gone_horrible"))

# +
# tr.draw()
# -

cni = ChainInputs(k=0.10, delta_k=0.009)
nbi = NEBInputs(v=True, early_stop_chain_rms_thre=0.002, tol=0.01,max_steps=2000)
m  = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=15))
start_chain = Chain.from_traj(tr,parameters=cni)

history, out_chain = m.find_mep_multistep(start_chain)

history.data.plot_chain_distances()

out_chain.plot_chain()

out_chain.to_trajectory().draw()

out_chain.plot_chain()

history.write_to_disk(Path("./wittig_triphenyl_2"))

initial_chain = history.children[0].data.chain_trajectory[-1]


n_cont = NEB(initial_chain=initial_chain,parameters=nbi)
n_cont.optimize_chain()

n_cont.optimized.plot_chain()

out_chain.plot_chain()

# +
# cleanup_nebs[0].write_to_disk(Path("./cleanup_neb"),write_history=True)

# +
# clean_out_chain = Chain.from_list_of_chains([cleanup_nebs[0].optimized, history.output_chain], parameters=cni)

# +
# nbi = NEBInputs(v=True, stopping_threshold=0, tol=0.01)
# tr_long = Trajectory([clean_out_chain[0].tdstructure, clean_out_chain[-1].tdstructure]).run_geodesic(nimages=len(clean_out_chain), sweep=False)
# initial_chain_long = Chain.from_traj(tr_long,parameters=cni)

# +
# nbi = NEBInputs(v=True, stopping_threshold=0, tol=0.01, max_steps=4000)
# neb_long = NEB(initial_chain_long,parameters=nbi)

# +
# neb_long.optimize_chain()

# +
# neb_long.write_to_disk(Path("./neb_long_45nodes"), write_history=True)

# +
# write_to_disk(history,Path("./wittig_early_stop/"))

# +
# history = TreeNode.read_from_disk(Path("./wittig_early_stop/"))

# +
# out_chain = history.output_chain

# +
# neb_long = NEB.read_from_disk(Path("./neb_long_unconverged"))

# +
# neb_long_continued = NEB.read_from_disk(Path("./neb_long_continuation"))
# -

neb_short = NEB.read_from_disk(Path("./neb_short"))

neb_long = NEB.read_from_disk(Path("./neb_long_45nodes"))

neb_cleanup = NEB.read_from_disk(Path("./cleanup_neb"))

history = TreeNode.read_from_disk(Path("./wittig_early_stop/"))

cni = ChainInputs()
nbi = NEBInputs(v=True, stopping_threshold=3, tol=0.01)
m  = MSMEP(neb_inputs=nbi, root_early_stopping=True, chain_inputs=cni, gi_inputs=GIInputs(nimages=15))

start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess.xyz", parameters=cni)


insertion_points = m._get_insertion_points_leaves(history.ordered_leaves,original_start=start_chain[0])

# +
list_of_cleanup_nebs = [TreeNode(data=neb_cleanup, children=[])]
new_leaves = history.ordered_leaves
print('before:',len(new_leaves))
for insertion_ind, neb_obj in zip(insertion_points, list_of_cleanup_nebs):
    new_leaves.insert(insertion_ind, neb_obj)
print('after:',len(new_leaves))

new_chains = [leaf.data.optimized for leaf in new_leaves]
clean_out_chain = Chain.from_list_of_chains(new_chains,parameters=start_chain.parameters)

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(1.618*fig,fig))

# plt.plot(neb_long_continued.optimized.integrated_path_length, (neb_long_continued.optimized.energies-out_chain.energies[0])*627.5,'o-',label="NEB (30 nodes)")
# plt.plot(neb_long.optimized.integrated_path_length, (neb_long.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (45 nodes)")
# plt.plot(neb_short.optimized.integrated_path_length, (neb_short.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (15 nodes)")
# plt.plot(clean_out_chain.integrated_path_length, (clean_out_chain.energies-clean_out_chain.energies[0])*627.5,'o-',label="AS-NEB")
plt.plot(integrated_path_length(neb_long.optimized), (neb_long.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (45 nodes)")
plt.plot(integrated_path_length(neb_short.optimized), (neb_short.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (15 nodes)")
plt.plot(integrated_path_length(clean_out_chain), (clean_out_chain.energies-clean_out_chain.energies[0])*627.5,'o-',label="AS-NEB")
plt.yticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.legend(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()


# +
nimages= 15
nimages_long = 45
n_steps_orig_neb = len(neb_short.chain_trajectory)
n_steps_msmep = sum([len(obj.chain_trajectory) for obj in history.get_optimization_history()])\
+ len(list_of_cleanup_nebs[0].data.chain_trajectory)
# n_steps_long_neb = len(neb_long.chain_trajectory+neb_long_continued.chain_trajectory)
n_steps_long_neb = len(neb_long.chain_trajectory)

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(1.16*fig,fig))
plt.bar(x=["AS-NEB",f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)'],
       height=[n_steps_msmep, n_steps_orig_neb, n_steps_long_neb])
plt.yticks(fontsize=fs)
plt.ylabel("Number of optimization steps",fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# +
nimages= 15
nimages_long = 45
n_steps_orig_neb = len(neb_short.chain_trajectory)*nimages
n_steps_msmep = sum([len(obj.chain_trajectory) for obj in history.get_optimization_history()])\
+ len(list_of_cleanup_nebs[0].data.chain_trajectory)
n_steps_msmep*=15
# n_steps_long_neb = len(neb_long.chain_trajectory+neb_long_continued.chain_trajectory)
n_steps_long_neb = len(neb_long.chain_trajectory)*nimages_long

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(1.16*fig,fig))
plt.bar(x=["AS-NEB",f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)'],
       height=[n_steps_msmep, n_steps_orig_neb, n_steps_long_neb],color='orange')
plt.yticks(fontsize=fs)
plt.ylabel("Number of gradient calls",fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()


# +
def get_mass_weighed_coords(chain):
    traj = chain.to_trajectory()
    coords = traj.coords
    weights = np.array([np.sqrt(get_mass(s)) for s in traj.symbols])
    mass_weighed_coords = coords  * weights.reshape(-1,1)
    return mass_weighed_coords

def integrated_path_length(chain):
    coords = get_mass_weighed_coords(chain)

    cum_sums = [0]

    int_path_len = [0]
    for i, frame_coords in enumerate(coords):
        if i == len(coords) - 1:
            continue
        next_frame = coords[i + 1]
        dist_vec = next_frame - frame_coords
        cum_sums.append(cum_sums[-1] + np.linalg.norm(dist_vec))

    cum_sums = np.array(cum_sums)
    int_path_len = cum_sums / cum_sums[-1]
    return np.array(int_path_len)


# -

plt.plot(integrated_path_length(neb_short.initial_chain), neb_short.initial_chain.energies,'o-')
plt.plot(integrated_path_length(neb_short.optimized), neb_short.optimized.energies,'o-')


neb_short.plot_opt_history(do_3d=True)



conc_checks = [m._chain_is_concave(c) for c in neb_short.chain_trajectory]

irc_checks = []
for c in neb_short.chain_trajectory:
    r,p = m._approx_irc(c)
    minimizing_gives_endpoints = r.is_identical(c[0]) and p.is_identical(c[-1])
    irc_checks.append(minimizing_gives_endpoints)

r,p = m._approx_irc(neb_short.chain_trajectory[0])

r.tdstructure

p.tdstructure

plt.plot(conc_checks,label='has no minima')
plt.plot(irc_checks, label='irc gives input structs')
plt.legend()
plt.show()

# ### Wittig Terachem

# + endofcell="--"
ind=0
settings = [

    (
        Molecule.from_smiles("CC(=O)C.CP(=C)(C)C"),
        {'charges': [], 'delete':[(5, 6), (1, 2)], 'double':[(1, 6), (2, 5)]},
        [
            (7, 5, 'Me'),
            (8, 5, 'Me'),
            (4, 5, 'Me'),
            (0, 1, 'Me'),
            (3, 1, 'Me'),
        ],
        [Changes3D(start=s, end=e, bond_order=1) for s, e in [(5, 6), (2, 1)]],
        [Changes3D(start=s, end=e, bond_order=2) for s, e in [(1, 6), (2, 5)]]

    )]

mol, d, cg, deleting_list, forming_list = settings[ind]

conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
# -

root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root.gum_mm_optimization()
# --

root = root.xtb_geom_optimization()
root.tc_model_basis = 'gfn2xtb'
root.tc_model_method = 'gfn2xtb'

root = root.tc_geom_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.gum_mm_optimization()
target = target.xtb_geom_optimization()
target.update_tc_parameters(root)

target = target.tc_geom_optimization()

tr = Trajectory([root, target]).run_geodesic(nimages=15, sweep=False)

tr.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_tc_xtb.xyz")

cni = ChainInputs(node_class=Node3D_TC)
# start_chain = Chain.from_traj(tr,parameters=cni)
# start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_b3lyp.xyz", parameters=cni)
start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_tc_xtb.xyz", parameters=cni)

# +
# root, target = start_chain[0], start_chain[-1]
# tr = Trajectory([root, target]).run_geodesic(nimages=45, sweep=False)
# -

nbi = NEBInputs(v=True, stopping_threshold=5, tol=0.01)
m  = MSMEP(neb_inputs=nbi, root_early_stopping=True, chain_inputs=cni, gi_inputs=GIInputs())

out_chain.plot_chain()

cleanup_neb = m.cleanup_nebs(start_chain,history)

# # Knoevenangel Condensation

# +
m = MSMEP(v=True,tol=0.01)
# root, target = m.create_endpoints_from_rxn_name("Knoevenangel-Condensation", reactions)
root = TDStructure.from_rxn_name("Knoevenangel-Condensation",reactions)
c3d_list = root.get_changes_in_3d(reactions["Knoevenangel-Condensation"])

target = root.copy()
target.apply_changed3d_list(c3d_list)
target.mm_optimization("gaff")
target.mm_optimization("uff")
target.mm_optimization("mmff94")
# -

root = TDStructure.from_rxn_name("Knoevenangel-Condensation",reactions)
root.mm_optimization("gaff",steps=40000)
# root.mm_optimization("uff")
# root.mm_optimization("mmff94")

root

# %%time
n_obj, out_chain = m.find_mep_multistep(Chain(nodes=[Node3D(root), Node3D(target)],k=0.1), do_alignment=True)

# # Blum-Ittah Aziridine

cni = ChainInputs(node_class=Node3D)
nbi = NEBInputs(v=True)

[r for r in reactions]

m = MSMEP(chain_inputs=cni, neb_inputs=nbi)
root, target = m.create_endpoints_from_rxn_name("", reactions)

# # Diels Alder

cni = ChainInputs(node_class=Node3D_TC)
nbi = NEBInputs(v=True)

m = MSMEP(chain_inputs=cni, neb_inputs=nbi)
root, target = m.create_endpoints_from_rxn_name("Diels-Alder-4+2", reactions)

# %%time
n_obj, out_chain = m.find_mep_multistep(Chain(nodes=[Node3D_TC(root), Node3D_TC(target)],k=0.1), do_alignment=True)

# +
# out_chain.plot_chain() # tol was probably too high
# -

out_chain.to_trajectory()

t = Trajectory([n.tdstructure for n in out_chain.nodes])

t.draw()

t.write_trajectory("../example_cases/diels_alder/msmep_tol001.xyz")

# # Beckmann Rearrangement

# +
nbi = NEBInputs(v=True, stopping_threshold=10, tol=0.01)

m  = MSMEP(neb_inputs=nbi, recycle_chain=True, root_early_stopping=True, split_method='minima')
rn = "Beckmann-Rearrangement"
root2, target2 = m.create_endpoints_from_rxn_name(rn, reactions)
# -

# %%time
cni = ChainInputs()
input_chain = Chain(nodes=[Node3D(root2),Node3D(target2)],parameters=cni)
history, out_chain2 = m.find_mep_multistep(input_chain)

sum([len(obj.chain_trajectory) for obj in history.get_optimization_history()])

tr_long = Trajectory([out_chain2[0].tdstructure, out_chain2[-1].tdstructure]).run_geodesic(nimages=len(out_chain2), sweep=False)
start_chain_long = Chain.from_traj(tr_long,parameters=cni)

nbi = NEBInputs(v=True, stopping_threshold=0, tol=0.01)
neb_long = NEB(initial_chain=start_chain_long,parameters=nbi)

neb_long.optimize_chain()

len(out_chain2)

neb_long.optimized.plot_chain()

out_chain2.plot_chain()

t = Trajectory([n.tdstructure for n in out_chain2.nodes])

t.draw()

# +
# %%time
nbi_loose = NEBInputs(v=True, stopping_threshold=5, tol=0.01)

m2  = MSMEP(neb_inputs=nbi_loose, recycle_chain=True, root_early_stopping=False, split_method='maxima')
cni = ChainInputs()
input_chain = Chain(nodes=[Node3D(root2),Node3D(target2)],parameters=cni)

history_loose, out_chain_loose = m2.find_mep_multistep(input_chain)
# -

out_chain_loose.plot_chain()

out_chain_loose.to_trajectory().draw();

nbi = NEBInputs(v=True, stopping_threshold=10, tol=0.01)



# # Alkene-Addition-Acidification-with-Rearrangement-Iodine

m = MSMEP(max_steps=2000,v=True,tol=0.01, nudge=0)
rn = "Alkene-Addition-Acidification-with-Rearrangement-Iodine"
root3, target3 = m.create_endpoints_from_rxn_name(rn, reactions)

# %%time
n_obj3, out_chain3 = m.find_mep_multistep((root3, target3), do_alignment=True)

# # Bamberger-Rearrangement

m = MSMEP(max_steps=2000,v=True,tol=0.01, nudge=0, k=0.01)
rn = "Bamberger-Rearrangement"
root4, target4 = m.create_endpoints_from_rxn_name(rn, reactions)

# %%time
n_obj4, out_chain4 = m.find_mep_multistep((root4, target4), do_alignment=True)

# # Ugi

nbi = NEBInputs(v=True, tol=0.005,max_steps=2000)
gii = GIInputs(friction=.01, extra_kwds={'sweep':False})

root = TDStructure.from_rxn_name("Ugi-Reaction",reactions)
root.gum_mm_optimization()

root

# +
# c3d_list = root.get_changes_in_3d(reactions["Ugi-Reaction"])

root = root.pseudoalign(c3d_list)
root.gum_mm_optimization()
# -

root

target = root.copy()
target.apply_changed3d_list(c3d_list)
target.gum_mm_optimization()

root_opt  = root.xtb_geom_optimization()
target_opt = target.xtb_geom_optimization()

root_opt

reference_ugi = Trajectory.from_xyz("/home/jdep/neb_dynamics/example_cases/ugi/msmep_tol0.01_max_2000.xyz")

root_opt = reference_ugi[0]
target_opt = reference_ugi[-1]

gi = Trajectory([root_opt,target_opt]).run_geodesic(nimages=15,friction=1,sweep=False)

gi_01 = Trajectory([root_opt,target_opt]).run_geodesic(nimages=15,friction=.1,sweep=False)
gi_001 = Trajectory([root_opt,target_opt]).run_geodesic(nimages=15,friction=.01,sweep=False)

import matplotlib.pyplot as plt

plt.plot(gi.energies_xtb(), 'o-',label='friction=1')
plt.plot(gi_01.energies_xtb(), 'o-',label='friction=0.1')
plt.plot(gi_001.energies_xtb(), 'o-',label='friction=0.01')
plt.legend()

cni = ChainInputs(k=0.1,step_size=1)
nbi = NEBInputs(v=True, tol=0.005,max_steps=2000,stopping_threshold=5)
m = MSMEP(neb_inputs=nbi, recycle_chain=True, gi_inputs=gii,split_method='maxima', root_early_stopping=True)
chain = Chain.from_traj(gi_001,cni)

history, out_chain = m.find_mep_multistep(chain)

history.data.optimized.plot_chain()

out_chain.plot_chain()

out_chain.to_trajectory().write_trajectory("maybe.xyz")

# !pwd

from neb_dynamics.helper_functions import _get_ind_maxima

_get_ind_maxima(history.data.optimized)

r1,p1 = m._approx_irc(history.data.optimized)

history.data.optimized.plot_chain()

# !pwd

out_chain.plot_chain()

out_chain.to_trajectory().write_trajectory(("/home/jdep/T3D_data/msmep_draft/ugi_tree_canonical/out_chain.xyz"))

ht.write_to_disk(Path("/home/jdep/T3D_data/msmep_draft/ugi_tree_canonical"))

# # Claisen-Rearrangement

# t = Trajectory.from_xyz("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_0-0_0_cneb.xyz")
t = Trajectory.from_xyz("/home/jdep/T3D_data/msmep_draft/claisen-0-0_with_conf_rearr.xyz")

cni = ChainInputs(node_class=Node3D)
nbi = NEBInputs(v=True)
gi = Trajectory([start, p.xtb_geom_optimization()]).run_geodesic(nimages=30)
c = Chain.from_traj(gi,cni)
n = NEB(c,nbi)

n.optimize_chain()

n.optimized.plot_chain()

start = t[0]
p = t[15]
pprime = t[-1]

start.tc_model_basis = "gfn2xtb"
start.tc_model_method = 'gfn2xtb'

p.tc_model_basis = "gfn2xtb"
p.tc_model_method = 'gfn2xtb'

p_opt = p.tc_geom_optimization()

pprime.tc_model_basis = "gfn2xtb"
pprime.tc_model_method = 'gfn2xtb'

# start_opt = start.tc_geom_optimization()
start_opt = start.xtb_geom_optimization()

# pprime_opt = pprime.tc_geom_optimization()
pprime_opt = pprime.xtb_geom_optimization()

# start,end = t[0],t[-1]
cni = ChainInputs(node_class=Node3D_TC)
nbi = NEBInputs(v=True)

# +

m = MSMEP(neb_inputs=nbi,recycle_chain=False)
# -

# %%time
c = Chain([Node3D(start),Node3D(end)],parameters=cni)
n_obj4, out_chain4 = m.find_mep_multistep(c, do_alignment=False)

from neb_dynamics.NEB import NEB

start_to_p = Trajectory([start_opt, p_opt]).run_geodesic(nimages=15)
start_to_p_chain = Chain.from_traj(start_to_p, parameters=cni)
neb = NEB(initial_chain=start_to_p_chain,parameters=nbi)
neb.optimize_chain()

neb.optimized.plot_chain()

p_to_pprime = Trajectory([p_opt, pprime_opt]).run_geodesic(nimages=15)
p_to_pprime_chain = Chain.from_traj(p_to_pprime, parameters=cni)
neb2 = NEB(initial_chain=p_to_pprime_chain,parameters=nbi)
neb2.optimize_chain()

foo = Trajectory.from_list_of_trajs([neb.optimized.to_trajectory(), neb2.optimized.to_trajectory()])

import matplotlib.pyplot as plt

cni = ChainInputs(node_class=Node3D)
foo_c = Chain.from_traj(foo, parameters=cni) 

s = 8
fs = 18
f, ax = plt.subplots(figsize=(1.8*s, s))
plt.plot(foo_c.integrated_path_length, (foo_c.energies-foo_c.energies[0])*627.5,'o-')
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.show()

""

