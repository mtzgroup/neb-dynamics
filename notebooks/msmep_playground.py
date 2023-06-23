# +
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.Node2d import Node2D_Flower


from retropaths.reactions.changes import Changes3D, Changes3DList, ChargeChanges

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
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Node3D_TC_TCPB import Node3D_TC_TCPB
from neb_dynamics.Inputs import NEBInputs, ChainInputs
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS

from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
# import os
# del os.environ['OE_LICENSE']

import networkx as nx

reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

# # play

h_dft = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig_DFT/initial_guess_msmep/"), 
                                chain_parameters=ChainInputs(node_class=Node3D_TC))

h_dft.output_chain.plot_chain()

c_dft = Chain.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig_DFT/initial_guess_msmep.xyz"), parameters=ChainInputs())
c_xtb = Chain.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/initial_guess_msmep.xyz"), parameters=ChainInputs())

plt.plot(c_dft.integrated_path_length, (c_dft.energies-c_dft.energies[0])*627.5, 'o-',label="b3lyp/3-21gs")
plt.plot(c_xtb.integrated_path_length, (c_xtb.energies-c_xtb.energies[0])*627.5, 'o-',label="GFN2XTB")
plt.legend()

orig = Trajectory.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig_DFT/initial_guess.xyz"))

start = orig[0]
end = orig[-1]

# +
method = 'b3lyp'
basis = '3-21gs'
kwds = {'restricted': False}


start.tc_model_basis = basis
start.tc_model_method = method
start.tc_kwds = kwds

end.update_tc_parameters(start)
# -

start.energy_tc_tcpb()

cni = ChainInputs(k=0.1, delta_k=0.09,node_class=Node3D_TC_TCPB, 
                  do_parallel=False, als_max_steps=3)
chain = Chain.from_traj(orig, parameters=cni)

nbi = NEBInputs(grad_thre=0.001*BOHR_TO_ANGSTROMS, 
                rms_grad_thre=0.0005*BOHR_TO_ANGSTROMS,
                v=True)
n = NEB(initial_chain=chain, parameters=nbi)

n.optimize_chain()

# # Wittig

# +
# mol1 =  Molecule.from_smiles('[P](=CC)(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3')
# mol1 =  Molecule.from_smiles('[P](=CC)(C)(C)C')
# mol2 =  Molecule.from_smiles('C(=O)C')

# mol = Molecule.from_smiles('[P](=CC)(C)(C)C.C(=O)C')
mol = Molecule.from_smiles('[P](=C)(C)(C)C.CC(=O)C')
# -


mol.draw(mode='d3')

td = TDStructure.from_RP(mol)

# +
p_ind = 0
cp_ind = 1

me1 = 2
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


root_opt = root.xtb_geom_optimization()
target_opt = target.xtb_geom_optimization()

# --

node_start = Node3D(root_opt)

node_end = Node3D(target_opt)

from neb_dynamics.helper_functions import RMSD


def get_vals(self, other):
    aligned_self = self.tdstructure.align_to_td(other.tdstructure)
    dist = RMSD(aligned_self.coords, other.tdstructure.coords)[0]
    en_delta = np.abs((self.energy - other.energy)*627.5)
    return dist, en_delta


tr = Trajectory([root_opt, target_opt]).run_geodesic(nimages=15, sweep=False)

tr.draw()

tr.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/initial_guess_tight_endpoints.xyz")

from neb_dynamics.constants import BOHR_TO_ANGSTROMS

# +
traj = Trajectory.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/initial_guess.xyz"), tot_charge=0, tot_spinmult=1)
tol = 0.01
cni = ChainInputs(k=0.01, node_class=Node3D)

nbi = NEBInputs(tol=tol, # tol means nothing in this case
    grad_thre=0.001*BOHR_TO_ANGSTROMS,
    rms_grad_thre=0.0005*BOHR_TO_ANGSTROMS,
    en_thre=0.001*BOHR_TO_ANGSTROMS,
    v=True,
    max_steps=4000,
    early_stop_chain_rms_thre=0.002,
    early_stop_force_thre=0.02,vv_force_thre=0.00*BOHR_TO_ANGSTROMS,)
chain = Chain.from_traj(traj=traj, parameters=cni)
m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs())
# -

n = NEB(initial_chain=chain,parameters=nbi)

n.optimize_chain()

n.optimized.plot_chain()



h, out = m.find_mep_multistep(chain)

cni = ChainInputs()
start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/initial_guess.xyz", parameters=cni)
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
