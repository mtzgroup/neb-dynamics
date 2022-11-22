# +
from retropaths.molecules.molecule import Molecule
from IPython.core.display import HTML
from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB, Chain, Node3D, NoneConvergedException
from neb_dynamics.MSMEP import MSMEP
import matplotlib.pyplot as plt
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
import numpy as np
from scipy.signal import argrelextrema
from retropaths.helper_functions import pairwise

HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

# +
# mol = Molecule.from_smiles("C(=O)(C)C.C=P(c1ccccc1)(c2ccccc2)c3ccccc3")
mol = Molecule.from_smiles("CC(=O)C.CP(=C)(C)C")
# mol2 = Molecule.from_smiles("CC(=C)C.c1ccc(cc1)P(=O)(c2ccccc2)c3ccccc3")
# mol2 = Molecule.from_smiles("CC(=C)C.CP(=O)(C)C")

mol.draw(mode='d3',node_index=True)

# +
d = {'charges':[],'delete':[(5,6),(1,2)], 'double':[(1,6),(2,5)]}
conds = Conditions()
rules = Rules()

temp = ReactionTemplate.from_components(name='Wittig', reactants=mol,changes_react_to_prod_dict=d, conditions=conds, rules=rules)
# -

# # NEB stuff

# +
deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,6), (2,1)]]
forming_list = [Changes3D(start=s,end=e, bond_order=2) for s,e in [(1,6), (2,5)]]

c3d_list = Changes3DList(deleted=deleting_list,forming=forming_list, charges=[])

# +
root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root = root.xtb_geom_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.mm_optimization("gaff")
target = target.xtb_geom_optimization()
# -

root.molecule_rp.get_bond_changes(target.molecule_rp)

root.get_changes_in_3d()

# +
traj = Trajectory([root.xtb_geom_optimization(), target.xtb_geom_optimization()])
gi = traj.run_geodesic(nimages=15)

chain = Chain.from_traj(gi,k=0.1,delta_k=0, step_size=1,node_class=Node3D)
tol = .03
n = NEB(initial_chain=chain,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
n.optimize_chain()
# -

t = Trajectory([node.tdstructure for node in n.optimized])

n.optimized.plot_chain()

# ### now, im going to split it up into fragments

# +
c_opt = n.optimized
frag1 = Trajectory([c_opt[0].tdstructure.xtb_geom_optimization(), c_opt[3].tdstructure.xtb_geom_optimization()])
gi_frag1 = frag1.run_geodesic(nimages=15)
c_frag1 = Chain.from_traj(gi_frag1,k=0.1,delta_k=0,node_class=Node3D,step_size=1)

tol = .03
n_frag1 = NEB(initial_chain=c_frag1,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
n_frag1.optimize_chain()
# -

n_frag1.chain_trajectory[-1].plot_chain() # this is kinda dumb

# +
frag2 = Trajectory([c_opt[3].tdstructure.xtb_geom_optimization(), c_opt[7].tdstructure.xtb_geom_optimization()])
gi_frag2 = frag2.run_geodesic(nimages=15)
c_frag2 = Chain.from_traj(gi_frag2,k=0.1,delta_k=0,node_class=Node3D,step_size=1)

tol = .03
n_frag2 = NEB(initial_chain=c_frag2,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
n_frag2.optimize_chain()
# -

n_frag2.chain_trajectory[-1].plot_chain()

c_opt[7].tdstructure

mod_target = c_opt[7].tdstructure.xtb_geom_optimization().copy()
mod_target.delete_bonds(c3d_list.deleted)
mod_target.mm_optimization("gaff")
mod_target = target.xtb_geom_optimization()

# +
frag3 = Trajectory([c_opt[7].tdstructure.xtb_geom_optimization(), mod_target])
gi_frag3 = frag3.run_geodesic(nimages=15)
c_frag3 = Chain.from_traj(gi_frag3,k=0.1,delta_k=0,node_class=Node3D,step_size=1)

tol = .03
n_frag3 = NEB(initial_chain=c_frag3,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
# -

n_frag3.optimize_chain()

foo = n_frag3.chain_trajectory[-1]

foo.plot_chain()

hmm = Trajectory([node.tdstructure for node in foo])

hmm[4].xtb_geom_optimization()

hmm.write_trajectory("../example_cases/wittig/wtf.xyz")

hmm[11].xtb_geom_optimization()

corrected_end = hmm[4].copy()
corrected_end.delete_bonds(c3d_list.deleted)
corrected_end.mm_optimization("gaff")
corrected_end.mm_optimization("uff")
corrected_end = corrected_end.xtb_geom_optimization()

# +
frag3_v2 = Trajectory([hmm[4].xtb_geom_optimization(), corrected_end])
gi_frag3_v2 = frag3_v2.run_geodesic(nimages=15)
c_frag3_v2 = Chain.from_traj(gi_frag3_v2,k=0.1,delta_k=0,node_class=Node3D,step_size=1)

tol = .03
n_frag3_v2 = NEB(initial_chain=c_frag3_v2,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
# -

n_frag3_v2.optimize_chain()

n_frag3_v2.chain_trajectory[-1].plot_chain()

c_tot = Chain(n_frag1.optimized.nodes+n_frag2.optimized.nodes+n_frag3_v2.optimized.nodes, k=0.1,step_size=1,delta_k=0)

t = Trajectory([n.tdstructure for n in c_tot])

t.write_trajectory("../example_cases/wittig/extracted_mechanism_att3.xyz")

# +
s = 8
fs = 18

f, ax = plt.subplots(figsize=(2.18*s, s))
plt.plot(c_tot.integrated_path_length, (c_tot.energies - c_tot.energies[0])*627.5, 'o--')
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
# -

# what if i try to get a TS.... for fun...
from neb_dynamics.NEB import TS_PRFO
tsopt = TS_PRFO(initial_node=c_tot[-8])

tsopt.ts

# # check out the 'true' path

mol = Molecule.from_smiles("CC(=O)C.CP(=C)(C)C")
mol.draw(mode='d3')

# +
d1 = {'charges':[],'delete':[], 'single':[(5,2),(6,1),(6,5),(2,1)]}
conds = Conditions()
rules = Rules()
cg = [
    (4,5,'Me'),
    (7,5,'Me'),
    (8,5,'Me'),
    (0,1,'Me'),
    (3,1,'Me'),
]

temp1 = ReactionTemplate.from_components(name='Wittig', reactants=mol,changes_react_to_prod_dict=d1, conditions=conds, rules=rules,collapse_groups=cg)
# temp1 = ReactionTemplate.from_components(name='Wittig', reactants=mol,changes_react_to_prod_dict=d1, conditions=conds, rules=rules)

# +
# Molecule.draw_list([temp1.reactants,temp1.products], mode='rdkit')
# -

# ### this is where making the endpoints happens

root_true = TDStructure.from_RP(temp1.reactants)

# +
forming_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,2), (6,1)]]

c3d_list = Changes3DList(deleted=[],forming=forming_list, charges=[])
# -

root_aligned = root_true.pseudoalign(c3d_list).xtb_geom_optimization()

target_aligned = root_aligned.copy()
target_aligned.add_bonds(c3d_list.forming)
target_aligned.mm_optimization('gaff')
target_aligned = target_aligned.xtb_geom_optimization()

traj_true = Trajectory([root_aligned.xtb_geom_optimization(), target_aligned.xtb_geom_optimization()])
gi_true = traj_true.run_geodesic(nimages=15, friction=.01)

chain_true = Chain.from_traj(traj=gi_true,k=0.1,delta_k=0,step_size=1,node_class=Node3D)
tol = .03
n_true = NEB(initial_chain=chain_true,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)

n_true.optimize_chain()

n_true.chain_trajectory[-1].plot_chain()

# #### continuation

target_aligned.molecule_rp.draw(mode='d3')

# +
deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,6), (2,1)]]

c3d_list = Changes3DList(deleted=deleting_list,forming=[], charges=[])
# -

final_struct = target_aligned.copy()
final_struct.delete_bonds(c3d_list.deleted)
final_struct.mm_optimization('gaff')
final_struct.mm_optimization('uff')
final_struct = final_struct.xtb_geom_optimization()

gi_true_pt2 = Trajectory([target_aligned, final_struct]).run_geodesic(nimages=15, friction=.01)

chain_true_pt2 = Chain.from_traj(traj=gi_true_pt2,k=0.1,delta_k=0,step_size=1,node_class=Node3D)
tol = .03
n_true_pt2 = NEB(initial_chain=chain_true_pt2,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)

n_true_pt2.optimize_chain()

nodes = n_true.chain_trajectory[-1].nodes.copy()

nodes.extend(n_true_pt2.chain_trajectory[-1].nodes.copy())

t_true = Trajectory([node.tdstructure for node in nodes])

# # Vis stuff

# c = Chain(nodes=nodes, k=0.1,delta_k=0,step_size=1)
c = Chain.from_xyz("../example_cases/wittig/true_mechanism_tol03_v2.xyz")
# c_tot = Chain.from_xyz("../example_cases/wittig/extracted_mechanism_att0.xyz")
c_tot = Chain.from_xyz("../example_cases/wittig/extracted_mechanism_att3.xyz")


# +
s = 8
fs = 18

f, ax = plt.subplots(figsize=(2.18*s, s))

plt.plot(c.integrated_path_length, (c.energies-c.energies[0])*627.5, 'o--', label='manual')
plt.plot(c_tot.integrated_path_length, (c_tot.energies-c.energies[0])*627.5, 'o--',label='extracted')
plt.plot(out.integrated_path_length, (out.energies-c.energies[0])*627.5, 'o--',label='extracted_auto')
# plt.plot(c_resamp.integrated_path_length, (c_resamp.energies-c.energies[0])*627.5, 'o--',label='extracted_resamp')
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.legend(fontsize=fs)
# -

t = Trajectory([n.tdstructure for n in c_tot.nodes])

t.draw();

# +


inds_min = argrelextrema(c_tot.energies, np.less, order=2)[0]
inds_min
# -

# # Automate Shit

# +
# let's automate shit
mol = Molecule.from_smiles("CC(=O)C.CP(=C)(C)C")
# mol = Molecule.from_smiles("C(=O)(C)C.C=P(c1ccccc1)(c2ccccc2)c3ccccc3")
d = {'charges':[],'delete':[(5,6),(1,2)], 'double':[(1,6),(2,5)]}
# d = {'charges':[],'delete':[(5,4),(1,0)], 'double':[(0,4),(1,5)]}
# d = {'charges':[], 'single':[(0,4),(1,5),(5,4),(0,1)]}
conds = Conditions()
rules = Rules()
cg = [
    (7,5,'Me'),
    (8,5,'Me'),
    (4,5,'Me'),
    (0,1,'Me'),
    (3,1,'Me'),
]

# cg = [
#     (18,5,'Me'),
#     (6,5,'Me'),
#     (12,5,'Me'),
#     (2,0,'Me'),
#     (3,0,'Me'),
# ]
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol,changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

# +
deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,6), (2,1)]]
forming_list = [Changes3D(start=s,end=e, bond_order=2) for s,e in [(1,6), (2,5)]]
# deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,4),(1,0)]]
# deleting_list = []
# forming_list = [Changes3D(start=s,end=e, bond_order=2) for s,e in [(0,4),(1,5)]]

c3d_list = Changes3DList(deleted=deleting_list,forming=forming_list, charges=[])
# -

root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root = root.xtb_geom_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.mm_optimization("gaff")
target = target.xtb_geom_optimization()

root

target

m = MSMEP(max_steps=2000, v=True, tol=0.01)

o = m.get_neb_chain(root, target,do_alignment=False)

o.plot_chain()

out = m.find_mep_multistep((root, target), do_alignment=True)

out.plot_chain()

(out.energies[4] - out.energies[0])*627.5

(out.energies[20] - out.energies[15])*627.5

t = Trajectory([n.tdstructure for n in out])

t.draw();

t.write_trajectory("../example_cases/wittig/auto_extracted_TPP_att0.xyz")

#









t1 = Trajectory.from_xyz("../example_cases/wittig/auto_extracted_att2.xyz")

t2 = Trajectory.from_xyz("../example_cases/wittig/auto_extracted_TPP_att0.xyz")

t1.draw();

t2.draw();

t2[0]

t2[-1]

t3 = Trajectory.from_xyz("../example_cases/claisen/cr_MSMEP_tol_01.xyz")

t3.draw();


