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


# +
def plot_node(node,linestyle='--',marker='o',ax=None,**kwds):
    plot_chain(chain=node.data.chain_trajectory[-1],linestyle=linestyle,marker=marker,ax=ax,**kwds)

def plot_chain(chain,linestyle='--',ax=None, marker='o',**kwds):
    if ax:
        ax.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)
    else:
        plt.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)


def plot_node2d(node,linestyle='--',marker='o',ax=None,start_point=0,end_point=1,
                **kwds):
    plot_chain2d(chain=node.data.chain_trajectory[-1],linestyle=linestyle,marker=marker,ax=ax,start_point=start_point,end_point=end_point,
                 **kwds)

def plot_chain2d(chain,linestyle='--',marker='o',ax=None,start_point=0,end_point=1,
                 **kwds):
    if ax:
        ax.plot((chain.integrated_path_length*end_point)+start_point,chain.energies,linestyle=linestyle,marker=marker,**kwds)
    else:
        plt.plot((chain.integrated_path_length*end_point)+start_point,chain.energies,linestyle=linestyle,marker=marker,**kwds)


""
# +
nimages = 10
np.random.seed(0)

start_point = [-2.59807434, -1.499999  ]
end_point = [2.5980755 , 1.49999912]


coords = np.linspace(start_point, end_point, nimages)
coords[1:-1] += [-1,1] # i.e. good initial guess
# coords[1:-1] += [-.05,.05] # i.e. bad initial guess
# coords[1:-1] -= np.random.normal(scale=.1, size=coords[1:-1].shape)
# coords[1:-1] -= 0.1
# coords[1:-1] -= 1
# coords[1:-1] += np.random.normal(scale=.15)

ks = .1
cni = ChainInputs(
    k=ks,
    node_class=Node2D_Flower,
    delta_k=0,
    step_size=.3,
    do_parallel=False,
    use_geodesic_interpolation=False,
)
gii = GIInputs(nimages=nimages)
nbi = NEBInputs(tol=.1, v=1, max_steps=500, climb=False, stopping_threshold=0)
chain = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni)

# +

n = NEB(initial_chain=chain,parameters=nbi)
n.optimize_chain()
# -

""
m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii)
h_root_node, out_chain = m.find_mep_multistep(input_chain=chain)

h_root_node.draw()


""
def plot_node_recursively(node, ax):
    if node.is_leaf:
        plot_chain(node.data.initial_chain, label=f'node{node.index} guess',linestyle='--', marker='x',ax=ax) 
        plot_node(node, label=f'node{node.index} neb',linestyle='-',ax=ax)
    else:
        plot_chain(node.data.initial_chain, label=f'node{node.index} guess',linestyle='--', marker='x',ax=ax) 
        plot_node(node, label=f'node{node.index} neb',linestyle='-',ax=ax)
        [plot_node_recursively(child,ax=ax) for child in node.children]


opt_hists = [h_root_node.data.initial_chain.coordinates]
for opt_hist in h_root_node.get_optimization_history():
    chain_traj_matrix = np.array([c.coordinates for c in opt_hist.chain_trajectory])
    opt_hists.extend(chain_traj_matrix)
# [animate_func(obj) for obj in h_root_node.get_optimization_history()]
output_matrix = np.array(opt_hists)
# print(output_matrix.shape)
# np.savetxt(fname="msmep_traj_flat.txt",X=output_matrix.flatten())

heyo = neb_long.chain_trajectory[-1][5].tdstructure.xtb_geom_optimization()

heyo.energy_xtb()

out_chain[0].energy

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18

f, ax = plt.subplots(figsize=(2.3 * fig, fig),ncols=2)
x = np.linspace(start=min_val, stop=max_val, num=1000)
y = x.reshape(-1, 1)

h = Node2D_Flower.en_func_arr([x, y])
cs = ax[0].contourf(x, x, h, cmap="Greys",alpha=.8)
_ = f.colorbar(cs,ax=ax[0])

# plot_chain(n.initial_chain,ax=ax[0], label='initial guess', linestyle='--')
# plot_chain(n.chain_trajectory[-1],ax=ax[0], label='neb', linestyle='-')

a = .1
b = 1
plot_node_recursively(h_root_node,ax=ax[0])

plot_chain(chain, label='initial guess',marker='x',color='blue',ax=ax[0],alpha=a)
plot_chain2d(chain, label='initial guess',marker='x',color='blue',ax=ax[1],alpha=a)

plot_chain(h_root_node.data.chain_trajectory[-1], label='root neb', marker='o',linestyle='-',color='blue',ax=ax[0],alpha=a)
plot_chain2d(h_root_node.data.chain_trajectory[-1], label='root neb', marker='o',linestyle='-',color='blue',ax=ax[1],alpha=a)

plot_chain(h_root_node.children[0].data.initial_chain, c='red', label='foo1 guess',linestyle='--', marker='x',ax=ax[0],alpha=a) 
plot_chain2d(h_root_node.children[0].data.initial_chain, c='red', label='foo1 guess',linestyle='--', marker='x',ax=ax[1],alpha=a,
            end_point=.61) 

plot_node(h_root_node.children[0], c='red', label='foo1 neb',ax=ax[0],linestyle='-',alpha=a) 
plot_node2d(h_root_node.children[0], c='red', label='foo1 neb',ax=ax[1],linestyle='-',alpha=a,
            end_point=.61) 



plot_chain(h_root_node.ordered_leaves[0].data.initial_chain, c='green', label='leaf1 guess',linestyle='--', marker='x',ax=ax[0],alpha=a) 
plot_chain2d(h_root_node.ordered_leaves[0].data.initial_chain, c='green', label='leaf1 guess',linestyle='--', marker='x',ax=ax[1],alpha=a,
            end_point=.25) 


plot_node(h_root_node.ordered_leaves[0], c='green', label='leaf1 neb',ax=ax[0],alpha=b) 
plot_node2d(h_root_node.ordered_leaves[0], c='green', linestyle='-',label='leaf1 neb',ax=ax[1],end_point=.25,alpha=b) 

plot_chain(h_root_node.ordered_leaves[1].data.initial_chain, c='purple', label='leaf2 guess',linestyle='--', marker='x',ax=ax[0],alpha=a) 
plot_chain2d(h_root_node.ordered_leaves[1].data.initial_chain, c='purple', label='leaf2 guess',linestyle='--', marker='x',ax=ax[1],alpha=a,
            start_point=.25,end_point=.36) 

plot_node(h_root_node.ordered_leaves[1], c='purple', label='leaf2 guess',linestyle='-', marker='o',ax=ax[0],alpha=b) 
plot_node2d(h_root_node.ordered_leaves[1], c='purple', label='leaf2 guess',linestyle='-', marker='o',ax=ax[1],alpha=b,
           start_point=.25,end_point=.36) 

plot_chain(h_root_node.ordered_leaves[2].data.initial_chain, c='darkorange', label='leaf2 guess',linestyle='--', marker='x',ax=ax[0],alpha=a) 
plot_chain2d(h_root_node.ordered_leaves[2].data.initial_chain, c='darkorange', label='leaf2 guess',linestyle='--', marker='x',ax=ax[1],alpha=a,
            start_point=.61, end_point=.4) 


plot_node(h_root_node.ordered_leaves[2], c='darkorange', label='leaf2 guess',linestyle='-', marker='o',ax=ax[0],alpha=b) 
plot_node2d(h_root_node.ordered_leaves[2], c='darkorange', label='leaf2 guess',linestyle='-', marker='o',ax=ax[1],alpha=b,
           start_point=.61, end_point=.4) 

# plot_node(h_root_node.ordered_leaves[2], c='orange', label='leaf3 neb') 

plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
# legend = ax[0].legend(fontsize=fs, bbox_to_anchor=(1.70,1.03))
# legend = ax[0].legend(fontsize=fs)

# frame = legend.get_frame()
# frame.set_color('gray')

# plt.savefig("/home/jdep/T3D_data/msmep_draft/flower_potential.svg",format='svg',bbox_inches='tight')
plt.tight_layout()
plt.show()
# +
# ################################################################################
# # #### Some BS
# +
c = Chain.from_xyz("/home/jdep/neb_dynamics/example_mep.xyz")

""
from scipy.signal import argrelextrema
import numpy as np
from retropaths.helper_functions import pairwise

""
# +
n_waters = 0
# smi = "[C]([H])([H])[H]"
# smi = "[C+](C)(C)C"
smi = "C12(C(=C)O[Si](C)(C)C)C(=O)OC3CCC1C23C4=CCCCC4"
# smi = "[Rh@SP1H](C#[O])(C#[O])C#[O]"
# smi = "[C]([H])([H])[H].[Cl]"
# smi = "[O-][H].O.O.O.O.[C+](C)(C)C"
# smi = "[O-][H]"
smi+= ".O"*n_waters
smi_ref = "O"*n_waters
# orig_smi = "[C]([H])([H])[H].[O-][H]"
# orig_smi = "[C+](C)(C)C.[O-][H]"

""
mol = Molecule.from_smiles(smi)

""
# -

""
mol.draw(mode='d3',size=(700,700))

""
ind=0

single_list = [(1,2),(2,17), (17,16),(16,0)]
double_list = [(15,14),(0,1)]
delete_list = [(0, 15),(0,14)]

forming_list = [Changes3D(start=s, end=e, bond_order=1) for s, e in single_list]
forming_list+= [Changes3D(start=s, end=e, bond_order=2) for s, e in double_list]

settings = [

    (
        mol,
        {'charges': [], 'delete':delete_list, 'single':single_list,"double":double_list},
        [],
        [Changes3D(start=s, end=e, bond_order=1) for s, e in delete_list], # deleting list
        forming_list

    )
]

mol, d, cg, deleting_list, forming_list = settings[ind]


conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
# -
# --

# +
# ind=0

# single_list = [(2,17),(2,1),(17,16)]
# double_list = [(15,16), (0,1)]
# delete_list = [(0, 15)]

# forming_list = [Changes3D(start=s, end=e, bond_order=1) for s, e in single_list]
# forming_list+= [Changes3D(start=s, end=e, bond_order=2) for s, e in double_list]

# settings = [

#     (
#         mol,
#         {'charges': [], 'delete':delete_list, 'single':single_list,"double":double_list},
#         [],
#         [Changes3D(start=s, end=e, bond_order=1) for s, e in delete_list], # deleting list
#         forming_list

#     )
# ]

# mol, d, cg, deleting_list, forming_list = settings[ind]


# conds = Conditions()
# rules = Rules()
# temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

# c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
# # -
# -

root = TDStructure.from_smiles(smi,tot_spinmult=1)
# root = root.pseudoalign(c3d_list)
# root = root.xtb_geom_optimization()

root.molecule_rp.draw(mode='d3')

temp.reactants.draw()

temp.products.draw(mode='rdkit')

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.mm_optimization('gaff')
target.mm_optimization("uff")
target.mm_optimization('mmff94')
# target = target.xtb_geom_optimization()

target = target.xtb_geom_optimization()

target

output = m.find_mep_multistep((root, target), do_alignment=False)

# # Wittig

cni = ChainInputs()
start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess.xyz", parameters=cni)
# tr.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess.xyz")

# +
# history, out_chain = m.find_mep_multistep(start_chain)

# +
# cleanup_nebs = cleanup_nebs(start_chain, history, m)

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
m  = MSMEP(neb_inputs=nbi, root_early_stopping=True)

insertion_points = m._get_insertion_points_leaves(history.ordered_leaves,original_start=start_chain[0])

insertion_points

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
plt.plot(neb_long.optimized.integrated_path_length, (neb_long.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (45 nodes)")
plt.plot(neb_short.optimized.integrated_path_length, (neb_short.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (15 nodes)")
plt.plot(clean_out_chain.integrated_path_length, (clean_out_chain.energies-clean_out_chain.energies[0])*627.5,'o-',label="AS-NEB")
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
# -

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

root.tc_model_basis = '6-31gs'
root.tc_model_method = 'b3lyp'

root = root.xtb_geom_optimization()
root = root.tc_geom_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.gum_mm_optimization()
target = target.xtb_geom_optimization()
target = target.tc_geom_optimization()

tr = Trajectory([root, target]).run_geodesic(nimages=15, sweep=False)
# --

# +
# tr.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_b3lyp.xyz")
# -

# cni = ChainInputs()
# start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_b3lyp.xyz", parameters=cni)


cni = ChainInputs(node_class=Node3D_TC)
# start_chain = Chain.from_traj(tr,parameters=cni)
start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_b3lyp.xyz", parameters=cni)

nbi = NEBInputs(v=True, stopping_threshold=3, tol=0.01)
m  = MSMEP(neb_inputs=nbi, root_early_stopping=True, chain_inputs=cni, gi_inputs=GIInputs())



history, out_chain = m.find_mep_multistep(start_chain)

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

