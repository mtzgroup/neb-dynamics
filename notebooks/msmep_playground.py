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
import networkx as nx

# reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")
# HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

# -
# # 2D Potentials

nimages = 10
np.random.seed(0)

# +
start_point = [-2.59807434, -1.499999  ]
end_point = [2.5980755 , 1.49999912]


coords = np.linspace(start_point, end_point, nimages)
coords[1:-1] += np.random.normal(scale=.1, size=coords[1:-1].shape)
# coords[1:-1] += np.random.normal(scale=.15)

ks = .01
cni = ChainInputs(
    k=ks,
    node_class=Node2D_Flower,
    delta_k=0,
    step_size=.5,
    do_parallel=False,
    use_geodesic_interpolation=False,
)
gii = GIInputs(nimages=nimages)
nbi = NEBInputs(tol=.1, v=1, max_steps=1000, climb=False)
chain = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni)
m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii)
h_root_node, out_chain = m.find_mep_multistep(input_chain=chain)
# -

h_root_node.draw()

f = plot_ethan(out_chain)
plt.plot(h_root_node.data.chain_trajectory[-1].coordinates[:,0],h_root_node.data.chain_trajectory[-1].coordinates[:,1],
           c='white',linestyle='--',marker='o')

# # Some BS

c = Chain.from_xyz("/home/jdep/neb_dynamics/example_mep.xyz")

from scipy.signal import argrelextrema
import numpy as np
from retropaths.helper_functions import pairwise

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


mol = Molecule.from_smiles(smi)

# -

mol.draw(mode='d3',size=(700,700))

# + endofcell="--"
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

# +
from retropaths.molecules.molecule import Molecule
from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory


from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Inputs import NEBInputs, ChainInputs
from neb_dynamics.HistoryTree import HistoryTree
from neb_dynamics.TreeNode import TreeNode

from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules

# +
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

# + endofcell="--"
conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
# -

root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root.gum_mm_optimization()
root = root.xtb_geom_optimization()
# --

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.gum_mm_optimization()
target = target.xtb_geom_optimization()

cni = ChainInputs()
nbi = NEBInputs(v=True)

m = MSMEP(neb_inputs=nbi, recycle_chain=False)

tr = Trajectory([root, target]).run_geodesic(nimages=15, sweep=False)
start_chain = Chain.from_traj(tr,parameters=cni)

m2 = MSMEP(neb_inputs=nbi, recycle_chain=True)

history, out_chain = m2.find_mep_multistep(start_chain,do_alignment=False)

# +
# history2, out_chain2 = m.find_mep_multistep(start_chain,do_alignment=False)
# -

ht = HistoryTree.from_history_list(history)


# ? ht.write_to_disk

def write_to_disk_neb(self, fp: Path, write_history=False):
        out_traj = self.chain_trajectory[-1].to_trajectory()
        out_traj.write_trajectory(fp)

        if write_history:
            out_folder = fp.resolve().parent / (fp.stem + "_history")
            if not out_folder.exists():
                out_folder.mkdir()

            for i, chain in enumerate(self.chain_trajectory):
                traj = chain.to_trajectory()
                traj.write_trajectory(out_folder / f"traj_{i}.xyz")


def write_to_disk_2(self, folder_name: Path):
    if not folder_name.exists():
        folder_name.mkdir()

    for i, node in enumerate(self.depth_first_ordered_nodes):
        write_to_disk_neb(node.data,
            fp=folder_name / f"node_{i}.xyz", write_history=True
        )

    np.savetxt(fname=folder_name / "adj_matrix.txt", X=self.adj_matrix)


ht.root.children[1].data.optimized.plot_chain()

write_to_disk_2(ht, fn)

out_chain.plot_chain()

ht.root.data.optimized.plot_chain()

ht.write_to_disk(folder_name=Path("/home/jdep/T3D_data/msmep_draft/history_tree"))



write_to_disk(ht.root.data, fp=Path("/home/jdep/T3D_data/msmep_draft/history_tree_root_neb.xyz"), write_history=True)



fp = Path("./hey.xyz")

fp.resolve().parent

ht = HistoryTree.from_history_list(history)

ht2 = HistoryTree.from_history_list(history2)

plot_opt_history(ht.root.children[1].data)


def plot_opt_history(obj: NEB):
    
    s = 8
    fs = 18
    f, ax = plt.subplots(figsize=(1.16 * s, s))
    
    
    for i, chain in enumerate(obj.chain_trajectory):
        if i == len(obj.chain_trajectory)-1:
            plt.plot(chain.integrated_path_length, chain.energies, 'o-',alpha=1)
        else:
            plt.plot(chain.integrated_path_length, chain.energies, 'o-',alpha=.1,color='gray')
            
    
    plt.xlabel("Integrated path length", fontsize=fs)
            
    plt.ylabel("Energy (kcal/mol)", fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.show()


plot_opt_history(history2[0])

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

m = MSMEP(max_steps=2000,v=True,tol=0.01, nudge=0)
rn = "Beckmann-Rearrangement"
root2, target2 = m.create_endpoints_from_rxn_name(rn, reactions)

# +
# %%time

input_chain = Chain(nodes=[Node3D(root2),Node3D(target2)],k=0.1)
n_obj2, out_chain2 = m.find_mep_multistep(input_chain, do_alignment=True)
# -

out_chain2.plot_chain()

t = Trajectory([n.tdstructure for n in out_chain2.nodes])

t.draw()

# mkdir ../example_cases/beckmann_rearrangement

t.write_trajectory("../example_cases/beckmann_rearrangement/msmep_tol001_v2.xyz")

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

# +
c3d_list = root.get_changes_in_3d(reactions["Ugi-Reaction"])

root = root.pseudoalign(c3d_list)
# -

target = root.copy()
target.apply_changed3d_list(c3d_list)
target.gum_mm_optimization()

root

root_opt  = root.xtb_geom_optimization()
target_opt = target.xtb_geom_optimization()

root_opt

reference_ugi = Trajectory.from_xyz("/home/jdep/neb_dynamics/example_cases/ugi/msmep_tol0.01_max_2000.xyz")

root_opt = reference_ugi[0]
target_opt = reference_ugi[-1]

gi = Trajectory([root_opt,target_opt]).run_geodesic(nimages=15,friction=1,sweep=False)

gi.draw();

gi_01 = Trajectory([root_opt,target_opt]).run_geodesic(nimages=15,friction=.1,sweep=False)
gi_001 = Trajectory([root_opt,target_opt]).run_geodesic(nimages=15,friction=.01,sweep=False)

import matplotlib.pyplot as plt

plt.plot(gi.energies_xtb(), 'o-',label='friction=1')
plt.plot(gi_01.energies_xtb(), 'o-',label='friction=0.1')
plt.plot(gi_001.energies_xtb(), 'o-',label='friction=0.01')
plt.legend()

cni = ChainInputs(k=0.01,step_size=1)
m = MSMEP(neb_inputs=nbi, recycle_chain=False, gi_inputs=gii)
chain = Chain.from_traj(gi_001,cni)

out_chain.plot_chain()

out_chain.to_trajectory().draw();

history, out_chain = m.find_mep_multistep(chain,do_alignment=False)

from neb_dynamics.HistoryTree import HistoryTree

ht = HistoryTree.from_history_list(history)

from pathlib import Path


def draw(self):
    foo = self.adj_matrix - np.identity(len(self.adj_matrix))
    g = nx.from_numpy_matrix(foo)
    nx.draw_networkx(g)


import networkx as nx
draw(ht)

type(ht)

out_chain.to_trajectory().draw();

out_chain.to_trajectory().write_trajectory(("/home/jdep/T3D_data/msmep_draft/ugi_tree_canonical/out_chain.xyz"))

ht.write_to_disk(Path("/home/jdep/T3D_data/msmep_draft/ugi_tree_canonical"))

# # Claisen-Rearrangement

# t = Trajectory.from_xyz("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_0-0_0_cneb.xyz")
t = Trajectory.from_xyz("/home/jdep/T3D_data/msmep_draft/claisen-0-0_with_conf_rearr.xyz")

start = t[0]
p = t[15]
pprime = t[-1]

start.tc_model_basis = "gfn2xtb"
start.tc_model_method = 'gfn2xtb'

start

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

m = MSMEP(neb_inputs=nbi,recycle_chain=False)

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


