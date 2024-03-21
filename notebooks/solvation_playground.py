# +
from neb_dynamics.Chain import Chain     
from neb_dynamics.nodes.Node3D import Node3D
from neb_dynamics.NEB import NEB
import retropaths.helper_functions as hf
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.molecules.molecule import Molecule

import numpy as np
import matplotlib.pyplot as plt


from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
from pathlib import Path

from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

from retropaths.abinitio.solvator import Solvator



n_waters = 0
smi = "[C](C)(C)(C)Cl"
# smi = "[C](C)(H)([H])F"
# smi+= ".O"*n_waters

td = TDStructure.from_smiles(smi)

solv = Solvator(n_solvent=5)

td_solv = solv.solvate_td(td)

from neb_dynamics.TreeNode import TreeNode

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Meyer-Schuster-Rearrangement/production_vpo_tjm_xtb_preopt_msmep/")

init_c = h.ordered_leaves[1].data.initial_chain

raw_r = TDStructure.from_coords_symbols(init_c[0].tdstructure.coords, init_c[0].tdstructure.symbols)

raw_p = TDStructure.from_coords_symbols(init_c[-1].tdstructure.coords, init_c[-1].tdstructure.symbols)

solv_r = solv.solvate_single_td(raw_r)

solv_p = solv.solvate_td(raw_p)

tr = Trajectory([solv_r, solv_p]).run_geodesic(nimages=12)

from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer

chain = Chain.from_traj(tr, ChainInputs(k=0.1, delta_k=0.09, node_freezing=True))

opt = VelocityProjectedOptimizer()

m = MSMEP(neb_inputs=NEBInputs(v=1,early_stop_force_thre=0.03, max_steps=500), 
          chain_inputs=ChainInputs(k=0.1, delta_k=0.09, node_freezing=True),gi_inputs=GIInputs(nimages=12), optimizer=opt)

h_solv, out_solv = m.find_mep_multistep(chain)

# +
mol = Molecule.from_smiles(smi)
d = {'charges': [(0,1),(4,-1)], 'delete':[(0, 4)]}

cg = []
deleting_list = [Changes3D(start=s, end=e, bond_order=1) for s, e in d["delete"]]
forming_list = []
# -

mol.draw(mode='d3', size=(800,800))

# +
conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

# c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
c3d_list = Changes3DList(deleted=forming_list, forming=deleting_list, charges=[])
# -

root = TDStructure.from_RP(temp.products)
root = root.pseudoalign(c3d_list)
root.mm_optimization("gaff", steps=5000)
root.mm_optimization("uff", steps=5000)
root.mm_optimization("mmff94", steps=5000)
root = root.xtb_geom_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.mm_optimization("gaff", steps=5000)
target.mm_optimization("uff", steps=5000)
target.mm_optimization("mmff94", steps=5000)
target = target.xtb_geom_optimization()

# # Now interpolate

pair = Trajectory([root, target])
traj = pair.run_geodesic(nimages=15)

plt.plot(traj.energies_xtb(),'o--')

chain = Chain.from_traj(traj, k=0.1,delta_k=0, step_size=0.33,node_class=Node3D)

tol = 0.01
n = NEB(initial_chain=chain,grad_thre=tol, en_thre=tol/450, rms_grad_thre=tol*(2/3), climb=False, vv_force_thre=0, max_steps=2000)
# n = NEB(initial_chain=chain,grad_thre=tol, en_thre=tol, rms_grad_thre=tol, climb=False, vv_force_thre=0, max_steps=2000)

n.optimize_chain()

n.optimized.plot_chain()

t = Trajectory([x.tdstructure for x in n.optimized])

t.draw()

m = MSMEP(tol=0.01,v=True, max_steps=2000)

start,end = t[0].xtb_geom_optimization(), t[5].xtb_geom_optimization()

output = m.find_mep_multistep((start,end),do_alignment=False)


