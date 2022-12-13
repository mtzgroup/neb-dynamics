# +
from neb_dynamics.MSMEP import MSMEP
import retropaths.helper_functions as hf
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.molecules.molecule import Molecule


from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
from pathlib import Path

reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")

# +
n_waters = 10
# smi = "[C]([H])([H])[H]"
smi = "[C+](C)(C)C"
# smi = "[C]([H])([H])[H].[Cl]"
# smi = "[O-][H].O.O.O.O.[C+](C)(C)C"
# smi = "[O-][H]"
smi+= ".O"*n_waters
smi_ref = "O"*n_waters
# orig_smi = "[C]([H])([H])[H].[O-][H]"
orig_smi = "[C+](C)(C)C.[O-][H]"


mol = Molecule.from_smiles(smi)

# -

mol.draw()

td = TDStructure.from_smiles(smi)

td

td.xtb_geom_optimization()

td.to_xyz("cme3_and_waters.xyz")

# fp = Path("/home/jdep/ch3_and_waters_and_oh.xyz")
td2 = TDStructure.from_fp(fp,tot_charge=-1)

td2.xtb_geom_optimization()

ref = TDStructure.from_smiles(smi_ref)
orig = TDStructure.from_smiles(orig_smi)

td2.energy_xtb() - ref.energy_xtb()

orig

orig.energy_xtb()


def calc_stabilization(n_waters):
    smi = "[O-][H]"
    smi+= ".O"*n_waters
    smi_ref = "O"*n_waters

    
    td = TDStructure.from_smiles(smi, tot_spinmult=1)
    td_ref = TDStructure.from_smiles(smi_ref, tot_spinmult=1)
    if not td.energy_xtb() : return np.nan, None
    
    init_en = td.energy_xtb() - td_ref.energy_xtb()

    td = td.xtb_geom_optimization()
    td_ref = td_ref.xtb_geom_optimization()
    
    if not td.energy_xtb() : return np.nan, None
    
    final_en = td.energy_xtb() - td_ref.energy_xtb()
    stabilization = final_en - init_en
    return stabilization, td

import numpy as np

# +
n_vals = [1,2,3,4,5,6,7,8,9,10]
n_iter = 5
tds = []
mean_s_vals = []
std_s_vals = []
for n_wat in n_vals:
    s_vals = []
    for x in range(n_iter):
        s,t = calc_stabilization(n_wat)
        s_vals.append(s)
        tds.append(t)
    mean_s_vals.append(np.mean(s_vals))
    std_s_vals.append(np.std(s_vals))
    
    
    
# -

import matplotlib.pyplot as plt

# +

plt.errorbar(x=n_vals, y=mean_s_vals, yerr=std_s_vals, fmt='o')
# -

td = TDStructure.from_smiles(smi, tot_spinmult=1)
td_ref = TDStructure.from_smiles(smi_ref, tot_spinmult=1)

init_en = td.energy_xtb() - td_ref.energy_xtb()

td = td.xtb_geom_optimization()
td_ref = td_ref.xtb_geom_optimization()

final_en = td.energy_xtb() - td_ref.energy_xtb()
stabilization = final_en - init_en

# +
# for i in mol.nodes:
#     print(i, mol.nodes[i])

# +
# ind=2
# settings = [

#     (
#         Molecule.from_smiles("C.Cl-Cl"),
#         {'charges': [], 'delete':[(0, 3), (1, 2)], 'single':[(0, 1), (2, 3)]},
#         [],
#         [Changes3D(start=s, end=e, bond_order=1) for s, e in [(0, 3), (1, 2)]],
#         [Changes3D(start=s, end=e, bond_order=2) for s, e in [(0, 1), (2, 3)]]

#     ),
#     (
#         Molecule.from_smiles("[C]([H])([H])[H].[Cl]"),
#         {'charges': [], 'delete':[], 'single':[(0, 1)]},
#         [],
#         [],
#         [Changes3D(start=s, end=e, bond_order=1) for s, e in [(0, 1)]]

#     ),
#     (
#         Molecule.from_smiles("Cl-Cl"),
#         {'charges': [], 'single':[], 'delete':[(0, 1)]},
#         [],
#         [Changes3D(start=s, end=e, bond_order=1) for s, e in [(0, 1)]],
#         []
        

#     )
# ]

# mol, d, cg, deleting_list, forming_list = settings[ind]


# conds = Conditions()
# rules = Rules()
# temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

# c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
# # -

# root = TDStructure.from_smiles(temp.reactants.smiles,tot_spinmult=1)
# # root = root.pseudoalign(c3d_list)
# root = root.xtb_geom_optimization()

# +
# target = root.copy()
# target.set_spinmult(3)
# target.add_bonds(c3d_list.forming)
# target.delete_bonds(c3d_list.deleted)
# target.mm_optimization("gaff", steps=5000)
# target.mm_optimization("uff", steps=5000)
# target.mm_optimization("mmff94", steps=5000)
# target = target.xtb_geom_optimization()

# +
# m = MSMEP(max_steps=2000,v=True,tol=0.01, nudge=0,step_size=1)
# result = m.find_mep_multistep((root, target),do_alignment=False)
# -

n, chain = result

chain.plot_chain()

chain.to_trajectory()

# # Diels Alder

root, target = m.create_endpoints_from_rxn_name("Diels Alder 4+2", reactions)

# %%time
n_obj, out_chain = m.find_mep_multistep((root, target), do_alignment=True)

out_chain.plot_chain() # tol was probably too high

t = Trajectory([n.tdstructure for n in out_chain.nodes])

t.draw()

t.write_trajectory("../example_cases/diels_alder/msmep_tol001.xyz")

# # Beckmann Rearrangement

m = MSMEP(max_steps=2000,v=True,tol=0.01, nudge=0)
rn = "Beckmann-Rearrangement"
root2, target2 = m.create_endpoints_from_rxn_name(rn, reactions)

root2

# %%time
n_obj2, out_chain2 = m.find_mep_multistep((root2, target2), do_alignment=True)

t = Trajectory([n.tdstructure for n in out_chain2.nodes])

# mkdir ../example_cases/beckmann_rearrangement

t.write_trajectory("../example_cases/beckmann_rearrangement/msmep_tol001.xyz")

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


