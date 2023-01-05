# +
from neb_dynamics.MSMEP import MSMEP
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

reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

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

mol.draw(mode='d3',size=(400,400))

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

m = MSMEP(v=True,tol=0.01)

output = m.find_mep_multistep((root, target), do_alignment=False)

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


