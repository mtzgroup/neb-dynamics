# +
# %%time
from retropaths.molecules.molecule import Molecule
from IPython.core.display import HTML
from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.abinitio.tdstructure import TDStructure

from neb_dynamics.MSMEP import MSMEP
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
import retropaths.helper_functions as hf
from pathlib import Path
import numpy as np

HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

tg = Path('target.xyz')
rt = Path('root.xyz')
if not tg.is_file() or False:
    mol = Molecule.from_smiles("C(=O)(C)C.C=P(c1ccccc1)(c2ccccc2)c3ccccc3")
    d = {'charges':[], 'single':[(0,4),(1,5),(5,4),(0,1)]}
    cg = [(18,5,'Me'),(6,5,'Me'),(12,5,'Me'),(2,0,'Me'),(3,0,'Me')]
    deleting_list = []
    forming_list = [Changes3D(start=s,end=e, bond_order=2) for s,e in [(0,4),(1,5)]]    
    
#     mol = Molecule.from_smiles("CC(=O)C.CP(=C)(C)C")
#     d = {"charges": [], "delete": [(5, 6), (1, 2)], "double": [(1, 6), (2, 5)]}
#     cg = [(7, 5, "Me"), (8, 5, "Me"), (4, 5, "Me"), (0, 1, "Me"), (3, 1, "Me"),]
#     deleting_list = [Changes3D(start=s, end=e, bond_order=1) for s, e in [(5, 6), (2, 1)]]
#     forming_list = [Changes3D(start=s, end=e, bond_order=2) for s, e in [(1, 6), (2, 5)]]
    
    conds = Conditions()
    rules = Rules()
    temp = ReactionTemplate.from_components(name="Wittig",reactants=mol,changes_react_to_prod_dict=d,conditions=conds,rules=rules,collapse_groups=cg,)
    c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
    root = TDStructure.from_RP(temp.reactants)
    root = root.pseudoalign(c3d_list)
    root = root.xtb_geom_optimization()
    target = root.copy()
    target.add_bonds(c3d_list.forming)
    target.delete_bonds(c3d_list.deleted)
    target.mm_optimization("gaff")
    target = target.xtb_geom_optimization()
    target.to_xyz(tg)
    root.to_xyz(rt)
else:
    target = TDStructure.from_xyz(tg.name)
    root = TDStructure.from_xyz(rt.name)

m = MSMEP(max_steps=2000, v=True, tol=0.01)
# -

# %%time
o = m.get_neb_chain(root, target, do_alignment=False)

# +
fn = Path('test.p')
if fn.is_file():
    test = hf.pload(fn)
else:
    test = o[5].coords
    hf.psave(test, fn)

np.all(np.isclose(o[5].coords, test, atol=0.0001))
# -


