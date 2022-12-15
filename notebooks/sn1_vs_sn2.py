# +
from neb_dynamics.MSMEP import MSMEP                                                                                         
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

# +
# reactions = hf.pload("/Users/janestrada/Documents/Stanford/Retropaths/retropaths/data/reactions.p")

# +
n_waters = 20

smi = "[C](C)(H)([H])F"
smi+= ".O"*n_waters

# +
mol = Molecule.from_smiles(smi)
d = {'charges': [(0,1),(4,-1)], 'delete':[(0, 4)]}

cg = []
deleting_list = [Changes3D(start=s, end=e, bond_order=1) for s, e in [(0,4)]]
forming_list = []


# -


mol.draw(mode='d3')

# +
conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
# -

root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root = root.xtb_geom_optimization()

root.to_xyz("./cme1f_solv.xyz")

root

# +
target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.mm_optimization("gaff", steps=5000)
target.mm_optimization("uff", steps=5000)
target.mm_optimization("mmff94", steps=5000)

target = target.xtb_geom_optimization()
# -

target

# +
# target.to_xyz("./cm2_f_solvated_raw.xyz")

# +
# fp = Path("./cm3_oh_solvated_raw.xyz")
# target = TDStructure.from_fp(fp)

# +
# target

# +
# target = target.xtb_geom_optimization()

# +
# target.to_xyz("./cm3_oh_solvated_opt.xyz")
# -

# # Endpoints have been created. Now we can do some numbers

pair = Trajectory([root, target])

traj = pair.run_geodesic(nimages=15)

ens = traj.energies

traj.write_trajectory("./cm2f_dissociation_geodesic.xyz")

# +
gi = Trajectory.from_xyz("./cm3f_dissociation_geodesic.xyz")
gi2 = Trajectory.from_xyz("./cm2f_dissociation_geodesic.xyz")

neb = Trajectory.from_xyz("./cm3f_dissociation_geodesic_cneb.xyz")
neb2 = Trajectory.from_xyz("./cm2f_dissociation_geodesic_cneb.xyz")
# -

gi.energies_xtb()

# +
s = 10
fs = 18
f,ax = plt.subplots(figsize=(2*s, s))
plt.plot(gi.energies_xtb(),'--',label='geodesic (tert)',c='blue')
plt.plot(gi2.energies_xtb(),'--',label='geodesic (secnd)',c='red')

plt.plot(neb.energies_xtb(),'o-',label='neb (tert)',c='blue')
plt.plot(neb2.energies_xtb(),'o-',label='neb (secnd)',c='red')
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.show()
# -

neb[-1].molecule_rp.smiles

# foo = Molecule.from_smiles('C[CH+]C.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.[F-]')
foo = Molecule.from_smiles('C[C+](C)C.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.[F-]')

foo.draw()

traj.write_trajectory("./cm3f_dissociation_geodesic.xyz")
