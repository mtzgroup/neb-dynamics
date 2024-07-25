# +
from retropaths.molecules.molecule import Molecule
from IPython.core.display import HTML
from retropaths.reactions.changes import Changes3DList, Changes3D
# from retropaths.abinitio.tdstructure import TDStructure
from neb_dynamics.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB, NoneConvergedException
from chain import Chain
from nodes.node3d import Node3D

from neb_dynamics.MSMEP import MSMEP
import matplotlib.pyplot as plt
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
import numpy as np
from scipy.signal import argrelextrema
from retropaths.helper_functions import pairwise
import retropaths.helper_functions as hf

HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

mol = Molecule.from_smiles("CC(C)=O.CC(=O)OO[H]")
mol.draw(mode='d3',size=(500,500))



# +
d1 = {'charges':[],'delete':[(7,8),(8,9),(2,1)], 'single':[(8,1),(9,7),(2,8)]}
conds = Conditions()
rules = Rules()
cg = []

temp1 = ReactionTemplate.from_components(name='TODD', reactants=mol,changes_react_to_prod_dict=d1, conditions=conds, rules=rules,collapse_groups=cg)
# temp1 = ReactionTemplate.from_components(name='Wittig', reactants=mol,changes_react_to_prod_dict=d1, conditions=conds, rules=rules)
# -

temp1.draw(mode='d3',size=(400,400))

from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Inputs import NEBInputs, GIInputs, ChainInputs
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

root = TDStructure.from_RP(temp1.reactants, charge=temp1.reactants.charge)

root

# +
deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(7,8),(8,9),(2,1)]]
forming_list = [Changes3D(start=s,end=e, bond_order=2) for s,e in [(8,1),(9,7),(2,8)]]

c3d_list = Changes3DList(deleted=deleting_list,forming=forming_list, charges=[])
# -

root = root.pseudoalign(c3d_list)

root

root_opt = root.xtb_geom_optimization()

root_opt

target = root_opt.copy()

target.add_bonds(c3d_list.forming)

target.delete_bonds(c3d_list.deleted)

target.gum_mm_optimization()

target_opt = target.xtb_geom_optimization()

# +
from retropaths.molecules.molecule import Molecule
from IPython.core.display import HTML
from retropaths.reactions.changes import Changes3DList, Changes3D
# from retropaths.abinitio.tdstructure import TDStructure
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.NEB import NEB, NoneConvergedException
from chain import Chain
from nodes.node3d import Node3D

from neb_dynamics.MSMEP import MSMEP
import matplotlib.pyplot as plt
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
import numpy as np
from scipy.signal import argrelextrema
from retropaths.helper_functions import pairwise
import retropaths.helper_functions as hf
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Inputs import NEBInputs, GIInputs, ChainInputs
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
# -

nbi = NEBInputs(v=True)
cni = ChainInputs(k=0.1,delta_k=0.09, use_maxima_recyling=True, node_freezing=True)
gii = GIInputs(nimages=15)
m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=VelocityProjectedOptimizer())

# traj = Trajectory([root_opt, target_opt]).run_geodesic(nimages=15)
root = TDStructure.from_xyz("/home/jdep/T3D_data/template_rxns/Baeyer-Villiger_oxidation/react.xyz")
target = TDStructure.from_xyz("/home/jdep/T3D_data/template_rxns/Baeyer-Villiger_oxidation/prod.xyz")
traj = Trajectory([root, target]).run_geodesic(nimages=15)
chain = Chain.from_traj(traj, cni)

h, out = m.find_mep_multistep(chain)

out.plot_chain()

out.to_trajectory()

from pathlib import Path

h.write_to_disk(Path("/home/jdep/T3D_data/template_rxns/Baeyer-Villiger_oxidation/without_h3o.xyz"))

out.write_to_disk(Path("/home/jdep/T3D_data/template_rxns/Baeyer-Villiger_oxidation/output_chain.xyz"))

step = 7
out[step*15].tdstructure.molecule_rp.draw(mode='oe')

out[(step*15)+14].tdstructure.molecule_rp.draw(mode='oe')

leaves = [l.data.chain_trajectory[-1] for l in h.ordered_leaves if l.data]

leaves[0].plot_chain()

out.plot_chain()


