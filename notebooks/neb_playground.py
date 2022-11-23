# -*- coding: utf-8 -*-
# +
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from neb_dynamics.NEB import Chain, NEB, Node3D
from neb_dynamics.MSMEP import MSMEP
from matplotlib.animation import FuncAnimation
from neb_dynamics.helper_functions import pairwise
from neb_dynamics.ALS import ArmijoLineSearch
from neb_dynamics.remapping_helpers import create_correct_product

from dataclasses import dataclass
# -

c = Chain.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v4/traj_0-1_0_cneb.xyz")
c_t = Chain.from_xyz("../example_cases/claisen/cr_MSMEP_tol_01.xyz")
out = Chain.from_xyz("../example_cases/claisen/cr_MSMEP_tol_0045_hydrogen_fix.xyz")


start = c[0].tdstructure
end = c[-1].tdstructure

m = MSMEP(v=True, max_steps=2000, tol=0.0045)

out = m.find_mep_multistep((start, end), do_alignment=True)

out.plot_chain()

t.write_trajectory("../example_cases/claisen/cr_MSMEP_tol_0045_hydrogen_fix.xyz")

t = Trajectory([n.tdstructure for n in out])
t.draw();


# +
s=8
fs = 18
f, ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(out.integrated_path_length, (out.energies-c.energies[0])*627.5,'o-', label="MSMEP_tol_0045_hydrogen_fix")
plt.plot(c_t.integrated_path_length, (c_t.energies-c.energies[0])*627.5,'o-', label='MSMEP_tol_01')
plt.plot(c.integrated_path_length, (c.energies-c.energies[0])*627.5,'o-', label='cNEB')

plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()


# -

def t_from_chain(chain):
    return Trajectory([n.tdstructure for n in chain])


# # Other rxns

from retropaths.helper_functions import pload
rxns = pload("/Users/janestrada/Retropaths/retropaths/data/reactions.p")

# +
from retropaths.reactions.template_utilities import \
    give_me_molecule_with_random_replacement_from_rules


# -

rxn = rxns["Ugi-Reaction"]
mol = give_me_molecule_with_random_replacement_from_rules(rxn, rxns.matching)

rxn.draw(mode='d3',size=(300,300),node_index=True, charges=True)

mol2 = rxn.apply_forward(mol)[0]

start = TDStructure.from_RP(mol)
# start = TDStructure.from_RP(mol.separate_graph_in_pieces()[0])
end = TDStructure.from_RP(mol2)

from retropaths.reactions.changes import Changes3D, Changes3DList, ChargeChanges

# +
# changes = rxn.changes_react_to_prod[0]
# c3d_list_frags = from_rxn_changes(changes, start)
# c3d_list = from_rxn_changes_full(changes)

c3d_list = start.get_changes_in_3d(rxn)


start = start.pseudoalign(c3d_list)
start.mm_optimization('uff', steps=2000)
start.mm_optimization('gaff', steps=2000)
start.mm_optimization('mmff94', steps=2000)
start = start.xtb_geom_optimization()

end_mod = start.copy()
end_mod.add_bonds(c3d_list.forming)
end_mod.delete_bonds(c3d_list.deleted)
end_mod.mm_optimization('uff', steps=2000)
end_mod.mm_optimization('gaff', steps=2000)
end_mod.mm_optimization('mmff94', steps=2000)
end_mod = end_mod.xtb_geom_optimization()
# -

# Jan: this is close. if I can have a function that takes two 3D structures, then outputs the "template" that connects them, I can generate the pseudoaligned endpoints for two arbitrary structures found through neb minimizations

m = MSMEP(v=True, max_steps=2000, tol=0.0045, nimages=15, friction=0.01, nudge=0.01, k=0.001)

# +
# t.write_trajectory("../example_cases/ugi/ugi_tol0045_no_converged.xyz")
# -

out = m.find_mep_multistep((start, end_mod),do_alignment=False)

out2 = Chain.from_xyz("../example_cases/ugi/auto_extracted_0.xyz")
out2.plot_chain()

t = Trajectory([n.tdstructure for n in out2])

# +
# t.write_trajectory("../example_cases/ugi/auto_extracted_0.xyz")
# -

t.draw();


o = m.get_neb_chain(start, end_mod, do_alignment=False)

TDStructureplot_chain()

t = Trajectory([n.tdstructure for n in o])

t[0]

t[5].xtb_geom_optimization()

t[-1].xtb_geom_optimization()

# +

# m.find_mep_multistep((start, end),do_alignment=True)
# -


