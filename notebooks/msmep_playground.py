# +
from neb_dynamics.MSMEP import MSMEP
import retropaths.helper_functions as hf
from retropaths.abinitio.trajectory import Trajectory

reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")
# -

m = MSMEP(max_steps=2000,v=True,tol=0.01, nudge=0)

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


