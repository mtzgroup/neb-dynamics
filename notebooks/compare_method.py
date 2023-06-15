# +
from neb_dynamics.CompetitorAnalyzer import CompetitorAnalyzer
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.helper_functions import create_friction_optimal_gi
from neb_dynamics.MSMEP import MSMEP

from retropaths.abinitio.trajectory import Trajectory

from pathlib import Path
import pandas as pd
import retropaths.helper_functions as hf

import matplotlib.pyplot as plt
# -

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

reactions = hf.pload("../../retropaths/data/reactions.p")

import os
del os.environ['OE_LICENSE']

directory = Path("/home/jdep/T3D_data/msmep_draft/comparisons")

comparisons_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons/")

ca = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='dlfind')
ca2 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='pygsm')
ca3 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='nebd')

# +
# ca3.submit_all_jobs()
# -

# # Make Structures

rns = ca.available_reaction_names

# for rn in rns:
# for rn in ["Ramberg-Backlund-Reaction-Bromine"]:
# for rn in ["Paal-Knorr-Furan-Synthesis"]:
for rn in ["Bamberger-Rearrangement"]:
# for rn in ["Claisen-Rearrangement-Aromatic"]:
    rxn = reactions[rn]
    m = MSMEP(NEBInputs(), ChainInputs(), GIInputs())
    r, p = m.create_endpoints_from_rxn_name(rn, reactions)


r

p

t = Trajectory([r, p]).run_geodesic(nimages=15)

t_opt = create_friction_optimal_gi(t, GIInputs())

plt.plot(t.energies_xtb(),'o-')
plt.plot(t_opt.energies_xtb(),'x-')

# t.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Ramberg-Backlund-Reaction-Bromine/initial_guess.xyz")
# t.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Paal-Knorr-Furan-Synthesis/initial_guess.xyz")
# t_opt.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Claisen-Rearrangement-Aromatic/initial_guess.xyz")
t_opt.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Bamberger-Rearrangement/initial_guess.xyz")

j = Janitor(history_object=asneb2,out_path=Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/tighter_conv/initial_guess_tight_endpoints_cleanups"),
           msmep_object=m)
