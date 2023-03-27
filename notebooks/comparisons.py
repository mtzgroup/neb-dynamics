from pathlib import Path
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
import numpy as np

from neb_dynamics.MSMEP import MSMEP
from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')


# # Helper Functions

def get_eA(chain_energies):
    return max(chain_energies) - chain_energies[0]


# # Variables

NIMAGES = 15

# # Make endpoints

out_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures")

import retropaths.helper_functions  as hf
reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")
m = MSMEP()

r, p = m.create_endpoints_from_rxn_name("Aza-Grob-Fragmentation-X-Bromine", reactions)

# +
rxn_name = "Aza-Grob-Fragmentation-X-Bromine"
reactions_object = reactions

rxn = reactions_object[rxn_name]
root = TDStructure.from_rxn_name(rxn_name, reactions_object)

c3d_list = root.get_changes_in_3d(rxn)

root = root.pseudoalign(c3d_list)
root.gum_mm_optimization()
root = root.xtb_geom_optimization()

target = root.copy()
target.apply_changed3d_list(c3d_list)
target.mm_optimization("gaff", steps=5000)
target.mm_optimization("uff", steps=5000)
# target = target.xtb_geom_optimization()


# -

target

#### extract templates with 1 reactant and no charges
single_mol_reactions = []
for rn in reactions:
    rxn = reactions[rn]
    n_reactants = len(rxn.reactants.force_smiles().split("."))
    if n_reactants == 1:
        if rxn.reactants.charge == 0:
            single_mol_reactions.append(rn)

#### create endpoints
failed = []
for rn in single_mol_reactions:
    try:
        rn_output_dir = out_dir / rn
        if rn_output_dir.exists():
            print(f"already did {rn}")
            continue
        rn_output_dir.mkdir(parents=False, exist_ok=True)
        r, p = m.create_endpoints_from_rxn_name(rn, reactions)
        r_opt, p_opt = r.xtb_geom_optimization(), p.xtb_geom_optimization()
        
        
        r_opt.to_xyz(rn_output_dir / "start.xyz")
        p_opt.to_xyz(rn_output_dir / "end.xyz")
    except:
        print(f"{rn} did not work. Check it out.")
        failed.append(rn)

# +
#### create gis
failed = []

frics = [0.001,0.01, 0.1, 1]

for rn in single_mol_reactions:
    # try:
    rn_output_dir = out_dir / rn
    output_file = rn_output_dir / "initial_guess.xyz"
    if output_file.exists():
        print(f'{output_file} exists')
        continue
    else:
        
        start_fp = rn_output_dir/"start.xyz"
        end_fp = rn_output_dir/"end.xyz"
        start = TDStructure.from_xyz(str(start_fp.resolve()))
        end = TDStructure.from_xyz(str(end_fp.resolve()))
        trajs = []
        for fric in frics:
            traj = Trajectory([start, end]).run_geodesic(nimages=NIMAGES, friction=fric)
            traj.write_trajectory(rn_output_dir / f'gi_fric{fric}.xyz')
            trajs.append(traj)

        eAs = [get_eA(t.energies_xtb()) for t in trajs]
        best_gi_ind = np.argmin(eAs)
        best_gi = trajs[best_gi_ind]
        print(f"Best {rn} gi had friction {frics[best_gi_ind]}")
        best_gi.write_trajectory(output_file)

            
                
        
        
#     except:
#         print(f"{rn} did not work. Check it out.")
#         failed.append(rn)
# -

# # write scripts for sbatching


