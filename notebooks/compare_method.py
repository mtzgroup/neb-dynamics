from neb_dynamics.CompetitorAnalyzer import CompetitorAnalyzer
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

comparisons_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons/")

ca = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='dlfind')
ca2 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='pygsm')
ca3 = CompetitorAnalyzer(comparisons_dir=comparisons_dir,method='nebd')

ca3.submit_all_jobs()

f = open(ca2.out_folder / 'Chan-Rearrangement' / 'log')
foobar = f.read().splitlines()

hey = ca2.out_folder / 'Chan-Rearrangement' / "string_trajectory.xyz"
c = Chain.from_xyz(str(hey.resolve()),parameters=ChainInputs())

en_history = []
for l in foobar:
    if "V_profile:" in l:
        en_history.append([float(x) for x in l.split()[1:]])


# +
def trajectory_from_pygsm_file(fp):
    """
    this is ugly as foooook
    """
    f = open(fp)
    all_lines = f.read().splitlines()
    coordinates = []
    n_atoms = int(all_lines[2])
    n_beads = 0
    bead = []
    atom_names = []
    for line in all_lines[2:]:
        if line == str(n_atoms):
            if len(bead) > 0:
                coordinates.append(bead)
                bead = []
            counter = 0
            n_beads += 1
            continue
        elif counter >= 0 and counter < n_atoms:
            counter+=1
            if len(line.split()) > 0:
                xyz_str = line.split()[1:]
                if n_beads == 1 : atom_names.append(line.split()[0])
                xyz = [float(val) for val in xyz_str]
                bead.append(xyz)
                
    return Trajectory.from_coords_symbols(coords=np.array(coordinates), symbs=atom_names)

# fp = ca2.out_folder / "Chan-Rearrangement" / 'opt_converged_000.xyz'
fp = ca2.out_folder / "Aza-Grob-Fragmentation-X-Bromine" / 'opt_converged_000.xyz'


hey = trajectory_from_pygsm_file(fp)
# -

hey.draw()

import numpy as np

from retropaths.abinitio.trajectory import Trajectory

t = 


def get_output_info(obj, reaction_name):
    if obj.method == 'dlfind':
        out_rn = obj.out_folder / reaction_name
        results_fp = out_rn / "scr.initial_guess" / 'nebinfo'
        f = open(results_fp)
        data = f.read().splitlines()
        header = ["path_length",'energy','work']
        res = [l.split() for l in data[1:]]
        df = pd.DataFrame(res, columns=header)
        path_len = df["path_length"].apply(float).values
        path_len_norm = path_len / max(path_len)
        energies = df["energy"].apply(float).values
    
    elif obj.method == 'pygsm':
        out_rn = obj.out_folder / reaction_name 
        results_fp = out_rn / "log"
        c = Chain.from_xyz(out_rn / "opt_converged_000.xyz",parameters=ChainInputs())
        path_len_norm = c.integrated_path_length
        # energies = 
    
    
    

    return path_len_norm, energies

pathlen, ens = get_output_info(ca, 'Chan-Rearrangement')

plt.plot(pathlen, ens*627.5, 'o-', label='dlfind')
plt.plot(c.integrated_path_length, en_history[-1], 'o-', label='pygsm')
plt.legend()
plt.show()


