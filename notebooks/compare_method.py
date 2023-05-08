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

# # WTF is happening with GFN1

# +
from neb_dynamics.Chain import Chain
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.Node3D_gfn1xtb import Node3D_gfn1xtb
from neb_dynamics.NEB import NEB


from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path
# -

import os
del os.environ['OE_LICENSE']

t = Trajectory.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/initial_guess_gfn1.xyz"))

import numpy as np

h.adj_matrix - np.identity(h.adj_matrix.shape[0])

start = t[0]
end = t[-1]
tr = Trajectory([start, end]).run_geodesic(nimages=15, sweep=False)

cni = ChainInputs(node_class=Node3D_gfn1xtb,delta_k=0.009, k=0.01)
init = Chain.from_traj(tr, parameters=cni)

m = MSMEP(neb_inputs=NEBInputs(vv_force_thre=0.00, tol=0.005, climb=True, v=True,early_stop_chain_rms_thre=0.001, early_stop_force_thre=0.01),chain_inputs=cni, gi_inputs=GIInputs(extra_kwds={'sweep':False}))

h, out = m.find_mep_multistep(init)

out.plot_chain()

clean_out = m.create_clean_msmep(h)

h.write_to_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/result_gfn1"))

from neb_dynamics.TreeNode import TreeNode

h = TreeNode.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/result_gfn1"))

h.adj_matrix

# ls /home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/result_gfn1/

out = Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/result_gfn1/")
t0 = Trajectory.from_xyz(out / 'node_7.xyz')

m = MSMEP(NEBInputs(), ChainInputs(), GIInputs())

seqs = m.make_sequence_of_chains(c0, 'maxima')

c0 = Chain.from_traj(t0, parameters=ChainInputs(node_class=Node3D_gfn1xtb))

c0.plot_chain()

c0.is_elem_step()

m = MSMEP(neb_inputs=NEBInputs(vv_force_thre=0.00, tol=0.005, climb=True, v=True,early_stop_chain_rms_thre=0.001, early_stop_force_thre=0.01),chain_inputs=cni, gi_inputs=GIInputs(extra_kwds={'sweep':False}))
clean_out = m.create_clean_msmep(h)

# +

# clean_out.to_trajectory().write_trajectory(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/result_gfn1_clean_out.xyz"))
# -

clean_out[0].is_identical(init[0])

clean_out.plot_chain()

import warnings
warnings.filterwarnings('ignore')

h, out = m.find_mep_multistep(init)

clean_msmep = m.create_clean_msmep(h)

clean_msmep.to_trajectory()

import networkx as nx

g = nx.from_numpy_array(hist.adj_matrix)

g

hist.adj_matrix

hist.children[1].data.optimized.to_trajectory()

hist.output_chain.plot_chain()

len(hist.children[0].children[0].children)

problematic_chain = hist.children[0].children[0].data.optimized

guess_to_chain = hist.children[0].children[0].data.initial_chain

r,p = problematic_chain._approx_irc()

aligned_self = r.tdstructure.align_to_td(guess_to_chain[0].tdstructure)

from neb_dynamics.helper_functions import RMSD


rmsd = RMSD(aligned_self.coords, guess_to_chain[0].tdstructure.coords)

aligned_self

problematic_chain.to_trajectory()

r.tdstructure

p.tdstructure

p.tdstructure

clean_chain = Chain.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/initial_guess_gfn1_msmep_clean.xyz"),ChainInputs(node_class=Node3D_gfn1xtb))

clean_chain.plot_chain()


