from retropaths.abinitio.trajectory import Trajectory
from pathlib import Path
import numpy as np

t = Trajectory.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/orca_gfn2_comp/initial_guess.xyz"))

start = t[0]
end = t[-1]

gi_noisy = Trajectory([start, end]).run_geodesic(nimages=20,nudge=5)
gi_notnoisy = Trajectory([start, end]).run_geodesic(nimages=20)

import torch


def _get_distances(path):
    dist = []
    prev_node = path[0]
    for node in path[1:]:
        dist.append(np.linalg.norm(node.coords - prev_node.coords))
        prev_node = node
    return np.array(dist)



def _get_delta_ens(ens):
    deltas = []
    prev_en = ens[0]
    for en in ens[1:]:
        deltas.append(np.abs(en - prev_en)) #### double check why it must/must not be absolute value
        prev_en = en
    return np.array(deltas)



def _get_grads(path):
    grads = []
    prev_node = path[0]
    for node in path[1:]:
        grads.append(np.linalg.norm(node.gradient_xtb()))
        prev_node = node
    return np.array(grads)



def path_work(path):
    ens = path.energies_xtb()
    distances = _get_distances(path)
    # delta_ens = _get_delta_ens(ens)
    delta_grads = _get_grads(path)
    # work = sum(delta_ens*distances)
    work = sum(delta_grads*distances)
    # work = sum(delta_ens)
    
    return work


path_work(gi_noisy)

path_work(gi_notnoisy)

import matplotlib.pyplot as plt

plt.plot(gi_noisy.energies_xtb(),'o-',label='noisy')
plt.plot(gi_notnoisy.energies_xtb(),'o-',label='notnoisy')
plt.legend()


