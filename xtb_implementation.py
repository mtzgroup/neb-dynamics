from xtb.interface import Calculator
from xtb.utils import get_method
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

data_dir = Path("./example_cases/")
# traj = Trajectory.from_xyz(data_dir/'diels_alder.xyz')
traj = Trajectory.from_xyz(data_dir/'PDDA_geodesic.xyz')

struct = traj[0]


# +
def en_func(tdstruct):
    coords = tdstruct.coords_bohr
    atomic_numbers = tdstruct.atomic_numbers
    
    calc = Calculator(get_method("GFN2-xTB"), numbers=np.array(atomic_numbers), positions=coords,
                     charge=tdstruct.charge, uhf=tdstruct.spinmult-1)
    res = calc.singlepoint()

    return res.get_energy()
    

en_func(struct)

# +
# Disable
import os
import sys
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# +
def grad_func(tdstruct):

    
    coords = tdstruct.coords_bohr
    atomic_numbers = tdstruct.atomic_numbers

    # blockPrint()
    calc = Calculator(get_method("GFN2-xTB"), numbers=np.array(atomic_numbers), positions=coords,
                     charge=tdstruct.charge, uhf=tdstruct.spinmult-1)
    res = calc.singlepoint()
    grad = res.get_gradient()

    
    # enablePrint()
    return grad

grad_func(struct)
# -

blockPrint()

# # NEB

from NEB_xtb import neb

original_chain_ens = [en_func(s) for s in traj]

plt.plot(original_chain_ens)

# +
foo1 = np.array([[1,2,3]])
foo2 = np.array([[3,2,1]])

np.tensordot(foo1, foo2)
# -

opt_chain, opt_chain_traj = neb().optimize_chain(chain=traj,grad_func=grad_func,en_func=en_func,k=10, max_steps=100)

opt_chain_ens = [en_func(s) for s in opt_chain]

plt.plot(list(range(len(original_chain_ens))),original_chain_ens,'x--', label='orig')
plt.plot(list(range(len(original_chain_ens))), opt_chain_ens, 'o',label='neb')
plt.legend()

out = Trajectory(opt_chain)


out.write_trajectory(data_dir/'pdda_neb_100_steps.xyz')


