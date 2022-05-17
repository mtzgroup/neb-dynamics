from xtb.interface import Calculator
from xtb.utils import get_method
from retropaths.abinitio.trajectory import Trajectory
from pathlib import Path
import numpy as np

data_dir = Path("./example_cases/")
traj = Trajectory.from_xyz(data_dir/'diels_alder.xyz')

struct = traj[0]


# +
def en_func(tdstruct):
    coords = tdstruct.coords
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


# -

s.spinmult


# +
def grad_func(tdstruct):

    
    coords = tdstruct.coords
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

# # NEB

from NEB_xtb import neb

opt_chain = neb().optimize_chain(chain=traj,grad_func=grad_func,en_func=en_func,k=10)

s = traj[9]

s.write_to_disk(data_dir/"wtf.xyz")

s.coords

grad_func(s)

s.charge


