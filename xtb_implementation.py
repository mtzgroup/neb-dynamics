# +
from xtb.interface import Calculator
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path
from NEB_xtb import neb
import numpy as np
import matplotlib.pyplot as plt
from ALS_xtb import ArmijoLineSearch

from ase.optimize.lbfgs import LBFGS
from ase.atoms import Atoms
from xtb.ase.calculator import XTB

# +

ANGSTROM_TO_BOHR = 1.88973
BOHR_TO_ANGSTROMS = 1/ANGSTROM_TO_BOHR
# -

data_dir = Path("./example_cases/")
# traj = Trajectory.from_xyz(data_dir/'diels_alder.xyz')
# traj = Trajectory.from_xyz(data_dir/'PDDA_geodesic.xyz')
traj = Trajectory.from_xyz(data_dir/"Claisen-Rearrangement_geodesic.xyz")


struct = traj[5]

struct.write_to_disk(data_dir/'orig.xyz')

f(struct)

# +
t=1
grad = n.grad_func(struct)
f = n.en_func

new_coords_bohr = struct.coords_bohr - t*grad
new_coords = new_coords_bohr*BOHR_TO_ANGSTROMS

struct_prime = TDStructure.from_coords_symbs(
            coords=new_coords,
            symbs=struct.symbols,
            tot_charge=struct.charge,
            tot_spinmult=struct.spinmult)
    
en_struct_prime = f(struct_prime)
# -

en_struct_prime

struct_prime.write_to_disk(data_dir/'wtf.xyz')

n = neb()
n.en_func(struct)



struct.charge

struct.spinmult

# +
tdstruct=struct
en_func = n.en_func
grad_func = n.grad_func
en_thre=0.0001
grad_thre=0.0001
maxsteps=1000


e0 = en_func(tdstruct)
g0 = grad_func(tdstruct)


# dr=0.1
dr = ArmijoLineSearch(struct=tdstruct, grad=g0, t=1, alpha=0.3, beta=0.8, f=en_func)

count = 0

coords1 = tdstruct.coords_bohr - dr*g0
tdstruct_prime = TDStructure.from_coords_symbs(
    coords=coords1*BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols,
    tot_charge=tdstruct.charge,
    tot_spinmult=tdstruct.spinmult
)


e1 = en_func(tdstruct_prime)
g1 = grad_func(tdstruct_prime)

struct_conv = (np.abs(e1-e0) < en_thre) and False not in (np.abs(g1-g0) < grad_thre).flatten()
print(f"DR -->{dr}")
while not struct_conv and count < maxsteps:
    count+=1

    e0 = e1
    g0 = g1

    dr = ArmijoLineSearch(struct=tdstruct, grad=g0, t=0.5, alpha=0.3, beta=0.8, f=en_func)
    coords1 = tdstruct.coords_bohr - dr*g0
    tdstruct_prime = TDStructure.from_coords_symbs(
        coords=coords1*BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols,
        tot_charge=tdstruct.charge,
        tot_spinmult=tdstruct.spinmult
    )


    e1 = en_func(tdstruct_prime)
    g1 = grad_func(tdstruct_prime)

    struct_conv = (np.abs(e1-e0) < en_thre) and False not in (np.abs(g1-g0) < grad_thre).flatten()

print(f"Converged --> {struct_conv} in {count} steps")

# +
# n.opt_func(tdstruct=struct,en_func=n.en_func,grad_func=n.grad_func)
# -

# # NEB

from NEB_xtb import neb

original_chain_ens = [en_func(s) for s in traj]

plt.plot(original_chain_ens)

# +
foo1 = np.array([[1,2,3]])
foo2 = np.array([[3,2,1]])

np.tensordot(foo1, foo2)
# -

# #### start time: 1045am
# #### end time: <=11:22am

opt_chain, opt_chain_traj = neb().optimize_chain(chain=traj,grad_func=grad_func,en_func=en_func,k=10, max_steps=1000)

opt_chain_ens = [en_func(s) for s in opt_chain]

plt.plot(list(range(len(original_chain_ens))),original_chain_ens,'x--', label='orig')
plt.plot(list(range(len(original_chain_ens))), opt_chain_ens, 'o',label='neb')
plt.legend()

out = Trajectory(opt_chain)


out.write_trajectory(data_dir/'pdda_neb_1000_steps_k_10.xyz')


