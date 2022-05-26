# +
from retropaths.abinitio.rootstructure import RootStructure
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from xtb.utils import get_method
from retropaths.abinitio.inputs import Inputs
from retropaths.abinitio.geodesic_input import GeodesicInput
from xtb.interface import Calculator
import matplotlib.pyplot as plt
from NEB_xtb import neb
from pathlib import Path
import random
import numpy as np

from ase.optimize.lbfgs import LBFGS
from ase.optimize.bfgs import BFGS
from ase.atoms import Atoms
from xtb.ase.calculator import XTB


out_dir = Path("/Users/janestrada/neb_dynamics/example_cases")
rxn_file = Path("/Users/janestrada/Retropaths/retropaths/data/reactions.p")



BOHR_TO_ANGSTROMS = 0.529177
# -

random.seed(1)

from retropaths.helper_functions import pload

# +
# foo = pload(rxn_file)

# +
# foo["Ganem-Oxidation-X-Chlorine"].draw()

# +
# foo["Vicarious-Nucleophilic-Substitution-(Para)-X-Iodine-and-EWG-Carboxyl"].draw()

# +
# [print(r) for r in foo]
# -

n = neb()

# +
# from ALS_xtb import ArmijoLineSearch
# def opt_func(tdstruct, en_func, grad_func, en_thre=0.0001, grad_thre=0.0001, maxsteps=5000):
# # en_thre=0.01
# # grad_thre=0.01
# # maxsteps=5
# # en_func = n.en_func
# # grad_func = n.grad_func
#     # dr = 0.1
    

#     e0 = en_func(tdstruct)
#     g0 = grad_func(tdstruct)
#     dr = ArmijoLineSearch(struct=tdstruct, grad=g0, t=1, alpha=0.3, beta=0.8, f=en_func)
#     print(f"DR -->{dr}")
#     count = 0

#     coords1 = tdstruct.coords_bohr - dr*g0
#     tdstruct_prime = TDStructure.from_coords_symbs(
#         coords=coords1*BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols,
#         tot_charge=tdstruct.charge,
#         tot_spinmult=tdstruct.spinmult
#     )


#     e1 = en_func(tdstruct_prime)
#     g1 = grad_func(tdstruct_prime)

#     struct_conv = (np.abs(e1-e0) < en_thre) and False not in (np.abs(g1-g0) < grad_thre).flatten()

#     while not struct_conv and count < maxsteps:
#         count+=1

#         e0 = e1
#         g0 = g1

#         coords1 = tdstruct.coords_bohr - dr*g0
#         tdstruct_prime = TDStructure.from_coords_symbs(
#             coords=coords1*BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols,
#             tot_charge=tdstruct.charge,
#             tot_spinmult=tdstruct.spinmult
#         )


#         e1 = en_func(tdstruct_prime)
#         g1 = grad_func(tdstruct_prime)

#         struct_conv = (np.abs(e1-e0) < en_thre) and False not in (np.abs(g1-g0) < grad_thre).flatten()
        
#     print(f"Converged --> {struct_conv} in {count} steps")
#     return tdstruct_prime


# +
rn = "Claisen-Rearrangement"
inps = Inputs(rxn_name=rn, reaction_file=rxn_file)


struct = TDStructure.from_rxn_name(rn, data_folder=rxn_file.parent)
rs = RootStructure(root=struct, 
                master_path=out_dir, 
                rxn_args=inps, 
                trajectory=Trajectory(traj_array=[]))

# relax endpoints
opt_init = n.opt_func(rs.pseudoaligned, en_func=n.en_func, grad_func=n.grad_func)
opt_final = n.opt_func(rs.transformed, en_func=n.en_func, grad_func=n.grad_func)
# opt_init = rs.pseudoaligned
# opt_final = rs.transformed
# -

n.en_func(opt_final)

### do geodesic interpolation
gi = GeodesicInput.from_endpoints(initial=opt_init, final=opt_final)
traj = gi.run(
    nimages=15, 
    friction=0.1,
    nudge=0.01
)

ens = [n.en_func(s) for s in traj]

plt.plot(ens)

opt_chain = n.optimize_chain(chain=traj,grad_func=n.grad_func, en_func=n.en_func, k=10)

opt_chain_energies = [n.en_func(s) for s in opt_chain[0]]

plt.title(f"{rn}")
plt.plot(ens, label='geodesic')
plt.scatter(list(range(len(opt_chain_energies))), opt_chain_energies, label='neb', color='orange')
plt.legend()

# +
# traj.write_trajectory(out_dir/f"{rn}_geodesic_opt_2.xyz")

# +
# opt_traj = Trajectory(opt_chain[0])

# +
# opt_traj.write_trajectory(out_dir/f"{rn}_neb_opt_2.xyz")
# -


