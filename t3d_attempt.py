# +
from retropaths.abinitio.rootstructure import RootStructure
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.inputs import Inputs
from retropaths.abinitio.geodesic_input import GeodesicInput
import matplotlib.pyplot as plt
from NEB_xtb import neb
from pathlib import Path


from ase.optimize.lbfgs import LBFGS
from ase.atoms import Atoms
from xtb.ase.calculator import XTB


out_dir = Path("/Users/janestrada/neb_dynamics/example_cases")
rxn_file = Path("/Users/janestrada/Retropaths/retropaths/data/reactions.p")
# -

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
rn = "Vicarious-Nucleophilic-Substitution-(Para)-X-Iodine-and-EWG-Carboxyl"
inps = Inputs(rxn_name=rn, reaction_file=rxn_file)


struct = TDStructure.from_rxn_name(rn, data_folder=rxn_file.parent)
rs = RootStructure(root=struct, 
                master_path=out_dir, 
                rxn_args=inps, 
                trajectory=Trajectory(traj_array=[]))

## relax endpoints
opt_init = n.opt_func(rs.pseudoaligned)
opt_final = n.opt_func(rs.transformed)
# -

n.en_funct(opt_final)

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
plt.plot(opt_chain_energies, label='neb')
plt.legend()

traj.write_trajectory(out_dir/f"{rn}_geodesic.xyz")

opt_traj = Trajectory(opt_chain[0])

opt_traj.write_trajectory(out_dir/f"{rn}_neb.xyz")


