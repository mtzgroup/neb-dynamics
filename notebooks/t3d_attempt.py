# +
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from ase.optimize.bfgs import BFGS
from ase.optimize.lbfgs import LBFGS
from pytest import approx
from neb_dynamics.geodesic_input import GeodesicInput
from retropaths.abinitio.inputs import Inputs
from retropaths.abinitio.rootstructure import RootStructure
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from xtb.ase.calculator import XTB
from xtb.interface import Calculator
from xtb.utils import get_method


from neb_dynamics.NEB import Node3D, Chain, NEB

out_dir = Path("/Users/janestrada/neb_dynamics/example_cases")
rxn_file = Path("/Users/janestrada/Retropaths/retropaths/data/reactions.p")


BOHR_TO_ANGSTROMS = 0.529177
# -

random.seed(1)

from neb_dynamics.helper_functions import pload

foo = pload(rxn_file)

# +
# [print(r) for r in foo]

# +
rn = "Claisen-Rearrangement"
inps = Inputs(rxn_name=rn, reaction_file=rxn_file)


struct = TDStructure.from_rxn_name(rn, data_folder=rxn_file.parent)
rs = RootStructure(root=struct, master_path=out_dir, rxn_args=inps, trajectory=Trajectory(traj_array=[]))


# +
tdstruct = rs.pseudoaligned
en_func = Node3D.en_func
grad_func = Node3D.grad_func

coords = tdstruct.coords

atoms = Atoms(
    symbols=tdstruct.symbols.tolist(),
    positions=coords
    # charges=0.0,
)

atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
opt = LBFGS(atoms)
opt.run(fmax=0.1)

opt_struct = TDStructure.from_coords_symbs(coords=atoms.positions, symbs=tdstruct.symbols, tot_charge=tdstruct.charge, tot_spinmult=tdstruct.spinmult)


# -

def opt_func(tdstruct):

    coords = tdstruct.coords

    atoms = Atoms(
        symbols=tdstruct.symbols.tolist(),
        positions=coords,
    )

    atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
    opt = LBFGS(atoms)
    opt.run(fmax=0.1)

    opt_struct = TDStructure.from_coords_symbs(coords=atoms.positions, symbs=tdstruct.symbols, tot_charge=tdstruct.charge, tot_spinmult=tdstruct.spinmult)

    return opt_struct



# +
# rn = "Claisen-Rearrangement"
# rn = "Diels Alder 4+2"
# rn = "Nucleophilic-Aliphatic-Substitution-Beta-acid-EWG1-Nitrile-and-EWG2-Nitrile-Lg-Iodine" # not valid????
# rn = "Riemschneider-Thiocarbamate-Synthesis-Water"
rn = "Chan-Rearrangement"
inps = Inputs(rxn_name=rn, reaction_file=rxn_file)


struct = TDStructure.from_rxn_name(rn, data_folder=rxn_file.parent)
mod_smi_struct = struct.molecule_rp.smiles+".[OH-]"
struct = TDStructure.from_smiles(mod_smi_struct, tot_charge=-1, tot_spinmult=1)


rs = RootStructure(root=struct, master_path=out_dir, rxn_args=inps, trajectory=Trajectory(traj_array=[]))
# -

rs.transformed.spinmult

# relax endpoints
opt_init = opt_func(rs.pseudoaligned)
opt_final = opt_func(rs.transformed)
# opt_init = rs.pseudoaligned
# opt_final = rs.transformed

### do geodesic interpolation/// but need to change the RP coords from Angstroms to Bohr.... cause im a fool
init_bohr = opt_init.coords_bohr
end_bohr = opt_final.coords_bohr
opt_init.update_coords(init_bohr)
opt_final.update_coords(end_bohr)

gi = GeodesicInput.from_endpoints(initial=opt_init, final=opt_final)
traj = gi.run(nimages=50, friction=0.1, nudge=0.01)

chain_geo = Chain.from_traj(traj, k=0.1, delta_k=0, step_size=1, node_class=Node3D)
plt.plot(chain_geo.energies)

traj.write_trajectory(Path("./Chan_traj_50_nodes.xyz"))

# +
# n = NEB(initial_chain=chain_geo, grad_thre_per_atom=0.0016, vv_force_thre=0)
# n.optimize_chain()
# -

opt_chain = n.optimized

plt.title(f"{rn}")
plt.plot((chain_geo.energies-opt_chain[0].energy)*627.5, '--',label="geodesic")
plt.plot((opt_chain.energies-opt_chain[0].energy)*627.5, 'o-', label="neb", color="orange")
plt.legend()

traj.write_trajectory(out_dir / f"{rn}_geodesic_opt.xyz")

n.write_to_disk(Path("./neb_chan_50_nodes.xyz"))

opt_traj = Trajectory(opt_chain[0])

opt_traj.write_trajectory(out_dir / f"{rn}_neb_opt.xyz")

# # What if, we take frames where a proton transfer happened, and dropped in a base in the viscinity?

# +
# load neb traj
traj_neb = Trajectory.from_xyz("./neb_chan.xyz")

# get frames where proton transfer happened
start = traj_neb[6]
end = traj_neb[7]

# write them to xyz files
start.write_to_disk("start_PT.xyz")
end.write_to_disk("end_PT.xyz")

# +
# load modified versions
start_mod = TDStructure.from_fp(Path("./start_PT_OH.xyz"), tot_charge=-1, tot_spinmult=3)
mod_start = start_mod.coords_bohr
start_mod.update_coords(mod_start)

end_mod = TDStructure.from_fp(Path("./end_PT_OH.xyz"), tot_charge=-1, tot_spinmult=3)
mod_end = end_mod.coords_bohr
end_mod.update_coords(mod_end)
# -

gi = GeodesicInput.from_endpoints(initial=start_mod, final=end_mod)
traj_geo_mod = gi.run(nimages=15, friction=0.01, nudge=0.001)

traj_geo_mod.write_trajectory(Path("traj_geo_mod.xyz"))

chain_geo_mod = Chain.from_traj(traj_geo_mod, k=0.1, delta_k=0,node_class=Node3D, step_size=.37)

n_mod = NEB(initial_chain=chain_geo_mod, grad_thre_per_atom=0.0016, vv_force_thre=0)
n_mod.optimize_chain()

opt_chain_mod = n_mod.optimized

plt.title(f"{rn}")
plt.plot((chain_geo_mod.energies-opt_chain_mod[0].energy)*627.5, '--',label="geodesic")
plt.plot((opt_chain_mod.energies-opt_chain_mod[0].energy)*627.5, 'o-', label="neb", color="orange")
plt.legend()

n_mod.write_to_disk(Path("./interesting.xyz"))

wtf = Trajectory([n.tdstructure for n in n_mod.chain_trajectory[-1].nodes])
wtf.write_trajectory(Path("wtf.xyz"))






