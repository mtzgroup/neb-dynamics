# +
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from ase.optimize.bfgs import BFGS
from ase.optimize.lbfgs import LBFGS
from pytest import approx
from retropaths.abinitio.geodesic_input import GeodesicInput
from retropaths.abinitio.inputs import Inputs
from retropaths.abinitio.rootstructure import RootStructure
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from xtb.ase.calculator import XTB
from xtb.interface import Calculator
from xtb.utils import get_method


out_dir = Path("/Users/janestrada/neb_dynamics/example_cases")
rxn_file = Path("/Users/janestrada/Retropaths/retropaths/data/reactions.p")


BOHR_TO_ANGSTROMS = 0.529177
# -

random.seed(1)

from neb_dynamics.helper_functions import pload

foo = pload(rxn_file)

foo["Claisen-Rearrangement"].draw()

# +
# [print(r) for r in foo]

# +
rn = "Claisen-Rearrangement"
inps = Inputs(rxn_name=rn, reaction_file=rxn_file)


struct = TDStructure.from_rxn_name(rn, data_folder=rxn_file.parent)
rs = RootStructure(root=struct, master_path=out_dir, rxn_args=inps, trajectory=Trajectory(traj_array=[]))


# +
tdstruct = rs.pseudoaligned
en_func = n.en_func
grad_func = n.grad_func

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

# +
en_thre = 0.01
grad_thre = 0.01
maxsteps = 100
e0 = en_func(tdstruct)
g0 = grad_func(tdstruct)
dr = ArmijoLineSearch(struct=tdstruct, grad=g0, t=1, alpha=0.3, beta=0.8, f=en_func)
print(f"DR -->{dr}")
count = 0

coords1 = tdstruct.coords_bohr - dr * g0
tdstruct_prime = TDStructure.from_coords_symbs(coords=coords1 * BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols, tot_charge=tdstruct.charge, tot_spinmult=tdstruct.spinmult)


e1 = en_func(tdstruct_prime)
g1 = grad_func(tdstruct_prime)

struct_conv = (np.abs(e1 - e0) < en_thre) and False not in (np.abs(g1 - g0) < grad_thre).flatten()

while not struct_conv and count < maxsteps:
    count += 1

    e0 = e1
    g0 = g1

    dr = ArmijoLineSearch(struct=tdstruct, grad=g0, t=1, alpha=0.3, beta=0.8, f=en_func)
    print(f"DR --> {dr}")
    coords1 = tdstruct_prime.coords_bohr - dr * g0
    tdstruct_prime = TDStructure.from_coords_symbs(coords=coords1 * BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols, tot_charge=tdstruct.charge, tot_spinmult=tdstruct.spinmult)

    e1 = en_func(tdstruct_prime)
    g1 = grad_func(tdstruct_prime)

    struct_conv = (np.abs(e1 - e0) < en_thre) and False not in (np.abs(g1 - g0) < grad_thre).flatten()

print(f"Converged --> {struct_conv} in {count} steps")


# -


def opt_func(tdstruct, en_func, grad_func, en_thre=0.0001, grad_thre=0.0001, maxsteps=5000):

    coords = tdstruct.coords_bohr

    atoms = Atoms(
        symbols=tdstruct.symbols.tolist(),
        positions=coords,
    )

    atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
    opt = LBFGS(atoms)
    opt.run(fmax=0.1)

    opt_struct = TDStructure.from_coords_symbs(coords=atoms.positions * BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols, tot_charge=tdstruct.charge, tot_spinmult=tdstruct.spinmult)

    return opt_struct

    e0 = en_func(tdstruct)
    g0 = grad_func(tdstruct)
    dr = ArmijoLineSearch(struct=tdstruct, grad=g0, t=1, alpha=0.3, beta=0.8, f=en_func)
    print(f"DR -->{dr}")
    count = 0

    coords1 = tdstruct.coords_bohr - dr * g0
    tdstruct_prime = TDStructure.from_coords_symbs(coords=coords1 * BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols, tot_charge=tdstruct.charge, tot_spinmult=tdstruct.spinmult)

    e1 = en_func(tdstruct_prime)
    g1 = grad_func(tdstruct_prime)

    struct_conv = (np.abs(e1 - e0) < en_thre) and False not in (np.abs(g1 - g0) < grad_thre).flatten()

    while not struct_conv and count < maxsteps:
        count += 1

        e0 = e1
        g0 = g1

        dr = ArmijoLineSearch(struct=tdstruct, grad=g0, t=1, alpha=0.3, beta=0.8, f=en_func)
        coords1 = tdstruct.coords_bohr - dr * g0
        tdstruct_prime = TDStructure.from_coords_symbs(coords=coords1 * BOHR_TO_ANGSTROMS, symbs=tdstruct.symbols, tot_charge=tdstruct.charge, tot_spinmult=tdstruct.spinmult)

        e1 = en_func(tdstruct_prime)
        g1 = grad_func(tdstruct_prime)

        struct_conv = (np.abs(e1 - e0) < en_thre) and False not in (np.abs(g1 - g0) < grad_thre).flatten()

    print(f"Converged --> {struct_conv} in {count} steps")
    return tdstruct_prime


def test_gfn2xtb_lbfgs():
    """Perform geometry optimization with GFN2-xTB and L-BFGS"""

    thr = 1.0e-5

    atoms = Atoms(
        symbols="NHCHC2H3OC2H3ONHCH3",
        positions=np.array(
            [
                [1.40704587284727, -1.26605342016611, -1.93713466561923],
                [1.85007200612454, -0.46824072777417, -1.50918242392545],
                [-0.03362432532150, -1.39269245193812, -1.74003582081606],
                [-0.56857009928108, -1.01764444489068, -2.61263467107342],
                [-0.44096297340282, -2.84337808903410, -1.48899734014499],
                [-0.47991761226058, -0.55230954385212, -0.55520222968656],
                [-1.51566045903090, -2.89187354810876, -1.32273881320610],
                [-0.18116520746778, -3.45187805987944, -2.34920431470368],
                [0.06989722340461, -3.23298998903001, -0.60872832703814],
                [-1.56668253918793, 0.00552120970194, -0.52884675001441],
                [1.99245341064342, -1.73097165236442, -3.08869239114486],
                [3.42884244212567, -1.30660069291348, -3.28712665743189],
                [3.87721962540768, -0.88843123009431, -2.38921453037869],
                [3.46548545761151, -0.56495308290988, -4.08311788302584],
                [4.00253374168514, -2.16970938132208, -3.61210068365649],
                [1.40187968630565, -2.43826111827818, -3.89034127398078],
                [0.40869198386066, -0.49101709352090, 0.47992424955574],
                [1.15591901335007, -1.16524842262351, 0.48740266650199],
                [0.00723492494701, 0.11692276177442, 1.73426297572793],
                [0.88822128447468, 0.28499001838229, 2.34645658013686],
                [-0.47231557768357, 1.06737634000561, 1.52286682546986],
                [-0.70199987915174, -0.50485938116399, 2.28058247845421],
            ]
        ),
    )

    atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
    opt = LBFGS(atoms)
    opt.run(fmax=0.1)

    # assert approx(atoms.get_potential_energy(), thr) == -897.4533662470938
    # assert approx(np.linalg.norm(atoms.get_forces(), ord=2), thr) == 0.19359647527783497


test_gfn2xtb_lbfgs()

# +
# rn = "Claisen-Rearrangement"
# rn = "Diels Alder 4+2"
# rn = "Nucleophilic-Aliphatic-Substitution-Beta-acid-EWG1-Nitrile-and-EWG2-Nitrile-Lg-Iodine" # not valid????
# rn = "Riemschneider-Thiocarbamate-Synthesis-Water"
rn = "Chan-Rearrangement"
inps = Inputs(rxn_name=rn, reaction_file=rxn_file)


struct = TDStructure.from_rxn_name(rn, data_folder=rxn_file.parent)
rs = RootStructure(root=struct, master_path=out_dir, rxn_args=inps, trajectory=Trajectory(traj_array=[]))

# relax endpoints
opt_init = opt_func(rs.pseudoaligned, en_func=n.en_func, grad_func=n.grad_func)
opt_final = opt_func(rs.transformed, en_func=n.en_func, grad_func=n.grad_func)
# opt_init = rs.pseudoaligned
# opt_final = rs.transformed
# -

n.en_func(opt_final)

### do geodesic interpolation
gi = GeodesicInput.from_endpoints(initial=opt_init, final=opt_final)
traj = gi.run(nimages=15, friction=0.1, nudge=0.01)

ens = [n.en_func(s) for s in traj]

plt.plot(ens)

opt_chain = n.optimize_chain(chain=traj, grad_func=n.grad_func, en_func=n.en_func, k=10)

opt_chain_energies = [n.en_func(s) for s in opt_chain[0]]

plt.title(f"{rn}")
plt.plot(ens, label="geodesic")
plt.scatter(list(range(len(opt_chain_energies))), opt_chain_energies, label="neb", color="orange")
plt.legend()

traj.write_trajectory(out_dir / f"{rn}_geodesic_opt.xyz")

opt_traj = Trajectory(opt_chain[0])

opt_traj.write_trajectory(out_dir / f"{rn}_neb_opt.xyz")
