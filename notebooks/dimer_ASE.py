# +
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
# from xtb.ase.calculator import XTB
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.io import Trajectory as ASETraj

def test_dimer_method(testdir):
    # Set up a small "slab" with an adatoms
    atoms = fcc100('Pt', size=(2, 2, 1), vacuum=10.0)
    add_adsorbate(atoms, 'Pt', 1.611, 'hollow')

    # Freeze the "slab"
    mask = [atom.tag > 0 for atom in atoms]
    atoms.set_constraint(FixAtoms(mask=mask))

    # Calculate using EMT
    atoms.calc = EMT()
    atoms.get_potential_energy()

    # Set up the dimer
    with DimerControl(initial_eigenmode_method='displacement',
                      displacement_method='vector', logfile=None,
                      mask=[0, 0, 0, 0, 1]) as d_control:
        d_atoms = MinModeAtoms(atoms, d_control)

        # Displace the atoms
        displacement_vector = [[0.0] * 3] * 5
        displacement_vector[-1][1] = -0.1
        d_atoms.displace(displacement_vector=displacement_vector)

        # Converge to a saddle point
        with MinModeTranslate(d_atoms, trajectory='dimer_method.xyz',
                              logfile='logfile_i_guess.txt') as dim_rlx:
            dim_rlx.run(fmax=0.001)


# -

test_dimer_method("./dimer_method_ase_stuff/")

t = ASETraj("./dimer_method.traj")

t[-1].write("ts_ase.xyz")

# # try on my shit

# +
from xtb.ase.calculator import XTB
from xtb.interface import Calculator
from xtb.utils import get_method

from ase.atoms import Atoms

from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.NEB import Chain, Node3D

import matplotlib.pyplot as plt
# -

traj = Trajectory.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement/traj_23-23_0_neb.xyz")

chain = Chain.from_traj(traj, k=1,delta_k=1,step_size=1,node_class=Node3D)

plt.plot(chain.energies, 'o--')

tdstruct = traj[10]
coords = tdstruct.coords
atoms = Atoms(
    symbols=tdstruct.symbols.tolist(),
    positions=coords
    # charges=0.0,
)

# Calculate using EMT
atoms.calc = XTB(method="GFN2-xTB", accuracy=0.1)
atoms.get_potential_energy()

# Set up the dimer
with DimerControl(initial_eigenmode_method='displacement',
                  displacement_method='vector', logfile=None) as d_control:
    d_atoms = MinModeAtoms(atoms, d_control)

    # Displace the atoms
    displacement_vector = [[0.0] * 3] * 14
    displacement_vector[-1][1] = -0.1
    d_atoms.displace(displacement_vector=displacement_vector)

    # Converge to a saddle point
    with MinModeTranslate(d_atoms, trajectory='dimer_method_CR.traj',
                          logfile='logfile_CR.txt') as dim_rlx:
        dim_rlx.run(fmax=0.001)


