# +
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)
# -

grad_tanh = grad(jnp.tanh)
print(grad_tanh(2.0))


grad_tanh(0.3)

# # Geodesic Cost function

# +
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from xtb.interface import Calculator, XTBException
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method
from retropaths.abinitio.abinitio import ANGSTROM_TO_BOHR


from chemcloud import CCClient
from chemcloud.models import AtomicInput, OptimizationInput, QCInputSpecification
from chemcloud.models import Molecule as TCMolecule
# -

t = Trajectory.from_xyz("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_0-0_0_cneb.xyz")

start = t[0].copy()
end = t[-1].copy()

gi = Trajectory([start,end])
gi = gi.run_geodesic(nimages=10)

gi.draw();

client = CCClient()


def _as_tc_molecule(coords):
    d = {"symbols":start.symbols, "geometry":coords, "molecular_multiplicity":1, "molecular_charge":0}
    tc_mol = TCMolecule.from_data(d)
    return tc_mol


def energy_tc(list_of_coords):
    tc_mols = [_as_tc_molecule(c) for c in list_of_coords]
    atomic_inputs = [AtomicInput(molecule=tcm, model={"method": "gfn2xtb", "basis": "gfn2xtb"}, driver="energy") for tcm in tc_mols]
    future_result = client.compute(atomic_inputs, engine="terachem_fe")
    result = future_result.get()
    return [r.return_result for r in result]



energy_tc(gi.coords*ANGSTROM_TO_BOHR)


# +
def cost_func(array_of_coords):
    ens = energy_tc(array_of_coords)
    
    eA = (jnp.array(ens) - ens[0])*627.5
    return max(eA)
    
    
    
# -

cost_func(jnp.array(gi.coords*ANGSTROM_TO_BOHR))

g_cost_func = grad(cost_func)

g_cost_func(gi.coords*ANGSTROM_TO_BOHR)

heyo = jnp.array(gi.coords[0])


natoms = len(start.symbols)

name=' '
string = ''
string += f'{natoms}\n{name}\n'
for i in range(natoms):
    string += f'{start.symbols[i]} {heyo[i,0]} {heyo[i,1]} {heyo[i,2]} \n'

string

TDStructure.from_xyz_string(string)


