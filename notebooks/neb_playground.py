# -*- coding: utf-8 -*-
# +
from neb_dynamics.chain import Chain
import neb_dynamics.chainhelpers as ch
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.TreeNode import TreeNode

from neb_dynamics.inputs import NEBInputs, ChainInputs, GIInputs
from neb_dynamics.msmep import MSMEP
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics.helper_functions import RMSD

import matplotlib.pyplot as plt

from scipy.signal import argrelmin
# -

from dataclasses import dataclass

# +
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed

@dataclass
class ChainBiaser:
    reference_chains: list  # list of Chain objects

    amplitude: float = 1.0
    sigma: float = 1.0

    distance_func: str = "simp_frechet"
    

    def node_wise_distance(self, chain):
        if self.distance_func == "per_node":
            tot_dist = sum(
                np.linalg.norm(abs(n1 - n2)) if not chain[0].is_a_molecule else RMSD(n1, n2)[0]
                for reference_chain in self.reference_chains
                for n1, n2 in zip(chain.coordinates[1:-1], reference_chain.coordinates[1:-1])
            )
        elif self.distance_func == "simp_frechet":
            tot_dist = sum(self._simp_frechet_helper(coords) for coords in chain.coordinates[1:-1])
        else:
            raise ValueError(f"Invalid distance func name: {self.distance_func}. Available are: {AVAIL_DIST_FUNC_NAMES}")
        
        return tot_dist / len(chain)

    def _simp_frechet_helper(self, coords):
        node_distances = np.array([
            min(
                np.linalg.norm(coords - ref_coord) if len(ref_coord.shape) == 1 else RMSD(coords, ref_coord)[0]
                for ref_coord in reference_chain.coordinates[1:-1]
            )
            for reference_chain in self.reference_chains
        ])
        return node_distances.min()

    def path_bias(self, distance):
        return self.amplitude * np.exp(-(distance**2) / (2 * self.sigma**2))

    def chain_bias(self, chain):
        dist_to_chain = self.node_wise_distance(chain)
        return self.path_bias(dist_to_chain)

    def grad_node_bias(self, chain, node, ind_node, dr=0.1):
        directions = ["x", "y", "z"]
        n_atoms = len(node.coords)
        grads = np.zeros((n_atoms, len(directions)))

        for i in range(n_atoms):
            for j in range(len(directions)):
                disp_vector = np.zeros((n_atoms, len(directions)))
                disp_vector[i, j] = dr

                displaced_coord = node.coords + disp_vector
                node_disp_direction = node.update_coords(displaced_coord)
                fake_chain = chain.copy()
                fake_chain.nodes[ind_node] = node_disp_direction

                grad_direction = self.chain_bias(fake_chain) - self.chain_bias(chain)
                grads[i, j] = grad_direction


        return grads / dr

    def grad_chain_bias(self, chain):
        grad_bias = grad_chain_bias_function(chain, self.grad_node_bias)
        mass_weights = ch._get_mass_weights(chain)
        
        grad_bias =  grad_bias * mass_weights.reshape(-1, 1)
        return grad_bias

def grad_chain_bias_function(chain, grad_node_bias_fn):
    return np.array(Parallel(n_jobs=-1)(delayed(grad_node_bias_fn)(chain, node, ind_node)
                         for ind_node, node in enumerate(chain[1:-1], start=1)))


# -

def update_chain_dynamics(chain: Chain, engine, dt, mass=1.0, biaser: ChainBiaser = None) -> Chain:
    import neb_dynamics.chainhelpers as ch

    grads = engine.compute_gradients(chain)
    grads[0] = np.zeros_like(grads[0])
    # grads[-1] = np.zeros_like(grads[0])
    enes = engine.compute_energies(chain)

    new_vel = chain.velocity
    new_vel += 0.5 * -1*grads * dt / mass
    position = chain.coordinates
    ref_start = position[0]
    ref_end = position[-1]
    new_position = position + new_vel * dt
    if biaser:
        grad_bias = biaser.grad_chain_bias(chain)
        new_position[1:-1] -= grad_bias
    new_position[0] = ref_start
    new_position[-1] = ref_end
    new_chain = chain.copy()
    new_nodes = []
    for coords, node in zip(new_position, new_chain):
        new_nodes.append(node.update_coords(coords))
    new_chain.nodes = new_nodes
    
    grads = engine.compute_gradients(new_chain)
    grads[0] = np.zeros_like(grads[0])
    grads[-1] = np.zeros_like(grads[0])
    new_vel +=  0.5 * -1*grads * dt / mass
    new_chain.velocity = new_vel

    return new_chain


# +

from neb_dynamics.convergence_helpers import chain_converged


# -

def run_dynamics(chain, max_steps, engine, neb_inputs: NEBInputs, dt=0.1, biaser: ChainBiaser = None):
    """
    Runs dynamics on the chain.
    """
    chain_trajectory = [chain]
    nsteps = 1
    chain_previous = chain
    while nsteps < max_steps + 1:            
        # try:
        new_chain = update_chain_dynamics(chain=chain_previous, engine=engine, dt=dt, biaser=biaser)
        # except Exception:
            # print("Electronic structure error")
            # return chain_trajectory
            # raise ElectronicStructureError(msg="QCOP failed.")

        max_rms_grad_val = np.amax(new_chain.rms_gperps)
        ind_ts_guess = np.argmax(new_chain.energies)
        ts_guess_grad = np.amax(np.abs(ch.get_g_perps(new_chain)[ind_ts_guess]))
        

        print(
            f"step {nsteps} // argmax(|TS gperp|) {np.amax(np.abs(ts_guess_grad))} // \
                max rms grad {max_rms_grad_val} // armax(|TS_triplet_gsprings|) \
                    {new_chain.ts_triplet_gspring_infnorm} // |velocity|={np.linalg.norm(new_chain.velocity)}//{' '*20}",
            end="\r",
        )
        chain_trajectory.append(new_chain)
        # if chain_converged(chain_prev=chain_previous, chain_new=new_chain, parameters=neb_inputs):
        #     print("Converged!!!")
        #     return chain_trajectory
        
        nsteps+=1
        chain_previous = new_chain
    return chain_trajectory

import numpy as np

from qcop.exceptions import ExternalProgramError

from neb_dynamics.engines import QCOPEngine, ASEEngine
import numpy as np

from xtb.ase.calculator import XTB

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Claisen-Rearrangement/debug2_msmep/")

ref_chain = h.output_chain

ref_chain = h.output_chain.copy()
ref_chain.nodes = [ref_chain[0], ref_chain.get_ts_node(), ref_chain[-1]]
bias = ChainBiaser(reference_chains=[ref_chain], amplitude=1)

# %%time
bias.grad_chain_bias(h.data.initial_chain)

# +
vel_dim = 5.0
chain = h.data.initial_chain
# chain = ch.run_geodesic(chain, nimages=40)
# chain = h.output_chain
mass_weights = ch._get_mass_weights(chain)
initial_vel = np.random.random(size=len(chain.coordinates.flatten()))

initial_vel = initial_vel.reshape(chain.coordinates.shape) #
initial_vel = initial_vel * mass_weights.reshape(-1, 1)
initial_vel = initial_vel / np.linalg.norm(initial_vel)


chain.velocity = initial_vel*vel_dim
for node in chain:
    node.converged = False
    node._cached_energy = None
    node._cached_gradient = None
    node._cached_result = None
# -

chain.parameters.k = 0.01
chain.parameters.delta_k=0.0

eng = ASEEngine(calculator=XTB())
# eng = QCOPEngine()

nbi = NEBInputs(v=True, early_stop_force_thre=0.03, skip_identical_graphs=False, node_rms_thre=5, node_ene_thre=1e10)
cni = ChainInputs(friction_optimal_gi=False, k=0.1, delta_k=0.09)
gii = GIInputs(nimages=len(h.data.initial_chain))
opt = VelocityProjectedOptimizer()
m = MSMEP(engine=eng, neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii)

# init_guess_tr = run_dynamics(chain, max_steps=500, dt=0.005, engine=eng, biaser=bias, neb_inputs=nbi)
init_guess_tr = run_dynamics(chain, max_steps=1000, dt=0.005, engine=eng, biaser=None, neb_inputs=nbi)


# +
def chain_energy(chain):
    # return sum(chain.energies)
    return np.linalg.norm(ch.compute_NEB_gradient(chain))

enes = [chain_energy(c) for c in init_guess_tr]



inds_mins, = argrelmin(np.array(enes))

plt.plot(enes)
print(inds_mins)

# +
# ch.visualize_chain(init_guess_tr[inds_mins[1]])
# ch.visualize_chain(init_guess_tr[909])

# +
init_c = init_guess_tr[inds_mins[0]].copy()
for node in init_c:
    node.converged = False
    node._cached_energy = None
    node._cached_gradient = None
    node._cached_result = None

eng.compute_gradients(init_c)
history0 = m.find_mep_multistep(init_c)
# -

history1 = m.find_mep_multistep(init_guess_tr[inds_mins[1]])

from qcio import view

view.view(h.output_chain.get_ts_node().structure,n.optimized.get_ts_node().structure) 

ch.visualize_chain(n.optimized)

n.plot_opt_history(1)

from qcio import DualProgramInput

from chemcloud import CCClient

client = CCClient()

# +
prog_inp = DualProgramInput(
    structure=n.optimized[9].structure,
    calctype="hessian",
    subprogram="terachem",
    subprogram_args={"model": {"method": "b3lyp", "basis": ""}},
)


# Submit calculation
future_result = client.compute("bigchem", prog_inp)

# -

prog_output = future_result.get()


prog_output

n.optimized[9].structure



# ### notes: 
# * 'dt' is path length dependent
# * TotalChainEnergy is a bad objective function cause you can cheat by having all nodes be in a single well

out_tr = eng.compute_geometry_optimization(h.output_chain[-1])

ch.visualize_chain(out_tr)

all_results = []



def _reset_chain(chain):
    for node in chain:
        node._cached_gradient = None
        node._cached_result = None
        node._cached_energy = None


orig_init = h.data.initial_chain
_reset_chain(orig_init)

for init_guess in [ct[ind] for ind in inds_mins]:
    _reset_chain(init_guess)
    history = m.find_mep_multistep(init_guess)
    all_results.append(history)

len(all_results)

ch.visualize_chain(all_results[0].output_chain)

eng2 = ASEEngine(calculator=XTB(), ase_opt_str='LBFGS')

out_tr[-1].structure.save("/home/jdep/T3D_data/ladderane/dynamics_attempt/end_preclaisen.xyz")

ct[-1][0].structure.save("/home/jdep/T3D_data/ladderane/dynamics_attempt/start.xyz")

# out_tr = eng.compute_geometry_optimization(ct[-1][30])
out_tr = eng2.compute_geometry_optimization(ct[-1][32])

len(out_tr)

# +
# ch.visualize_chain(out_tr)
# -

from neb_dynamics.msmep import MSMEP

m = MSMEP(engine=eng)

history = m.find_mep_multistep(input_chain=ct[-1])

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/pygsm/")

ch.visualize_chain(h.output_chain)















# +
"""An example of how to create a FileInput object for a QC Program."""

from pathlib import Path

from qcio import FileInput, Structure

from qcop import compute
from chemcloud import CCClient

# Input files for QC Program
inp_file = Path("./tc.in").read_text()  # Or your own function to create tc.in

# Structure object to XYZ file
structure = Structure.from_smiles("O.O.O", program='openbabel')
xyz_str = structure.to_xyz()  # type: ignore

# Create a FileInput object for TeraChem
file_inp = FileInput(
    files={"tc.in": inp_file, "coords.xyz": xyz_str}, cmdline_args=["tc.in"]
)

# This will write the files to disk in a temporary directory and then run
# "terachem tc.in" in that directory.
# output = compute("terachem", file_inp, print_stdout=True)
client = CCClient()
future_output = client.compute("terachem", file_inp)

output = future_output.get()

# Data
output.stdout
output.input_data
output.results.files  # Has all the files terachem creates
output.results.files.keys()  # Print out file names


output.pstdout


# -
from qcio import Structure

from pathlib import Path


def create_submission_scripts(
    out_dir: Path,
    out_name: str, 
    submission_dir: Path,
    reference_dir: Path,
    reference_start_name: str = 'start.xyz',
    reference_end_name: str = 'start.xyz',
    prog: str = 'xtb',
    eng: str = 'qcop',
    sig: int = 1,
    nrt: float = 1,  # node rms threshold
    net: float = 1,  # node ene threshold
    
    ):

    if out_name is None:
        out_name = reference_start_name.stem
    template = [
    "#!/bin/bash",
    "",
    "#SBATCH -t 12:00:00",
    "#SBATCH -J nebjan_test",
    "#SBATCH --qos=gpu_short",
    "#SBATCH --gres=gpu:1",
    "",
    "work_dir=$PWD",
    "",
    "cd $SCRATCH",
    "",
    "# Load modules",
    "ml TeraChem",
    "source /home/jdep/.bashrc",
    "source activate neb",
    "export OMP_NUM_THREADS=1",
    "# Run the job",
    "create_msmep_from_endpoints.py ",
    ]
    
    start_fp = reference_dir / reference_start_name
    end_fp = reference_dir / reference_end_name
    
    out_fp = out_dir / out_name
    
    cmd = f"/home/jdep/.cache/pypoetry/virtualenvs/neb-dynamics-G7__rX26-py3.9/bin/python /home/jdep/neb_dynamics/neb_dynamics/scripts/create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.002 \
        -sig {sig} -nimg 12 -min_ends 1 \
        -es_ft 0.03 -name {out_fp} -prog {prog} -eng {eng} -node_rms_thre {nrt} -node_ene_thre {net} &> {out_fp}_out"
    
    new_template = template.copy()
    new_template[-1] = cmd
    
    with open(
        submission_dir / out_name , "w+"
    ) as f:
        f.write("\n".join(new_template))


ref_dir = Path("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures")
all_rns = [p.stem for p in ref_dir.glob("*")]

# rn = '1-2-Amide-Phthalamide-Synthesis'
for rn in all_rns:
    create_submission_scripts(
        out_dir=Path('/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/results_asneb'),
        out_name=rn,
        submission_dir=Path('/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/submissions_dir'), 
        reference_dir=Path(f'/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rn}'),
        reference_start_name='start_raw.xyz', reference_end_name='end_raw.xyz',
        nrt=10, net=10
    )





