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

# +

from pathlib import Path

from qcio import FileInput, Structure

from qcop import compute
from chemcloud import CCClient

# Input files for QC Program
inp_file = Path("./tc2.in").read_text()  # Or your own function to create tc.in

# Structure object to XYZ file
# structure = Structure.from_smiles("O.O.O", program='openbabel')
structure = Structure.open("/home/jdep/T3D_data/DavidDumas_Cesium/end.xyz")
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


# output.pstdout

# -

output.results.files.keys()

tempout = open("/tmp/foobar.xyz","w+")
tempout.write(output.results.files['scr.coords/coors.xyz'])
tempout.close()

from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file

from neb_dynamics import StructureNode

tr = read_multiple_structure_from_file('/tmp/foobar.xyz')
nodes = [StructureNode(structure=a) for a in tr]

ch.visualize_chain(nodes)

from neb_dynamics.helper_functions import _create_df

import os
from pathlib import Path

# +
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed

@dataclass
class ChainBiaser:
    AVAIL_DIST_FUNC_NAMES = ["per_node", "simp_frechet"]
    reference_chains: list  # list of Chain objects

    amplitude: float = 1.0
    sigma: float = 1.0

    distance_func: str = "simp_frechet"
    

    def node_wise_distance(self, chain):
        if self.distance_func == "per_node":
            tot_dist = 0
            for reference_chain in self.reference_chains:
                for n1, n2 in zip(chain.coordinates[1:-1], reference_chain.coordinates[1:-1]):
                    tot_dist+= RMSD(n1, n2)[0]
                    
        elif self.distance_func == "simp_frechet":
            tot_dist = 0
            # tot_dist = sum(self._simp_frechet_helper(coords) for coords in chain.coordinates[1:-1])
            for ref_chain in self.reference_chains:
                tot_dist += self.frechet_distance(path1=chain.coordinates, path2=ref_chain.coordinates)
        else:
            raise ValueError(f"Invalid distance func name: {self.distance_func}. Available are: {self.AVAIL_DIST_FUNC_NAMES}")
        
        return tot_dist / len(chain)

    def euclidean_distance(self, p1, p2):
        """Calculates the Euclidean distance between two points p1 and p2."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def frechet_distance(self, path1, path2):
        """Calculates the Fréchet distance between two paths."""
        n, m = len(path1), len(path2)
        ca = np.full((n, m), -1.0)  # Memoization table
    
        def recursive_calculation(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            elif i == 0 and j == 0:
                ca[i, j] = self.euclidean_distance(path1[0], path2[0])
            elif i > 0 and j == 0:
                ca[i, j] = max(recursive_calculation(i - 1, 0), self.euclidean_distance(path1[i], path2[0]))
            elif i == 0 and j > 0:
                ca[i, j] = max(recursive_calculation(0, j - 1), self.euclidean_distance(path1[0], path2[j]))
            elif i > 0 and j > 0:
                ca[i, j] = max(
                    min(
                        recursive_calculation(i - 1, j),
                        recursive_calculation(i - 1, j - 1),
                        recursive_calculation(i, j - 1)
                    ),
                    self.euclidean_distance(path1[i], path2[j])
                )
            else:
                ca[i, j] = float('inf')
            return ca[i, j]
    
        return recursive_calculation(n - 1, m - 1)

    
    
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
        energy_weights = chain.energies_kcalmol[1:-1]
        grad_bias =  grad_bias * mass_weights.reshape(-1, 1)
        out_arr = [grad*weight for grad, weight in zip(grad_bias, energy_weights)]
        grad_bias = np.array(out_arr)

        
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
        energy_weights= chain.energies_kcalmol[1:-1]
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
        try:
            new_chain = update_chain_dynamics(chain=chain_previous, engine=engine, dt=dt, biaser=biaser)
        except Exception:
            print("Electronic structure error")
            return chain_trajectory
            raise ElectronicStructureError(msg="QCOP failed.")

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

ALL_DYNAMICS = []

bias = ChainBiaser(distance_func='simp_frechet', reference_chains=[h.output_chain], amplitude=5, sigma=3)

# %%time
bias.grad_chain_bias(h.output_chain)

# +
vel_dim = 0.0
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

chain.parameters.k = 0.1
chain.parameters.delta_k=0.09

eng = ASEEngine(calculator=XTB())
# eng = QCOPEngine()

nbi = NEBInputs(v=True, early_stop_force_thre=0.03, skip_identical_graphs=False, node_rms_thre=5, node_ene_thre=1)
cni = ChainInputs(friction_optimal_gi=False, k=0.1, delta_k=0.09)
gii = GIInputs(nimages=len(h.data.initial_chain))
opt = VelocityProjectedOptimizer()
m = MSMEP(engine=eng, neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii)

init_guess_tr = run_dynamics(chain, max_steps=500, dt=0.1, engine=eng, biaser=bias, neb_inputs=nbi)
# init_guess_tr = run_dynamics(chain, max_steps=1000, dt=0.005, engine=eng, biaser=None, neb_inputs=nbi)
# -

# bias2 = ChainBiaser(distance_func='simp_frechet', reference_chains=[h.output_chain, init_guess_tr[-1]], amplitude=5)
# init_guess_tr2 = run_dynamics(chain, max_steps=500, dt=0.005, engine=eng, biaser=bias2, neb_inputs=nbi)


ch.plot_opt_history(init_guess_tr,1)

N_STARTS = 10

for i in range(N_STARTS):
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
    
    chain.parameters.k = 0.01
    chain.parameters.delta_k=0.0
    
    eng = ASEEngine(calculator=XTB())
    # eng = QCOPEngine()
    
    nbi = NEBInputs(v=True, early_stop_force_thre=0.03, skip_identical_graphs=False, node_rms_thre=5, node_ene_thre=1)
    cni = ChainInputs(friction_optimal_gi=False, k=0.1, delta_k=0.09)
    gii = GIInputs(nimages=len(h.data.initial_chain))
    opt = VelocityProjectedOptimizer()
    m = MSMEP(engine=eng, neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii)
    
    # init_guess_tr = run_dynamics(chain, max_steps=500, dt=0.005, engine=eng, biaser=bias, neb_inputs=nbi)
    init_guess_tr = run_dynamics(chain, max_steps=1000, dt=0.005, engine=eng, biaser=None, neb_inputs=nbi)
    ALL_DYNAMICS.append(init_guess_tr)

# +
eng = ASEEngine(calculator=XTB())
# eng = QCOPEngine()

nbi = NEBInputs(v=True, early_stop_force_thre=0.0, skip_identical_graphs=False, node_rms_thre=5, node_ene_thre=1)
cni = ChainInputs(friction_optimal_gi=False, k=0.1, delta_k=0.09)
gii = GIInputs(nimages=len(h.data.initial_chain))
opt = VelocityProjectedOptimizer()
m = MSMEP(engine=eng, neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii)
# -

n, es_res = m.minimize_chain(init_guess_tr[-10])

ch.visualize_chain(n.chain_trajectory[-1])

ch.plot_opt_history(ALL_DYNAMICS[2], 1)



# +
def chain_energy(chain):
    # return sum(chain.energies)
    return np.linalg.norm(ch.compute_NEB_gradient(chain))

all_init_guesses = []
f, ax = plt.subplots()
# for tr in ALL_DYNAMICS:
for tr in [init_guess_tr]:
    enes = [chain_energy(c) for c in tr]
    
    
    
    inds_mins, = argrelmin(np.array(enes))
    for ind in inds_mins:
        all_init_guesses.append(tr[ind])
    
    plt.plot(enes)
plt.show()
# -

all_histories = []
for init_c in all_init_guesses:
    n, es_res = m.minimize_chain(init_c)
    all_histories.append(n)

all_histories[10].plot_opt_history()

# + active=""
# from qcio import Program
# -

prog_inp = DualProgramInput(
    structure=outputs[12].results.final_structure,
    calctype="hessian",
    subprogram="terachem",
    subprogram_args={"model": {"method": "b3lyp", "basis": "6-31g"}},
)

future_res = client.compute('bigchem', prog_inp)

huh = future_res.get()



dr = 1
ts = outputs[12].results.final_structure
ts_plus_coords = huh.results.normal_modes_cartesian[0]*dr + ts.geometry
ts_minus_coords = huh.results.normal_modes_cartesian[0]*(-1)*dr + ts.geometry

from neb_dynamics.nodes.node import StructureNode

ts = StructureNode(structure=ts)
ts_plus = ts.update_coords(ts_plus_coords)
ts_minus = ts.update_coords(ts_minus_coords)

pi = ProgramInput(structure=ts.structure, model={"method": "b3lyp", "basis": "6-31g"}, calctype='energy')
eng  = QCOPEngine(program_input=pi, program='terachem', compute_program='chemcloud')

opts = []
for node in [ts_minus, ts_plus]:
    opts.append(eng.compute_geometry_optimization(node))

all_program_inputs = []
for n in all_histories:
    ts_node = n.chain_trajectory[-1].get_ts_node()
    structure = ts_node.structure
    prog_inp = DualProgramInput(
        structure=structure,
        calctype="transition_state",
        keywords={'maxiter': 50, 'transition': True},
        subprogram="terachem",
        subprogram_args={"model": {"method": "b3lyp", "basis": "6-31g"}},
    )
    all_program_inputs.append(prog_inp)

res = client.compute('geometric', all_program_inputs)

outputs = res.get()

from qcio import view

outputs[2].success

view.view(outputs[12].results.final_structure)

outputs[0].input_data

ch.visualize_chain(all_histories[8].output_chain)

from qcio import view

view.view(h.output_chain.get_ts_node().structure,n.optimized.get_ts_node().structure) 

ch.visualize_chain(n.optimized)

n.plot_opt_history(1)

# ### notes: 
# * 'dt' is path length dependent
# * TotalChainEnergy is a bad objective function cause you can cheat by having all nodes be in a single well

from qcio import Structure

from pathlib import Path


def create_submission_scripts(
    out_dir: Path,
    out_name: str, 
    submission_dir: Path,
    reference_dir: Path,
    reference_start_name: str = 'start.xyz',
    reference_end_name: str = 'end.xyz',
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


# ref_dir = Path("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures")
ref_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures")
all_rns = [Path(p).stem for p in open("/home/jdep/T3D_data/msmep_draft/comparisons_dft/reactions_todo_multistep.txt").read().splitlines()]
# all_rns = [p.stem for p in ref_dir.glob("*") if p.stem in ]

import os

# rn = '1-2-Amide-Phthalamide-Synthesis'
rns_to_submit = []
for rn in all_rns:
    # out_dir = Path('/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/results_asneb')
    out_dir = Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb')
    # if (out_dir/f'{rn}_out').exists():
    #     continue
    # else:
    # print(rn)
    # rns_to_submit.append(Path('/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/submissions_dir')/rn)
    rns_to_submit.append(Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/submissions_dir')/rn)
    create_submission_scripts(
        out_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb'),
        out_name=rn,
        submission_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/submissions_dir'), 
        reference_dir=Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}'),
        reference_start_name='start_xtb.xyz', reference_end_name='end_xtb.xyz',
        prog='terachem',
        nrt=1, net=1
    )

from neb_dynamics.TreeNode import TreeNode


h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/results_asneb/Alcohol-Bromination/")


import neb_dynamics.chainhelpers as ch
from neb_dynamics.elementarystep import check_if_elem_step

ch.visualize_chain(h.output_chain)

from neb_dynamics.engines import QCOPEngine
from qcio import ProgramInput, Structure

pi = ProgramInput(structure=Structure.from_smiles("O"), 
                  model={'method':'ub3lyp','basis':'3-21gs'}, 
                  keywords={'pcm':'cosmo','epsilon':80.4},
                 calctype='energy')
eng = QCOPEngine(program_input=pi, program='terachem')

output = check_if_elem_step(h.output_chain, engine=eng)

# +

h = TreeNode.read_from_disk("/home/jdep/T3D_data/ladderane/bugfixed/")
# -

ch.visualize_chain(h.output_chain)

sub_chain = h.output_chain.copy()

sub_nodes = sub_chain.nodes[:40]

sub_chain.nodes = sub_nodes

ch.visualize_chain(sub_chain)

eng2 = ASEEngine(ase_opt_str='MDMin', calculator=XTB())

out_tr = eng2.compute_geometry_optimization(sub_chain[6])

ch.visualize_chain(Chain(out_tr, ChainInputs()))

from neb_dynamics.nodes.nodehelpers import is_identical

is_identical(h.output_chain[-1], output.minimization_results[-1])



# ?h.output_chain[-1].__eq__

view.view(*[node.structure for node in output.minimization_results])

view.view(output.minimization_results[0].structure, h.output_chain[0].structure)

# +

output.minimization_results[0] == 

# +

from qcio import view
# -

for rn in rns_to_submit:
    os.system(f'sbatch {str(rn.resolve())}')



