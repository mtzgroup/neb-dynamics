# -*- coding: utf-8 -*-
from neb_dynamics.chain import Chain
import neb_dynamics.chainhelpers as ch
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.TreeNode import TreeNode


# +
# ch.visualize_chain(h.data.initial_chain)
# -

def update_chain_dynamics(chain: Chain, engine, dt, mass=1.0) -> Chain:
    import neb_dynamics.chainhelpers as ch

    grads = engine.compute_gradients(chain)
    grads[0] = np.zeros_like(grads[0])
    grads[-1] = np.zeros_like(grads[0])
    enes = engine.compute_energies(chain)

    new_vel = chain.velocity
    new_vel += 0.5 * -1*grads * dt / mass
    position = chain.coordinates
    ref_start = position[0]
    ref_end = position[-1]
    new_position = position + new_vel * dt
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


from qcop.exceptions import ExternalProgramError


def run_dynamics(chain, max_steps, engine, dt=0.1):
    """
    Runs dynamics on the chain.
    """
    chain_trajectory = [chain]
    nsteps = 1
    chain_previous = chain
    while nsteps < max_steps + 1:            
        try:
            new_chain = update_chain_dynamics(chain=chain_previous, engine=engine, dt=dt)
        except ExternalProgramError:
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
        
        nsteps+=1
        chain_previous = new_chain
    return chain_trajectory

from neb_dynamics.engines import QCOPEngine, ASEEngine
import numpy as np

from xtb.ase.calculator import XTB

h = TreeNode.read_from_disk('/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Wittig/debug_msmep/')

vel_dim = 5.0
chain = h.data.initial_chain
# chain = h.output_chain
initial_vel = np.random.random(size=len(chain.coordinates.flatten()))
initial_vel = initial_vel.reshape(chain.coordinates.shape) / np.linalg.norm(initial_vel)
chain.velocity = initial_vel*vel_dim
for node in chain:
    node._cached_energy = None
    node._cached_gradient = None
    node._cached_result = None

chain.parameters.k = 0.0
chain.parameters.delta_k=0.0

eng = ASEEngine(calculator=XTB())

ct = run_dynamics(chain=chain, max_steps=500, engine=eng, dt=0.01)

ch.plot_opt_history(ct, do_3d=True)

ch.visualize_chain(ct[150])
# out_tr = eng.compute_geometry_optimization(ct[-1][3])

# +
# ch.visualize_chain(out_tr)
# -

from neb_dynamics.msmep import MSMEP

m = MSMEP(engine=eng)

history = m.find_mep_multistep(input_chain=ct[-1])



















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





