# -*- coding: utf-8 -*-
from dataclasses import dataclass

from qcio import ProgramInput
from neb_dynamics import NEBInputs, ChainInputs, GIInputs
from neb_dynamics import QCOPEngine
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer

from neb_dynamics.inputs import RunInputs

inputs = RunInputs(engine_name='ase',program='xtb', path_min_method='neb', chain_inputs={'friction_optimal_gi':True}, path_min_inputs={'path_resolution':200},
                  gi_inputs={'nimages':20})

from neb_dynamics import MSMEP

m = MSMEP(inputs=inputs)

from neb_dynamics.nodes.nodehelpers import create_pairs_from_smiles

a = 'CCl.[Br-]'
b = 'CBr.[Cl-]'


a_s, b_s = create_pairs_from_smiles(a,b)

from qcio import view

from neb_dynamics import StructureNode

react = StructureNode(structure=a_s)
prod = StructureNode(structure=b_s)

h = m.run_recursive_minimize(input_chain=[react, prod])

import neb_dynamics.chainhelpers as ch

ch.visualize_chain(h.output_chain)

# +
    from types import SimpleNamespace
    from qcio import ProgramInput, ProgramArgs
    from xtb.ase.calculator import XTB
    from neb_dynamics import ASEEngine
    
    import json

    @dataclass 
    class RunInputs:
        engine: str
        program: str
    
        path_min: str = 'NEB'
        path_min_kwds: dict = None
    
        chain_inputs: dict = None
        
        program_kwds: ProgramArgs = None
        gi_inputs: dict = None
        optimizer_kwds: dict = None
    
    
        def __post_init__(self):
            
            if self.path_min.upper() == "NEB":
                default_kwds = NEBInputs().__dict__
            
            elif self.path_min.upper() == "FSM":
                default_kwds = {
                "stepsize": 0.5,
                "ngradcalls": 3,
                "max_cycles": 500,
                "path_resolution": 1 / 10,  # BOHR,
                "max_atom_displacement": 0.1,
                "early_stop_scaling": 3,
                "use_geodesic_tangent": True,
                "dist_err": 0.1,
                "min_images": 4,
                "distance_metric": "GEODESIC",
                "verbosity": 1,
            }
            #     default_kwds = FSMInputs()
            elif self.path_min.upper() == "PYGSM":
                default_kwds = PYGSMInputs()
            
            if self.path_min_kwds is None:
                self.path_min_kwds = default_kwds
            
            else:
                for key, val in self.path_min_kwds.items():
                    default_kwds[key] = val
                    
                self.path_min_kwds = SimpleNamespace(**default_kwds)
    
    
            if self.gi_inputs is None:
                self.gi_inputs = GIInputs()
            else:
                self.gi_inputs = GIInputs(**self.gi_inputs)
    
            if self.program_kwds is None:
                if self.program == "xtb":
                    program_args = ProgramArgs(
                        model={"method": "GFN2xTB"},
                        keywords={})
    
                elif "terachem" in self.program:
                    program_args = ProgramArgs(
                        model={"method": "ub3lyp", "basis": "3-21g"},
                        keywords={})
                else:
                    raise ValueError("Need to specify program arguments")
            else:
                program_args = ProgramArgs(**self.program_kwds)
            self.program_kwds = program_args
    
            if self.chain_inputs is None:
                self.chain_inputs = ChainInputs()
            
            else:
                self.chain_inputs = ChainInputs(**self.chain_inputs)
    
    
            if self.optimizer_kwds is None:
                self.optimizer_kwds = {"timestep": 1.0}
    
            
            if self.engine == 'qcop' or self.engine == 'chemcloud':
                eng = QCOPEngine(program_input=self.program_kwds, 
                                 program=self.program, 
                                 compute_program=self.engine
                                )
            elif self.engine == 'ase':
                assert self.program == 'xtb', f"{self.program} not yet supported with ASEEngine"
                calc = XTB()
                eng = ASEEngine(calculator=calc)
            else:
                raise ValueError(f"Unsupported engine: {self.engine}")
    
            self.engine_object = eng
                
                
    
        @classmethod
        def open(cls, fp):
            with open(fp) as f:
                data = f.read()
            data_dict = json.loads(data)
            return cls(**data_dict)
                    
             
        
        
# -

ri = RunInputs.open("/home/jdep/debug/runfile.in")
ri

huh = RunInputs(engine='qcop', program='xtb',  path_min='fsm', path_min_kwds={'distance_metric':'linear'},)
huh

# +
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.neb import NEB
from neb_dynamics import ChainInputs
from neb_dynamics.chain import Chain
import neb_dynamics.chainhelpers as ch
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file
from neb_dynamics import StructureNode

import matplotlib.pyplot as plt
from pathlib import Path

from qcio import Structure, view, ProgramInput

from neb_dynamics.nodes.nodehelpers import displace_by_dr
from neb_dynamics.nodes.node import Node
from neb_dynamics import QCOPEngine



# +
import numpy as np

# Function to compute normalized gradient (reaction path direction) in non-mass-weighted coordinates
def compute_normalized_gradient(gradient):
    norm = np.linalg.norm(gradient)
    if norm == 0:
        raise ValueError("Zero gradient! Cannot normalize.")
    return gradient / norm

# Function to compute the geodesic correction
def geodesic_correction(geometry, new_geometry, step_direction, engine):
    """
    Apply geodesic correction to the new geometry to ensure that it
    follows the IRC tangent direction.
    
    :param geometry: Current geometry (numpy array)
    :param new_geometry: New geometry after taking a step
    :param step_direction: Direction of the previous step (normalized gradient)
    :return: Corrected geometry
    """
    # Compute the gradient at the new geometry
    new_gradient = engine.compute_gradients([new_geometry])[0]
    
    # Project the new gradient onto the previous step direction
    dot = np.dot(new_gradient.flatten(), step_direction.flatten()) 
    projection = dot * step_direction
    print(f"{dot=}")
    
    # Apply geodesic correction by subtracting the projection
    corrected_geometry_coords = new_geometry.coords - projection
    corrected_geometry = new_geometry.update_coords(corrected_geometry_coords)
    
    return corrected_geometry

# Function to take a single IRC step with geodesic correction
def irc_step_with_correction(geometry: Node, step_size, engine, prev_direction=None):
    # Compute gradient and normalized gradient (direction of motion)
    gradient = engine.compute_gradients([geometry])[0]
    direction = compute_normalized_gradient(gradient)
    
    # Take an initial IRC step along the normalized gradient
    new_geometry_coords = geometry.coords - step_size * direction
    new_geometry = geometry.update_coords(new_geometry_coords)

    
    # If there's a previous step direction, apply geodesic correction
    if prev_direction is not None:
        new_geometry = geodesic_correction(geometry=geometry, new_geometry=new_geometry, step_direction=prev_direction, engine=engine)
    
    return new_geometry, direction

# Schlegel-Gonzalez IRC integration algorithm with geodesic correction
def schlegel_gonzalez_irc_with_geodesic(geometry, engine, step_size=0.01, max_steps=100):
    """
    Integrates the IRC path using the Schlegel-Gonzalez algorithm
    with geodesic corrections in non-mass-weighted coordinates.

    :param geometry: Initial geometry (coordinates) as a numpy array
    :param step_size: Step size for each IRC integration step
    :param max_steps: Maximum number of steps to integrate
    :return: IRC path as a list of geometries
    """
    irc_path = [geometry]
    prev_direction = None

    for step in range(max_steps):
        # Take an IRC step with geodesic correction
        geometry, prev_direction = irc_step_with_correction(geometry=geometry, step_size=step_size, prev_direction=prev_direction, engine=engine)
        engine.compute_gradients([geometry])
        
        # Append the new geometry to the IRC path
        irc_path.append(geometry)
        
        # Optional: Add convergence check or re-adjust step size if necessary
        if np.linalg.norm(geometry.gradient) < 1e-5:
            print(f"Converged after {step+1} steps.")
            break

    return irc_path


# -

def load_qchem_result(path_dir):
    
    path_dir = Path(path_dir)
    path_string = path_dir / 'stringfile.txt'
    structs = read_multiple_structure_from_file(path_string)
    nodes = [StructureNode(structure=s) for s in structs]
    enes_file = path_dir / 'Vfile.txt'
    enes_data = open(enes_file).read().splitlines()[1:]
    enes = [float(line.split()[-2]) for line in enes_data]
    for node, ene in zip(nodes, enes):
        node._cached_energy = ene
    chain = Chain(nodes=nodes) 
    return chain


hcn = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/hcn/fsm_short_neb.xyz")
hcn_linear = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/hcn/fsm_linear_short_neb.xyz")
hcn_qchem = load_qchem_result("/home/jdep/T3D_data/fneb_draft/hcn/fsm.files.1/")

plt.plot(hcn_linear.optimized.energies)
plt.plot(hcn.optimized.energies)

# +
# ch.visualize_chain(hcn_qchem)
# -

ts_hcn_res = eng._compute_ts_result(hcn.optimized.get_ts_node())

ts_hcn_qchem_res = eng._compute_ts_result(hcn_qchem.get_ts_node())

# +
# view.view(ts_hcn_res)
# view.view(ts_hcn_qchem_res)

# +

fsm = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/wittig/fsm_short_geodesic_neb.xyz")
qchem = load_qchem_result("/home/jdep/T3D_data/fneb_draft/wittig/fsm_short.files.1/")
# -

fsm_long = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/claisen/fsm_geodesic_neb.xyz")
qchem_long = load_qchem_result("/home/jdep/T3D_data/fneb_draft/claisen/fsm.files.1/")

plt.plot(fsm.optimized.integrated_path_length, fsm.optimized.energies, 'o-')
plt.plot(qchem.integrated_path_length, qchem.energies, 'o-')

eng = QCOPEngine(program
                 _input=ProgramInput(structure=fsm.optimized.get_ts_node().structure, model={'method':'hf','basis':'6-31g'}, calctype='energy'), compute_program='qcop', program='terachem-pbs')

ts = eng.compute_transition_state(fsm.optimized.get_ts_node())

ts_hcn_res.return_result.save("/home/jdep/T3D_data/fneb_draft/hcn/ts_hf.xyz")

# +
c_backwards = Chain.from_xyz("/home/jdep/T3D_data/fneb_draft/hcn/irc_negative.xyz", ChainInputs())

c_forward = Chain.from_xyz("/home/jdep/T3D_data/fneb_draft/hcn/irc_forward.xyz", ChainInputs())

c_backwards.nodes.reverse()

c_irc = Chain(c_backwards.nodes+[ts_node]+c_forward.nodes)

eng.compute_energies(c_irc)
# -

ts_qchem = eng.compute_transition_state(qchem.get_ts_node())

ts.save("/home/jdep/T3D_data/fneb_draft/claisen/ts_hf.xyz")

ts_node = StructureNode(structure=ts)

ts_node = StructureNode(structure=ts_hcn_res.return_result)
eng.compute_energies([ts_node])


outputs = build_full_irc_chain(ts_node, engine=eng, dr=0.01, step_size=0.001)

rxn_coordinate = ch.get_rxn_coordinate(c_irc)

hcn_qchem[0].energy, hcn.optimized[0].energy, c_irc[0].energy

c_irc[0].structure.save("/home/jdep/T3D_data/fneb_draft/hcn/hcn_reactant.xyz")

c_irc[-1].structure.save("/home/jdep/T3D_data/fneb_draft/hcn/hcn_product.xyz")

c_fsm_linear = hcn_linear.optimized

c_fsm = hcn.optimized
c_qchem = hcn_qchem



def get_projections(c: Chain, eigvec, reference):
    # ind_ts = c.energies.argmax()
    ind_ts = 0

    all_dists = []
    for i, node in enumerate(c):
        # _, aligned_ci = align_geom(c[i].coords, reference)
        # _, aligned_start = align_geom(c[ind_ts].coords, reference)
        # displacement = aligned_ci.flatten() - c[ind_ts].coords.flatten()
        # displacement = aligned_ci.flatten() - aligned_start.flatten()
        displacement = c[i].coords.flatten() - c[ind_ts].coords.flatten()
        # _, disp_aligned = align_geom(displacement.reshape(c[i].coords.shape), eigvec.reshape(c[i].coords.shape))
        all_dists.append(np.dot(displacement, eigvec))
        # all_dists.append(np.dot(disp_aligned.flatten(), eigvec))
                                     
    # plt.plot(all_dists)
    return all_dists


dists_fsm_linear = get_projections(c_fsm_linear, rxn_coordinate, c_irc[0].coords)
dists_fsm = get_projections(c_fsm, rxn_coordinate, c_irc[0].coords)
dists_qchem = get_projections(c_qchem, rxn_coordinate, c_irc[0].coords)
dists_irc = get_projections(c_irc, rxn_coordinate, c_irc[0].coords)

from neb_dynamics.neb import NEB
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics import NEBInputs

# +
# n = NEB(initial_chain=c_fsm, optimizer=VelocityProjectedOptimizer(timestep=0.5), parameters=NEBInputs(v=True, max_steps=100), engine=eng)

# +
# es_res = n.optimize_chain()

# +
# dists_neb = ch.get_projections(n.optimized, rxn_coordinate)

# +
# n.grad_calls_made
# -





# +
s=6
fs=18
f, ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(np.array(dists_fsm), c_fsm.energies_kcalmol, 'o-', label='fsm')
plt.plot(np.array(dists_fsm_linear), c_fsm_linear.energies_kcalmol, 'o-', label='fsm(linear)')
# plt.plot(np.array(dists_qchem), c_qchem.energies_kcalmol, 'o-', label='qchem')
plt.plot(np.array(dists_irc), c_irc.energies_kcalmol, label='irc', color='black', lw=3)

# plt.plot(c_fsm.integrated_path_length, c_fsm.energies_kcalmol, 'o-', label='fsm')
# plt.plot(c_qchem.integrated_path_length, c_qchem.energies_kcalmol, 'o-', label='qchem')
# plt.plot(c_irc.integrated_path_length, c_irc.energies_kcalmol, label='irc', color='black', lw=3)
plt.ylabel("Energies (kcal/mol)", fontsize=fs)
plt.xlabel("Reaction coordinate", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
# -

len(c_fsm_linear)

ch.visualize_chain(hcn.optimized)

hcn_qchem.coordinates[-1]

ch.visualize_chain(c_irc)

c_irc.energies

# +

hcn_qchem.energies
# -

ch.visualize_chain(hcn_qchem)

wittig = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/wittig/fsm_geodesic_neb.xyz")
wittig_qchem = load_qchem_result("/home/jdep/T3D_data/fneb_draft/wittig/fsm.files.2/")

# +

# ch.visualize_chain(wittig.optimized)
ch.visualize_chain(wittig_qchem)
# -

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb/Wittig/")

plt.plot(wittig.optimized.integrated_path_length, wittig.optimized.energies,'o-')
plt.plot(wittig_qchem.integrated_path_length, wittig_qchem.energies,'o-')

len(wittig.optimized), len(wittig_qchem)

# +
# wittig.optimized.plot_chain()

# +
# ch.visualize_chain(wittig.optimized)
# -

tsg1_jan = wittig.optimized[16]
tsg2_jan = wittig.optimized[26]

try:
    ts_res1_jan = eng._compute_ts_result(h.ordered_leaves[0].data.optimized.get_ts_node(), keywords={'maxiter':500})
except Exception as e:
    failed_out1 = e

try:
    ts_res1_jan = eng._compute_ts_result(tsg1_jan, keywords={'maxiter':500})
except Exception as e:
    failed_out1 = e

# +

try:
    ts_res2_jan = eng._compute_ts_result(tsg2_jan, keywords={'maxiter':500})
except Exception as e:
    failed_out2 = e
# -

irc1_res = build_full_irc_chain(StructureNode(structure=ts_res1_jan.return_result), engine=eng)

irc2_res = build_full_irc_chain(StructureNode(structure=ts_res2_jan.return_result), engine=eng)

coord1 = ch.get_rxn_coordinate(irc1_res[1])
coord2 = ch.get_rxn_coordinate(irc2_res[1])

irc2_res[0].nodes.reverse()
irc2_res[1].nodes.reverse()

# full_irc = Chain(irc1_res[0].nodes+irc2_res[0].nodes)
full_irc = Chain([irc1_res[0][0]]+irc1_res[1].nodes+[irc1_res[0][-1], irc2_res[0][0]]+irc2_res[1].nodes+[irc2_res[0][-1]])

# +

from neb_dynamics.geodesic_interpolation.coord_utils import align_geom
# -

align_geom(



# +
dists_fsm_1 = get_projections(wittig.optimized, coord1)
dists_fsm_2 = get_projections(wittig.optimized, coord2)


dists_qchem_1 = get_projections(wittig_qchem, coord1)
dists_qchem_2 = get_projections(wittig_qchem, coord2)

dists_irc_1 = get_projections(full_irc, coord1)
dists_irc_2 = get_projections(full_irc, coord2)

# +
s=6
fs=18
f, ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(wittig.optimized.integrated_path_length, wittig.optimized.energies_kcalmol, 'o-', label='fsm')
plt.plot(wittig_qchem.integrated_path_length, wittig_qchem.energies_kcalmol, 'o-', label='qchem')


plt.plot(full_irc.integrated_path_length, full_irc.energies_kcalmol, 'o--',label='irc', color='black', lw=3)
plt.ylabel("Rxn coordinate 2", fontsize=fs)
plt.xlabel("Rxn coordinate 1", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
# plt.savefig("attempt_at_wittig_irc.svg")

# +
s=6
fs=18
f, ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(dists_fsm_1, dists_fsm_2, 'o-', label='fsm')
plt.plot(dists_qchem_1, dists_qchem_2, 'o-', label='qchem')


plt.plot(np.array(dists_irc_1), np.array(dists_irc_2), 'o--',label='irc', color='black', lw=3)
plt.ylabel("Rxn coordinate 2", fontsize=fs)
plt.xlabel("Rxn coordinate 1", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
# plt.savefig("attempt_at_wittig_irc.svg")

# +

ch.visualize_chain(full_irc)

# +

len(wittig.optimized)

# +

full_irc.write_to_disk("/home/jdep/T3D_data/fneb_draft/wittig/irc_disjoined_intermediates.xyz")
# -

RMSD(full_irc[0].coords, wittig.optimized[0].coords)

full_irc.plot_chain()

# +

view.view(ts_res1_jan)
# -

hess_ress1 = eng._compute_hessian_result(StructureNode(structure=ts_res1_jan.return_result))

hess_ress2 = eng._compute_hessian_result(StructureNode(structure=ts_res2_jan.return_result))

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
    tcin_fp: str = None,
    es_ft: float = 0.03,
    met: str = "asneb"
    
    ):

    # if out_name is None:
    #     out_name = reference_start_name.stem
    template = [
    "#!/bin/bash",
    "",
    # "#SBATCH -t 24:00:00",
    "#SBATCH -t 12:00:00",
    "#SBATCH -J asneb",
    # "#SBATCH -p gpuq",
    "#SBATCH --qos=gpu_short",
    "#SBATCH --gres=gpu:1",
    "",
    "work_dir=$PWD",
    "",
    "cd $SCRATCH",
    "",
    "# Load modules",
    "ml TeraChem",
    "terachem -s 11111 || true &",
    "source /home/jdep/.bashrc",
    "source activate neb",
    "export OMP_NUM_THREADS=1",
    "# Run the job",
    "create_msmep_from_endpoints.py ",
    ]
    
    start_fp = reference_dir / reference_start_name
    end_fp = reference_dir / reference_end_name
    
    out_fp = out_dir / out_name
    print(out_fp)


    if tcin_fp:
        tcin = f"-tcin {tcin_fp}"
    else:
        tcin = ""
    
    cmd = f"/home/jdep/.cache/pypoetry/virtualenvs/neb-dynamics-G7__rX26-py3.9/bin/python /home/jdep/neb_dynamics/neb_dynamics/scripts/create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.002 \
        -sig {sig} -nimg 12 -min_ends 1 \
        -es_ft {es_ft} -name {out_fp} -prog {prog} -eng {eng} -node_rms_thre {nrt} -met {met} -node_ene_thre {net} {tcin} &> {out_fp}_out "
    
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

ch.visualize_chain(c_irc)

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
        eng='qcop',
        prog='terachem-pbs',
        tcin_fp='/home/jdep/T3D_data/msmep_draft/comparisons_dft/tc.in',
        es_ft=0.03,
        met='asneb',
        nrt=1, net=1)
    # create_submission_scripts(
    #     out_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb'),
    #     out_name=rn+"_asfneb",
    #     submission_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/submissions_dir'), 
    #     reference_dir=Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}'),
    #     reference_start_name='start_xtb.xyz', reference_end_name='end_xtb.xyz',
    #     eng='qcop',
    #     prog='terachem',
    #     tcin_fp='/home/jdep/T3D_data/msmep_draft/comparisons_dft/tc.in',
    #     es_ft=10,
    #     met='asfneb',
    #     nrt=1, net=5)
    
    # create_bs_submission(
    #     out_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb'),
    #     out_name=rn,
    #     submission_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/submissions_dir'), 
    #     reference_dir=Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}'),
    #     reference_start_name='start_xtb.xyz', reference_end_name='end_xtb.xyz',
    #     eng='chemcloud',
    #     prog='terachem',
    #     tcin_fp='/home/jdep/T3D_data/msmep_draft/comparisons_dft/tc.in',
    #     nrt=1, net=1)


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



