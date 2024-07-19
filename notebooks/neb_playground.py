# -*- coding: utf-8 -*-
from neb_dynamics.engine import QCOPEngine

from neb_dynamics.nodes.node import Node
from qcio.models.inputs import Structure, ProgramInput

h2 = Structure(
    symbols=["H", "H"],
    geometry=[[0, 0.0, 0.0], [0, 0, 1.4]],  # type: ignore
)


ProgramInput(

eng = QCOPEngine(

# +
"""Example of using geometric subprogram to optimize H2 bond length.

Constraints docs: https://geometric.readthedocs.io/en/latest/constraints.html
"""

from qcio import DualProgramInput, Structure

from qcop import compute, exceptions

# Create Structure
h2 = Structure(
    symbols=["H", "H"],
    geometry=[[0, 0.0, 0.0], [0, 0, 1.4]],  # type: ignore
)

# Define the program input
prog_inp = DualProgramInput(
    calctype="optimization",  # type: ignore
    structure=h2,
    subprogram="xtb",
    subprogram_args={  # type: ignore
        "model": {"method": "GFN2xTB", "basis": "6-31g"}
    },
    keywords={
        "check": 3,
        # This is obviously a stupid constraint, but it's just an example to show how
        # to use them
        "constraints": {
            "freeze": [
                {"type": "distance", "indices": [0, 1], "value": 1.4},
            ],
        },
    },
)

# Run calculation
try:
    output = compute("geometric", prog_inp, propagate_wfn=False, rm_scratch_dir=False)
except exceptions.QCOPBaseError as e:
    # Calculation failed
    output = e.program_output
    print(output.stdout)  # or output.pstdout for short
    # Input data used to generate the calculation
    print(output.input_data)
    # Provenance of generated calculation
    print(output.provenance)
    print(output.traceback)
    raise

else:
    # Check results
    print("Energies:", output.results.energies)
    print("Structures:", output.results.structures)
    print("Trajectory:", output.results.trajectory)
    # Stdout from the program
    print(output.stdout)  # or output.pstdout for short
    # Input data used to generate the calculation
    print(output.input_data)
    # Provenance of generated calculation
    print(output.provenance)
# -

output.results.trajectory

output.results.trajectory[0].input_data

# +
from neb_dynamics.engine import QCOPEngine
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file
from neb_dynamics.nodes.node import Node
from qcio.models.inputs import ProgramInput

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs

import neb_dynamics.chainhelpers as ch

from neb_dynamics.engine import QCOPEngine
# -

c = Chain.from_xyz("/home/jdep/T3D_data/AutoMG_v0/msmep_results/results_pair149_msmep.xyz", parameters=ChainInputs())

from pathlib import Path

from neb_dynamics.inputs import NEBInputs
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.neb import NEB

# +
test_data_dir: Path = Path("/home/jdep/neb_dynamics/tests")
# tr = Trajectory.from_xyz(test_data_dir / "test_traj.xyz")
tol = 0.001
cni = ChainInputs(
    k=0.01,
    delta_k=0.009,
    do_parallel=True,
    node_freezing=True)

nbi = NEBInputs(
    tol=tol,  # * BOHR_TO_ANGSTROMS,
    barrier_thre=0.1,  # kcalmol,
    climb=False,

    rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
    max_rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS*2.5,
    ts_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
    ts_spring_thre=tol,  # * BOHR_TO_ANGSTROMS*3,

    v=1,
    max_steps=200,
    early_stop_force_thre=0.0)  # *BOHR_TO_ANGSTROMS)
initial_chain = c
prog_inp = ProgramInput(structure=initial_chain[0].structure, calctype='energy',
                        model={'method': 'GFN2xTB', 'basis': 'GFN2xTB'})
# symbols = tr.symbols

opt = VelocityProjectedOptimizer(timestep=1.0)
n = NEB(initial_chain=initial_chain, parameters=nbi, optimizer=opt,engine_inputs={'program_input': prog_inp, 'program': 'xtb'})


# +
import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """Suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# # Usage
# with suppress_output():
#     # Your code here
#     print("This will not be printed")
#     # Any other code that generates output


# -

with suppress_output():
    es_out = n.optimize_chain()

n.grad_calls_made

es_out.number_grad_calls

eng = QCOPEngine(program_input=ProgramInput(structure=c[0].structure,calctype='energy',model={'method':"GFN2xTB"}), program='xtb')

c[0].structure.geometry

c[1].structure.geometry

out1 = eng.compute_gradients(c)

c[0].structure.geometry

c[1].structure.geometry

out2 = eng.compute_gradients(c.nodes)

import neb_dynamics.chainhelpers as ch

import numpy as np

np.amax(abs(ch.get_g_perps(c)))

c.energies

out1

out2



from neb_dynamics.trajectory import Trajectory

tr = Trajectory.from_xyz("/home/jdep/T3D_data/AutoMG_v0/msmep_results/results_pair149_msmep.xyz")

grads = [td.gradient_xtb() for td in tr]

c._zero_velocity()

c.velocity

import numpy as np

from neb_dynamics.convergence_helpers import _check_rms_grad_converged

?_check_rms_grad_converged

_check_rms_grad_converged(c, threshold=0.001)

np.zeros_like(a=c.coordinates)

c.gradients[0]

c.gradients[0] / grads[0]

import numpy as np

np.amax(ch.get_g_perps(c))





huh = c[0].update_coords(c[-1].coords)

from neb_dynamics.qcio_structure_helpers import structure_to_molecule

structure_to_molecule(c[0].structure).draw()

structure_to_molecule(huh.structure).draw()

pi = ProgramInput(
    structure=c[0].structure,
    calctype='energy',
    model={'method':'GFN2xTB', 'basis':'GFN2xTB'}
)

c[0].structure.__dict__

foo = {'hey':1}

foo['lol']

type(c[0])

n = Node(c[0].structure)

type(n)

eng = QCOPEngine(prog_input=pi, program='xtb')

from qcop import compute

compute(

eng.compute_energies(c)

from qcio.models.inputs import ProgramInput

compute(

structs[0] is structs[0]

from qcio.models.outputs import ProgramOutput

from qcop import compute

pi = ProgramInput(
    structure=structs[0],
    calctype='energy',
    model={"method": "GFN2xTB"},  # type: ignore
    keywords={},
)

res = compute("xtb", pi)

res.results.gradient

ProgramInput

pi.__dict__.copy()

len(structs)

mol = structure_to_molecule(structs[-1])

mol.draw()

len(structs)

from nodes.node import Node
from neb_dynamics.qcio_structure_helpers import   split_structure_into_frags, structure_to_molecule

# +
from qcio.models.structure import Structure

structures = Structure.from_smiles("COCO.CC", force_field="MMFF94")

with open("/home/jdep/debug.xyz",'w+') as f:
    f.write(structures.to_xyz())
f.close()
# -

mol = structure_to_molecule(structures)

from neb_dynamics.tdstructure import TDStructure

td = TDStructure.from_xyz('/tmp/tmp1l4yxrqg.xyz')

td2 = td.copy()

td2.gum_mm_optimization()

td2

mol.draw()

split_structure_into_frags(structures)

node = Node(structure=Structure.from_smiles("COCO"))
node2 = Node(structure=Structure.from_smiles("COCO"))



node.is_identical(node2)

from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory

tr = Trajectory([TDStructure.from_smiles("COCO") for i in range(5)])

start = TDStructure.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/1-2-Amide-Phthalamide-Synthesis/start_opt.xyz")

end = TDStructure.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/1-2-Amide-Phthalamide-Synthesis/end_opt.xyz")

tr = Trajectory([start, end]).run_geodesic(nimages=12)

tr.energies_and_gradients_tc()

# +
from neb_dynamics.TreeNode import TreeNode
from chain import Chain
from nodes.node3d import Node3D
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.SD import SteepestDescent
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.optimizers.Linesearch import Linesearch
from neb_dynamics.NEB_TCDLF import NEB_TCDLF
from pathlib import Path

from neb_dynamics.Refiner import Refiner
from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -


h = TreeNode.read_from_disk("/home/jdep/T3D_data/Ugi_NOSIG/msmep_results/results_pair0_msmep/")

h2 = TreeNode.read_from_disk("/home/jdep/debugging_throwaway_results/ugi_shit/debug/")

h2.output_chain.plot_chain()

output = h2.ordered_leaves[1].data.optimized._approx_irc()

output[0].is_identical(h2.ordered_leaves[1].data.optimized[0])

output[1].tdstructure

h.output_chain.plot_chain()

h2.output_chain[-1].is_identical(h.output_chain[-1])

len(h.get_optimization_history())

len(h2.get_optimization_history())

h.output_chain.plot_chain()

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/1-2-Amide-Phthalamide-Synthesis/ASNEB_03_NOSIG_NOMR/")

h.ordered_leaves[1].data.plot_opt_history(1)

h.output_chain.plot_chain()

c = Chain.from_xyz("/home/jdep/T3D_data/template_rxns/Ugi-Reaction/conformer_sampling/best_ugi_chain.xyz", ChainInputs())

c2 = Chain.from_xyz("/home/jdep/T3D_data/template_rxns/Ugi-Reaction/ugi_apr132024_msmep.xyz", ChainInputs())

import matplotlib.pyplot as plt

# +
fs=18
s=5
f,ax = plt.subplots(figsize=(1.61*s, s))
plt.plot(c.integrated_path_length, c.energies,'o-', label='conformer sampled msmep')
plt.plot(c2.integrated_path_length, c2.energies,'-', alpha=.7, label='pseudoaligned msmep')
plt.xlabel("Normalized path length",
           fontsize=fs)
plt.ylabel("Energies (kcal/mol)",
           fontsize=fs)
plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)


plt.legend(fontsize=fs)#, bbox_to_anchor=(1.01, 1.05))
# -

from retropaths.abinitio.tdstructure import TDStructure as TD2
from retropaths.abinitio.trajectory import Trajectory as Tr2

Tr2([TD2(td.molecule_obmol) for td in c.to_trajectory().traj])

TD2(c[4].tdstructure.molecule_obmol).xtb_geom_optimization()

c[0].tdstructure.molecule_rp.draw(mode='d3')

from neb_dynamics.trajectory import Trajectory

tr = Trajectory.from_xyz("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement/start_conformers/crest_conformers.xyz")

tr2 = Trajectory.from_xyz("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement/end_conformers/crest_conformers.xyz")

# +
start_confs = tr.traj
end_confs = tr2.traj

from itertools import product

pairs_to_do = list(product(start_confs, end_confs))

template = open("/home/jdep/T3D_data/template_rxns/Ugi-Reaction/new_template.sh").read().splitlines()

data_dir = Path("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement/pairs_to_do/")
submissions_dir = Path("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement/submissions/")
for i, (start, end) in enumerate(pairs_to_do):

    print(f'***Doing pair {i}')
    start_fp = data_dir / f'start_pair_{i}.xyz'
    end_fp = data_dir / f'end_pair_{i}.xyz'
    start.to_xyz(start_fp)
    end.to_xyz(end_fp)
    out_fp = Path(f'/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement/conformer_sampling/results_pair{i}_msmep')


    cmd = f"create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.001 -sig 0 -mr 0 -nc node3d -preopt 0 -climb 0 -nimg 12 -name {out_fp} -min_ends 1"

    new_template = template.copy()
    new_template[-1] = cmd

    with open(submissions_dir / f'submission_{i}.sh', 'w+') as f:
        f.write("\n".join(new_template))
# -

from neb_dynamics.tdstructure import TDStructure

# all_rns = open("/home/jdep/T3D_data/msmep_draft/comparisons_dft/reactions_todo.txt").read().splitlines()
all_rns = open("/home/jdep/T3D_data/msmep_draft/comparisons/reactions_todo_xtb.txt").read().splitlines()

import shutil

# +
# rn = 'Lobry-de-Bruyn-Van-Ekenstein-Transformation'
# p = f"/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/production_msmep/"
do_refine = True
multi = []
elem = []
failed = []
skipped = []
n_steps = []

n_opt_steps = []
n_opt_splits = []

success_names = []


for i, rn in enumerate(all_rns):
    # p = Path(rn) / 'production_vpo_tjm_xtb_preopt_msmep'
    # p = Path(rn) / 'production_maxima_recycling_msmep'
    p = Path(rn) / 'ASNEB_5_noSIG'
    print(p.parent)

    try:
        with open("/home/jdep/T3D_data/msmep_draft/comparisons/status_refinement.txt","w+") as f:
            f.write(f"doing {p}")
        h = TreeNode.read_from_disk(p)

        # refine reactions
        refined_fp = Path(rn) / 'refined_ASNEB_5_noSIG'
        print('Refinement done: ', refined_fp.exists())
        if do_refine and not refined_fp.exists():
            # if refined_fp.exists():
            #     shutil.rmtree(refined_results)
            print("Refining...")

            refiner = Refiner(cni=ChainInputs(k=0.1, delta_k=0.09,
                                              node_class=Node3D_TC,
                                              use_maxima_recyling=False,
                                              do_parallel=True,
                                              node_freezing=True))
            refined_leaves = refiner.create_refined_leaves(h.ordered_leaves)
            refiner.write_leaves_to_disk(refined_fp, refined_leaves)


            tot_grad_calls = sum([leaf.get_num_grad_calls() for leaf in refined_leaves if leaf])
            print(f"Refinement took: {tot_grad_calls} calls")
            with open(refined_fp.parent/'refined_ASNEB_5_noSIG_grad_calls.txt','w+') as f:
                f.write(f"Refinement took: {tot_grad_calls} gradient calls")


        es = len(h.output_chain)==12
        print('elem_step: ', es)
        print([len(obj.chain_trajectory) for obj in h.get_optimization_history()])
        n_splits = len(h.get_optimization_history())
        print(sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()]))
        tot_steps = sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()])

        n_opt_steps.append(tot_steps)
        n_opt_splits.append(n_splits)




        if es:
            elem.append(p)
        else:
            multi.append(p)


        n = len(h.output_chain) / 12
        n_steps.append((i,n))
        success_names.append(rn)

    except Exception as e:
        print(e)
        print(f"{rn} had an error")
        continue




        es = len(neb_obj.optimized)==12
        print('elem_step: ', es)
        print(len(neb_obj.chain_trajectory))
        # sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()])
        if es:
            elem.append(p)
        else:
            multi.append(p)
        n = len(neb_obj.optimized) / 12
        n_steps.append((i,n))
        tot_steps = len(neb_obj.chain_trajectory)
        n_opt_steps.append(tot_steps)
        n_opt_splits.append(0)

        success_names.append(rn)


    except FileNotFoundError:
        failed.append(p)

    except TypeError:
        failed.append(p)

    except KeyboardInterrupt:
        skipped.append(p)


    print("")

# -



from neb_dynamics.trajectory import Trajectory

from neb_dynamics.tdstructure import TDStructure

tsg = TDStructure.from_xyz("/home/jdep/T3D_data/tsg.xyz")

tsg.tc_model_method = 'wb97xd3'
tsg.tc_model_basis = 'def2-svp'

ts = tsg.tc_geom_optimization('ts')

ts

ts_plus = ts.displace_by_dr(0.5*ts.tc_nma_calculation()[0])

ts_minus = ts.displace_by_dr(-0.5*ts.tc_nma_calculation()[0])

ts_plus.update_tc_parameters(ts)
ts_minus.update_tc_parameters(ts)

ts_plus_opt = ts_plus.tc_local_geom_optimization()

profile = Trajectory([ts_minus_opt, ts, ts_plus_opt ])

profile.update_tc_parameters(ts)

c = Chain.from_traj(profile, ChainInputs(node_class=Node3D_TC))

c.gradients

c.write_to_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Lossen-Rearrangement/exact_ts_profile.xyz")

ens = profile.energies_tc()

(ens[1]-ens[0])*627.5

ts_plus_opt

ts_minus_opt = ts_minus.tc_local_geom_optimization()

ts_minus_opt

tr = Trajectory.from_xyz('/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Claisen-Rearrangement/start_opt_msmep/node_0.xyz').run_geodesic(nimages=100)

init_c = Chain.from_traj(tr, parameters=ChainInputs(k=0.0, node_class=Node3D_TC, node_freezing=True))

opt = VelocityProjectedOptimizer(timestep=0.5, activation_tol=0.1)

n = NEB(initial_chain=init_c, parameters=NEBInputs(v=1), optimizer=opt)

output = n.optimize_chain()

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Robinson-Gabriel-Synthesis/explicit_solvent_xtb/")

from neb_dynamics.tdstructure import TDStructure

coco = TDStructure.from_smiles("COCOCOCO")

# %%time
coco_opt_2 = coco.run_tc_local(method=coco.tc_model_method,
                               basis=coco.tc_model_basis,
                               calculation='minimize')

# %%time
coco_opt_3 = coco.run_tc_local(method=coco.tc_model_method,
                               basis=coco.tc_model_basis,
                               calculation='minimize')

coco.tc_kwds = {'new_minimizer': True}

# %%time
coco_opt = coco.tc_local_geom_optimization()

coco.tc_kwds = {'new_minimizer':False}

# %%time
coco_opt_4 = coco.tc_local_geom_optimization()

coco_opt_2

coco_opt_3

coco_opt

coco_opt_4

coco_opt_2

h.output_chain.plot_chain()

h.data.plot_opt_history(1)

from retropaths.abinitio.trajectory import Trajectory

t = Trajectory.from_xyz('/home/jdep/T3D_data/colton_debug/sn2_endpoints.xyz', tot_charge=-1)

start = t[0]
end = t[-1]

from retropaths.abinitio.solvator import Solvator

solv = Solvator(n_solvent=5)

start_solv = solv.solvate_td(start)

start_solv

end_solv = solv.solvate_td(end)

t_solv = Trajectory([start_solv, end_solv]).run_geodesic(nimages=12)

# +
cni = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D, do_parallel=True, node_freezing=True)
# cni = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D, do_parallel=True, node_freezing=True)
nbi = NEBInputs(tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, v=1, _use_dlf_conv=False, early_stop_force_thre=0.01, early_stop_chain_rms_thre=1)
gii = GIInputs(nimages=12)
# chain = Chain.from_traj(tr, parameters=cni)
chain = Chain.from_traj(t_solv, parameters=cni)
# neb = NEB_TCDLF(initial_chain=chain, parameters=nbi)

# optimizer = Linesearch(step_size=0.33*len(chain), min_step_size=.01*len(chain), activation_tol=0.1)
# optimizer = SteepestDescent(step_size_per_atom=0.01)
# optimizer = VelocityProjectedOptimizer(timestep=.1, activation_tol=0.5)
# optimizer = Linesearch(step_size=1.0, min_step_size=0.33)
optimizer = BFGS(step_size=3, min_step_size=0.1, use_linesearch=False, bfgs_flush_thre=0.80,
                 activation_tol=0.1, bfgs_flush_steps=500)

neb = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)
# neb2 = NEB(initial_chain=chain2, parameters=nbi, optimizer=optimizer)

# -

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=optimizer)

h, out = m.find_mep_multistep(chain)

out.plot_chain()

# %%time
try:
    neb.optimize_chain()
except Exception as e:
    print(e)




neb.plot_opt_history(0)

r, p = neb.optimized._approx_irc()

from retropaths.abinitio.tdstructure import TDStructure

td = TDStructure.from_smiles("O.O.O.O.[O-][H]")

foobar = td.xtb_geom_optimization()

foobar

tsg = neb.optimized.get_ts_guess()

tsg.tc_model_method

tsg.tc_model_basis

ts = tsg.tc_geom_optimization('ts')

ts.to_xyz("/home/jdep/T3D_data/colton_debug/ts_sn2.xyz")

from neb_dynamics.tdstructure import TDStructure

ts = TDStructure.from_xyz("/home/jdep/T3D_data/colton_debug/ts_sn2.xyz", charge=-1)

ts.tc_model_method = 'wb97xd3'
ts.tc_model_basis = 'def2-svp'
ts.tc_kwds = {"pcm":"cosmo","epsilon":80}
# ts.tc_kwds = {}

ts.tc_freq_calculation()

nma = ts.tc_nma_calculation()

ind = 1
Trajectory([ts.displace_by_dr(nma[ind]), ts, ts.displace_by_dr(-1*nma[ind])] )

m = MSMEP(neb_inputs=nbi,
          chain_inputs=cni, gi_inputs=GIInputs(nimages=15),
          optimizer=optimizer)

neb2 = NEB_TCDLF(initial_chain=chain, parameters=nbi)

# %%time
try:
    neb2.optimize_chain()
except Exception as e:
    print(e)


neb2.plot_opt_history()

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=16),optimizer=optimizer)

h, out = m.find_mep_multistep(chain)

# ## h.ordered_leaves[0].data.plot_opt_history(0)

out.plot_chain()

# +
# neb.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_SD_deltak_notConv"))
# -

import time

# %%time
try:
    neb.optimize_chain()
except Exception as e:
    print(e)


neb.plot_opt_history(1)

# +
# neb.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_VPO_deltak_nimg16_rms_en_conv"))
# -



neb.optimized.gradients[14]

neb.optimized.plot_chain()

598/(14*3)

np.argmax(np.abs(neb.optimized.gradients))

neb.plot_opt_history(1)

tr_dlf = Trajectory.from_xyz('/home/jdep/T3D_data/dlfind_vs_jan/dlfind/scr.claisen_initial_guess/nebpath.xyz')
tr_dlf.update_tc_parameters(tr[0])


from retropaths.abinitio.tdstructure import TDStructure

ci_tr = Trajectory.from_xyz('/home/jdep/T3D_data/dlfind_vs_jan/dlfind/scr.claisen_initial_guess/optim.xyz')
ci_tr.update_tc_parameters(tr[0])

ci = chain.parameters.node_class(ci_tr[-1])

dlf_neb = Chain.from_traj(tr_dlf, parameters=cni)

dlf_neb.nodes.insert(0, neb.chain_trajectory[-1][0])

dlf_neb.nodes.insert(10, ci)

dlf_neb.nodes.append(neb.chain_trajectory[-1][-1])

import numpy as np

np.loadtxt('/home/jdep/T3D_data/dlfind_vs_jan/dlfind/scr.claisen_initial_guess/nebinfo')[:,0]

plt.plot(dlf_neb.path_length, dlf_neb.energies,'o-', label='dlf')
# plt.plot(np.loadtxt('/home/jdep/T3D_data/dlfind_vs_jan/dlfind/scr.claisen_initial_guess/nebinfo')[:,0], np.loadtxt('/home/jdep/T3D_data/dlfind_vs_jan/dlfind/scr.claisen_initial_guess/nebinfo')[:,1]*627.5,'o-', label='dlf')
plt.plot(neb.optimized.path_length, neb.optimized.energies,'o-', label='jan')
plt.legend()


Trajectory.from_xyz("/home/jdep/T3D_data/dlfind_vs_jan/dlfind/scr.claisen_initial_guess/optim.xyz")

dlf_neb.get_ts_guess()

h, out = m.find_mep_multistep(chain)

out.plot_chain()

# %%time
try:
    neb2.optimize_chain()
except:
    print("done")

# %%time
neb.optimize_chain(remove_all=False)

neb2.plot_opt_history()

# +
# chain_traj = neb.get_
# -

neb.plot_opt_history(1, 1)

# +
# chain_traj = neb.get_chain_trajectory(Path('/tmp/tmpn_nqmg77'))

# +
# neb.optimize_chain(remove_all=False)

# +
# chain_traj = neb.chain_trajectory
# -

h_loser = NEB.read_from_disk("./hey_loser")

# +

[node._cached_energy for node in chain_traj[0]]
# -

neb.chain_trajectory = chain_traj

neb.write_to_disk(Path('hey_loser'))

# !pwd

# +
# amadori = NEB.read_from_disk('/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Amadori-Rearrangement/debug_msmep/')
# -



# +

neb.plot_opt_history(0,0)

# +
# # %%time
# neb2.optimize_chain()

# +
# # %%time
# neb.optimize_chain(remove_all=True)
# -

h, out = m.find_mep_multistep(chain)

leaves = [leaf.data for leaf in h.ordered_leaves if leaf.data]

leaves_jan = [leaf.data for leaf in h_jan.ordered_leaves if leaf.data]

tsg1_jan = leaves_jan[0].optimized.get_ts_guess()

tsg1_dlf = leaves[0].optimized.get_ts_guess()

# +
#### JAN: TODO: write a function that scrapes the DLF energies and adds them to the _cached_energies for creating chain trajectories
# -





import numpy as np



len(h.data.chain_trajectory)

import matplotlib.pyplot as plt

tsg1_jan.tc_model_method = 'uwb97xd3'
tsg1_jan.tc_model_basis = 'def2-svp'

tsg1_jan.tc_freq_calculation()

tsg1_dlf.tc_freq_calculation()

out.plot_chain()

h_jan = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Wittig/local_jan_msmep")

len(h_jan.get_optimization_history())

h.

len(h.get_optimization_history())

h.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/dlfind/wittig_msmep"))

out.to_trajectory()

from neb_dynamics.constants import BOHR_TO_ANGSTROMS


# +
print(f"doing reference")
# cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D_TC, node_freezing=True)
cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D, node_freezing=True)
optimizer2 = Linesearch(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn)
chain = Chain.from_traj(gi, cni)

nbi = NEBInputs(v=True,tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, climb=True,
               _use_dlf_conv=True)

# +
# n_ref = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer2)
# n_ref.optimize_chain()

# +
print(f"doing reference")
cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D_TC, node_freezing=True)
# cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D, node_freezing=True)
optimizer = BFGS(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn,
                 bfgs_flush_steps=10000, bfgs_flush_thre=0.10)
chain = Chain.from_traj(gi, cni)

# nbi = NEBInputs(v=True,tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, climb=True)
nbi = NEBInputs(v=True,tol=0.001*BOHR_TO_ANGSTROMS, max_steps=70, climb=True,
                en_thre=0.001*BOHR_TO_ANGSTROMS,
               _use_dlf_conv=True)
# -

n_bfgs = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)
n_bfgs.optimize_chain()

traj = new_chain.to_trajectory()

foobar = traj.energies_and_gradients_tc()

new_chain.plot_chain()



new_chain.plot_chain()

n_bfgs_cont = NEB(initial_chain=n_bfgs.chain_trajectory[-1], parameters=nbi, optimizer=optimizer)
n_bfgs_cont.optimize_chain()

import matplotlib.pyplot as plt

len(n_bfgs.chain_trajectory)

n_bfgs.chain_trajectory[-1].plot_chain()

plt.plot(n_ref.optimized.integrated_path_length, n_ref.optimized.energies_kcalmol, 'o-',label='ref')
plt.plot(n_bfgs.optimized.integrated_path_length, n_bfgs.optimized.energies_kcalmol, 'o-',label='bfgs')

cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D_TC, node_freezing=True)
n_ref = NEB.read_from_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_linesearch"), chain_parameters=cni)

# n_bfgs.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_bfgs"))
cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D_TC, node_freezing=True)
n_bfgs = NEB.read_from_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_bfgs"), chain_parameters=cni)

n_bfgs.plot_opt_history(do_3d=True)

n_bfgs.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_bfgs_dlf_conv"))

len(n_bfgs.chain_trajectory)

# +
# n_bfgs = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)
# n_bfgs.optimize_chain()
# -

t = Trajectory.from_xyz("/home/jdep/T3D_data/dlfind_vs_jan/dlfind/scr.claisen_initial_guess/nebpath.xyz")
t.traj.insert(0, r)
t.traj.append(p)
t.update_tc_parameters(r)

c = Chain.from_traj(t, parameters=cni)

import matplotlib.pyplot as plt



import numpy as np


def hard_reset_chain(chain):
    for node in chain:
        node._cached_gradient = None
        node._cached_energy = None
        node.converged = False



hard_reset_chain(jan_chain)

len(jan_chain)

gperps_jan, gparr_jan = jan_chain.pe_grads_spring_forces_nudged()

gperps_dlf, gparr_dlf = c.pe_grads_spring_forces_nudged()

len(gperps_dlf)

np.linalg.norm(gperps_dlf[4]) / np.sqrt(11*3 + 3)

a, b = conv_chain.pe_grads_spring_forces_nudged()

for node in conv_chain.nodes:
    node.tdstructure.update_tc_parameters(r)

hard_reset_chain(conv_chain)

c4_grad = c[4].tdstructure.gradient_tc()

tan = c._create_tangent_path(c[3],c[4],c[5])

unit_tan = tan / np.linalg.norm(tan)

hard_reset_chain(c)

# for grad in c:
for grad in jan_chain:
    print(grad._cached_gradient, grad.converged)

from neb_dynamics.constants import BOHR_TO_ANGSTROMS
import numpy as np

# +
conv_chain = None
conv_step = None

for i, chain in enumerate(n_bfgs.chain_trajectory):
# for i, chain in enumerate(n_ref.chain_trajectory):
    gperp, _ = chain.pe_grads_spring_forces_nudged()
    tol = 0.001*BOHR_TO_ANGSTROMS



    grad_conv = np.linalg.norm(gperp) / np.sqrt(11*3 + 3) <= tol
    rms_grad_conv = np.sqrt(sum(np.square(gperp.flatten())) / len(gperp.flatten())) <= (tol / (1.5))

    if grad_conv and rms_grad_conv:
        conv_chain = chain
        conv_step = i
        break
# -

conv_step

n_bfgs._chain_converged(n_bfgs.chain_trajectory[conv_step-1], conv_chain)

n_bfgs.parameters._use_dlf_conv = True
n_bfgs.parameters.v = 3
n_bfgs.parameters.en_thre = n_bfgs.parameters.tol

n_bfgs._chain_converged(n_bfgs.chain_trajectory[conv_step-1], conv_chain)

# +
# jan_chain = n_bfgs.chain_trajectory[223]
# jan_chain = n_bfgs.chain_trajectory[-1]
jan_chain = conv_chain

plt.plot(jan_chain.integrated_path_length, jan_chain.energies,'o-', label='jan')

plt.plot(c.integrated_path_length, c.energies,'o-', label='dlfind')
plt.legend()
# -

tsg_jan = jan_chain.get_ts_guess()

tsg_jan.update_tc_parameters(r)

tsg_jan.

tsg_jan.tc_freq_calculation()

c.get_maximum_gperp()

n_bfgs.plot_projector_history()

n_bfgs.chain_trajectory[-1].to_trajectory()

import numpy as np

flush_steps = [5, 10, 20, 50, 100]
flush_thres = [.50, .80, .90, .99]

from itertools import product

results_nebs = []
for fs, ft in list(product(flush_steps, flush_thres)):
    print(f"doing: force steps:{fs} force thre: {ft}")
    cni = ChainInputs(k=0.1, delta_k=0.09)
    chain = Chain.from_traj(gi, cni)
    opt = BFGS(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn, bfgs_flush_steps=fs, bfgs_flush_thre=ft)
    nbi = NEBInputs(v=True,tol=0.001, max_steps=500)

    n = NEB(initial_chain=chain, parameters=nbi, optimizer=opt)
    try:
        n.optimize_chain()
        results_nebs.append(n)
    except NoneConvergedException as e:
        results_nebs.append(e.obj)
    except:
        results_nebs.append(None)

from neb_dynamics.optimizers.Linesearch import Linesearch

# +
print(f"doing reference")
cni = ChainInputs(k=0.1, delta_k=0.09)
optimizer2 = Linesearch(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn)
chain = Chain.from_traj(gi, cni)

nbi = NEBInputs(v=True,tol=0.001, max_steps=500, do_bfgs=False)

n_ref = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer2)
n_ref.optimize_chain()
# -

n = results_nebs[7]
n.optimized.plot_chain()

n_ref.optimized.get_eA_chain()

n.optimized.get_eA_chain()

n_ref.optimized.plot_chain()

for neb, params in zip(results_nebs, list(product(flush_steps, flush_thres))):
    fs, ft = params
    neb.write_to_disk(Path(f"/home/jdep/T3D_data/bfgs_results/claisen_fs_{fs}_ft_{ft}.xyz"))

import pandas as pd

conditions = list(product(flush_steps, flush_thres))
results = [] # f_steps, f_thre, n_steps
for n_result, (f_steps, f_thre)  in zip(results_nebs, conditions):
    results.append([f_steps, f_thre, len(n_result.chain_trajectory)])

df = pd.DataFrame(results, columns=['f_steps','f_thre','n_steps'])

import matplotlib.pyplot as plt

df

plt.plot(n.optimized.integrated_path_length, n.optimized.energies, label='ref')
plt.plot(results_nebs[6].optimized.integrated_path_length, results_nebs[6].optimized.energies, label='new opt')
plt.legend()

results_nebs[6].optimized.plot_chain()

len(n.chain_trajectory)

df.sort_values(by='n_steps')

# # Foobar

from neb_dynamics.TreeNode import TreeNode

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig2/react_msmep/")

from neb_dynamics.MSMEPAnalyzer import MSMEPAnalyzer
from neb_dynamics.NEB import NEB
from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path

p = Path('/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/')
msma = MSMEPAnalyzer(parent_dir=p, msmep_root_name='ASNEB_03_NOSIGNOMR')
msma_dft = MSMEPAnalyzer(parent_dir=p, msmep_root_name='dft_early_stop')

# +
# list of problems: 17,

# sys14 is interesting. SP is a second order at b3lyp/6-31g but a 1st order with wb97xd3/def2svp
# -

name = 'system95'
msma_obj = msma
# n = NEB.read_from_disk(p / name / 'dft_early_stop_msmep' / 'node_0.xyz')
n = NEB.read_from_disk(msma_obj.parent_dir / name /  (str(msma_obj.msmep_root_name)+"_msmep") / 'node_0.xyz')
h = msma_obj.get_relevant_tree(name)
out = msma_obj.get_relevant_chain(name)
sp = msma_obj.get_relevant_saddle_point(name)

h.output_chain.plot_chain()
h.output_chain.to_trajectory().draw();

sp.tc_model_method = 'wb97xd3'

sp.tc_model_basis = 'def2-svp'

sp.tc_freq_calculation()


