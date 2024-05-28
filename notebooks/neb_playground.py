# -*- coding: utf-8 -*-
# +
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.Chain import Chain
from neb_dynamics.nodes.Node3D import Node3D
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

ts = TDStructure.from_xyz("/home/jdep/T3D_data/colton_debug/ts_sn2.xyz", tot_charge=-1)

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

h.output_chain.to_trajectory()

# from neb_dynamics.MSMEPAnalyzer import MSMEPAnalyzer
from neb_dynamics.NEB import NEB
from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path

p = Path('/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/')
msma = MSMEPAnalyzer(parent_dir=p, msmep_root_name='react')
msma_dft = MSMEPAnalyzer(parent_dir=p, msmep_root_name='dft_early_stop')

h = TreeNode.read_from_disk("/home/jdep/T3D_data/template_rxns/Ugi-Reaction/ugi_apr132024_msmep/")

h.output_chain[-1].tdstructure.molecule_rp.draw(mode='d3')

name = 'system23'
msma_obj = msma
# n = NEB.read_from_disk(p / name / 'dft_early_stop_msmep' / 'node_0.xyz')
n = NEB.read_from_disk(msma_obj.parent_dir / name /  (str(msma_obj.msmep_root_name)+"_msmep") / 'node_0.xyz')
h = msma_obj.get_relevant_tree(name)
out = msma_obj.get_relevant_chain(name)
sp = msma_obj.get_relevant_saddle_point(name)

n.optimized.get_ts_guess()

out.plot_chain()

distances = [msma_obj._distance_to_sp(c, sp) for c in n.chain_trajectory]

out.to_trajectory()

sp

out.to_trajectory()

plt.plot(distances,'o-')

np.argmin(distances)

sp

n.chain_trajectory[7].get_ts_guess()

sp

out.get_ts_guess()

out.plot_chain()

sp

out.get_ts_guess()



sp40 = TDStructure.from_xyz(p/'system40'/'sp.xyz')

tsg = out.to_trajectory()[26]

sp.tc_model_method = 'B3LYP-D3BJ'
sp.tc_model_basis = 'def2-svp'
sp.tc_kwds = {'reference':'uks'}

sp_opt = sp.tc_geom_optimization('ts')

sp_opt

tsg

tsg.update_tc_parameters(sp)

ts = tsg.tc_geom_optimization('ts')

ts.tc_freq_calculation()

sp

ts.to_xyz(p / name / 'sp_discovered.xyz')

# # TCPB Shit

from retropaths.abinitio.tdstructure import TDStructure

import tcpb 

import os
import sys
import atexit
import time
import signal
import socket
import logging
import subprocess
import tempfile
from contextlib import closing

# +
"""A parallel engine using a farm of TCPB servers as backend."""
import threading
import socket
import numpy as np
from collections import namedtuple, OrderedDict, defaultdict
from multiprocessing.dummy import Pool
import os


logger = logging.getLogger(__name__)


TCPBAddress = namedtuple("TCPBWorker", ('host', 'port'))
"""A TCPB server address, in the form of host/port tuple"""

# -

DEBYE_TO_AU = 0.393430307
class TCPBEngine:
    """TCPB engine using a pool of TeraChem TCPB workers as backend

    Example:  If you have 1 server on node fire-20-05 on ports 30001,
    and two on fire-13-01 on ports 40001, 40002, then
    >>> servers = [('fire-20-05', 30001),
                   ('fire-13-01', 40001), ('fire-13-01', 40002)]
    >>> engine = TCPBEngine(servers)
    >>> engine.status()
    TCPBEngine has 3 workers connected.
      [Ready] fire-20-06:30001
      [Ready] fire-13-01:40001
      [Ready] fire-13-01:40002
    Total number of available workers: 3

    Alternatively, one can let the engine to discover servers running on a node
    with the `scan_node` method.
    >>> engine = TCPBEngine()
    >>> engine.status()
    TCPBEngine has 0 workers connected.
    Total number of available workers: 0
    >>> engine.scan_node('fire-13-01')
    >>> engine.status()
    TCPBEngine has 2 workers connected.
      [Ready] fire-13-01:40001
      [Ready] fire-13-01:40002
    Total number of available workers: 2

    Then one can setup() and run calculations like on a normal engine.
    To add or remove servers from the worker pool, use `add_worker` and `remove_worker`.

    """


    @property
    def max_workers(self):
        """Redefine max_workers here to basically reflect the length of worker list,
        and at the same time make it read-only"""
        return len(self.workers)

    def __init__(self, workers=(), cwd="", max_cache_size=300, resubmit=True, propagate_orb=True,
                 *tcpb_args, **tcpb_kwargs):
        """Initialize TCPBWorkers and the engine

        Args:
            workers:    List/tuple of TCPBAddress [in forms of (host, port)]
                        which specifies the initial workers
            cwd:        Current working directory for terachem instance
            max_cache_size: Maximum number of cached results.
            resubmit:   Whether to resubmit a job if a worker becomes unavailable
            propagate_orb:  Whether to use the orbitals of the previous calculation as
                        starting guess for the next.  Currently, one orbital set is saved
                        for each host, and data transfer between host is not supported.
            """
        self.max_cache_size = max_cache_size
        self.workers = OrderedDict()
        self.tcpb_args = tcpb_args
        self.tcpb_kwargs = tcpb_kwargs
        self.error_count = defaultdict(int)
        self.options = dict()
        self.orbfiles = {}
        self.lock = threading.Lock()
        self.resubmit = resubmit
        self.propagate_orb = propagate_orb
        self.cwd = cwd
        for worker in workers:
            self.add_worker(worker)

    def _try_port(self, address):
        """Used by scan_node to scan a port on a host.
        Returns a connected TCPB client if the port hosts a TCPB server.
        """
        host, port = address
        try:
            # Check if the port is open.  Would get a RuntimeError if not
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)  # A short timeout of 1 sec
            sock.connect(address)
            client = tcpb.TCProtobufClient(host, port, *self.tcpb_args, **self.tcpb_kwargs)
            client.tcsock = sock
            client.tcsock.settimeout(30.0)
            if client.is_available():   # Check if accepting job
                return True, port, client
            else:
                return False, port, client
        except Exception as e:
            return None, port, e

    def scan_node(self, host, port_range=(1024, 65536), threads=None,
                  busy_servers=False, max_servers=None):
        """Scan node for all ports that have TCPB server running.
        A worker will be added to each available server.

        * Remember, with great power, comes great responsibility.

        Args:
            host:   Hostname on which to search for servers.
            port_range: Range to search.  Default is all valid ports.
            threads:    Number of threads.  Default is number of CPUs.
            busy_servers:   Whether or not to add servers that are
                    currently busy computing jobs.  Default is False.
            max_servers:  Maximum number of servers to connect. Default is no limit.
        """
        pool = Pool(threads)
        logger.info('Scanning host %s port %d to %d for TCPB servers.',
                    host, *port_range)
        n_added = 0
        addrs = ((host, port) for port in xrange(*port_range)
                 if (host, port) not in self.workers)
        for success, port, client in pool.imap_unordered(self._try_port, addrs, 100):
            if success is not None and success or busy_servers:
                address = TCPBAddress(host, port)
                logger.info('Found TCPB server at port %d', port)
                if n_added == max_servers:
                    logger.info("Number of servers added reached maximum %d.  Not adding.", max_servers)
                else:
                    self.workers[address] = client
        pool.close()
        pool.join()
        logger.info("Found %d new workers.  The engine now has %d workers.",
                    n_added, self.max_workers)

    @staticmethod
    def validate_address(address):

        if not isinstance(address, (tuple, list)) or len(address) != 2:
            raise TypeError("Worker ID must be a (host, port) tuple")
        if not isinstance(address[0], str):
            raise TypeError("Hostname must be a string")
        if not isinstance(address[1], int) or address[1] < 0:
            raise TypeError("port number must be a positive integer")
        return TCPBAddress(*address)

    def add_worker(self, address):
        """Add a tcpb worker to the worker pool"""
        logger.debug('Trying to add worker at address %s', str(address))
        address = self.validate_address(address)
        with self.lock:
            if address in self.workers:
                logger.warn('Address %s cannot be added: already in worker pool.',
                            str(address))
            client = tcpb.TCProtobufClient(address.host, address.port,
                                           *self.tcpb_args, **self.tcpb_kwargs)
            client.connect()
            self.workers[address] = client

    def remove_worker(self, address):
        """Remove a worker from worker pool"""
        address = self.validate_address(address)
        with self.lock:
            if address not in self.workers:
                raise KeyError("Cannot remove worker.  Not in worker pool.")
            try:
                self.workers[address].disconnect()
            except:
                pass
            del self.workers[address]

    def setup(self, mol, restricted=None, closed_shell=None, propagate_orb=None, **kwargs):
        """Setup the client based on information provided in a Molecule object
        Args:
            mol:    A Molecule object, the info in which will be used to
                    configure the client
            restricted: Whether to use restricted WF. Default is only to use
                    restricted when multiplicity is 1.
            closed_shell: Whether the system is closed shell.  Default is
                    to use closed shell wave function when multiplicity is 1
                    and restricted has not been set to false.
            propagate_orb:   Whether to use the previous orbital set as starting guess
                    for the next.  Current, cannot cross hosts.
            Other keyword arguments will be passed to job_spec() directly
        """
        with self.lock:
            if propagate_orb is not None:
                self.propagate_orb = propagate_orb
            self.orbfiles = {}
            if mol is None:
                atoms = charge = spinmult = None
            else:
                if not isinstance(mol, Molecule):
                    raise TypeError("mol must be a Molecule object.")
                atoms = list(mol.atoms)
                charge, spinmult = mol.charge, mol.multiplicity
                if restricted is not None:
                    self.options["restricted"] = restricted
                elif self.options.setdefault('restricted') is None:
                    self.options['restricted'] = mol.multiplicity == 1
                    logger.info('Infering `restricted` from multiplicity. (%s)', str(self.options['restricted']))
                if closed_shell is not None:
                    self.options['closed_shell'] = closed_shell
                elif self.options.setdefault('closed_shell') is None:
                    self.options['closed_shell'] = self.options['restricted'] and mol.multiplicity == 1
                    logger.info('Infering `closed_shell` from multiplicity. (%s)', str(self.options['closed_shell']))
            self.options.update(dict(atoms=atoms, charge=charge, spinmult=spinmult, **kwargs))

    def parse_options(self, args, separator='='):
        """Parse TeraChem arguments in a list of key/value strings"""
        if args is None:
            return
        tc_options = {'method': 'b3lyp', 'basis': '3-21g'}
        if isinstance(args, dict):  # No need to parse
            tc_options.update(args)
        else: 
            logger.info('Parsing TeraChem options')
            if isinstance(args, basestring):     # If args is just a string, assume it is the filename
                args = [args]
            for entry in args:
                raw = entry.split(separator, 1)
                if len(raw) == 1: 
                    filename = raw[0]
                    logging.info('Loading TeraChem input from file %s', filename)
                    with open(filename) as f:
                        for line in f:
                            splitted = line.strip().split(None, 1)
                            if len(splitted) != 2:
                                logger.info("Input file line ignored: %s", line.strip())
                            else:
                                key, value = splitted
                                key = key.lower()
                                value = value.strip()
                                if key in ['charge', 'spinmult']:
                                    value = int(value)
                                tc_options[key] = value
                else:
                    key, value = raw
                    key = key.lower()
                    value = value.strip()
                    if value[0] == '"' and value[-1] == '"' or value[0] == "'" and value[-1] == "'":
                        value = value[1:-1]
                    if key in ['charge', 'spinmult']:
                        value = int(value)
                    tc_options[key] = value
            for key, value in tc_options.iteritems():
                logger.info('%12s %12s', key, value)

        with self.lock:
            if tc_options['method'].lower().startswith('r'):
                tc_options['restricted'] = True
                tc_options['method'] = tc_options['method'][1:]
            elif tc_options['method'].lower().startswith('u'):
                tc_options['restricted'] = False
                tc_options['closed_shell'] = False
                tc_options['method'] = tc_options['method'][1:]
                self.closed_shell = False
            carry_over = ["atoms", "charge", "spinmult", "restricted", "closed_shell"]
            self.options = {key: self.options.get(key, None) for key in carry_over} 
            self.options.update(tc_options)

    def worker_by_index(self, index):
        """Get the client associated with a worker's index
        Returns a (address, client) tuple
        """
        with self.lock:
            return list(self.workers.items())[index]

    def compute__(self, geom, job_type, workerid, *args, **kwargs):
        address, worker = self.worker_by_index(workerid)
        with self.lock:
            options = self.options.copy()
        options.update(kwargs)
        if self.propagate_orb and 'guess' not in kwargs:
            if address.host in self.orbfiles:
                options['guess'] = self.orbfiles[address.host]
                logger.debug('Using previous orbitals as starting guess: %s', options['guess'])
            else:
                logger.debug('No previous orbital guess available')
        elif 'guess' in kwargs:
            logger.debug('Using %s as initial orbital guess', options['guess'])
        logger.debug("Perform calculation on worker %s:%d", *address)
        result = worker.compute_job_sync(jobType=job_type, geom=geom, unitType="bohr",
                                         bond_order=True, *args, **options)
        if self.propagate_orb:
            if 'orbfile' in result:
                self.orbfiles[address.host] = result['orbfile']
            elif 'orbfile_a' in result and 'orbfile_b' in result:
                self.orbfiles[address.host] = result['orbfile_a'] + " " + result['orbfile_b']
            else:
                logger.debug('Result does not contain orbital file info.')
        # Try to extract S^2
        if options.get('closed_shell', True):
            s2 = 0.0
        else:
            s2 = np.nan
            job_dir = result['job_dir']
            job_id = result['server_job_id'] 
            
            #If using pre-October 2020 TeraChem, use commented output_file line instead
            #output_file = result['output_file'] = "{}/{}.log".format(job_dir, job_id)
            output_file = result['output_file'] = "{}/{}/tc.out".format(self.cwd,job_dir)
            try:
                with open(output_file) as f:
                    for line in f:
                        if line.startswith('SPIN S-SQUARED:'):
                            s2 = float(line.replace('SPIN S-SQUARED:', '').split()[0])
                            break
            except IOError or OSError:
                logger.warn('Cannot access TeraChem output to extract <S^2>')
        result['S2'] = s2
        result['S2_exact'] = (options['spinmult'] ** 2 - 1) * 0.25
        # Convert dipole from Debye to a.u. (TCPB returns debye)
        result['dipole_vector'] = np.array(result['dipole_vector']) * DEBYE_TO_AU
        result['dipole_moment'] = np.array(result['dipole_moment']) * DEBYE_TO_AU
        return result

    def available__(self, workerid):
        """Check if a TCPB client is available for submission
        This subroutine itself should be exception proof

        workerid:   Which worker to check"""
        if workerid < 0 or workerid >= len(self.workers):
            logger.error("Worker index %d is out of range!", workerid)
            return False
        address, worker = self.worker_by_index(workerid)
        try:
            result = worker.is_available()
            if not result:
                logger.warn("Worker %d at address %s reported as unavailable.",
                            workerid, str(address))
            return result
        except (RuntimeError, tcpb.exceptions.ServerError, tcpb.exceptions.TCPBError):
            logger.debug("Worker %s:%d reported ERROR.", *address)
            count = self.error_count[address] + 1
            self.error_count[address] = count
            if count > 5:
                self.remove_worker(address)
                logger.info('Worker %s:%d removed', *address)
            return False
        except Exception:
            logger.error('Unknown exception thrown in TCPBEngine.available__', exc_info=True)
            return False

    def status(self):
        """Report on worker status"""
        n_ready = 0
        with self.lock:
            print ("TCPBEngine has %d workers connected." % self.max_workers)
            for key, worker in self.workers.iteritems():
                try:
                    if worker.is_available():
                        status = "Ready"
                        n_ready += 1
                    else:
                        status = " Busy"
                except RuntimeError:
                    status = "Error"
                print ("  [{}] {}:{}".format(status, *key))
            print ("Total number of available workers: %d" % n_ready)

    def disconnect(self):
        """Disconnect all servers"""
        with self.lock:
            logger.info('Trying to disconnect the clients and shutdown any housed servers.')
            try:
                for worker in self.workers.itervalues():
                    worker.disconnect()
            except:
                pass
            self.workers = {}
            for server in getattr(self, 'servers', ()):
                logger.debug('Trying to shut down hosted server %s', repr(server))
                server.shutdown()


class TCPBServer(object):
    """Maintains a TCPB server instance"""
    def __init__(self, terachem_exe='terachem', port=None, gpus='0',
                 output='server.out', wait=3, cwd=None):
        """Start a TeraChem protobuf server.

        Args:
            terachem_exc:   ['terachem'] TeraChem executable
            port:           [None] If specified, use this port between server and client
                            (instead of choosing a random unused port)
            gpus:           ['0'] List of GPUs to be used by the server
            output:         Filename (or file object) for the terachem output file
            cwd:            Working directory for the server.  The directory must exist!
            wait:           [3] Number of seconds to wait before the client is connected.

        All other arguments are passed to TCPB (methods, basis, etc ...)
        """
        self.gpus = gpus
        self.wait = wait
        self.terachem = terachem_exe
        if isinstance(output, str):
            self.output = open(output, 'w+')
        else:
            self.output = output
        if cwd is None:
            self.cwd = tempfile.mkdtemp()
            # raise AttributeError("No working directory for server specified in yaml file. Please use /u/username as directory (cwd).")
        else:
            self.cwd = cwd
        self.proc = None
        self.wait = wait
        self.start_server(port)
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def find_free_port(self):
        """Identify a free port number to be used.
        This sets the `port` field of the object."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            self.port = s.getsockname()[1]
            print('Selecting port %d', self.port)

    def start_server(self, port=None):
        """Start terachem server"""
        self.shutdown()
        self.port = port
        print('Starting server for TeraChem engine in %s.', self.cwd)
        if self.port is None:
            self.find_free_port()
        cmd = [self.terachem, '-g', self.gpus, '-s', str(self.port)]
        self.proc = subprocess.Popen(cmd, stdout=self.output, cwd=self.cwd)
        self.active_time = self.wait + time.time()

    def shutdown(self, signum=None, stack=None):
        """Stop the server instance. Not sure if we'll ever need this run manually, but
        it will be executed at least once when the program exits."""
        if signum is not None:
            print('TCPB Servers received shutdown signal : ' + str(signum))
        if self.proc is not None and self.proc.poll() is None:
                print("Shutting down TeraChem server at port %d", self.port)
                self.proc.kill()
        self.proc = None

    @property
    def address(self):
        """Returns the current server address (host, port).  If there is no active server,
        None is returned."""
        if time.time() < self.active_time:
            time.sleep(self.active_time - time.time())
        if self.proc is not None and self.proc.poll() is None:
            return socket.gethostname(), self.port
        else:
            return None


def prepare_engine(gpus, cwd, terachem_exe, wait):
    
    servers, server_addr = [], []
    for gpu_list in gpus.split(','):
        server = TCPBServer(gpus=gpu_list, cwd=cwd, terachem_exe=terachem_exe)
        servers.append(server)
        server_addr.append(server.address)
        
        
        
    pcwd = os.getcwd()
    if pcwd not in sys.path:
        sys.path.append(pcwd)
        
        
    time.sleep(wait)


    engine = TCPBEngine(server_addr, cwd = cwd)
    engine.servers = servers
    
    return engine


def prepare_tcpb_options(td):
    d = {'atoms':td.symbols, 'charge':td.charge, 'spinmult':td.spinmult,
          'closed_shell':False, 'restricted':False
          }
    return d


engine = prepare_engine(gpus, cwd, terachem_exe, wait)

engine.servers[0].active_time

td = TDStructure.from_smiles("O.O")

mol = None
gpus = '0'
wait = 3
cwd = None
terachem_exe = 'terachem'
basestring = 'terachem'

# +
options = prepare_tcpb_options(td)

# restricted = bool(mol.multiplicity == 1)
restricted = True
engine.setup(mol, restricted=restricted, closed_shell=restricted)
engine.parse_options(options)

# +
geom = td.coords

result = engine.compute__(geom, 'gradient', 0)
# -

result["energy"]

result['gradient']

engine.disconnect()

# +

td.gradient_tc_tcpb()
# -

from neb_dynamics.Node3D_TC_TCPB import Node3D_TC_TCPB
from neb_dynamics.Node3D_TC import Node3D_TC

from neb_dynamics.Chain import Chain

from neb_dynamics.Inputs import ChainInputs

c = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Claisen-Rearrangement/initial_guess_msmep.xyz", ChainInputs(node_class=Node3D_TC_TCPB, do_parallel=False))

tr = c.to_trajectory()

td_ref = tr[0]
td_ref.tc_model_method = 'wb97xd3'
td_ref.tc_model_basis = 'def2-svp'
td_ref.tc_kwds = {'restricted': False}

tr.update_tc_parameters(td_ref)

gi = tr.run_geodesic(nimages=8)

# %%time
gi.energies_tc()

# %%time
[td.energy_tc_tcpb() for td in gi]



# # BFGS



import numpy as np

# +
import numpy as np
from numpy.linalg import inv
from scipy import optimize as opt
import math

class BFGS:
    """
    Constructor BFGS (gracefully stolen from https://github.com/Paulnkk/Nonlinear-Optimization-Algorithms/blob/main/bfgs.py)
    """
    def __init__ (self, f, fd, H, xk, eps):
        self.fd = fd
        self.H = H
        self.xk = xk
        self.eps = eps
        self.f = f
        return
    """
    BFGS-Method 
    """

    def work (self):
        f = self.f
        fd = self.fd
        H = self.H
        xk = self.xk
        eps = self.eps
        """
        Initial Matrix for BFGS (Identitiy Matrix)
        """
        E =  np.array([   [1.,         0.],
                            [0.,         1.] ])
        xprev = xk
        it = 0
        maxit = 10000

        while (np.linalg.norm(fd(xk)) > eps) and (it < maxit):
            Hfd = inv(E)@fd(xprev)
            xk = xprev - Hfd
            sk = np.subtract(xk, xprev)
            yk = np.subtract(fd(xk), fd(xprev))

            b1 = (1 / np.dot(yk, sk))*(np.outer(yk, yk))
            sub1b2 = np.outer(sk, sk)
            Esk = E @ (sk)
            sub2b2 = (1 / np.dot(sk, Esk))
            sub3b2 = np.matmul(E, sub1b2)
            sub4b2 = np.matmul(sub3b2, E)
            b2 = sub2b2 * sub4b2
            E1 = np.add(E, b1)
            E = np.subtract(E1, b2)

            xprev = xk
            print(f'\t{xk}')
            print("Log-Values(BFGS): ", math.log10(f(xk)))
            it += 1

        return xk, it


# -

def func(val):
    x,y=val
    return x**2 + y**2


def grad(val):
    x,y=val
    return np.array([2*x, 2*y])


opt = BFGS(f=func,fd=grad,H=np.eye(2), xk=np.array([-2000,30.23]), eps=.01)

opt.work()

# # Foobar2

from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.Chain import Chain
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Inputs import ChainInputs

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Wittig/initial_guess_msmep/")

h.output_chain.plot_chain()

from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Inputs import NEBInputs, GIInputs

new_step2 = Trajectory([step1_neb.optimized[-1].tdstructure, step2_neb.optimized[-1]]).run_geodesic(nimages=15)

new_step2_chain = Chain.from_traj(new_step2, ChainInputs(k=0.1, delta_k=0.01, step_size=3))

neb = NEB(initial_chain=new_step2_chain, parameters=NEBInputs(v=1,tol=0.001))

neb.optimize_chain()

neb.initial_chain[0].tdstructure.to_xyz("/home/jdep/fuckwittig.xyz")

neb.chain_trajectory[-1].to_trajectory()

leaves = h.ordered_leaves

step1_neb = leaves[0].data

step2_neb = leaves[1].data

from neb_dynamics.constants import BOHR_TO_ANGSTROMS


def get_chain_at_cutoff(chain_traj, cut=0.01):
    beep = None
    for chain in chain_traj:
        if chain.get_maximum_gperp() <= cut:
            beep = chain
            break
    return beep


s1_gi = step1_neb.initial_chain.to_trajectory()

s1_opt = step1_neb.optimized.to_trajectory()

s1_003 = get_chain_at_cutoff(step1_neb.chain_trajectory, cut=0.003*BOHR_TO_ANGSTROMS).to_trajectory()

s1_01 = get_chain_at_cutoff(step1_neb.chain_trajectory, cut=0.01*BOHR_TO_ANGSTROMS).to_trajectory()

s1_0065 = get_chain_at_cutoff(step1_neb.chain_trajectory, cut=0.0065*BOHR_TO_ANGSTROMS).to_trajectory()

ref = step1_neb.initial_chain[0].tdstructure

ref.tc_model_method = 'b3lyp'
ref.tc_model_basis = 'def2-svp'
ref.tc_kwds = {'reference':'uks'}

s1_gi.update_tc_parameters(ref)

s1_gi_chain = Chain.from_traj(s1_gi, ChainInputs(node_class=Node3D_TC))

s1_gi_chain.get_maximum_gperp()

s1_01.update_tc_parameters(ref)

s1_003.update_tc_parameters(ref)

s1_0065.update_tc_parameters(ref)

s1_0065_chain = Chain.from_traj(s1_0065, ChainInputs(node_class=Node3D_TC))

s1_0065_chain.get_maximum_gperp()

s1_01_chain = Chain.from_traj(s1_01, ChainInputs(node_class=Node3D_TC))

s1_01_chain.get_maximum_gperp()

s1_003_chain = Chain.from_traj(s1_003, ChainInputs(node_class=Node3D_TC))

s1_003_chain.get_maximum_gperp()

s1_003_chain.get_maximum_gperp()

s1_opt_chain = Chain.from_traj(s1_opt, ChainInputs(node_class=Node3D_TC))

s1_opt_chain.get_maximum_gperp()

labels =['gi','0.01','0.0065', '0.003','0.001']

import matplotlib.pyplot as plt

plt.plot(labels, [s1_gi_chain.get_maximum_gperp(),s1_01_chain.get_maximum_gperp(),s1_0065_chain.get_maximum_gperp(), s1_003_chain.get_maximum_gperp(),s1_opt_chain.get_maximum_gperp()],'o-')

start, end = s1_opt[0], s1_opt[-1]
start.update_tc_parameters(ref)
end.update_tc_parameters(ref)

start_opt = start.tc_geom_optimization()

end_opt = end.tc_geom_optimization()


