# +
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.Inputs import ChainInputs, GIInputs
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from neb_dynamics.nodes.Node3D import Node3D
from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB

from neb_dynamics.NEB_TCDLF import NEB_TCDLF
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import NEBInputs
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
# from retropaths.abinitio.trajectory import Trajectory
# from retropaths.abinitio.tdstructure import TDStructure
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.helper_functions import create_friction_optimal_gi
from neb_dynamics.optimizers.SD import SteepestDescent

from pathlib import Path
import numpy as np
# -

from neb_dynamics.NEB import NoneConvergedException
from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.Linesearch import Linesearch
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer as VPO

from neb_dynamics.NEB import NEB

start = TDStructure.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Azaindole-Synthesis/start_opt.xyz")

end = TDStructure.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Azaindole-Synthesis/end_opt.xyz")

tr = Trajectory([start, end]).run_geodesic(nimages=12)

# +
cni = ChainInputs(k=0.1, delta_k=0.09, do_parallel=True, node_freezing=True)

# optimizer1 = BFGS(step_size=3, min_step_size=0.1, bfgs_flush_steps=50, 
#                  bfgs_flush_thre=0.8, use_linesearch=False)
optimizer2 = VPO(timestep=0.1)

nbi = NEBInputs(v=1,tol=0.01*BOHR_TO_ANGSTROMS)
init_c = Chain.from_traj(tr, parameters=cni)
# n1 = NEB(initial_chain=init_c, parameters=nbi, optimizer=optimizer1)
# n1.optimize_chain()
# all_nebs_bfgs.append(n1)

n2 = NEB(initial_chain=init_c, parameters=nbi, optimizer=optimizer2)
n2.optimize_chain()
# all_nebs_vpo.append(n2)
# -

n2.plot_opt_history(1)

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Wittig/local_jan_msmep/")

init_guess = h.output_chain.to_trajectory().run_geodesic(nimages=12)

# +
# c = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen_b3lyp_631g_02012024.xyz", ChainInputs() )
# c = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/permutation_experiments/claisen_original.xyz", ChainInputs() )

# +
# start = c[0].tdstructure
# end = c[-1].tdstructure

# +
from neb_dynamics.optimizers.BFGS import BFGS

from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer as VPO
# -

# tr = Trajectory([start, end]).run_geodesic(nimages=len(c))
tr = init_guess

tols = [0.1, 0.01*BOHR_TO_ANGSTROMS, 0.001, 0.001*BOHR_TO_ANGSTROMS]
all_nebs_bfgs = []
all_nebs_vpo = []

# +
for tol in tols:

    cni = ChainInputs(k=0.1, delta_k=0.09, do_parallel=True, node_freezing=True)

    optimizer1 = BFGS(step_size=3, min_step_size=0.1, bfgs_flush_steps=50, 
                     bfgs_flush_thre=0.8, use_linesearch=False)
    optimizer2 = VPO(timestep=0.1)

    nbi = NEBInputs(v=1,tol=tol)
    init_c = Chain.from_traj(tr, parameters=cni)
    n1 = NEB(initial_chain=init_c, parameters=nbi, optimizer=optimizer1)
    n1.optimize_chain()
    all_nebs_bfgs.append(n1)
    
    n2 = NEB(initial_chain=init_c, parameters=nbi, optimizer=optimizer2)
    n2.optimize_chain()
    all_nebs_vpo.append(n2)
    
    
    
    
# -

from retropaths.reactions.pot import Pot


def get_minima_from_tree(tree_obj):
    minima = []
    for i, leaf in enumerate(tree_obj.ordered_leaves):
        if leaf.data:
            if i == 0:
                minima.append(leaf.data.chain_trajectory[-1][0].tdstructure)
            minima.append(leaf.data.chain_trajectory[-1][-1].tdstructure)
    return minima


tds = get_minima_from_tree(h)

molecules = [td.molecule_rp for td in tds]

import retropaths.helper_functions as hf

pot = Pot(root=molecules[0])
edges = list(hf.pairwise(list(range(len(tds)))))
edges_to_do = []
for i, mol_to_add in enumerate(molecules[1:], start=1):
    child_ind = i
    parent_ind = i-1

    pot.graph.add_node(child_ind, molecule=mol_to_add, converged=False)
    if (child_ind, parent_ind) in edges or (parent_ind, child_ind) in edges:
        pot.graph.add_edge(child_ind, parent_ind, reaction='neb0')
    else:
        edges_to_do.append((child_ind, parent_ind))

import matplotlib.pyplot as plt

# +
for tol, neb_obj in zip(tols, all_nebs_bfgs):
    plt.plot(neb_obj.chain_trajectory[-1].integrated_path_length, 
             neb_obj.chain_trajectory[-1].energies_kcalmol, 'o-',
             label=f'{round(tol,5)}_bfgs')
    
for tol, neb_obj in zip(tols, all_nebs_bfgs):
    plt.plot(neb_obj.optimized.integrated_path_length, 
             neb_obj.optimized.energies_kcalmol, 'o-',
             label=f'{round(tol,5)}_vpo')
# plt.plot(gi_chain.integrated_path_length, tr.energies_xtb(),'o-',label='gi')
plt.legend()
# -

chains_dft_bfgs = []
for neb_obj in all_nebs_bfgs:
    out_tr = neb_obj.optimized.to_trajectory()
    c = Chain.from_traj(out_tr, parameters=ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D_TC))
    c.gradients
    chains_dft_bfgs.append(c)


chains_dft_vpo = []
for neb_obj in all_nebs_vpo:
    out_tr = neb_obj.optimized.to_trajectory()
    c = Chain.from_traj(out_tr, parameters=ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D_TC))
    c.gradients
    chains_dft_vpo.append(c)

neb_obj = all_nebs_vpo[-1]
out_tr = neb_obj.chain_trajectory[-1].to_trajectory()
c = Chain.from_traj(out_tr, parameters=ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D_TC))
c.gradients
chains_dft_vpo.append(c)

gi_chain = Chain.from_traj(tr, 
                           parameters=ChainInputs(k=0.1, delta_k=0.09, 
                                                  node_class=Node3D_TC))

all_nebs_bfgs[-2].plot_opt_history(1)

# +
for tol, c in zip(tols, chains_dft_bfgs):
    if tol==0.1:
        continue
    plt.plot(c.integrated_path_length, 
             c.energies_kcalmol, 'o-',
             label=f'{round(tol,5)}_bfgs')
    
    
for tol, c in zip(tols, chains_dft_vpo):
    if tol==0.1:
        continue
    plt.plot(c.integrated_path_length, 
             c.energies_kcalmol, 'o-',
             label=f'{round(tol,5)}_vpo')

    
plt.plot(gi_chain.integrated_path_length, 
         gi_chain.energies_kcalmol, 'o-', label='gi')
plt.legend()

# +
# plt.plot(chains_dft[1].integrated_path_length, chains_dft[1].energies_kcalmol,'o-')
# plt.plot(chains_dft[2].integrated_path_length, chains_dft[2].energies_kcalmol,'o-')
plt.plot(chains_dft[3].integrated_path_length, chains_dft[2].energies_kcalmol,'o-')


plt.plot(c.integrated_path_length, c.energies_kcalmol,'o-')
# -

# chains_dft_bfgs[1].write_to_disk("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen_tol_0.01_b3lyp_631g_conv_using_xtb.xyz")
chains_dft_bfgs[2].write_to_disk("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen_tol_0.001_b3lyp_631g_conv_using_xtb.xyz")



# +

plt.plot(n_seeded[0].optimized.integrated_path_length, n_seeded[0].optimized.energies_kcalmol,'o-')
plt.plot(n_seeded[1].optimized.integrated_path_length, n_seeded[1].optimized.energies_kcalmol, 'o-')
# -

n_seeded[0].write_to_disk("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen_xtb_seed_tol0.01")

n_seeded = []
for init_seed in chains_dft_bfgs[1:3]:
    optimizer = BFGS(step_size=3, min_step_size=0.1, bfgs_flush_steps=50, 
                     bfgs_flush_thre=0.8, use_linesearch=False)

    nbi = NEBInputs(v=1,tol=tol)
    init_seed.parameters.node_freezing = True
    n = NEB(initial_chain=init_seed, parameters=nbi, optimizer=optimizer)
    n.optimize_chain()
    n_seeded.append(n)

n_seeded[1].plot_opt_history(1)

len(n_seeded[0].chain_trajectory)

len(n_seeded[1].chain_trajectory)

n_seeded[0].grad_calls_made

chains_dft[1].get_ts_guess()

c.get_ts_guess()

# # bias away from node 0-1

cb = ChainBiaser(reference_chains=[h.ordered_leaves[0].data.optimized])

tol = 0.01

nbi = NEBInputs(
        grad_thre=tol * BOHR_TO_ANGSTROMS,
        rms_grad_thre=(tol / 2) * BOHR_TO_ANGSTROMS,
        en_thre=(tol)* BOHR_TO_ANGSTROMS,  # loose energy threshold cause DLF doesnt use en thresh
        v=1,
        max_steps=500,
        early_stop_chain_rms_thre=1,  # not really caring about chain distances
        early_stop_force_thre=0.01,
        early_stop_still_steps_thre=100,
        vv_force_thre=0.0,                                                                                                                                                                                         
        _use_dlf_conv=False,
        preopt_with_xtb=False)

cni_bias = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D, 
                       node_freezing=True, do_parallel=True,
                       cb=cb,
                  )

cni_bias.do_chain_biasing = True

from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer

optimizer = VelocityProjectedOptimizer()

m = MSMEP(neb_inputs=nbi, chain_inputs=cni_bias, optimizer=optimizer, 
         gi_inputs=GIInputs(nimages=12))

chain_bias = Chain.from_traj(h.data.initial_chain.to_trajectory(),
                            cni_bias)

h_bias, out_bias = m.find_mep_multistep(chain_bias)

pot.draw_from_single_path(list(pot.paths_from(3))[0])


def sort_td(td):
    c = td.coords
    symbols = td.symbols
    sorted_inds = np.argsort(td.symbols)
    coords_sorted = np.array(c[sorted_inds])
    symbols_sorted = symbols[sorted_inds]
    
    sorted_td =  TDStructure.from_coords_symbols(coords=coords_sorted, symbols=symbols_sorted, 
                                           tot_charge=td.charge, tot_spinmult=td.spinmult)
    
    return sorted_td


def permute_indices_td(td, element):
    td = sort_td(td)
    coords = td.coords
    symbols = td.symbols
    
    inds_element = np.nonzero(td.symbols == element)[0]
    permutation_element = np.random.permutation(inds_element)
    
    inds_original = np.arange(len(td.symbols))
    inds_permuted = inds_original.copy()
    inds_permuted[inds_element] = permutation_element
    
    coords_permuted = coords[inds_permuted]
    permuted_td = TDStructure.from_coords_symbols(coords_permuted, symbols)
    permuted_td.update_tc_parameters(td)
    return permuted_td


start = c[0].tdstructure

end = c[-1].tdstructure

start_shuffled = permute_indices_td(start, 'C')

tr = Trajectory([start_shuffled, end]).run_geodesic(nimages=12)

# +
nbi = NEBInputs(tol=0.01,
                early_stop_force_thre=0.03, 
                early_stop_chain_rms_thre=1, v=True, preopt_with_xtb=False, max_steps=500)

cni = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D, node_freezing=True)
gii = GIInputs(nimages=12)
optimizer = BFGS(bfgs_flush_steps=200, bfgs_flush_thre=0.80, use_linesearch=False, 
                 step_size=3, 
                 min_step_size= 0.1,
                 activation_tol=0.1
            )
# -

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=optimizer)

chain = Chain.from_traj(tr, cni)

h, out = m.find_mep_multistep(chain)

out.plot_chain()

# # Foobar

from neb_dynamics.NEB import NEB

import retropaths.helper_functions as hf

reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")

rxn = reactions['1,3-Dipolar-Cycloaddition-Type-II']

nbi = NEBInputs(tol=0.001*BOHR_TO_ANGSTROMS,
                early_stop_force_thre=0.01*BOHR_TO_ANGSTROMS, 
                early_stop_chain_rms_thre=1, v=True, preopt_with_xtb=True, max_steps=500)
# cni = ChainInputs(k=0.1, delta_k=0.09,node_class=Node3D_TC,node_freezing=True)
cni = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D, node_freezing=True)
gii = GIInputs(nimages=12)
optimizer = BFGS(bfgs_flush_steps=200, bfgs_flush_thre=0.80, use_linesearch=False, 
                 step_size=3, 
                 min_step_size= 0.1,
                 activation_tol=0.1
            )

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=optimizer)

chain = Chain.from_traj(tr, cni)

out.plot_chain()

h, out = m.find_mep_multistep(chain)

from retropaths.abinitio.tdstructure import TDStructure

# +
rn = 'Huigsen-Cycloaddition'
rxn = reactions[rn]
# root = TDStructure.from_rxn_name('1,3-Dipolar-Cycloaddition-Type-II', reactions)
root = TDStructure.from_rxn_name(rn, reactions)

# root = TDStructure.from_smiles('C=C.N=[N+]=[N-].[Cu+]')
root = TDStructure.from_smiles('CC#CC.CN=[N+]=[N-].[Cu+].[Cu+]')
c3d_list = root.get_changes_in_3d(rxn)
root = root.pseudoalign(c3d_list)

# -

def move_atom_i_near_atom_j(td, atom_i, atom_j,disp=2):
    new_coords = td.coords
    new_coords[atom_i] = td.coords[atom_j]+disp
    td_out = td.update_coords(new_coords)
    return td_out


cu1_index = 8
cu2_index = 9
nitrogen_index = 6
carbon_index = 0

root = move_atom_i_near_atom_j(root, cu1_index, nitrogen_index, disp=1.8)
root = move_atom_i_near_atom_j(root, cu2_index, carbon_index, disp=-1.8)

target = root.copy()
target.apply_changed3d_list(c3d_list)

target = move_atom_i_near_atom_j(target, cu1_index, nitrogen_index, disp=1.8)
target = move_atom_i_near_atom_j(target, cu2_index, carbon_index, disp=-1.8)

# +
# foobar = TreeNode.read_from_disk("/home/jdep/T3D_data/asneb_catalysis/1,3-Dipolar-Cycloaddition-Type-II/cu_1_catalyzed")

# +
# foobar.data.initial_chain.to_trajectory()

# +
# foobar.output_chain.to_trajectory()

# +
# foobar.output_chain.plot_chain()
# -

root.tc_model_method = 'ub3lyp'
root.tc_model_basis = 'lanl2dzdp_ecp'

tr = Trajectory([root, target]).run_geodesic(nimages=12)

tr.update_tc_parameters(root)

initial_chain = Chain.from_traj(tr, parameters=cni)

# +
# get_xtb_seed(initial_chain, nbi, optimizer)
# -

leaves = [l.data for l in h.ordered_leaves if l.data]

m2 = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=25), optimizer=optimizer)
h_2, out_2 = m2.find_mep_multistep(leaves[1].chain_trajectory[-1])



leaves[1].plot_opt_history()
# leaves[1].optimized.to_trajectory()

h, out = m.find_mep_multistep(input_chain=initial_chain)

out.plot_chain()

opt = out[0].do_geometry_optimization()

out[0].tdstructure

opt.tdstructure



opt.energy

out[0].energy

rn = 'Benzimidazolone-Synthesis-1-X-Iodine'

h = TreeNode.read_from_disk(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/start_opt_2024_msmep')

tsg = h.ordered_leaves[0].data.optimized.get_ts_guess()

tsg.tc_model_method = 'wb97xd3'
tsg.tc_model_basis = 'def2-svp'

tsg.tc_freq_calculation()

h.output_chain.to_trajectory()

cni = ChainInputs(k=0.1, delta_k=0.09,node_class=Node3D_TC,node_freezing=True)

optimizer = BFGS(step_size=200, min_step_size=0.5, use_linesearch=False, bfgs_flush_thre=0.90,
                 activation_tol=20, bfgs_flush_steps=500)


optimizer_xtb = VelocityProjectedOptimizer(timestep=0.1)

# +
tol = 0.001
nbi_msmep = NEBInputs(
        grad_thre=tol * BOHR_TO_ANGSTROMS,
        rms_grad_thre=(tol / 2) * BOHR_TO_ANGSTROMS,
        v=1,
        max_steps=200,
        early_stop_chain_rms_thre=1,  # not really caring about chain distances,
        early_stop_force_thre=0.01,
        vv_force_thre=0.0,
        # _use_dlf_conv=True,
        _use_dlf_conv=False,
    )

nbi = NEBInputs(
        grad_thre=tol * BOHR_TO_ANGSTROMS,
        rms_grad_thre=(tol / 2) * BOHR_TO_ANGSTROMS,
        v=1,
        max_steps=500,
        vv_force_thre=0.0,
        # _use_dlf_conv=True,
        _use_dlf_conv=False,
    )

nbi_xtb = NEBInputs(
        grad_thre=tol * BOHR_TO_ANGSTROMS,
        rms_grad_thre=(tol / 2) * BOHR_TO_ANGSTROMS,
        v=1,
        max_steps=500,
        vv_force_thre=0.0,
        # _use_dlf_conv=True,
        _use_dlf_conv=False,
    )
# -

h = TreeNode.read_from_disk(folder_name=Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Robinson-Gabriel-Synthesis/wb97xd3_def2svp_2024/"),
                           neb_parameters=nbi, 
                           chain_parameters=cni)

step1 = h.ordered_leaves[0].data

neb_obj = step1

# initial_chain_tr = rgs.ordered_leaves[1].data.optimized.to_trajectory()
# initial_chain_tr = rgs.data.initial_chain.to_trajectory().run_geodesic(nimages=12)
# initial_chain_tr = neb_obj.initial_chain.to_trajectory().run_geodesic(nimages=12)
post_xtb_chain = get_xtb_seed(neb_obj.initial_chain, neb_params=nbi_xtb, optimizer=optimizer_xtb)

post_xtb_chain.plot_chain()

initial_chain_tr = post_xtb_chain.to_trajectory()

ref = initial_chain_tr[0]
ref.tc_model_method = 'wb97xd3'
ref.tc_model_basis = 'def2-svp'

initial_chain_tr.update_tc_parameters(ref)

initial_chain = Chain.from_traj(initial_chain_tr, parameters=cni)

# optimizer = SteepestDescent(step_size_per_atom=0.01)
# optimizer = VelocityProjectedOptimizer(timestep=1.0, activation_tol=0.1)
optimizer = BFGS(step_size=200, min_step_size=0.5, use_linesearch=False, bfgs_flush_thre=0.90,
                 activation_tol=20, bfgs_flush_steps=500)
# optimizer = BFGS(step_size=3, min_step_size=.1,use_linesearch=False, bfgs_flush_thre=0.80, 
#                  activation_tol=0.1, bfgs_flush_steps=20)


n = NEB(initial_chain=initial_chain,parameters=nbi, optimizer=optimizer)

# %%time
n.optimize_chain()

step1.optimized.plot_chain()

len(step1.chain_trajectory)

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=12),optimizer=optimizer)

h, out = m.find_mep_multistep(initial_chain)

h.write_to_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Robinson-Gabriel-Synthesis/wb97xd3_def2svp_2024"))

out.write_to_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Robinson-Gabriel-Synthesis/wb97xd3_def2svp_2024_out_chain.xyz"))

c = Chain.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Robinson-Gabriel-Synthesis/wb97xd3_def2svp_2024_out_chain.xyz"), cni)

c.to_trajectory()

np.linalg.norm(rgs.ordered_leaves[1].data.optimized.get_maximum_grad_magnitude())

clean1 = NEB.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Robinson-Gabriel-Synthesis/sd_msmep_cleanups/cleanup_neb_-1.xyz")

clean1.optimized.to_trajectory()

rgs.output_chain.plot_chain(0)

t = Trajectory.from_xyz('/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Robinson-Gabriel-Synthesis/initial_guess_msmep_clean.xyz')

# t = Trajectory.from_xyz('/home/jdep/T3D_data/geometry_spawning/claisen_results/claisen_ts_profile.xyz')
t = Trajectory.from_xyz('/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Robinson-Gabriel-Synthesis/initial_guess.xyz')

r,p = t[0], t[-1]

t = Trajectory.from_xyz('/home/jdep/T3D_data/geometry_spawning/claisen_results/claisen_ts_profile.xyz')
r,p = t[0], t[-1]

r.tc_model_method = 'wb97xd3'
r.tc_model_basis = 'def2-svp'

gi = Trajectory([r,p]).run_geodesic(nimages=15)

gi.update_tc_parameters(r)

# +
cni = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D_TC_TCPB, node_freezing=True, do_parallel=True)
# nbi = NEBInputs(v=True,tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, climb=False,_use_dlf_conv=True)
nbi = NEBInputs(v=True,tol=0.00045*BOHR_TO_ANGSTROMS, max_steps=500, climb=False, _use_dlf_conv=True)
nbi_dlf = NEBInputs(v=True,tol=0.00045, max_steps=500, climb=False, _use_dlf_conv=True)
gii = GIInputs(nimages=10)
optimizer_bfgs = BFGS(step_size=0.01, bfgs_flush_steps=200, bfgs_flush_thre=0.40, use_linesearch=False)
optimizer = SteepestDescent(step_size_per_atom=0.01)




chain = Chain.from_traj(gi, cni)
# -

seed = get_xtb_seed(chain, nbi, optimizer)

import matplotlib.pyplot as plt

info = np.loadtxt("/tmp/tmppk2mw_80/nebinfo")

n_seed.plot_opt_history(1)

plt.plot(n.chain_trajectory[-1].energies_kcalmol,'o-', label='jan')
plt.plot(info[:,1]*627.5,'o-', label='dlf')
plt.legend()

n_seed.optimized.plot_chain()

# %%time
# try:
n_seed = NEB(initial_chain=seed, parameters=nbi, optimizer=optimizer)
    # n = NEB_TCDLF(initial_chain=chain, parameters=nbi)
    # n.optimize_chain(remove_all=False)
n_seed.optimize_chain()
# except:
#     print("Done")

# %%time
# try:
# n = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer_bfgs)
n = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)
    # n = NEB_TCDLF(initial_chain=chain, parameters=nbi)
    # n.optimize_chain(remove_all=False)
n.optimize_chain()
# except:
#     print("Done")

n.plot_opt_history(1)

# %%time
n_dlf = NEB_TCDLF(initial_chain=chain, parameters=nbi_dlf)
n_dlf.optimize_chain(remove_all=False)
# except:
#     print("Done")

info = np.loadtxt('/home/jdep/T3D_data/dlfind_vs_jan/dlfind/tmplylcor_1/nebinfo')

import matplotlib.pyplot as plt

n_dlf.optimized.get_ts_guess()

n.optimized.get_ts_guess()

plt.plot(n.optimized.path_length, n.optimized.energies_kcalmol,'o-')
plt.plot(n_dlf.optimized.path_length, n_dlf.optimized.energies_kcalmol,'o-')
plt.plot(n_seed.optimized.path_length, n_seed.optimized.energies_kcalmol,'*-')

# +
# n.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_bfgs_xtbseed_k01"))

# +
# n.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_"))

# +
# n2_dft.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_bfgs_k01"))

# +
# huh = NEB.read_from_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_bfgs_xtbseed_k01"))

# +
# n2_dft.optimized.get_ts_guess()
# -

n2_dft.optimized.plot_chain()

n.optimized.plot_chain()

# +
# n.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_SD"))
# -

cni_xtb = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D, node_freezing=True, do_parallel=True)
chain_xtb = Chain.from_traj(gi, cni_xtb)

optimizer2 = BFGS(step_size=0.01, 
                 bfgs_flush_steps=20, bfgs_flush_thre=0.4, use_linesearch=False)
n2 = NEB(initial_chain=chain_xtb, parameters=nbi, optimizer=optimizer2)
n2.optimize_chain()

# +
# %%time

optimizer2_dft = BFGS(step_size=0.01, bfgs_flush_steps=20, bfgs_flush_thre=0.4, use_linesearch=False)
# n2_dft = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer2)

xtb_seed_tr = n2.chain_trajectory[-1].copy().to_trajectory()
xtb_seed = Chain.from_traj(xtb_seed_tr,cni)
# n2_dft = NEB(initial_chain=xtb_seed, parameters=nbi, optimizer=optimizer2_dft)
n2_dft = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer2_dft)
n2_dft.optimize_chain()
# -

n2_dft.plot_opt_history(1)

n2_dft.write_to_disk(Path("/home/jdep/T3D_data/dlfind_vs_jan/jan_bfgs_xtbseed_k01"))

n.optimized.plot_chain()

n2_dft.optimized.plot_chain()

# %%time
n_dlf = NEB_TCDLF(initial_chain=chain, parameters=nbi)
n_dlf.optimize_chain()

len(n_dlf.chain_trajectory)

n.optimized.plot_chain()

n_dlf.optimized.plot_chain()

n_dlf.optimized.to_trajectory()



# +
# m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, _use_dlf_as_backend=True, optimizer=None)

# +
# h, out = m.find_mep_multistep(chain)

# +
# cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D, node_freezing=True)
cni = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D_TC_TCPB, node_freezing=True, do_parallel=True)

optimizer = BFGS(step_size=0.33*gi[0].atomn, min_step_size=.001*gi[0].atomn, 
                 bfgs_flush_steps=10000, bfgs_flush_thre=0.4)
# optimizer2 = Linesearch(step_size=0.33*gi[0].atomn, min_step_size=.001*gi[0].atomn)

chain = Chain.from_traj(gi, cni)

nbi = NEBInputs(v=True,tol=0.001*BOHR_TO_ANGSTROMS, max_steps=10, climb=False, early_stop_force_thre=0.01*BOHR_TO_ANGSTROMS, 
                early_stop_chain_rms_thre=2, early_stop_still_steps_thre=100,
                _use_dlf_conv=True)
# -

n = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)

# %%time
try:
    n.optimize_chain()
except:
    print("DONE")

import time

# %%time
try :
    time.sleep(10)
    raise ValueError("loser")
except:
    print("hi")

# m = MSMEP(neb_inputs=nbi, chain_inputs=cni, optimizer=optimizer2, gi_inputs=GIInputs(nimages=15))
m2 = MSMEP(neb_inputs=nbi, chain_inputs=cni, optimizer=optimizer2, gi_inputs=GIInputs(nimages=15))

h2, out2 = m2.find_mep_multistep(chain)

h, out = m.find_mep_multistep(chain)

out.plot_chain()

out.to_trajectory()

results = []
thres = np.arange(start=0.1, stop=1.0, step=.1)
for fthre in thres:
    print(f"\ndoing {fthre}\n")
    cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D, node_freezing=True)
    # cni = ChainInputs(k=0.1, delta_k=0, node_class=Node3D, node_freezing=True)
    
    optimizer = BFGS(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn, 
                     bfgs_flush_steps=10000, bfgs_flush_thre=fthre)

    chain = Chain.from_traj(gi, cni)

    nbi = NEBInputs(v=True,tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, climb=False,
                   _use_dlf_conv=True)
    n = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)
    try:
        n.optimize_chain()
        results.append(n)
    except NoneConvergedException as e:
        results.append(e.obj)


# +

cni = ChainInputs(k=0.01, delta_k=0.0, node_class=Node3D, node_freezing=True)

optimizer = Linesearch(step_size=0.33*gi[0].atomn, min_step_size=.01*gi[0].atomn)

chain = Chain.from_traj(gi, cni)

nbi = NEBInputs(v=True,tol=0.001*BOHR_TO_ANGSTROMS, max_steps=500, climb=False,
               _use_dlf_conv=True)
n = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)

n.optimize_chain()
    # results.append(n)
# -

steps_taken = [len(neb.chain_trajectory) for neb in results]
steps_taken.append(len(n.chain_trajectory))

import matplotlib.pyplot as plt

fs = 18 
plt.plot([round(x, 1) for x in thres]+['SD-ls'], steps_taken, 'o-')
plt.ylabel("Opt steps", fontsize=fs)
plt.xticks(fontsize=fs-5)
plt.xlabel("Flush threshold", fontsize=fs)

data_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Robinson-Gabriel-Synthesis/")

# start = TDStructure.from_xyz(data_dir / 'react.xyz')
start = TDStructure.from_xyz(data_dir / 'start_opt.xyz')

# end = TDStructure.from_xyz(data_dir / 'prod.xyz')
end = TDStructure.from_xyz(data_dir / 'end_opt.xyz')

start.tc_model_method = 'wb97xd3'
start.tc_model_basis = 'def2-svp'
start.tc_kwds = {'restricted':False}

end.update_tc_parameters(start)

tr = Trajectory([start, end])

# +
# start_opt = start.()

# +
# end_opt = end.tc_geom_optimization()
# -

# gi = Trajectory([start_opt, end_opt]).run_geodesic(nimages=10)
gi = Trajectory([start, end]).run_geodesic(nimages=10)

all_mols = [td._as_tc_molecule() for td in gi]

all_inps = [td._prepare_input("gradient") for td in gi]

from chemcloud.client import CCClient

client = CCClient()

# %%time
result = client.compute('terachem',all_inps,collect_files=True)
output = result.get()

c0_guesses = [op.files['scr.geometry/c0'] for op in output]

all_inps_mod = []
# for mol, wfn_guess in zip(all_mols, c0_guesses):
for td, wfn_guess in zip(gi, c0_guesses):
    # prog_inp = ProgramInput(
    #     molecule=mol,
    #     model={"method": start.tc_model_method, "basis": start.tc_model_basis},
    #     calctype="gradient",
    #     keywords={'guess': "c0"},
    #     # files={'c0':wfn_guess}
    #     files = {'c0': b''}
    # )
    td2 = td.copy()
    td2.tc_c0 = wfn_guess
    prog_inp = td2._prepare_input("gradient")
    all_inps_mod.append(prog_inp)

# %%time
future_result = client.compute("terachem", all_inps_mod, collect_files=True, rm_scratch_dir=False)
output2 = future_result.get()

t = Trajectory.from_xyz("/home/jdep/T3D_data/dlfind_vs_jan/dlfind/scr.claisen_initial_guess/nebpath.xyz")
# t.update_tc_parameters(r)

r = t[0]
r.tc_model_method = 'wb97xd3'
r.tc_model_basis = 'def2-svp'

t.update_tc_parameters(r)

cni = ChainInputs(k=0.0, delta_k=0.0, node_class=Node3D_TC, node_freezing=True)
c = Chain.from_traj(t, parameters=cni)

import matplotlib.pyplot as plt

c.plot_chain()

np.linalg.norm(c.gradients)

np.linalg.norm(c.gradients) / len(c)

plt.plot(c.integrated_path_length, c.energies_kcalmol,'o-',label='me')
plt.plot(data_dlf[:, 0]/data_dlf[-1][0], data_dlf[:, 1]*627.5,'o-',label='dlf')
plt.legend()

ind = 7

output[ind].provenance

data_dlf = np.loadtxt('/home/jdep/T3D_data/dlfind_vs_jan/dlfind/scr.claisen_initial_guess/nebinfo')

ene, grads = gi.energies_and_gradients_tc()

# +
client = CCClient()
prog_input = [td._prepare_input(method='gradient') for td in gi]

future_result = client.compute(
    ES_PROGRAM, prog_input
)
output_list = future_result.get()

for res in output_list:
    if not res.success:
        res.ptraceback
        print(f"{ES_PROGRAM} failed.")


grads = np.array([output.return_result for output in output_list])
ens = np.array([output.results.energy for output in output_list])
# -

output_list[3].provenance

for ol in output_list:
    print(ol.provenance.hostname)

# def get_all_tsg_in_tree(h: TreeNode):
    all_tsg = []
    for neb_obj in h.get_optimization_history():
        start = neb_obj.initial_chain[0]
        end = neb_obj.initial_chain[-1]

        # gi = Node3D(neb_obj.initial_chain.get_ts_guess())
        tsg = Node3D(neb_obj.optimized.get_ts_guess())
        if any([ref_tsg.is_identical(tsg) for ref_tsg in all_tsg]):
            contin`a
            out.plotue
        else:
            all_tsg.append(tsg)

    return all_tsg

# # Wittig

# +
from retropaths.molecules.molecule import Molecule
from IPython.core.display import HTML
from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Chain import Chain
from neb_dynamics.nodes.Node3D import Node3D

from neb_dynamics.MSMEP import MSMEP
import matplotlib.pyplot as plt
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
import numpy as np
from scipy.signal import argrelextrema
from retropaths.helper_functions import pairwise
import retropaths.helper_functions as hf

HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

tds = TDStructure.from_smiles("C/C(C)=C/CCC(C)(C)C12CCC(=O)C1C2.[B]([F])([F])[F]")

from retropaths.molecules.molecule import Molecule
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.reactions.changes import Changes3D, Changes3DList, ChargeChanges
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
from retropaths.reactions.template import ReactionTemplate

# +
# mol1 =  Molecule.from_smiles('[P](=CC)(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3')
# mol1 =  Molecule.from_smiles('[P](=CC)(C)(C)C')
# mol2 =  Molecule.from_smiles('C(=O)C')

# mol = Molecule.from_smiles('[P](=CC)(C)(C)C.C(=O)C')
# mol = Molecule.from_smiles('[P](=C)(C)(C)C.CC(=O)C')
mol = Molecule.from_smiles("C/C(C)=C/CCC(C)(C)C12CCC(=O)C1C2.[B]([F])([F])[F]")
# -


td = TDStructure.from_RP(mol)

mol.draw(mode='d3', size=(500,500))

# +
delete_list = [
    (14, 9),
    
]

single_list = [
    (1, 11),
    (1, 3),
    
]

# +

ind=0
settings = [

    (
        td.molecule_rp,
        {'charges': [], 'delete':delete_list, 'single':single_list},
        [],
        [Changes3D(start=s, end=e, bond_order=1) for s, e in delete_list],
        [Changes3D(start=s, end=e, bond_order=1) for s, e in single_list]

    )]

mol, d, cg, deleting_list, forming_list = settings[ind]

# +
# """
# wittig
# """
# p_ind = 0
# cp_ind = 1

# me1 = 2
# me2 = 3
# me3 = 4

# o_ind = 7
# co_ind = 6

# ind=0
# settings = [

#     (
#         td.molecule_rp,
#         {'charges': [], 'delete':[(co_ind, o_ind), (cp_ind, p_ind)], 'double':[(o_ind, p_ind), (cp_ind, co_ind)]},
#         [
#             (me1, p_ind, 'Me'),
#             (me2, p_ind, 'Me'),
#             (me3, p_ind, 'Me'),
  
#         ],
#         [Changes3D(start=s, end=e, bond_order=1) for s, e in [(co_ind, o_ind), (cp_ind, p_ind)]],
#         [Changes3D(start=s, end=e, bond_order=2) for s, e in [(o_ind, p_ind), (cp_ind, co_ind)]]

#     )]

# mol, d, cg, deleting_list, forming_list = settings[ind]
# -

conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='colton', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

temp.products.draw(mode='d3')

c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])

root = TDStructure.from_RP(temp.reactants).xtb_geom_optimization()
root_ps = root.pseudoalign(c3d_list)

root_ps

root_ps_opt = root_ps.xtb_geom_optimization()

# target = root_ps_opt.copy()
target = root_ps.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.gum_mm_optimization()


target_opt = target.xtb_geom_optimization()

# tr = Trajectory([root_ps_opt, target_opt]).run_geodesic(nimages=15)
tr = Trajectory([root_ps, target_opt]).run_geodesic(nimages=15)

tr.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Wittig/initial_guess_fixed.xyz")

from neb_dynamics.optimizers.Linesearch import Linesearch
from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer

from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs

optimizer3 = VelocityProjectedOptimizer(step_size=0.10*root_ps_opt.atomn, min_step_size=0.001*root_ps_opt.atomn, als_max_steps=3)

# +
cni = ChainInputs(k=0.1, delta_k=0.09, node_freezing=True)
# cni = ChainInputs(k=0, node_freezing=True)
chain = Chain.from_traj(tr,parameters=cni)

nbi = NEBInputs(v=True, tol=0.001, early_stop_force_thre=0.03,early_stop_chain_rms_thre=0.002, early_stop_still_steps_thre=20, climb=False, _use_dlf_conv=True, max_steps=100, en_thre=0.01)

optimizer = BFGS(step_size=0.10*root_ps_opt.atomn, min_step_size=0.001*root_ps_opt.atomn, bfgs_flush_thre=.40, bfgs_flush_steps=20)
# optimizer = Linesearch(step_size=0.10*root_ps_opt.atomn, min_step_size=0.001*root_ps_opt.atomn, als_max_steps=5)
# optimizer = BFGS(step_size=1, min_step_size=0.001, bfgs_flush_thre=.99, bfgs_flush_steps=20)

# m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=15), optimizer=optimizer)
m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=15), optimizer=optimizer)
# -

h_xtb,out_xtb = m.find_mep_multistep(chain)

chain_to_upsample = h_xtb.ordered_leaves[1].data.optimized

m2 = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=40), optimizer=optimizer)
h_upsample, out_upsample = m2.find_mep_multistep(chain_to_upsample)


out_upsample.to_trajectory()

h_xtb.ordered_leaves[1].data.optimized.to_trajectory()

node_start = Node3D(root_opt)

node_end = Node3D(target_opt)

from neb_dynamics.helper_functions import RMSD


def get_vals(self, other):
    aligned_self = self.tdstructure.align_to_td(other.tdstructure)
    dist = RMSD(aligned_self.coords, other.tdstructure.coords)[0]
    en_delta = np.abs((self.energy - other.energy)*627.5)
    return dist, en_delta


tr = Trajectory([root_opt, target_opt]).run_geodesic(nimages=15, sweep=False)

tr.draw()

tr.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/initial_guess_tight_endpoints.xyz")

from neb_dynamics.constants import BOHR_TO_ANGSTROMS

# +
traj = Trajectory.from_xyz(Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/initial_guess.xyz"), tot_charge=0, tot_spinmult=1)
tol = 0.01
cni = ChainInputs(k=0.01, node_class=Node3D)

nbi = NEBInputs(tol=tol, # tol means nothing in this case
    grad_thre=0.001*BOHR_TO_ANGSTROMS,
    rms_grad_thre=0.0005*BOHR_TO_ANGSTROMS,
    en_thre=0.001*BOHR_TO_ANGSTROMS,
    v=True,
    max_steps=4000,
    early_stop_chain_rms_thre=0.002,
    early_stop_force_thre=0.02,vv_force_thre=0.00*BOHR_TO_ANGSTROMS,)
chain = Chain.from_traj(traj=traj, parameters=cni)
m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs())
# -

n = NEB(initial_chain=chain,parameters=nbi)

n.optimize_chain()

n.optimized.plot_chain()



h, out = m.find_mep_multistep(chain)

cni = ChainInputs()
start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/initial_guess.xyz", parameters=cni)
# tr.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_phe.xyz")

# +
# history.write_to_disk(Path("wittig_gone_horrible"))

# +
# tr.draw()
# -

cni = ChainInputs(k=0.10, delta_k=0.009)
nbi = NEBInputs(v=True, early_stop_chain_rms_thre=0.002, tol=0.01,max_steps=2000)
m  = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=15))
start_chain = Chain.from_traj(tr,parameters=cni)

history, out_chain = m.find_mep_multistep(start_chain)

history.data.plot_chain_distances()

out_chain.plot_chain()

out_chain.to_trajectory().draw()

out_chain.plot_chain()

history.write_to_disk(Path("./wittig_triphenyl_2"))

initial_chain = history.children[0].data.chain_trajectory[-1]


n_cont = NEB(initial_chain=initial_chain,parameters=nbi)
n_cont.optimize_chain()

n_cont.optimized.plot_chain()

out_chain.plot_chain()

# +
# cleanup_nebs[0].write_to_disk(Path("./cleanup_neb"),write_history=True)

# +
# clean_out_chain = Chain.from_list_of_chains([cleanup_nebs[0].optimized, history.output_chain], parameters=cni)

# +
# nbi = NEBInputs(v=True, stopping_threshold=0, tol=0.01)
# tr_long = Trajectory([clean_out_chain[0].tdstructure, clean_out_chain[-1].tdstructure]).run_geodesic(nimages=len(clean_out_chain), sweep=False)
# initial_chain_long = Chain.from_traj(tr_long,parameters=cni)

# +
# nbi = NEBInputs(v=True, stopping_threshold=0, tol=0.01, max_steps=4000)
# neb_long = NEB(initial_chain_long,parameters=nbi)

# +
# neb_long.optimize_chain()

# +
# neb_long.write_to_disk(Path("./neb_long_45nodes"), write_history=True)

# +
# write_to_disk(history,Path("./wittig_early_stop/"))

# +
# history = TreeNode.read_from_disk(Path("./wittig_early_stop/"))

# +
# out_chain = history.output_chain

# +
# neb_long = NEB.read_from_disk(Path("./neb_long_unconverged"))

# +
# neb_long_continued = NEB.read_from_disk(Path("./neb_long_continuation"))
# -

neb_short = NEB.read_from_disk(Path("./neb_short"))

neb_long = NEB.read_from_disk(Path("./neb_long_45nodes"))

neb_cleanup = NEB.read_from_disk(Path("./cleanup_neb"))

history = TreeNode.read_from_disk(Path("./wittig_early_stop/"))

cni = ChainInputs()
nbi = NEBInputs(v=True, stopping_threshold=3, tol=0.01)
m  = MSMEP(neb_inputs=nbi, root_early_stopping=True, chain_inputs=cni, gi_inputs=GIInputs(nimages=15))

start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess.xyz", parameters=cni)


insertion_points = m._get_insertion_points_leaves(history.ordered_leaves,original_start=start_chain[0])

# +
list_of_cleanup_nebs = [TreeNode(data=neb_cleanup, children=[])]
new_leaves = history.ordered_leaves
print('before:',len(new_leaves))
for insertion_ind, neb_obj in zip(insertion_points, list_of_cleanup_nebs):
    new_leaves.insert(insertion_ind, neb_obj)
print('after:',len(new_leaves))

new_chains = [leaf.data.optimized for leaf in new_leaves]
clean_out_chain = Chain.from_list_of_chains(new_chains,parameters=start_chain.parameters)

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(1.618*fig,fig))

# plt.plot(neb_long_continued.optimized.integrated_path_length, (neb_long_continued.optimized.energies-out_chain.energies[0])*627.5,'o-',label="NEB (30 nodes)")
# plt.plot(neb_long.optimized.integrated_path_length, (neb_long.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (45 nodes)")
# plt.plot(neb_short.optimized.integrated_path_length, (neb_short.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (15 nodes)")
# plt.plot(clean_out_chain.integrated_path_length, (clean_out_chain.energies-clean_out_chain.energies[0])*627.5,'o-',label="AS-NEB")
plt.plot(integrated_path_length(neb_long.optimized), (neb_long.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (45 nodes)")
plt.plot(integrated_path_length(neb_short.optimized), (neb_short.optimized.energies-clean_out_chain.energies[0])*627.5,'o-',label="NEB (15 nodes)")
plt.plot(integrated_path_length(clean_out_chain), (clean_out_chain.energies-clean_out_chain.energies[0])*627.5,'o-',label="AS-NEB")
plt.yticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.legend(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()


# +
nimages= 15
nimages_long = 45
n_steps_orig_neb = len(neb_short.chain_trajectory)
n_steps_msmep = sum([len(obj.chain_trajectory) for obj in history.get_optimization_history()])\
+ len(list_of_cleanup_nebs[0].data.chain_trajectory)
# n_steps_long_neb = len(neb_long.chain_trajectory+neb_long_continued.chain_trajectory)
n_steps_long_neb = len(neb_long.chain_trajectory)

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(1.16*fig,fig))
plt.bar(x=["AS-NEB",f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)'],
       height=[n_steps_msmep, n_steps_orig_neb, n_steps_long_neb])
plt.yticks(fontsize=fs)
plt.ylabel("Number of optimization steps",fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# +
nimages= 15
nimages_long = 45
n_steps_orig_neb = len(neb_short.chain_trajectory)*nimages
n_steps_msmep = sum([len(obj.chain_trajectory) for obj in history.get_optimization_history()])\
+ len(list_of_cleanup_nebs[0].data.chain_trajectory)
n_steps_msmep*=15
# n_steps_long_neb = len(neb_long.chain_trajectory+neb_long_continued.chain_trajectory)
n_steps_long_neb = len(neb_long.chain_trajectory)*nimages_long

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(1.16*fig,fig))
plt.bar(x=["AS-NEB",f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)'],
       height=[n_steps_msmep, n_steps_orig_neb, n_steps_long_neb],color='orange')
plt.yticks(fontsize=fs)
plt.ylabel("Number of gradient calls",fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()


# +
def get_mass_weighed_coords(chain):
    traj = chain.to_trajectory()
    coords = traj.coords
    weights = np.array([np.sqrt(get_mass(s)) for s in traj.symbols])
    mass_weighed_coords = coords  * weights.reshape(-1,1)
    return mass_weighed_coords

def integrated_path_length(chain):
    coords = get_mass_weighed_coords(chain)

    cum_sums = [0]

    int_path_len = [0]
    for i, frame_coords in enumerate(coords):
        if i == len(coords) - 1:
            continue
        next_frame = coords[i + 1]
        dist_vec = next_frame - frame_coords
        cum_sums.append(cum_sums[-1] + np.linalg.norm(dist_vec))

    cum_sums = np.array(cum_sums)
    int_path_len = cum_sums / cum_sums[-1]
    return np.array(int_path_len)


# -

plt.plot(integrated_path_length(neb_short.initial_chain), neb_short.initial_chain.energies,'o-')
plt.plot(integrated_path_length(neb_short.optimized), neb_short.optimized.energies,'o-')


neb_short.plot_opt_history(do_3d=True)



conc_checks = [m._chain_is_concave(c) for c in neb_short.chain_trajectory]

irc_checks = []
for c in neb_short.chain_trajectory:
    r,p = m._approx_irc(c)
    minimizing_gives_endpoints = r.is_identical(c[0]) and p.is_identical(c[-1])
    irc_checks.append(minimizing_gives_endpoints)

r,p = m._approx_irc(neb_short.chain_trajectory[0])

r.tdstructure

p.tdstructure

plt.plot(conc_checks,label='has no minima')
plt.plot(irc_checks, label='irc gives input structs')
plt.legend()
plt.show()

# ### Wittig Terachem

# + endofcell="--"
ind=0
settings = [

    (
        Molecule.from_smiles("CC(=O)C.CP(=C)(C)C"),
        {'charges': [], 'delete':[(5, 6), (1, 2)], 'double':[(1, 6), (2, 5)]},
        [
            (7, 5, 'Me'),
            (8, 5, 'Me'),
            (4, 5, 'Me'),
            (0, 1, 'Me'),
            (3, 1, 'Me'),
        ],
        [Changes3D(start=s, end=e, bond_order=1) for s, e in [(5, 6), (2, 1)]],
        [Changes3D(start=s, end=e, bond_order=2) for s, e in [(1, 6), (2, 5)]]

    )]

mol, d, cg, deleting_list, forming_list = settings[ind]

conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])
# -

root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root.gum_mm_optimization()
# --

root = root.xtb_geom_optimization()
root.tc_model_basis = 'gfn2xtb'
root.tc_model_method = 'gfn2xtb'

root = root.tc_geom_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.gum_mm_optimization()
target = target.xtb_geom_optimization()
target.update_tc_parameters(root)

target = target.tc_geom_optimization()

tr = Trajectory([root, target]).run_geodesic(nimages=15, sweep=False)

tr.write_trajectory("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_tc_xtb.xyz")

cni = ChainInputs(node_class=Node3D_TC)
# start_chain = Chain.from_traj(tr,parameters=cni)
# start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_b3lyp.xyz", parameters=cni)
start_chain = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/initial_guess_tc_xtb.xyz", parameters=cni)

# +
# root, target = start_chain[0], start_chain[-1]
# tr = Trajectory([root, target]).run_geodesic(nimages=45, sweep=False)
# -

nbi = NEBInputs(v=True, stopping_threshold=5, tol=0.01)
m  = MSMEP(neb_inputs=nbi, root_early_stopping=True, chain_inputs=cni, gi_inputs=GIInputs())

out_chain.plot_chain()

cleanup_neb = m.cleanup_nebs(start_chain,history)
