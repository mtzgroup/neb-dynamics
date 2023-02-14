# +
import matplotlib.pyplot as plt
import numpy as np

from retropaths.abinitio.trajectory import Trajectory
from retropaths.molecules.molecule import Molecule
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.TS_PRFO import TS_PRFO
from neb_dynamics.HistoryTree import HistoryTree
from neb_dynamics.helper_functions import _get_ind_minima

from pathlib import Path
# -


# # Claisen

# +
params = ChainInputs()
traj_tol01 = Chain.from_xyz("./claisen/cr_MSMEP_tol_01.xyz",params)
traj_tol0045 = Chain.from_xyz("./claisen/cr_MSMEP_tol_0045.xyz",params)
traj_tol0045_hf = Chain.from_xyz("./claisen/cr_MSMEP_tol_0045_hydrogen_fix.xyz",params)

# reference = Chain.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-NEB_v3/traj_0-0_0_neb.xyz")
# reference = Chain.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_0-0_0_cneb.xyz")
reference = Chain.from_xyz("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_0-0_0_cneb.xyz",params)

# +
s = 8
fs = 18

ens_tol01 = traj_tol01.energies
ens_tol0045  = traj_tol0045.energies
ens_tol0045_hf  = traj_tol0045_hf.energies
ens_reference = reference.energies

f,ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(reference.integrated_path_length, (ens_reference - ens_reference[0])*627.5,'o-',label="reference cneb")
# plt.plot(traj_tol01.integrated_path_length, (ens_tol01-ens_reference[0])*627.5,'o-',label="MSMEP tol=0.01 no-hydrogen-fix")
plt.plot(traj_tol0045.integrated_path_length, (ens_tol0045 - ens_reference[0])*627.5,'o-',label="MSMEP tol=0.0045 no-hydrogen-fix")
plt.plot(traj_tol0045_hf.integrated_path_length, (ens_tol0045_hf - ens_reference[0])*627.5,'o-',label="MSMEP tol=0.0045 hydrogen-fix")

plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (kcal/mol) ['reference' neb is 0]",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# -
ts = TS_PRFO(initial_node=traj_tol0045_hf.nodes[7], dr=1,max_step_size=.01)

ts.ts


# # Wittig

# ht = HistoryTree.read_from_disk(folder_name=Path("/home/jdep/T3D_data/msmep_draft/wittig_tree_pre_bugfix/"))
ht = HistoryTree.read_from_disk(folder_name=Path("/home/jdep/T3D_data/msmep_draft/"))

ht.root.data.plot_opt_history()


def plot_chains_at_depth(history_tree, depth,p_split, shift=.1, space=0.02):
    chains_at_depth = history_tree.get_nodes_at_depth(depth)

    for i, node in enumerate(chains_at_depth):
        chain = node.data.optimized
        if not np.mod(i,2):
            plt.plot(chain.integrated_path_length*(p_split - space), chain.energies-shift, 'o-', label=f"depth_{depth}_ind_{i}")
        else:
            plt.plot(chain.integrated_path_length*(1-p_split)+p_split, chain.energies-shift, 'o-', label=f"depth_{depth}_ind_{i}")


def get_cutoffs(parent_chain, prev_factor=1):
    root = parent_chain
    ind_min = _get_ind_minima(root)
    cutoffs = root.integrated_path_length[ind_min]
    return cutoffs*prev_factor


def get_p_splits(parent_chain, prev_factor=1, ind=0):
    root = parent_chain
    ind_min = _get_ind_minima(root)
    cutoffs = root.integrated_path_length[ind_min]
    if cutoffs.size > 0:
        return cutoffs[ind]*prev_factor
    else:
        p_split = prev_factor
    return p_split


def plot_from_parent(parent_node, name="root", depth=1):
    cuts = get_cutoffs(parent_node.data.optimized)
    root = parent_node.data.optimized
    plt.plot(root.integrated_path_length, root.energies, 'o-', label=name)
    for _, cut  in zip(ht.get_nodes_at_depth(depth), cuts):
        plot_chains_at_depth(history_tree=ht, depth=depth, p_split=cut, shift=depth*.1)


# +
s = 8
fs = 18

f,ax = plt.subplots(figsize=(1.8*s, s))



# p_split0 = get_p_splits(ht.root.data.optimized)


plot_from_parent(ht.root)
# plot_from_parent(ht.root.children[0], name="depth_2_parent", depth=2)
# for i, child in enumerate(ht.root.children):
#     plot_from_parent(child,name=f"depth_2_{i}", depth=2)
# cuts = get_cutoffs(ht.root.data.optimized)
# root = ht.root.data.optimized
# plt.plot(root.integrated_path_length, root.energies, 'o-', label=f"root")
# for cut in cuts:
#     plot_chains_at_depth(history_tree=ht, depth=1, p_split=cut)

# parent = ht.get_nodes_at_depth(1)[0]
# for i, node in enumerate(ht.get_nodes_at_depth(2)):
#     p_split = get_p_splits(parent.data.optimized, prev_factor=p_split0, ind=1)
#     plot_chains_at_depth(ht, depth=2, p_split=p_split, shift=.2)
        
plt.legend()

# +
plt.plot(ht.root.children[0].data.optimized.integrated_path_length*(p_split-space),ht.root.children[0].data.optimized.energies-shift, 'o-',label='depth1', color='orange')
# plt.plot((dep1_1.integrated_path_length*((1-p_split)+space))+p_split,dep1_1.energies-shift, 'o-', color='orange')


# plt.plot(dep2_0.integrated_path_length*.5,dep2_0.energies-2*shift, 'o-',label='depth2', color='red')

# plt.plot(dep3_0.integrated_path_length*.49,dep3_0.energies-3*shift, 'o-',label='depth2', color='black')



plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (Hartrees)",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()
# -

from dataclasses import dataclass

HistoryTree.from_history_list(history2)

dep2_0[-1].energy

# +
# t_witt = Chain.from_xyz("./wittig/auto_extracted_att2.xyz")
# t_witt = Chain.from_xyz("../example_mep.xyz")
# -

t_witt.to_trajectory().draw();

# +
s = 8
fs = 18

ens_witt = t_witt.energies

f,ax = plt.subplots(figsize=(1.8*s, s))

plt.plot(t_witt.integrated_path_length, (ens_witt - ens_witt[0])*627.5,'o-',label="extracted tol=0.01")

plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# -

traj_witt = t_witt.to_trajectory()

traj_witt.draw()

start, intermediate, end = traj_witt[0], traj_witt[14], traj_witt[-1]

start_opt = start.tc_geom_optimization()
int_opt = intermediate.tc_geom_optimization()
end_opt = end.tc_geom_optimization()

pair_start_to_int = Trajectory([start_opt, int_opt])
gi_start_to_int = pair_start_to_int.run_geodesic(nimages=15, sweep=False)

pair_int_to_end = Trajectory([int_opt, end_opt])
gi_int_to_end = pair_int_to_end.run_geodesic(nimages=15, sweep=False)

from neb_dynamics.Inputs import NEBInputs, ChainInputs

ChainInputs(

NEBInputs(v=True, 
neb_start_to_int = 

# # Ugi

# ht = HistoryTree.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/ugi_tree_canonical/"))
ht = HistoryTree.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/ugi_tree_2/"))


def get_aggregate_leaves(tree, depth):
    leaves = []
    for d in range(depth):
        nodes = tree.get_nodes_at_depth(d)
        for n in nodes:
            if len(n.children)==0:
                leaves.append(n)
    return leaves


ht.adj_matrix

ht.draw()

len(get_aggregate_leaves(ht,ht.max_depth+1))

leaves = get_aggregate_leaves(ht,ht.max_depth+1)

t = Trajectory.from_list_of_trajs([leaf.data.optimized.to_trajectory() for leaf in leaves])

c = Chain.from_traj(t, parameters=ChainInputs())

mol = Molecule.from_smiles('CNC(=O)C(c1ccccc1)NC=O.O')
mol.draw()

t[-1].molecule_rp.smiles

t_ref = Trajectory.from_xyz("/home/jdep/T3D_data/msmep_draft/ugi_tree_canonical/out_chain.xyz")

c_ref = Chain.from_traj(t_ref,parameters=ChainInputs())

# +
s = 8
fs = 18


shift = .1

f,ax = plt.subplots(figsize=(1.8*s, s))

plt.plot(c_ref.integrated_path_length, (c_ref.energies-c_ref.energies[0])*627.5, 'o-')
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.xlabel("Integrated Path Length", fontsize=fs)
plt.ylabel("Energy (kcal/mol)", fontsize=fs)
plt.show()
# -

t_ref.draw();

# +
s = 8
fs = 18


shift = .1

f,ax = plt.subplots(figsize=(1.8*s, s))


root = ht.root.data.optimized
plt.plot(root.integrated_path_length, root.energies, 'o-', label='root')

children = ht.root.children
child = children[0].data.optimized
plt.plot(child.integrated_path_length*(root.integrated_path_length[4]), child.energies-shift, 'o-', label='child')
# -

# t_ugi = Chain.from_xyz("./ugi/auto_extracted_0.xyz")
t_ugi = Chain.from_xyz("/home/jdep/neb_dynamics/example_cases/ugi/msmep_tol0.01_max_2000.xyz")

# +
s = 8
fs = 18

ens_ugi = t_ugi.energies

f,ax = plt.subplots(figsize=(2*s, s))

plt.plot(t_ugi.integrated_path_length, (ens_ugi - ens_ugi[0])*627.5,'o-',label="extracted tol=0.01")

plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# -


