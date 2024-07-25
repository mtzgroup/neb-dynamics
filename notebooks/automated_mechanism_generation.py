# +
from qcio import ProgramInput, ProgramOutput, Structure

from chemcloud import CCClient

water = Structure(
    symbols=["O", "H", "H"],
    geometry=[
        [0.0000, 0.00000, 0.0000],
        [0.2774, 0.89290, 0.2544],
        [0.6067, -0.23830, -0.7169],
    ],
)

client = CCClient()

prog_inp = ProgramInput(
    structure=water,
    model={"method": "b3lyp", "basis": "6-31g"},
    calctype="gradient",  # Or "gradient" or "hessian"
    keywords={},
)
future_result = client.compute("terachem", prog_inp, collect_files=True)
prog_output: ProgramOutput = future_result.get()
# ProgramOutput object containing all returned data
print(prog_output.stdout)
print(prog_output)
# The energy value requested

if prog_output.success:
    print(prog_output.results.energy)
    print(prog_output.files.keys())
else:
    print(prog_output.traceback)

# -

prog_inp.calctype = 'energy'

prog_output.results.gradient

# +
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.helper_functions import RMSD
from chain import Chain
from neb_dynamics.Inputs import ChainInputs
from neb_dynamics.molecule import Molecule
from neb_dynamics.treenode import TreeNode
from neb_dynamics.pot import Pot


import numpy as np
import matplotlib.pyplot as plt
import contextlib, os

from pathlib import Path
import subprocess
from itertools import product
import itertools
from dataclasses import dataclass
from IPython.core.display import HTML
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel

import numpy as np
import matplotlib.pyplot as plt
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

# # Network stuff

# +
from neb_dynamics.NetworkBuilder import NetworkBuilder, ReactionData
from neb_dynamics.Inputs import ChainInputs
from pathlib import Path
import os

from neb_dynamics.molecule import Molecule
from neb_dynamics.tdstructure import TDStructure
# -

start = TDStructure.from_smiles("CC(C)(C)C(C)Br.O")
end = TDStructure.from_smiles("C(C)(C)(C(C)C)O.Br")

# +
# from neb_dynamics.solvator import Solvator

# solv = Solvator(n_solvent=8)

# start = solv.solvate_single_td(start)
# end = solv.solvate_single_td(end)
# -

from neb_dynamics.nodes.node3d_water import Node3D_Water
from nodes.node3d import Node3D

from neb_dynamics.Inputs import NetworkInputs

# +
"""Must run script like this: python -m examples.xtb"""

from qcio import CalcType, ProgramInput, Structure

from qcop import compute

# Create the structure
# Can also open a structure from a file
# structure = Structure.open("path/to/h2o.xyz")
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[  # type: ignore
        [0.0, 0.0, 0.0],
        [0.52421003, 1.68733646, 0.48074633],
        [1.14668581, -0.45032174, -1.35474466],
    ],
)

# Define the program input
prog_input = ProgramInput(
    structure=structure,
    calctype=CalcType.energy,
    model={"method": "GFN2xTB"},  # type: ignore
    keywords={"max_iterations": 150},
)


output = compute("xtb", prog_input)
print(output)

# +
ugi_fp = Path("/home/jdep/T3D_data/template_rxns/Ugi-Reaction/")
nosigugi_fp = Path("/home/jdep/T3D_data/Ugi_NOSIG/")
yessigugi_fp = Path("/home/jdep/T3D_data/Ugi_YESSIG/")

cr_fp = Path("/home/jdep/T3D_data/AutoMG_v0/")

enolcr_fp = Path("/home/jdep/T3D_data/EnolateClaisen")

debug_fp = Path("/home/jdep/T3D_data/debug_automg/")

nosigCR = Path("/home/jdep/T3D_data/ClaisenNoSIG/")

rgs_fp = Path("/home/jdep/T3D_data/RGS_Network")

enol_solv_fp = Path("/home/jdep/T3D_data/EnolateClaisen_8Water")

cr_rxnmapper = Path("/home/jdep/T3D_data/ClaisenRxnMapper")
sn1_rxnmapper = Path("/home/jdep/T3D_data/SN1RxnMapper")


network_inps = NetworkInputs(
    subsample_confs=True,
    n_max_conformers=5,
    use_slurm=False,
    verbose=1,
    tolerate_kinks=False,
    network_nodes_are_conformers=False,
    maximum_barrier_height=50000
)

bob = NetworkBuilder(
    data_dir=sn1_rxnmapper,
    start=start.molecule_rp.smiles,
    end=end.molecule_rp.smiles,
    chain_inputs=ChainInputs(k=0.1, delta_k=0.09, skip_identical_graphs=True, use_maxima_recyling=True, node_class=Node3D_Water),
    network_inputs=network_inps

)
# -

pot = bob.run_and_return_network()


# +
# pot.draw()

# +
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import networkx as nx

# %matplotlib ipympl

alpha = 1
plot_enes = True
n = 100

g = pot.graph
positions = nx.spring_layout(g)

s = 1.5
plt.rcParams["figure.figsize"] = [7.50 * s, 3.50 * s]
plt.rcParams["figure.autolayout"] = True

def scatter_positions(ax, positions, zs, plot_enes=True):
    xs, ys, labels = [], [], []
    for node_ind, pos in positions.items():
        if len(g.nodes[node_ind]['node_energies']) < 1:
            continue
        ene = min(g.nodes[node_ind]['node_energies'])
        for neighbor in g.neighbors(node_ind):
            if neighbor == node_ind:
                continue
            eA = float(g.edges[(neighbor, node_ind)]['reaction'].split(":")[-1])  # kcal/mol
            TS_ene = ene + (eA / 627.5)
            start_p = positions[node_ind]
            end_p = positions[neighbor]
            TS_p = (end_p + start_p) / 2
            xs.append(TS_p[0])
            ys.append(TS_p[1])
            zs.append(TS_ene if plot_enes else -np.exp(TS_ene))
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(ene if plot_enes else -np.exp(ene))
    return xs, ys, zs

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs, ys, zs = scatter_positions(ax, positions, [], plot_enes)

kernel = RBF(1.0, (-20.0, 1e5))
gpr = GaussianProcessRegressor(kernel=kernel)
train_data = np.array([xs, ys]).T
gpr.fit(train_data, zs)
print(gpr.score(train_data, zs))

x_vals = np.linspace(min(xs), max(xs), num=n)
y_vals = np.linspace(min(ys), max(ys), num=n)
xx, yy = np.meshgrid(x_vals, y_vals)
zz = np.array([[0.0] * len(xx)] * len(xx))

for i, col_val in enumerate(xx):
    for j, row_val in enumerate(col_val):
        pred_val = gpr.predict(np.array([[xx[i, j], yy[i, j]]]))[0]
        pred_val = max(min(pred_val, max(zs)), min(zs))
        zz[i, j] = pred_val

ax.plot_trisurf(xs, ys, zs, cmap='viridis', alpha=alpha)

for node_ind, pos in positions.items():
    if len(g.nodes[node_ind]['node_energies']) < 1:
        continue
    ene = min(g.nodes[node_ind]['node_energies'])
    for neighbor in g.neighbors(node_ind):
        if neighbor == node_ind:
            continue
        eA = float(g.edges[(neighbor, node_ind)]['reaction'].split(":")[-1])  # kcal/mol
        TS_ene = ene + (eA / 627.5)
        TS_p = (positions[node_ind] + positions[neighbor]) / 2
        ax.scatter(TS_p[0], TS_p[1], TS_ene if plot_enes else -np.exp(TS_ene), marker='x', color='red', s=80, depthshade=False)
    ax.scatter(pos[0], pos[1], ene if plot_enes else -np.exp(ene), s=50, color='black')
    ax.text(pos[0] + .05, pos[1] + .05, ene - .001 if plot_enes else -np.exp(ene - .001), s=f'node_{node_ind}')

ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlim3d(min(zs), max(zs))
ax.view_init(azim=45, elev=40)
plt.show()

# +
# p = [0, 3, 4, 18, 6]
# p = [0, 3, 4, 5, 6]
# p = [0, 5, 4]

# p = [0, 1, 2, 3, 4]
p = [0, 1, 2, 22, 23, 24, 4]

# -

list_of_chains  = path_to_list_of_chains(p, bob.leaf_objects)

c = path_to_chain(p, bob.leaf_objects)

c[11].is_identical(c[12])

c.to_trajectory()

c.plot_chain()

c_sig = Chain.from_xyz("/home/jdep/T3D_data/ugi_yessig_06102024.xyz", ChainInputs())

Molecule.draw_list(get_mechanism_mols(c_sig),mode='d3')

tsg = c_sig[30].tdstructure

tsto_trajectorydel_method = 'ub3lyp'

tsg.tc_model_basis = '6-31gss'

ts_opt = tsg.tc_geom_optimization("ts")

ts_opt

c_sig.plot_chain()

import matplotlib.pyplot as plt

s=7
fs=18
f, ax = plt.subplots(figsize=(2.16*s, s))
# plt.plot(c_nosig.integrated_path_length, c_nosig.energies_kcalmol,'o-')
plt.plot(c_sig.integrated_path_length, c_sig.energies_kcalmol,'o-')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel("Energies (kcal/mol)", fontsize=fs)
plt.xlabel("Reaction progression", fontsize=fs)

c_nosig = Chain.from_xyz("/home/jdep/T3D_data/ugi_nosig_06102024.xyz", ChainInputs())

c_nosig.plot_chain()

# +
# p = [0, 3, 4, 18, 25, 6]
# p.reverse()

# +
# pot.draw_shortest_to_node(6, mode='d3', weight='barrier')
# -

from neb_dynamics.treenode import TreeNode

fake_neb_objects = [NEB(initial_chain=None, parameters=NEBInputs(), optimizer=VelocityProjectedOptimizer) for i in range(len(list_of_chains))]

for n, chain in zip(fake_neb_objects, list_of_chains):
    n.optimized = chain

fake_leaves = [TreeNode(data=n, children=[], index=0) for n in fake_neb_objects]

from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB

ref = Refiner(v=True, cni=ChainInputs(k=0.1, delta_k=0.09, skip_identical_graphs=False, node_class=Node3D_TC_TCPB, do_parallel=False), nbi=NEBInputs(v=True, tol=0.002*BOHR_TO_ANGSTROMS), gii=GIInputs(nimages=12))

from neb_dynamics.MSMEP import MSMEP
from nodes.node3d import Node3D

m = MSMEP(neb_inputs=NEBInputs(v=True, tol=0.002*BOHR_TO_ANGSTROMS), chain_inputs=ChainInputs(k=0.1, delta_k=0.09, skip_identical_graphs=False, node_class=Node3D, do_parallel=False), gi_inputs=GIInputs(nimages=12), optimizer=VelocityProjectedOptimizer(timestep=0.5))

c.parameters = ChainInputs(k=0.1, delta_k=0.09, skip_identical_graphs=False, node_class=Node3D, do_parallel=False)

jj = Janitor(history_object=c, reaction_leaves=fake_neb_objects,
             msmep_object=m)

clean_msmep = jj.create_clean_msmep()


def merge_by_indices(
    insertions_inds, insertions_vals, orig_inds, orig_values
):

    insertions_inds = insertions_inds.copy()
    insertions_vals = insertions_vals.copy()
    orig_inds = orig_inds.copy()
    orig_values = orig_values.copy()
    out = []
    while len(insertions_inds) > 0:
        if 0 <= insertions_inds[0] <= orig_inds[0]:
            out.extend([leaf.data.optimized for leaf in insertions_vals[0].ordered_leaves])
            insertions_inds.pop(0)
            insertions_vals.pop(0)
        elif insertions_inds[0] == -1 and len(orig_values) == 0:
            out.extend([leaf.data.optimized for leaf in insertions_vals[0].ordered_leaves])
            insertions_inds.pop(0)
            insertions_vals.pop(0)
        else:
            out.append(orig_values[0].optimized)
            orig_inds.pop(0)
            orig_values.pop(0)

    if len(orig_values) > 0:
        out.extend([n.optimized for n in orig_values])

    return out


orig_leaves = jj.reaction_leaves
new_leaves = merge_by_indices(
            insertions_inds=jj.insertion_points,
            insertions_vals=jj.cleanup_trees,
            orig_inds=list(range(len(orig_leaves))),
            orig_values=orig_leaves,
        )
print("after:", len(new_leaves))
# new_chains = [leaf.data.optimized for leaf in new_leaves]
clean_out_chain = Chain.from_list_of_chains(
    new_leaves, parameters=jj.starting_chain.parameters
)

pot.gradph.

sorted(list(enumerate([(pot.graph.degree[i], pot.graph.nodes[i].energy) for i in range(len(pot.graph.nodes))])),key=lambda x:x[1], reverse=True)

pot.graph.nodes[4]['molecule'].draw(mode='d3')

pot.draw()

clean_out_chain.plot_chain()
clean_out_chain.write_to_disk("/home/jdep/T3D_data/ugi_nosig_06102024.xyz")

clean_out_chain[155].is_identical(clean_out_chain[156])

# +

c.plot_chain()
c.write_to_disk("/home/jdep/T3D_data/ugi_yessig_06102024.xyz")
# -

clean_out_chain.write_to_disk("/home/jdep/T3D_data/hijan.xyz")

len(list_of_cleanup_nebs)

jj._merge_cleanups_and_leaves

clean_out_chain.plot_chain()



jj.cleanup_trees[1].output_chain.plot_chain()

[leaf.data.optimized.plot_chain() for leaf in jj.cleanup_trees[0].ordered_leaves[1:]]



frag1 = clean_out_chain.copy()
frag1.nodes = clean_out_chain.nodes[:24]



frag2 = clean_out_chain.copy()
frag2.nodes = clean_out_chain.nodes[36:]

joined = Chain.from_list_of_chains([frag1, frag2], ChainInputs())

clean_out_chain.write_to_disk("/home/jdep/T3D_data/ugi_best.xyz")



pot.draw_reaction_graph()

fs = 18
s = 8
f, ax = plt.subplots(figsize=(1.16*s, s))
plt.plot(clean_out_chain.integrated_path_length, clean_out_chain.energies_kcalmol, 'o-')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel("Reaction coordinate", fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)",fontsize=fs)
plt.savefig("/home/jdep/T3D_data/msmep_draft/figures/UGI_path.svg")

jj.cleanup_trees[0].data.optimized[0].is_identical(c[24])

clean_msmep[35].is_identical(clean_msmep[36])

output = ref.create_refined_leaves(fake_leaves)

# +
# r = bob.run_and_return_network()

# +
# pot.draw()
# -

import networkx as nx

g = pot.graph
path = nx.shortest_path(g, source=0, target=4, weight='barrier')

from neb_dynamics.helper_functions import pairwise



import numpy as np

from chain import Chain

c = path_to_chain(path, leaf_objects=bob.leaf_objects)

c.plot_chain()

c[60].tdstructure.molecule_rp.draw(mode='d3')



pot.draw_shortest_to_node(4, mode='d3', weight='barrier')

pot.draw()

all_mols = [pot.graph.nodes[i]['molecule'] for i,_ in enumerate(pot.graph.nodes)]

# +
# Molecule.draw_list(all_mols, mode='d3',names=[f"node {i}" for i in range(len(pot.graph.nodes))])
# -

pot.draw_reaction_graph()

pot.draw_shortest_to_node(4, mode='d3', weight='work')

# +

# pot = bob.run_and_return_network()
# -

pot.draw_molecules_in_nodes()

import networkx as nx
import matplotlib.pyplot as plt

pot.draw()

bob.run_msmeps()

# +
from  dataclasses import dataclass
from neb_dynamics.pot import Pot
@dataclass
class KineticModel:
    pot: Pot
    temperature: float = 273 # Kelvin
    nsteps: int = 10000
    timestep: float = .1

    @property
    def transition_matrix(self):
        g = self.pot.graph
        adjmat = nx.adjacency_matrix(g,
                             weight='barrier',
                             nodelist=range(len(g.nodes))
                            ).A

        inf_barrier = np.inf
        adjmat[adjmat==0] = inf_barrier
        np.fill_diagonal(adjmat,inf_barrier)
        T = self.temperature
        beta=1. / ((0.008314)*T)

        trans_mat = np.exp(-beta*adjmat)*self.timestep
#         state_probabilities = [np.exp(-beta*g.nodes[i]['node_energy']) for i,_ in enumerate(g.nodes)]
#         state_probs = np.zeros_like(trans_mat)
#         for i, _ in enumerate(state_probs):
#             for j,_ in enumerate(state_probs):
#                 state_probs[i,j] = state_probabilities[i]/state_probabilities[j]

#         trans_mat = state_probs*trans_mat


        for i, _ in enumerate(trans_mat):
            col = trans_mat[:, i]
            # row = trans_mat[i, :]
            sumcol= sum(col)
            # sumrow = sum(row)
            trans_mat[i, i] = 1 - sumcol
            # trans_mat[i,i] = 1-sumrow
            assert np.isclose(sum(trans_mat[:,i]), 1.0), f'Probabilities not summing up to 1 ({sum(trans_mat[:,1])}) with timestep: {self.timestep}. Use a smaller one.'
        return trans_mat

    def run_simulation(self, population):
        g = self.pot.graph

        trans_mat = self.transition_matrix

        # population = np.zeros_like(trans_mat)

        traj = []
        pop = population.copy()

        pop_prev = pop
        keep_going = True
        change_thre=.0000001
        nsteps=self.nsteps
        count=0
        for i in range(nsteps):

        # while keep_going:
            count+=1
            new_pop = pop_prev@trans_mat.T
            delta_p = new_pop - pop_prev
            change = np.amax(abs(delta_p))
            pop_prev = new_pop

            if change < change_thre:
                keep_going=False
                # print(f"Done in {count} steps! {change}")
            traj.append(new_pop)

        return np.array(traj), new_pop

    def run_and_plot_populated_species(self, population, thresh=1/100):
        traj, fin_pop = kinx.run_simulation(population=population)
        inds_nonzero = np.where(fin_pop>thresh)

        f, ax = plt.subplots()
        for ind in inds_nonzero[0]:
            ax.plot(xs=list(range(len(traj[:, ind].tolist()))), ys=traj[:, ind].tolist(), label=ind)
        plt.legend(loc='right')
        plt.show()

        def _get_mols_from_inds(inds, graph):
            return [graph.nodes[i]['molecule'] for i in inds]

        mols = _get_mols_from_inds(inds_nonzero[0], pot.graph)
        yields = fin_pop[inds_nonzero]

        grouped_mols = []
        grouped_yields = {}
        for i, (mol,pyield) in enumerate(zip(mols, yields)):
            if mol not in grouped_mols:
                grouped_yields[len(grouped_mols)] = pyield
                grouped_mols.append(mol)


            else:

                ind_match = np.where([mol==refmol for refmol in grouped_mols])[0][0]
                grouped_yields[ind_match] += pyield



        sort_inds = np.argsort(fin_pop[inds_nonzero])
        g = self.pot.graph

        return Molecule.draw_list(grouped_mols, names=[f'yield={round(y*100,2)} %' for y in [b for a,b in grouped_yields.items()]], mode='d3')
# -

kinx = KineticModel(pot=pot,temperature=25+273, nsteps=18000, timestep=.1) # 18000*.1 timesteps = 30min

kinx.run_and_plot_populated_species(population)

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

eigvals, eigvecs = np.linalg.eigh(kinx.transition_matrix)

eigvals

population=np.zeros(len(pot.graph.nodes))
population.reshape(-1,1)
population[0]=1.0

# +
# pot.draw_molecules_in_nodes()
# -

traj, fin_pop = kinx.run_simulation(population=population)

# f, ax = plt.subplots()
# for ind in inds_nonzero[0]:
plt.plot(ys=traj[:, 0])
plt.show()

# +
# pot.draw_from_node(12)
# -

kinx.run_and_plot_populated_species(population)

evals, evecs = np.linalg.eigh(kinx.transition_matrix)

evals

list(enumerate(evals))

list(enumerate(np.round(evecs[10], 3)))

pot.graph.nodes[7]['molecule'].draw()

rd = ReactionData(data=bob.leaf_objects)

# +
all_spreads = []
all_rxn_spreads = []

for c in all_ps:
    max_en = c.energies_kcalmol.max()
    inds_to_consider = np.where(abs(c.energies_kcalmol - max_en) <= 5)[0]

    tds = [td for i, td in enumerate(c.to_trajectory()) if i in inds_to_consider]
    # print([td.coords for td in tds])
    # tds = [td.update_coords(td.coords_bohr) for td in tds]
    # print([td.coords for td in tds])

    sasas = [compute_SASA(td) for td in tds]
    spread = max(sasas) - min(sasas)
    std = np.std(sasas)
    all_spreads.append(spread)


    rxn_sasa = [compute_SASA(c[0].tdstructure), compute_SASA(c[-1].tdstructure)]
    rxn_spread = max(rxn_sasa) - min(rxn_sasa)
    all_rxn_spreads.append(rxn_spread)
# -

np.mean(all_spreads), np.std(all_spreads), np.median(all_spreads)

np.mean(all_rxn_spreads), np.std(all_rxn_spreads), np.median(all_rxn_spreads)

all_ps[0].plot_chain()

all_tsg = rd.get_all_TS('0-1')
print(len(all_tsg))
sub_tsg = Trajectory(bob.subselect_confomers(all_tsg, n_max=len(all_tsg), rmsd_cutoff=1))
print(len(sub_tsg))

# +
# bob.run_msmeps()
# -



x_points = np.linspace(min(xs), max(xs))
y_points = np.linspace(min(ys), max(ys))

# +
# Evaluating the spline at the new x values
y_new = spl(x_new)

# Plotting the original data and the interpolated spline
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'o', label='Original Points')
plt.plot(x_new, y_new, label='Spline Interpolation')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Spline Interpolation with BSpline')
plt.legend()
plt.grid(True)
plt.show()
# -

pot.draw_neighbors_of_node(12)



ps = pot.paths_from(4)
path = next(ps)
# path = next(ps)
# path = next(ps)
# path = next(ps)
pot.draw_from_single_path(path, mode='d3')

rd = ReactionData(data=bob.leaf_objects)

all_paths = rd.get_all_paths('0-2', barrier_thre=28)

all_paths[1].energies_kcalmol

tsg = all_paths[0].get_ts_guess()

from openbabel import openbabel

for atom in openbabel.OBMolAtomIter(tsg.molecule_obmol):
    print(atom.)

tsg.molecule_obmol.


def get_eA_from_edge(pot, child_node, parent_node):
    return float(pot.graph.edges[(child_node, parent_node)]['reaction'].split(':')[-1])


from neb_dynamics.helper_functions import pairwise


# +
def path_to_keys(path_indices):
    pairs = list(pairwise(path_indices))
    labels = [f"{a}-{b}" for a,b in pairs]
    return labels

def get_best_chain(list_of_chains):
    eAs = [c.get_eA_chain() for c in list_of_chains]
    return list_of_chains[np.argmin(eAs)]

def calculate_barrier(chain):
    return (chain.energies.max() - chain[0].energy)*627.5

def path_to_chain(path, leaf_objects):
    labels = path_to_keys(path)
    node_list = []
    for l in labels:

        node_list.extend(get_best_chain(leaf_objects[l]).nodes)
    c = Chain(node_list, ChainInputs())
    return c


# -

get_eA_from_edge(pot, child_node=1, parent_node=0)

# +
g = pot.graph
shortest_weighed_path = nx.shortest_path(g, source=0, target=4, weight='barrier')
shortest_path = nx.shortest_path(g, source=0, target=4, weight=None)

p = shortest_weighed_path
p2 = shortest_path
jan_path = path_to_chain(p, leaf_objects=bob.leaf_objects)
jan_path2 = path_to_chain(p2, leaf_objects=bob.leaf_objects)
print(p)
# -

work(jan_path),work(jan_path2)

plt.plot(jan_path.path_length, jan_path.energies,label='shortest_weighed')
plt.plot(jan_path2.path_length, jan_path2.energies,label='shortest')
plt.legend()
plt.show()

# +
key = '0-1'
bt = 80
rd = ReactionData(data=bob.leaf_objects)

rd.plot_all_paths(key, barrier_thre=bt)
all_ps = rd.get_all_paths(key, barrier_thre=bt)

# +

ind_rn = 0
print(key)
c = all_ps[ind_rn]
c.plot_chain()
Molecule.draw_list([c[0].tdstructure.molecule_rp, c[-1].tdstructure.molecule_rp],
                   mode='d3', size=(400,400), charges=True)
# -

# jan_path = path_to_chain([0,1,2, 3, 4], leaf_objects=rd.data)
jan_path = path_to_chain([0,1,2, 3, 4], leaf_objects=rd.data)

from neb_dynamics.Janitor import Janitor
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer

m = MSMEP(neb_inputs=NEBInputs(v=True), chain_inputs=ChainInputs(), gi_inputs=GIInputs(nimages=12), optimizer=VelocityProjectedOptimizer(timestep=0.5))
jj = Janitor(history_object=jan_path, msmep_object=m)

jan_clean = jj.create_clean_msmep()

jan_path.plot_chain()

rd = ReactionData(data=bob.leaf_objects)
for key in rd.data.keys():
    population = len(rd.data[key])
    if population >= 1:
        parent_node = int(key.split('-')[0])
        child_node = int(key.split('-')[1])
        eA = get_eA_from_edge(pot, child_node=child_node, parent_node=parent_node)
        print(population, key, round(eA,1))

pot.draw()







c.to_trajectory()

tsgs = rd.get_all_TS(key)

sub_tsgs = bob.subselect_confomers(tsgs, n_max=3, rmsd_cutoff=1)

Trajectory(sub_tsgs).energies_xtb()

Trajectory(sub_tsgs)

pot.draw()


def vis_nma(td, nma, dr=0.1):
    return Trajectory([td.displace_by_dr(-dr*nma), td, td.displace_by_dr(dr*nma)])


