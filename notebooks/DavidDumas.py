# +
from retropaths.molecules.molecule import Molecule
from IPython.core.display import HTML
from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.abinitio.tdstructure import TDStructure
# from neb_dynamics.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB, NoneConvergedException
from chain import Chain
from nodes.node3d import Node3D

from neb_dynamics.MSMEP import MSMEP
import matplotlib.pyplot as plt
from retropaths.reactions.template import ReactionTemplate
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
import numpy as np
from scipy.signal import argrelextrema
from retropaths.helper_functions import pairwise
import retropaths.helper_functions as hf
import networkx as nx

HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

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

#  # Molecule Creation

# mol = Molecule.from_smiles("C1=CC(=CC=C1C(C)(C)C2=CC=C(C=C2)O[H])O[H].O=C=O.[Na+].[O-][H].[Na+].[O-][H]")
mol = Molecule.from_smiles("C1=CC(=CC=C1C(C)(C)C2=CC=C(C=C2)O[H])O[H].[Cs+].[Cs+].C(=O)([O-])[O-]")
# mol = Molecule.from_smiles("C1=CC(=CC=C1C(C)(C)C2=CC=C(C=C2)[O-])[O-].[Cs+].[Cs+].C(=O)([O-])[O-].O")
mol.draw(mode='d3',size=(500,500))

# forming = [(11,18),(35,23),(17,18),(38,21)]
# forming = [(11,18),(17,18), (15,33)]
# forming_s = [(5,19),(38, 9)]
forming_s = [(19,9),(17,22),(18,22)]
# forming_d = [(6,22)]
forming_d = []
forming=forming_s+forming_d

# deleting = [(35,11),(38,15)]
# deleting = [(33,11)]
# deleting = [(5,6),(6,9),(19,22),(38,16)]
deleting = [(9,6), (22,19)]

# c_changes = [(17,-1), (22,+1), (20,-1)]
c_changes = [(6,+1), (22,+1), (17,-1),(18,-1)]

# +
d1 = {'charges': c_changes,
      'delete':deleting,
      'single':forming_s,
     'double':forming_d}
conds = Conditions()
rules = Rules()
cg = []

temp1 = ReactionTemplate.from_components(name='David', reactants=mol,changes_react_to_prod_dict=d1, conditions=conds, rules=rules,collapse_groups=cg)
# temp1 = ReactionTemplate.from_components(name='Wittig', reactants=mol,changes_react_to_prod_dict=d1, conditions=conds, rules=rules)
# -

temp1.draw(mode='d3',size=(500,500), charges=True, node_index=True)

from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Inputs import NEBInputs, GIInputs, ChainInputs
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

root = TDStructure.from_RP(temp1.reactants, charge=temp1.reactants.charge)

# +
deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in deleting]
forming_list = [Changes3D(start=s,end=e, bond_order=2) for s,e in forming]

c3d_list = Changes3DList(deleted=deleting_list,forming=forming_list, charges=c_changes)
# -

root = root.pseudoalign(c3d_list)

root

# +
# root_opt = root.xtb_geom_optimization()

# +
# root_opt
# -

# target = root_opt.copy()
target = root.copy()

target.add_bonds(c3d_list.forming)

target.delete_bonds(c3d_list.deleted)

target = TDStructure.from_RP(temp1.products)

target.gum_mm_optimization()
target.mm_optimization("gaff")

target.gum_mm_optimization()

target.molecule_rp.draw(mode='d3')

from neb_dynamics.tdstructure import TDStructure as TD2_neb

td2 = TD2_neb(molecule_obmol=target.molecule_obmol)

td2_opt_tr = td2.xtb_geom_optimization(return_traj=True)

td2_root = TD2_neb(molecule_obmol=root.molecule_obmol)

td2_root_opt_tr = td2_root.xtb_geom_optimization(return_traj=True)

# +
# target_opt = target.xtb_geom_optimization()
# -

target_opt = td2_opt_tr[-1]
target_opt

root_opt = td2_root_opt_tr[-1]

tr = Trajectory([root_opt, target_opt]).run_geodesic(nimages=12)

nbi = NEBInputs(v=True)
cni = ChainInputs(k=0.1,delta_k=0.09, use_maxima_recyling=True, node_freezing=True)
gii = GIInputs(nimages=15)
m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=VelocityProjectedOptimizer(timestep=.1, activation_tol=0.01))

root_opt.to_xyz("/home/jdep/T3D_data/DavidDumas_Frag/start.xyz")

target_opt.to_xyz("/home/jdep/T3D_data/DavidDumas_Frag/end.xyz")

# traj = Trajectory([root_opt, target_opt]).run_geodesic(nimages=15)
# root = TDStructure.from_xyz("/home/jdep/T3D_data/template_rxns/DavidDumas_ortho/start.xyz")
root = TDStructure.from_xyz("/home/jdep/T3D_data/DavidDumas_ortho/start.xyz")

target_opt.to_xyz("/home/jdep/T3D_data/DavidDumas_Cesium/end.xyz")

# +
root = TDStructure.from_xyz("/home/jdep/T3D_data/DavidDumas_Cesium/start.xyz")
root.mm_optimization("gaff")
dr= np.zeros_like(root.coords)

n = -3
dr[17] = [n,n,n]
dr[18] = [n,n,n]
dr[19] = [n,n,n]
# dr[20] = [n,n,n]

m = -3
dr[21] = [m,m,m]
# dr[21] = [m,m,m]
# dr[40] = [m,m,m]

root = root.displace_by_dr(dr)
# -

root_opt = root.xtb_geom_optimization()

root.mm_optimization('gaff')

list(enumerate(root.symbols))

# +
# root_opt = root.xtb_geom_optimization()
# -

from neb_dynamics.tdstructure import TDStructure

# root_opt = root.xtb_geom_optimization()
root = TDStructure.from_xyz("/home/jdep/T3D_data/DavidDumas_Frag/start.xyz")
target = TDStructure.from_xyz("/home/jdep/T3D_data/DavidDumas_Frag/end.xyz")

# # Network Creation

from neb_dynamics.NetworkBuilder import NetworkBuilder, ReactionData
from pathlib import Path

# +
ugi_p = Path("/home/jdep/T3D_data/template_rxns/Auto/")
dd_frag_p = Path("/home/jdep/T3D_data/DavidDumas_Frag/")
cr_p  = Path("/home/jdep/T3D_data/AutoMG_v0/")


bob_frag = NetworkBuilder(start=None,end=None,
                    data_dir=dd_frag_p,
                     subsample_confs=True,
                     use_slurm=True,
                    n_max_conformers=20, tolerate_kinks=False,
                    verbose=False)

# +
# bob_frag.create_submission_scripts()
# -

pot = bob_frag.create_rxn_network()

pot.draw(mode='d3')

pot.draw_from_node(7)

pot.draw_molecules_in_nodes()

pot.draw_from_node(10)

pot.draw_from_node(7)

c_xtb  = bob_frag.leaf_objects['4-7'][0]

from neb_dynamics.Refiner import Refiner
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer

opt = VelocityProjectedOptimizer(timestep=0.5, activation_tol=0.1)
ref = Refiner(method='ub3lyp',v=True, cni=ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D_TC, use_maxima_recyling=True),
              nbi=NEBInputs(max_steps=500, v=True, preopt_with_xtb=False), gii=GIInputs(nimages=12), optimizer=opt)

output = ref.refine_xtb_chain(c_xtb)

tsg.tc_model_basis = 'def2-svp'

tsg.energy_tc()

tsg.tc_freq_calculation()

tsg

tsg.energy_xtb()

from neb_dynamics.tdstructure import TDStructure

TDStructure.from_xyz("/home/jdep/triforce.xyz")

len(bob_frag.leaf_objects['0-20'])

bob_frag.leaf_objects['0-20'][2].to_trajectory()

pot.draw_from_node(20)

pot.draw_molecules_in_nodes()

import networkx as nx

nx.adjacency_matrix(pot.graph)

nx.shortest_path(pot.graph, 0, 81)

pot.draw_from_node(81)

c2 = path_to_chain([0, 42, 4, 26, 84, 83, 82, 81], bob_frag.leaf_objects)

c2.plot_chain()

p =[0, 42, 4, 26, 84, 83, 82, 81]
p.reverse()

# +

pot.draw_from_single_path(p, mode='d3')
# -

nx.shortest_path(pot.graph, source=0, target=4,weight='barrier')

import numpy as np
from chain import Chain
from neb_dynamics.Inputs import ChainInputs

c = path_to_chain([0, 1, 42, 8, 9, 48, 4], bob_frag.leaf_objects)

c.write_to_disk("/home/jdep/T3D_data/hi_jonathan.xyz")

from neb_dynamics.molecule import Molecule



ind=48
Molecule.draw_list([c[ind].tdstructure.molecule_rp, c[ind+12].tdstructure.molecule_rp], mode='d3', size=(400,400))

pot = bob_frag.create_rxn_network()

pot.draw_from_node(1)

# pot = bob.run_and_return_network()
pot_cesium = bob_cesium.create_rxn_network()

bob_sodium = NetworkBuilder(start=None,end=None,data_dir=Path("/home/jdep/T3D_data/DavidDumas_ortho"),
                     subsample_confs=False,
                     use_slurm=True,
                    n_max_conformers=20, tolerate_kinks=False,
                    verbose=False)

pot.draw_from_single_path()

pot_sodium = bob_sodium.create_rxn_network()

c.plot_chain()

pot_sodium.draw()

pot_sodium.draw_molecules_in_nodes()

shortest = nx.shortest_path(pot_sodium.graph, weight='barrier', source=0, target=25)
shortest.reverse()
pot_sodium.draw_from_single_path(shortest)

# +
# bob_sodium.leaf_objects['6-23'][0].to_trajectory()
# -

shortest = nx.shortest_path(pot_cesium.graph, weight='barrier', source=0, target=3)
shortest.reverse()
pot_cesium.draw_from_single_path(shortest)

pot_cesium.draw(mode='d3')

pot_cesium.draw_molecules_in_nodes()

# +
# pot_sodium.draw()
# pot_cesium.draw_from_node(3)
# -

pot.draw_neighbors_of_node(4)

nx.shortest_path(pot_cesium.graph, 0,3, weight='barrier')

from neb_dynamics.Inputs import ChainInputs

c = path_to_chain([0, 1, 4, 3], bob_cesium.leaf_objects)

c.plot_chain()

c.to_trajectory()

from neb_dynamics.Janitor import Janitor
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer

m = MSMEP(neb_inputs=NEBInputs(v=True), chain_inputs=ChainInputs(),gi_inputs=GIInputs(nimages=12),optimizer=VelocityProjectedOptimizer(timestep=0.5))

from neb_dynamics.treenode import TreeNode

j = Janitor(history_object=TreeNode(data=neb,children=[],index=0), reaction_leaves=[path_to_chain([0,1], bob.leaf_objects),path_to_chain([1,4],bob.leaf_objects)],
           msmep_object=m)

j.create_clean_msmep()

c.plot_chain()
c.to_trajectory()

c[24].tdstructure

bob2 = NetworkBuilder(start=root,end=target,data_dir=Path("/home/jdep/T3D_data/DavidDumas_ortho/"),
                     subsample_confs=False,
                     use_slurm=True,
                    n_max_conformers=20, tolerate_kinks=True,
                    verbose=False)

pot2 = bob2.create_rxn_network()

p = nx.shortest_path(pot.graph, source=0,target=3,weight='barrier')
p

cs = []
for i, p in enumerate(list(pot.paths_from(3))):
    p.reverse()
    c = path_to_chain(p, bob.leaf_objects)
    plt.plot(c.path_length, c.energies,'-', label=f'path_{i}')
    cs.append(c)
plt.legend()

c.plot_chain()

cs[3].plot_chain()
cs[3].to_trajectory()

r = bob.leaf_objects['0-6'][0][0].tdstructure
ts = bob.leaf_objects['0-6'][0].get_ts_guess()
p = bob.leaf_objects['0-6'][0][-1].tdstructure

(ts.energy_xtb() - r.energy_xtb())*627.5

r.symbols

symbsCS = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
       'C', 'C', 'O', 'O', 'O', 'C', 'O', 'Cs', 'O', 'Cs', 'O', 'H', 'H',
       'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
       'H', 'H', 'H']
r_cs = TDStructure.from_coords_symbols(coords=r.coords, symbols=symbsCS)

c = path_to_chain(p, bob.leaf_objects)
# c = path_to_chain([0,1,2,3,4,5], bob.leaf_objects)

c.plot_chain()
c.to_trajectory()

# +
# pot.draw_neighbors_of_node(18)

# +
# c.to_trajectory()
# -

from neb_dynamics.Inputs import ChainInputs

pot.draw_from_node(22)

node6_confs = pot.graph.nodes[6]['conformers']

node5_confs = pot.graph.nodes[5]['conformers']

sub_node6 = bob.subselect_confomers(Trajectory([td.tdstructure for td in node6_confs]))

sub_node5 = bob.subselect_confomers(Trajectory([td.tdstructure for td in node5_confs]))

bob.create_submission_scripts(start_confs=sub_node6, end_confs=sub_node5, sub_id_name='_manual_',file_stem_name='manual')

# +
# bob.run_msmeps()

# +
# pot = bob.run_and_return_network()

# +
# pot.graph.edges[(7,6)]['list_of_nebs'][0].to_trajectory()
# -

from neb_dynamics.Refiner import Refiner
from neb_dynamics.nodes.Node3D_TC import Node3D_TC

ref = Refiner(cni=ChainInputs(node_class=Node3D_TC),resample_chain=False)

all_cxtb  = rd.get_all_paths('0-1')

c_xtb = all_cxtb[0]

c_dft = ref.convert_to_dft(c_xtb)

tsg = c_dft.get_ts_guess()

ts = tsg.tc_geom_optimization('ts')

c_dft[-3].tdstructure

pot.draw_from_node(7)

edge = (7,6)
pot.graph.edges[edge]['list_of_nebs'][0].plot_chain()
pot.graph.edges[edge]['list_of_nebs'][0].to_trajectory()

rd= ReactionData(bob.leaf_objects)

for key in rd.data:
    population = len(rd.data[key])
    # if population > 10:
    print(population, key)

pot.draw()

rd.get_all_paths(

from pathlib import Path

h.write_to_disk(Path("/home/jdep/T3D_data/template_rxns/Baeyer-Villiger_oxidation/without_h3o.xyz"))

out.write_to_disk(Path("/home/jdep/T3D_data/template_rxns/Baeyer-Villiger_oxidation/output_chain.xyz"))

step = 7
out[step*15].tdstructure.molecule_rp.draw(mode='oe')

out[(step*15)+14].tdstructure.molecule_rp.draw(mode='oe')

leaves = [l.data.chain_trajectory[-1] for l in h.ordered_leaves if l.data]

leaves[0].plot_chain()

out.plot_chain()


