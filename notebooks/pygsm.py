from neb_dynamics.trajectory import Trajectory
from neb_dynamics.tdstructure import TDStructure

td = TDStructure.from_smiles("COCO")

# +
# import pyGSM
# -

from pyGSM.coordinate_systems.delocalized_coordinates import DelocalizedInternalCoordinates
from pyGSM.coordinate_systems.primitive_internals import PrimitiveInternalCoordinates
from pyGSM.coordinate_systems.topology import Topology
from pyGSM.growing_string_methods import DE_GSM
from pyGSM.level_of_theories.ase import ASELoT
from pyGSM.optimizers.eigenvector_follow import eigenvector_follow
from pyGSM.optimizers.lbfgs import lbfgs
from pyGSM.potential_energy_surfaces import PES
from pyGSM.utilities import nifty
from pyGSM.utilities.elements import ElementData
from pyGSM.molecule import Molecule


# +
def post_processing(gsm, analyze_ICs=False, have_TS=True):
    plot(fx=gsm.energies, x=range(len(gsm.energies)), title=gsm.ID)

    ICs = []
    ICs.append(gsm.nodes[0].primitive_internal_coordinates)

    # TS energy
    if have_TS:
        minnodeR = np.argmin(gsm.energies[:gsm.TSnode])
        TSenergy = gsm.energies[gsm.TSnode] - gsm.energies[minnodeR]
        print(" TS energy: %5.4f" % TSenergy)
        print(" absolute energy TS node %5.4f" % gsm.nodes[gsm.TSnode].energy)
        minnodeP = gsm.TSnode + np.argmin(gsm.energies[gsm.TSnode:])
        print(" min reactant node: %i min product node %i TS node is %i" % (minnodeR, minnodeP, gsm.TSnode))

        # ICs
        ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
        ICs.append(gsm.nodes[gsm.TSnode].primitive_internal_values)
        ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open('IC_data_{:04d}.txt'.format(gsm.ID), 'w') as f:
            f.write("Internals \t minnodeR: {} \t TSnode: {} \t minnodeP: {}\n".format(minnodeR, gsm.TSnode, minnodeP))
            for x in zip(*ICs):
                f.write("{0}\t{1}\t{2}\t{3}\n".format(*x))

    else:
        minnodeR = 0
        minnodeP = gsm.nR
        print(" absolute energy end node %5.4f" % gsm.nodes[gsm.nR].energy)
        print(" difference energy end node %5.4f" % gsm.nodes[gsm.nR].difference_energy)
        # ICs
        ICs.append(gsm.nodes[minnodeR].primitive_internal_values)
        ICs.append(gsm.nodes[minnodeP].primitive_internal_values)
        with open('IC_data_{}.txt'.format(gsm.ID), 'w') as f:
            f.write("Internals \t Beginning: {} \t End: {}".format(minnodeR, gsm.TSnode, minnodeP))
            for x in zip(*ICs):
                f.write("{0}\t{1}\t{2}\n".format(*x))

    # Delta E
    deltaE = gsm.energies[minnodeP] - gsm.energies[minnodeR]
    print(" Delta E is %5.4f" % deltaE)

def cleanup_scratch(ID):
    cmd = "rm scratch/growth_iters_{:03d}_*.xyz".format(ID)
    os.system(cmd)
    cmd = "rm scratch/opt_iters_{:03d}_*.xyz".format(ID)


# -

def gsm_to_ase_atoms(gsm: DE_GSM):
    # string
    frames = []
    for energy, geom in zip(gsm.energies, gsm.geometries):
        at = Atoms(symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom])
        at.info["energy"] = energy
        frames.append(at)

    # TS
    ts_geom = gsm.nodes[gsm.TSnode].geometry
    ts_atoms = Atoms(symbols=[x[0] for x in ts_geom], positions=[x[1:4] for x in ts_geom])

    return frames, ts_atoms


import os

from xtb.ase.calculator import XTB

from ase import Atoms
import ase.io

tr = Trajectory.from_xyz("/home/jdep/T3D_data/dlfind_vs_jan/claisen_initial_guess.xyz")

atoms_reactant = tr[0].to_ASE_atoms()
atoms_product = tr[-1].to_ASE_atoms()


calculator = XTB(method="GFN2-xTB", accuracy=0.001)
fixed_reactant=False
fixed_product=False


# optimizer
optimizer_method = "eigenvector_follow"  # OR "lbfgs"
line_search = 'NoLineSearch'  # OR: 'backtrack'
only_climb = True
# 'opt_print_level': args.opt_print_level,
step_size_cap = 0.1  # DMAX in the other wrapper
# molecule
coordinate_type = "TRIC"
# 'hybrid_coord_idx_file': args.hybrid_coord_idx_file,
# 'frozen_coord_idx_file': args.frozen_coord_idx_file,
# 'prim_idx_file': args.prim_idx_file,
# GSM
# gsm_type = "DE_GSM"  # SE_GSM, SE_Cross
num_nodes = 11  # 20 for SE-GSM
# 'isomers_file': args.isomers,   # driving coordinates, this is a file
add_node_tol = 0.1  # convergence for adding new nodes
conv_tol = 0.0005  # Convergence tolerance for optimizing nodes
conv_Ediff = 100.  # Energy difference convergence of optimization.
# 'conv_dE': args.conv_dE,
conv_gmax = 100.  # Max grad rms threshold
# 'BDIST_RATIO': args.BDIST_RATIO,
# 'DQMAG_MAX': args.DQMAG_MAX,
# 'growth_direction': args.growth_direction,
ID = 0
# 'gsm_print_level': args.gsm_print_level,
max_gsm_iterations = 100
max_opt_steps = 3  # 20 for SE-GSM
# 'use_multiprocessing': args.use_multiprocessing,
# 'sigma': args.sigma,
# LOT
lot = ASELoT.from_options(calculator,
                          geom=[[x.symbol, *x.position] for x in atoms_reactant],
                          ID=ID)
# PES
pes_obj = PES.from_options(lot=lot, ad_idx=0, multiplicity=1)
# Build the topology
nifty.printcool("Building the topologies")
element_table = ElementData()
elements = [element_table.from_symbol(sym) for sym in atoms_reactant.get_chemical_symbols()]
topology_reactant = Topology.build_topology(
    xyz=atoms_reactant.get_positions(),
    atoms=elements
)
topology_product = Topology.build_topology(
    xyz=atoms_product.get_positions(),
    atoms=elements
)
# Union of bonds
# debated if needed here or not
for bond in topology_product.edges():
    if bond in topology_reactant.edges() or (bond[1], bond[0]) in topology_reactant.edges():
        continue
    print(" Adding bond {} to reactant topology".format(bond))
    if bond[0] > bond[1]:
        topology_reactant.add_edge(bond[0], bond[1])
    else:
        topology_reactant.add_edge(bond[1], bond[0])
# +
# primitive internal coordinates
nifty.printcool("Building Primitive Internal Coordinates")

prim_reactant = PrimitiveInternalCoordinates.from_options(
    xyz=atoms_reactant.get_positions(),
    atoms=elements,
    topology=topology_reactant,
    connect=coordinate_type == "DLC",
    addtr=coordinate_type == "TRIC",
    addcart=coordinate_type == "HDLC",
)
# -
prim_product = PrimitiveInternalCoordinates.from_options(
    xyz=atoms_product.get_positions(),
    atoms=elements,
    topology=topology_product,
    connect=coordinate_type == "DLC",
    addtr=coordinate_type == "TRIC",
    addcart=coordinate_type == "HDLC",
)
# add product coords to reactant coords
prim_reactant.add_union_primitives(prim_product)
# Delocalised internal coordinates
nifty.printcool("Building Delocalized Internal Coordinates")
deloc_coords_reactant = DelocalizedInternalCoordinates.from_options(
    xyz=atoms_reactant.get_positions(),
    atoms=elements,
    connect=coordinate_type == "DLC",
    addtr=coordinate_type == "TRIC",
    addcart=coordinate_type == "HDLC",
    primitives=prim_reactant
)
# Molecules
nifty.printcool("Building the reactant object with {}".format(coordinate_type))
from_hessian = optimizer_method == "eigenvector_follow"
molecule_reactant = Molecule.from_options(
    geom=[[x.symbol, *x.position] for x in atoms_reactant],
    PES=pes_obj,
    coord_obj=deloc_coords_reactant,
    Form_Hessian=from_hessian
)
molecule_product = Molecule.copy_from_options(
    molecule_reactant,
    xyz=atoms_product.get_positions(),
    new_node_id=num_nodes - 1,
    copy_wavefunction=False
)
# optimizer
nifty.printcool("Building the Optimizer object")
opt_options = dict(print_level=1,
                   Linesearch=line_search,
                   update_hess_in_bg=not (only_climb or optimizer_method == "lbfgs"),
                   conv_Ediff=conv_Ediff,
                   conv_gmax=conv_gmax,
                   DMAX=step_size_cap,
                   opt_climb=only_climb)
if optimizer_method == "eigenvector_follow":
    optimizer_object = eigenvector_follow.from_options(**opt_options)
elif optimizer_method == "lbfgs":
    optimizer_object = lbfgs.from_options(**opt_options)
else:
    raise NotImplementedError
# GSM
nifty.printcool("Building the GSM object")
gsm = DE_GSM.from_options(
    reactant=molecule_reactant,
    product=molecule_product,
    nnodes=num_nodes,
    CONV_TOL=conv_tol,
    CONV_gmax=conv_gmax,
    CONV_Ediff=conv_Ediff,
    ADD_NODE_TOL=add_node_tol,
    growth_direction=0,  # I am not sure how this works
    optimizer=optimizer_object,
    ID=ID,
    print_level=1,
    mp_cores=1,  # parallelism not tested yet with the ASE calculators
    interp_method="DLC",
)
# optimize reactant and product if needed
if not fixed_reactant:
    nifty.printcool("REACTANT GEOMETRY NOT FIXED!!! OPTIMIZING")
    path = os.path.join(os.getcwd(), 'scratch', f"{ID:03}", "0")
    optimizer_object.optimize(
        molecule=molecule_reactant,
        refE=molecule_reactant.energy,
        opt_steps=100,
        path=path
    )

# +

if not fixed_product:
    nifty.printcool("PRODUCT GEOMETRY NOT FIXED!!! OPTIMIZING")
    path = os.path.join(os.getcwd(), 'scratch', f"{ID:03}", str(num_nodes - 1))
    optimizer_object.optimize(
        molecule=molecule_product,
        refE=molecule_product.energy,
        opt_steps=100,
        path=path
    )
# -
# set 'rtype' as in main one (???)
if only_climb:
    rtype = 1
# elif no_climb:
#     rtype = 0
else:
    rtype = 2


# +

# %%time
# do GSM
nifty.printcool("Main GSM Calculation")
gsm.go_gsm(max_gsm_iterations, max_opt_steps, rtype=rtype)
# write the results into an extended xyz file
string_ase, ts_ase = gsm_to_ase_atoms(gsm)
ase.io.write(f"opt_converged_{gsm.ID:03d}_ase.xyz", string_ase)
ase.io.write(f'TSnode_{gsm.ID}.xyz', string_ase)
from pyGSM.utilities.cli_utils import plot
# -

import numpy as np

# post processing taken from the main wrapper, plots as well
post_processing(gsm, have_TS=True)
cleanup_scratch(gsm.ID)


def _split_gsm_geom(geom):
    xyz = []
    symbols = []
    for row in geom:
        symbols.append(row[0])
        xyz.append(row[1:])
    return symbols, np.array(xyz)


def td_from_gsm_geom(geom):
    symbs, coords = _split_gsm_geom(geom)

    return TDStructure.from_coords_symbols(coords=coords, symbols=symbs)


tr_gsm = Trajectory([td_from_gsm_geom(g) for g in gsm.geometries])

c_gsm = Chain.from_traj(tr_gsm, cni)

# ## neb

from neb_dynamics.Inputs import ChainInputs, NEBInputs
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from neb_dynamics.NEB import NEB

from chain import Chain

cni = ChainInputs(k=0.1, delta_k=0.01, node_freezing=True, use_maxima_recyling=True)

nbi = NEBInputs(v=1, tol=0.001*BOHR_TO_ANGSTROMS, climb=True)

tr_xtb = tr.run_geodesic(nimages=11)

c = Chain.from_traj(tr_xtb, parameters=cni)

from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer

opt = VelocityProjectedOptimizer()

n = NEB(initial_chain=c, parameters=nbi, optimizer=opt)

output = n.optimize_chain()

n.grad_calls_made

c_neb = output[2]

import matplotlib.pyplot as plt

plt.plot(c_gsm.path_length, c_gsm.energies, 'o-', label='gsm')
plt.plot(c_neb.path_length, c_neb.energies, 'o-', label='neb')
plt.legend()

c_neb.get_ts_guess()

c_gsm.get_ts_guess()


