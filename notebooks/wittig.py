from retropaths.abinitio.tdstructure import TDStructure

td = TDStructure.from_smiles("O")

from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB

node = Node3D_TC_TCPB(td)

node2 = node.copy()

node.gradient

import numpy as np

np.array(wtf).reshape(td.coords.shape)

dir(wtf)


def _tcpb_input_string(tdstruct):
    tc_inp_str = f"""method {tdstruct.tc_model_method}
    basis {tdstruct.tc_model_basis}
    """
    kwds_strings = "\n".join(f"{pair[0]}  {pair[1]}" for pair in tdstruct.tc_kwds.items())

    tc_inp_str+=kwds_strings
    
    return tc_inp_str


_tcpb_input_string(td)

# +
import sys
from ase.io import read
from ase import units
import time


# Set information about the server
host = "localhost"

# Get Port
port = 8888

str_inp = _tcpb_input_string(td)

fname = '/tmp/inputfile.in'
with open(fname,'w') as f:
    f.write(str_inp)


# tcfile = '/home/jdep/tc2.in' ### HERE
tcfile = fname







# +
# Set global treatment (for how TeraChem will handle wavefunction initial guess)
# 0 means continue and use casguess, 1 is keep global variables the same, but recalc casguess, 2 means reinitialize everything
globaltreatment = {"Cont": 0, "Cont_Reset": 1, "Reinit": 2}

# Information about initial QM region
# structures = read("/home/jdep/heyjona.xyz", index=":")  ### HERE
structures = [td]

# qmattypes = structures[0].get_chemical_symbols() ### HERE
qmattypes = td.symbols

# Attempts to connect to the TeraChem server
print(f"Attempting to connect to TeraChem server using host {host} and {port}.")
status = tc.connect(host, port)
if status == 0:
    print("Connected to TC server.")
elif status == 1:
    raise ValueError("Connection to TC server failed.")
elif status == 2:
    raise ValueError(
        "Connection to TC server succeeded, but the server is not available."
    )
else:
    raise ValueError("Status on tc.connect function is not recognized!")


# Setup TeraChem
status = tc.setup(str(tcfile), qmattypes) ### HERE
# status = tc.setup(tc_inp_str, qmattypes) ### HERE
if status == 0:
    print("TC setup completed with success.")
elif status == 1:
    raise ValueError(
        "No options read from TC input file or mismatch in the input options!"
    )
elif status == 2:
    raise ValueError("Failed to setup TC.")
else:
    raise ValueError("Status on tc_setup function is not recognized!")
# -

from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS

BOHR_TO_ANGSTROMS

# +
# %%time


for i, structure in enumerate(structures):
    qmcoords = structure.coords.flatten() / units.Bohr
    qmcoords = qmcoords.tolist()

    # Compute energy and gradient
    time.sleep(0.010)  # TCPB needs a small delay between calls
    if i == 0:
        totenergy, qmgrad, mmgrad, status = tc.compute_energy_gradient(
            qmattypes, qmcoords, globaltreatment=globaltreatment["Cont_Reset"]
        )
        print("Starting new positions file")
    else:
        print(f"Continuing with job {i}")
        totenergy, qmgrad, mmgrad, status = tc.compute_energy_gradient(
            qmattypes, qmcoords, globaltreatment=globaltreatment["Cont"]
        )
        

    print(f"Status: {status}")
    if status == 0:
        print("Successfully computed energy and gradients")
    elif status == 1:
        raise ValueError("Mismatch in the variables passed to compute_energy_gradient")
    elif status == 2:
        raise ValueError("Error in compute_energy_gradient.")
    else:
        raise ValueError(
            "Status on compute_energy_gradient function is not recognized!"
        )
# -

# %%time
td.energy_tc_local()

totenergy

# !ml Amber

tds = TDStructure.from_smiles("CCCCCCCCCC")

# %%time
tds.energy_tc()

# %%time
tds.energy_tc()

# +
from qcio import Molecule, ProgramInput, SinglePointOutput

from chemcloud import CCClient

water = Molecule(
    symbols=["O", "H", "H"],
    geometry=[
        [0.0000, 0.00000, 0.0000],
        [0.2774, 0.89290, 0.2544],
        [0.6067, -0.23830, -0.7169],
    ],
)

client = CCClient()

prog_inp = ProgramInput(
    molecule=water,
    model={"method": "wb97xd3", "basis": "def2-svp"},
    calctype="energy",  # Or "gradient" or "hessian"
    keywords={},
    files={'c0':c0}
)
future_result = client.compute("terachem", prog_inp, collect_files=True, collect_wavefunction=True)
output: SinglePointOutput = future_result.get()
# SinglePointOutput object containing all returned data
print(output.stdout)
print(output)
# The energy value requested
print(output.return_result)
print(output.files.keys())
# -

output.files.keys()

c0 = output.files['scr.geometry/c0']

# +
from retropaths.molecules.molecule import Molecule
from IPython.core.display import HTML
from retropaths.reactions.changes import Changes3DList, Changes3D
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Chain import Chain
from neb_dynamics.Node3D import Node3D

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
mol = Molecule.from_smiles('[P](=C)(C)(C)C.CC(=O)C')
# -

td = TDStructure.from_RP(mol)

# +
p_ind = 0
cp_ind = 1

me1 = 2
me2 = 3
me3 = 4

o_ind = 7
co_ind = 6

ind=0
settings = [

    (
        td.molecule_rp,
        {'charges': [], 'delete':[(co_ind, o_ind), (cp_ind, p_ind)], 'double':[(o_ind, p_ind), (cp_ind, co_ind)]},
        [
            (me1, p_ind, 'Me'),
            (me2, p_ind, 'Me'),
            (me3, p_ind, 'Me'),
  
        ],
        [Changes3D(start=s, end=e, bond_order=1) for s, e in [(co_ind, o_ind), (cp_ind, p_ind)]],
        [Changes3D(start=s, end=e, bond_order=2) for s, e in [(o_ind, p_ind), (cp_ind, co_ind)]]

    )]

mol, d, cg, deleting_list, forming_list = settings[ind]
# -

conds = Conditions()
rules = Rules()
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol, changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

c3d_list = Changes3DList(deleted=deleting_list, forming=forming_list, charges=[])

root = TDStructure.from_RP(temp.reactants).xtb_geom_optimization()
root_ps = root.pseudoalign(c3d_list)

target = root_ps_opt.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.gum_mm_optimization()

# +
# for r in reactions:
#     print(r)

# +
reactions = hf.pload("../../retropaths/data/reactions.p")
# rxn_name = "Claisen-Rearrangement"
# rxn_name = "Ene-Reaction-N=N"
# rxn_name = "Knoevenangel-Condensation"
rxn_name = "Ugi-Reaction"

rxn = reactions[rxn_name]
root = TDStructure.from_rxn_name(rxn_name, reactions)
c3d_list = root.get_changes_in_3d(rxn)

root = root.pseudoalign(c3d_list)
root = root.xtb_geom_optimization()

# +
target = root.copy()
target.apply_changed3d_list(c3d_list)

target.mm_optimization("gaff")

target = target.xtb_geom_optimization()
target
# -

# %%time
m = MSMEP(max_steps=2000, v=True, tol=0.01,friction=0.05,nudge=0, k=0.001)
n2, out = m.find_mep_multistep((root, target), do_alignment=False)

out.plot_chain()

t = Trajectory([n.tdstructure for n in out])

t.draw();

t.write_trajectory(f"../example_cases/{rxn_name}/msmep_tol001.xyz")

# # NEB stuff

# +
deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,6), (2,1)]]
forming_list = [Changes3D(start=s,end=e, bond_order=2) for s,e in [(1,6), (2,5)]]

c3d_list = Changes3DList(deleted=deleting_list,forming=forming_list, charges=[])

# +
root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root = root.xtb_geom_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.mm_optimization("gaff")
target = target.xtb_geom_optimization()
# -

root.molecule_rp.get_bond_changes(target.molecule_rp)

root.get_changes_in_3d()

# +
traj = Trajectory([root.xtb_geom_optimization(), target.xtb_geom_optimization()])
gi = traj.run_geodesic(nimages=15)

chain = Chain.from_traj(gi,k=0.1,delta_k=0, step_size=1,node_class=Node3D)
tol = .03
n = NEB(initial_chain=chain,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
n.optimize_chain()
# -

t = Trajectory([node.tdstructure for node in n.optimized])

n.optimized.plot_chain()

# ### now, im going to split it up into fragments

# +
c_opt = n.optimized
frag1 = Trajectory([c_opt[0].tdstructure.xtb_geom_optimization(), c_opt[3].tdstructure.xtb_geom_optimization()])
gi_frag1 = frag1.run_geodesic(nimages=15)
c_frag1 = Chain.from_traj(gi_frag1,k=0.1,delta_k=0,node_class=Node3D,step_size=1)

tol = .03
n_frag1 = NEB(initial_chain=c_frag1,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
n_frag1.optimize_chain()
# -

n_frag1.chain_trajectory[-1].plot_chain() # this is kinda dumb

# +
frag2 = Trajectory([c_opt[3].tdstructure.xtb_geom_optimization(), c_opt[7].tdstructure.xtb_geom_optimization()])
gi_frag2 = frag2.run_geodesic(nimages=15)
c_frag2 = Chain.from_traj(gi_frag2,k=0.1,delta_k=0,node_class=Node3D,step_size=1)

tol = .03
n_frag2 = NEB(initial_chain=c_frag2,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
n_frag2.optimize_chain()
# -

n_frag2.chain_trajectory[-1].plot_chain()

c_opt[7].tdstructure

mod_target = c_opt[7].tdstructure.xtb_geom_optimization().copy()
mod_target.delete_bonds(c3d_list.deleted)
mod_target.mm_optimization("gaff")
mod_target = target.xtb_geom_optimization()

# +
frag3 = Trajectory([c_opt[7].tdstructure.xtb_geom_optimization(), mod_target])
gi_frag3 = frag3.run_geodesic(nimages=15)
c_frag3 = Chain.from_traj(gi_frag3,k=0.1,delta_k=0,node_class=Node3D,step_size=1)

tol = .03
n_frag3 = NEB(initial_chain=c_frag3,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
# -

n_frag3.optimize_chain()

foo = n_frag3.chain_trajectory[-1]

foo.plot_chain()

hmm = Trajectory([node.tdstructure for node in foo])

hmm[4].xtb_geom_optimization()

hmm.write_trajectory("../example_cases/wittig/wtf.xyz")

hmm[11].xtb_geom_optimization()

corrected_end = hmm[4].copy()
corrected_end.delete_bonds(c3d_list.deleted)
corrected_end.mm_optimization("gaff")
corrected_end.mm_optimization("uff")
corrected_end = corrected_end.xtb_geom_optimization()

# +
frag3_v2 = Trajectory([hmm[4].xtb_geom_optimization(), corrected_end])
gi_frag3_v2 = frag3_v2.run_geodesic(nimages=15)
c_frag3_v2 = Chain.from_traj(gi_frag3_v2,k=0.1,delta_k=0,node_class=Node3D,step_size=1)

tol = .03
n_frag3_v2 = NEB(initial_chain=c_frag3_v2,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)
# -

n_frag3_v2.optimize_chain()

n_frag3_v2.chain_trajectory[-1].plot_chain()

c_tot = Chain(n_frag1.optimized.nodes+n_frag2.optimized.nodes+n_frag3_v2.optimized.nodes, k=0.1,step_size=1,delta_k=0)

t = Trajectory([n.tdstructure for n in c_tot])

t.write_trajectory("../example_cases/wittig/extracted_mechanism_att3.xyz")

# +
s = 8
fs = 18

f, ax = plt.subplots(figsize=(2.18*s, s))
plt.plot(c_tot.integrated_path_length, (c_tot.energies - c_tot.energies[0])*627.5, 'o--')
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
# -

# what if i try to get a TS.... for fun...
from neb_dynamics.NEB import TS_PRFO
tsopt = TS_PRFO(initial_node=c_tot[-8])

tsopt.ts

# # check out the 'true' path

mol = Molecule.from_smiles("CC(=O)C.CP(=C)(C)C")
mol.draw(mode='d3')

# +
d1 = {'charges':[],'delete':[], 'single':[(5,2),(6,1),(6,5),(2,1)]}
conds = Conditions()
rules = Rules()
cg = [
    (4,5,'Me'),
    (7,5,'Me'),
    (8,5,'Me'),
    (0,1,'Me'),
    (3,1,'Me'),
]

temp1 = ReactionTemplate.from_components(name='Wittig', reactants=mol,changes_react_to_prod_dict=d1, conditions=conds, rules=rules,collapse_groups=cg)
# temp1 = ReactionTemplate.from_components(name='Wittig', reactants=mol,changes_react_to_prod_dict=d1, conditions=conds, rules=rules)

# +
# Molecule.draw_list([temp1.reactants,temp1.products], mode='rdkit')
# -

# ### this is where making the endpoints happens

root_true = TDStructure.from_RP(temp1.reactants)

# +
forming_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,2), (6,1)]]

c3d_list = Changes3DList(deleted=[],forming=forming_list, charges=[])
# -

root_aligned = root_true.pseudoalign(c3d_list).xtb_geom_optimization()

target_aligned = root_aligned.copy()
target_aligned.add_bonds(c3d_list.forming)
target_aligned.mm_optimization('gaff')
target_aligned = target_aligned.xtb_geom_optimization()

traj_true = Trajectory([root_aligned.xtb_geom_optimization(), target_aligned.xtb_geom_optimization()])
gi_true = traj_true.run_geodesic(nimages=15, friction=.01)

chain_true = Chain.from_traj(traj=gi_true,k=0.1,delta_k=0,step_size=1,node_class=Node3D)
tol = .03
n_true = NEB(initial_chain=chain_true,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)

n_true.optimize_chain()

n_true.chain_trajectory[-1].plot_chain()

# #### continuation

target_aligned.molecule_rp.draw(mode='d3')

# +
deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,6), (2,1)]]

c3d_list = Changes3DList(deleted=deleting_list,forming=[], charges=[])
# -

final_struct = target_aligned.copy()
final_struct.delete_bonds(c3d_list.deleted)
final_struct.mm_optimization('gaff')
final_struct.mm_optimization('uff')
final_struct = final_struct.xtb_geom_optimization()

gi_true_pt2 = Trajectory([target_aligned, final_struct]).run_geodesic(nimages=15, friction=.01)

chain_true_pt2 = Chain.from_traj(traj=gi_true_pt2,k=0.1,delta_k=0,step_size=1,node_class=Node3D)
tol = .03
n_true_pt2 = NEB(initial_chain=chain_true_pt2,max_steps=1000,en_thre=tol/450, rms_grad_thre=tol*(2/3), grad_thre=tol)

n_true_pt2.optimize_chain()

nodes = n_true.chain_trajectory[-1].nodes.copy()

nodes.extend(n_true_pt2.chain_trajectory[-1].nodes.copy())

t_true = Trajectory([node.tdstructure for node in nodes])

# # Vis stuff

# c = Chain(nodes=nodes, k=0.1,delta_k=0,step_size=1)
c = Chain.from_xyz("../example_cases/wittig/true_mechanism_tol03_v2.xyz")
# c_tot = Chain.from_xyz("../example_cases/wittig/extracted_mechanism_att0.xyz")
c_tot = Chain.from_xyz("../example_cases/wittig/extracted_mechanism_att3.xyz")
c_aut = Chain.from_xyz("../example_cases/wittig/auto_extracted_att2.xyz")
c_aut2 = Chain.from_xyz("../example_cases/wittig/auto_extracted_att2_tol01.xyz")


# +
s = 8
fs = 18

f, ax = plt.subplots(figsize=(2.18*s, s))

plt.plot(c.integrated_path_length, (c.energies-c.energies[0])*627.5, 'o--', label='manual')
# plt.plot(c_tot.integrated_path_length, (c_tot.energies-c.energies[0])*627.5, 'o--',label='extracted')
plt.plot(c_aut.integrated_path_length, (c_aut.energies-c.energies[0])*627.5, 'o--',label='extracted_auto (tol 0.0045)')
plt.plot(c_aut2.integrated_path_length, (c_aut2.energies-c.energies[0])*627.5, 'o--',label='extracted_auto (tol 0.01)')
# plt.plot(c_resamp.integrated_path_length, (c_resamp.energies-c.energies[0])*627.5, 'o--',label='extracted_resamp')
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.legend(fontsize=fs)
# -

t = Trajectory([n.tdstructure for n in c_tot.nodes])

t.draw();

# +


inds_min = argrelextrema(c_tot.energies, np.less, order=2)[0]
inds_min
# -

# # Automate Shit

# +
# let's automate shit
mol = Molecule.from_smiles("CC(=O)C.CP(=C)(C)C")
# mol = Molecule.from_smiles("C(=O)(C)C.C=P(c1ccccc1)(c2ccccc2)c3ccccc3")
d = {'charges':[],'delete':[(5,6),(1,2)], 'double':[(1,6),(2,5)]}
# d = {'charges':[],'delete':[(5,4),(1,0)], 'double':[(0,4),(1,5)]}
# d = {'charges':[], 'single':[(0,4),(1,5),(5,4),(0,1)]}
conds = Conditions()
rules = Rules()
cg = [
    (7,5,'Me'),
    (8,5,'Me'),
    (4,5,'Me'),
    (0,1,'Me'),
    (3,1,'Me'),
]

# cg = [
#     (18,5,'Me'),
#     (6,5,'Me'),
#     (12,5,'Me'),
#     (2,0,'Me'),
#     (3,0,'Me'),
# ]
temp = ReactionTemplate.from_components(name='Wittig', reactants=mol,changes_react_to_prod_dict=d, conditions=conds, rules=rules, collapse_groups=cg)

# +
deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,6), (2,1)]]
forming_list = [Changes3D(start=s,end=e, bond_order=2) for s,e in [(1,6), (2,5)]]
# deleting_list = [Changes3D(start=s,end=e, bond_order=1) for s,e in [(5,4),(1,0)]]
# deleting_list = []
# forming_list = [Changes3D(start=s,end=e, bond_order=2) for s,e in [(0,4),(1,5)]]

c3d_list = Changes3DList(deleted=deleting_list,forming=forming_list, charges=[])
# -

root = TDStructure.from_RP(temp.reactants)
root = root.pseudoalign(c3d_list)
root = root.xtb_geom_optimization()

target = root.copy()
target.add_bonds(c3d_list.forming)
target.delete_bonds(c3d_list.deleted)
target.mm_optimization("gaff")
target = target.xtb_geom_optimization()

root

target

# +
# reference = Trajectory.from_xyz("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v4/traj_0-0_0_cneb.xyz")
# reference_geo = Trajectory.from_xyz("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v4/traj_0-0_0.xyz")

# +
# plt.plot((np.array(reference.energies)-reference.energies[0])*627.5, 'o-')

# +
# root, target = reference[0].copy(), reference[-1].copy()

# +
# geo_chain = Chain.from_traj(reference_geo,k=0.1,delta_k=0,step_size=2,node_class=Node3D)

# +
# tol = 4.5e-3
# chain = Chain.from_traj(traj=reference_geo, k=.1, delta_k=0.0, step_size=2, node_class=Node3D)
# n = NEB(initial_chain=chain, grad_thre=tol, en_thre=tol/450, rms_grad_thre=tol*(2/3), climb=True, vv_force_thre=0, max_steps=10000)


# +
# n.optimize_chain()

# +
# plt.plot((n.optimized.energies-n.optimized.energies[0])*627.5,'o-')

# +
# m = MSMEP(max_steps=2000, v=True, tol=0.0045,friction=0.001)
# o = m.get_neb_chain(root, target,do_alignment=False)
# n, c = o

# +
# c.plot_chain()
# -

ref2 = Chain.from_xyz("../example_cases/claisen/cr_MSMEP_tol_0045_hydrogen_fix.xyz")
ref2.plot_chain()

# +
ref = Chain.from_xyz("../example_cases/claisen/cr_MSMEP_tol_01.xyz")

ref.plot_chain()

root = ref2[0].tdstructure
target = ref2[-1].tdstructure
# -

# %%time
m = MSMEP(max_steps=2000, v=True, tol=0.0045,friction=0.1,nudge=0)
n2, out = m.find_mep_multistep((root, target), do_alignment=False)

out.plot_chain()

t = Trajectory([n.tdstructure for n in out])

t.draw();

plt.plot(out.energies,'o-',label='MSMEP')
plt.plot(reference.energies, 'x-', label='old method')
plt.legend()
plt.show()

reference[0]

t[0]

print("hey")

t.draw()

t.write_trajectory("../example_cases/wittig/auto_extracted_att2_tol01.xyz")

# +
# t.write_trajectory("../example_cases/wittig/auto_extracted_TPP_att0.xyz")
# -

#









t1 = Trajectory.from_xyz("../example_cases/wittig/auto_extracted_att2.xyz")

t2 = Trajectory.from_xyz("../example_cases/wittig/auto_extracted_TPP_att0.xyz")

t1.draw();

t2.draw();

t2[0]

t2[-1]

t3 = Trajectory.from_xyz("../example_cases/claisen/cr_MSMEP_tol_01.xyz")

t3.draw();

# +
from dataclasses import dataclass, field
from neb_dynamics.NEB import NEB

@dataclass
class TreeNode:
    data: NEB
    children: list = field(default_factory=list)
    
    
    def add_child(self, child: TreeNode):
        self.children.append(child)
@dataclass
class DataTree:
    root: TreeNode


# -

dt = DataTree(root=TreeNode(data=o))

dt.root.children

dt.root.add_child(o)

len(dt.root.children)


