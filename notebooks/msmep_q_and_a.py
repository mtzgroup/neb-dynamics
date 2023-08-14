# +
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from IPython.core.display import HTML
from neb_dynamics.MSMEP import MSMEP
import retropaths.helper_functions as hf
from neb_dynamics.CompetitorAnalyzer import CompetitorAnalyzer
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.Inputs import NEBInputs, GIInputs, ChainInputs
from neb_dynamics.Chain import Chain
from neb_dynamics.NEB import NEB
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.helper_functions import RMSD
from kneed import KneeLocator
from neb_dynamics.TreeNode import TreeNode
from retropaths.molecules.elements import ElementData
from retropaths.abinitio.tdstructure import TDStructure
import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

# +
from openbabel import pybel

ob_log_handler = pybel.ob.OBMessageHandler()
pybel.ob.obErrorLog.StopLogging()

# +
# import os
# del os.environ['OE_LICENSE']
# -

# # Can I seed ab initio runs with XTB?

neb_xtb = NEB.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen/node_3.xyz"))

neb_dft_opt = NEB.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen_forcethre_0/node_3_neb.xyz"))

neb_dft_ft_003 = NEB.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen_forcethre_0,003/seeding_xtb_neb.xyz"))

neb_dft_ft_03 = NEB.read_from_disk(Path("/home/jdep/T3D_data/msmep_draft/seeding_experiments/claisen_forcethre_0,03/seeding_xtb_neb.xyz"))

# +
# neb_dft_ft_03.optimized.is_elem_step()
# -

neb_xtb.initial_chain[0].is_identical(neb_xtb.optimized[0])

xvals = ['xtb', 'dft - es03', 'dft - es003', 'dft - opt']
yvals = [len(neb_xtb.chain_trajectory), len(neb_dft_ft_03.chain_trajectory), len(neb_dft_ft_003.chain_trajectory), len(neb_dft_opt.chain_trajectory)]

plt.bar(x=xvals, height=yvals)



def set_tc_args(traj, method, basis, kwds={'maxiter':10000}):
    for td in traj:
        td.tc_model_basis = basis
        td.tc_model_method = method


m = 'wb97xd3'
b = 'def2-svp'

# +
xtb_ens = neb_xtb.optimized.to_trajectory().energies_xtb()



dft_init_tr = neb_dft_opt.initial_chain.to_trajectory()
set_tc_args(dft_init_tr, m, b)
dft_init_ens = dft_init_tr.energies_tc()
# dft_init_ens = [td.energy_tc() for td in dft_init_tr]
# -

dft_opt_tr = neb_dft_opt.optimized.to_trajectory()
set_tc_args(dft_opt_tr, 'wb97xd3', 'def2-svp')
dft_opt_ens = dft_opt_tr.energies_tc()
# dft_opt_ens = [td.energy_tc() for td in dft_opt_tr]

dft_ft_03_tr = neb_dft_ft_03.optimized.to_trajectory()
set_tc_args(dft_ft_03_tr, 'wb97xd3', 'def2-svp')
dft_ft_03_ens = dft_ft_03_tr.energies_tc()
# dft_ft_03_ens = [td.energy_tc() for td in dft_ft_03_tr]

dft_ft_003_tr = neb_dft_ft_003.optimized.to_trajectory()
set_tc_args(dft_ft_003_tr, 'wb97xd3', 'def2-svp')
dft_ft_003_ens = dft_ft_003_tr.energies_tc()
# dft_ft_003_ens = [td.energy_tc() for td in dft_ft_003_tr]

# +



plt.plot(neb_xtb.optimized.integrated_path_length, xtb_ens,'o--', label='xtb')
plt.plot(neb_dft_opt.optimized.integrated_path_length, dft_opt_ens,'-', linewidth=6, label='dft - optseed')
# plt.plot(neb_dft_opt.initial_chain.integrated_path_length, dft_init_ens,'o-', label='dft - init')
plt.plot(neb_dft_ft_03.optimized.integrated_path_length, dft_ft_03_ens,'o-', label=' dft - es03 seed')
plt.plot(neb_dft_ft_003.optimized.integrated_path_length, dft_ft_003_ens,'x-', label=' dft - es003 seed')

plt.legend()
# -

neb_dft_opt.initial_chain.energies

plt.plot([  0.        ,   1.32675863,   4.50511477,   8.71484519,
        14.44812173,  23.41113873,  32.27156787,  38.79849834,
        26.621906  ,   1.91414245, -18.4693448 , -19.7537819 ])

tr.energies_tc()

neb_dft_opt.initial_chain[1].energy

len(neb_dft_ft_003.optimized)

len(neb_dft_ft_003.optimized)



r_raw.energy_tc() - neb_dft_ft_003.optimized[0].energy

r_raw = neb_dft_ft_003.optimized[1].tdstructure
p_raw = neb_dft_ft_003.optimized[-2].tdstructure

method = 'wb97xd3'
basis = 'def2-svp'

r_raw.tc_model_method = method
r_raw.tc_model_basis = basis

p_raw.tc_model_method = method
p_raw.tc_model_basis = basis

r_opt = r_raw.tc_geom_optimization()

p_opt = p_raw.tc_geom_optimization()

neb_dft_ft_003_fixed = neb_dft_ft_003.optimized.copy()

neb_dft_ft_003_fixed.nodes[0] = Node3D_TC(r_opt)

neb_dft_ft_003_fixed.nodes[-1] = Node3D_TC(p_opt)

Node3D_TC(r_opt).is_identical(neb_dft_ft_003.optimized[0])

neb_dft_ft_003_fxed.plot_chain()

neb_dft_ft_003_fixed[0]._cached_energy

r_opt.energy_tc()

p_opt.energy_tc()

reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")

nbi = NEBInputs(v=True, grad_thre=0.001,rms_grad_thre=0.001,max_steps=2000, early_stop_force_thre=0.003,early_stop_still_steps_thre=500)
cni = ChainInputs(k=0.1, delta_k=0.09,step_size=3)
gii = GIInputs(nimages=10)
m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii)

# +
# r,p = m.create_endpoints_from_rxn_name('Claisen-Rearrangement',reactions)

# +
# tr = Trajectory([r, p])
# chain = Chain.from_traj(tr,parameters=cni)

# +
# h, out = m.find_mep_multistep(chain)
# -

h = TreeNode.read_from_disk(Path("./claisen"),nbi, cni)

out = h.output_chains

out.get_maximum_gperp()

out_leaf = [l for l in h.ordered_leaves if l.data]


xtb_chain = out_leaf[0].data.optimized
init_chain = out_leaf[0].data.initial_chain

# +
method = 'wb97xd3'
basis = 'def2-svp'

xtb_tr = xtb_chain.to_trajectory()
for td in xtb_tr:
    td.tc_model_basis = basis
    td.tc_model_method = method
# -

dft_start = xtb_tr[0].tc_geom_optimization()

dft_end = xtb_tr[-1].tc_geom_optimization()

dft_tr = Trajectory([dft_start]+xtb_tr.traj[1:-1]+[dft_end]) # same as the other chain but without the endpoints
dft_tr2= Trajectory([dft_start, dft_end]).run_geodesic(nimages=10)

nbi_dft = NEBInputs(v=True, grad_thre=0.001,rms_grad_thre=0.001,max_steps=2000, early_stop_force_thre=0.003,early_stop_still_steps_thre=500)
cni_dft = ChainInputs(k=0.1, delta_k=0.09,step_size=3, node_class=Node3D_TC)
gii_dft = GIInputs(nimages=10)

dft_chain = Chain.from_traj(dft_tr, parameters=cni_dft)

m_dft = MSMEP(neb_inputs=nbi_dft, chain_inputs=cni_dft, gi_inputs=gii_dft)

h_dft, out_dft = m_dft.find_mep_multistep(dft_chain)

# ** XTB seems to have biased the path to some channel that is far away from the DFT endpoints

dft_chain2 = Chain.from_traj(dft_tr2, parameters=cni_dft)

h_dft2, out_dft2 = m_dft.find_mep_multistep(dft_chain2)

# # Could  I have split based on initial guess? 
#
# **preliminary answer**: maybe

reactions = hf.pload("../../retropaths/data/reactions.p")


directory = Path("/home/jdep/T3D_data/msmep_draft/comparisons/")

ca = CompetitorAnalyzer(comparisons_dir=directory,method='dlfind')

rns = ca.available_reaction_names

from neb_dynamics.constants import BOHR_TO_ANGSTROMS


def reaction_converged(fp):
    c = Chain.from_xyz(fp, ChainInputs(k=0.1, delta_k=0.09))
    return c.get_maximum_grad_magnitude() <= 0.001*BOHR_TO_ANGSTROMS


elem_rns = []
multi_step_rns = []
failed = []


def get_eA_chain(chain):
    eA = max(chain.energies_kcalmol)
    return eA


def get_relevant_leaves(rn):
    data_dir = ca.out_folder / rn
    fp = data_dir / 'initial_guess_msmep'
    adj_mat_fp = fp / 'adj_matrix.txt'
    adj_mat = np.loadtxt(adj_mat_fp)
    if adj_mat.size == 1:
        return [Chain.from_xyz(fp / f'node_0.xyz', ChainInputs(k=0.1, delta_k=0.09))]
    else:
    
        a = np.sum(adj_mat,axis=1)
        inds_leaves = np.where(a == 1)[0] 
        chains = [Chain.from_xyz(fp / f'node_{ind}.xyz',ChainInputs(k=0.1, delta_k=0.09)) for ind in inds_leaves]
        return chains


# +

tol = 0.001*BOHR_TO_ANGSTROMS

succ = []
failed = []
for i, rn in enumerate(rns):
    try:
        cs = get_relevant_leaves(rn)
        if all([x.get_maximum_grad_magnitude() <= tol for x in cs]):
            eAs = [get_eA_chain(c) for c in cs]
            maximum_barrier = max(eAs)
            n_steps = len(eAs)

            succ.append(rn)
        else:
            print(f"{rn} has not converged")
            failed.append(rn)

    except:
        failed.append(rn)

# +
# for ind in range(len(rns)):


#     rn = rns[ind]
#     print(f"Doing: {rn}")
#     reaction_structs = ca.structures_dir / rn

#     guess_path = reaction_structs / 'initial_guess.xyz'

#     tr = Trajectory.from_xyz(guess_path)

#     nbi = NEBInputs()
#     cni = ChainInputs()
#     gii = GIInputs()
#     m = MSMEP(neb_inputs=nbi, gi_inputs=gii, chain_inputs=cni)

#     chain = Chain.from_traj(tr,parameters=cni)
#     try:
#         is_elem_step_bool, method = chain.is_elem_step()


#         if is_elem_step_bool:
#             elem_rns.append(rn)
#         else:
#             multi_step_rns.append(rn)
#     except:
#         failed.append(rn)BOHR_TO_ANGSTROMS
# -
succ=['Semmler-Wolff-Reaction', 'Grob-Fragmentation-X-Fluorine', 'Elimination-Lg-Alkoxide', 'Elimination-Alkene-Lg-Bromine', 'Elimination-with-Alkyl-Shift-Lg-Chlorine', 'Aza-Grob-Fragmentation-X-Bromine', 'Ramberg-Backlund-Reaction-Bromine', 'Elimination-Alkene-Lg-Iodine', 'Aza-Grob-Fragmentation-X-Chlorine', 'Decarboxylation-CG-Nitrite', 'Amadori-Rearrangement', 'Rupe-Rearrangement', 'Grob-Fragmentation-X-Chlorine', 'Elimination-Alkene-Lg-Sulfonate', 'Elimination-with-Alkyl-Shift-Lg-Hydroxyl', 'Semi-Pinacol-Rearrangement-Nu-Iodine', 'Grob-Fragmentation-X-Sulfonate', 'Oxazole-Synthesis-EWG-Carbonyl-EWG3-Nitrile', 'Oxazole-Synthesis', 'Fries-Rearrangement-para', 'Buchner-Ring-Expansion-O', 'Chan-Rearrangement', 'Irreversable-Azo-Cope-Rearrangement', 'Claisen-Rearrangement', 'Paal-Knorr-Furan-Synthesis', 'Chapman-Rearrangement', 'Ramberg-Backlund-Reaction-Chlorine', 'Overman-Rearrangement-Pt2', 'Hemi-Acetal-Degradation', 'Vinylcyclopropane-Rearrangement', 'Sulfanyl-anol-Degradation', 'Cyclopropanation-Part-2', 'Oxindole-Synthesis-X-Fluorine', 'Curtius-Rearrangement', 'Oxazole-Synthesis-EWG-Nitrite-EWG3-Nitrile', 'Elimination-Lg-Iodine', 'Aza-Vinylcyclopropane-Rearrangement', 'Elimination-Acyl-Chlorine', 'Imine-Tautomerization-EWG-Phosphonate-EWG3-Nitrile', 'Elimination-Lg-Chlorine', 'Semi-Pinacol-Rearrangement-Nu-Chlorine', 'Elimination-Lg-Hydroxyl', 'Aza-Grob-Fragmentation-X-Sulfonate', 'Elimination-Acyl-Iodine', 'Imine-Tautomerization-EWG-Nitrite-EWG3-Nitrile', 'Imine-Tautomerization-EWG-Carbonyl-EWG3-Nitrile', 'Elimination-Acyl-Sulfonate', 'Elimination-with-Hydride-Shift-Lg-Iodine', 'Elimination-Alkene-Lg-Chlorine', 'Semi-Pinacol-Rearrangement-Nu-Sulfonate', 'Thiocarbamate-Resonance', 'Elimination-with-Hydride-Shift-Lg-Chlorine', 'Meisenheimer-Rearrangement', 'Imine-Tautomerization-EWG-Carboxyl-EWG3-Nitrile', 'Mumm-Rearrangement', 'Claisen-Rearrangement-Aromatic', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Cl', '2-Sulfanyl-anol-Degradation', 'Meisenheimer-Rearrangement-Conjugated', 'Elimination-with-Hydride-Shift-Lg-Bromine', 'Azaindole-Synthesis', 'Oxy-Cope-Rearrangement', 'Beckmann-Rearrangement', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Br', 'Decarboxylation-CG-Carboxyl', 'Benzimidazolone-Synthesis-1-X-Bromine', 'Benzimidazolone-Synthesis-1-X-Iodine', 'Ramberg-Backlund-Reaction-Fluorine', 'Elimination-Acyl-Bromine', 'Oxazole-Synthesis-EWG-Phosphonate-EWG3-Nitrile', 'Decarboxylation-Carbamic-Acid', 'Grob-Fragmentation-X-Iodine', 'Imine-Tautomerization-EWG-Nitrile-EWG3-Nitrile', 'Grob-Fragmentation-X-Bromine', 'Elimination-To-Form-Cyclopropanone-Chlorine', 'Enolate-Claisen-Rearrangement', 'Elimination-with-Alkyl-Shift-Lg-Sulfonate', 'Petasis-Ferrier-Rearrangement', 'Buchner-Ring-Expansion-C', 'Madelung-Indole-Synthesis', 'Thio-Claisen-Rearrangement', 'Semi-Pinacol-Rearrangement-Alkene', 'Decarboxylation-CG-Carbonyl', 'Semi-Pinacol-Rearrangement-Nu-Bromine', 'Robinson-Gabriel-Synthesis', 'Newman-Kwart-Rearrangement', 'Azo-Vinylcyclopropane-Rearrangement', 'Buchner-Ring-Expansion-N', 'Elimination-Lg-Bromine', 'Lobry-de-Bruyn-Van-Ekenstein-Transformation', 'Oxindole-Synthesis-X-Bromine', 'Electrocyclic-Ring-Opening', 'Ester-Pyrolysis', 'Knorr-Quinoline-Synthesis', 'Lossen-Rearrangement', 'Pinacol-Rearrangement', 'Piancatelli-Rearrangement', 'Elimination-Water-Imine', 'Skraup-Quinoline-Synthesis', 'Wittig']#[]
failed=['Elimination-with-Hydride-Shift-Lg-Sulfonate', 'Fries-Rearrangement-ortho', 'Oxazole-Synthesis-EWG-Nitrile-EWG3-Nitrile', 'Indole-Synthesis-1', 'Elimination-To-Form-Cyclopropanone-Sulfonate', 'Oxindole-Synthesis-X-Iodine', 'Nazarov-Cyclization', 'Baker-Venkataraman-Rearrangement', 'Elimination-with-Alkyl-Shift-Lg-Iodine', 'Elimination-with-Alkyl-Shift-Lg-Bromine', 'Oxazole-Synthesis-EWG-Alkane-EWG3-Nitrile', 'Meyer-Schuster-Rearrangement', 'Ramberg-Backlund-Reaction-Iodine', 'Aza-Grob-Fragmentation-X-Iodine', 'Oxindole-Synthesis-X-Chlorine', 'Elimination-Amine-Imine', 'Camps-Quinoline-Synthesis', 'Oxazole-Synthesis-EWG-Carboxyl-EWG3-Nitrile', 'Elimination-with-Hydride-Shift-Lg-Hydroxyl', 'Aza-Grob-Fragmentation-X-Fluorine', 'Indole-Synthesis-Hemetsberger-Knittel', 'Bradsher-Cyclization-2', 'Elimination-To-Form-Cyclopropanone-Bromine', 'Bradsher-Cyclization-1', 'Elimination-To-Form-Cyclopropanone-Iodine', 'Bamford-Stevens-Reaction', '1-2-Amide-Phthalamide-Synthesis', 'Elimination-Lg-Sulfonate', 'Oxa-Vinylcyclopropane-Rearrangement', 'Bamberger-Rearrangement', 'Wittig_DFT'] #[]

elem_rns = ['Semmler-Wolff-Reaction', 'Grob-Fragmentation-X-Fluorine', 'Elimination-Lg-Alkoxide', 'Elimination-Alkene-Lg-Bromine', 'Elimination-with-Alkyl-Shift-Lg-Chlorine', 'Aza-Grob-Fragmentation-X-Bromine', 'Fries-Rearrangement-ortho', 'Elimination-Alkene-Lg-Iodine', 'Grob-Fragmentation-X-Chlorine', 'Elimination-Alkene-Lg-Sulfonate', 'Elimination-with-Alkyl-Shift-Lg-Hydroxyl', 'Semi-Pinacol-Rearrangement-Nu-Iodine', 'Grob-Fragmentation-X-Sulfonate', 'Elimination-with-Alkyl-Shift-Lg-Iodine', 'Elimination-with-Alkyl-Shift-Lg-Bromine', 'Buchner-Ring-Expansion-O', 'Meyer-Schuster-Rearrangement', 'Chan-Rearrangement', 'Aza-Grob-Fragmentation-X-Iodine', 'Chapman-Rearrangement', 'Curtius-Rearrangement', 'Oxazole-Synthesis-EWG-Carboxyl-EWG3-Nitrile', 'Elimination-Lg-Iodine', 'Aza-Vinylcyclopropane-Rearrangement', 'Elimination-Lg-Chlorine', 'Semi-Pinacol-Rearrangement-Nu-Chlorine', 'Aza-Grob-Fragmentation-X-Fluorine', 'Elimination-Acyl-Iodine', 'Imine-Tautomerization-EWG-Nitrite-EWG3-Nitrile', 'Elimination-with-Hydride-Shift-Lg-Iodine', 'Elimination-Alkene-Lg-Chlorine', 'Thiocarbamate-Resonance', 'Elimination-with-Hydride-Shift-Lg-Chlorine', 'Meisenheimer-Rearrangement', 'Claisen-Rearrangement-Aromatic', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Cl', 'Elimination-with-Hydride-Shift-Lg-Bromine', 'Oxy-Cope-Rearrangement', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Br', 'Benzimidazolone-Synthesis-1-X-Bromine', 'Imine-Tautomerization-EWG-Nitrile-EWG3-Nitrile', 'Enolate-Claisen-Rearrangement', 'Buchner-Ring-Expansion-C', 'Thio-Claisen-Rearrangement', 'Semi-Pinacol-Rearrangement-Nu-Bromine', 'Newman-Kwart-Rearrangement', 'Elimination-Lg-Bromine', 'Lobry-de-Bruyn-Van-Ekenstein-Transformation', 'Oxindole-Synthesis-X-Bromine', 'Electrocyclic-Ring-Opening', 'Lossen-Rearrangement', 'Pinacol-Rearrangement']
elem_rns = [x for x in elem_rns if x in succ]

multi_step_rns = ['Elimination-with-Hydride-Shift-Lg-Sulfonate', 'Ramberg-Backlund-Reaction-Bromine', 'Oxazole-Synthesis-EWG-Nitrile-EWG3-Nitrile', 'Aza-Grob-Fragmentation-X-Chlorine', 'Decarboxylation-CG-Nitrite', 'Amadori-Rearrangement', 'Rupe-Rearrangement', 'Indole-Synthesis-1', 'Elimination-To-Form-Cyclopropanone-Sulfonate', 'Oxindole-Synthesis-X-Iodine', 'Nazarov-Cyclization', 'Baker-Venkataraman-Rearrangement', 'Oxazole-Synthesis-EWG-Carbonyl-EWG3-Nitrile', 'Oxazole-Synthesis', 'Oxazole-Synthesis-EWG-Alkane-EWG3-Nitrile', 'Fries-Rearrangement-para', 'Irreversable-Azo-Cope-Rearrangement', 'Ramberg-Backlund-Reaction-Iodine', 'Claisen-Rearrangement', 'Paal-Knorr-Furan-Synthesis', 'Oxindole-Synthesis-X-Chlorine', 'Ramberg-Backlund-Reaction-Chlorine', 'Overman-Rearrangement-Pt2', 'Hemi-Acetal-Degradation', 'Vinylcyclopropane-Rearrangement', 'Elimination-Amine-Imine', 'Sulfanyl-anol-Degradation', 'Cyclopropanation-Part-2', 'Camps-Quinoline-Synthesis', 'Oxindole-Synthesis-X-Fluorine', 'Oxazole-Synthesis-EWG-Nitrite-EWG3-Nitrile', 'Elimination-with-Hydride-Shift-Lg-Hydroxyl', 'Elimination-Acyl-Chlorine', 'Imine-Tautomerization-EWG-Phosphonate-EWG3-Nitrile', 'Elimination-Lg-Hydroxyl', 'Aza-Grob-Fragmentation-X-Sulfonate', 'Imine-Tautomerization-EWG-Carbonyl-EWG3-Nitrile', 'Elimination-Acyl-Sulfonate', 'Indole-Synthesis-Hemetsberger-Knittel', 'Semi-Pinacol-Rearrangement-Nu-Sulfonate', 'Imine-Tautomerization-EWG-Carboxyl-EWG3-Nitrile', 'Mumm-Rearrangement', 'Bradsher-Cyclization-2', 'Elimination-To-Form-Cyclopropanone-Bromine', '2-Sulfanyl-anol-Degradation', 'Meisenheimer-Rearrangement-Conjugated', 'Bradsher-Cyclization-1', 'Azaindole-Synthesis', 'Beckmann-Rearrangement', 'Decarboxylation-CG-Carboxyl', 'Elimination-To-Form-Cyclopropanone-Iodine', 'Bamford-Stevens-Reaction', 'Benzimidazolone-Synthesis-1-X-Iodine', '1-2-Amide-Phthalamide-Synthesis', 'Ramberg-Backlund-Reaction-Fluorine', 'Elimination-Acyl-Bromine', 'Elimination-Lg-Sulfonate', 'Oxazole-Synthesis-EWG-Phosphonate-EWG3-Nitrile', 'Decarboxylation-Carbamic-Acid', 'Oxa-Vinylcyclopropane-Rearrangement', 'Grob-Fragmentation-X-Iodine', 'Grob-Fragmentation-X-Bromine', 'Elimination-To-Form-Cyclopropanone-Chlorine', 'Elimination-with-Alkyl-Shift-Lg-Sulfonate', 'Petasis-Ferrier-Rearrangement', 'Madelung-Indole-Synthesis', 'Semi-Pinacol-Rearrangement-Alkene', 'Decarboxylation-CG-Carbonyl', 'Robinson-Gabriel-Synthesis', 'Azo-Vinylcyclopropane-Rearrangement', 'Buchner-Ring-Expansion-N', 'Ester-Pyrolysis', 'Knorr-Quinoline-Synthesis', 'Piancatelli-Rearrangement', 'Elimination-Water-Imine', 'Skraup-Quinoline-Synthesis', 'Wittig', 'Wittig_DFT']
multi_step_rns = [x for x in multi_step_rns if x in succ]

# +
# failed=['Bamberger-Rearrangement']
# -

comparisons_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons/")
ca = CompetitorAnalyzer(comparisons_dir,'asneb')


# +
# t = Trajectory.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Aza-Grob-Fragmentation-X-Fluorine/initial_guess.xyz")
# -

def plot_rxn(name):
    s = 5
    fs = 18
    # clean_fp = ca.out_folder / name / 'initial_guess_msmep_clean.xyz'
    # # clean_fp = ca.out_folder / name / 'production_results' / 'initial_guess_msmep_clean.xyz'
    # if clean_fp.exists():
    #     data_dir = clean_fp
    # else:
    #     data_dir = ca.out_folder / name / 'initial_guess_msmep.xyz'
    #     # data_dir = ca.out_folder / name / 'production_results' / 'initial_guess_msmep.xyz'
    cs = get_relevant_leaves(name)
    c = Chain.from_list_of_chains(cs, ChainInputs())
    f, ax = plt.subplots(figsize=(1.618*s, s))
    
    plt.plot(c.integrated_path_length, (c.energies-c.energies[0])*627.5,'o-')
    plt.yticks(fontsize=fs)
    plt.ylabel("Energy (kcal/mol)",fontsize=fs)
    plt.xlabel("Normalized path length",fontsize=fs)
    plt.xticks(fontsize=fs)
    # plt.text(.02, .9, f"grad_max: {round(c.get_maximum_grad_magnitude(),5)}", transform=ax.transAxes, fontsize=fs)
    plt.title(name, fontsize=fs)
    plt.show()
    return data_dir, c


dd, c = plot_rxn(elem_rns[1])

# reaction_name = 'Lobry-de-Bruyn-Van-Ekenstein-Transformation'
# reaction_name = 'Robinson-Gabriel-Synthesis'
reaction_name = 'Semmler-Wolff-Reaction'
dd, c = plot_rxn(reaction_name)

# +
s=5
fs=18
f, ax = plt.subplots(figsize=(2.16*s, s))

ms = 7
lw = 2
hexvals=[
    '#ff1b1b',
'#dd0090',
'#5855c3']

out = c
out_pathlen = out.integrated_path_length
plt.plot(out.integrated_path_length, (out.energies-out.energies[0])*627.5-0,'o-', label="AS-NEB"
        ,markersize=ms,linewidth=lw, color='green')


plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.xlabel("Normalized path length",fontsize=fs)

# plt.legend(fontsize=fs)


plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{reaction_name}_energy_paths.svg")


plt.show()
# -

a = np.loadtxt(dd.parent / 'initial_guess_msmep' / 'adj_matrix.txt')

a

get_n_leaves(a)


def get_n_leaves(adj_mat):
    n_leaves=0
    for row in adj_mat:
        if len(row.nonzero()[0]) == 1:
            n_leaves+=1
    return n_leaves


get_n_leaves(a)

actual_failed = []
actual_elem_step = []
actual_multi_step = []
# for rn in rns:
for rn in succ:
    # try:
    data_dir = ca.out_folder / rn / 'initial_guess_msmep' / 'adj_matrix.txt'
    adj = np.loadtxt(data_dir)
    if not adj.shape:
        actual_elem_step.append(rn)
    else:
        # if get_n_leaves(adj)==1:
        #     actual_elem_step.append(rn)
        # else:
        actual_multi_step.append(rn)
    # except:
    #     actual_failed.append(rn)


# +
# # dd, c = plot_rxn(actual_multi_step[4])
# for i in range(len(actual_multi_step)):
#     dd, c = plot_rxn(actual_multi_step[i])
# -

set_elem_rns = set(elem_rns) - set(actual_failed)
set_multi_rns = set(multi_step_rns) - set(actual_failed)

overlap_elem_steps = set(actual_elem_step).intersection(set_elem_rns)

mispred_elem_steps = set_elem_rns - set(actual_elem_step)

overlap_multi = set(actual_multi_step).intersection(set_multi_rns)

mispred_multi_steps = set_multi_rns - set(actual_multi_step) 

print(f"N predicted multistep: {len(set_multi_rns)}")
print(f"N actual multistep: {len(actual_multi_step)}")
print(f"\tN True Positives: {len(overlap_multi)}")
print(f"\tN False Positives: {len(mispred_multi_steps)}")

print(f"N predicted elemstep: {len(set_elem_rns)}")
print(f"N actual elemstep: {len(actual_elem_step)}")
print(f"\tN True Positives: {len(overlap_elem_steps)}")
print(f"\tN False Positives: {len(mispred_elem_steps)}")


# # How often do transient minima appear ? 

def has_transient_minima(neb_obj):
    for chain in neb_obj.chain_trajectory:
        if not chain._chain_is_concave():
            return True
    return False


count = 0
examples = []
wtf = []
for elem_rn in actual_elem_step:
    try:
        neb_obj = NEB.read_from_disk(ca.out_folder / elem_rn / 'initial_guess_msmep' / 'node_0')
        if has_transient_minima(neb_obj):
            examples.append(elem_rn)
            count+=1
    except:
        wtf.append(elem_rn)

examples[1]

neb_obj.optimized.is_elem_step()

neb_obj = NEB.read_from_disk(ca.out_folder / 'Elimination-with-Hydride-Shift-Lg-Sulfonate' / 'initial_guess_msmep' / 'node_0')

neb_obj.plot_opt_history(do_3d=True)


