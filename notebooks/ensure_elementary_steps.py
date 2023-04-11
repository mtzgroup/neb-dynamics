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
from neb_dynamics.helper_functions import RMSD
from kneed import KneeLocator
from neb_dynamics.TreeNode import TreeNode
from retropaths.molecules.elements import ElementData
import warnings
warnings.filterwarnings('ignore')
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

# # Could  I have split based on initial guess? 
#
# **preliminary answer**: maybe

reactions = hf.pload("../../retropaths/data/reactions.p")


directory = Path("/home/jdep/T3D_data/msmep_draft/comparisons")

ca = CompetitorAnalyzer(comparisons_dir=directory,method='dlfind')

rns = ca.available_reaction_names

# +
# elem_rns = []
# multi_step_rns = []
# failed = []

# +
# for ind in range(len(rns)):


#     rn = rns[ind]
#     reaction_structs = ca.structures_dir / rn

#     guess_path = reaction_structs / 'initial_guess.xyz'

#     tr = Trajectory.from_xyz(guess_path)

#     nbi = NEBInputs()
#     cni = ChainInputs()
#     gii = GIInputs()
#     m = MSMEP(neb_inputs=nbi, gi_inputs=gii, chain_inputs=cni)

#     chain = Chain.from_traj(tr,parameters=cni)
#     try:
#         r,p = m._approx_irc(chain)
#         minimizing_gives_endpoints = r.is_identical(chain[0]) and p.is_identical(chain[-1])


#         if minimizing_gives_endpoints:
#             elem_rns.append(rn)
#         else:
#             multi_step_rns.append(rn)
#     except:
#         failed.append(rn)
# -

elem_rns=['Grob-Fragmentation-X-Fluorine', 'Elimination-Lg-Alkoxide', 'Elimination-Alkene-Lg-Bromine', 'Elimination-with-Hydride-Shift-Lg-Sulfonate', 'Elimination-with-Alkyl-Shift-Lg-Chlorine', 'Aza-Grob-Fragmentation-X-Bromine', 'Fries-Rearrangement-ortho', 'Elimination-Alkene-Lg-Iodine', 'Aza-Grob-Fragmentation-X-Chlorine', 'Decarboxylation-CG-Nitrite', 'Amadori-Rearrangement', 'Rupe-Rearrangement', 'Elimination-To-Form-Cyclopropanone-Sulfonate', 'Grob-Fragmentation-X-Chlorine', 'Elimination-Alkene-Lg-Sulfonate', 'Elimination-with-Alkyl-Shift-Lg-Hydroxyl', 'Oxindole-Synthesis-X-Iodine', 'Semi-Pinacol-Rearrangement-Nu-Iodine', 'Baker-Venkataraman-Rearrangement', 'Oxazole-Synthesis', 'Elimination-with-Alkyl-Shift-Lg-Iodine', 'Elimination-with-Alkyl-Shift-Lg-Bromine', 'Buchner-Ring-Expansion-O', 'Meyer-Schuster-Rearrangement', 'Chan-Rearrangement', 'Irreversable-Azo-Cope-Rearrangement', 'Claisen-Rearrangement', 'Aza-Grob-Fragmentation-X-Iodine', 'Chapman-Rearrangement', 'Overman-Rearrangement-Pt2', 'Hemi-Acetal-Degradation', 'Vinylcyclopropane-Rearrangement', 'Elimination-Amine-Imine', 'Sulfanyl-anol-Degradation', 'Cyclopropanation-Part-2', 'Oxindole-Synthesis-X-Fluorine', 'Curtius-Rearrangement', 'Oxazole-Synthesis-EWG-Nitrite-EWG3-Nitrile', 'Elimination-with-Hydride-Shift-Lg-Hydroxyl', 'Elimination-Lg-Iodine', 'Aza-Vinylcyclopropane-Rearrangement', 'Elimination-Acyl-Chlorine', 'Imine-Tautomerization-EWG-Phosphonate-EWG3-Nitrile', 'Elimination-Lg-Chlorine', 'Semi-Pinacol-Rearrangement-Nu-Chlorine', 'Aza-Grob-Fragmentation-X-Fluorine', 'Elimination-Lg-Hydroxyl', 'Aza-Grob-Fragmentation-X-Sulfonate', 'Elimination-Acyl-Iodine', 'Imine-Tautomerization-EWG-Nitrite-EWG3-Nitrile', 'Imine-Tautomerization-EWG-Carbonyl-EWG3-Nitrile', 'Elimination-Acyl-Sulfonate', 'Elimination-with-Hydride-Shift-Lg-Iodine', 'Elimination-Alkene-Lg-Chlorine', 'Indole-Synthesis-Hemetsberger-Knittel', 'Semi-Pinacol-Rearrangement-Nu-Sulfonate', 'Thiocarbamate-Resonance', 'Elimination-with-Hydride-Shift-Lg-Chlorine', 'Meisenheimer-Rearrangement', 'Imine-Tautomerization-EWG-Carboxyl-EWG3-Nitrile', 'Mumm-Rearrangement', 'Bradsher-Cyclization-2', 'Claisen-Rearrangement-Aromatic', 'Elimination-To-Form-Cyclopropanone-Bromine', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Cl', '2-Sulfanyl-anol-Degradation', 'Meisenheimer-Rearrangement-Conjugated', 'Elimination-with-Hydride-Shift-Lg-Bromine', 'Bradsher-Cyclization-1', 'Azaindole-Synthesis', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Br', 'Decarboxylation-CG-Carboxyl', 'Benzimidazolone-Synthesis-1-X-Bromine', 'Elimination-To-Form-Cyclopropanone-Iodine', 'Benzimidazolone-Synthesis-1-X-Iodine', '1-2-Amide-Phthalamide-Synthesis', 'Elimination-Acyl-Bromine', 'Elimination-Lg-Sulfonate', 'Decarboxylation-Carbamic-Acid', 'Oxa-Vinylcyclopropane-Rearrangement', 'Grob-Fragmentation-X-Iodine', 'Imine-Tautomerization-EWG-Nitrile-EWG3-Nitrile', 'Grob-Fragmentation-X-Bromine', 'Elimination-To-Form-Cyclopropanone-Chlorine', 'Enolate-Claisen-Rearrangement', 'Elimination-with-Alkyl-Shift-Lg-Sulfonate', 'Petasis-Ferrier-Rearrangement', 'Buchner-Ring-Expansion-C', 'Semi-Pinacol-Rearrangement-Alkene', 'Decarboxylation-CG-Carbonyl', 'Semi-Pinacol-Rearrangement-Nu-Bromine', 'Robinson-Gabriel-Synthesis', 'Newman-Kwart-Rearrangement', 'Azo-Vinylcyclopropane-Rearrangement', 'Elimination-Lg-Bromine', 'Bamberger-Rearrangement', 'Lobry-de-Bruyn-Van-Ekenstein-Transformation', 'Oxindole-Synthesis-X-Bromine', 'Electrocyclic-Ring-Opening', 'Ester-Pyrolysis', 'Lossen-Rearrangement', 'Pinacol-Rearrangement']

multi_step_rns=['Semmler-Wolff-Reaction', 'Ramberg-Backlund-Reaction-Bromine', 'Oxazole-Synthesis-EWG-Nitrile-EWG3-Nitrile', 'Indole-Synthesis-1', 'Grob-Fragmentation-X-Sulfonate', 'Oxazole-Synthesis-EWG-Carbonyl-EWG3-Nitrile', 'Oxazole-Synthesis-EWG-Alkane-EWG3-Nitrile', 'Fries-Rearrangement-para', 'Ramberg-Backlund-Reaction-Iodine', 'Paal-Knorr-Furan-Synthesis', 'Oxindole-Synthesis-X-Chlorine', 'Ramberg-Backlund-Reaction-Chlorine', 'Camps-Quinoline-Synthesis', 'Oxazole-Synthesis-EWG-Carboxyl-EWG3-Nitrile', 'Oxy-Cope-Rearrangement', 'Beckmann-Rearrangement', 'Bamford-Stevens-Reaction', 'Ramberg-Backlund-Reaction-Fluorine', 'Oxazole-Synthesis-EWG-Phosphonate-EWG3-Nitrile', 'Madelung-Indole-Synthesis', 'Thio-Claisen-Rearrangement', 'Buchner-Ring-Expansion-N', 'Knorr-Quinoline-Synthesis', 'Piancatelli-Rearrangement', 'Elimination-Water-Imine', 'Skraup-Quinoline-Synthesis']

failed=['Nazarov-Cyclization']

# reaction_structs = ca.structures_dir / 'Thio-Claisen-Rearrangement'
reaction_structs = ca.structures_dir / 'Imine-Tautomerization-EWG-Carbonyl-EWG3-Nitrile'
guess_path = reaction_structs / 'initial_guess.xyz'

tr = Trajectory.from_xyz(guess_path)

nbi = NEBInputs(v=True,early_stop_chain_rms_thre=0.002,tol=0.00045)
cni = ChainInputs()
gii = GIInputs()
m = MSMEP(neb_inputs=nbi, gi_inputs=gii, chain_inputs=cni)

c = Chain.from_traj(tr,parameters=cni)

history, out_chain = m.find_mep_multistep(c)

out_chain.plot_chain()

 nbi_complete = nbi = NEBInputs(v=True,early_stop_chain_rms_thre=0.0,tol=0.00045)
neb_complete = NEB(initial_chain=c,parameters=nbi_complete)

neb_complete.optimize_chain()


# # Let's get error bars for an optimization

# +
def get_mass(symbol):
    ED = ElementData()
    return ED.from_symbol(symbol).mass_amu

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

history = TreeNode.read_from_disk(Path("./wittig_early_stop/"))

neb_short = NEB.read_from_disk(Path("./neb_short"))

neb_short.plot_opt_history(do_3d=True)

neb_short.plot_grad_delta_mag_history()

neb_short.plot_projector_history()


def get_projector(chain1,chain2, var = 'gradients'):
    if var == 'tangents':
        chain1_vec = chain1.unit_tangents  
        chain2_vec = chain2   .unit_tangents
    
    elif var == 'gradients':
    
        chain1_vec = chain1.gradients 
        chain2_vec = chain2.gradients 
        
    else:
        raise ValueError(f"Incorrect input method: {var}")
    
    # projector = sum([np.dot(t1.flatten(),t2.flatten()) for t1,t2  in zip(chain1_vec, chain2_vec)]) / len(chain1_vec)
    projector = sum([np.dot(t1.flatten(),t2.flatten()) for t1,t2  in zip(chain1_vec, chain2_vec)]) 
    return projector


var = 'gradients'
traj = [ ]
for ind in range(1,296):
    proj = get_projector(neb_short.chain_trajectory[ind - 1], neb_short.chain_trajectory[ind], var=var) 
    normalization = get_projector(neb_short.chain_trajectory[ind - 1], neb_short.chain_trajectory[ind - 1], var=var)
    traj.append(proj / normalization)

list(enumerate(traj))

plt.plot(traj)
plt.ylim(0,1.1)

# +
chain1_tangentsst_per_opt_step = []
# end_cost_per_opt_step = []

# running_knee_dist = []

# running_std_dist = []
# running_std_ene = []



# for end_chain in range(1, len(neb_complete.chain_trajectory)):
#     chain1 = neb_complete.chain_trajectory[end_chain-1]
#     chain2 = neb_complete.chain_trajectory[end_chain]

#     distances = []
#     en_diffs = []

#     for node1,node2 in zip(chain1.nodes, chain2.nodes):
#         dist,_ = RMSD(node1.coords, node2.coords)
#         distances.append(dist)

#         en_diff = node2.energy - node1.energy
#         en_diffs.append(abs(en_diff))

#     # print(distances)
#     # dist_cost_per_opt_step.append(sum(distances)/len(chain1))
#     dist_cost_per_opt_step.append(chain1._distance_to_chain(chain2))
#     end_cost_per_opt_step.append(sum(en_diffs)/len(chain1))
    
    
    
#     running_std_dist.append(np.std(dist_cost_per_opt_step))
#     running_std_ene.append(np.std(end_cost_per_opt_step))


# -


def get_elbow(list_of_vals):
    kn = KneeLocator(list(range(len(list_of_vals))), list_of_vals, curve='convex',direction='decreasing',S=1,online=True)
    return kn.elbow


list_of_vals = dist_cost_per_opt_step
kn = KneeLocator(list(range(len(list_of_vals))), list_of_vals, curve='convex',direction='decreasing',S=1,online=True)
kn.elbow_y

# +
start = 0
end = -1


xvals = list(range(len(dist_cost_per_opt_step[start:end])))

plt.plot(xvals, dist_cost_per_opt_step[start:end],'o-', label='distances change')
# plt.plot(xvals, end_cost_per_opt_step[start:end],'o-',label='energies change')
axes = plt.gca()

elbow = get_elbow(dist_cost_per_opt_step)
# elbow = get_elbow(end_cost_per_opt_step)
print(elbow)
miny,maxy = axes.get_ylim()
# plt.vlines(x=26,ymin=miny,ymax=maxy, linestyle='--',color='gray', label='force thre')
# plt.vlines(x=33,ymin=miny,ymax=maxy, linestyle='--',color='purple', label='chain rms thre')
plt.vlines(x=elbow,ymin=miny,ymax=maxy, linestyle='--',color='green', label='elbow energy')




plt.legend()
plt.show()
# -


