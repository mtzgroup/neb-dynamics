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
from retropaths.abinitio.tdstructure import TDStructure
import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

# +
# import os
# del os.environ['OE_LICENSE']
# -

# # Could  I have split based on initial guess? 
#
# **preliminary answer**: maybe

reactions = hf.pload("../../retropaths/data/reactions.p")


directory = Path("/home/jdep/T3D_data/msmep_draft/comparisons/")

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

# +
# rns[90:]
# -

failed=['Nazarov-Cyclization']

comparisons_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons/")
ca = CompetitorAnalyzer(comparisons_dir,'asneb')


# +
# t = Trajectory.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/asneb/Aza-Grob-Fragmentation-X-Fluorine/initial_guess.xyz")
# -

def plot_rxn(name):
    s = 5
    fs = 18
    clean_fp = ca.out_folder / name / 'initial_guess_msmep_clean.xyz'
    # clean_fp = ca.out_folder / name / 'production_results' / 'initial_guess_msmep_clean.xyz'
    if clean_fp.exists():
        data_dir = clean_fp
    else:
        data_dir = ca.out_folder / name / 'initial_guess_msmep.xyz'
        # data_dir = ca.out_folder / name / 'production_results' / 'initial_guess_msmep.xyz'
    c = Chain.from_xyz(data_dir, ChainInputs(k=0))
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


reactions["Robinson-Gabriel-Synthesis

# reaction_name = 'Lobry-de-Bruyn-Van-Ekenstein-Transformation'
reaction_name = 'Robinson-Gabriel-Synthesis'
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

c.to_trajectory().draw();

actual_failed = []
actual_elem_step = []
actual_multi_step = []
for rn in rns:
    try:
        data_dir = ca.out_folder / rn / 'initial_guess_msmep.xyz'
        c = Chain.from_xyz(data_dir, ChainInputs())
        # print(rn)
        # c.plot_chain()
        if c._chain_is_concave():
            actual_elem_step.append(rn)
        else:
            actual_multi_step.append(rn)
    except:
        actual_failed.append(rn)


set_elem_rns = set(elem_rns) - set(actual_failed)
set_multi_rns = set(multi_step_rns) - set(actual_failed)

actual_multi_step

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

len(wtf)

neb_obj = NEB.read_from_disk(ca.out_folder / 'Chan-Rearrangement' / 'initial_guess_msmep' / 'node_0')

neb_obj.optimized

neb_obj.plot_opt_history(do_3d=True)


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

# +
chain_traj = neb_short.chain_trajectory
distances = [None] # None for the first chain
for i,chain in enumerate(chain_traj):
    if i == 0 :
        continue
    
    prev_chain = chain_traj[i-1]
    dist = prev_chain._distance_to_chain(chain)
    distances.append(dist)
    


fs = 18
s = 8


from kneed import KneeLocator
kn = KneeLocator(x=list(range(len(distances)))[1:], y=distances[1:], curve='convex', direction='decreasing')


f,ax = plt.subplots(figsize=(1.16*s, s))

plt.text(.65,.9, s=f"elbow: {kn.elbow}\nelbow_yval: {round(kn.elbow_y,4)}", transform=ax.transAxes,fontsize=fs)

plt.plot(distances,'o-')
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Distance to previous chain",fontsize=fs)
plt.xlabel("Chain id",fontsize=fs)

plt.show()

    

neb_short.plot_grad_delta_mag_history()
# -

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


