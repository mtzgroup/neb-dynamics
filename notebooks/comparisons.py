# +
from pathlib import Path
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
import numpy as np
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.Node2d import Node2D_Flower, Node2D
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Node3D import Node3D

from neb_dynamics.Node3D_gfn1xtb import Node3D_gfn1xtb
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS

from neb_dynamics.TreeNode import TreeNode

from itertools import product
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from neb_dynamics.CompetitorAnalyzer import CompetitorAnalyzer

import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
# import os
# del os.environ['OE_LICENSE']

from neb_dynamics.Janitor import Janitor
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from neb_dynamics.MSMEP import MSMEP
from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

NIMAGES = 15


# # Helper Functions

def get_eA(chain_energies):
    return max(chain_energies) - chain_energies[0]


# +
def plot_chain(chain,linestyle='--',ax=None, marker='o',**kwds):
    if ax:
        ax.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)
    else:
        plt.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)

        
def plot_neb(neb,linestyle='--',marker='o',ax=None,**kwds):
    plot_chain(chain=neb.chain_trajectory[-1],linestyle='-',marker=marker,ax=ax,**kwds)
    plot_chain(chain=neb.initial_chain,linestyle='--',marker=marker,ax=ax,**kwds)


# +
def plot_chain2d(chain,linestyle='--',ax=None, marker='o',**kwds):
    if ax:
        ax.plot(chain.integrated_path_length,chain.energies,linestyle=linestyle,marker=marker,**kwds)
    else:
        plt.plot(chain.integrated_path_length,chain.energies,linestyle=linestyle,marker=marker,**kwds)

        
def plot_neb2d(neb,linestyle='--',marker='o',ax=None,**kwds):
    plot_chain2d(chain=neb.chain_trajectory[-1],linestyle='-',marker=marker,ax=ax,**kwds)
    plot_chain2d(chain=neb.initial_chain,linestyle='--',marker=marker,ax=ax,**kwds)


# -

# # 2D potentials

# +
ind = 0

the_noise = [-1,1]

noises_bool = [
    True,
    False

]




start_points = [
     [-2.59807434, -1.499999  ],
    [-3.77931026, -3.283186  ]
]

end_points = [
    [2.5980755 , 1.49999912],
    [2.99999996, 1.99999999]

]
tols = [
    0.1,
    0.05,

]

step_sizes = [
    1,
    1
]


k_values = [
    1,#.05,
    50

]



nodes = [Node2D_Flower, Node2D]
node_to_use = nodes[ind]
start_point = start_points[ind]
end_point = end_points[ind]
tol = tols[ind]

ss = step_sizes[ind]
ks = k_values[ind]
do_noise = noises_bool[ind]

# +
nimages = NIMAGES
np.random.seed(0)



coords = np.linspace(start_point, end_point, nimages)
if do_noise:
    coords[1:-1] += the_noise # i.e. good initial guess

    
cni_ref = ChainInputs(
    k=ks,
    node_class=node_to_use,
    delta_k=0,
    step_size=ss,
    # step_size=.01,
    do_parallel=False,
    use_geodesic_interpolation=False,
    min_step_size=.001
)
gii = GIInputs(nimages=nimages)
nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=0, node_freezing=False, 
               vv_force_thre=0)
chain_ref = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni_ref)
# -

n_ref = NEB(initial_chain=chain_ref,parameters=nbi)
n_ref.optimize_chain()

gii = GIInputs(nimages=nimages)
nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_chain_rms_thre=0.02, early_stop_force_thre=1, node_freezing=False)
# nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=3, node_freezing=False)
m = MSMEP(neb_inputs=nbi_msmep,chain_inputs=cni_ref, gi_inputs=gii)
history, out_chain = m.find_mep_multistep(chain_ref)

obj = n_ref
distances = obj._calculate_chain_distances()
forces = [c.get_maximum_grad_magnitude() for c in obj.chain_trajectory]

# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)

ax.plot(distances, 'o-',color='blue', label='chain distances')
plt.xticks(fontsize=fs)

plt.yticks(fontsize=fs)

ax2 = ax.twinx()
ax2.plot(forces, 'o-',color='orange',label='max(|∇$_{\perp}$|)')
plt.yticks(fontsize=fs)
ax.set_ylabel("Distance between chains",fontsize=fs)
ax2.set_ylabel("Maximum gradient component absolute value",fontsize=fs)
f.legend(fontsize=fs, loc='upper left')

# +
nimages_long = len(out_chain)

coords_long = np.linspace(start_point, end_point, nimages_long)
if do_noise:
    coords_long[1:-1] += the_noise # i.e. good initial guess
gii = GIInputs(nimages=nimages_long)
nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, early_stop_force_thre=0, node_freezing=False)
cni_ref2 = ChainInputs(
    k=1,
    node_class=node_to_use,
    delta_k=0,
    step_size=1,
    do_parallel=False,
    use_geodesic_interpolation=False,
    min_step_size=0.001
)
# chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref)
chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref2)

n_ref_long = NEB(initial_chain=chain_ref_long,parameters=nbi)
n_ref_long.optimize_chain()

# +
#### get energies for countourplot
gridsize = 100
min_val = -4
max_val = 4
# min_val = -.05
# max_val = .05

x = np.linspace(start=min_val, stop=max_val, num=gridsize)
y = x.reshape(-1, 1)
# -

h_flat_ref = np.array([node_to_use.en_func_arr(pair) for pair in product(x,x)])
h_ref = h_flat_ref.reshape(gridsize,gridsize).T

# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
# x = np.linspace(start=min_val, stop=max_val, num=1000)
# y = x.reshape(-1, 1)

cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
# cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

plot_chain(n_ref.initial_chain, c='orange',label='initial guess')
plot_chain(n_ref.chain_trajectory[-1], c='green',linestyle='-',label=f'NEB({nimages} nodes)')
plot_chain(n_ref_long.chain_trajectory[-1], c='gold',linestyle='-',label=f'NEB({nimages_long} nodes)', marker='*', ms=12)
plot_chain(out_chain, c='red',marker='o',linestyle='-',label='AS-NEB')

plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/results_2D_potential_ind{ind}.svg")
plt.show()

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
# ax.set_facecolor("lightgray")

plot_chain2d(n_ref.initial_chain, c='orange',label='initial guess')
plot_chain2d(n_ref.chain_trajectory[-1], c='green',linestyle='-',label=f'neb({nimages} nodes)')
plot_chain2d(out_chain, c='red',marker='o',linestyle='-',label='as-neb')
plot_chain2d(n_ref_long.chain_trajectory[-1], c='silver',linestyle='-',label=f'neb({nimages_long} nodes)')
plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# +
n_steps_orig_neb = len(n_ref.chain_trajectory)-1
n_steps_msmep = sum([len(obj.chain_trajectory) for obj in history.get_optimization_history()])-1 
n_steps_long_neb = len(n_ref_long.chain_trajectory)-1

n_grad_orig_neb = n_steps_orig_neb*(NIMAGES-2)
n_grad_msmep = n_steps_msmep*(NIMAGES-2)
n_grad_long_neb = n_steps_long_neb*(nimages_long-2)

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f,ax = plt.subplots(figsize=(1.16*fig,fig))
# plt.bar(x=["AS-NEB",f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)'],
#        height=[n_steps_msmep, n_steps_orig_neb, n_steps_long_neb])

bars = ax.bar(x=[f'NEB({nimages} nodes)',f'NEB({nimages_long} nodes)',"AS-NEB",],
       height=[n_grad_orig_neb, n_grad_long_neb, n_grad_msmep])

ax.bar_label(bars,fontsize=fs)


plt.yticks(fontsize=fs)
# plt.ylabel("Number of optimization steps",fontsize=fs)
plt.text(.63,.95, f"{round((n_grad_long_neb / n_grad_msmep), 2)}x improvement",transform=ax.transAxes,fontsize=fs,
        bbox={'visible':True,'fill':False})
plt.ylabel("Number of gradient calls",fontsize=fs)

plt.xticks(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/results_2D_potential_ind{ind}_barplot.svg")
plt.show()
# -

# # Visualize Big data

ca = CompetitorAnalyzer(comparisons_dir=Path("/home/jdep/T3D_data/msmep_draft/comparisons/"),method='asneb')

# +
# ca.submit_all_jobs()
# -

rns = ca.available_reaction_names

# +
rn = rns[0]

def get_relevant_chain(rn):
    data_dir = ca.out_folder / rn
    clean_chain = data_dir / 'initial_guess_msmep_clean.xyz'
    msmep_chain = data_dir / 'initial_guess_msmep.xyz'

    if clean_chain.exists():
        chain_to_use = Chain.from_xyz(clean_chain, ChainInputs())
    elif not clean_chain.exists() and msmep_chain.exists():
        chain_to_use = Chain.from_xyz(msmep_chain,ChainInputs())
    else: # somehow the thing failed
        print(f"{rn} failed.")
        chain_to_use = None

    return chain_to_use

def get_relevant_tree(rn):
    data_dir = ca.out_folder / rn
    fp = data_dir / 'initial_guess_msmep'
    tree = TreeNode.read_from_disk(fp)
    return tree

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
# ChainInputs(k=0.1, delta_k=0.09)
# -

def plot_hist(data_list, label, **kwargs):
    fs = 18
    plt.hist([x[1] for x in data_list], **kwargs)
    plt.ylabel("Count",fontsize=fs)
    plt.xlabel(label,fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.show()


def get_eA_chain(chain):
    eA = max(chain.energies_kcalmol)
    return eA


def get_eA_leaf(leaf):
    chain = leaf.data.optimized
    eA = max(chain.energies_kcalmol)
    return eA


def descending_order(data_list):
    return sorted(data_list, key=lambda x: x[1], reverse=True)


# +
from openbabel import pybel

ob_log_handler = pybel.ob.OBMessageHandler()
pybel.ob.obErrorLog.StopLogging()


# +
all_max_barriers = []
all_n_steps = []
peak_barriers = []
all_n_atoms = []



tol = 0.001*BOHR_TO_ANGSTROMS

succ=['Semmler-Wolff-Reaction', 'Grob-Fragmentation-X-Fluorine', 'Elimination-Lg-Alkoxide', 'Elimination-Alkene-Lg-Bromine', 'Elimination-with-Alkyl-Shift-Lg-Chlorine', 'Aza-Grob-Fragmentation-X-Bromine', 'Ramberg-Backlund-Reaction-Bromine', 'Elimination-Alkene-Lg-Iodine', 'Aza-Grob-Fragmentation-X-Chlorine', 'Decarboxylation-CG-Nitrite', 'Amadori-Rearrangement', 'Rupe-Rearrangement', 'Grob-Fragmentation-X-Chlorine', 'Elimination-Alkene-Lg-Sulfonate', 'Elimination-with-Alkyl-Shift-Lg-Hydroxyl', 'Semi-Pinacol-Rearrangement-Nu-Iodine', 'Grob-Fragmentation-X-Sulfonate', 'Oxazole-Synthesis-EWG-Carbonyl-EWG3-Nitrile', 'Oxazole-Synthesis', 'Fries-Rearrangement-para', 'Buchner-Ring-Expansion-O', 'Chan-Rearrangement', 'Irreversable-Azo-Cope-Rearrangement', 'Claisen-Rearrangement', 'Paal-Knorr-Furan-Synthesis', 'Chapman-Rearrangement', 'Ramberg-Backlund-Reaction-Chlorine', 'Overman-Rearrangement-Pt2', 'Hemi-Acetal-Degradation', 'Vinylcyclopropane-Rearrangement', 'Sulfanyl-anol-Degradation', 'Cyclopropanation-Part-2', 'Oxindole-Synthesis-X-Fluorine', 'Curtius-Rearrangement', 'Oxazole-Synthesis-EWG-Nitrite-EWG3-Nitrile', 'Elimination-Lg-Iodine', 'Aza-Vinylcyclopropane-Rearrangement', 'Elimination-Acyl-Chlorine', 'Imine-Tautomerization-EWG-Phosphonate-EWG3-Nitrile', 'Elimination-Lg-Chlorine', 'Semi-Pinacol-Rearrangement-Nu-Chlorine', 'Elimination-Lg-Hydroxyl', 'Aza-Grob-Fragmentation-X-Sulfonate', 'Elimination-Acyl-Iodine', 'Imine-Tautomerization-EWG-Nitrite-EWG3-Nitrile', 'Imine-Tautomerization-EWG-Carbonyl-EWG3-Nitrile', 'Elimination-Acyl-Sulfonate', 'Elimination-with-Hydride-Shift-Lg-Iodine', 'Elimination-Alkene-Lg-Chlorine', 'Semi-Pinacol-Rearrangement-Nu-Sulfonate', 'Thiocarbamate-Resonance', 'Elimination-with-Hydride-Shift-Lg-Chlorine', 'Meisenheimer-Rearrangement', 'Imine-Tautomerization-EWG-Carboxyl-EWG3-Nitrile', 'Mumm-Rearrangement', 'Claisen-Rearrangement-Aromatic', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Cl', '2-Sulfanyl-anol-Degradation', 'Meisenheimer-Rearrangement-Conjugated', 'Elimination-with-Hydride-Shift-Lg-Bromine', 'Azaindole-Synthesis', 'Oxy-Cope-Rearrangement', 'Beckmann-Rearrangement', 'Fritsch-Buttenberg-Wiechell-Rearrangement-Br', 'Decarboxylation-CG-Carboxyl', 'Benzimidazolone-Synthesis-1-X-Bromine', 'Benzimidazolone-Synthesis-1-X-Iodine', 'Ramberg-Backlund-Reaction-Fluorine', 'Elimination-Acyl-Bromine', 'Oxazole-Synthesis-EWG-Phosphonate-EWG3-Nitrile', 'Decarboxylation-Carbamic-Acid', 'Grob-Fragmentation-X-Iodine', 'Imine-Tautomerization-EWG-Nitrile-EWG3-Nitrile', 'Grob-Fragmentation-X-Bromine', 'Elimination-To-Form-Cyclopropanone-Chlorine', 'Enolate-Claisen-Rearrangement', 'Elimination-with-Alkyl-Shift-Lg-Sulfonate', 'Petasis-Ferrier-Rearrangement', 'Buchner-Ring-Expansion-C', 'Madelung-Indole-Synthesis', 'Thio-Claisen-Rearrangement', 'Semi-Pinacol-Rearrangement-Alkene', 'Decarboxylation-CG-Carbonyl', 'Semi-Pinacol-Rearrangement-Nu-Bromine', 'Robinson-Gabriel-Synthesis', 'Newman-Kwart-Rearrangement', 'Azo-Vinylcyclopropane-Rearrangement', 'Buchner-Ring-Expansion-N', 'Elimination-Lg-Bromine', 'Lobry-de-Bruyn-Van-Ekenstein-Transformation', 'Oxindole-Synthesis-X-Bromine', 'Electrocyclic-Ring-Opening', 'Ester-Pyrolysis', 'Knorr-Quinoline-Synthesis', 'Lossen-Rearrangement', 'Pinacol-Rearrangement', 'Piancatelli-Rearrangement', 'Elimination-Water-Imine', 'Skraup-Quinoline-Synthesis', 'Wittig']#[]
failed=['Elimination-with-Hydride-Shift-Lg-Sulfonate', 'Fries-Rearrangement-ortho', 'Oxazole-Synthesis-EWG-Nitrile-EWG3-Nitrile', 'Indole-Synthesis-1', 'Elimination-To-Form-Cyclopropanone-Sulfonate', 'Oxindole-Synthesis-X-Iodine', 'Nazarov-Cyclization', 'Baker-Venkataraman-Rearrangement', 'Elimination-with-Alkyl-Shift-Lg-Iodine', 'Elimination-with-Alkyl-Shift-Lg-Bromine', 'Oxazole-Synthesis-EWG-Alkane-EWG3-Nitrile', 'Meyer-Schuster-Rearrangement', 'Ramberg-Backlund-Reaction-Iodine', 'Aza-Grob-Fragmentation-X-Iodine', 'Oxindole-Synthesis-X-Chlorine', 'Elimination-Amine-Imine', 'Camps-Quinoline-Synthesis', 'Oxazole-Synthesis-EWG-Carboxyl-EWG3-Nitrile', 'Elimination-with-Hydride-Shift-Lg-Hydroxyl', 'Aza-Grob-Fragmentation-X-Fluorine', 'Indole-Synthesis-Hemetsberger-Knittel', 'Bradsher-Cyclization-2', 'Elimination-To-Form-Cyclopropanone-Bromine', 'Bradsher-Cyclization-1', 'Elimination-To-Form-Cyclopropanone-Iodine', 'Bamford-Stevens-Reaction', '1-2-Amide-Phthalamide-Synthesis', 'Elimination-Lg-Sulfonate', 'Oxa-Vinylcyclopropane-Rearrangement', 'Bamberger-Rearrangement', 'Wittig_DFT'] #[]
# for i, rn in enumerate(rns):
for i, rn in enumerate(succ):
    # try:
    cs = get_relevant_leaves(rn)
    # if all([x.get_maximum_grad_magnitude() <= tol for x in cs]):
    eAs = [get_eA_chain(c) for c in cs]
    
    c = Chain.from_list_of_chains(cs, ChainInputs())
    max_delta_en = (max(c.energies) - c.energies[0])*627.5
    
    
    maximum_barrier = max(eAs)
    peak_barrier = max_delta_en
    n_steps = len(eAs)
    n_atoms = c.n_atoms

    all_max_barriers.append((i, maximum_barrier))
    peak_barriers.append((i, peak_barrier))
    all_n_steps.append((i, n_steps))
    all_n_atoms.append((i, n_atoms))
    
        # succ.append(rn)
    # else:
    #     print(f"{rn} has not converged")
        # failed.append(rn)
            
    # except:
        # failed.append(rn)
# -

descending_order(all_n_steps)

plot_hist(all_n_atoms, "Number atoms", bins=20)

plot_hist(peak_barriers, "Peak E$_A$")

plot_hist(all_n_steps, "Number steps rxn", bins=30)

descending_order(peak_barriers)

# look_at_me = rns[19]
look_at_me = succ[19]
look_at_me

cs = get_relevant_leaves(look_at_me)

c = Chain.from_list_of_chains(cs,ChainInputs())

c.plot_chain()

cs[ind].plot_chain()

ind = 2
print(get_eA_chain(cs[ind]))
cs[ind].to_trajectory()

# # Stats

# +
# N.B.: The -1 comes from the fact that Wittig_DFT is still included in the dataset for whatever reason. It should not be. 


print(f"Tot N reactions: {len(rns)-1}")
print(f"\tTot N converged: {len(succ)}")
print(f"\tTot N unconverged: {len(failed)-1}")
print(f"Convergence percentage: {round(len(succ) / (len(rns)-1), 3)*100}%")


# -

def get_error_message(rn):
    data_dir = ca.out_folder / rn
    out_fp = data_dir / 'out.txt'
    datum = open(out_fp).read().splitlines()
    return datum


# +
reasons = []
wtfs = []

for f in failed[:-1]: # ignoring 'Wittig_DFT'
    datum = get_error_message(f)
    # print(datum[-1])
    if 'AttributeError' in datum[-1] or "TypeError: 'NoneType'" in datum[-1]:
        reasons.append("Insufficient\nOptimization Steps")
    elif 'step' in datum[-1]:
        reasons.append("Insufficient\nTime")
    elif 'scf' in datum[-1]:
        reasons.append("Electronic\nStructure Error")
    else:
        reasons.append("Small bug\nto be fixed\nasap")
        wtfs.append(f)
# -

f, ax = plt.subplots()
plt.hist(reasons)

# # Repro NEB-TS

ind = 101
r = TDStructure.from_xyz(f"/home/jdep/T3D_data/nebts_repro/configurations/system{ind}-react.xyz")
p = TDStructure.from_xyz(f"/home/jdep/T3D_data/nebts_repro/configurations/system{ind}-prod.xyz")
sp = TDStructure.from_xyz(f"/home/jdep/T3D_data/nebts_repro/configurations/system{ind}-sp.xyz")

tr = Trajectory([r,p]).run_geodesic(nimages=10)

# +
tolerance = 0.01


cni = ChainInputs(k=0.1, delta_k=0.09,step_size=3, min_step_size=0.33)
nbi = NEBInputs(v=True, grad_thre=tolerance*BOHR_TO_ANGSTROMS, rms_grad_thre=1, en_thre=1)
c = Chain.from_traj(tr, parameters=cni)
# -

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=10), skip_identical_graphs=False)

h, out = m.find_mep_multistep(c)


