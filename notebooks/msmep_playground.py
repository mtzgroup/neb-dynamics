# ### refinement


# # Plots

# ### Initial Guess Comparison (ASNEB v NEB)


# Lobry-de-Bruyn-Van-Ekenstein-Transformation
# Meyer Schuster
# Madelung Indole
# Halogenation-Amide-w-Thionyl-Chloride
# Imidoyl-Chloride-Formation


from neb_dynamics.TreeNode import TreeNode
from neb_dynamics import StructureNode, Chain
import neb_dynamics.chainhelpers as ch
from pathlib import Path
from neb_dynamics.neb import NEB
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from qcio import Structure, view
import numpy as np


def grep(file_path, search_term):
    import re

    matches = []
    with open(file_path, 'r') as f:
        # This mimics 'grep' by printing matching lines
        for line in f:
            if re.search(search_term, line):
                matches.append(line.strip())
    return matches


# +
rns= ["Lobry-de-Bruyn-Van-Ekenstein-Transformation", "Meyer-Schuster-Rearrangement", 
     "Madelung-Indole-Synthesis", "Halogenation-Amide-w-Thionyl-Chloride", "Imidoyl-Chloride-Formation"]

# rns= ["Alcohol-Bromination", 
#       "Bohlmann-Rahtz-Pyridine-Synthesis-EWG-Phosphonate", 
#       "Darzens-Condensation-Ketone-X-Sulfonate", 
#       "Elimination-Alcohol-POCl3", 
#       "Fries-Rearrangement-ortho",
#       "Halogenation-Amide-w-Thionyl-Chloride",
#       "Imidoyl-Chloride-Formation",
#       "Imine-Tautomerization-EWG-Nitrile-EWG3-Nitrile",
#       "Isocyanate-Carbonyl-Addition-Nu-Hydroxyl",
#       "Knoevenagel-Condensation-EWG1-Carbonyl-and-EWG2-Nitrile-EWG3-Nitrile", 
#       "Knorr-Quinoline-Synthesis",
#       "Lobry-de-Bruyn-Van-Ekenstein-Transformation",
#       "Madelung-Indole-Synthesis",

#       "Nucleophilic-Acyl-Substitution-Acid-Halide-Lg-Bromine-and-Nu-Amino",
#       "Meyer-Schuster-Rearrangement"
#      ]

# +
# fp = Path(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/tsg_folder/{rns[0]}_tsg_0_optim.xyz")
fp = Path(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/tsg_folder/{rns[0]}_tsg_2.tmp/irc_irc.xyz")
node_info = Structure.from_xyz_multi(open(fp).read())
enes = [float(s.extras['xyz_comments'][-1]) for s in node_info]
node_list = [StructureNode(structure=s) for s in node_info]
for ene, node in zip(enes, node_list):
    node._cached_energy = ene

c = Chain.model_validate({"nodes":node_list})
# -

h.output_chain[-1].structure.save("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Lobry-de-Bruyn-Van-Ekenstein-Transformation/indices_swap/end_opt.xyz")

h2 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Lobry-de-Bruyn-Van-Ekenstein-Transformation/indices_swap/debug")

a = Structure.open("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Lobry-de-Bruyn-Van-Ekenstein-Transformation/indices_swap/start_opt.xyz")
# b = Structure.open("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Lobry-de-Bruyn-Van-Ekenstein-Transformation/indices_swap/end_opt.xyz")
b = Structure.open("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Lobry-de-Bruyn-Van-Ekenstein-Transformation/indices_swap/end_opt_swap.xyz")

# view.view(h2.output_chain[0].structure, h2.output_chain[-1].structure, show_indices=True)
view.view(a,b, show_indices=True)
# view.view(h2.data.initial_chain[0].structure, h2.data.initial_chain[-1].structure, show_indices=True)

ri = RunInputs()

tsres = ri.engine._compute_ts_result(h2.output_chain.get_ts_node())

view.view(tsres)

from neb_dynamics.helper_functions import compute_irc_chain

irc = compute_irc_chain(ts_node = StructureNode(structure=tsres.return_result), engine=ri.engine)

ch.visualize_chain(irc)

# ch.visualize_chain(c)
ch.visualize_chain(h2.output_chain)

costs = []
costs2 = []
for rn in rns:
    h =  TreeNode.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rn}/ASNEB_climb_DFTv2")
    nebgc = int(grep(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rn}/out_ASNEB_climb_DFTv2", ">>>")[0].split()[2])
    geomgc = int(grep(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rn}/out_ASNEB_climb_DFTv2", "<<<")[0].split()[2])
    costs.append((nebgc, geomgc, len(h.ordered_leaves)))

    neblonggc = int(grep(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/neblong_logs/{rn}.out", ">>>")[0].split()[2])
    costs2.append((neblonggc, 0, 1))


costs

costs2

# +
import matplotlib.pyplot as plt
import numpy as np

asneb = costs
neb = costs2

# Extract components
asneb_bottom = [x[0] for x in asneb]
asneb_top = [x[1] for x in asneb]

neb_bottom = [x[0] for x in neb]
neb_top = [x[1] for x in neb]

x = np.arange(len(asneb))
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))

# Bars for asneb
rects1_bot = ax.bar(x - width/2, asneb_bottom, width, label='ASNEB NEB gradcalls', color='skyblue')
rects1_top = ax.bar(x - width/2, asneb_top, width, bottom=asneb_bottom, label='Geom opt gradcalls', color='steelblue')

# Bars for neb
rects2_bot = ax.bar(x + width/2, neb_bottom, width, label='NEB gradcalls', color='lightcoral')

# Formatting
ax.set_xticks(x)
ax.set_xticklabels([f'Rxn {i+1}' for i in range(len(asneb))], fontsize=fs)
plt.yticks(fontsize=fs)
ax.legend(fontsize=fs)
ax.set_ylim(0, 15000)

plt.tight_layout()

# +
import matplotlib.pyplot as plt
import numpy as np

fs = 12
# Data
asneb = costs
neb = costs2

# Calculate medians
asneb_m1 = np.median([x[0] for x in asneb])
asneb_m2 = np.median([x[1] for x in asneb])

neb_m1 = np.median([x[0] for x in neb])
neb_m2 = np.median([x[1] for x in neb])

# Plotting
labels = ['asneb', 'neb']
v1 = [asneb_m1, neb_m1]
v2 = [asneb_m2, neb_m2]

fig, ax = plt.subplots(figsize=(5, 6))

ax.bar(labels, v1, label='NEB gradient calls', color='tab:blue', width=0.5)
ax.bar(labels, v2, bottom=v1, label='Geom. opt. gradient calls', color='tab:orange', width=0.5)

ax.set_ylim(0,8000)
ax.set_ylabel('Median cost',fontsize=fs)
ax.legend(fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.tight_layout()
# plt.savefig('median_stacked_barplot.png')

# +
import matplotlib.pyplot as plt
import numpy as np

# Data
asneb = costs
neb = costs2

# Extract only Part 1 (P1) components
asneb_p1 = [x[0] for x in asneb]
neb_p1 = [x[0] for x in neb]

data_to_plot = [asneb_p1, neb_p1]
labels = ['asneb', 'neb']

# Create the plot
fig, ax = plt.subplots(figsize=(3, 2))

# Plot the boxplots
# 'zorder=1' ensures boxes are in the back, 'showfliers=False' to avoid double-plotting outliers if any
bp = ax.boxplot(data_to_plot, labels=labels, showfliers=False, zorder=1)

# Overlay individual data points (jittered)
for i, data in enumerate(data_to_plot):
    # i+1 is the x-position for the boxplot
    # Add some random jitter to the x-axis so points don't overlap perfectly
    x = np.random.normal(i + 1, 0.04, size=len(data))
    ax.scatter(x, data, alpha=0.7, color='crimson', edgecolor='black', zorder=2, label='Data Points' if i == 0 else "")

ax.set_ylabel('NEB grad calls', fontsize=fs)

ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

# Add a small legend for the points if desired
# ax.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=8, markeredgecolor='black')], ['Data Points'])

plt.tight_layout()


# +
import matplotlib.pyplot as plt
import numpy as np

# Data
asneb = costs
neb = costs2

# Extracting the "cost" (first value)
asneb_costs = np.array([x[0] for x in asneb])
neb_costs = np.array([x[0] for x in neb])

# Calculate Delta (neb - asneb)
# deltas = neb_costs - asneb_costs
deltas = asneb_costs - neb_costs

# Groups
groups = [f'Rxn {i+1}' for i in range(len(deltas))]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))

# Colors based on positive or negative delta
colors = ['lightcoral' if d < 0 else 'skyblue' for d in deltas]

bars = ax.bar(groups, deltas, color=colors, edgecolor='black', alpha=0.8)

# Add a horizontal line at zero
ax.axhline(0, color='black', linewidth=0.8)

# Adding labels and title
ax.set_ylabel('Delta Cost (ASNEB - NEB)', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

# Labeling bars with their values
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 0 if height > 0 else -1),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()
# plt.savefig('delta_cost_barplot.png')

# +
# costs = [ # ngradcalls NEB, ngradcall geomOpt
#     (2626, 241),
#     (3796, 128),
#     (2067, 209),
#     (2535, 606),
#     (2730, 262)
# ]


# -

np.median([c[0]/c[2] for c in costs]), np.median([c[0] for c in costs]),  np.median([c[0]+c[1] for c in costs])

np.median([c[0]/c[2] for c in costs2]), np.median([c[0] for c in costs2]),  np.median([c[0]+c[1] for c in costs2])

np.median([v[0] for v in costs])
np.median([v[0]+v[1] for v in costs])

# +
# h2 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Meyer-Schuster-Rearrangement/solvation/mep_output_msmep/")

# +
# view.view(h2.output_chain[0].structure, show_indices=True)

# +
# ch.visualize_chain(h2.data.initial_chain)
# -

rxnind = 0
print(rns[rxnind])
h = TreeNode.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rns[rxnind]}/ASNEB_climb_DFTv2")

# +

h2 = TreeNode.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Lobry-de-Bruyn-Van-Ekenstein-Transformation/indices_swap/debug")
# -

# ch.visualize_chain(h.output_chain)
ch.visualize_chain(h2.output_chain)

# +
n = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rns[rxnind]}/neblong_results.xyz")

nebgc = int(grep(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rns[rxnind]}/out_ASNEB_climb_DFTv2", ">>>")[0].split()[2])
geomgc = int(grep(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rns[rxnind]}/out_ASNEB_climb_DFTv2", "<<<")[0].split()[2])
neblonggc = int(grep(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/neblong_logs/{rns[rxnind]}.out", ">>>")[0].split()[2])

# +
s=5
f, ax = plt.subplots(ncols=2, figsize=(1.16*s,s))
fs = 12

enes = h.ordered_leaves[0].data.optimized.energies_kcalmol
ax[0].plot(h.ordered_leaves[0].data.initial_chain.integrated_path_length*splitloc, h.ordered_leaves[0].data.initial_chain.energies_kcalmol, 'o--', color='tab:blue', alpha=initalpha)
ax[0].plot(h.ordered_leaves[0].data.optimized.integrated_path_length*splitloc, enes, 'o-', color='tab:blue', alpha=1)

ax[0].plot((h.ordered_leaves[1].data.initial_chain.integrated_path_length*(1-splitloc))+splitloc, h.ordered_leaves[1].data.initial_chain.energies_kcalmol+enes[-1], 'o--', color='tab:orange', alpha=initalpha)
ax[0].plot((h.ordered_leaves[1].data.optimized.integrated_path_length*(1-splitloc))+splitloc, h.ordered_leaves[1].data.optimized.energies_kcalmol+enes[-1], 'o-', color='tab:orange', alpha=1)

ax[0].plot(n.optimized.integrated_path_length, n.optimized.energies_kcalmol, 's-', color='tab:gray', alpha=1)
ax[0].plot(n.initial_chain.integrated_path_length, n.initial_chain.energies_kcalmol, 's--', color='tab:gray', alpha=initalpha)

ea_init = h.data.initial_chain.get_eA_chain()

plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
ax[0].set_ylabel("Relative energies (kcal/mol)", fontsize=fs)
ax[0].set_xlabel("Reaction progress", fontsize=fs)

ax[1].bar(x=['NEB','ASNEB'], height=[neblonggc, nebgc], label='NEB calls')
ax[1].bar(x=['NEB','ASNEB'], height=[0, geomgc], bottom=[neblonggc, nebgc], label='Geometry opt. calls')
ax[1].set_ylim(0, 15000)
ax[1].set_ylabel("Cost (gradient calls)", fontsize=fs)

plt.tight_layout()
plt.legend()
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rns[rxnind]}_chains.svg")
plt.show()
# ax.plot(h.ordered_leaves[0].data.optimized.integrated_path_length*.5, h.data.initial_chain.energies_kcalmol, 'o-', color='tab:orange', alpha=1)

# +
f, ax = plt.subplots(nrows=2)

initalpha = .3
# splitlocs = [.628, .53, .48, .5, .5]
splitlocs = [.628, .53, .4, .5, .5]
splitloc = splitlocs[rxnind]

# ax[0].plot(h.data.initial_chain.integrated_path_length, h.data.initial_chain.energies_kcalmol, 'x--', color='gray', alpha=initalpha, label='initial chain0')
# ax[0].plot(h.data.optimized.integrated_path_length, h.data.optimized.energies_kcalmol, 'x-', color='gray', alpha=.6, label='final chain0')
ax[0].plot(n.initial_chain.integrated_path_length, n.initial_chain.energies_kcalmol, 'x--', color='gray', alpha=initalpha, label='initial chain0')
ax[0].plot(n.optimized.integrated_path_length, n.optimized.energies_kcalmol, 'x-', color='gray', alpha=.6, label='final chain0')
enes = h.ordered_leaves[0].data.optimized.energies_kcalmol
enesglob = h.output_chain.energies_kcalmol
ax[1].plot(h.ordered_leaves[0].data.initial_chain.integrated_path_length*splitloc, h.ordered_leaves[0].data.initial_chain.energies_kcalmol, 'o-', color='tab:blue', alpha=initalpha)
ax[1].plot(h.ordered_leaves[0].data.optimized.integrated_path_length*splitloc, enes, 'o-', color='tab:blue', alpha=1)

ax[1].plot((h.ordered_leaves[1].data.initial_chain.integrated_path_length*(1-splitloc))+splitloc, h.ordered_leaves[1].data.initial_chain.energies_kcalmol+enes[-1], 'o-', color='tab:orange', alpha=initalpha)
ax[1].plot((h.ordered_leaves[1].data.optimized.integrated_path_length*(1-splitloc))+splitloc, h.ordered_leaves[1].data.optimized.energies_kcalmol+enes[-1], 'o-', color='tab:orange', alpha=1)

ea_init = h.data.initial_chain.get_eA_chain()

ax[0].vlines(x=splitloc, ymin=-200, ymax=ea_init*ea_init, color='gray', linestyle='--', alpha=.2)
ax[1].vlines(x=splitloc, ymin=-200, ymax=ea_init*ea_init, color='gray', linestyle='-', alpha=1)
ax[0].set_ylim(1*min(enesglob)-.05*ea_init, ea_init+.2*ea_init)
ax[1].set_ylim(1*min(enesglob)-.05*ea_init, ea_init+.2*ea_init)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rns[rxnind]}_chains.svg")
plt.show()
# ax.plot(h.ordered_leaves[0].data.optimized.integrated_path_length*.5, h.data.initial_chain.energies_kcalmol, 'o-', color='tab:orange', alpha=1)
# -

len(h.children[0].children[0].children[0].children[0].children[].data.chain_trajectory)

from neb_dynamics.helper_functions import RMSD

ch.visualize_chain([chain.initial_chain[1] for chain in h.get_optimization_history()])

len(h.get_optimization_history())

ch.visualize_chain([h.ordered_leaves[0].data.optimized.get_ts_node(), n.optimized[10], 
                    h.ordered_leaves[1].data.optimized.get_ts_node(), n.optimized[17]])

# +

n.optimized[10].structure.save("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Meyer-Schuster-Rearrangement/tsres/tsg_node10.xyz")
n.optimized[17].structure.save("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Meyer-Schuster-Rearrangement/tsres/tsg_node17.xyz")
# -

ch.visualize_chain(n.optimized)

# +
f, ax = plt.subplots(nrows=1)
fs = 12


ax.plot(h.ordered_leaves[0].data.initial_chain.integrated_path_length*splitloc, h.ordered_leaves[0].data.initial_chain.energies_kcalmol, 'o-', color='tab:blue', alpha=initalpha)
ax.plot(h.ordered_leaves[0].data.optimized.integrated_path_length*splitloc, enes, 'o-', color='tab:blue', alpha=1)

ax.plot((h.ordered_leaves[1].data.initial_chain.integrated_path_length*(1-splitloc))+splitloc, h.ordered_leaves[1].data.initial_chain.energies_kcalmol+enes[-1], 'o-', color='tab:orange', alpha=initalpha)
ax.plot((h.ordered_leaves[1].data.optimized.integrated_path_length*(1-splitloc))+splitloc, h.ordered_leaves[1].data.optimized.energies_kcalmol+enes[-1], 'o-', color='tab:orange', alpha=1)

ea_init = h.data.initial_chain.get_eA_chain()


plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)", fontsize=fs)
plt.xlabel("Reaction progress", fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rns[rxnind]}_chains.svg")
plt.show()
# ax.plot(h.ordered_leaves[0].data.optimized.integrated_path_length*.5, h.data.initial_chain.energies_kcalmol, 'o-', color='tab:orange', alpha=1)
# -

from qcinf import structure_to_smiles
from qcio import ProgramOutput
structure_to_smiles(h.output_chain[-1].structure)

structs = Structure.from_xyz_multi(open("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/tsg_folder/Madelung-Indole-Synthesis_tsg_1_optim.xyz").read())

view.view(structs[-1])

h.data.optimized[6].structure.save("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Lobry-de-Bruyn-Van-Ekenstein-Transformation/tsres/tsg_node6.xyz")



ch.visualize_chain(h.data.optimized)

h.data.plot_opt_history(0)

ch.visualize_chain(h.data.optimized)

min(enes)

ch.visualize_chain(h.data.optimized)

for rn in rns:
    h = TreeNode.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rn}/ASNEB_climb_DFTv2")
    for i, leaf in enumerate(h.ordered_leaves):
        tsg = leaf.data.optimized.get_ts_node()
        tsg.structure.save(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/tsg_folder/{rn}_tsg_{i}.xyz")

results = {}
for rn in rns:
    h = TreeNode.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/{rn}/ASNEB_climb_DFT")
    results[rn] = []
    for i, leaf in enumerate(h.ordered_leaves):
        results[rn].append(leaf.data.optimized[0].structure)
        results[rn].append(Structure.open(f"/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/tsg_folder/{rn}_tsg_{i}_optim.xyz"))
        results[rn].append(leaf.data.optimized[-1].structure)

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Dakin-Oxidation/ASNEB_climb_XTB/")

oldminima = []
for leaf in h.ordered_leaves:
    oldminima.append((leaf.data.optimized[0].structure, leaf.data.optimized[-1].structure))

ind=3
from qcinf import structure_to_smiles
# structure_to_smiles(oldminima[ind][0]), structure_to_smiles(oldminima[ind][1])
structure_to_smiles(oldminima[ind][0]), structure_to_smiles(oldminima[ind][1])

paths = sorted(list(Path("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/refinement/").glob("results_pair_*.xyz")))

# ch.visualize_chain(nebs[0].output_chain)
ch.visualize_chain(h.ordered_leaves[0].data.optimized)

nebs = [TreeNode.read_from_disk(p.parent / p.stem) for p in paths]

newminima= []
for neb in nebs:
    for leaf in neb.ordered_leaves:
        newminima.append((leaf.data.optimized[0].structure, leaf.data.optimized[-1].structure))

ch.visualize_chain(nebs[1].output_chain)

ind=4
from qcinf import structure_to_smiles
structure_to_smiles(newminima[ind][0]), structure_to_smiles(newminima[ind][1])

matches = list(zip(h.ordered_leaves, nebs))

# +
f, ax = plt.subplots()
fs = 12

prev_ene_xtb = 0
prev_ene_dft = 0

for i, (xtb, dft) in enumerate(matches):
    # Calculate offsets
    path_offset = i * 0.25
    
    # Calculate energies with cumulative offset
    enes_xtb = xtb.data.optimized.energies_kcalmol + prev_ene_xtb
    # enes_xtb = dft.data.initial_chain.energies_kcalmol + prev_ene_dft
    enes_dft = dft.output_chain.energies_kcalmol + prev_ene_dft
    
    # Plotting
    ax.plot((xtb.data.optimized.integrated_path_length * 0.25) + path_offset, enes_xtb, '--', color='tab:gray', linewidth=3)
    # ax.plot((dft.data.initial_chain.integrated_path_length * 0.25) + path_offset, enes_xtb, '--', color='tab:gray', linewidth=3)
    ax.plot((dft.output_chain.integrated_path_length * 0.25) + path_offset, enes_dft, 'o-')
    
    # Update cumulative energy for the next iteration
    prev_ene_xtb = enes_xtb[-1]
    prev_ene_dft = enes_dft[-1]



met1_handle = Line2D([0], [0], color='gray', linestyle='--', label='XTB', linewidth=3)

# 2. Define the "rainbow" handle
# Note: Matplotlib legends don't natively support gradient lines in one handle, 
# but we can mimic the style or use a specific color to represent it.
# If you want it to visually pop, we can use a distinct color or a custom marker.
met2_handle = Line2D([0], [0], color='rebeccapurple', linewidth=2, label='DFT') 
# Adding the legend using the handles
ax.legend(handles=[met1_handle, met2_handle], fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)", fontsize=fs)
plt.xlabel("Reaction progress", fontsize=fs)
import matplotlib.pyplot as plt

# Assuming 'ax' is your axes object
# Define your boundaries and matching colors
segments = [
    (0.0, 0.25, 'tab:blue'),
    (0.25, 0.50, 'tab:orange'),
    (0.50, 0.75, 'tab:green'), 
    (0.75, 1.0, 'tab:red')
]

for x_start, x_end, col in segments:
    plt.axvspan(x_start, x_end, color=col, alpha=0.1, lw=1.0)

# Ensure the background blocks stay behind your data
plt.gca().set_axisbelow(True)
plt.savefig("/home/jdep/T3D_data/msmep_draft/figures/refinement_dakin.svg")
plt.show()

# -

h2 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Robinson-Gabriel-Synthesis/mep_output_msmep/")

view.view(h2.output_chain[0].structure, h2.output_chain[-1].structure, show_indices=True)

h3 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Robinson-Gabriel-Synthesis/mep2")

h2.output_chain[0].structure.save("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Robinson-Gabriel-Synthesis/start.xyz")
h2.output_chain[-1].structure.save("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Robinson-Gabriel-Synthesis/end.xyz")

# +

max(h3.output_chain.energies_kcalmol)-2.2, max(h2.output_chain.energies_kcalmol)-3.28, 
# -

from neb_dynamics import RunInputs
ri = RunInputs()

tsres2 = ri.engine._compute_ts_result(h2.output_chain.get_ts_node())

tsres3 = ri.engine._compute_ts_result(h3.output_chain.get_ts_node())

view.view(tsres2)

ch.visualize_chain(h2.output_chain)
# ch.visualize_chain(h3.output_chain)

s=5
f, ax = plt.subplots(figsize=(1.18*s,s))
fs = 12
ax.plot(h2.output_chain.integrated_path_length, h2.output_chain.energies_kcalmol, 'o-', label="Amide Oyxgen in ring")
ax.plot(h3.output_chain.integrated_path_length, h3.output_chain.energies_kcalmol, 'o-', label="Ketone Oxygen in ring")
plt.legend(fontsize=fs, loc='lower center')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)", fontsize=fs)
plt.xlabel("Reaction progress", fontsize=fs)
plt.savefig("/home/jdep/T3D_data/msmep_draft/figures/robinson_oxygen_comparison.svg")

# +

h2= TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Meyer-Schuster-Rearrangement/solvation/mep_output_msmep/")

# +

ch.visualize_chain(h2.data.initial_chain)

# +

h3 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures/Meyer-Schuster-Rearrangement/ASNEB_climb_XTB")
# -

ch.visualize_chain(h2.output_chain)

# +

ch.visualize_chain(h2.output_chain)
# -

len(h2.get_optimization_history())



ch.visualize_chain(h2.output_chain)

comparisons = list(zip(oldminima, newminima))

comp = comparisons[3]
view.view(comp[0][0], comp[1][0], comp[0][1], comp[1][1]) 

# +
tol = 0.002
nbi = NEBInputs(
    tol=tol * BOHR_TO_ANGSTROMS,
    barrier_thre=0.1,  # kcalmol,

    rms_grad_thre=tol * BOHR_TO_ANGSTROMS,
    max_rms_grad_thre=tol * BOHR_TO_ANGSTROMS*2.5,
    ts_grad_thre=tol * BOHR_TO_ANGSTROMS*2.5,
    ts_spring_thre=tol * BOHR_TO_ANGSTROMS*1.5,

    v=1,
    max_steps=500,

)
# -
h = TreeNode.read_from_disk(
    "/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb/Wittig/")
neb12 = NEB.read_from_disk(
    "/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_neb/Wittig12_neb.xyz")
neb24 = NEB.read_from_disk(
    "/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_neb/Wittig24_neb.xyz")


# +
rn = 'Wittig'
s = 5
fs = 18
f, ax = plt.subplots(figsize=(2*s, s))

plt.plot(h.output_chain.integrated_path_length, h.output_chain.energies_kcalmol,
         'o-', color='black', label='ASNEB')  # , linewidth=3.3)
plt.plot(neb12.optimized.integrated_path_length, neb12.optimized.energies_kcalmol,
         '^-', color='blue', label='NEB(12)')  # , alpha=.3)
plt.plot(neb24.optimized.integrated_path_length, neb24.optimized.energies_kcalmol,
         'x-', color='red', label=f'NEB({len(neb_long.optimized)})')  # , alpha=.3)

# plt.plot(pl_dft, joined.energies_kcalmol, 'o-',label='ASNEB_DFT')#, linewidth=3.3)

plt.legend(fontsize=fs)


plt.xlabel('Normalized path length', fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.savefig(
    f"/home/jdep/T3bD_data/msmep_draft/figures/{rn}_comparison_paths.svg")
plt.show()
# -

gcalls = [2230, 1251, 2506]

neb12.optimized.energies

neb24.optimized.energies

h.output_chain.energies

# +

neb24.plot_opt_history(1)
# -

h.output_chain.energies

neb24.optimized.energies

neb12.plot_opt_history(1)

neb24.plot_opt_history(1)

# +
rn = 'Wittig'
# rn = 'Robinson-Gabriel-Synthesis'
# rn = 'Lobry-de-Bruyn-Van-Ekenstein-Transformation'
# rn = 'Bamford-Stevens-Reaction'
# rn = 'Ramberg-Backlund-Reaction-Bromine'
# rn = 'Rupe-Rearrangement'

ref_p = Path(
    f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_01_NOSIG_NOMR_v2/")
h = TreeNode.read_from_disk(ref_p, neb_parameters=nbi)
neb = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB_12_nodes_neb",
                         chain_parameters=ChainInputs(k=0.1, delta_k=0.09),
                         neb_parameters=nbi)
neb_long = NEB.read_from_disk(
    f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB_03_NOSIG_NOMR_neb", chain_parameters=ChainInputs(k=0.1, delta_k=0.09))
# neb_long = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB_005_noMR_neb")

# neb_long2 = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB36_neb")

ngc_asneb = open(
    f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_{ref_p.stem}').read().splitlines()
ngc_asneb = sum([int(line.split()[2]) for line in ngc_asneb if '>>>' in line])


ngc_neb = open(
    f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_12_nodes').read().splitlines()
ngc_neb = sum([int(line.split()[2]) for line in ngc_neb if '>>>' in line])

# ngc_neb_long = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_01_noSIG').read().splitlines()
ngc_neb_long = open(
    f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_03_NOSIG_NOMR').read().splitlines()
ngc_neb_long = sum([int(line.split()[2])
                   for line in ngc_neb_long if '>>>' in line])

# ngc_neb_long2 = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_36imgs').read().splitlines()
# ngc_neb_long2 = sum([int(line.split()[2]) for line in ngc_neb_long2 if '>>>' in line])
# -

hdb = TreeNode.read_from_disk(
    "/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Claisen-Rearrangement-Aromatic/ASNEB_03_NOSIG_NOMR_GI/")

hdb.output_chain.plot_chain()
hdb.output_chain.to_trajectory()

ref = Refiner()
leaves = ref.read_leaves_from_disk(Path(
    f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_03_NOSIG_NOMR_v2_refined/"))
joined = ref.join_output_leaves(leaves)

# +
s = 5
fs = 18
f, ax = plt.subplots(figsize=(1*s, s))
plt.bar(x=['NEB(12)', f'NEB({len(neb_long.optimized)})', 'ASNEB'], height=[
        ngc_neb, ngc_neb_long, ngc_asneb], color='white', hatch='/', edgecolor='black')

plt.ylabel("Number of gradient calls", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.savefig(
    f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_comparison_barplots.svg")
plt.show()

# +

self = neb
s = 8
fs = 18
ms = 5

all_chains = self.chain_trajectory


ens = np.array([c.energies-c.energies[0] for c in all_chains])
all_integrated_path_lengths = np.array(
    [c.integrated_path_length for c in all_chains])
opt_step = np.array(list(range(len(all_chains))))
s = 7
fs = 18
ax = plt.figure(figsize=(1.16*s, s)).add_subplot(projection='3d')

# Plot a sin curve using the x and y axes.
x = opt_step
ys = all_integrated_path_lengths
zs = ens
for i, (xind, y) in enumerate(zip(x, ys)):
    if i == len(h.data.chain_trajectory):
        ax.plot([xind]*len(y), y, 'o-', zs=zs[i], color='red',
                markersize=ms, label='early stop chain')
    elif i < len(ys) - 1:
        ax.plot([xind]*len(y), y, 'o-', zs=zs[i],
                color='gray', markersize=ms, alpha=.1)
    else:
        ax.plot([xind]*len(y), y, 'o-', zs=zs[i], color='blue',
                markersize=ms, label='optimized chain')
ax.grid(False)

ax.set_xlabel('optimization step', fontsize=fs)
ax.set_ylabel('integrated path length', fontsize=fs)
ax.set_zlabel('energy (hartrees)', fontsize=fs)

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-45)
plt.tight_layout()
plt.legend(fontsize=fs)
plt.savefig(
    f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_early_stop_chain_traj.svg")
plt.show()


# -


def get_mechanism_mols(chain, iter_dist=12):
    out_mols = [chain[0].tdstructure.molecule_rp]
    nsteps = int(len(chain)/iter_dist)
    for ind in range(nsteps):
        r = chain[ind*12].tdstructure.molecule_rp
        if r != out_mols[-1]:
            out_mols.append(r)

    p = chain[-1].tdstructure.molecule_rp
    if p != out_mols[-1]:
        out_mols.append(p)
    return out_mols


# +
s = 5
fs = 18
ct = neb.chain_trajectory

avg_rms_gperp = []
max_rms_gperp = []
avg_rms_g = []
barr_height = []
ts_gperp = []
inf_norm_g = []
inf_norm_gperp = []
springs_g = []


for ind in range(1, len(ct)):
    avg_rms_g.append(sum(ct[ind].rms_gradients[1:-1]) / (len(ct[ind])-2))
    avg_rms_gperp.append(sum(ct[ind].rms_gperps[1:-1]) / (len(ct[ind])-2))
    max_rms_gperp.append(max(ct[ind].rms_gperps))
    springs_g.append(ct[ind].ts_triplet_gspring_infnorm)
    barr_height.append(abs(ct[ind].get_eA_chain() - ct[ind-1].get_eA_chain()))
    ts_node_ind = ct[ind].energies.argmax()
    ts_node_gperp = np.max(ct[ind].get_g_perps()[ts_node_ind])
    ts_gperp.append(ts_node_gperp)
    inf_norm_val_g = inf_norm_g.append(np.max(ct[ind].gradients))
    inf_norm_val_gperp = inf_norm_gperp.append(np.max(ct[ind].get_g_perps()))


f, ax = plt.subplots(figsize=(1.6*s, s))
plt.plot(avg_rms_gperp, color='blue')  # , label='RMS Grad$_{\perp}$')
plt.plot(max_rms_gperp, color='orange')  # , label='Max RMS Grad$_{\perp}$')
plt.plot(springs_g, color='purple')
# plt.plot(avg_rms_g, label='RMS Grad')
plt.plot(ts_gperp, color='green')  # ,label='TS gperp')
# plt.plot(inf_norm_g,label='Inf norm G')
# plt.plot(inf_norm_gperp,label='Inf norm Gperp')
# plt.ylabel("argmax(|Gradient$_{\perp}$|)", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

xmin = ax.get_xlim()[0]
xmax = ax.get_xlim()[1]
ymin = ax.get_ylim()[0]
ymax = ax.get_ylim()[1]
ax.hlines(y=self.parameters.rms_grad_thre, xmin=xmin, xmax=xmax,
          label='avg(RMS Grad$_{\perp}$)', linewidth=3, linestyle='--', color='blue')
ax.hlines(y=self.parameters.max_rms_grad_thre, xmin=xmin, xmax=xmax,
          label='max(RMS Grad$_{\perp}$)', linewidth=5, alpha=.8, linestyle='--', color='orange')
ax.hlines(y=self.parameters.ts_grad_thre, xmin=xmin, xmax=xmax,
          label='infnorm(TS Grad$_{\perp}$)', linewidth=3, linestyle='--', color='green')
ax.hlines(y=self.parameters.ts_spring_thre, xmin=xmin, xmax=xmax,
          label='infnorm(TS triplet Grad$_{spr}$)', linewidth=3, linestyle='--', color='purple')


self.parameters.early_stop_force_thre = 0.03*BOHR_TO_ANGSTROMS


# ax.hlines(y=self.parameters.early_stop_force_thre, xmin=xmin, xmax=xmax, label='early stop threshold', linestyle='--', linewidth=3, color='red')


# ax.vlines(x=18, ymin=ymin, ymax=ymax, linestyle='--', color='red', label='early stop', linewidth=4)

# ax2 = plt.twinx()
# plt.plot(barr_height, 'o--',label='barr_height_delta', color='purple')
# plt.ylabel("Barrier height data", fontsize=fs)

plt.yticks(fontsize=fs)


# ax2.hlines(y=self.pxarameters.barrier_thre, xmin=xmin, xmax=xmax, label='barrier_thre', linestyle='--', color='purple')
# f.legend(fontsize=15, bbox_to_anchor=(1.35,.8))
f.legend(fontsize=fs)
plt.ylim(0, 0.02)
plt.tight_layout()
plt.savefig(
    f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_early_stop_convergence.svg")
plt.show()


# +
s = 5
fs = 18
f, ax = plt.subplots(figsize=(2*s, s))
normalize_pl = 1
if normalize_pl:
    pl_h = h.output_chain.integrated_path_length
    pl_neb = neb.optimized.integrated_path_length
    pl_neb_long = neb_long.optimized.integrated_path_length
    pl_dft = joined.integrated_path_length
    # pl_neb_long2 = neb_long2.optimized.integrated_path_length
    xl = "Reaction progression"
else:
    pl_h = h.output_chain.path_length
    pl_neb = neb.optimized.path_length
    pl_neb_long = neb_long.optimized.path_length
    # pl_neb_long2 = neb_long2.optimized.path_length
    xl = "Path length"

plt.plot(pl_h, h.output_chain.energies_kcalmol, 'o-',
         color='black', label='ASNEB')  # , linewidth=3.3)
plt.plot(pl_neb, neb.optimized.energies_kcalmol, '^-',
         color='blue', label='NEB(12)')  # , alpha=.3)
plt.plot(pl_neb_long, neb_long.optimized.energies_kcalmol, 'x-',
         color='red', label=f'NEB({len(neb_long.optimized)})')  # , alpha=.3)

# plt.plot(pl_dft, joined.energies_kcalmol, 'o-',label='ASNEB_DFT')#, linewidth=3.3)

plt.legend(fontsize=fs)


plt.xlabel(xl, fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.savefig(
    f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_comparison_paths.svg")
plt.show()


# -

def recreate_gis(leaves):
    out_cs = []
    for leaf in leaves:
        gi = leaf.data.initial_chain.to_trajectory().run_geodesic(nimages=12)
        c = Chain.from_traj(gi, ChainInputs())
        out_cs.append(c)
    return out_cs


# +
s = 5
fs = 18
f, ax = plt.subplots(figsize=(2*s, s))
plt.plot(h.output_chain.integrated_path_length,
         h.output_chain.energies_kcalmol, 'o-', color='black', label='ASNEB')
plt.plot(neb.initial_chain.integrated_path_length,
         neb.initial_chain.energies_kcalmol, '^--', color='blue', label='GI(12)')
plt.plot(neb_long.initial_chain.integrated_path_length,
         neb_long.initial_chain.energies_kcalmol, '^--', color='red', label='GI(24)')

colors = ['green', 'purple', 'gold', 'gray']
man_gis = recreate_gis(h.ordered_leaves)
last_val = 0
for i, (leaf, manual) in enumerate(zip(h.ordered_leaves, man_gis)):
    final_point = h.output_chain.integrated_path_length[12*i+11]
    start_point = h.output_chain.integrated_path_length[12*i]
    path_len_leaf = final_point - start_point

    plt.plot((manual.integrated_path_length*path_len_leaf)+start_point,
             manual.energies_kcalmol+last_val, 'o--', color=colors[i], label=f'GI leaf {i}')
    last_val = manual.energies_kcalmol[-1]
    # last_val = h.output_chain.path_length[12*i+11]


plt.legend(fontsize=fs)
plt.xlabel("Reaction progression", fontsize=fs)
plt.ylabel("Energies (kcal/mol)", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.savefig(
    f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_comparison_initial_guesses.svg")
plt.show()
# -
# ## Comparison of mechanisms


reactions = pload("/home/jdep/retropaths/data/reactions.p")


def get_out_chain(path):
    try:
        h = TreeNode.read_from_disk(path)
        print(h.get_num_opt_steps())
        out = h.output_chain
    except IndexError:
        n = NEB.read_from_disk(path / 'node_0.xyz')
        out = n.optimized
        print(len(n.chain_trajectory))

    except FileNotFoundError as e:
        print(f'{path} does not exist.')
        raise e

    return out


def _join_output_leaves(self, refined_leaves):
    joined_nodes = []
    [
        joined_nodes.extend(leaf.data.chain_trajectory[-2].nodes)
        for leaf in refined_leaves
        if leaf
    ]
    joined_chain = Chain(nodes=joined_nodes, parameters=self.cni)
    return joined_chain


def build_report(rn):
    # dft_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/production_vpo_tjm_xtb_preopt_msmep')
    dft_path = Path(
        f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/ASNEB_03_NOSIG_NOMR')
    xtb_path = Path(
        f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_03_NOSIG_NOMR_v2')
    refine_path = Path(
        f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_03_NOSIG_NOMR_v2_refined')
    refinement_done = False

    if rn in reactions:
        rp_rn = reactions[rn]
    else:
        rp_rn = None
    print('dft')
    out_dft = get_out_chain(dft_path)
    print('xtb')
    out_xtb = get_out_chain(xtb_path)
    if refine_path.exists():
        refinement_done = True
        ref = Refiner(v=1)
        print("refinement")
        refined_results = ref.read_leaves_from_disk(refine_path)

        # joined = ref.join_output_leaves(refined_results)
        joined = _join_output_leaves(ref, refined_results)

    plt.plot(out_dft.path_length, out_dft.energies_kcalmol, 'o-', label='dft')
    plt.plot(out_xtb.path_length, out_xtb.energies_kcalmol, 'o-', label='xtb')
    plt.ylabel("Energies (kcal/mol)")
    plt.xlabel("Path length")

    if refinement_done:
        plt.plot(joined.path_length, joined.energies_kcalmol,
                 'o-', label='refinement')
    plt.legend()
    plt.show()

    out_trajs = [out_dft, out_xtb]
    if refinement_done:
        out_trajs.append(joined)

    if rp_rn:
        return rp_rn.draw(size=(200, 200)), out_trajs
    else:
        return rp_rn, out_trajs


ind = 0


df_sub[['?' in val for val in df_sub['agrees?'].values]]

df_sub.loc[38]['experimental link']

rn = 'Bamford-Stevens-Reaction'
a, b = build_report(rn)
a

# +
c = b[2]
s = 5
fs = 18
f, ax = plt.subplots(figsize=(2*s, s))

plt.plot(c.integrated_path_length, c.energies_kcalmol, 'o-', color='black')

# plt.legend(fontsize=fs)


plt.xlabel('Reaction progression', fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.savefig(
    f"/home/jdep/T3D_data/msmep_draft/figures/SI_{rn}_refined_path.svg")
plt.show()
Molecule.draw_list(get_mechanism_mols(c), mode='d3')
# -

c.to_trajectory().draw()

ref = Refiner()
leaves = ref.read_leaves_from_disk(Path(
    "/home/jdep/T3D_data/msmep_draft/comparisons/structures/Azaindole-Synthesis/ASNEB_03_NOSIG_NOMR_v2_refined/"))
joined = ref.join_output_leaves(leaves)

tsg = b[0].get_ts_guess()

tsg.tc_model_method = 'wb97xd3'
tsg.tc_model_basis = 'def2-svp'

tsg.tc_freq_calculation()

for ind in range(10):
    rn, trajs = build_report(all_rns[ind])
    display(rn)


# +
df = pd.read_csv(
    "/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_yessig.csv")

all_rns = df['reaction_name'].to_list()


# -

def report(df, rn):
    row = df[df['reaction_name'] == rn]
    p = row['file_path'].values[0]
    p_ref = ∂
    return p


report(df, all_rns[0])

# ## Comparison of deployment strategies

h = TreeNode.read_from_disk(
    "/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/1-2-Amide-Phthalamide-Synthesis/ASNEB_03_NOSIG_NOMR/")

h.output_chain.plot_chain()
h.output_chain.to_trajectory().draw()

df_precond = pd.read_csv(
    "/home/jdep/T3D_data/msmep_draft/comparisons_dft/dataset_results_xtb_precondition.csv")
df_gi = pd.read_csv(
    "/home/jdep/T3D_data/msmep_draft/comparisons_dft/dataset_results_gi.csv")
df_ref = pd.read_csv(
    "/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_refinement.csv")

df_ref['n_grad_calls'].median(), df_precond['n_grad_calls'].median(
), df_gi['n_grad_calls'].median(),

# +
fs = 18
s = 7
f, ax = plt.subplots(figsize=(1*s, s))
offset = .5

lw = 3

xlabels = ['Refinement', 'XTB-Seed', 'GI-Seed']
x = np.arange(len(xlabels))
plt.boxplot(x=[
    df_ref.dropna()['n_grad_calls'],
    df_precond.dropna()['n_grad_calls'],
    df_gi.dropna()['n_grad_calls']],
    positions=x,
    medianprops={'linewidth': lw, 'color': 'black'},
    boxprops={'linewidth': lw},
    capprops={'linewidth': lw-1},
    patch_artist=True)
# fill with colors
for patch in boxesnosig['boxes']:
    patch.set_facecolor('#E2DADB')


plt.ylabel("Gradient calls",
           fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=fs)

plt.xlabel("Early stop gradient threshold", fontsize=fs)
xmin, xmax = ax.get_xlim()
plt.yticks(fontsize=fs)

plt.ylim(0, 4000)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/noSIG_yesSIG_boxplot.svg")
plt.show()

# +
fs = 18
s = 7
f, ax = plt.subplots(figsize=(1*s, s))
offset = .5

lw = 3


xlabels = ['Refinement', 'XTB-Seed', 'GI-Seed']
bottom = np.zeros(len(xlabels))
x = np.arange(len(xlabels))
elem_step_heights = [
    len(df_ref[df_ref['n_rxn_steps'] == 1]),
    len(df_precond[df_precond['n_rxn_steps'] == 1]),
    len(df_gi[df_gi['n_rxn_steps'] == 1]),

]

multi_step_heights = [
    len(df_ref[df_ref['n_rxn_steps'] != 1]),
    len(df_precond[df_precond['n_rxn_steps'] != 1]),
    len(df_gi[df_gi['n_rxn_steps'] != 1]),

]


plt.bar(x=xlabels, height=elem_step_heights,
        bottom=bottom, color='#FE5D9F',
        label='Single step rxn',
        width=offset)

bottom += elem_step_heights

plt.bar(x=xlabels, height=multi_step_heights,
        bottom=bottom, color='#52FFEE',
        label='Multi step rxn',
        width=offset)


plt.ylabel("Count",
           fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=fs)

xmin, xmax = ax.get_xlim()
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/noSIG_yesSIG_boxplot.svg")
plt.show()
