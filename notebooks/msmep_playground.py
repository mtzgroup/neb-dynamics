# +
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.helper_functions import RMSD
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs
from neb_dynamics.molecule import Molecule
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.pot import Pot

import pandas as pd

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

import matplotlib.patches as mpatches


from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/start_opt_msmep/")

h.output_chain.plot_chain()

h.ordered_leaves[0].data.plot_convergence_metrics(1)

h.output_chain.plot_chain()
h.output_chain.to_trajectory().draw();

tsg = h.output_chain.get_ts_guess()

tsg

tsg_opt = tsg.xtb_sella_geom_optimization()

tsg.energy_xtb()

tsg_opt.energy_xtb()

# # Database

# +

IGNORING_RXNS = ['Aza-Grob-Fragmentation-X-Bromine', 'Bamberger-Rearrangement']


# -

def subset_to_multi_step_rns(df, by_list=False):
    if by_list:
        return df[df['reaction_name'].isin(by_list)]
    return df[df['n_rxn_steps']>1].reindex()


def sanitize(row):
    return row['reaction name'].replace("'","").replace("[","").replace("]","").replace(",","")


df_jan = pd.read_csv("/home/jdep/T3D_data/msmep_draft/msmep_reaction_successes.csv")
df_jan = df_jan.dropna()
df_jan['reaction name'] = df_jan.apply(sanitize, axis=1)
df_gi = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons_dft/dataset_results_gi.csv")

# nosig_nomrs_dfs = [pd.read_csv(f"/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_{label}_nosig_nomr.csv").dropna() for label in ['5','1','05','03','01','005','0']]
# nosig_nomrs_dfs = [subset_to_elem_step_rns(pd.read_csv(f"/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_{label}_nosig_nomr.csv").dropna()) for label in ['5','1','05','03','01','005','0']]
nosig_nomrs_dfs = [subset_to_multi_step_rns(pd.read_csv(f"/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_{label}_nosig_nomr.csv").dropna()) for label in ['5','1','05','03','01','005','0']]
for df in nosig_nomrs_dfs:
    df['TOTAL_GRAD_CALLS'] = df['n_grad_calls']+df['n_grad_calls_geoms']

true_ms = []
for fp in nosig_nomrs_dfs[-1][nosig_nomrs_dfs[-1]['n_rxn_steps']>1]['file_path']:
    p = Path(fp)/'ASNEB_003_yesSIG'
    h = TreeNode.read_from_disk(p)
    if len(h.ordered_leaves) > 1:
        true_ms.append(fp)

true_ms_names = [st.split("/")[-1] for st in true_ms]

nosig_nomrs_dfs = [subset_to_multi_step_rns(pd.read_csv(f"/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_{label}_nosig_nomr.csv").dropna(),
                                           by_list=true_ms_names ) for label in ['5','1','05','03','01','005','0']]

df_sub = df_jan[df_jan['reaction name'].isin(nosig_nomrs_dfs[3]['reaction_name'])]

len(df_sub)

df_sub.loc[12]

df_sub['agrees?'].value_counts()


def subset_to_elem_step_rns(df, by_list=False, inp_list=None):
    if by_list:
        assert inp_list is not None, 'please input a list to subset by'
        return df[df['reaction_name'].isin(inp_list)]
    return df[df['n_rxn_steps']==1].reindex()


import pandas as pd

df['n_rxn_steps'].argmax()

len(true_ms)

# +
df_neb = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_NEBS.csv")

df_neb.drop(df_neb[df_neb['reaction_name'].isin(IGNORING_RXNS)].index, inplace=True)

# df_neb = subset_to_multi_step_rns(df_neb, by_list=True, inp_list=true_ms_names)

# +
# print(len(df_neb))
# -

def print_stats(df):
    print(f"median: {df['n_grad_calls'].median()}, mean: {df['n_grad_calls'].mean()}, std: {df['n_grad_calls'].std()}")


print_stats(df_neb)

df = nosig_nomrs_dfs[3]

df.iloc[13]

# +
fs=18
s=8
f,ax = plt.subplots(figsize=(1*s, s))
offset=.5
# es_labels = ['0.5 no MR','0.5 yes MR', 'NEB']
es_labels = ['5','1','05','03','01','005','0']
x = np.arange(len(es_labels))
lw = 3

# colname = 'n_grad_calls'
# colname = 'n_grad_calls_geoms'
colname = 'TOTAL_GRAD_CALLS'
# colname = 'n_opt_splits'


# boxesyessig = plt.boxplot(x=[df[colname].dropna() for df in (df_xtb_5_yessig, df_xtb_5_yessig_yesmr)]+[df_neb[colname].dropna()],
#            positions=x, widths=offset-.1,
#            medianprops={'linewidth':lw, 'color':'black'},
#             boxprops={'linewidth':lw},
#            capprops={'linewidth':lw-1},
#            patch_artist=True)

boxesyessig = plt.boxplot(x=[df[colname].dropna() for df in nosig_nomrs_dfs],
           positions=x, widths=offset-.1,
           medianprops={'linewidth':lw, 'color':'black'},
            boxprops={'linewidth':lw},
           capprops={'linewidth':lw-1},
           patch_artist=True)


for patch in boxesyessig['boxes']:
    patch.set_facecolor('#6D696A')

# plt.ylabel("Gradient calls",fontsize=fs)
# plt.ylabel("Gradient calls from geom opts",fontsize=fs)
plt.ylabel("Total Gradient Calls",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

ax.set_xticks(x)

ax.set_xticklabels(es_labels,fontsize=fs)

plt.xlabel("Early stop gradient threshold",fontsize=fs)

xmin, xmax = ax.get_xlim()
plt.yticks(fontsize=fs)
# plt.hlines(xmin=xmin,xmax=xmax, y=df_neb[colname].median(),
#            linestyles='-', color='red',linewidth=lw,
#            label='NEB median value')
# plt.legend(fontsize=fs)


nosig_patch = mpatches.Patch(color='#E2DADB', label='no SIG')
yessig_patch = mpatches.Patch(color='#6D696A', label='yes SIG')
handles, labels = ax.get_legend_handles_labels()


handles.extend([nosig_patch, yessig_patch])
# plt.legend(handles=handles,fontsize=fs)
# plt.ylim(0,60000)
# plt.ylim(0,40)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/noSIG_yesSIG_boxplot_farout.svg")
plt.show()
# -

df_neb['n_grad_calls'].median(), df_neb['n_grad_calls'].std(), df_neb['n_grad_calls'].mean()

nosig_nomrs_dfs[0]['n_grad_calls'].median(), nosig_nomrs_dfs[0]['n_grad_calls'].std(), nosig_nomrs_dfs[0]['n_grad_calls'].mean()

nosig_nomrs_dfs[0]['TOTAL_GRAD_CALLS'].median()

[len(df.dropna()) for df in nosig_nomrs_dfs]

# +
fs=18
s=8
f,ax = plt.subplots(figsize=(1*s, s))
offset=.5
# es_labels = ['0.5 no MR','0.5 yes MR', 'NEB']
es_labels = ['ASNEB','NEB']
df = nosig_nomrs_dfs[3]
x = np.arange(len(es_labels))
lw = 3

colname = 'n_grad_calls'
# colname = 'n_grad_calls_geoms'
# colname = 'TOTAL_GRAD_CALLS'
# colname = 'n_opt_splits'


# boxesyessig = plt.boxplot(x=[df[colname].dropna() for df in (df_xtb_5_yessig, df_xtb_5_yessig_yesmr)]+[df_neb[colname].dropna()],
#            positions=x, widths=offset-.1,
#            medianprops={'linewidth':lw, 'color':'black'},
#             boxprops={'linewidth':lw},
#            capprops={'linewidth':lw-1},
#            patch_artist=True)

boxesyessig = plt.boxplot(x=[df['TOTAL_GRAD_CALLS'].dropna(), df_neb['n_grad_calls'].dropna()],
           positions=x, widths=offset-.1,
           medianprops={'linewidth':lw, 'color':'black'},
            boxprops={'linewidth':lw},
           capprops={'linewidth':lw-1},
           patch_artist=True)


for patch in boxesyessig['boxes']:
    patch.set_facecolor('#6D696A')

# plt.ylabel("Gradient calls",fontsize=fs)
# plt.ylabel("Gradient calls from geom opts",fontsize=fs)
plt.ylabel("NEB Gradient Calls",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

ax.set_xticks(x)

ax.set_xticklabels(es_labels,fontsize=fs)

xmin, xmax = ax.get_xlim()
plt.yticks(fontsize=fs)


nosig_patch = mpatches.Patch(color='#E2DADB', label='no SIG')
yessig_patch = mpatches.Patch(color='#6D696A', label='yes SIG')
handles, labels = ax.get_legend_handles_labels()


handles.extend([nosig_patch, yessig_patch])

# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/noSIG_yesSIG_boxplot_farout.svg")
plt.show()
# -

nosig_nomrs_dfs[3]

[df[colname].min() for df in nosig_nomrs_dfs]

nosig_nomrs_dfs[0][colname].argmin()





nosig_nomrs_dfs[0].iloc[19]

[len(df.dropna()) for df in nosig_nomrs_dfs]

nosig_nomrs_dfs[-1][colname]u.argmax()

nosig_nomrs_dfs[-1].iloc[24]

h1 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Baker-Venkataraman-Rearrangement/ASNEB_0_NOSIG_NOMR")

h2 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Baker-Venkataraman-Rearrangement/ASNEB_0_NOSIG_NOMR_v2")

h1.output_chain.plot_chain()
h1.output_chain.to_trajectory().draw();

h2.output_chain.plot_chain()
h2.output_chain.to_trajectory().draw();

h2.output_chain.plot_chain()
h2.output_chain.to_trajectory().draw();

df['TOTAL_GRAD_CALLS'].sort_values()

h = TreeNode.read_from_disk('/home/jdep/T3D_data/msmep_draft/comparisons/structures/Azaindole-Synthesis/debug/')

h.output_chain.plot_chain()

h.output_chain.to_trajectory()

h.output_chain[-24].tdstructure

h.output_chain[-12].tdstructure

td_list = h.data.initial_chain[-1].tdstructure.split_td_into_frags()


def _is_conformer_identical(self, other) -> bool:
    if self._is_connectivity_identical(other):
        aligned_self = self.tdstructure.align_to_td(other.tdstructure)

        global_dist = RMSD(aligned_self.coords, other.tdstructure.coords)[0]
        per_frag_dists = []
        self_frags = self.tdstructure.split_td_into_frags()
        other_frags = other.tdstructure.split_td_into_frags()
        for frag_self, frag_other in zip(self_frags, other_frags):
            aligned_frag_self = frag_self.align_to_td(frag_other)
            frag_dist = RMSD(aligned_frag_self.coords, frag_other.coords)[0]
            per_frag_dists.append(frag_dist)
        print(f"{per_frag_dists=}")
        print(f"{global_dist=}")

        en_delta = np.abs((self.energy - other.energy) * 627.5)

        global_rmsd_identical = global_dist < self.RMSD_CUTOFF
        fragment_rmsd_identical = max(per_frag_dists) < self.RMSD_CUTOFF
        rmsd_identical = global_rmsd_identical and fragment_rmsd_identical
        energies_identical = en_delta < self.KCAL_MOL_CUTOFF
        # print(f"\nbarrier_to_conformer_rearr: {barrier} kcal/mol\n{en_delta=}\n")

        if rmsd_identical and energies_identical: #and barrier_accessible:
            return True
        else:
            return False
    else:
        return False



_is_conformer_identical(h.output_chain[-2], h.output_chain[-1])

a = h.output_chain[12*21].tdstructure

b = h.output_chain[12*22].tdstructure

len(a.molecule_rp.separate_graph_in_pieces())


np.sqrt(np.dot(a.align_to_td(b).coords.flatten(), b.coords.flatten()))/atomn


def _is_conformer_identical(self, other) -> bool:
    if self._is_connectivity_identical(other):
        aligned_self = self.tdstructure.align_to_td(other.tdstructure)
        dist = RMSD(aligned_self.coords, other.tdstructure.coords)[0]
        en_delta = np.abs((self.energy - other.energy) * 627.5)

        rmsd_identical = dist < self.RMSD_CUTOFF
        energies_identical = en_delta < self.KCAL_MOL_CUTOFF

        print(dist)

        if rmsd_identical and energies_identical: #and barrier_accessible:
            return True
        else:
            return False


    else:
        return False

a = h.output_chain[621].tdstructure.align_to_td(h.output_chain[641].tdstructure).coords
b = h.output_chain[641].coords

a = h.output_chain[621].tdstructure
b = h.output_chain[641].tdstructure

dir(a)

atomn = len(a)

np.sqrt(np.dot(a.flatten(), b.flatten()))/atomn



h.output_chain[621].tdstructure

h.output_chain[-12].tdstructure

_is_conformer_identical(h.output_chain[-12], h.output_chain[-1])

_is_conformer_identical(h.output_chain[621], h.output_chain[641])

nosig_nomrs_dfs[0].iloc[12]['file_path']

# +
fs=18
s=8
f,ax = plt.subplots(figsize=(1*s, s))
offset=.5
# es_labels = ['0.5 no MR','0.5 yes MR', 'NEB']
es_labels = ['5','1','05','01','005']
x = np.arange(len(es_labels))
lw = 3

colname = 'n_grad_calls_geoms'
# colname = 'n_opt_splits'


# boxesyessig = plt.boxplot(x=[df[colname].dropna() for df in (df_xtb_5_yessig, df_xtb_5_yessig_yesmr)]+[df_neb[colname].dropna()],
#            positions=x, widths=offset-.1,
#            medianprops={'linewidth':lw, 'color':'black'},
#             boxprops={'linewidth':lw},
#            capprops={'linewidth':lw-1},
#            patch_artist=True)

boxesyessig = plt.boxplot(x=[df[colname].dropna() for df in yessig_yesmrs_dfs],
           positions=x, widths=offset-.1,
           medianprops={'linewidth':lw, 'color':'black'},
            boxprops={'linewidth':lw},
           capprops={'linewidth':lw-1},
           patch_artist=True)


for patch in boxesyessig['boxes']:
    patch.set_facecolor('#6D696A')

plt.ylabel("Gradient calls",
           fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

ax.set_xticks(x)

ax.set_xticklabels(es_labels,fontsize=fs)

plt.xlabel("Early stop gradient threshold",fontsize=fs)
xmin, xmax = ax.get_xlim()
plt.yticks(fontsize=fs)

plt.legend(fontsize=fs)


nosig_patch = mpatches.Patch(color='#E2DADB', label='no SIG')
yessig_patch = mpatches.Patch(color='#6D696A', label='yes SIG')
handles, labels = ax.get_legend_handles_labels()


handles.extend([nosig_patch, yessig_patch])
plt.legend(handles=handles,fontsize=fs)
# plt.ylim(0,7000)
# plt.ylim(0,40)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/noSIG_yesSIG_boxplot_farout.svg")
plt.show()

# +
fs=18
s=8
f,ax = plt.subplots(figsize=(1*s, s))
offset=.5
es_labels = ['0.5','0.1','0.05','0.03','0.01','no stop', 'NEB']
x = np.arange(len(es_labels))
lw = 3

colname = 'n_grad_calls'
# colname = 'n_opt_splits'

boxesnosig = plt.boxplot(x=[df[colname].dropna() for df in nosig_dfs]+[np.nan],
           positions=x-offset, widths=offset-.1,
           medianprops={'linewidth':lw, 'color':'black'},
           boxprops={'linewidth':lw},
           capprops={'linewidth':lw-1},
           patch_artist=True)
# fill with colors
for patch in boxesnosig['boxes']:
    patch.set_facecolor('#E2DADB')




boxesyessig = plt.boxplot(x=[df[colname].dropna() for df in yessig_dfs]+[df_neb[colname].dropna()],
           positions=x, widths=offset-.1,
           medianprops={'linewidth':lw, 'color':'black'},
            boxprops={'linewidth':lw},
           capprops={'linewidth':lw-1},
           patch_artist=True)


for patch in boxesyessig['boxes']:
    patch.set_facecolor('#6D696A')

plt.ylabel("Gradient calls",
           fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

ax.set_xticks(x-(offset/2))

ax.set_xticklabels(es_labels,fontsize=fs)

plt.xlabel("Early stop gradient threshold",fontsize=fs)
xmin, xmax = ax.get_xlim()
plt.yticks(fontsize=fs)
plt.hlines(xmin=xmin,xmax=xmax, y=df_neb[colname].median(),
           linestyles='-', color='red',linewidth=lw,
           label='NEB median value')
plt.legend(fontsize=fs)


nosig_patch = mpatches.Patch(color='#E2DADB', label='no SIG')
yessig_patch = mpatches.Patch(color='#6D696A', label='yes SIG')
handles, labels = ax.get_legend_handles_labels()


handles.extend([nosig_patch, yessig_patch])
plt.legend(handles=handles,fontsize=fs)
# plt.ylim(0,7000)
# plt.ylim(0,40)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/noSIG_yesSIG_boxplot_farout.svg")
plt.show()
# -



plt.bar(x=es_labels,height=[len(df.dropna()) for df in yessig_yesmrs_dfs])
print([len(df.dropna()) for df in yessig_yesmrs_dfs])

plt.bar(x=es_labels,height=[len(df.dropna()) for df in yessig_dfs]+[len(df_neb.dropna())])
print([len(df.dropna()) for df in yessig_dfs])

plt.bar(x=es_labels,height=[len(df.dropna()) for df in nosig_dfs]+[len(df_neb.dropna())])
print([len(df.dropna()) for df in nosig_dfs])


def  get_success_rate(df):
    return len(df[df['success']==1])/len(df)


def get_failed(df):
    return df[df['success']==0]['reaction_name'].values


def get_tptn(df, reference_elem, reference_multi):
    elems =  df[df['n_rxn_steps']==1]['reaction_name']
    multis = df[df['n_rxn_steps']!=1]['reaction_name']
    tp = sum(elems.isin(reference_elem))
    fp = sum(~elems.isin(reference_elem))
    print(elems[~elems.isin(reference_elem)])

    tn = sum(multis.isin(reference_multi))
    fn = sum(~multis.isin(reference_multi))

    return np.array([[tp, fp], [fn, tn]])


h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Benzimidazolone-Synthesis-1-X-Iodine/ASNEB_5_yesSIG_yesMR")
# h2 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Claisen-Rearrangement-Aromatic/ASNEB_003_yesSIG/")

h.data.plot_convergence_metrics()

for df in yessig_yesmrs_dfs:
    df['TOTAL_GRAD_CALLS'] = df['n_grad_calls']+df['n_grad_calls_geoms']

pe_grads_nudged, springs = h.output_chain.pe_grads_spring_forces_nudged()

np.linalg.norm(springs), np.linalg.norm(pe_grads_nudged)

# +
grads = springs
rms_grads = []
for grad in grads:
    rms_gradient = np.sqrt(sum(np.square(grad.flatten())) / len(grad))
    rms_grads.append(rms_gradient)

max(rms_grads)

# +
grads = pe_grads_nudged
rms_grads = []
for grad in grads:
    rms_gradient = np.sqrt(sum(np.square(grad.flatten())) / len(grad))
    rms_grads.append(rms_gradient)

max(rms_grads)
# -

h.data.plot_convergence_metrics(1)

h.output_chain.plot_chain()

h2.output_chain.plot_chain()

# +

es_labels = ['0.5','0.1','0.05','0.03','0.01','no stop', 'yes MR']
df_list = yessig_dfs+[df_xtb_5_yessig_yesmr]
ind=0
df = yessig_dfs[-1]
elem_rxns = df[df['n_rxn_steps']==1]['reaction_name']
multi_rxns = df[df['n_rxn_steps']!=1]['reaction_name']


for ind, compared in enumerate(es_labels):
    fs=18
    f, ax = plt.subplots()
    # compared = es_labels[ind]


    tptn = get_tptn(df_list[ind], elem_rxns, multi_rxns)

    im = ax.imshow(tptn,alpha=.4)
    # Loop over data tptn and create text annotations.
    for i in range(len(tptn)):
        for j in range(len(tptn)):
            text = ax.text(j, i, tptn[i, j],
                           ha="center", va="center", color="black", fontsize=fs)

    xlabels = ['Single Step','Multi Step']
    ylabels = ['Pred. Single Step','Pred. Multi Step']
    plt.title(f'{compared} vs Reference',fontsize=fs)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(tptn)), labels=xlabels,fontsize=fs)
    ax.set_yticks(np.arange(len(tptn)), labels=ylabels,fontsize=fs)
    plt.show()
# -

yessig_dfs[-1]

# +
yessig_dfs[-1]

~yessig_dfs[-1]['reaction_name'][yessig_dfs[-1]['n_rxn_steps']==1].isin(elem_rxns)

# +
fs=18
s=8
f,ax = plt.subplots(figsize=(1*s, s))
offset=.5
lw = 3
es_labels = ['0.5','0.1','0.05','0.03','0.01','no stop']
bottom = np.zeros(len(es_labels))
xs = np.arange(len(es_labels))

elem_step_heights = [len(df[df['n_rxn_steps']==1].dropna()) for df in nosig_dfs]

multi_step_heights = [len(df[df['n_rxn_steps']!=1].dropna()) for df in nosig_dfs]


boxesnosig = plt.bar(x=xs-offset, height=elem_step_heights,
                     bottom=bottom, color='#FE5D9F',
                     label='Single step rxn (no SIG)',
                    width=offset-.1)
bottom+=elem_step_heights

boxesnosig = plt.bar(x=xs-offset, height=multi_step_heights,
                     bottom=bottom, color='#52FFEE',
                     label='Multi step rxn (no SIG)',
                    width=offset-.1)


###########################
bottom = np.zeros(len(es_labels))
elem_step_heights = [len(df[df['n_rxn_steps']==1].dropna()) for df in yessig_dfs]
multi_step_heights = [len(df[df['n_rxn_steps']!=1].dropna()) for df in yessig_dfs]


boxesnosig = plt.bar(x=xs, height=elem_step_heights,
                     bottom=bottom, label='Single step rxn (yes SIG)', color='#FE5D9F',
                    width=offset-.1, hatch='//')
bottom+=elem_step_heights

boxesnosig = plt.bar(x=xs, height=multi_step_heights,
                     bottom=bottom, label='Multi step rxn (yes SIG)', color='#52FFEE',
                    width=offset-.1,hatch='//')



ax.set_xticks(xs-(offset/2))
ax.set_xticklabels(es_labels,fontsize=fs)

plt.ylabel("Count",
           fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.xlabel("Early stop gradient threshold",fontsize=fs)
# -

# # Dataset creation

from pathlib import Path


def vis_nma(td, nma, dr=0.1):
    return Trajectory([td.displace_by_dr(-dr*nma), td, td.displace_by_dr(dr*nma)])


from neb_dynamics.NEB import NoneConvergedException
from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.Linesearch import Linesearch
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer as VPO

from neb_dynamics.NEB import NEB


def get_n_grad_calls(fp, xtb=False):
    if xtb:
        # output  = open(fp.parent / 'out_nomaxima_recycling.txt').read().splitlines()
        # output  = open(fp.parent / 'out_greedy').read().splitlines()
        output  = open(fp.parent / 'out_ASNEB_005_yesMR').read().splitlines()
        # output  = open(fp.parent / 'refined_grad_calls.txt').read().splitlines()
    else:
        output  = open(fp.parent / 'out_production_maxima_recycling').read().splitlines()
        # output  = open(fp.parent / 'out_mr_gi').read().splitlines()






    try:
        return int(output[-1].split()[2])
    except:
        print(output[-10:])

# all_rns = open("/home/jdep/T3D_data/msmep_draft/comparisons_dft/reactions_todo.txt").read().splitlines()
all_rns = open("/home/jdep/T3D_data/msmep_draft/comparisons/reactions_todo_xtb.txt").read().splitlines()
all_rns_xtb = open("/home/jdep/T3D_data/msmep_draft/comparisons/reactions_todo_xtb.txt").read().splitlines()


xtb_data_dir = Path(all_rns_xtb[0]).parent


def rn_succeeded(name, success_names):
    if name in success_names:
        return 1
    return 0


def mod_variable(var, all_rns, success_names):
    ind = 0
    mod_var = []
    for i, rn in enumerate(all_rns):
        if rn in success_names:
            mod_var.append(var[ind])
            ind+=1
        else:
            mod_var.append(None)

    return mod_var


from neb_dynamics.Refiner import Refiner


# +
def create_df(msmep_filename, v=True, refinement_results=False):

    do_refine = False
    multi = []
    elem = []
    failed = []
    skipped = []
    n_steps = []
    n_grad_calls = []
    n_grad_calls_geoms = []

    activation_ens = []

    n_opt_steps = []
    n_opt_splits = []

    success_names = []

    tsg_list = []
    sys_size = []
    xtb = True
    for i, rn in enumerate(all_rns):

        p = Path(rn) / msmep_filename
        if v: print(p)

        try:
            out = open(p.parent / f"out_{p.stem}").read().splitlines()
            # if 'Traceback (most recent call last):' in out[:50] or 'Terminated' in out:
            #     raise TypeError('failure')
            if 'Warning! A chain has electronic structure errors.                         Returning an unoptimized chain...' in out:
                raise FileNotFoundError('failure')

            if refinement_results:
                clean_fp = Path(str(p.resolve())+".xyz")
                ref = Refiner()
                ref_results = ref.read_leaves_from_disk(p)

            else:
                h = TreeNode.read_from_disk(p)
                clean_fp = p.parent / (str(p.stem)+'_clean.xyz')

            if clean_fp.exists():
                try:
                    out_chain = Chain.from_xyz(clean_fp, ChainInputs())

                except:
                    if v: print("\t\terror in energies. recomputing")
                    tr = Trajectory.from_xyz(clean_fp)
                    out_chain = Chain.from_traj(tr, ChainInputs())
                    grads = out_chain.gradients
                    if v: print(f"\t\twriting to {clean_fp}")
                    out_chain.write_to_disk(clean_fp)


                if not out_chain._energies_already_computed:
                    if v: print("\t\terror in energies. recomputing")
                    tr = Trajectory.from_xyz(clean_fp)
                    out_chain = Chain.from_traj(tr, ChainInputs())
                    grads = out_chain.gradients
                    if v: print(f"\t\twriting to {clean_fp}")
                    out_chain.write_to_disk(clean_fp)
            elif not clean_fp.exists() and refinement_results:
                print(clean_fp,' did not succeed')
                raise FileNotFoundError("boo")


            es = len(out_chain)==12
            if v: print('elem_step: ', es)
            if refinement_results:
                n_splits = sum([len(leaf.get_optimization_history()) for leaf in ref_results])
                tot_steps = sum([len(leaf.data.chain_trajectory) for leaf in ref_results])
                act_en = max([leaf.data.chain_trajectory[-1].get_eA_chain() for leaf in ref_results])

            else:
                if v: print([len(obj.chain_trajectory) for obj in h.get_optimization_history()])
                n_splits = len(h.get_optimization_history())
                if v: print(sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()]))
                tot_steps = sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()])
                act_en = max([leaf.data.chain_trajectory[-2].get_eA_chain() for leaf in h.ordered_leaves])

            n_opt_steps.append(tot_steps)
            n_opt_splits.append(n_splits)


            ng_line = [line for line in out if len(line)>3 and line[0]=='>']
            if v: print(ng_line)
            ng = sum([int(ngl.split()[2]) for ngl in ng_line])

            ng_geomopt = [line for line in out if len(line)>3 and line[0]=='<']
            ng_geomopt = sum([int(ngl.split()[2]) for ngl in ng_geomopt])
            # ng = 69
            if v: print(ng, ng_geomopt)

            activation_ens.append(act_en)

            tsg_list.append(out_chain.get_ts_guess())
            sys_size.append(len(out_chain[0].coords))



            if es:
                elem.append(p)
            else:
                multi.append(p)


            n = len(out_chain) / 12
            n_steps.append((i,n))
            success_names.append(rn)
            n_grad_calls.append(ng)
            n_grad_calls_geoms.append(ng_geomopt)

        except FileNotFoundError as e:
            if v: print(e)
            failed.append(p)

        except IndexError as e:
            if v: print(e)
            failed.append(p)

#         except KeyboardInterrupt:
#             skipped.append(p)


        if v: print("")

    import pandas as pd
    df = pd.DataFrame()
    df['reaction_name'] = [fp.split("/")[-1]for fp in all_rns]

    df['success'] = [rn_succeeded(fp, success_names) for fp in all_rns]

    df['n_grad_calls'] = mod_variable(n_grad_calls, all_rns, success_names)
    if v: print(n_grad_calls)
    if v: print(mod_variable(n_grad_calls, all_rns, success_names))

    df['n_grad_calls_geoms'] = mod_variable(n_grad_calls_geoms, all_rns, success_names)

    df["n_opt_splits"] = mod_variable(n_opt_splits, all_rns, success_names)
    # df["n_opt_splits"] = [0 for rn in all_rns]

    df['n_rxn_steps'] = mod_variable([x[1] for x in n_steps], all_rns, success_names)
    # df['n_rxn_steps'] = [0 for rn in all_rns]

    df['n_opt_steps'] = mod_variable(n_opt_steps, all_rns, success_names)

    df['file_path'] = all_rns

    df['activation_ens'] = mod_variable(activation_ens, all_rns, success_names)

    df['activation_ens'].plot(kind='hist')

    # df['n_opt_splits'].plot(kind='hist')
    # print(success_names)

    return df
# -

df = create_df("ASNEB_03_NOSIG_NOMR_v2_refined", refinement_results=True)

df.dropna()['n_grad_calls'].plot(kind='box')

len(df.dropna())



df.to_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_03_refinement.csv")

leaves = Refiner().read_leaves_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Bamberger-Rearrangement/ASNEB_03_NOSIG_NOMR_v2_refined"))
output = Refiner().join_output_leaves(leaves)

df.iloc[60]

df['activation_ens'].argmax()

# ### refinement

from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs
from nodes.Node import Node

from neb_dynamics.helper_functions import RMSD

from neb_dynamics.trajectory import Trajectory

# +
# rn = 'Lobry-de-Bruyn-Van-Ekenstein-Transformation'
# p = f"/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/production_msmep/"
do_refine = False
multi = []
elem = []
failed = []
skipped = []
n_steps = []
n_grad_calls = []

activation_ens = []

n_opt_steps = []
n_opt_splits = []

success_names = []

tsg_list = []
sys_size = []
xtb = True
for i, rn in enumerate(all_rns):

    p = Path(rn) / 'ASNEB_5_yesSIG'
    print(p.parent)



    # refine reactions
    refined_fp = Path(rn) / 'refined_results_5_yesSIG'
    print('Refinement done: ', refined_fp.exists())
    if refined_fp.exists():

        try:
            out = open(p.parent / "refined_grad_calls.txt").read().splitlines()
            # if out[-5][:3] == 'DIE' or 'ValueError: Error in compute_energy_gradient.' in out or 'IndexError: index 0 is out of bounds for axis 0 with size 0' in out:
            if 'Traceback (most recent call last):' in out[:50] or 'Terminated' in out:
                raise TypeError('failure')




            refiner = Refiner(cni=ChainInputs(k=0.1, delta_k=0.09,
                                                  node_class=Node3D_TC))

            leaves = refiner.read_leaves_from_disk(refined_fp)
            out_chain = refiner.join_output_leaves(leaves)




            es = len(out_chain)==12
            print('elem_step: ', es)
            opt_hist = []
            for leaf in leaves:
                opt_hist.append(leaf.data)


            print([len(obj.chain_trajectory) for obj in opt_hist])
            n_splits = len(opt_hist)
            print(sum([len(obj.chain_trajectory) for obj in opt_hist]))
            tot_steps = sum([len(obj.chain_trajectory) for obj in opt_hist])

            n_opt_steps.append(tot_steps)
            n_opt_splits.append(n_splits)
            ng = get_n_grad_calls(p, xtb=xtb)
            print(ng)
            n_grad_calls.append(ng)
            # activation_ens.append(max([leaf.data.optimized.get_eA_chain() for leaf in h.ordered_leaves]))
            activation_ens.append(max([leaf.data.chain_trajectory[-2].get_eA_chain() for leaf in leaves]))

            tsg_list.append(h.output_chain.get_ts_guess())
            sys_size.append(len(h.output_chain[0].coords))



            if es:
                elem.append(p)
            else:
                multi.append(p)


            n = len(out_chain) / 12
            n_steps.append((i,n))
            success_names.append(rn)
        except:
            failed.append(rn)
    else:
        failed.append(rn)

#     except FileNotFoundError:
#         failed.append(p)

#     except TypeError:
#         failed.append(p)

#     except KeyboardInterrupt:
#         skipped.append(p)


    print()
# -

#  ### non refinement

# +
# rn = 'Lobry-de-Bruyn-Van-Ekenstein-Transformation'
# p = f"/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/production_msmep/"
do_refine = False
multi = []
elem = []
failed = []
skipped = []
n_steps = []
n_grad_calls = []

activation_ens = []

n_opt_steps = []
n_opt_splits = []

success_names = []
n_steps = []

tsg_list = []
sys_size = []
xtb = True
for i, rn in enumerate(all_rns):
    # p = Path(rn) / 'production_vpo_tjm_xtb_preopt_msmep'
    p = Path(rn) / 'NEB_03_NOSIG_NOMR_neb'
    # p = Path(rn) / 'production_mr_gi_msmep'
    # p = Path(rn) / 'production_vpo_tjm_msmeap'
    print(p.parent)

    try:
        # out = open(p.parent / "out_production_maxima_recycling").read().splitlines()
        # out = open(p.parent / "out_nomaxima_recycling.txt").read().splitlines()
        out = open(p.parent / "out_NEB_03_NOSIG_NOMR").read().splitlines()
        # if out[-5][:3] == 'DIE' or 'ValueError: Error in compute_energy_gradient.' in out or 'IndexError: index 0 is out of bounds for axis 0 with size 0' in out:
        # if 'Traceback (most recent call last):' in out[:50] or 'Terminated' in out:
        #     raise TypeError('failure')

        neb = NEB.read_from_disk(p)

        tot_steps = len(neb.chain_trajectory)

        n_opt_steps.append(tot_steps)
        output  = open(p.parent / 'out_NEB_03_NOSIG_NOMR').read().splitlines()
        ng = [int(line.split()[2]) for line in output if '>>>' in line]
        print(ng)
        n_grad_calls.append(ng)
        activation_ens.append(neb.optimized.get_eA_chain())

        sys_size.append(len(neb.optimized[0].coords))

        success_names.append(rn)
        n_steps.append(len(neb.optimized)/12)

    except FileNotFoundError as e:
        print(e)
        failed.append(p)

    except TypeError as e:
        print(e)
        failed.append(p)

    except KeyboardInterrupt:
        skipped.append(p)


    print("")
# -

for label in ['5','1','05','03','01','005','0']:
# label = '03'
    df = create_df(f'ASNEB_{label}_NOSIG_NOMR_v2',v=False)

    df["TOTAL_GRAD_CALLS"] = df['n_grad_calls']+df['n_grad_calls_geoms']

    df[df['n_rxn_steps']>1]['TOTAL_GRAD_CALLS'].plot(kind='box')
    plt.title(label)
    plt.show()

    # df = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_05_yessig_yesmr.csv")
    df[['n_grad_calls','n_grad_calls_geoms']].plot(kind="box")
    # df['n_grad_calls_geoms'].plot(kind="box")

    df.to_csv(f"/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_{label}_nosig_nomr.csv")

# # Plots

# ### Initial Guess Comparison (ASNEB v NEB)

from neb_dynamics.Inputs import NEBInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

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
# +
# rn = 'Wittig'
# rn = 'Robinson-Gabriel-Synthesis'
# rn = 'Lobry-de-Bruyn-Van-Ekenstein-Transformation'
# rn = 'Bamford-Stevens-Reaction'
rn = 'Ramberg-Backlund-Reaction-Bromine'
# rn = 'Rupe-Rearrangement'

ref_p = Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_01_NOSIG_NOMR_v2/")
h = TreeNode.read_from_disk(ref_p, neb_parameters=nbi)
neb = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB_12_nodes_neb",
                         chain_parameters=ChainInputs(k=0.1, delta_k=0.09),
                        neb_parameters=nbi)
neb_long = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB_03_NOSIG_NOMR_neb", chain_parameters=ChainInputs(k=0.1, delta_k=0.09))
# neb_long = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB_005_noMR_neb")

# neb_long2 = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB36_neb")

ngc_asneb = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_{ref_p.stem}').read().splitlines()
ngc_asneb = sum([int(line.split()[2]) for line in ngc_asneb if '>>>' in line])


ngc_neb = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_12_nodes').read().splitlines()
ngc_neb = sum([int(line.split()[2]) for line in ngc_neb if '>>>' in line])

# ngc_neb_long = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_01_noSIG').read().splitlines()
ngc_neb_long = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_03_NOSIG_NOMR').read().splitlines()
ngc_neb_long = sum([int(line.split()[2]) for line in ngc_neb_long if '>>>' in line])

# ngc_neb_long2 = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_36imgs').read().splitlines()
# ngc_neb_long2 = sum([int(line.split()[2]) for line in ngc_neb_long2 if '>>>' in line])
# -

hdb = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Claisen-Rearrangement-Aromatic/ASNEB_03_NOSIG_NOMR_GI/")

hdb.output_chain.plot_chain()
hdb.output_chain.to_trajectory()

ref = Refiner()
leaves = ref.read_leaves_from_disk(Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_03_NOSIG_NOMR_v2_refined/"))
joined = ref.join_output_leaves(leaves)

# +
s=5
fs=18
f, ax = plt.subplots(figsize=(1*s, s))
plt.bar(x=['NEB(12)',f'NEB({len(neb_long.optimized)})','ASNEB'],height=[ngc_neb, ngc_neb_long, ngc_asneb], color='white',hatch='/', edgecolor='black')

plt.ylabel("Number of gradient calls",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_comparison_barplots.svg")
plt.show()

# +

self = neb
s = 8
fs = 18
ms = 5

all_chains = self.chain_trajectory


ens = np.array([c.energies-c.energies[0] for c in all_chains])
all_integrated_path_lengths = np.array([c.integrated_path_length for c in all_chains])
opt_step = np.array(list(range(len(all_chains))))
s=7
fs=18
ax = plt.figure(figsize=(1.16*s, s)).add_subplot(projection='3d')

# Plot a sin curve using the x and y axes.
x = opt_step
ys = all_integrated_path_lengths
zs = ens
for i, (xind, y) in enumerate(zip(x, ys)):
    if i == len(h.data.chain_trajectory):
        ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='red',markersize=ms, label='early stop chain')
    elif i < len(ys) -1:
        ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='gray',markersize=ms,alpha=.1)
    else:
        ax.plot([xind]*len(y), y, 'o-',zs=zs[i], color='blue',markersize=ms, label='optimized chain')
ax.grid(False)

ax.set_xlabel('optimization step', fontsize=fs)
ax.set_ylabel('integrated path length', fontsize=fs)
ax.set_zlabel('energy (hartrees)', fontsize=fs)

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-45)
plt.tight_layout()
plt.legend(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_early_stop_chain_traj.svg")
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
s=5
fs=18
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



f,ax = plt.subplots(figsize=(1.6*s, s))
plt.plot(avg_rms_gperp,color='blue') #, label='RMS Grad$_{\perp}$')
plt.plot(max_rms_gperp,color='orange')#, label='Max RMS Grad$_{\perp}$')
plt.plot(springs_g,color='purple')
# plt.plot(avg_rms_g, label='RMS Grad')
plt.plot(ts_gperp, color='green')#,label='TS gperp')
# plt.plot(inf_norm_g,label='Inf norm G')
# plt.plot(inf_norm_gperp,label='Inf norm Gperp')
# plt.ylabel("argmax(|Gradient$_{\perp}$|)", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

xmin= ax.get_xlim()[0]
xmax= ax.get_xlim()[1]
ymin= ax.get_ylim()[0]
ymax= ax.get_ylim()[1]
ax.hlines(y=self.parameters.rms_grad_thre, xmin=xmin, xmax=xmax, label='avg(RMS Grad$_{\perp}$)', linewidth=3, linestyle='--', color='blue')
ax.hlines(y=self.parameters.max_rms_grad_thre, xmin=xmin, xmax=xmax, label='max(RMS Grad$_{\perp}$)', linewidth=5,alpha=.8, linestyle='--', color='orange')
ax.hlines(y=self.parameters.ts_grad_thre, xmin=xmin, xmax=xmax, label='infnorm(TS Grad$_{\perp}$)', linewidth=3,linestyle='--', color='green')
ax.hlines(y=self.parameters.ts_spring_thre, xmin=xmin, xmax=xmax, label='infnorm(TS triplet Grad$_{spr}$)', linewidth=3,linestyle='--', color='purple')



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
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_early_stop_convergence.svg")
plt.show()



# +
s=5
fs=18
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

plt.plot(pl_h, h.output_chain.energies_kcalmol, 'o-', color='black',label='ASNEB')#, linewidth=3.3)
plt.plot(pl_neb, neb.optimized.energies_kcalmol, '^-', color='blue',label='NEB(12)')#, alpha=.3)
plt.plot(pl_neb_long, neb_long.optimized.energies_kcalmol, 'x-', color='red',label=f'NEB({len(neb_long.optimized)})')#, alpha=.3)

# plt.plot(pl_dft, joined.energies_kcalmol, 'o-',label='ASNEB_DFT')#, linewidth=3.3)

plt.legend(fontsize=fs)


plt.xlabel(xl,fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_comparison_paths.svg")
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
s=5
fs=18
f, ax = plt.subplots(figsize=(2*s, s))
plt.plot(h.output_chain.integrated_path_length, h.output_chain.energies_kcalmol, 'o-', color='black',label='ASNEB')
plt.plot(neb.initial_chain.integrated_path_length, neb.initial_chain.energies_kcalmol, '^--', color='blue',label='GI(12)')
plt.plot(neb_long.initial_chain.integrated_path_length, neb_long.initial_chain.energies_kcalmol, '^--', color='red',label='GI(24)')

colors = ['green','purple', 'gold', 'gray']
man_gis = recreate_gis(h.ordered_leaves)
last_val = 0
for i, (leaf, manual) in enumerate(zip(h.ordered_leaves, man_gis)):
    final_point = h.output_chain.integrated_path_length[12*i+11]
    start_point = h.output_chain.integrated_path_length[12*i]
    path_len_leaf = final_point - start_point

    plt.plot((manual.integrated_path_length*path_len_leaf)+start_point, manual.energies_kcalmol+last_val, 'o--', color=colors[i],label=f'GI leaf {i}')
    last_val = manual.energies_kcalmol[-1]
    # last_val = h.output_chain.path_length[12*i+11]


plt.legend(fontsize=fs)
plt.xlabel("Reaction progression",fontsize=fs)
plt.ylabel("Energies (kcal/mol)",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_comparison_initial_guesses.svg")
plt.show()
# -
# ## Comparison of mechanisms


from retropaths.helper_functions import pload

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


from neb_dynamics.Refiner import Refiner


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
    dft_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/ASNEB_03_NOSIG_NOMR')
    xtb_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_03_NOSIG_NOMR_v2')
    refine_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_03_NOSIG_NOMR_v2_refined')
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
        plt.plot(joined.path_length, joined.energies_kcalmol, 'o-', label='refinement')
    plt.legend()
    plt.show()


    out_trajs = [out_dft, out_xtb]
    if refinement_done:
        out_trajs.append(joined)

    if rp_rn:
        return rp_rn.draw(size=(200,200)), out_trajs
    else:
        return rp_rn, out_trajs

ind = 0

from IPython.core.display import display

df_sub[['?' in  val for val in df_sub['agrees?'].values]]

df_sub.loc[38]['experimental link']

rn = 'Bamford-Stevens-Reaction'
a, b = build_report(rn)
a

# +
c = b[2]
s=5
fs=18
f, ax = plt.subplots(figsize=(2*s, s))

plt.plot(c.integrated_path_length, c.energies_kcalmol, 'o-', color='black')

# plt.legend(fontsize=fs)


plt.xlabel('Reaction progression',fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/SI_{rn}_refined_path.svg")
plt.show()
Molecule.draw_list(get_mechanism_mols(c), mode='d3')
# -

c.to_trajectory().draw();

ref = Refiner()
leaves = ref.read_leaves_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Azaindole-Synthesis/ASNEB_03_NOSIG_NOMR_v2_refined/"))
joined = ref.join_output_leaves(leaves)

tsg = b[0].get_ts_guess()

tsg.tc_model_method = 'wb97xd3'
tsg.tc_model_basis = 'def2-svp'

tsg.tc_freq_calculation()

for ind in range(10):
    rn, trajs = build_report(all_rns[ind])
    display(rn)



# +
df = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_yessig.csv")

all_rns = df['reaction_name'].to_list()


# -

def report(df, rn):
    row = df[df['reaction_name']==rn]
    p = row['file_path'].values[0]
    p_ref = ∂
    return p


report(df, all_rns[0])

# ## Comparison of deployment strategies

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/1-2-Amide-Phthalamide-Synthesis/ASNEB_03_NOSIG_NOMR/")

h.output_chain.plot_chain()
h.output_chain.to_trajectory().draw();

df_precond = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons_dft/dataset_results_xtb_precondition.csv")
df_gi = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons_dft/dataset_results_gi.csv")
df_ref = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_refinement.csv")

df_ref['n_grad_calls'].median(), df_precond['n_grad_calls'].median(), df_gi['n_grad_calls'].median(),

# +
fs=18
s=7
f,ax = plt.subplots(figsize=(1*s, s))
offset=.5

lw = 3

xlabels = ['Refinement','XTB-Seed','GI-Seed']
x = np.arange( len(xlabels))
plt.boxplot(x=[
                df_ref.dropna()['n_grad_calls'],
                df_precond.dropna()['n_grad_calls'],
                df_gi.dropna()['n_grad_calls']],
           positions=x,
           medianprops={'linewidth':lw, 'color':'black'},
           boxprops={'linewidth':lw},
           capprops={'linewidth':lw-1},
           patch_artist=True)
# fill with colors
for patch in boxesnosig['boxes']:
    patch.set_facecolor('#E2DADB')






plt.ylabel("Gradient calls",
           fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

ax.set_xticks(x)
ax.set_xticklabels(xlabels,fontsize=fs)

plt.xlabel("Early stop gradient threshold",fontsize=fs)
xmin, xmax = ax.get_xlim()
plt.yticks(fontsize=fs)

plt.ylim(0,4000)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/noSIG_yesSIG_boxplot.svg")
plt.show()

# +
fs=18
s=7
f,ax = plt.subplots(figsize=(1*s, s))
offset=.5

lw = 3


xlabels = ['Refinement','XTB-Seed','GI-Seed']
bottom = np.zeros(len(xlabels))
x = np.arange( len(xlabels))
elem_step_heights = [
                len(df_ref[df_ref['n_rxn_steps']==1]),
                len(df_precond[df_precond['n_rxn_steps']==1]),
                len(df_gi[df_gi['n_rxn_steps']==1]),

                ]

multi_step_heights = [
                len(df_ref[df_ref['n_rxn_steps']!=1]),
                len(df_precond[df_precond['n_rxn_steps']!=1]),
                len(df_gi[df_gi['n_rxn_steps']!=1]),

                ]


plt.bar(x=xlabels, height=elem_step_heights,
                     bottom=bottom, color='#FE5D9F',
                     label='Single step rxn',
                    width=offset)

bottom+=elem_step_heights

plt.bar(x=xlabels, height=multi_step_heights,
                     bottom=bottom, color='#52FFEE',
                     label='Multi step rxn',
                    width=offset)




plt.ylabel("Count",
           fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)

ax.set_xticks(x)
ax.set_xticklabels(xlabels,fontsize=fs)

xmin, xmax = ax.get_xlim()
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/noSIG_yesSIG_boxplot.svg")
plt.show()
# -


