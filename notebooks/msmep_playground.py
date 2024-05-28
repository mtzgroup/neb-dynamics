# +
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.Inputs import ChainInputs, GIInputs
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from neb_dynamics.nodes.Node3D import Node3D
from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB

from neb_dynamics.NEB_TCDLF import NEB_TCDLF
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import NEBInputs
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
# from retropaths.abinitio.trajectory import Traj|ectory
# from retropaths.abinitio.tdstructure import TDStructure
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.helper_functions import create_friction_optimal_gi
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.helper_functions import RMSD
from neb_dynamics.NEB import NEB

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


import pandas as pd
from pathlib import Path
import numpy as np

from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

# # Database

# +

IGNORING_RXNS = ['Aza-Grob-Fragmentation-X-Bromine', 'Bamberger-Rearrangement']


# -

def subset_to_multi_step_rns(df, by_list=False, inp_list=None):
    if by_list:
        assert inp_list is not None, 'please input a list to subset by'
        return df[df['reaction_name'].isin(inp_list)]
    return df[df['n_rxn_steps']>1].reindex()


def sanitize(row):
    return row['reaction name'].replace("'","").replace("[","").replace("]","").replace(",","")


df_jan = pd.read_csv("/home/jdep/T3D_data/msmep_draft/msmep_reaction_successes.csv")
df_jan = df_jan.dropna()
df_jan['reaction name'] = df_jan.apply(sanitize, axis=1)
df_gi = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons_dft/dataset_results_gi.csv")


def subset_to_elem_step_rns(df, by_list=False, inp_list=None):
    if by_list:
        assert inp_list is not None, 'please input a list to subset by'
        return df[df['reaction_name'].isin(inp_list)]
    return df[df['n_rxn_steps']==1].reindex()


import pandas as pd

# +
df_xtb_5_yessig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_5_yessig.csv")

df_xtb_1_yessig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_1_yessig.csv")

df_xtb_05_yessig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_05_yessig.csv")


df_xtb_03_yessig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_03_yessig.csv")


df_xtb_01_yessig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_01_yessig.csv")


df_xtb_005_yessig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_005_yessig.csv")


df_xtb_003_yessig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_003_yessig.csv")



df_xtb_5_yessig_yesmr = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_5_yessig_yesmr.csv")
yessig_yesmrs_dfs = [pd.read_csv(f"/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_{label}_yessig_yesmr.csv").dropna() for label in ['5','1','05','01','005','0']]
# -


true_ms = []
for fp in df_xtb_003_yessig[df_xtb_003_yessig['n_rxn_steps']>1]['file_path']:
    p = Path(fp)/'ASNEB_003_yesSIG'
    h = TreeNode.read_from_disk(p)
    if len(h.ordered_leaves) > 1:
        true_ms.append(fp)

len(true_ms)

true_ms_names = [st.split("/")[-1] for st in true_ms]

df_jan_sub = df_jan[df_jan['reaction name'].isin(true_ms_names)]

rn = 'Wittig'
h = TreeNode.read_from_disk(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_5_yesSIG_yesMR')
clean_fp = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_5_yesSIG_yesMR_clean.xyz')
if clean_fp.exists():
    out_c = Chain.from_xyz(clean_fp, ChainInputs())
else:
    out_c = h.output_chain
# h = TreeNode.read_from_disk(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/production_maxima_recycling_msmep')

[len(obj.chain_trajectory) for obj in h.get_optimization_history()]

len(df_jan[df_jan['agrees?']=='yes*'])

# +
# yessig_dfs = [
#     df_xtb_5_yessig,
#     df_xtb_1_yessig,
#     df_xtb_05_yessig,

#     df_xtb_03_yessig,
# df_xtb_01_yessig,
# df_xtb_005_yessig,
# df_xtb_003_yessig
# ]
yessig_dfs = [
    df_xtb_5_yessig,
    df_xtb_1_yessig,
    df_xtb_05_yessig,

    df_xtb_03_yessig,
df_xtb_01_yessig,
df_xtb_003_yessig
]

[df.drop(df[df['reaction_name'].isin(IGNORING_RXNS)].index, inplace=True) for df in yessig_dfs]

multi_rns = subset_to_multi_step_rns(yessig_dfs[-1])['reaction_name'].values
# -

[df.drop(df[df['reaction_name'].isin(IGNORING_RXNS)].index, inplace=True) for df in yessig_yesmrs_dfs]

# +
# yessig_dfs = [subset_to_multi_step_rns(df, by_list=True, inp_list=multi_rns) for df in yessig_dfs]


for df in yessig_dfs:
    df['n_grad_calls'].replace(to_replace=0, value=np.nan, inplace=True)

# +
df_xtb_5_nosig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_5_nosig.csv")

df_xtb_1_nosig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_1_nosig.csv")

df_xtb_05_nosig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_05_nosig.csv")

df_xtb_03_nosig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_03_nosig.csv")

df_xtb_01_nosig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_01_nosig.csv")

df_xtb_005_nosig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_005_nosig.csv")

df_xtb_003_nosig = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_003_nosig.csv")


# +
# nosig_dfs = [
#     df_xtb_5_nosig,
#     df_xtb_1_nosig,
#     df_xtb_05_nosig,
#     df_xtb_03_nosig,
# df_xtb_01_nosig,
# df_xtb_005_nosig,
# df_xtb_003_nosig]

nosig_dfs = [
    df_xtb_5_nosig,
    df_xtb_1_nosig,
    df_xtb_05_nosig,
    df_xtb_03_nosig,
df_xtb_01_nosig,
df_xtb_003_nosig]


[df.drop(df[df['reaction_name'].isin(IGNORING_RXNS)].index, inplace=True) for df in nosig_dfs]
# nosig_dfs = [subset_to_multi_step_rns(df, by_list=True, inp_list=multi_rns) for df in nosig_dfs]
for df in nosig_dfs:
    df['n_grad_calls'].replace(to_replace=0, value=np.nan, inplace=True)

# +
df_neb = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_NEBS.csv")

df_neb.drop(df_neb[df_neb['reaction_name'].isin(IGNORING_RXNS)].index, inplace=True)

# df_neb = subset_to_multi_step_rns(df_neb, by_list=True, inp_list=multi_rns)

# +
# print(len(df_neb))
# -

def print_stats(df):
    print(f"median: {df['n_grad_calls'].median()}, mean: {df['n_grad_calls'].mean()}, std: {df['n_grad_calls'].std()}")


print_stats(df_neb)

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Claisen-Rearrangement-Aromatic/ASNEB_0_yesSIG_yesMR")





def steepest_descent(td, ss=.1, max_steps=10):
    tds = []
    last_td = td.copy()
    for i in range(max_steps):
        grad = last_td.gradient_xtb()
        new_coords = last_td.coords -1*ss*grad
        td_new = last_td.update_coords(new_coords)
        tds.append(td_new)
        last_td = td_new.copy()
    return Trajectory(tds)


huh = chain[ind_geom].tdstructure

h.data.optimized.plot_chain()

# +
ind_geom = 5
chain = h.data.optimized
# chain = h.ordered_leaves[0].data.optimized



unit_tans = chain.unit_tangents

correlations = []
correlations2 = []
correlations3 = []
# tr = chain[ind_geom].do_geom_opt_trajectory()[:10]
tr = steepest_descent(chain[ind_geom].tdstructure, max_steps=100, ss=1)
for i, _ in enumerate(tr):
    if i == 0:
        continue
    
    disp = tr[i].coords - tr[i-1].coords
    disp_unit = disp/np.linalg.norm(disp)
    
    tans = unit_tans[ind_geom]
    tans2 = unit_tans[ind_geom-1]
    tans3 = unit_tans[ind_geom+1]
    
    corr = np.dot(disp_unit.flatten(), tans.flatten())
    corr2 = np.dot(disp_unit.flatten(), tans2.flatten())
    corr3 = np.dot(disp_unit.flatten(), tans3.flatten())

    correlations.append(corr)
    correlations2.append(corr2)
    correlations3.append(corr3)
# -

plt.plot(correlations)
plt.plot(correlations2)
plt.plot(correlations3)

tr

ind = 1
disp =  tr5[ind].coords - tr5[ind-1].coords
tans = h.data.optimized.unit_tangents[7]

np.dot(disp.flatten()/np.linalg.norm(disp), tans.flatten())

# +
fs=18
s=8
f,ax = plt.subplots(figsize=(1*s, s))
offset=.5
# es_labels = ['0.5 no MR','0.5 yes MR', 'NEB']
es_labels = ['5','1','05','01','005','0']
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

boxesyessig = plt.boxplot(x=[df[colname].dropna() for df in yessig_yesmrs_dfs],
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
plt.ylim(0,6000)
# plt.ylim(0,40)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/noSIG_yesSIG_boxplot_farout.svg")
plt.show()

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

def animate_chain_trajectory(chain_traj, min_y=-100, max_y=100, 
                             max_x=1.1, min_x=-0.1, norm_path_len=True):
    # %matplotlib notebook
    import matplotlib.pyplot as plt
    import matplotlib.animation
    import numpy as np


    figsize = 5
    s=4

    fig, ax = plt.subplots(figsize=(1.618 * figsize, figsize))

    ax.set_xlim(min_x,max_x)
    ax.set_ylim(min_y, max_y)

    (line,) = ax.plot([], [], "o--", lw=3)


    def animate(chain):
            if norm_path_len:
                x = chain.integrated_path_length
            else:
                x = chain.path_length
                
            y = chain.energies_kcalmol



            color = 'lightblue'

            line.set_data(x, y)
            line.set_color("skyblue")




            return 

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=chain_traj)
    ani.save("/home/jdep/wittig_node2.gif")


    from IPython.display import HTML
    return HTML(ani.to_jshtml())

from pathlib import Path


def rn_succeeded(name, success_names):
    if name in success_names:
        return 1
    return 0


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

# ### refinement

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
    p = Path(rn) / 'NEB_005_noMR_neb'
    # p = Path(rn) / 'production_mr_gi_msmep'
    # p = Path(rn) / 'production_vpo_tjm_msmeap'
    print(p.parent)
    
    try:
        # out = open(p.parent / "out_production_maxima_recycling").read().splitlines()
        # out = open(p.parent / "out_nomaxima_recycling.txt").read().splitlines()
        out = open(p.parent / "out_NEB_005_noMR").read().splitlines()
        # if out[-5][:3] == 'DIE' or 'ValueError: Error in compute_energy_gradient.' in out or 'IndexError: index 0 is out of bounds for axis 0 with size 0' in out:
        if 'Traceback (most recent call last):' in out[:50] or 'Terminated' in out:
            raise TypeError('failure')
        
        neb = NEB.read_from_disk(p)
        
        tot_steps = len(neb.chain_trajectory) 
        
        n_opt_steps.append(tot_steps)
        output  = open(p.parent / 'out_NEB_005_noMR').read().splitlines()
        ng = int(output[-1].split()[2])
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



for i, rn in enumerate([all_rns[-2]]):
    
    p = Path(rn) / 'ASNEB_05_yesSIG'
    print(p)



    out = open(p.parent / f"out_{p.stem}").read().splitlines()
    # if 'Traceback (most recent call last):' in out[:50] or 'Terminated' in out:
    #     raise TypeError('failure')

    h = TreeNode.read_from_disk(p)
    clean_fp = p.parent / (str(p.stem)+'_clean.xyz')
    if clean_fp.exists():
        try:
            out_chain = Chain.from_xyz(clean_fp, ChainInputs())
        except:
            print("\t\terror in energies. recomputing")
            tr = Trajectory.from_xyz(clean_fp)
            out_chain = Chain.from_traj(tr, ChainInputs())
            grads = out_chain.gradients
            print(f"\t\twriting to {clean_fp}")
            out_chain.write_to_disk(clean_fp)


        if not out_chain._energies_already_computed:
            print("\t\terror in energies. recomputing")
            tr = Trajectory.from_xyz(clean_fp)
            out_chain = Chain.from_traj(tr, ChainInputs())
            grads = out_chain.gradients
            print(f"\t\twriting to {clean_fp}")
            out_chain.write_to_disk(clean_fp)
    else:
        out_chain = h.output_chain



    # refine reactions
    refined_fp = Path(rn) / 'refined_results'
    print('Refinement done: ', refined_fp.exists())



    es = len(out_chain)==12
    print('elem_step: ', es)
    print([len(obj.chain_trajectory) for obj in h.get_optimization_history()])
    n_splits = len(h.get_optimization_history())
    print(sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()]))
    tot_steps = sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()])

    n_opt_steps.append(tot_steps)
    n_opt_splits.append(n_splits)


    ng_line = [line for line in out if len(line)>3 and line[0]=='>']
    print(ng_line)
    ng = sum([int(ngl.split()[2]) for ngl in ng_line])
    # ng = 69
    print(ng)

    # activation_ens.append(max([leaf.data.optimized.get_eA_chain() for leaf in h.ordered_leaves]))
    activation_ens.append(max([leaf.data.chain_trajectory[-2].get_eA_chain() for leaf in h.ordered_leaves]))

    tsg_list.append(h.output_chain.get_ts_guess())
    sys_size.append(len(h.output_chain[0].coords))



    if es:
        elem.append(p)
    else:
        multi.append(p)


    n = len(out_chain) / 12
    n_steps.append((i,n))
    success_names.append(rn)
    n_grad_calls.append(ng)


def create_df(msmep_filename):

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
        print(p)

        try:
            
            out = open(p.parent / f"out_{p.stem}").read().splitlines()
            # if 'Traceback (most recent call last):' in out[:50] or 'Terminated' in out:
            #     raise TypeError('failure')

            h = TreeNode.read_from_disk(p)
            clean_fp = p.parent / (str(p.stem)+'_clean.xyz')
            if clean_fp.exists():
                try:
                    out_chain = Chain.from_xyz(clean_fp, ChainInputs())
                except:
                    print("\t\terror in energies. recomputing")
                    tr = Trajectory.from_xyz(clean_fp)
                    out_chain = Chain.from_traj(tr, ChainInputs())
                    grads = out_chain.gradients
                    print(f"\t\twriting to {clean_fp}")
                    out_chain.write_to_disk(clean_fp)


                if not out_chain._energies_already_computed:
                    print("\t\terror in energies. recomputing")
                    tr = Trajectory.from_xyz(clean_fp)
                    out_chain = Chain.from_traj(tr, ChainInputs())
                    grads = out_chain.gradients
                    print(f"\t\twriting to {clean_fp}")
                    out_chain.write_to_disk(clean_fp)
            else:
                out_chain = h.output_chain



            # refine reactions
            refined_fp = Path(rn) / 'refined_results'
            print('Refinement done: ', refined_fp.exists())

            if not refined_fp.exists() and do_refine:
                print("Refining...")

                refiner = Refiner(cni=ChainInputs(k=0.1, delta_k=0.09, 
                                                  node_class=Node3D_TC, 
                                                  node_conf_en_thre=1.5))
                refined_leaves = refiner.create_refined_leaves(h.ordered_leaves)
                refiner.write_leaves_to_disk(refined_fp, refined_leaves)


                tot_grad_calls = sum([leaf.get_num_grad_calls() for leaf in refined_leaves if leaf])
                print(f"Refinement took: {tot_grad_calls} calls")
                with open(refined_fp.parent/'refined_grad_calls.txt','w+') as f:
                    f.write(f"Refinement took: {tot_grad_calls} gradient calls")


            es = len(out_chain)==12
            print('elem_step: ', es)
            print([len(obj.chain_trajectory) for obj in h.get_optimization_history()])
            n_splits = len(h.get_optimization_history())
            print(sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()]))
            tot_steps = sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()])

            n_opt_steps.append(tot_steps)
            n_opt_splits.append(n_splits)


            ng_line = [line for line in out if len(line)>3 and line[0]=='>']
            print(ng_line)
            ng = sum([int(ngl.split()[2]) for ngl in ng_line])
            
            ng_geomopt = [line for line in out if len(line)>3 and line[0]=='<']
            ng_geomopt = sum([int(ngl.split()[2]) for ngl in ng_geomopt])
            # ng = 69
            print(ng, ng_geomopt)
            
            # activation_ens.append(max([leaf.data.optimized.get_eA_chain() for leaf in h.ordered_leaves]))
            activation_ens.append(max([leaf.data.chain_trajectory[-2].get_eA_chain() for leaf in h.ordered_leaves]))

            tsg_list.append(h.output_chain.get_ts_guess())
            sys_size.append(len(h.output_chain[0].coords))



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
            print(e)
            failed.append(p)

        except IndexError as e:
            print(e)
            failed.append(p)

        except KeyboardInterrupt:
            skipped.append(p)


        print("")
        
    import pandas as pd
    df = pd.DataFrame()
    df['reaction_name'] = [fp.split("/")[-1]for fp in all_rns]

    df['success'] = [rn_succeeded(fp, success_names) for fp in all_rns]

    df['n_grad_calls'] = mod_variable(n_grad_calls, all_rns, success_names)
    print(n_grad_calls)
    print(mod_variable(n_grad_calls, all_rns, success_names))
    
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

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Claisen-Rearrangement-Aromatic/ASNEB_5_yesSIG")

c = h.output_chain

tans = c.unit_tangents

gperps, gsprings = c.pe_grads_spring_forces_nudged()

len(tans)

len(gperps)

df = create_df('ASNEB_5_yesSIG_yesMR')

# df = pd.read_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_05_yessig_yesmr.csv")
df[['n_grad_calls','n_grad_calls_geoms']].plot(kind="box")
# df['n_grad_calls_geoms'].plot(kind="box")

df.to_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_5_yessig_yesmr.csv")

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Claisen-Rearrangement-Aromatic/ASNEB_5_yesSIG_yesMR/")
h2 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Claisen-Rearrangement-Aromatic/ASNEB_003_yesSIG/")

h2.output_chain[12].energy

h2.output_chain.plot_chain()

# +
from xtb.ase.calculator import XTB
from xtb.interface import Calculator, XTBException
from xtb.libxtb import VERBOSITY_MUTED
from xtb.utils import get_method

from ase.optimize import LBFGSLineSearch, BFGS, FIRE, LBFGS

# -

 def xtb_geom_optimization(self, return_traj=False, tol=0.1):
    from ase.io.trajectory import Trajectory as ASETraj
    from neb_dynamics.trajectory import Trajectory


    atoms = self.to_ASE_atoms()

    atoms.calc = XTB(method="GFN2-xTB", accuracy=0.001)
    opt = LBFGS(atoms, logfile=None, trajectory='logfile.traj')
    
    opt.run(fmax=tol)

    aT = ASETraj('logfile.traj')
    traj_list = []
    for i, _ in enumerate(aT):
        traj_list.append(
            TDStructure.from_ase_Atoms(
                aT[i], charge=self.charge, spinmult=self.spinmult
            )
        )
    traj = Trajectory(traj_list)
    traj.update_tc_parameters(self)

    if return_traj:
        print("len opt traj: ", len(traj))
        return traj
    else:
        return traj[-1]

tols = [0.5, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.0001]

results = []
for tol in tols:
    results.append(xtb_geom_optimization(h.output_chain[6].tdstructure, tol=tol))

grads = [np.linalg.norm(td.gradient_xtb()) for td in results]

ene_difs = [td.energy_xtb() - results[-1].energy_xtb() for td in results]



# + active=""
# (np.log(grads)[-2]-np.log(grads)[-5]) / (np.log(ene_difs)[-2]-np.log(ene_difs)[-5])
# -

grads[3] - grads[0]

np.log(ene_difs)

plt.plot(ene_difs)

grad = h.output_chain[12].tdstructure.gradient_xtb()

np.linalg.norm(grad)

(1000*np.linalg.norm(grad)**2)*627.5

np.linalg.norm(grad)

(h2.output_chain[12].energy - h.output_chain[12].energy)

opt_node = h.output_chain[11].do_geometry_optimization()
opt_node.energy

(-28.356670146641616 - -28.349385924750713)*627.5

h.output_chain[11].energy

plt.plot(h.output_chain.path_length, h.output_chain.energies)
plt.plot(h2.output_chain.path_length, h2.output_chain.energies)

(h.output_chain[12].energy -h2.output_chain[12].energy)*627.5

h.ordered_leaves[0].data.plot_opt_history(1)

h4.output_chain.plot_chain()

c = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Claisen-Rearrangement-Aromatic/debug_clean.xyz", ChainInputs())

c.to_trajectory()

h4.ordered_leaves[0].data.plot_opt_history(1)

h4.ordered_leaves[1].data.plot_opt_history(1)

plt.plot([c.ts_triplet_gspring_infnorm for c in h.ordered_leaves[0].data.chain_trajectory])
plt.plot([c.ts_triplet_gspring_infnorm for c in h3.ordered_leaves[0].data.chain_trajectory])
plt.plot([c.ts_triplet_gspring_infnorm for c in h4.data.chain_trajectory])


from neb_dynamics.constants import BOHR_TO_ANGSTROMS

0.002*5*BOHR_TO_ANGSTROMS

opt_chain = h3.ordered_leaves[0].data.optimized

np.amax(abs(opt_chain.get_g_perps()[opt_chain.energies.argmax()]))

h2.data.plot_convergence_metrics()

plt.plot([np.amax(abs(chain.get_g_perps())) for chain in h2.data.chain_trajectory])

plt.plot([chain.ts_triplet_gspring_infnorm for chain in h2.ordered_leaves[1].data.chain_trajectory])

# +

h2.ordered_leaves[1].data.plot_opt_history(1)
# -

import matplotlib.pyplot as plt

len(gperps)

np.amax(gsprings[6])

np.amax(gperps[6])

c.to_trajectory()

df['n_grad_calls'].argmax()

df.iloc[19]['file_path']

df.iloc[24]

df['n_opt_splits'].argmax()

h.ordered_leaves[0].data.optimized.plot_chain()

h.output_chain.to_trajectory()

df[df['reaction_name']=='Elimination-To-Form-Cyclopropanone-Bromine']

df.to_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_5_yessig_yesmr.csv")



import pandas as pd


def plot_dist(obj,xlabel='number gradient calls'):
    f, ax = plt.subplots()
    fs = 18
    plt.hist(obj, bins=None, density=False)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel("Count",fontsize=fs)
    plt.text(.65,.7,s=f'avg: {round(np.mean(obj),1)}\nmed: {round(np.median(obj),1)}\nstd: {round(np.std(obj),1)}',
            transform=ax.transAxes,fontsize=fs)

    plt.show()


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


# +
import pandas as pd
df = pd.DataFrame()
df['reaction_name'] = [fp.split("/")[-1]for fp in all_rns]

df['success'] = [rn_succeeded(fp, success_names) for fp in all_rns]

df['n_grad_calls'] = mod_variable(n_grad_calls, all_rns, success_names)

# df["n_opt_splits"] = mod_variable(n_opt_splits, all_rns, success_names)
df["n_opt_splits"] = [0 for rn in all_rns]

df['n_rxn_steps'] = mod_variable([x for x in n_steps], all_rns, success_names)
# df['n_rxn_steps'] = [0 for rn in all_rns]

df['n_opt_steps'] = mod_variable(n_opt_steps, all_rns, success_names)

df['file_path'] = all_rns

df['activation_ens'] = mod_variable(activation_ens, all_rns, success_names)

df['activation_ens'].plot(kind='hist')

df['n_grad_calls'].median(), df['n_grad_calls'].mean(), df['n_grad_calls'].std()

df['n_grad_calls'].median()

df['n_grad_calls'].plot(kind='hist')
# df['n_opt_splits'].plot(kind='hist')

# df.to_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_NEBS_12.csv")
# -

df['n_rxn_steps']

df.to_csv("/home/jdep/T3D_data/msmep_draft/comparisons/dataset_results_NEBS.csv")

df_xtbnomr

df_xtbyesmr['n_grad_calls'].median(),df_xtbyesmr['n_grad_calls'].std(), df_xtbyesmr['n_grad_calls'].mean()

df_xtbnomr['n_grad_calls'].median(),df_xtbnomr['n_grad_calls'].std(), df_xtbnomr['n_grad_calls'].mean()

# +
rn = 'Irreversable-Azo-Cope-Rearrangement'
data_dir = Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/")
# msmep_fp = data_dir / 'ASNEB_005_noMR'
msmep_fp = data_dir / 'ASNEB_005_noSIG'
msmep2_fp = data_dir / 'ASNEB_005_yesSIG'
clean_fp = (msmep2_fp.parent / (str(msmep2_fp.name)+"_clean.xyz"))
if clean_fp.exists():
    clean_msmep = Chain.from_xyz(clean_fp, ChainInputs())
else:
    clean_msmep = None

# msmep_fp = data_dir / 'debug_msmep'
# msmep2_fp = data_dir / 'debug_msmep'
neb_fp = data_dir / 'NEB_005_noMR_neb'
# neb_fp = data_dir / 'debug_neb_neb'
# -

h = TreeNode.read_from_disk(msmep_fp)
h2 = TreeNode.read_from_disk(msmep2_fp)

# +
# h.output_chain.to_trajectory()
# -

h2.output_chain.plot_chain()

h.output_chain.plot_chain()
len(h.output_chain)

if clean_msmep:
    print("Clean exists")
    clean_msmep.plot_chain()
else:
    h2.output_chain.plot_chain()

# +
output  = open(neb_fp.parent / 'out_NEB_005_noMR').read().splitlines()
grad_calls_neb = int(output[-1].split()[2])

# output  = open(msmep_fp.parent / 'out_ASNEB_005_noMR').read().splitlines()
output  = open(neb_fp.parent / 'out_greedy').read().splitlines()
grad_calls_msmep = int(output[-1].split()[2])

# output  = open(msmep2_fp.parent / 'out_debug').read().splitlines()
# grad_calls_msmep2 = int(output[-1].split()[2])

neb = NEB.read_from_disk(neb_fp)
msmep = TreeNode.read_from_disk(msmep_fp)
# msmep2 = TreeNode.read_from_disk(msmep2_fp)

# +
s=5
f, ax = plt.subplots(figsize=(2.5*s, s),ncols=2)
fs = 18
ax[0].plot(neb.optimized.path_length, neb.optimized.energies,'o-', label='NEB')
ax[0].plot(msmep.output_chain.path_length, msmep.output_chain.energies,'o-', label='ASNEB')
# ax[0].plot(msmep2.output_chain.path_length, msmep2.output_chain.energies,'o-', label='ASNEB2')
ax[0].legend(fontsize=fs)
ax[0].set_xticks(ax[0].get_xticks(),fontsize=fs)
ax[0].set_yticks(ax[0].get_yticks(), fontsize=fs)

ax[1].bar(x=['ASNEB','NEB'],height=[grad_calls_msmep, grad_calls_neb])
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.show()
# -

opt = neb.optimized[7].do_geometry_optimization()


opt.is_identical(neb.optimized[-1])

msmep.output_chain[-12].is_identical(neb.optimized[-1])

msmep.output_chain[-12].is_identical(neb.optimized[-1])


def plot_dist(obj,xlabel='number gradient calls'):
    f, ax = plt.subplots()
    fs = 18
    plt.hist(obj, bins=100, density=False)

    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel("Count",fontsize=fs)
    plt.text(.65,.7,s=f'avg: {round(np.mean(obj),1)}\nmed: {round(np.median(obj),1)}\nstd: {round(np.std(obj),1)}',
            transform=ax.transAxes,fontsize=fs)

    plt.show()


s=3
fs= 15
f,ax = plt.subplots(figsize=(1.618*s,s))
plt.hist(activation_ens_dft, label='dft')
plt.hist(activation_ens_xtb, label='xtb', alpha=.6)
plt.xlabel('Max activation energy', fontsize=fs)
plt.ylabel("Count", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.show()

# +
# h.data.optimized.to_trajectory()
# -

from neb_dynamics.Janitor import Janitor
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Oxazole-Synthesis/ASNEB_005_yesSIG/", chain_parameters=ChainInputs(use_maxima_recyling=False, skip_identical_graphs=False))
h2 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Oxazole-Synthesis/ASNEB_005_noSIG/", chain_parameters=ChainInputs(use_maxima_recyling=False, skip_identical_graphs=False))

c = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Oxazole-Synthesis/start_opt_msmep_clean.xyz", ChainInputs())

h.output_chain.plot_chain()
# h.output_chain.to_trajectory()

# +
# h2.output_chain.plot_chain()
# -

c.plot_chain()
c.to_trajectory().draw();

m = MSMEP(neb_inputs=NEBInputs(v=True), 
          chain_inputs=ChainInputs(k=0.1, delta_k=0.09, skip_identical_graphs=False, node_freezing=True, use_maxima_recyling=False), 
          gi_inputs=GIInputs(nimages=12), optimizer=VelocityProjectedOptimizer(timestep=0.5))


data_dir = Path(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/")
paths = list(data_dir.glob("system*"))

start = TDStructure.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system107/react.xyz")
end = TDStructure.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system107/prod.xyz")
sp = TDStructure.from_xyz("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system107/sp.xyz")

all_dists = []
en_dists = []
all_grad_calls = []
for path in paths:
    
    name = path.name
    data_fp = Path(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/{name}/asneb_dft_msmep/")
    if data_fp.exists():
        print(path.name)
        db = TreeNode.read_from_disk(data_fp, chain_parameters=ChainInputs(use_maxima_recyling=True, skip_identical_graphs=False))
        fp = open(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/{name}/out_greedy_dft").read().splitlines()
        grad_calls = int(fp[-1].split()[2])
        ts_correct = TDStructure.from_xyz(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/{name}/sp.xyz")

        tsg = db.output_chain.get_ts_guess()

        dist = RMSD(ts_correct.coords, tsg.coords)[0]
        en_dist = abs(ts_correct.energy_xtb() - tsg.energy_xtb())*627.5
        all_dists.append(dist)
        all_grad_calls.append(grad_calls)
        en_dists.append(en_dist)
    else:
        continue
        # print(f"***{data_fp} failed")
        # print()

sorted_list = sorted(list(enumerate(all_dists, start=0)), key=lambda x: x[1], reverse=True)
renamed_list = [(paths[val[0]].name,val[1]) for val in sorted_list]

sorted_list2 = sorted(list(enumerate(all_grad_calls, start=0)), key=lambda x: x[1], reverse=True)
renamed_list2 = [(paths[val[0]].name, val[1]) for val in sorted_list2]

plt.hist(all_grad_calls)

plt.hist(all_dists)

plt.hist(en_dists)

renamed_list2



ind=91
db = TreeNode.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system{ind}/asneb_dft_msmep/", chain_parameters=ChainInputs(use_maxima_recyling=True, skip_identical_graphs=False))
# db2 = TreeNode.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system{ind}/asneb2_xtb_msmep/", chain_parameters=ChainInputs(use_maxima_recyling=True, skip_identical_graphs=False))
fp = open(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system{ind}/out_greedy_dft").read().splitlines()
grad_calls = int(fp[-1].split()[2])
sp = TDStructure.from_xyz(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system{ind}/sp.xyz")
db.output_chain.plot_chain(norm_path=False)
db.output_chain.to_trajectory().draw();
# db2.output_chain.plot_chain()
# db2.output_chain.to_trajectory().draw();

min_guess = db.output_chain[12].tdstructure

min_guess.tc_model_method = 'uwb97xd3'
min_guess.tc_model_basis = 'def2-svp'

minimized = min_guess.tc_geom_optimization()

minimized

grad_calls

jj = Janitor(history_object=db, msmep_object=m, out_path=None)

db.data.plot_convergence_metrics()

output = jj.create_clean_msmep()

[t.get_num_grad_calls() for t in jj.cleanup_trees]

output.plot_chain()
output.to_trajectory().draw();

# +
# db2.output_chain.plot_chain()
# db2.output_chain.to_trajectory()
# -

db.output_chain.plot_chain()
db.output_chain.to_trajectory()

h2.output_chain.plot_chain()

success_names[28]

sorted(n_steps, key=lambda x:x[1], reverse=True)

s=3
fs= 15
f,ax = plt.subplots(figsize=(1.618*s,s))
plt.hist([x[1] for x in n_steps], label='dft')
plt.hist([x[1] for x in n_steps_xtb], label='xtb', alpha=.6)
# plt.hist(activation_ens_xtb, label='xtb', alpha=.6)
plt.xlabel('N steps', fontsize=fs)
plt.ylabel("Count", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.show()

sum([b>a for a,b in zip(n_opt_steps, n_opt_steps_refinement)]) / len(n_opt_steps)


list(enumerate(list(zip(n_opt_steps,n_opt_steps_refinement))))

leaves = ref.read_leaves_from_disk(Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Claisen-Rearrangement-Aromatic//"))

leaves[0].data.plot_opt_history(1)

leaves[1].get_num_opt_steps()

success_names[4]

len(failed)

failed


import retropaths.helper_functions as hf



reactions = hf.pload('/home/jdep/retropaths/data/reactions.p')



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


def build_report(rn):
    dft_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/production_vpo_tjm_xtb_preopt_msmep')
    xtb_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/production_vpo_tjm_msmep')
    refine_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/refined_results')
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

        joined = ref.join_output_leaves(refined_results)
    
    
    
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


just_rn_names = [Path(fp).stem for fp in all_rns]

from neb_dynamics.Refiner import Refiner

multi[1].parent.stem

ind=0
# report = build_report(just_rn_names[ind])
report = build_report(multi[ind].parent.stem)
report[0]

report[1][0].to_trajectory()

s=3
f,ax = plt.subplots(figsize=(1*s, 1*s))
plt.boxplot([n_opt_splits], labels=['n_splits'])
plt.show()
f,ax = plt.subplots(figsize=(1*s, 1*s))
plt.boxplot([n_opt_steps], labels=['n_steps'])
plt.show()

len(elem)

plt.hist(n_opt_steps)

plt.hist([x[1] for x in n_steps])
plt.ylabel("Count")
plt.xlabel("N steps")

list(enumerate(multi))

ind = 14
multi[ind]

# rn = 'Irreversable-Azo-Cope-Rearrangement'
rn = multi[ind]
# rn = elem[ind]
h = TreeNode.read_from_disk(rn)
# h = TreeNode.read_from_disk('/home/jdep/T3D_data/msmep_draft/comparisons/structures/Semi-Pinacol-Rearrangement-Alkene/production_vpo_tjm_msmep')
# h2 =  TreeNode.read_from_disk(rn.parent/ 'production_vpo_msmep/')
# h = TreeNode.read_from_disk(multi[ind].parent.resolve() / 'production_vpo_msmep/')
# h2 =  TreeNode.read_from_disk(multi[ind].parent.resolve() /'production_vpo4_msmep/')
print([len(obj.chain_trajectory) for obj in h.get_optimization_history()])
print(sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()]))
# print([len(obj.chain_trajectory) for obj in h2.get_optimization_history()])
# print(sum([len(obj.chain_trajectory) for obj in h2.get_optimization_history()]))#

h.output_chain.plot_chain()
h.output_chain.to_trajectory()

h.ordered_leaves[-1].data.optimized.plot_chain()

# %%time
p_opt = p_raw.tc_local_geom_optimization()



h = NEB.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Newman-Kwart-Rearrangement/production_vpo_tjm_msmep/node_0.xyz")

h.optimized.to_trajectory()





# %%time
r_opt = r_raw.tc_local_geom_optimization()

obj = h.ordered_leaves[1]
obj.data.plot_convergence_metrics(), obj.data.plot_opt_history(1)

h.output_chain.plot_chain()
h.output_chain.to_trajectory().draw();

all_data = list(Path('/home/jdep/T3D_data/msmep_draft/comparisons/structures/').iterdir())

st = TDStructure.from_xyz('/home/jdep/T3D_data/msmep_draft/comparisons/structures/Indole-Synthesis-1/start.xyz')
en = TDStructure.from_xyz('/home/jdep/T3D_data/msmep_draft/comparisons/structures/Indole-Synthesis-1/end.xyz')

# tr = h.data.initial_chain.to_trajectory()
tr = Trajectory([st, en]).run_geodesic(nimages=12)

h_ref = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Lobry-de-Bruyn-Van-Ekenstein-Transformation/production_vpo_tjm_xtb_preopt_msmep/")

# +
ref = tr[0]
ref.tc_model_method = 'gfn2xtb'
ref.tc_model_basis = 'gfn2xtb'

tr.update_tc_parameters(ref)

chain_local = Chain.from_traj(tr, parameters=cni_xtbloc)
chain_tc = Chain.from_traj(tr, parameters=cni_xtbtc)
# -

# %%time
try:
    n = NEB(initial_chain=chain_local, parameters=nbi, optimizer=optimizer)
    n.optimize_chain()
except NoneConvergedException:
    print("done")




# +
prog_input = chain_tc[4].tdstructure._prepare_input(method='gradient')

output = qcop.compute('terachem', prog_input, propagate_wfn=True, collect_files=True)
# -

output.results.energy

# %%time
try:
    n = NEB(initial_chain=chain_tc, parameters=nbi, optimizer=optimizer)
    n.optimize_chain()
except NoneConvergedException:J
    print("done")


def sort_td(td):
    c = td.coords
    symbols = td.symbols
    sorted_inds = np.argsort(td.symbols)
    coords_sorted = np.array(c[sorted_inds])
    symbols_sorted = symbols[sorted_inds]
    
    sorted_td =  TDStructure.from_coords_symbols(coords=coords_sorted, symbols=symbols_sorted, 
                                           tot_charge=td.charge, tot_spinmult=td.spinmult)
    
    return sorted_td


def permute_indices_td(td, element):
    td = sort_td(td)
    coords = td.coords
    symbols = td.symbols
    
    inds_element = np.nonzero(td.symbols == element)[0]
    permutation_element = np.random.permutation(inds_element)
    
    inds_original = np.arange(len(td.symbols))
    inds_permuted = inds_original.copy()
    inds_permuted[inds_element] = permutation_element
    
    coords_permuted = coords[inds_permuted]
    permuted_td = TDStructure.from_coords_symbols(coords_permuted, symbols)
    permuted_td.update_tc_parameters(td)
    return permuted_td


start = c[0].tdstructure

end = c[-1].tdstructure

start_shuffled = permute_indices_td(start, 'C')

tr = Trajectory([start_shuffled, end]).run_geodesic(nimages=12)

# +
nbi = NEBInputs(tol=0.01,
                early_stop_force_thre=0.03, 
                early_stop_chain_dist_thre=1, v=True, preopt_with_xtb=False, max_steps=500)

cni = ChainInputs(k=0.1, delta_k=0.09, node_class=Node3D, node_freezing=True)
gii = GIInputs(nimages=12)
optimizer = BFGS(bfgs_flush_steps=200, bfgs_flush_thre=0.80, use_linesearch=False, 
                 step_size=3, 
                 min_step_size= 0.1,
                 activation_tol=0.1
            )
# -

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=optimizer)

chain = Chain.from_traj(tr, cni)

h, out = m.find_mep_multistep(chain)

out.plot_chain()

# # Plots

# ### Initial Guess Comparison (ASNEB v NEB)

# +
rn = 'Robinson-Gabriel-Synthesis'

ref_p = Path(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_01_noSIG/")
h = TreeNode.read_from_disk(ref_p)
neb = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB_12_nodes_neb")
neb_long = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB_01_noSIG_neb")
# neb_long = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB_005_noMR_neb")

# neb_long2 = NEB.read_from_disk(f"/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/NEB36_neb")

ngc_asneb = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_{ref_p.stem}').read().splitlines()
ngc_asneb = sum([int(line.split()[2]) for line in ngc_asneb if '>>>' in line])


ngc_neb = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_12_nodes').read().splitlines()
ngc_neb = sum([int(line.split()[2]) for line in ngc_neb if '>>>' in line])

# ngc_neb_long = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_01_noSIG').read().splitlines()
ngc_neb_long = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_005_noMR').read().splitlines()
ngc_neb_long = sum([int(line.split()[2]) for line in ngc_neb_long if '>>>' in line])

# ngc_neb_long2 = open(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/out_NEB_36imgs').read().splitlines()
# ngc_neb_long2 = sum([int(line.split()[2]) for line in ngc_neb_long2 if '>>>' in line])

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



for ind in range(1, len(ct)):
    avg_rms_g.append(sum(ct[ind].rms_gradients[1:-1]) / (len(ct[ind])-2))
    avg_rms_gperp.append(sum(ct[ind].rms_gperps[1:-1]) / (len(ct[ind])-2))
    max_rms_gperp.append(max(ct[ind].rms_gperps))
    barr_height.append(abs(ct[ind].get_eA_chain() - ct[ind-1].get_eA_chain()))
    ts_node_ind = ct[ind].energies.argmax()
    ts_node_gperp = np.max(ct[ind].get_g_perps()[ts_node_ind])
    ts_gperp.append(ts_node_gperp)
    inf_norm_val_g = inf_norm_g.append(np.max(ct[ind].gradients))
    inf_norm_val_gperp = inf_norm_gperp.append(np.max(ct[ind].get_g_perps()))



f,ax = plt.subplots(figsize=(1.6*s, s))
# plt.plot(avg_rms_gperp, label='RMS Grad$_{\perp}$')
# plt.plot(max_rms_gperp, label='Max RMS Grad$_{\perp}$')
# plt.plot(avg_rms_g, label='RMS Grad')
plt.plot(ts_gperp,label='TS gperp')
# plt.plot(inf_norm_g,label='Inf norm G')
# plt.plot(inf_norm_gperp,label='Inf norm Gperp')
plt.ylabel("argmax(|Gradient$_{\perp}$|)", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

xmin= ax.get_xlim()[0]
xmax= ax.get_xlim()[1]
ymin= ax.get_ylim()[0]
ymax= ax.get_ylim()[1]
# ax.hlines(y=self.parameters.rms_grad_thre, xmin=xmin, xmax=xmax, label='rms_grad_thre', linestyle='--', color='blue')
# ax.hlines(y=self.parameters.max_rms_grad_thre, xmin=xmin, xmax=xmax, label='max_rms_grad_thre', linestyle='--', color='orange')
ax.hlines(y=self.parameters.ts_grad_thre, xmin=xmin, xmax=xmax, label='full convergence threshold', linewidth=3,linestyle='--', color='green')    



self.parameters.early_stop_force_thre = 0.5*BOHR_TO_ANGSTROMS


ax.hlines(y=self.parameters.early_stop_force_thre, xmin=xmin, xmax=xmax, label='early stop threshold', linestyle='--', linewidth=3, color='red')    


# ax.vlines(x=18, ymin=ymin, ymax=ymax, linestyle='--', color='red', label='early stop', linewidth=4)    

# ax2 = plt.twinx()
# plt.plot(barr_height, 'o--',label='barr_height_delta', color='purple')
# plt.ylabel("Barrier height data", fontsize=fs)

plt.yticks(fontsize=fs)


# ax2.hlines(y=self.pxarameters.barrier_thre, xmin=xmin, xmax=xmax, label='barrier_thre', linestyle='--', color='purple')
f.legend(fontsize=15, bbox_to_anchor=(1.35,.8))
plt.tight_layout()
plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_early_stop_convergence.svg")
plt.show()



# +
s=5
fs=18
f, ax = plt.subplots(figsize=(1.6*s, s))
normalize_pl = 0
if normalize_pl:
    pl_h = h.output_chain.integrated_path_length
    pl_neb = neb.optimized.integrated_path_length
    pl_neb_long = neb_long.optimized.integrated_path_length
    # pl_neb_long2 = neb_long2.optimized.integrated_path_length
    xl = "Normalized Path length"
else:
    pl_h = h.output_chain.path_length
    pl_neb = neb.optimized.path_length
    pl_neb_long = neb_long.optimized.path_length
    # pl_neb_long2 = neb_long2.optimized.path_length
    xl = "Path length"

plt.plot(pl_h, h.output_chain.energies_kcalmol, 'o-', color='black',label='ASNEB')
plt.plot(pl_neb, neb.optimized.energies_kcalmol, '^-', color='blue',label='NEB(12)')
plt.plot(pl_neb_long, neb_long.optimized.energies_kcalmol, 'x-', color='red',label=f'NEB({len(neb_long.optimized)})')
# plt.plot(pl_neb_long2, neb_long2.optimized.energies_kcalmol, '*-', color='green',label=f'NEB({len(neb_long2.optimized)})')

plt.legend(fontsize=fs)


plt.xlabel(xl,fontsize=fs)
plt.ylabel("Relative energies (kcal/mol)",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/{rn}_comparison_paths.svg")
plt.show()

# +
s=5
fs=18
f, ax = plt.subplots(figsize=(1.6*s, s))
normalize_pl = 0
if normalize_pl:
    pl_h = h.output_chain.integrated_path_length
    pl_neb = neb.optimized.integrated_path_length
    pl_neb_long = neb_long.optimized.integrated_path_length
    xl = "Normalized Path length"
else:
    pl_h = h.output_chain.path_length
    pl_neb = neb.optimized.path_length
    pl_neb_long = neb_long.optimized.path_length
    xl = "Path length"

plt.plot(pl_h, h.output_chain.energies_kcalmol, 'o-', color='black',label='ASNEB')
plt.plot(pl_neb, neb.optimized.energies_kcalmol, '^-', color='blue',label='NEB(12)')
plt.plot(pl_neb_long, neb_long.optimized.energies_kcalmol, 'x-', color='red',label=f'NEB({len(neb_long.optimized)})')

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
plt.plot(h.output_chain.path_length, h.output_chain.energies, 'o-', color='black',label='ASNEB')
plt.plot(neb.initial_chain.path_length, neb.initial_chain.energies, '^--', color='blue',label='GI(12)')
plt.plot(neb_long.initial_chain.path_length, neb_long.initial_chain.energies, '^--', color='red',label='GI(24)')

colors = ['green','purple', 'gold', 'gray']
man_cis = recreate_gis(h.ordered_leaves)
last_val = 0
for i, (leaf, manual) in enumerate(zip(h.ordered_leaves, man_cis)):
    # plt.plot(leaf.data.initial_chain.path_length+last_val, leaf.data.initial_chain.energies, 'o--', color=colors[i],label=f'GI leaf {i}')
    plt.plot(manual.path_length+last_val, manual.energies, 'o--', color=colors[i],label=f'GI leaf {i}')
    last_val = h.output_chain.path_length[12*i+11]


plt.legend(fontsize=fs)
plt.xlabel("Path length",fontsize=fs)
plt.ylabel("Energies (Hartree)",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
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


def build_report(rn):
    dft_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}/production_vpo_tjm_xtb_preopt_msmep')
    xtb_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/ASNEB_005_yesSIG')
    refine_path = Path(f'/home/jdep/T3D_data/msmep_draft/comparisons/structures/{rn}/refined_results')
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

        joined = ref.join_output_leaves(refined_results)
    
    
    
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

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Fries-Rearrangement-ortho/ASNEB_5_yesSIG/")

h.output_chain.plot_chain()
h.output_chain.to_trajectory()

a, b = build_report('Enolate-Claisen-Rearrangement')
a

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
