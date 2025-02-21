# -*- coding: utf-8 -*-
# +
from dataclasses import dataclass
from pathlib import Path

from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.chainhelpers import visualize_chain
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file

from neb_dynamics.neb import NEB
from neb_dynamics import StructureNode, Chain
from neb_dynamics.helper_functions import RMSD, get_fsm_tsg_from_chain
from neb_dynamics.inputs import RunInputs

from qcio import Structure, view, ProgramOutput
from neb_dynamics.helper_functions import compute_irc_chain
import neb_dynamics.chainhelpers as ch
import pandas as pd


from neb_dynamics.geodesic_interpolation.coord_utils import align_geom
import numpy as np

import matplotlib.pyplot as plt
# -
po = ProgramOutput.open("/home/jdep/T3D_data/fneb_draft/draft_data/system20/ub3lyp_321gs_tsres.qcio")


len(po.results.trajectory) - 66

# +

po.pstdout
# -

view.view(po)



# +

view.view(tsres)
# -

from neb_dynamics.pathminimizers.dimer import DimerMethod3D

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_manual/Semmler-Wolff-Reaction/wtf/")

tsreses = [ProgramOutput.open(fp) for fp in Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_manual/Semmler-Wolff-Reaction/").glob("wtf*tsres*")]

# +

view.view(tsreses[1])
# -

ch.visualize_chain(h.output_chain)

for i, g in enumerate(h.children[0].data.chain_trajectory[200].springgradients):
    print(i, np.dot(abs(g.flatten()), abs(g.flatten())) / len(g))

h.children[0].data.plot_opt_history(1)



visualize_chain(h.output_chain)

# +

# h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_manual/Rupe-Rearrangement/mep_output_msmep")
# h = TreeNode.read_from_disk("/u/jdep/click_fsm/mep_output_msmep_pair0", charge=2)
# sysns = []
# for i in range(1, 122):
# sysn = i
sysn = 97
start = Structure.open(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system{sysn}/react.xyz")
end = Structure.open(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system{sysn}/prod.xyz")
sp = Structure.open(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system{sysn}/sp.xyz")
sn = StructureNode(structure=start)
en = StructureNode(structure=end)
sysns.append(len(sp.geometry))
# -

from neb_dynamics.msmep import MSMEP

import pandas as pd

ri = RunInputs.open("/u/jdep/click_fsm/default_inputs.toml")

# +

gtols  = np.arange(0, 1.1, step=.1)

# +

DATA = []
gtols  = np.arange(0, 1, step=.1)
for gt in gtols:
    ri.path_min_inputs.ngradcalls = 20
    ri.chain_inputs.friction_optimal_gi = False
    ri.chain_inputs.k = 1
    ri.path_min_inputs.stepsize=1
    ri.path_min_inputs.grad_tol = gt
    ri.chain_inputs.node_rms_thre = 1
    ri.path_min_inputs.skip_identical_graphs = False
    
    m = MSMEP(ri)
    
    # h.data.initial_chain.parameters.k = 1
    # h2 = m.run_recursive_minimize(h.data.initial_chain)
    try:
        h2 = m.run_recursive_minimize(Chain.model_validate({"nodes":[sn, en], "parameters":ri.chain_inputs}))
    except Exception:
        DATA.append([gt]+[None]*2)
        continue
    try:
        tsres = ri.engine._compute_ts_result(h2.ordered_leaves[0].data.optimized.get_ts_node())
    except Exception as e:
        tsres = e.program_output
    hessres = ri.engine._compute_hessian_result(h2.ordered_leaves[0].data.optimized.get_ts_node(),use_bigchem=False)
    
    DATA.append([gt, h2, tsres, hessres ])
# -


all_cs = [d[1].output_chain for d in DATA if d[1]]

for i, d in enumerate(DATA):
    if d[1]:
        c = d[1].output_chain
        plt.plot(c.integrated_path_length, c.energies_kcalmol, 'o-', label=f"gtol:{DATA[i][0]}")
plt.legend()

ngcs = [d[1].data.grad_calls_made if d[1] else None for d in DATA ]

tsreses = [d[2] if d[2] else None for d in DATA ]

ngcs_tot = []
for a,b in zip(ngcs, tsreses):
    if b:
        if b.success:
            hesscalls = int(b.stdout.split("\n")[2].split("(")[1].split()[0])
            ngcs_tot.append(a+len(b.results.trajectory)-hesscalls)
            continue
    ngcs_tot.append(None)


nfreqs = [sum(np.array(d[-1].results.freqs_wavenumber) < 0) for d in DATA if d[1]]

nfreqs

plt.scatter(x=gtols, y=ngcs_tot)

ch.visualize_chain(DATA[-2][1].output_chain)

view.view(tsreses[-2])
# view.view(tsreses[-2])

gtols[np.argmin(ngcs_tot)]

ngcs

# +
# hessres = ri.engine._compute_hessian_result(StructureNode(structure=tsres.return_result), use_bigchem=False)
# hessres = ri.engine._compute_hessian_result(h2.ordered_leaves[0].data.optimized[2], use_bigchem=False)

# +
# hessres.results.freqs_wavenumber
# -

tsres.pstdout

view.view(tsres)
# view.view(hessres)

irc = compute_irc_chain(ts_node=StructureNode(structure=tsres.return_result), engine=ri.engine)

ch.visualize_chain(h2.ordered_leaves[0].data.optimized)
# ch.visualize_chain(irc)

view.view(h.output_chain[-1].structure, show_indices=True)

ch.visualize_chain(h.output_chain)

p = Path("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system1/")

allsys = [p.parent / sys for sys in list(p.parent.glob("sys*"))]

# +
name = 'xtb4_msmep'
data = []
failcount = 0
unsucc = []
for sysn in allsys:
    try:
        ngc = [float(l.split()[2]) for l in open(sysn / f'out_{name}').read().splitlines() if ">>>" in l][0]
        ngc_geom = [float(l.split()[2]) for l in open(sysn / f'out_{name}').read().splitlines() if "<<<" in l][0]
        ts_output = ProgramOutput.open(sysn / f'{name}_tsres_0.qcio')
        
        if not ts_output.success:
            print(f"*****{sysn}")
            dist = None
        else:
            sp = Structure.open(sysn / 'sp.xyz')
            spnode = StructureNode(structure=sp)
            # tsnode = StructureNode(structure=ts_output.return_result)
            # enes = RunInputs().engine.compute_energies([spnode, tsnode])
            # dist = abs(tsnode.energy - spnode.energy)*627.5
            dist = RMSD(sp.geometry, ts_output.return_result.geometry)[0]
            # print(sysn.stem, dist[0])
            

        data.append([str(sysn.stem), ngc, ngc_geom, dist])
    except Exception as e:
        print(e)
        failcount+=1
        print(sysn, ' failed')


# data = np.array(data)
# -

len(unsucc), failcount

import pandas as pd

df = pd.DataFrame(data, columns=['name', 'ngc','geom', 'ts_dist'])

df.iloc[:, 1].mean(), df.iloc[:, 1].std(), df.iloc[:, 1].mean() ,df.iloc[:, 1].max(), df.iloc[:, 1].min()

df['ts_dist'].plot(kind='hist')

df[df['ts_dist']>1.0]

inds = list(range(0, 125, 25))
eng = RunInputs().engine
THRE = 2.0
succeses = []
weird_failures = []
for p in allsys:
    # print("\n\n\n\n\n\n\n\n\n\n***\n\n\n\n\n\n\n\n\n\n", p)
    if not (p / "xtb4_msmep").exists(): continue
    try:
        h = TreeNode.read_from_disk(p / "xtb4_msmep")
        # h = TreeNode.read_from_disk(p / "deb")
        sp = Structure.open(p / 'sp.xyz')
        for i in inds:
        
            # tsres = eng._compute_ts_result(StructureNode(structure=sp))
            c = h.data.chain_trajectory[i]
            ts_ind = c.energies.argmax()
            # print(f"ts_gperp: {np.amax(abs(c.gperps[ts_ind]))} || ts_spring: {c.ts_triplet_gspring_infnorm} || rms: {sum(c.rms_gradients)/len(c)}")
            conditions = (p, i, np.amax(abs(c.gperps[ts_ind])), c.ts_triplet_gspring_infnorm, sum(c.rms_gradients)/len(c))
            
            try:
                tsres = eng._compute_ts_result(c.get_ts_node())
            except Exception as e:
                tsres = e.program_output
            if tsres.success:
                dist = RMSD(tsres.return_result.geometry, sp.geometry)[0]
        
                if dist < THRE:
                    succeses.append(conditions)
    except Exception:
        weird_failures.append((p, i))

reshape = {}
vals = [] 
for row in succeses:
    if row[0].stem in reshape.keys():
        continue
    else:
        reshape[row[0].stem] = row[1:]
        vals.append(row[2:])

dataframe = pd.DataFrame(vals, columns=['ts_gperp', 'ts_spring', 'rms'])

# +

dataframe['sums'] = [sum(v) for v in vals]
dataframe['avgs'] = [sum(v)/len(v) for v in vals]
dataframe['mins'] = [min(v) for v in vals]
dataframe['maxs'] = [max(v) for v in vals]
# -

dataframe['ts_gperp'].median(), dataframe['ts_spring'].median(), dataframe['rms'].median()

dataframe['ts_gperp'].plot(kind='kde')
dataframe['ts_spring'].plot(kind='kde')
dataframe['rms'].plot(kind='kde')
plt.vlines(x=0.05765000839220463, ymin=0, ymax=40, color='blue', linestyles='--')
plt.vlines(x=0.02301635371365334, ymin=0, ymax=40, color='orange', linestyles='--')
plt.vlines(x=0.02395753780581609, ymin=0, ymax=40, color='green', linestyles='--')
plt.legend()

dataframe['sums'].argmin(),  dataframe['mins'].argmin(), dataframe['maxs'].argmin()

n = NEB.read_from_disk('/home/jdep/for_kuke/bugfixed.xyz')

# pos = [ProgramOutput.open(lol) for lol in Path("/home/jdep/for_kuke/").glob("bugfixed*.qcio")]
pos = [node._cached_result for node in n.optimized]

for indx in range(len(pos)):
    lines = pos[indx].stdout.split()
    print([lines[i+1] for i, l in enumerate(lines) if "SQUARED" in l])

from neb_dynamics.inputs import RunInputs
ri = RunInputs.open("/home/jdep/for_kuke/toml2.toml")

for node in n.optimized:
    node._cached_energy = None



# +

ri.engine.compute_energies(n.optimized)
# -

n.optimized.plot_chain()

c = n.optimized.copy()
for node in c:
    node._cached_energy = None

ri = RunInputs.open("/home/jdep/for_kuke/toml2.toml")

# +

ri.engine.compute_energies(c)

# +

c.energies
# -

plt.plot(c.energies, label='truther')
plt.plot(n.optimized.energies, label='faker')
plt.legend()

opt_tr = ri.engine.compute_geometry_optimization(n.optimized[1])

(opt_tr[-1].energy-n.optimized[0].energy)*627.5

# +

view.view(opt_tr[-1].structure, n.optimized[0].structure, show_indices=True)
# -

ch.visualize_chain(opt_tr)

# h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/debug/wittig/fsm")
# fp = Path("/home/jdep/T3D_data/msmep_draft/pes/ch4/results/")
# name = 'pair_16'
# h = TreeNode.read_from_disk(fp / name)
# h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/debug/Claisen-Aromatic/mep_output_msmep/")
# name = 'Semmler-Wolff-Reaction'
# p = Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/') / name
name = 'system25'
p = Path('/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/') / name
# h = TreeNode.read_from_disk(p / "xtb4_msmep")
h = TreeNode.read_from_disk(p / "debug")
# h = TreeNode.read_from_disk(p / "dft_msmep")
# h = TreeNode.read_from_disk(p / "deb")x
sp = Structure.open(p / 'sp.xyz')
# h = TreeNode.read_from_disk(p / "ref")

h.data.plot_opt_history(1)

visualize_chain(h.output_chain)

h.data.plot_convergence_metrics()

# p = Path("/home/jdep/T3D_data/msmep_draft/debug/Claisen-Aromatic/")
# po_names = list(p.glob("xtb4_msmep_tsres*"))
po_names = list(p.glob("debug*tsres*"))
# po_names = list(p.glob("dft_msmep_tsres*"))
# po_names = list(Path("/home/jdep/T3D_data/msmep_draft/pes/ch4/results").glob("*.qcio"))
# po_names = list(p.glob("deb_tsres*"))
pos = [ProgramOutput.open(name) for name in po_names]

view.view(pos[0])

view.view(pos[0].return_result, sp)

eng = RunInputs().engine

# +

tsres = eng._compute_hessian_result(StructureNode(structure=pos[0].return_result))
# -

res = eng.compute_sd_irc(StructureNode(structure=pos[0].return_result))

pos = [ProgramOutput.open(x) for x in Path("/home/jdep/T3D_data/msmep_draft/pes/ch4/results/").glob("*tsres*.qcio") ]

tss = [po.return_result for po in pos if po.success]

unique = [tss[0]]
for structure in tss[1:]:
    dists = np.array([RMSD(structure.geometry, ref.geometry)[0] for ref in unique])
    if all(dists >0.1):
        unique.append(structure)


len(unique)

view.view(*unique)

hessres = eng._compute_hessian_result(StructureNode(structure=unique[-1]))

# +
# view.view(pos[0])
# view.view(pos[0].return_result, sp)

# +
# ts = Structure.open("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system1/sp.xyz")

# +
# view.view(po)
# -

visualize_chain(h.output_chain)

po1 = ProgramOutput.open("/home/jdep/T3D_data/msmep_draft/debug/Claisen-Aromatic/mep_output_msmep_tsres_0.qcio")
po2 = ProgramOutput.open("/home/jdep/T3D_data/msmep_draft/debug/Claisen-Aromatic/mep_output_msmep_tsres_1.qcio")

view.view(po1.return_result, po2.return_result)

# +
# visualize_chain(h.output_chain)
# -

dd = Path("/home/jdep/T3D_data/fneb_draft/benchmark/")

ind_guess = round(len(neb.optimized) / 2)
print(ind_guess)
ind_guesses = [ind_guess-1, ind_guess,ind_guess+1]
enes_guess = [neb.optimized.energies[ind_guesses[0]], neb.optimized.energies[ind_guesses[1]], neb.optimized.energies[ind_guesses[2]]]
ind_tsg = ind_guesses[np.argmax(enes_guess)]
print(ind_tsg)

# +

view.view(gi[0].structure, neb.optimized[0].structure)
# -

ri = RunInputs.open("/home/jdep/T3D_data/fneb_draft/benchmark/launch.toml")
eng = ri.engine
eng.program = 'terachem'
# eng = RunInputs().engine

from neb_dynamics.scripts.main_cli import pseuirc

# +
name = 'system25'
gradcallspernode = 5


print(name)
neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fneb.xyz")
# neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fneb_bugfix.xyz")
# neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fneb_linear.xyz")
# neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fneb_oldminima.xyz")
# neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/neb.xyz")
# neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/redofailed.xyz")
# neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/debug/reactant_neb.xyz")
sp = Structure.open(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/{name}/{name}/sp_terachem.xyz")
fsmqcio_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fsm_ts.qcio")
# fsmqcio_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fsm_ts_bugfix.qcio")
# fsmqcio_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fsm_ts_linear.qcio")
# fsmqcio_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/neb_ts.qcio")
# fsmqcio_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/debug/tsg.qcio")
fsmgi_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/gi_ts.qcio")
if not neb_fp.exists() or not fsmqcio_fp.exists() or not fsmgi_fp.exists():
    print('\tfailed! check this entry.')
neb = NEB.read_from_disk(neb_fp)
gi = Chain.from_xyz(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/gi.xyz", ChainInputs())
po_fsm = ProgramOutput.open(fsmqcio_fp)
po_gi = ProgramOutput.open(fsmgi_fp)
sp = Structure.open(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/{name}/{name}/sp_terachem.xyz")

ngc_fsm = ((len(neb.optimized)-2)*gradcallspernode + 2)  + len(po_fsm.results.trajectory)
ngc_gi = len(po_gi.results.trajectory)

if po_fsm.success:
    d_fsm, _  = RMSD(po_fsm.return_result.geometry, sp.geometry)
else:
    d_fsm = None
    
if po_gi.success:
    d_gi, _  = RMSD(po_gi.return_result.geometry, sp.geometry)
else:
    d_gi = None

# view.view(sp, po_fsm.return_result, po_gi.return_result, titles=['SP', f"FSM {round(d_fsm, 3)}", f"GI {round(d_gi, 3)}"])
# -


for node in neb.optimized:
    node._cached_energy = None

eng_true.compute_energies(neb.optimized)

sp_node = StructureNode(structure=sp)
eng_true.compute_energies([sp_node])

# +

from neb_dynamics.constants import BOHR_TO_ANGSTROMS
# -

sp.geometry

RMSD(get_fsm_tsg_from_chain(neb.optimized).coords, sp_node.coords)[0]*BOHR_TO_ANGSTROMS



[l for l in po_fsm.stdout.split('\n') if "Imaginary" in l]

(get_fsm_tsg_from_chain(neb.optimized).energy - sp_node.energy)*627.5

# +

visualize_chain(lolz.optimized)
# -

huh = ProgramOutput.open("/home/jdep/T3D_data/fneb_draft/benchmark/system25/fsm_ts_linear.qcio")
huh2 = ProgramOutput.open("/home/jdep/T3D_data/fneb_draft/benchmark/system25/fsm_ts.qcio")
lolz = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/benchmark/system25/fneb_linear.xyz")

[l for l in huh2.stdout.split("\n") if "Imaginary" in l]

[l for l in huh.stdout.split("\n") if "Imaginary" in l]

view.view(huh.input_data.structure, po_fsm.input_data.structure, sp)

[l for l in po_gi.stdout.split("\n") if "Imaginary" in l]

tsgs = [gi.get_ts_node().structure, neb.optimized.get_ts_node().structure, get_fsm_tsg_from_chain(lolz.optimized).structure]

for i, s in enumerate(tsgs):
    s.save(f"/home/jdep/neb_dynamics/notebooks/GM2024/data/tsg_{i}.xyz")




visualize_chain(lolz.optimized)

view.view(huh)

tsres = eng_true._compute_ts_result(gi.get_ts_node())

tsres.save("/home/jdep/T3D_data/fneb_draft/benchmark/system8/gi_ts.qcio")

view.view(po_fsm.return_result, sp)

visualize_chain(neb.optimized)

view.view(po_fsm)

# +

view.view(gi.get_ts_node().structure, neb.optimized.get_ts_node().structure, sp)

# +
# sub[['name','dE_fsm_geo']]

# +
# tsres = eng_true._compute_ts_result(get_fsm_tsg_from_chain(neb.optimized))
# tsres = RunInputs().engine._compute_ts_result(get_fsm_tsg_from_chain(neb.optimized))
# -

r = Structure.open("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system4/sp_terachem_minus.xyz")
p = Structure.open("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system4/sp_terachem_plus.xyz")

# view.view(po_fsm.return_result, sp)
view.view(neb.optimized[0].structure, neb.optimized[-1].structure, r, p)
# view.view(po_gi)

# +
# visualize_chain(neb.optimized)
# -

view.view(po_fsm.input_data.structure, sp)

for node in c:
    node._cached_energy = None

c = neb.optimized

eng_true.compute_energies(c)

visualize_chain(c)

a = Structure.open("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system2/react.xyz")
b = Structure.open("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system2/prod.xyz")
sp = Structure.open("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system2/sp.xyz")

eng = RunInputs.open("/home/jdep/T3D_data/fneb_draft/benchmark/launch.toml").engine

tsres = eng._compute_ts_result(neb.optimized.get_ts_node())

tsres.save("/home/jdep/T3D_data/fneb_draft/benchmark/system2/fsm_ts.qcio")

view.view(tsres.return_result, sp)

# +
# view.view(po_fsm)
# -

defeng = RunInputs().engine

# tsres = defeng._compute_ts_result(get_fsm_tsg_from_chain(neb.optimized))
tsres = defeng._compute_ts_result(neb.optimized.get_ts_node())

view.view(tsres.return_result, sp)p

# +

tsres2 = defeng._compute_ts_result(neb.optimized[8])
# -

view.view(tsres.return_result, sp)

visualize_chain(neb.optimized)


def get_projections(all_coords, eigvec, reference=None):
    # ind_ts = c.energies.argmax()
    ind_ts = 0
    # ind_ts = 8

    all_dists = []
    for i, coords in enumerate(all_coords):
        # _, aligned_ci = align_geom(all_coords[i], reference)
        if reference:
            _, aligned_start = align_geom(c[ind_ts].coords, reference)
        # displacement = aligned_ci.flatten() - c[ind_ts].coords.flatten()
            displacement = aligned_ci.flatten() - aligned_start.flatten()
        else:
            displacement = coords.flatten() - all_coords[ind_ts].flatten()
        # _, disp_aligned = align_geom(displacement.reshape(c[i].coords.shape), eigvec.reshape(c[i].coords.shape))
        all_dists.append(np.dot(displacement, eigvec))
        # all_dists.append(np.dot(disp_aligned.flatten(), eigvec))
                                     
    # plt.plot(all_dists)
    return all_dists


# +
# n0 = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/geo.xyz")
n0 = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/hcn/geo2.xyz")
# n0 = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/hcn/fsm.xyz")
# n0 = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/hcn/hellomartin.xyz")
# n1 = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/hcn/line.xyz")
n1 = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/hcn/line2.xyz")
n2 = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/hcn/neb.xyz")

lol = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/hcn/start_neb.xyz")
# -

from neb_dynamics.nodes.nodehelpers import displace_by_dr

# c_irc = Chain(nodes=tsmin_tr+tsplus_tr)
c_irc = Chain.from_xyz("/home/jdep/T3D_data/fneb_draft/animation/hcn/irc.xyz", ChainInputs())


def animate_trajectory(traj, xmin=-0.1, xmax=1.1, ymin=-1, ymax=200):
    import matplotlib.pyplot as plt
    import matplotlib.animation
    import numpy as np
    from IPython.display import HTML
    
    
    x = [list(chain.coordinates) for chain in traj]
    y = [list(chain.energies_kcalmol) for chain in traj]
    
    fig, ax = plt.subplots()
    fs = 18
    
    l, = ax.plot([],[], 'o-', label='fsm')
    rxn_coord = ch.get_rxn_coordinate(c_irc)
    # disps = np.array(get_projections(c_irc.coordinates, rxn_coord))
    disps = c_irc.integrated_path_length
    
    ax.plot(disps, c_irc.energies_kcalmol, '-', color='black', label='irc')
    ax.scatter([disps[501]], [max(c_irc.energies_kcalmol)], marker='x', color='black', s=50, label='TS')
    
    # ax.axis([0,1,0,100])
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    def animate(i):    
        disps = np.array(get_projections(x[i], rxn_coord))
        # l.set_data(disps, y[i])
        l.set_data(traj[i].integrated_path_length, traj[i].energies_kcalmol)
    
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(traj))
    plt.ylabel("Energies (kcal/mol)",fontsize=fs)
    plt.xlabel("Reaction coordinate",fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    return ani
    # return HTML(ani.to_jshtml())

# +


# visualize_chain(lol.optimized)
# -

def align_chain_to_node(chain, refnode):
    wtfnew_nodes = []
    for node in chain:
        _, lolz = align_geom(refgeom=refnode.coords, geom=node.coords)
        wtfnew_nodes.append(node.update_coords(lolz))
    return Chain(wtfnew_nodes)


new_ct = []
for c in n0.chain_trajectory:
    new_ct.append(align_chain_to_node(c, n0.optimized[-1]))

new_ct_line = []
for c in n1.chain_trajectory:
    new_ct_line.append(align_chain_to_node(c, n1.optimized[-1]))

eng = RunInputs().engine

# +

for c in new_ct_line:
    eng.compute_gradients(c)
# -

for c in new_ct:
    eng.compute_gradients(c)

c_irc = Chain(nodes=wtfnew_nodes)

defeng.compute_energies(c_irc)

# +
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import HTML

fs = 15
fig, ax = plt.subplots()


l, = ax.plot([],[], 'o-')
rxn_coord = ch.get_rxn_coordinate(c_irc)
disps = np.array(get_projections(c_irc.coordinates, rxn_coord))

# ax.plot(disps, c_irc.energies_kcalmol, '-', color='black')
ax.plot(c_irc.integrated_path_length, c_irc.energies_kcalmol, '-', color='black')
ax.scatter([c_irc.integrated_path_length[501]], [max(c_irc.energies_kcalmol)], marker='x', color='black', s=50, label='TS')
# ax.plot(get_projections(new_ct[1].coordinates, rxn_coord), new_ct[1].energies_kcalmol, 'o-', label='geodesic-growth')
# ax.plot(get_projections(new_ct_line[1].coordinates, rxn_coord), new_ct_line[1].energies_kcalmol, 'o-', label='linear-growth')
ax.plot(new_ct[1].integrated_path_length, new_ct[1].energies_kcalmol, 'o-', label='geodesic-growth')
ax.plot(new_ct_line[1].integrated_path_length, new_ct_line[1].energies_kcalmol, 'o-', label='linear-growth')

# ax.plot(get_projections(n0.optimized.coordinates, rxn_coord), n0.optimized.energies_kcalmol, 'o-', label='geodesic-converged')
# ax.plot(get_projections(n1.optimized.coordinates, rxn_coord), n1.optimized.energies_kcalmol, 'o-', label='linear-converged')
plt.legend(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Energies (kcal/mol)",fontsize=fs)
plt.xlabel("Rxn coordinate",fontsize=fs)
plt.yticks(fontsize=fs)
plt.show()
# -

fig.savefig("/home/jdep/neb_dynamics/notebooks/GM2024/data/comparison.png")

for c in new_ct_line:
    defeng.compute_energies(c)

for c in new_ct:
    defeng.compute_energies(c)

# huh = animate_trajectory(n0.chain_trajectory, ymax=500, xmin=-4, xmax=0.1, ymin=-30)
# huh = animate_trajectory(new_ct_line, ymax=1500, xmin=-0.1, xmax=1.1, ymin=-30)
huh = animate_trajectory(new_ct, ymax=200, xmin=-0.1, xmax=1.1, ymin=-30)
huh

high_ene_tr = []
for i, c in enumerate(new_ct_line):
    high_ene_tr.append((c[int(len(c)/2)-1], c[int(len(c)/2)], i))

for i in range(len(new_ct_line)):
    print(i)
    fs = 15
    fig, ax = plt.subplots()
    
    c = new_ct_line[i]
    l, = ax.plot([],[], 'o-')
    rxn_coord = ch.get_rxn_coordinate(c_irc)
    disps = np.array(get_projections(c_irc.coordinates, rxn_coord))
    
    # ax.plot(disps, c_irc.energies_kcalmol, '-', color='black')
    ax.plot(c_irc.integrated_path_length, c_irc.energies_kcalmol, '-', color='black')
    ax.scatter([c_irc.integrated_path_length[501]], [max(c_irc.energies_kcalmol)], marker='x', color='black', s=50, label='TS')
    # ax.plot(get_projections(new_ct[1].coordinates, rxn_coord), new_ct[1].energies_kcalmol, 'o-', label='geodesic-growth')
    # ax.plot(get_projections(new_ct_line[1].coordinates, rxn_coord), new_ct_line[1].energies_kcalmol, 'o-', label='linear-growth')
    ax.plot(c.integrated_path_length, c.energies_kcalmol, 'o-', label='linear-growth')
    
    # ax.plot(get_projections(n0.optimized.coordinates, rxn_coord), n0.optimized.energies_kcalmol, 'o-', label='geodesic-converged')
    # ax.plot(get_projections(n1.optimized.coordinates, rxn_coord), n1.optimized.energies_kcalmol, 'o-', label='linear-converged')
    plt.legend(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.ylabel("Energies (kcal/mol)",fontsize=fs)
    plt.xlabel("Rxn coordinate",fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.show()

ch.visualize_chain([n[1] for n in high_ene_tr])

huh.save("/home/jdep/T3D_data/fneb_draft/animation/geo.gif")
# huh.save("/home/jdep/T3D_data/fneb_draft/animation/line.gif")

tsres = defeng._compute_ts_result(get_fsm_tsg_from_chain(lol.optimized))

# +
lol = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/hcn/neb.xyz")
# irc = Chain.from_xyz("/home/jdep/T3D_data/fneb_draft/animation/hcn/irc.xyz", ChainInputs())
lol2 = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/animation/hcn/geo2.xyz")


sp = Structure.open("/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/system1/sp.xyz")

# -

gi = ch.run_geodesic([lol2.optimized[0], lol2.optimized[-1]], nimages=50)
defeng.compute_energies(gi)

irc = c_irc

len_list = [len(c) for c in lol2.chain_trajectory]

pivot_inds = []
prev_val = len_list[0]
for i, val in enumerate(len_list):
    if val != prev_val:
        prev_val = val
        pivot_inds.append(i)
pivot_inds


step = 1
i = pivot_inds[step]-1
plt.plot(lol2.chain_trajectory[0].integrated_path_length, lol2.chain_trajectory[0].energies_kcalmol,'o-')
plt.plot(lol2.chain_trajectory[i].integrated_path_length, lol2.chain_trajectory[i].energies_kcalmol,'o-')
print(len(lol2.chain_trajectory[i]))

# innleft = 1
innleft = step
gi2 = ch.run_geodesic([lol2.chain_trajectory[i][innleft], lol2.chain_trajectory[i][innleft+1]], nimages=10)
# gi2 = ch.run_geodesic([lol2.chain_trajectory[i][2], lol2.chain_trajectory[i][4]], nimages=10)

defeng = RunInputs().engine

defeng.compute_energies(gi2)

coords = np.linspace(lol2.optimized[0].coords, lol2.optimized[-1].coords, num=15)
line_nodes = [gi[0].update_coords(x) for x in coords]

line = Chain(line_nodes)

defeng.compute_energies(line)

line[7]._cached_energy = np.nan

# +


fs = 18

c_gi2 = Chain(nodes=lol2.chain_trajectory[i][:innleft]+gi2.nodes+lol2.chain_trajectory[i][innleft+1:], parameters=ChainInputs())
# c_gi2 = Chain(nodes=lol2.chain_trajectory[i][:2]+gi2.nodes+lol2.chain_trajectory[i][4+1:], parameters=ChainInputs())

defeng.compute_energies(c_gi2)

plt.plot(line.integrated_path_length, line.energies_kcalmol,'-', label='original linear')
plt.plot(gi.integrated_path_length, gi.energies_kcalmol,'-', label='original gi')
plt.plot(irc.integrated_path_length, irc.energies_kcalmol,'-', label='irc', color='black')

plt.plot(c_gi2.integrated_path_length, c_gi2.energies_kcalmol,'-', label=f'gi after step {step}', color='green')
plt.plot([c_gi2.integrated_path_length[1],c_gi2.integrated_path_length[-2]], [c_gi2.energies_kcalmol[1], c_gi2.energies_kcalmol[-2]],'^', color='green',ms=15)


plt.ylim(0,100)
plt.ylabel("Energies (kcal/mol)", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
# -

view.view(neb.optimized.get_ts_node().structure, get_fsm_tsg_from_chain(neb.optimized).structure, sp)

eng = RunInputs.open("/home/jdep/T3D_data/fneb_draft/benchmark/debug.toml").engine

tsres = eng._compute_ts_result(neb.optimized.get_ts_node())

# +
# tsres = eng._compute_ts_result(neb.optimized[3])

# +
# view.view(tsres, animate=False)
# -

# view.view(po_fsm.return_result, sp, po_gi.return_result)
view.view(po_fsm.return_result, sp)
# view.view(sp, neb.optimized.get_ts_node().structure)



visualize_chain(neb.optimized)

f,ax = plt.subplots()
fs=18
# N = len(df)
N = len(df[df['fsm_success']==1])
kcal1 = [sum(df['dE_fsm'] <= 1.0)/N, sum(df['dE_gi'] <= 1.0)/N]
kcal05 = [sum(df['dE_fsm'] <= 0.5)/N, sum(df['dE_gi'] <= 0.5)/N]
kcal01 = [sum(df['dE_fsm'] <= 0.1)/N, sum(df['dE_gi'] <= 0.1)/N]
labels=["FSM-GI","GI"]
plt.plot(labels, kcal1,'o-', label='∆=1.0(kcal/mol)')
plt.plot(labels, kcal05,'*-', label='∆=0.5(kcal/mol)')
plt.plot(labels, kcal01,'x-', label='∆=0.1(kcal/mol)')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.ylabel("Percent",fontsize=fs+5)
plt.show()

defaulteng = RunInputs(engine_name='ase').engine
eng_true = RunInputs.open("/home/jdep/T3D_data/fneb_draft/benchmark/launch.toml").engine
eng_true.program = 'terachem'
gradcallspernode = 5
data = []
# name = 'system5'
all_names = list(dd.glob("sys*"))
for fp in all_names:
    name = fp.stem
    print(name)
    
    # neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fneb.xyz")
    # neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fneb_bugfix.xyz")
    neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fneb_linear.xyz")
    # neb_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/neb.xyz")
    # fsmqcio_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fsm_ts.qcio")
    # fsmqcio_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fsm_ts_bugfix.qcio")
    fsmqcio_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/fsm_ts_linear.qcio")
    # fsmqcio_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/neb_ts.qcio")
    fsmgi_fp = Path(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/gi_ts.qcio")
    if not neb_fp.exists() or not fsmqcio_fp.exists() or not fsmgi_fp.exists():
        print('\tfailed! check this entry.')
        continue
    neb = NEB.read_from_disk(neb_fp)
    gi = Chain.from_xyz(f"/home/jdep/T3D_data/fneb_draft/benchmark/{name}/gi.xyz", ChainInputs())
    po_fsm = ProgramOutput.open(fsmqcio_fp)
    po_gi = ProgramOutput.open(fsmgi_fp)
    sp = Structure.open(f"/home/jdep/T3D_data/msmep_draft/comparisons_benchmark/{name}/{name}/sp_terachem.xyz")

    ngc_fsm = ((len(neb.optimized)-2)*gradcallspernode + 2)
    ngc_fsm_algo = ngc_fsm

    nimaginary_fsm = None
    nimaginary_gi = None
    if po_fsm.stdout:
        imgdata=[l for l in po_fsm.stdout.split('\n') if "Frequencies (cm^-1)" in l]
        if len(imgdata)>0:
            nimaginary_fsm = int(imgdata[0].split()[0])
    if po_gi.stdout:
        imgdata=[l for l in po_gi.stdout.split('\n') if "Frequencies (cm^-1)" in l]
        if len(imgdata)>0:
            nimaginary_gi = int(imgdata[0].split()[0])
    
    if po_fsm.success:
        d_fsm, _  = RMSD(po_fsm.return_result.geometry_angstrom, sp.geometry_angstrom)
        # sp_ene, spfsm_ene = eng_true.compute_energies([StructureNode(structure=sp), StructureNode(structure=po_fsm.return_result)])
        sp_ene, spfsm_ene = defaulteng.compute_energies([StructureNode(structure=sp), StructureNode(structure=po_fsm.return_result)])
        if "Valid Hessian" in po_fsm.stdout.split("\n")[0]:
            hesscalls = int(po_fsm.stdout.split("\n")[2].split("(")[1].split()[0])
        else:
            hesscalls = 0
        ngc_fsm += len(po_fsm.results.trajectory) - hesscalls
    else:
        d_fsm = None
        sp_ene, spfsm_ene = None, None
        
    if po_gi.success:
        d_gi, _  = RMSD(po_gi.return_result.geometry_angstrom, sp.geometry_angstrom)
        
        
        if "Valid Hessian" in po_gi.stdout.split("\n")[0]:
            hesscalls = int(po_gi.stdout.split("\n")[2].split("(")[1].split()[0])
        else:
            hesscalls = 0
        
        # spgi_ene = eng_true.compute_energies([StructureNode(structure=po_gi.return_result)])[0]
        spgi_ene = defaulteng.compute_energies([StructureNode(structure=po_gi.return_result)])[0]
        
        ngc_gi = len(po_gi.results.trajectory) - hesscalls
    else:
        spgi_ene = None
        d_gi = None
        ngc_gi = None

    row = [name, po_fsm.success, po_gi.success, d_fsm, d_gi, sp_ene, spfsm_ene, spgi_ene, ngc_fsm, ngc_gi, ngc_fsm_algo, nimaginary_fsm, nimaginary_gi]
    data.append(row)
    # view.view(sp, po_fsm.return_result, po_gi.return_result, titles=['SP', f"FSM {round(d_fsm, 3)}", f"GI {round(d_gi, 3)}"])

df = pd.DataFrame(data, columns=["name", "fsm_success", "gi_success", "d_fsm", "d_gi", "sp_ene", "spfsm_ene", "spgi_ene", "ngc_fsm", "ngc_gi", "ngc_fsm_algo", 'nimaginary_fsm', 'nimaginary_gi'])

df["dE_gi"] = abs(df['sp_ene'] - df["spgi_ene"])*627.5

df["dE_fsm"] = abs(df['sp_ene'] - df["spfsm_ene"])*627.5

fs = 18
df['nimaginary_fsm'].plot(kind='hist', label='fsm ts guess')
df['nimaginary_gi'].plot(kind='hist', label='gi ts guess', alpha=.7)
plt.ylabel("Count",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.xlabel("Number of imaginary modes", fontsize=fs)

fs = 18
bins_orig = plt.hist(df['dE_fsm'], label='fsm-gi')
df['dE_gi'].plot(kind='hist', label='gi', bins=bins_orig[1], alpha=.8)
# df_linear['dE_fsm'].plot(kind='hist', label='fsm-line', alpha=.7)
plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Count", fontsize=fs)
plt.xlabel("|E$_{TS_{ref}}$ - E$_{TS_{method}}$|",fontsize=fs)

# +

fs = 18
bins_orig = plt.hist(df3['nimaginary_fsm_geo'], label='fsm-gi')
df3['nimaginary_gi_geo'].plot(kind='hist', label='gi', bins=bins_orig[1], alpha=.8)
df3['nimaginary_fsm_line'].plot(kind='hist', label='fsm-line', alpha=.7, bins=bins_orig[1])
plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Count", fontsize=fs)
plt.xlabel("Number of imaginary modes of TS guess",fontsize=fs)
# -

fs = 18
bins_orig = plt.hist(df3['dE_fsm_geo'], label='fsm-gi')
df3['dE_gi_geo'].plot(kind='hist', label='gi', bins=bins_orig[1], alpha=.8)
df3['dE_fsm_line'].plot(kind='hist', label='fsm-line', alpha=.7, bins=bins_orig[1])
plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylabel("Count", fontsize=fs)
plt.xlabel("|E$_{TS_{ref}}$ - E$_{TS_{method}}$|",fontsize=fs)

# df.to_csv("./GM2024/data/dataset.csv",index=False)
df.to_csv("./GM2024/data/lineardf.csv",index=False)

df3['ngc_fsm_algo_line'].median(), df3['ngc_fsm_algo_line'].mean(), df3['ngc_fsm_algo_line'].std()

df3['ngc_fsm_algo_geo'].median(), df3['ngc_fsm_algo_geo'].mean(), df3['ngc_fsm_algo_geo'].std()

fs=18
# (df['ngc_fsm'] - df['ngc_fsm_algo']).mean(), (df['ngc_fsm'] - df['ngc_fsm_algo']).median()
# plt.violinplot(df['ngc_fsm_algo'], showmedians=True)
# plt.violinplot(positions=[2], dataset=df2['ngc_fsm_algo'], showmedians=True)
plt.violinplot(df3['ngc_fsm_algo_line'], showmedians=True)
plt.violinplot(positions=[2], dataset=df3['ngc_fsm_algo_geo'], showmedians=True)
plt.plot([2], [df3['ngc_fsm_algo_geo'].mean()],'o',color='black')
plt.plot([1], [df3['ngc_fsm_algo_line'].mean()],'o',color='black')
plt.xticks([1,2], labels=['linear','geodesic'],fontsize=fs)
plt.ylabel("Cost algorithm (gradient calls)",fontsize=fs)
# plt.ylim(0,200)
plt.yticks(fontsize=fs)

df3.iloc[72]

df3['ngc_fsm_algo_geo'].argmax()

# +

(df3['ngc_fsm_line'] - df3['ngc_fsm_algo_line']).median(), (df3['ngc_fsm_line'] - df3['ngc_fsm_algo_line']).mean(), (df3['ngc_fsm_line'] - df3['ngc_fsm_algo_line']).std()
# -

(df3['ngc_fsm_geo'] - df3['ngc_fsm_algo_geo']).median(), (df3['ngc_fsm_geo'] - df3['ngc_fsm_algo_geo']).mean(), (df3['ngc_fsm_geo'] - df3['ngc_fsm_algo_geo']).std()

fs=18
# (df['ngc_fsm'] - df['ngc_fsm_algo']).mean(), (df['ngc_fsm'] - df['ngc_fsm_algo']).median()
# plt.violinplot(df['ngc_fsm'] - df['ngc_fsm_algo'],showmedians=True)
# plt.violinplot(positions=[2], dataset=df2['ngc_fsm'] - df2['ngc_fsm_algo'],showmedians=True)
plt.violinplot(df3['ngc_fsm_line'] - df3['ngc_fsm_algo_line'],showmedians=True)
plt.violinplot(positions=[2], dataset=df3['ngc_fsm_geo'] - df3['ngc_fsm_algo_geo'],showmedians=True)
plt.xticks([1,2], labels=['linear','geodesic'],fontsize=fs)
plt.ylabel("Cost TS opt (gradient calls)",fontsize=fs)
# plt.ylim(0,200)
plt.yticks(fontsize=fs)

df3['ngc_fsm_line'].median(),df3['ngc_fsm_line'].mean(), df3['ngc_fsm_line'].std()

df3['ngc_fsm_geo'].median(),df3['ngc_fsm_geo'].mean(), df3['ngc_fsm_geo'].std()

(df['ngc_fsm'] - df['ngc_fsm_algo']).mean(), (df['ngc_fsm'] - df['ngc_fsm_algo']).median()

df.to_csv("./GM2024/data/lineardf.csv",index=False)

# +
# df2 = pd.read_csv("./GM2024/data/dataframe.csv")
df2 = pd.read_csv("./GM2024/data/dataset.csv")
# df2 = df

df = pd.read_csv("./GM2024/data/lineardf.csv")

# +

df['ngc_fsm_algo'].mean(), df['ngc_fsm_algo'].median()
# -

(df['ngc_fsm'] - df['ngc_fsm_algo']).mean(), (df['ngc_fsm'] - df['ngc_fsm_algo']).median()

df2['ngc_fsm_algo'].mean(), df2['ngc_fsm_algo'].median()

# +

(df2['ngc_fsm'] - df2['ngc_fsm_algo']).mean(), (df2['ngc_fsm'] - df2['ngc_fsm_algo']).median()
# -

df3 = pd.merge(left=df2, right=df, suffixes=("_geo","_line"), on='name')

# +
percents = [
    .70, 
    .85,
    len(df[df["gi_success"]==1]) / len(df),
    len(df[df["fsm_success"]==1]) / len(df),
    
]
labels = [
    "IDPP-TS*", 
    "NEB(0.01)-TS*",
    "GI-TS", 
    "FSM-TS"]
# -
# failednames = ['system114',
#  'system116',
#  'system13',
#  'system14',
#  'system17',
#  'system3',
#  'system31',
#  'system39',
#  'system49',
#  'system50',
#  'system58',
#  'system63',
#  'system74',
#  'system75',
#  'system84',
#  'system90',
#  'system97',
#  'system98',
#  'system99']
failednames=['system114', 'system116', 'system93', 'system97', 'system98', 'system99']

df3 = df3[~df3['name'].isin(failednames)]

df3[df3['dE_fsm_geo']>1.0][['dE_fsm_geo','name']]



sub_fail = df3[(df3['fsm_success_geo']==0)|(df3['dE_fsm_geo']>1.0)]
sub_nice = df3[(df3['gi_success_geo']==0)&(df3['dE_fsm_geo']<1.0)|(df3['dE_gi_geo']>1.0)]

# +

sub_nice[['name','dE_fsm_line', 'dE_fsm_geo','dE_gi_geo']]
# -

len(sub), len(df3)

with open("/home/jdep/T3D_data/fneb_draft/benchmark/todo.txt", "w+") as f:
    for name in sub['name'].values:
        f.write(f"{name}\n")

len(df3[df3["dE_fsm_line"]<0.1]) / len(df3), len(df3[df3["dE_fsm_geo"]<0.1]) / len(df3)

df3['ngc_fsm_geo'].median(), df3['ngc_fsm_line'].median()

# +

fs = 18
vals = [0.1, 0.5, 1.0]
symbols = ["o","x","^"]
for i, val in enumerate(vals):
    valfsm_line = len(df3[df3["dE_fsm_line"]<val]) / len(df3)
    valfsm_geo = len(df3[df3["dE_fsm_geo"]<val]) / len(df3)
    valgi = len(df3[df3["dE_gi_geo"]<val]) / len(df3)
    print(valfsm_line, valgi, valfsm_geo)
    plt.plot(["FSM(linear)","GI", "FSM(geo)"], [valfsm_line , valgi, valfsm_geo],f'{symbols[i]}-', ms=10, label=f"∆={val} kcal/mol")
plt.legend(fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel(f"Percent success (N={len(df3)})", fontsize=fs)
plt.show()
# -

df3['ngc_fsm_algo_geo'].argmax()

df3['ngc_fsm_algo_geo'].median(), df3['ngc_fsm_algo_geo'].mean()

# +

df3['ngc_fsm_geo'].median(), df3['ngc_fsm_algo_geo'].mean()
# -

from neb_dynamics.inputs import RunInputs

inputs = RunInputs(engine_name='ase',program='xtb', path_min_method='neb', chain_inputs={'friction_optimal_gi':True}, path_min_inputs={'path_resolution':200},
                  gi_inputs={'nimages':20})

from neb_dynamics import MSMEP

m = MSMEP(inputs=inputs)

from neb_dynamics.nodes.nodehelpers import create_pairs_from_smiles

a = 'CCl.[Br-]'
b = 'CBr.[Cl-]'


a_s, b_s = create_pairs_from_smiles(a,b)

from qcio import view

from neb_dynamics import StructureNode

react = StructureNode(structure=a_s)
prod = StructureNode(structure=b_s)

h = m.run_recursive_minimize(input_chain=[react, prod])

import neb_dynamics.chainhelpers as ch

ch.visualize_chain(h.output_chain)

ri = RunInputs.open("/home/jdep/debug/runfile.in")
ri

huh = RunInputs(engine='qcop', program='xtb',  path_min='fsm', path_min_kwds={'distance_metric':'linear'},)
huh

# +
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.neb import NEB
from neb_dynamics import ChainInputs
from neb_dynamics.chain import Chain
import neb_dynamics.chainhelpers as ch
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file
from neb_dynamics import StructureNode

import matplotlib.pyplot as plt
from pathlib import Path

from qcio import Structure, view, ProgramInput

from neb_dynamics.nodes.nodehelpers import displace_by_dr
from neb_dynamics.nodes.node import Node
from neb_dynamics import QCOPEngine



# +
import numpy as np

# Function to compute normalized gradient (reaction path direction) in non-mass-weighted coordinates
def compute_normalized_gradient(gradient):
    norm = np.linalg.norm(gradient)
    if norm == 0:
        raise ValueError("Zero gradient! Cannot normalize.")
    return gradient / norm

# Function to compute the geodesic correction
def geodesic_correction(geometry, new_geometry, step_direction, engine):
    """
    Apply geodesic correction to the new geometry to ensure that it
    follows the IRC tangent direction.
    
    :param geometry: Current geometry (numpy array)
    :param new_geometry: New geometry after taking a step
    :param step_direction: Direction of the previous step (normalized gradient)
    :return: Corrected geometry
    """
    # Compute the gradient at the new geometry
    new_gradient = engine.compute_gradients([new_geometry])[0]
    
    # Project the new gradient onto the previous step direction
    dot = np.dot(new_gradient.flatten(), step_direction.flatten()) 
    projection = dot * step_direction
    print(f"{dot=}")
    
    # Apply geodesic correction by subtracting the projection
    corrected_geometry_coords = new_geometry.coords - projection
    corrected_geometry = new_geometry.update_coords(corrected_geometry_coords)
    
    return corrected_geometry

# Function to take a single IRC step with geodesic correction
def irc_step_with_correction(geometry: Node, step_size, engine, prev_direction=None):
    # Compute gradient and normalized gradient (direction of motion)
    gradient = engine.compute_gradients([geometry])[0]
    direction = compute_normalized_gradient(gradient)
    
    # Take an initial IRC step along the normalized gradient
    new_geometry_coords = geometry.coords - step_size * direction
    new_geometry = geometry.update_coords(new_geometry_coords)

    
    # If there's a previous step direction, apply geodesic correction
    if prev_direction is not None:
        new_geometry = geodesic_correction(geometry=geometry, new_geometry=new_geometry, step_direction=prev_direction, engine=engine)
    
    return new_geometry, direction

# Schlegel-Gonzalez IRC integration algorithm with geodesic correction
def schlegel_gonzalez_irc_with_geodesic(geometry, engine, step_size=0.01, max_steps=100):
    """
    Integrates the IRC path using the Schlegel-Gonzalez algorithm
    with geodesic corrections in non-mass-weighted coordinates.

    :param geometry: Initial geometry (coordinates) as a numpy array
    :param step_size: Step size for each IRC integration step
    :param max_steps: Maximum number of steps to integrate
    :return: IRC path as a list of geometries
    """
    irc_path = [geometry]
    prev_direction = None

    for step in range(max_steps):
        # Take an IRC step with geodesic correction
        geometry, prev_direction = irc_step_with_correction(geometry=geometry, step_size=step_size, prev_direction=prev_direction, engine=engine)
        engine.compute_gradients([geometry])
        
        # Append the new geometry to the IRC path
        irc_path.append(geometry)
        
        # Optional: Add convergence check or re-adjust step size if necessary
        if np.linalg.norm(geometry.gradient) < 1e-5:
            print(f"Converged after {step+1} steps.")
            break

    return irc_path


# -

def load_qchem_result(path_dir):
    
    path_dir = Path(path_dir)
    path_string = path_dir / 'stringfile.txt'
    structs = read_multiple_structure_from_file(path_string)
    nodes = [StructureNode(structure=s) for s in structs]
    enes_file = path_dir / 'Vfile.txt'
    enes_data = open(enes_file).read().splitlines()[1:]
    enes = [float(line.split()[-2]) for line in enes_data]
    for node, ene in zip(nodes, enes):
        node._cached_energy = ene
    chain = Chain(nodes=nodes) 
    return chain


hcn = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/hcn/fsm_short_neb.xyz")
hcn_linear = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/hcn/fsm_linear_short_neb.xyz")
hcn_qchem = load_qchem_result("/home/jdep/T3D_data/fneb_draft/hcn/fsm.files.1/")

plt.plot(hcn_linear.optimized.energies)
plt.plot(hcn.optimized.energies)

# +
# ch.visualize_chain(hcn_qchem)
# -

ts_hcn_res = eng._compute_ts_result(hcn.optimized.get_ts_node())

ts_hcn_qchem_res = eng._compute_ts_result(hcn_qchem.get_ts_node())

# +
# view.view(ts_hcn_res)
# view.view(ts_hcn_qchem_res)

# +

fsm = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/wittig/fsm_short_geodesic_neb.xyz")
qchem = load_qchem_result("/home/jdep/T3D_data/fneb_draft/wittig/fsm_short.files.1/")
# -

fsm_long = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/claisen/fsm_geodesic_neb.xyz")
qchem_long = load_qchem_result("/home/jdep/T3D_data/fneb_draft/claisen/fsm.files.1/")

plt.plot(fsm.optimized.integrated_path_length, fsm.optimized.energies, 'o-')
plt.plot(qchem.integrated_path_length, qchem.energies, 'o-')

eng = QCOPEngine(program
                 _input=ProgramInput(structure=fsm.optimized.get_ts_node().structure, model={'method':'hf','basis':'6-31g'}, calctype='energy'), compute_program='qcop', program='terachem-pbs')

ts = eng.compute_transition_state(fsm.optimized.get_ts_node())

ts_hcn_res.return_result.save("/home/jdep/T3D_data/fneb_draft/hcn/ts_hf.xyz")

# +
c_backwards = Chain.from_xyz("/home/jdep/T3D_data/fneb_draft/hcn/irc_negative.xyz", ChainInputs())

c_forward = Chain.from_xyz("/home/jdep/T3D_data/fneb_draft/hcn/irc_forward.xyz", ChainInputs())

c_backwards.nodes.reverse()

c_irc = Chain(c_backwards.nodes+[ts_node]+c_forward.nodes)

eng.compute_energies(c_irc)
# -

ts_qchem = eng.compute_transition_state(qchem.get_ts_node())

ts.save("/home/jdep/T3D_data/fneb_draft/claisen/ts_hf.xyz")

ts_node = StructureNode(structure=ts)

ts_node = StructureNode(structure=ts_hcn_res.return_result)
eng.compute_energies([ts_node])


outputs = build_full_irc_chain(ts_node, engine=eng, dr=0.01, step_size=0.001)

rxn_coordinate = ch.get_rxn_coordinate(c_irc)

hcn_qchem[0].energy, hcn.optimized[0].energy, c_irc[0].energy

c_irc[0].structure.save("/home/jdep/T3D_data/fneb_draft/hcn/hcn_reactant.xyz")

c_irc[-1].structure.save("/home/jdep/T3D_data/fneb_draft/hcn/hcn_product.xyz")

c_fsm_linear = hcn_linear.optimized

c_fsm = hcn.optimized
c_qchem = hcn_qchem



def get_projections(c: Chain, eigvec, reference):
    # ind_ts = c.energies.argmax()
    ind_ts = 0

    all_dists = []
    for i, node in enumerate(c):
        # _, aligned_ci = align_geom(c[i].coords, reference)
        # _, aligned_start = align_geom(c[ind_ts].coords, reference)
        # displacement = aligned_ci.flatten() - c[ind_ts].coords.flatten()
        # displacement = aligned_ci.flatten() - aligned_start.flatten()
        displacement = c[i].coords.flatten() - c[ind_ts].coords.flatten()
        # _, disp_aligned = align_geom(displacement.reshape(c[i].coords.shape), eigvec.reshape(c[i].coords.shape))
        all_dists.append(np.dot(displacement, eigvec))
        # all_dists.append(np.dot(disp_aligned.flatten(), eigvec))
                                     
    # plt.plot(all_dists)
    return all_dists


dists_fsm_linear = get_projections(c_fsm_linear, rxn_coordinate, c_irc[0].coords)
dists_fsm = get_projections(c_fsm, rxn_coordinate, c_irc[0].coords)
dists_qchem = get_projections(c_qchem, rxn_coordinate, c_irc[0].coords)
dists_irc = get_projections(c_irc, rxn_coordinate, c_irc[0].coords)

from neb_dynamics.neb import NEB
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics import NEBInputs

# +
# n = NEB(initial_chain=c_fsm, optimizer=VelocityProjectedOptimizer(timestep=0.5), parameters=NEBInputs(v=True, max_steps=100), engine=eng)

# +
# es_res = n.optimize_chain()

# +
# dists_neb = ch.get_projections(n.optimized, rxn_coordinate)

# +
# n.grad_calls_made
# -





# +
s=6
fs=18
f, ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(np.array(dists_fsm), c_fsm.energies_kcalmol, 'o-', label='fsm')
plt.plot(np.array(dists_fsm_linear), c_fsm_linear.energies_kcalmol, 'o-', label='fsm(linear)')
# plt.plot(np.array(dists_qchem), c_qchem.energies_kcalmol, 'o-', label='qchem')
plt.plot(np.array(dists_irc), c_irc.energies_kcalmol, label='irc', color='black', lw=3)

# plt.plot(c_fsm.integrated_path_length, c_fsm.energies_kcalmol, 'o-', label='fsm')
# plt.plot(c_qchem.integrated_path_length, c_qchem.energies_kcalmol, 'o-', label='qchem')
# plt.plot(c_irc.integrated_path_length, c_irc.energies_kcalmol, label='irc', color='black', lw=3)
plt.ylabel("Energies (kcal/mol)", fontsize=fs)
plt.xlabel("Reaction coordinate", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
# -

len(c_fsm_linear)

ch.visualize_chain(hcn.optimized)

hcn_qchem.coordinates[-1]

ch.visualize_chain(c_irc)

c_irc.energies

# +

hcn_qchem.energies
# -

ch.visualize_chain(hcn_qchem)

wittig = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/wittig/fsm_geodesic_neb.xyz")
wittig_qchem = load_qchem_result("/home/jdep/T3D_data/fneb_draft/wittig/fsm.files.2/")

# +

# ch.visualize_chain(wittig.optimized)
ch.visualize_chain(wittig_qchem)
# -

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb/Wittig/")

plt.plot(wittig.optimized.integrated_path_length, wittig.optimized.energies,'o-')
plt.plot(wittig_qchem.integrated_path_length, wittig_qchem.energies,'o-')

len(wittig.optimized), len(wittig_qchem)

# +
# wittig.optimized.plot_chain()

# +
# ch.visualize_chain(wittig.optimized)
# -

tsg1_jan = wittig.optimized[16]
tsg2_jan = wittig.optimized[26]

try:
    ts_res1_jan = eng._compute_ts_result(h.ordered_leaves[0].data.optimized.get_ts_node(), keywords={'maxiter':500})
except Exception as e:
    failed_out1 = e

try:
    ts_res1_jan = eng._compute_ts_result(tsg1_jan, keywords={'maxiter':500})
except Exception as e:
    failed_out1 = e

# +

try:
    ts_res2_jan = eng._compute_ts_result(tsg2_jan, keywords={'maxiter':500})
except Exception as e:
    failed_out2 = e
# -

irc1_res = build_full_irc_chain(StructureNode(structure=ts_res1_jan.return_result), engine=eng)

irc2_res = build_full_irc_chain(StructureNode(structure=ts_res2_jan.return_result), engine=eng)

coord1 = ch.get_rxn_coordinate(irc1_res[1])
coord2 = ch.get_rxn_coordinate(irc2_res[1])

irc2_res[0].nodes.reverse()
irc2_res[1].nodes.reverse()

# full_irc = Chain(irc1_res[0].nodes+irc2_res[0].nodes)
full_irc = Chain([irc1_res[0][0]]+irc1_res[1].nodes+[irc1_res[0][-1], irc2_res[0][0]]+irc2_res[1].nodes+[irc2_res[0][-1]])



align_geom(



# +
dists_fsm_1 = get_projections(wittig.optimized, coord1)
dists_fsm_2 = get_projections(wittig.optimized, coord2)


dists_qchem_1 = get_projections(wittig_qchem, coord1)
dists_qchem_2 = get_projections(wittig_qchem, coord2)

dists_irc_1 = get_projections(full_irc, coord1)
dists_irc_2 = get_projections(full_irc, coord2)

# +
s=6
fs=18
f, ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(wittig.optimized.integrated_path_length, wittig.optimized.energies_kcalmol, 'o-', label='fsm')
plt.plot(wittig_qchem.integrated_path_length, wittig_qchem.energies_kcalmol, 'o-', label='qchem')


plt.plot(full_irc.integrated_path_length, full_irc.energies_kcalmol, 'o--',label='irc', color='black', lw=3)
plt.ylabel("Rxn coordinate 2", fontsize=fs)
plt.xlabel("Rxn coordinate 1", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
# plt.savefig("attempt_at_wittig_irc.svg")

# +
s=6
fs=18
f, ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(dists_fsm_1, dists_fsm_2, 'o-', label='fsm')
plt.plot(dists_qchem_1, dists_qchem_2, 'o-', label='qchem')


plt.plot(np.array(dists_irc_1), np.array(dists_irc_2), 'o--',label='irc', color='black', lw=3)
plt.ylabel("Rxn coordinate 2", fontsize=fs)
plt.xlabel("Rxn coordinate 1", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
# plt.savefig("attempt_at_wittig_irc.svg")

# +

ch.visualize_chain(full_irc)

# +

len(wittig.optimized)

# +

full_irc.write_to_disk("/home/jdep/T3D_data/fneb_draft/wittig/irc_disjoined_intermediates.xyz")
# -

RMSD(full_irc[0].coords, wittig.optimized[0].coords)

full_irc.plot_chain()

# +

view.view(ts_res1_jan)
# -

hess_ress1 = eng._compute_hessian_result(StructureNode(structure=ts_res1_jan.return_result))

hess_ress2 = eng._compute_hessian_result(StructureNode(structure=ts_res2_jan.return_result))

from pathlib import Path


def create_submission_scripts(
    out_dir: Path,
    out_name: str, 
    submission_dir: Path,
    reference_dir: Path,
    reference_start_name: str = 'start.xyz',
    reference_end_name: str = 'end.xyz',
    prog: str = 'xtb',
    eng: str = 'qcop',
    sig: int = 1,
    nrt: float = 1,  # node rms threshold
    net: float = 1,  # node ene threshold
    tcin_fp: str = None,
    es_ft: float = 0.03,
    met: str = "asneb"
    
    ):

    # if out_name is None:
    #     out_name = reference_start_name.stem
    template = [
    "#!/bin/bash",
    "",
    # "#SBATCH -t 24:00:00",
    "#SBATCH -t 12:00:00",
    "#SBATCH -J asneb",
    # "#SBATCH -p gpuq",
    "#SBATCH --qos=gpu_short",
    "#SBATCH --gres=gpu:1",
    "",
    "work_dir=$PWD",
    "",
    "cd $SCRATCH",
    "",
    "# Load modules",
    "ml TeraChem",
    "terachem -s 11111 || true &",
    "source /home/jdep/.bashrc",
    "source activate neb",
    "export OMP_NUM_THREADS=1",
    "# Run the job",
    "create-mep ",
    ]
    
    start_fp = reference_dir / reference_start_name
    end_fp = reference_dir / reference_end_name
    
    out_fp = out_dir / out_name
    print(out_fp)


    if tcin_fp:
        tcin = f"-tcin {tcin_fp}"
    else:
        tcin = ""
    
    # cmd = f"/home/jdep/.cache/pypoetry/virtualenvs/neb-dynamics-G7__rX26-py3.9/bin/python /home/jdep/neb_dynamics/neb_dynamics/scripts/create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.002 \
    #     -sig {sig} -nimg 12 -min_ends 1 \
    #     -es_ft {es_ft} -name {out_fp} -prog {prog} -eng {eng} -node_rms_thre {nrt} -met {met} -node_ene_thre {net} {tcin} &> {out_fp}_out "
    cmd = f"create-mep run {start_fp} {end_fp} --minimize-ends --name {out_fp} --inputs {tcin_fp} --recursive &> {out_fp}_out"
    
    new_template = template.copy()
    new_template[-1] = cmd
    
    with open(
        submission_dir / out_name , "w+"
    ) as f:
        f.write("\n".join(new_template))

# ref_dir = Path("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/structures")
ref_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures")
all_rns = [Path(p).stem for p in open("/home/jdep/T3D_data/msmep_draft/comparisons_dft/reactions_todo_multistep.txt").read().splitlines()]
# all_rns = [p.stem for p in ref_dir.glob("*") if p.stem in ]

import os

# rn = '1-2-Amide-Phthalamide-Synthesis'
rns_to_submit = []
for rn in all_rns:
    # out_dir = Path('/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/results_asneb')
    out_dir = Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb')
    # if (out_dir/f'{rn}_out').exists():
    #     continue
    # else:
    # print(rn)
    # rns_to_submit.append(Path('/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/submissions_dir')/rn)
    rns_to_submit.append(Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/submissions_dir')/rn)
    create_submission_scripts(
        out_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb'),
        out_name=rn,
        submission_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/submissions_dir'), 
        reference_dir=Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}'),
        reference_start_name='start_xtb.xyz', reference_end_name='end_xtb.xyz',
        eng='qcop',
        prog='terachem-pbs',
        tcin_fp='/home/jdep/T3D_data/msmep_draft/comparisons_dft/input.toml',
        es_ft=0.03,
        met='asneb',
        nrt=1, net=1)
    # create_submission_scripts(
    #     out_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb'),
    #     out_name=rn+"_asfneb",
    #     submission_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/submissions_dir'), 
    #     reference_dir=Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}'),
    #     reference_start_name='start_xtb.xyz', reference_end_name='end_xtb.xyz',
    #     eng='qcop',
    #     prog='terachem',
    #     tcin_fp='/home/jdep/T3D_data/msmep_draft/comparisons_dft/tc.in',
    #     es_ft=10,
    #     met='asfneb',
    #     nrt=1, net=5)
    
    # create_bs_submission(
    #     out_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb'),
    #     out_name=rn,
    #     submission_dir=Path('/home/jdep/T3D_data/msmep_draft/comparisons_dft/submissions_dir'), 
    #     reference_dir=Path(f'/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/{rn}'),
    #     reference_start_name='start_xtb.xyz', reference_end_name='end_xtb.xyz',
    #     eng='chemcloud',
    #     prog='terachem',
    #     tcin_fp='/home/jdep/T3D_data/msmep_draft/comparisons_dft/tc.in',
    #     nrt=1, net=1)


from neb_dynamics.TreeNode import TreeNode


h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/full_retropaths_launch/results_asneb/Alcohol-Bromination/")


import neb_dynamics.chainhelpers as ch
from neb_dynamics.elementarystep import check_if_elem_step

ch.visualize_chain(h.output_chain)

from neb_dynamics.engines import QCOPEngine
from qcio import ProgramInput, Structure

pi = ProgramInput(structure=Structure.from_smiles("O"), 
                  model={'method':'ub3lyp','basis':'3-21gs'}, 
                  keywords={'pcm':'cosmo','epsilon':80.4},
                 calctype='energy')
eng = QCOPEngine(program_input=pi, program='terachem')

output = check_if_elem_step(h.output_chain, engine=eng)

# +

h = TreeNode.read_from_disk("/home/jdep/T3D_data/ladderane/bugfixed/")
# -

ch.visualize_chain(h.output_chain)

sub_chain = h.output_chain.copy()

sub_nodes = sub_chain.nodes[:40]

sub_chain.nodes = sub_nodes

ch.visualize_chain(sub_chain)

eng2 = ASEEngine(ase_opt_str='MDMin', calculator=XTB())

out_tr = eng2.compute_geometry_optimization(sub_chain[6])

ch.visualize_chain(Chain(out_tr, ChainInputs()))

from neb_dynamics.nodes.nodehelpers import is_identical

is_identical(h.output_chain[-1], output.minimization_results[-1])



# ?h.output_chain[-1].__eq__

view.view(*[node.structure for node in output.minimization_results])

view.view(output.minimization_results[0].structure, h.output_chain[0].structure)

# +

output.minimization_results[0] == 

# +

from qcio import view
# -

for rn in rns_to_submit:
    os.system(f'sbatch {str(rn.resolve())}')



