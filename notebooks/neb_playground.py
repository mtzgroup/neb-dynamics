# -*- coding: utf-8 -*-
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.neb import NEB


h = TreeNode.read_from_disk('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb/Wittig')
neb = NEB.read_from_disk('/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_neb/Wittig24_neb')

import neb_dynamics.chainhelpers as ch

# ch.visualize_chain(h.output_chain)
ch.visualize_chain(neb.optimized)

# +
# ch.visualize_chain(history.output_chain)
# -

from neb_dynamics.TreeNode import TreeNode
from qcio import DualProgramInput

# +
# h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/results_asneb/Claisen-Rearrangement-Aromatic/")
fneb = NEB.read_from_disk("/home/jdep/debug/wittig/db_neb/")
fneb2 = NEB.read_from_disk("/home/jdep/debug/wittig/db2_neb/")
fneb3 = NEB.read_from_disk("/home/jdep/debug/wittig/db3_neb/")

fneb4 = NEB.read_from_disk("/home/jdep/debug/wittig/db_nogitan_neb/")
fneb5 = NEB.read_from_disk("/home/jdep/debug/wittig/db_nogitan2_neb/")
# -

fneb = NEB.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Beckmann-Rearrangement/db4_neb.xyz")
asfneb = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Beckmann-Rearrangement/asdb/")
# asneb = TreeNode.read_from_disk("/home/jdep/debug/qchem/cope/db4")

ch.plot_opt_history(asfneb.ordered_leaves[1].data.chain_trajectory)

ch.visualize_chain(asfneb.output_chain)

# +
# view.view(fneb.optimized[0].structure, fneb.optimized[-1].structure, show_indices=True)

# +
# 118 ++ 1000ish
# -

# ch.visualize_chain(fneb.optimized)
ch.visualize_chain(asfneb.output_chain)

plt.plot(fneb.optimized.energies_kcalmol, 'o-'

# ch.visualize_chain(fneb.optimized)
# ch.visualize_chain(fneb2.optimized)
# ch.visualize_chain(fneb3.optimized)
ch.visualize_chain(fneb3.optimized)

tsg1 = h.ordered_leaves[0].data.optimized.get_ts_node()

tsg2 = h.ordered_leaves[1].data.optimized.get_ts_node()

from neb_dynamics.neb import NEB

fneb = NEB.read_from_disk("/home/jdep/T3D_data/fneb_draft/hardexample1/fneb_geo_dft_neb.xyz")

ch.visualize_chain(fneb.optimized)


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


# +

from pathlib import Path
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file
# -

qchem = load_qchem_result("/home/jdep/T3D_data/fneb_draft/wittig/fsm.files/")

# +

tsg3 = qchem.get_ts_node()
# -

dpi1 = DualProgramInput(structure=tsg1.structure, 
                       keywords={'max_iter':500},
                       subprogram='terachem',
                       subprogram_args={'model':{'method':'hf','basis':'6-31g'}}, 
                       calctype='transition_state')

dpi2 = DualProgramInput(structure=tsg2.structure, 
                       keywords={'max_iter':500},
                       subprogram='terachem',
                       subprogram_args={'model':{'method':'hf','basis':'6-31g'}}, 
                       calctype='transition_state')

dpi3 = DualProgramInput(structure=tsg3.structure, 
                       keywords={'max_iter':500},
                       subprogram='terachem',
                       subprogram_args={'model':{'method':'hf','basis':'6-31g'}}, 
                       calctype='transition_state')

from chemcloud import CCClient

client = CCClient()

future_res = client.compute('geometric', [dpi1, dpi2])

future_res2 = client.compute('geometric', dpi3)

output2 = future_res2.get()

output = future_res.get()

output[0].return_result.save("/home/jdep/T3D_data/fneb_draft/wittig/irc/ts1_asfneb.xyz")

output[1].return_result.save("/home/jdep/T3D_data/fneb_draft/wittig/irc/ts2_asfneb.xyz")

view.view(output[0].return_result, output[1].return_result, output[2].return_result)

view.view(output[0].return_result, output[1].return_result, output2.return_result)

view.view(dpi3.structure)

from pathlib import Path

from qcio import DualProgramInput, Structure

struct = Structure.from_smiles("COCO")
dpi = DualProgramInput(
    calctype = "neb",
    structure = struct, # this probably would have to be ignored
    keywords = {"recursive": True},  # Optional
    subprogram = "xtb",
    subprogram_args = {
        'model':{"method": "GFN2-XTB", "basis": "GFN2-XTB"},
        'keywords':{}}  # Optional}
)

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
    "create_msmep_from_endpoints.py ",
    ]
    
    start_fp = reference_dir / reference_start_name
    end_fp = reference_dir / reference_end_name
    
    out_fp = out_dir / out_name
    print(out_fp)


    if tcin_fp:
        tcin = f"-tcin {tcin_fp}"
    else:
        tcin = ""
    
    cmd = f"/home/jdep/.cache/pypoetry/virtualenvs/neb-dynamics-G7__rX26-py3.9/bin/python /home/jdep/neb_dynamics/neb_dynamics/scripts/create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.002 \
        -sig {sig} -nimg 12 -min_ends 1 \
        -es_ft {es_ft} -name {out_fp} -prog {prog} -eng {eng} -node_rms_thre {nrt} -met {met} -node_ene_thre {net} {tcin} &> {out_fp}_out "
    
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
        tcin_fp='/home/jdep/T3D_data/msmep_draft/comparisons_dft/tc.in',
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



