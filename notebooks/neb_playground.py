# -*- coding: utf-8 -*-
# # Redo inptus

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
    tcin_fp: str = None
    
    ):

    if out_name is None:
        out_name = reference_start_name.stem
    template = [
    "#!/bin/bash",
    "",
    "#SBATCH -t 24:00:00",
    "#SBATCH -J asneb",
    "#SBATCH -p gpuq",
    # "#SBATCH --qos=gpu",
    "#SBATCH --gres=gpu:1",
    "",
    "work_dir=$PWD",
    "",
    "cd $SCRATCH",
    "",
    "# Load modules",
    "ml TeraChem",
    "source /home/jdep/.bashrc",
    "source activate neb",
    "export OMP_NUM_THREADS=1",
    "# Run the job",
    "create_msmep_from_endpoints.py ",
    ]
    
    start_fp = reference_dir / reference_start_name
    end_fp = reference_dir / reference_end_name
    
    out_fp = out_dir / out_name

    if tcin_fp:
        tcin = f"-tcin {tcin_fp}"
    else:
        tcin = ""
    
    cmd = f"/home/jdep/.cache/pypoetry/virtualenvs/neb-dynamics-G7__rX26-py3.9/bin/python /home/jdep/neb_dynamics/neb_dynamics/scripts/create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.002 \
        -sig {sig} -nimg 12 -min_ends 1 \
        -es_ft 0.03 -name {out_fp} -prog {prog} -eng {eng} -node_rms_thre {nrt} -node_ene_thre {net} {tcin} &> {out_fp}_out "
    
    new_template = template.copy()
    new_template[-1] = cmd
    
    with open(
        submission_dir / out_name , "w+"
    ) as f:
        f.write("\n".join(new_template))


def create_bs_submission(
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
    tcin_fp: str = None
    
    ):

    if out_name is None:
        out_name = reference_start_name.stem
    template = [
    "",
    ]
    
    start_fp = reference_dir / reference_start_name
    end_fp = reference_dir / reference_end_name
    
    out_fp = out_dir / out_name

    if tcin_fp:
        tcin = f"-tcin {tcin_fp}"
    else:
        tcin = ""
    
    cmd = f"/home/jdep/.cache/pypoetry/virtualenvs/neb-dynamics-G7__rX26-py3.9/bin/python /home/jdep/neb_dynamics/neb_dynamics/scripts/create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.002 \
        -sig {sig} -nimg 12 -min_ends 1 \
        -es_ft 0.03 -name {out_fp} -prog {prog} -eng {eng} -node_rms_thre {nrt} -node_ene_thre {net} {tcin} &> {out_fp}_out &\n"
    
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
        prog='terachem',
        tcin_fp='/home/jdep/T3D_data/msmep_draft/comparisons_dft/tc.in',
        nrt=1, net=1)
    
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



