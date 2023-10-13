from pathlib import Path
from argparse import ArgumentParser
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB
from neb_dynamics.Chain import Chain
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Janitor import Janitor
from neb_dynamics.Node3D_gfn1xtb import Node3D_gfn1xtb
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
import numpy as np


def read_single_arguments():
    """
    Command line reader
    """
    description_string = "will take path to an xyz file of geodesic trajectory and relax it using XTB neb"
    parser = ArgumentParser(description=description_string)
    parser.add_argument(
        "-f",
        "--fp",
        dest="f",
        type=str,
        required=True,
        help="file path",
    )

    parser.add_argument(
        "-c",
        "--charge",
        dest="c",
        type=int,
        default=0,
        help='total charge of system'
    )

    parser.add_argument(
        "-s",
        "--spinmult",
        dest='s',
        type=int,
        default=1,
        help='total spinmultiplicity of system'

    )
    
    parser.add_argument(
        '-dc',
        '--do_cleanup',
        dest='dc',
        type=bool,
        default=True,
        help='whether to do conformer-conformer NEBs at the end'
        
    )
    
    return parser.parse_args()


def main():
    # import os
    # del os.environ['OE_LICENSE']
    args = read_single_arguments()

    fp = Path(args.f)

    traj = Trajectory.from_xyz(fp, tot_charge=args.c, tot_spinmult=args.s)
    tol = 0.001
    # cni = ChainInputs(k=0.01,delta_k=0.00, node_class=Node3D, step_size=1,friction_optimal_gi=True)
    cni = ChainInputs(k=0.1,delta_k=0.09, node_class=Node3D,step_size=3,  min_step_size=0.33, friction_optimal_gi=True)
    #method = 'gfn2xtb'
    #basis = 'gfn2xtb'
    #for td in traj:
    #    td.tc_model_method = method
    #    td.tc_model_basis = basis

    # nbi = NEBInputs(tol=tol, v=True, max_steps=2000,early_stop_chain_rms_thre=0.002,
    #                 early_stop_force_thre=0.005, early_stop_still_steps_thre=200, vv_force_thre=0.00)
    # nbi = NEBInputs(tol=tol, # tol means nothing in this case
    #                 grad_thre=0.001*BOHR_TO_ANGSTROMS,
    #                 rms_grad_thre=0.0005*BOHR_TO_ANGSTROMS,
    #                 en_thre=0.001*BOHR_TO_ANGSTROMS,
    #                 v=True, 
    #                 max_steps=2000,
    #                 # early_stop_chain_rms_thre=0.0014,
    #                 # early_stop_force_thre=0.003, 
    #                 # early_stop_still_steps_thre=200,
    #                 node_freezing=False,
    nbi = NEBInputs(grad_thre=0.001*BOHR_TO_ANGSTROMS,
                rms_grad_thre=0.0005*BOHR_TO_ANGSTROMS,
                en_thre=0.0001*BOHR_TO_ANGSTROMS,
                v=1, 
                max_steps=2000,
                early_stop_chain_rms_thre=0.002, 
                early_stop_force_thre=0.01, 
            
                early_stop_still_steps_thre=500,
                vv_force_thre=0.0,
                node_freezing=False)
    chain = Chain.from_traj(traj=traj, parameters=cni)
    m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=GIInputs(nimages=15,extra_kwds={"sweep":False}))
    history, out_chain = m.find_mep_multistep(chain)
    data_dir = fp.parent
    
    out_chain.to_trajectory().write_trajectory(data_dir/f"{fp.stem}_msmep.xyz")
    history.write_to_disk(data_dir/f"{fp.stem}_msmep")
    
    
    if args.dc:
        op = data_dir/f"{fp.stem}_cleanups"
        j = Janitor(
            history_object=history,
            out_path=op,
            msmep_object=m
        )
        
        clean_msmep = j.create_clean_msmep()
        
        if clean_msmep:
            clean_msmep.to_trajectory().write_trajectory(data_dir/f"{fp.stem}_msmep_clean.xyz")

if __name__ == "__main__":
    main()