#!/home/jdep/.conda/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Node3D_TC import Node3D_TC
from neb_dynamics.Node3D import Node3D

from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs
import numpy as np
from neb_dynamics.constants import BOHR_TO_ANGSTROMS


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
    
    return parser.parse_args()


def main():
    #import os
    #del os.environ['OE_LICENSE']
    args = read_single_arguments()

    nodes = {'node3d': Node3D, 'node3d_tc': Node3D_TC, 'node3d_tc_local': Node3D_TC_Local, 'node3d_tcpb': Node3D_TC_TCPB}
    nc = nodes[args.nc]

    fp = Path(args.f)

    traj = Trajectory.from_xyz(fp, tot_charge=args.c, tot_spinmult=args.s)
    tol = 0.001
    # cni = ChainInputs(k=0.01,delta_k=0.00, node_class=Node3D, step_size=1,friction_optimal_gi=True)
    if args.nc != "node3d":
        method = 'b3lyp' 
        basis = '3-21gs'
        # kwds = {'restricted': False}
        kwds = {'restricted': False, 'pcm':'cosmo','epsilon':80}
        for td in traj:
            td.tc_model_method = method
            td.tc_model_basis = basis
            td.tc_kwds = kwds
            
    if args.nc == 'node3d_tcpb':
        do_parallel=False
    else:
        do_parallel=True
    
    cni = ChainInputs(k=0.1,delta_k=0.09, node_class=nc,step_size=3,  min_step_size=0.33, friction_optimal_gi=True, do_parallel=do_parallel,
                      als_max_steps=3)

    nbi = NEBInputs(tol=tol, # tol means nothing in this case
                    grad_thre=0.001*BOHR_TO_ANGSTROMS,
                    rms_grad_thre=0.0005*BOHR_TO_ANGSTROMS,
                    en_thre=0.001*BOHR_TO_ANGSTROMS,
                    v=True, 
                    max_steps=4000,
                    vv_force_thre=0.0)
    
    chain = Chain.from_traj(traj=traj, parameters=cni)
    n = NEB(initial_chain=chain, parameters=nbi)
    try: 
        n.optimize_chain()
        data_dir = fp.parent
        n.write_to_disk(data_dir/f"{fp.stem}_neb.xyz",write_history=True)
    except NoneConvergedException as e:
        data_dir = fp.parent
        e.obj.write_to_disk(data_dir/f"{fp.stem}_failed.xyz",write_history=True)
        
	    

if __name__ == "__main__":
    main()
