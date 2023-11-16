#!/home/jdep/.conda/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB
from neb_dynamics.Chain import Chain
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.NEB import NEB, NoneConvergedException

from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB

from neb_dynamics.nodes.Node3D import Node3D
from neb_dynamics.Janitor import Janitor
from neb_dynamics.nodes.Node3D_gfn1xtb import Node3D_gfn1xtb
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.helper_functions import get_seeding_chain
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
    
    parser.add_argument(
        '-nc',
        '--node_class',
        dest='nc',
        type=str,
        default="node3d_tc",
        help='what node type to use. options are: node3d_tc, node3d_tc_local, node3d_tcpb'
        
    )
    
    parser.add_argument(
        '-ft',
        '--force_thre',
        dest='ft',
        type=float,
        default=0.003,
        help='force threshold of the seeding chain'
        
    )
    
    return parser.parse_args()


def main():
    # import os
    # del os.environ['OE_LICENSE']
    args = read_single_arguments()
    
    nodes = {'node3d': Node3D, 'node3d_tc': Node3D_TC, 'node3d_tc_local': Node3D_TC_Local, 'node3d_tcpb': Node3D_TC_TCPB}
    nc = nodes[args.nc]

    fp = Path(args.f)
    
    method = 'wb97xd3'
    basis = 'def2-svp'
    kwds = {}
        
    if args.nc == 'node3d_tcpb':
        do_parallel=False
    else:
        do_parallel=True

    
    
    cni = ChainInputs(k=0.1,delta_k=0.0, node_class=nc,step_size=3,  min_step_size=0.33, friction_optimal_gi=True, do_parallel=do_parallel,
                      als_max_steps=3, node_freezing=False)
    nbi = NEBInputs(grad_thre=0.001*BOHR_TO_ANGSTROMS,
                rms_grad_thre=0.0005*BOHR_TO_ANGSTROMS,
                en_thre=0.0001*BOHR_TO_ANGSTROMS,
                v=1, 
                max_steps=2000,
                early_stop_chain_rms_thre=0.002, 
                early_stop_force_thre=0.01, 
            
                early_stop_still_steps_thre=500,
                vv_force_thre=0.0)
    
    neb_xtb = NEB.read_from_disk(fp)
    
    
    start = neb_xtb.initial_chain[0].tdstructure
    start.tc_model_method = method
    start.tc_model_basis = basis
    start.tc_kwds = kwds
            
    start_opt = start.tc_geom_optimization()
    
    end = neb_xtb.initial_chain[-1].tdstructure
    end.tc_model_method = method
    end.tc_model_basis = basis
    end.tc_kwds = kwds
            
    end_opt = end.tc_geom_optimization()
    
    
    seeding_chain = get_seeding_chain(neb_xtb, args.ft)
    
    xtb_traj = seeding_chain.to_trajectory()
    for td in xtb_traj:
        td.tc_model_basis = basis
        td.tc_model_method = method
        
    
    tr = Trajectory([start_opt]+xtb_traj.traj+[end_opt])
    chain = Chain.from_traj(tr, parameters=cni)
    
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
