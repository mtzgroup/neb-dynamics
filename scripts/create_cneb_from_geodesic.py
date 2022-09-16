#!/home/jdep/.conda/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.NEB import NEB, Node3D, Chain, NoneConvergedException
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
    
    return parser.parse_args()


def main():
    args = read_single_arguments()

    fp = Path(args.f)

    traj = Trajectory.from_xyz(fp, tot_charge=args.c, tot_spinmult=args.s)


    tol = 4.5e-3
    chain = Chain.from_traj(traj=traj, k=.1, delta_k=0.0, step_size=2, node_class=Node3D)
    n = NEB(initial_chain=chain, grad_thre=tol, en_thre=tol/450, rms_grad_thre=tol*(2/3), climb=True, vv_force_thre=0, max_steps=10000)
    try: 
        n.optimize_chain()
        data_dir = fp.parent
        n.write_to_disk(data_dir/f"{fp.stem}_cneb.xyz")
    except NoneConvergedException as e:
        e.obj.write_to_disk(data_dir/f"{fp.stem}_failed.xyz")
        
	    

if __name__ == "__main__":
    main()
