#!/home/jdep/.conda/envs/rp/bin/python


from pathlib import Path
from argparse import ArgumentParser
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.NEB import NEB, Node3D, Chain
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

    chain = Chain.from_traj(traj,k=0.1)
    n = NEB(initial_chain=chain, grad_thre=0.01, climb=False)
    
    n.optimize_chain()

    list_of_tdstructs = np.array([s.tdstructure for s in n.optimized.nodes])

    opt_traj = Trajectory(traj_array=list_of_tdstructs, tot_charge=args.c, tot_spinmult=args.s)

    data_dir = fp.parent

    opt_traj.write_trajectory(data_dir/f"{fp.stem}_neb.xyz")


if __name__ == "__main__":
    main()
