#!/home/jdep/.conda/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB, Node3D, Chain, NoneConvergedException
from neb_dynamics.MSMEP import MSMEP
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
    out = m.find_mep_multistep((traj[0], traj[-1]), do_alignment=True)
    data_dir = fp.parent
    t = Trajectory([n.tdstructure for n in out])
    t.write_trajectory(data_dir/f"{fp.stem}_msmep_cneb.xyz")
    n.write_to_disk(data_dir/f"{fp.stem}_cneb.xyz")
        
	    

if __name__ == "__main__":
    main()
