#!/home/jdep/.conda/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Node3D import Node3D
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs
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


    tol = 0.001
    cni = ChainInputs(k=0.01)
    nbi = NEBInputs(tol=tol, v=True, max_steps=2000)
    chain = Chain.from_traj(traj=traj, parameters=cni)
    n = NEB(initial_chain=chain, parameters=nbi)
    try: 
        n.optimize_chain()
        data_dir = fp.parent
        n.write_to_disk(data_dir/f"{fp.stem}_neb.xyz")
    except NoneConvergedException as e:
        e.obj.write_to_disk(data_dir/f"{fp.stem}_failed.xyz")
        
	    

if __name__ == "__main__":
    main()
