#!/home/jdep/.conda/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Node3D_TC import Node3D_TC
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


    tol = 0.01
    cni = ChainInputs(k=0.01, node_class=Node3D_TC)
    method = 'gfn2xtb'
    basis = 'gfn2xtb'
    for td in traj:
        td.tc_model_method = method
        td.tc_model_basis = basis

    nbi = NEBInputs(tol=tol, v=True, max_steps=2000)
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
