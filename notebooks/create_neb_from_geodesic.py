from argparse import ArgumentParser
from pathlib import Path

from retropaths.abinitio.trajectory import Trajectory

from NEB_xtb import neb


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

    parser.add_argument("-c", "--charge", dest="c", type=int, default=0, help="total charge of system")

    parser.add_argument("-s", "--spinmult", dest="s", type=int, default=1, help="total spinmultiplicity of system")

    return parser.parse_args()


def main():
    args = read_single_arguments()

    fp = Path(args.f)

    traj = Trajectory.from_xyz(fp, tot_charge=args.c, tot_spinmult=args.s)

    n = neb()

    opt_chain, chain_changes = n.optimize_chain(chain=traj, grad_func=n.grad_func, en_func=n.en_func, k=10)

    opt_traj = Trajectory(traj_array=opt_chain, tot_charge=args.c, tot_spinmult=args.s)

    data_dir = fp.parent

    opt_traj.write_trajectory(data_dir / f"{fp.stem}_neb.xyz")


if __name__ == "__main__":
    main()
