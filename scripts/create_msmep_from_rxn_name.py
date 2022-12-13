#!/home/jdep/.conda/envs/rp/bin/python
import random
from argparse import ArgumentParser
from pathlib import Path
import time


from retropaths.abinitio.tdstructure import TDStructure
import retropaths.helper_functions as hf
from retropaths.abinitio.trajectory import Trajectory
from retropaths.helper_functions import pairwise
from retropaths.molecules.molecule import Molecule
from retropaths.reactions.changes import Changes3D, Changes3DList
from retropaths.reactions.conditions import Conditions
from retropaths.reactions.rules import Rules
from retropaths.reactions.template import ReactionTemplate
from scipy.signal import argrelextrema

from neb_dynamics.Chain import Chain
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Node3D import Node3D
from neb_dynamics.trajectory import Trajectory

import os

random.seed(1)


def read_single_arguments():
    """
    Command line reader
    """


    description_string = "script for filding multi step minimum energy paths from a reaction name"
    parser = ArgumentParser(description=description_string)

    parser.add_argument(
        "-f",
        "--fp",
        dest="f",
        type=str,
        required=True,
        help="file path to reactions pickle obj",
    )

    parser.add_argument(
        "-n",
        "--name",
        dest="n",
        type=str,
        required=True,
        help="name of reaction",
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
        "-o",
        "--outdir",
        dest='o',
        type=str,
        default=os.getcwd(),
        help='where to output the final trajectory'

    )


    parser.add_argument(
        "-of",
        "--outfile",
        dest='of',
        type=str,
        default=None,
        help='name_of_output_file'

    )


    return parser.parse_args()

def main():
    args = read_single_arguments()

    fp = Path(args.f)
    reactions = hf.pload(fp)
    rxn_name = args.n
    outdir = Path(args.o)
    outfile = args.of
    tol = 0.01
    max_steps=2000

    if outfile is None: outfile = f"msmep_tol{tol}_max_{max_steps}.xyz"

    m = MSMEP(max_steps=max_steps, v=True, tol=tol)
    root,target = m.create_endpoints_from_rxn_name(rxn_name=rxn_name, reactions_object=reactions)

    start = time.time()

    n_obj, out_chain = m.find_mep_multistep((root, target), do_alignment=True)
    end = time.time()

    print(f"Total time (s): {end - start}")

    t = out_chain.to_trajectory()
    t.write_trajectory(outdir/outfile)


if __name__ == "__main__":
    main()
