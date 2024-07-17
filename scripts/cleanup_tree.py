#!/home/jdep/.conda/envs/neb/bin/python
from pathlib import Path
from argparse import ArgumentParser
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.tdstructure import TDStructure


from neb_dynamics.NEB import NEB
from chain import Chain
from neb_dynamics.nodes.Node3D_TC import Node3D_TC

from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB
from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.Linesearch import Linesearch
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer

from neb_dynamics.nodes.Node3D import Node3D
from neb_dynamics.Janitor import Janitor
from neb_dynamics.nodes.Node3D_gfn1xtb import Node3D_gfn1xtb
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.helper_functions import create_friction_optimal_gi
from neb_dynamics.optimizers.BFGS import BFGS

from neb_dynamics.constants import BOHR_TO_ANGSTROMS
import numpy as np

import sys



def read_single_arguments():
    """
    Command line reader
    """
    description_string = "will take path to an xyz file of geodesic trajectory and relax it using XTB neb"
    parser = ArgumentParser(description=description_string)

    parser.add_argument(
        "-c", "--charge", dest="c", type=int, default=0, help="total charge of system"
    )

    parser.add_argument(
        "-nimg",
        "--nimages",
        dest="nimg",
        type=int,
        default=8,
        help="number of images in the chain",
    )

    parser.add_argument(
        "-s",
        "--spinmult",
        dest="s",
        type=int,
        default=1,
        help="total spinmultiplicity of system",
    )

    parser.add_argument(
        "-nc",
        "--node_class",
        dest="nc",
        type=str,
        default="node3d",
        help="what node type to use. options are: node3d, node3d_tc, node3d_tc_local, node3d_tcpb",
    )


    parser.add_argument(
        "-fp",
        "--file_path",
        dest="fp",
        required=True,
        type=str,
        help="path to the tree folder",
    )

    parser.add_argument(
        "-tol",
        "--tolerance",
        dest="tol",
        required=True,
        type=float,
        help="convergence threshold in H/Bohr^2",
        default=0.001,
    )

    parser.add_argument(
        "-name",
        "--file_name",
        dest="name",
        required=False,
        type=str,
        help="name of folder to output to || defaults to react_msmep",
        default=None,
    )

    parser.add_argument(
        "-climb",
        "--do_climb",
        dest="climb",
        required=False,
        type=int,
        help="whether to use cNEB",
        default=0
    )

    parser.add_argument(
        "-mr",
        "--maxima_recyling",
        dest="mr",
        required=False,
        type=int,
        help="whether to use maxima recyling",
        default=0
    )

    parser.add_argument(
        "-es_ft",
        "--early_stop_ft",
        dest="es_ft",
        required=False,
        type=float,
        help="force threshold for early stopping",
        default=0.03
    )

    parser.add_argument(
        "-preopt",
        "--preopt_with_xtb",
        dest="preopt",
        required=True,
        type=int,
        help="whether to preconverge chains using xtb",
        default=1
    )


    return parser.parse_args()


def main():
    args = read_single_arguments()
    fp = Path(args.fp)

    nodes = {'node3d': Node3D, 'node3d_tc': Node3D_TC, 'node3d_tc_local': Node3D_TC_Local, 'node3d_tcpb': Node3D_TC_TCPB}
    # nodes = {"node3d": Node3D, "node3d_tc": Node3D_TC}
    nc = nodes[args.nc]

    method = "wb97xd3" #terachem
    basis = "def2-svp"
    kwds = {}

    if args.nc == "node3d_tc_local":
        do_parallel = False
    else:
        do_parallel = True

    tol = args.tol

    cni = ChainInputs(
        k=0.1,
        delta_k=0.09,
        node_class=nc,
        friction_optimal_gi=True,
        use_maxima_recyling=bool(args.mr),
        do_parallel=do_parallel,
        node_freezing=True,
        skip_identical_graphs=0,
        tc_model_method=method,
        tc_model_basis=basis,
        tc_kwds = kwds
    )

    optimizer = VelocityProjectedOptimizer(timestep=.5, activation_tol=0.1)

    nbi = NEBInputs(
        tol=tol * BOHR_TO_ANGSTROMS,
        barrier_thre=0.1, #kcalmol,
        climb=bool(args.climb),

        rms_grad_thre=tol * BOHR_TO_ANGSTROMS*2,
        max_rms_grad_thre=tol * BOHR_TO_ANGSTROMS*5,
        ts_grad_thre=tol * BOHR_TO_ANGSTROMS*5,

        v=1,
        max_steps=500,

        early_stop_chain_rms_thre=1,
        early_stop_force_thre=args.es_ft,
        early_stop_still_steps_thre=100,

        _use_dlf_conv=False,
        preopt_with_xtb=bool(int(args.preopt))
    )
    print(f"{args.preopt=}")
    print(f"NEBinputs: {nbi}\nChainInputs: {cni}\nOptimizer: {optimizer}")
    gii = GIInputs(nimages=args.nimg, extra_kwds={"sweep": False})
    sys.stdout.flush()
    tree = TreeNode.read_from_disk(fp, chain_parameters=cni, neb_parameters=nbi, gi_parameters=gii, optimizer=optimizer)

    m = MSMEP(
    neb_inputs=nbi,
    chain_inputs=cni,
    gi_inputs=gii,
    optimizer=optimizer,
    )
    data_dir = fp.parent
    if args.name:
        foldername = data_dir / args.name
        filename = data_dir / (args.name + ".xyz")

    else:
        foldername = data_dir / f"{fp.stem}_cleanups"
        filename = data_dir / f"{fp.stem}_cleanups.xyz"

    op = foldername
    j = Janitor(history_object=tree, out_path=op, msmep_object=m)

    clean_msmep = j.create_clean_msmep()
    j.write_to_disk(op)

    if clean_msmep:
        clean_msmep.write_to_disk(data_dir / filename)

        tot_num_steps = sum([t.get_num_opt_steps() for t in j.cleanup_trees])
        print(f">>> Made {tot_num_steps} optimization steps.")
    else:
        print(f">>> Made 0 optimization steps.")




if __name__ == "__main__":
    main()
