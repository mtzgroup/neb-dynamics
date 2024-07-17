#!/home/jdep/.conda/envs/neb/bin/python
from pathlib import Path
from argparse import ArgumentParser
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.Refiner import Refiner
from neb_dynamics.NEB import NEB, NoneConvergedException
from chain import Chain
from neb_dynamics.nodes.Node3D_TC import Node3D_TC

from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer

from neb_dynamics.nodes.node3d import Node3D
from neb_dynamics.nodes.Node3D_Water import Node3D_Water
from neb_dynamics.Janitor import Janitor
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.helper_functions import create_friction_optimal_gi

from neb_dynamics.constants import BOHR_TO_ANGSTROMS

import sys


def read_single_arguments():
    """
    Command line reader
    """
    description_string = "will take path to an xyz file of geodesic \
        trajectory and relax it using XTB neb"
    parser = ArgumentParser(description=description_string)

    parser.add_argument(
        "-c",
        "--charge",
        dest="c",
        type=int,
        default=0,
        help="total charge of syrgsstem",
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
        "-fp",
        "--filepath",
        dest="fp",
        type=str,
        help="path to TreeNode data to use as reference",
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
        "-dc",
        "--do_cleanup",
        dest="dc",
        type=int,
        default=0,
        help="whether to do conformer-conformer NEBs at the end",
    )

    parser.add_argument(
        "-nc",
        "--node_class",
        dest="nc",
        type=str,
        default="node3d",
        help="what node type to use. options are: node3d, \
            node3d_tc, node3d_tc_local, node3d_tcpb",
    )

    parser.add_argument(
        "-tol",
        "--tolerance",
        dest="tol",
        required=True,
        type=float,
        help="convergence threshold in H/Bohr^2",
        default=0.002,
    )

    parser.add_argument(
        "-sig",
        "--skip_identical_graphs",
        dest="sig",
        required=True,
        type=int,
        help="whether to skip optimizations for identical graph endpoints",
        default=1,
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
        default=0,
    )

    parser.add_argument(
        "-mr",
        "--maxima_recyling",
        dest="mr",
        required=False,
        type=int,
        help="whether to use maxima recyling",
        default=0,
    )

    parser.add_argument(
        "-es_ft",
        "--early_stop_ft",
        dest="es_ft",
        required=False,
        type=float,
        help="force threshold for early stopping",
        default=0.03,
    )

    parser.add_argument(
        "-es_rms",
        "--early_stop_rms",
        dest="es_rms",
        required=False,
        type=float,
        help="chain RMS threshold for early stopping",
        default=1,
    )

    parser.add_argument(
        "-es_ss",
        "--early_stop_still_steps",
        dest="es_ss",
        required=False,
        type=float,
        help="number of still steps until an early stop check",
        default=500,
    )

    parser.add_argument(
        "-preopt",
        "--preopt_with_xtb",
        dest="preopt",
        required=True,
        type=int,
        help="whether to preconverge chains using xtb",
        default=1,
    )

    parser.add_argument(
        "-met",
        "--method",
        dest="method",
        required=False,
        type=str,
        help="what type of NEB to use. options: [asneb, neb]",
        default="asneb",
    )

    parser.add_argument(
        "-delk",
        "--delta_k",
        dest="delk",
        required=False,
        type=float,
        help="parameter for the delta k",
        default=0.09,
    )

    return parser.parse_args()


def main():
    args = read_single_arguments()

    nodes = {
        "node3d": Node3D,
        "node3d_tc": Node3D_TC,
        "node3d_tc_local": Node3D_TC_Local,
        "node3d_tcpb": Node3D_TC_TCPB,
        "node3d_water": Node3D_Water,
    }
    # nodes = {"node3d": Node3D, "node3d_tc": Node3D_TC}
    nc = nodes[args.nc]

    if args.nc != "node3d":
        method = "uwb97xd3"  # terachem
        # method = "ub3lyp"  # terachem
        # method = 'ub3lyp'
        # method = "wb97x-d3" # psi4
        basis = "def2-svp"
        # basis = '3-21g'
        kwds = {"gpus": 1}
        # kwds = {'maxit':'500'}
        # kwds = {'dftd': 'd3-bj'}
        # method = 'ub3lyp'
        # basis = '3-21gs'
    #        kwds = {
    #         'convthre': 0.0001,
    #         'maxit':    700,
    #         'diismaxvecs':    20,
    #         'precision':   'mixed',
    #         'dftgrid':     1,
    #         'threall':     1.0e-14,
    # levelshift:    'yes'
    # levelshiftvala: 2.7
    # levelshiftvalb: 1.0
    #         'fon':   'yes',
    #         'fon_anneal': 'yes',
    #         'fon_target': 0.0,
    #         'fon_temperature':  0.1,
    #         'fon_anneal_iter':  500,
    #         'fon_converger': 'yes',
    #         'fon_coldstart': 'yes',
    #         'fon_tests': 1,
    #         'purify': 'no'
    #        }

    if args.nc == "node3d_tc_local":
        do_parallel = False
    else:
        do_parallel = True

    tol = args.tol

    cni = ChainInputs(
        k=0.1,
        delta_k=args.delk,
        node_class=nc,
        friction_optimal_gi=True,
        use_maxima_recyling=bool(args.mr),
        do_parallel=do_parallel,
        node_freezing=True,
        skip_identical_graphs=bool(args.sig),
    )

    optimizer = VelocityProjectedOptimizer(timestep=0.5, activation_tol=0.1)

    nbi = NEBInputs(
        tol=tol * BOHR_TO_ANGSTROMS,
        barrier_thre=0.1,  # kcalmol,
        climb=bool(args.climb),
        rms_grad_thre=tol * BOHR_TO_ANGSTROMS,
        max_rms_grad_thre=tol * BOHR_TO_ANGSTROMS * 2.5,
        ts_grad_thre=tol * BOHR_TO_ANGSTROMS * 2.5,
        ts_spring_thre=tol * BOHR_TO_ANGSTROMS * 1.5,
        v=1,
        max_steps=500,
        early_stop_chain_rms_thre=args.es_rms,
        early_stop_force_thre=args.es_ft * BOHR_TO_ANGSTROMS,
        early_stop_still_steps_thre=args.es_ss,
        _use_dlf_conv=False,
        preopt_with_xtb=bool(int(args.preopt)),
    )
    print(f"{args.preopt=}")
    print(f"NEBinputs: {nbi}\nChainInputs: {cni}\nOptimizer: {optimizer}")
    sys.stdout.flush()

    gii = GIInputs(nimages=args.nimg, extra_kwds={"sweep": False})
    h = TreeNode.read_from_disk(args.fp)

    if args.method == "asneb":
        m = MSMEP(
            neb_inputs=nbi,
            chain_inputs=cni,
            gi_inputs=gii,
            optimizer=optimizer,
        )

        refiner = Refiner(cni=cni)
        refined_leaves = refiner.create_refined_leaves(h.ordered_leaves)
        out_chain = refiner.join_output_leaves(refined_leaves)

        tot_grad_calls = sum([obj.data.grad_calls_made for obj in refined_leaves])
        geom_grad_calls = sum([obj.data.geom_grad_calls_made for obj in refined_leaves])
        print(f">>> Made {tot_grad_calls} gradient calls total.")
        print(
            f"<<< Made {geom_grad_calls} gradient for geometry\
               optimizations."
        )
        fp = Path(args.fp)
        data_dir = fp.parent

        if args.name:
            foldername = data_dir / args.name
            filename = data_dir / (args.name + ".xyz")

        else:
            foldername = data_dir / f"{fp.stem}_refined"
            filename = data_dir / f"{fp.stem}_refined.xyz"

        out_chain.write_to_disk(filename)
        refiner.write_leaves_to_disk(foldername, refined_leaves)

    elif args.method == "neb":
        # n = NEB(initial_chain=chain, parameters=nbi, optimizer=optimizer)
        # fp = Path(args.st)
        # data_dir = fp.parent
        # if args.name:
        #    filename = data_dir / (args.name + "_neb.xyz")

        # else:
        #    filename = data_dir / f"{fp.stem}_neb.xyz"

        # try:
        #   n.optimize_chain()

        #    n.write_to_disk(data_dir/f"{filename}", write_history=True)
        # except NoneConvergedException as e:
        #   e.obj.write_to_disk(data_dir/f"{filename}", write_history=True)

        # tot_grad_calls = n.grad_calls_made
        # print(f">>> Made {tot_grad_calls} gradient calls total.")
        print("Unsupported for now")
    else:
        raise ValueError("Incorrect input method. Use 'asneb' or 'neb'")


if __name__ == "__main__":
    main()
