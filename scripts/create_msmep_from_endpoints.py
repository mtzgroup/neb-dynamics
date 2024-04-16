#!/home/jdep/.conda/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.tdstructure import TDStructure


from neb_dynamics.NEB import NEB
from neb_dynamics.Chain import Chain
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
        "-dc",
        "--do_cleanup",
        dest="dc",
        type=bool,
        default=False,
        help="whether to do conformer-conformer NEBs at the end",
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
        "-st",
        "--start",
        dest="st",
        required=True,
        type=str,
        help="path to the first xyz structure",
    )

    parser.add_argument(
        "-en",
        "--end",
        dest="en",
        required=True,
        type=str,
        help="path to the final xyz structure",
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
        "-sig",
        "--skip_identical_graphs",
        dest="sig",
        required=True,
        type=int,
        help="whether to skip optimizations for identical graph endpoints",
        default=1,
    )

    parser.add_argument(
        "-min_ends",
        "--minimize_endpoints",
        dest="min_ends",
        required=True,
        type=int,
        help="whether to minimize the endpoints before starting",
        default=False,
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

    nodes = {'node3d': Node3D, 'node3d_tc': Node3D_TC, 'node3d_tc_local': Node3D_TC_Local, 'node3d_tcpb': Node3D_TC_TCPB}
    # nodes = {"node3d": Node3D, "node3d_tc": Node3D_TC}
    nc = nodes[args.nc]

    start = TDStructure.from_xyz(args.st, tot_charge=args.c, tot_spinmult=args.s)
    end = TDStructure.from_xyz(args.en, tot_charge=args.c, tot_spinmult=args.s)

    if args.nc != "node3d":
        method = "wb97xd3" #terachem
        # method = "wb97x-d3" # psi4
        basis = "def2-svp"
        kwds = {'maxit':'500'}
        #method = 'ub3lyp'
        #basis = '3-21gs'
#        kwds = {
#         'convthre': 0.0001,
#         'maxit':    700,
#         'diismaxvecs':    20,
#         'precision':   'mixed',
#         'dftgrid':     1,
#         'threall':     1.0e-14,
         #levelshift:    'yes'
         #levelshiftvala: 2.7
         #levelshiftvalb: 1.0
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



        start.tc_model_method = method
        end.tc_model_method = method

        start.tc_model_basis = basis
        end.tc_model_basis = basis
        
        start.tc_kwds =  kwds
        end.tc_kwds = kwds

        if int(args.min_ends):
            start = start.tc_local_geom_optimization()
            end = end.tc_local_geom_optimization()


    else:
        if int(args.min_ends):
            start = start.xtb_geom_optimization()
            end = end.xtb_geom_optimization()

    traj = Trajectory([start, end]).run_geodesic(nimages=args.nimg)

    if args.nc == "node3d_tc_local":
        do_parallel = False
    else:
        do_parallel = True

    tol = args.tol

    # cni = ChainInputs(k=0.1,delta_k=0.09, node_class=nc,friction_optimal_gi=True, do_parallel=do_parallel, node_freezing=True)
    cni = ChainInputs(
        k=0.1,
        delta_k=0.09,
        node_class=nc,
        friction_optimal_gi=True,
        use_maxima_recyling=bool(args.mr),
        do_parallel=do_parallel,
        node_freezing=True,
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
    sys.stdout.flush()

    gii = GIInputs(nimages=args.nimg, extra_kwds={"sweep": False})
    traj = create_friction_optimal_gi(traj, gii)
    print(traj[0].spinmult)
    chain = Chain.from_traj(traj=traj, parameters=cni)
    m = MSMEP(
        neb_inputs=nbi,
        chain_inputs=cni,
        gi_inputs=gii,
        skip_identical_graphs=bool(args.sig),
        optimizer=optimizer,
    )
    history, out_chain = m.find_mep_multistep(chain)
    
    leaves_nebs = [obj for obj in history.get_optimization_history() if obj]
    tot_grad_calls = sum([obj.grad_calls_made for obj in leaves_nebs])
    print(f">>> Made {tot_grad_calls} gradient calls total.")

    fp = Path(args.st)
    data_dir = fp.parent

    if args.name:
        foldername = data_dir / args.name
        filename = data_dir / (args.name + ".xyz")

    else:
        foldername = data_dir / f"{fp.stem}_msmep"
        filename = data_dir / f"{fp.stem}_msmep.xyz"

    out_chain.to_trajectory().write_trajectory(filename)
    history.write_to_disk(foldername)

    if args.dc:
        op = data_dir / f"{filename.stem}_cleanups"
        j = Janitor(history_object=history, out_path=op, msmep_object=m)

        clean_msmep = j.create_clean_msmep()

        if clean_msmep:
            clean_msmep.to_trajectory().write_trajectory(
                data_dir / f"{fp.stem}_msmep_clean.xyz"
            )


if __name__ == "__main__":
    main()
