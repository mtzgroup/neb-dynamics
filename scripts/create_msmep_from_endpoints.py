#!/home/jdep/.conda/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.tdstructure import TDStructure


from neb_dynamics.NEB import NEB
from neb_dynamics.Chain import Chain
from neb_dynamics.nodes.Node3D_TC import Node3D_TC

# from neb_dynamics.Node3D_TC_Local import Node3D_TC_Local
# from neb_dynamics.Node3D_TC_TCPB import Node3D_TC_TCPB
from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.Linesearch import Linesearch

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
        default=True,
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

    return parser.parse_args()


def main():
    args = read_single_arguments()

    # nodes = {'node3d': Node3D, 'node3d_tc': Node3D_TC, 'node3d_tc_local': Node3D_TC_Local, 'node3d_tcpb': Node3D_TC_TCPB}
    nodes = {"node3d": Node3D, "node3d_tc": Node3D_TC}
    nc = nodes[args.nc]

    start = TDStructure.from_xyz(args.st, tot_charge=args.c, tot_spinmult=args.s)
    end = TDStructure.from_xyz(args.en, tot_charge=args.c, tot_spinmult=args.s)

    if args.nc != "node3d":
        # method = 'b3lyp'
        method = "wb97xd3"
        basis = "def2-svp"
        # method = 'gfn2xtb'
        # basis = 'gfn2xtb'
        # kwds = {'reference':'uks'}
        start.tc_model_method = method
        end.tc_model_method = method

        start.tc_model_basis = basis
        end.tc_model_basis = basis

        if int(args.min_ends):
            start = start.tc_geom_optimization()
            end = end.tc_geom_optimization()

        # kwds = {'reference':'uks'}
        kwds = {"restricted": False}
        # kwds = {'restricted': False, 'pcm':'cosmo','epsilon':80}

    else:
        if int(args.min_ends):
            start = start.xtb_geom_optimization()
            end = end.xtb_geom_optimization()

    traj = Trajectory([start, end]).run_geodesic(nimages=args.nimg)

    if args.nc == "node3d_tcpb" or args.nc == "node3d_tc_local":
        do_parallel = False
    else:
        do_parallel = True

    tol = args.tol

    # cni = ChainInputs(k=0.1,delta_k=0.09, node_class=nc,friction_optimal_gi=True, do_parallel=do_parallel, node_freezing=True)
    cni = ChainInputs(
        k=0.01,
        node_class=nc,
        friction_optimal_gi=True,
        do_parallel=do_parallel,
        node_freezing=True,
    )
    # cni = ChainInputs(k=0, node_class=nc,friction_optimal_gi=True, do_parallel=do_parallel, node_freezing=False)
    # cni = ChainInputs(k=0, node_class=nc,friction_optimal_gi=True, do_parallel=do_parallel, node_freezing=True)
    # optimizer = BFGS(bfgs_flush_steps=1000, bfgs_flush_thre=0.1, step_size=0.33*traj[0].atomn, min_step_size=0.01*traj[0].atomn)
    optimizer = Linesearch(
        step_size=0.33 * traj[0].atomn, min_step_size=0.001 * traj[0].atomn
    )
    # nbi = NEBInputs(grad_thre=tol*BOHR_TO_ANGSTROMS,
    #            rms_grad_thre=(tol/2)*BOHR_TO_ANGSTROMS,
    #            en_thre=(tol/10)*BOHR_TO_ANGSTROMS,
    #            v=1,
    #            max_steps=500,
    #            early_stop_chain_rms_thre=0.002,
    #            early_stop_force_thre=0.003,

    #            early_stop_still_steps_thre=500,
    #            vv_force_thre=0.0)
    nbi = NEBInputs(
        grad_thre=tol * BOHR_TO_ANGSTROMS,
        rms_grad_thre=(tol / 2) * BOHR_TO_ANGSTROMS,
        en_thre=(tol)
        * BOHR_TO_ANGSTROMS,  # loose energy threshold cause DLF doesnt use en thresh
        v=1,
        max_steps=500,
        early_stop_chain_rms_thre=1,  # not really caring about chain distances
        early_stop_force_thre=tol * 10 * BOHR_TO_ANGSTROMS,
        early_stop_still_steps_thre=100,
        vv_force_thre=0.0,
        _use_dlf_conv=True,
    )

    gii = GIInputs(nimages=args.nimg, extra_kwds={"sweep": False})
    traj = create_friction_optimal_gi(traj, gii)
    chain = Chain.from_traj(traj=traj, parameters=cni)
    m = MSMEP(
        neb_inputs=nbi,
        chain_inputs=cni,
        gi_inputs=gii,
        skip_identical_graphs=bool(args.sig),
        optimizer=optimizer,
    )
    history, out_chain = m.find_mep_multistep(chain)

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
