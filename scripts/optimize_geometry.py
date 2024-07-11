#!/home/jdep/.conda/envs/neb/bin/python
from pathlib import Path
from argparse import ArgumentParser
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure


from neb_dynamics.NEB import NEB
from neb_dynamics.Chain import Chain
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB

from neb_dynamics.nodes.Node3D import Node3D
from neb_dynamics.Janitor import Janitor
from neb_dynamics.nodes.Node3D_gfn1xtb import Node3D_gfn1xtb
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
import numpy as np

def read_single_arguments():
    """
    Command line reader
    """
    description_string = "will take path to an xyz file of geodesic trajectory and relax it using XTB neb"
    parser = ArgumentParser(description=description_string)


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
        '-nc',
        '--node_class',
        dest='nc',
        type=str,
        default="node3d",
        help='what node type to use. options are: node3d, node3d_tc, node3d_tc_local, node3d_tcpb'

    )

    parser.add_argument(
        '-fp',
        '--file_path',
        dest='fp',
        required=True,
        type=str,
        help='path to the xyz structure'

    )

    return parser.parse_args()


def main():
    args = read_single_arguments()

    nodes = {'node3d': Node3D, 'node3d_tc': Node3D_TC, 'node3d_tc_local': Node3D_TC_Local, 'node3d_tcpb': Node3D_TC_TCPB}
    nc = nodes[args.nc]

    fp = Path(args.fp)

    td = TDStructure.from_xyz(args.fp, tot_charge=args.c, tot_spinmult=args.s)

    if args.nc != "node3d":
        method = 'wb97xd3'
        basis = 'def2-svp'
        kwds = {'restricted': False}
        # kwds = {'restricted': False, 'pcm':'cosmo','epsilon':80}
        td.tc_model_method = method
        td.tc_model_basis = basis
        td.tc_kwds = kwds

    node = nodes[args.nc](td)
    node_opt = node.do_geometry_optimization()
    node_opt.tdstructure.to_xyz(fp.parent / (fp.stem+"_opt.xyz"))




if __name__ == "__main__":
    main()
