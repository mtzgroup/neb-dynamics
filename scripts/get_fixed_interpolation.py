#!/Users/janestrada/opt/anaconda3/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser


from retropaths.abinitio.inputs import Inputs
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.rootstructure import RootStructure
from retropaths.abinitio.sampler import Sampler


from neb_dynamics.NEB import Chain, Node3D
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.remapping_helpers import create_correct_interpolation
import numpy as np
import matplotlib.pyplot as plt









def read_single_arguments():
    """
    Command line reader
    """
    description_string = "will take path to an xyz file of geodesic trajectory and relax it using XTB neb"
    parser = ArgumentParser(description=description_string)


    parser.add_argument(
        "-o",
        "--out",
        dest="o",
        type=Path,
        help='out path for the results'
    )

    parser.add_argument(
        "-rp",
        "--reaction_path",
        dest="rp",
        type=Path,
        help='path to the reactions.p file'
    )

    parser.add_argument(
        "-rn",
        "--reaction_name",
        dest="rn",
        type=str,
        help='name of the reaction'
    )

    parser.add_argument(
        "-s",
        "--start",
        dest="start",
        type=int,
        help='index of start conformer'
    )

    parser.add_argument(
        "-e",
        "--end",
        dest="end",
        type=int,
        help='index of end conformer',
        required=True
    )

    

    
    return parser.parse_args()


def main():
    args = read_single_arguments()
    out_path = args.o
    rxn_name = args.rn
    reaction_path = args.rp


    start_ind = args.start
    end_ind = args.end
    # reaction_path = Path("/Users/janestrada/Retropaths/retropaths/data/reactions.p")
    # out_path=Path("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement")


    inp = Inputs(rxn_name=rxn_name, reaction_file=reaction_path)
    root_structure = RootStructure(
        root=TDStructure.from_rxn_name(rxn_name, data_folder=reaction_path.parent),
        master_path=out_path,
        rxn_args=inp,
        trajectory=Trajectory([]),
    )

    root_conformers, root_energies = root_structure.crest(
        tdstruct=root_structure.root, id="root"
    ).conf_and_energy
    transformed_conformers, transformed_energies = root_structure.crest(
        tdstruct=root_structure.transformed,
        id="transformed"
    ).conf_and_energy
    sampler = Sampler(mode="distance")
    sub_root_conformers, _  = sampler.run(
    conformers_to_subsample=root_conformers,
    bonds_between_frags=root_structure._get_bonds_between_fragments(),
    energies=root_energies,
    cutoff=7
    ) 
    sub_trans_conformers, _ = sampler.run(
    conformers_to_subsample=transformed_conformers,
    bonds_between_frags=root_structure._get_bonds_between_fragments(),
    energies=transformed_energies,
    cutoff=7
    )

    subselected_conf_pairs = sampler._get_conf_pairs(
    start_conformers=sub_root_conformers,
    end_conformers=sub_trans_conformers
    )
    root_conformers = sub_root_conformers
    transformed_conformers = sub_trans_conformers

    trajs = create_correct_interpolation(start_ind, end_ind, root_conformers=root_conformers, transformed_conformers=transformed_conformers)
    for i, traj_array in enumerate(trajs):
        traj = Trajectory(traj_array, tot_charge=0, tot_spinmult=1)
        traj.write_trajectory(out_path/f'traj_{start_ind}-{end_ind}_{i}.xyz')
    
        
	    

if __name__ == "__main__":
    main()
