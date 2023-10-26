#!/home/jdep/.conda/envs/rp/bin/python
from pathlib import Path
from argparse import ArgumentParser
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs


from neb_dynamics.ReactionProfileGenerator import ReactionProfileGenerator

import numpy as np

def read_single_arguments():
    """
    Command line reader
    """
    description_string = "will take path to an xyz of an MSMEP folder"
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
        '-m',
        '--method',
        dest='m',
        type=str,
        default="wb97xd3",
        help='what electronic structure method to use'
        
    )
    
    parser.add_argument(
        '-b',
        '--basis',
        dest='b',
        type=str,
        default="def2-svp",
        help='what electronic structure basis set to use'
        
    )
    
    return parser.parse_args()


def main():
    # import os
    # del os.environ['OE_LICENSE']
    args = read_single_arguments()

    fp = Path(args.f)
    
    adj_mat_fp = fp / 'adj_matrix.txt'
    adj_mat = np.loadtxt(adj_mat_fp)
    if adj_mat.size == 1:
        chain_list =  [Chain.from_xyz(fp / f'node_0.xyz' , ChainInputs())]
    else:
    
        a = np.sum(adj_mat,axis=1)
        inds_leaves = np.where(a == 1)[0] 
        chain_list = [Chain.from_xyz(fp / f'node_{ind}.xyz', ChainInputs()) for ind in inds_leaves]
    
    ref_td = chain_list[0][0].tdstructure
    method = args.m
    basis = args.b
    kwds = {'reference':'uks'}
    # kwds = {'restricted': False}
    # kwds = {'restricted': False, 'pcm':'cosmo','epsilon':80}  
    
    all_inps = []
    all_rpg = []
    for chain in chain_list:
        rpg = ReactionProfileGenerator(input_obj=chain, method=method, basis=basis, kwds=kwds)
        inps = rpg.create_pseudo_irc_inputs()
        all_inps.extend(inps)
        all_rpg.append(rpg)
        
    all_results = rpg.compute_and_return_results(all_inps)
    print("all inps:", all_inps)
    
    all_profiles = []
    for i, rpg in zip(range(0, len(all_results), 2), all_rpg):
        all_profiles.append(rpg.create_profile(all_results[i:i+2]))
        
        
    dft_dir = fp.parent / f"RPG_{method}-{basis}"
    dft_dir.mkdir(exist_ok=True)
    print("--->", all_profiles)
    for i, chain in enumerate(all_profiles):
        chain.write_to_disk(dft_dir / f'chain_{i}.xyz')
    
        
    
        
	    

if __name__ == "__main__":
    main()
