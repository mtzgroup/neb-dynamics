from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.helper_functions import RMSD
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs
from retropaths.reactions.pot import Pot

import numpy as np
import matplotlib.pyplot as plt
import contextlib, os

from pathlib import Path
import subprocess
from itertools import product
import itertools
from dataclasses import dataclass


@dataclass
class NetworkBuilder:
    start: TDStructure
    end: TDStructure
    
    data_dir: Path
    
    n_max_conformers: int = 10
    subsample_confs: bool = True
    conf_rmsd_cutoff: float = 0.5
    
    use_slurm: bool = False
    
    verbose: bool = True
    
    
    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        
        self.start_confs_dd = self.data_dir / 'start_confs'
        self.start_confs_dd.mkdir(exist_ok=True)

        self.end_confs_dd = self.data_dir / 'end_confs'
        self.end_confs_dd.mkdir(exist_ok=True)
        
        self.pairs_data_dir = self.data_dir / 'pairs_to_do'
        self.pairs_data_dir.mkdir(exist_ok=True)
        
        self.msmep_data_dir = self.data_dir / 'msmep_results'
        self.msmep_data_dir.mkdir(exist_ok=True)
        
        self.submissions_dir = self.data_dir / "submissions"
        self.submissions_dir.mkdir(exist_ok=True)
        
        self.leaf_objects = None
        
        
    
    
    @contextlib.contextmanager
    def remember_cwd(self):
        curdir= os.getcwd()
        try: yield
        finally: os.chdir(curdir)

    def sample_all_conformers(self, td: TDStructure, dd: Path, fn: str):
        confs_fp = dd / fn
        td.to_xyz(confs_fp)
        with self.remember_cwd():
            os.chdir(dd)

            results_conf_fp = dd / 'crest_conformers.xyz'
            results_rots_fp = dd / 'crest_rotamers.xyz'

            if results_conf_fp.exists() and results_rots_fp.exists():
                if self.verbose: print("\tConformers already computed.")
            else:
                if self.verbose: print("\tRunning conformer sampling...")
                output = subprocess.run(['crest',f'{str(confs_fp.resolve())}','--gfn2'], capture_output=True)
                with open(dd / 'crest_output.txt', 'w+') as fout:
                    fout.write(output.stdout.decode("utf-8"))
                if self.verbose: print("\tDone!")

            conformers = Trajectory.from_xyz(results_conf_fp)
            rotamers = Trajectory.from_xyz(results_rots_fp)
            all_confs = Trajectory(traj=conformers.traj+rotamers.traj)

        return all_confs
    
    @staticmethod
    def subselect_confomers(conformer_pool, n_max=20, rmsd_cutoff=0.5):

        subselected_pool = [conformer_pool[0]]
        
        for conf in conformer_pool[1:]:
            distances = []
            for reference in subselected_pool:
                dist = RMSD(conf.coords, reference.coords)[0]
                distances.append(dist >= rmsd_cutoff)
            if all(distances) and len(subselected_pool) < n_max:
                subselected_pool.append(conf)
        return subselected_pool
    
    
    def create_endpoint_conformers(self):


        start_conformers = self.sample_all_conformers(td=self.start, dd=self.start_confs_dd, fn='start.xyz')
        end_conformers = self.sample_all_conformers(td=self.end, dd=self.end_confs_dd, fn='end.xyz')
        
        if self.subsample_confs:
            ### subselect to n conformers for each
            sub_start_confs = self.subselect_confomers(conformer_pool=start_conformers, 
                                                       n_max=self.n_max_conformers, 
                                                       rmsd_cutoff=self.conf_rmsd_cutoff)
            sub_end_confs = self.subselect_confomers(conformer_pool=end_conformers, 
                                                     n_max=self.n_max_conformers, 
                                                     rmsd_cutoff=self.conf_rmsd_cutoff)
            start_conformers = sub_start_confs
            end_conformers = sub_end_confs
        return start_conformers, end_conformers
    
    def create_pairs_of_structures(self, start_confs, end_confs):
        # generate all pairs of structures
        pairs_to_do = list(itertools.product(start_confs, end_confs))
        return pairs_to_do
        
    def create_submission_scripts(self):
        start_confs, end_confs = self.create_endpoint_conformers()
        pairs_to_do = self.create_pairs_of_structures(start_confs=start_confs, end_confs=end_confs)

        

        template = ['#!/bin/bash',
                     '',
                     '#SBATCH -t 2:00:00',
                     '#SBATCH -J nebjan_test',
                     '#SBATCH --qos=gpu_short',
                     '#SBATCH --gres=gpu:1',
                     '',
                     'work_dir=$PWD',
                     '',
                     'cd $SCRATCH',
                     '',
                     '# Load modules',
                     'ml TeraChem',
                     'source /home/jdep/.bashrc',
                     'source activate neb',
                     'export OMP_NUM_THREADS=1',
                     '# Run the job',
                     'create_msmep_from_endpoints.py ']


        

        for i, (start, end) in enumerate(pairs_to_do):

            if self.verbose: print(f'\t***Creating pair {i} submission')
            start_fp = self.pairs_data_dir / f'start_pair_{i}.xyz'
            end_fp = self.pairs_data_dir / f'end_pair_{i}.xyz'
            start.to_xyz(start_fp)
            end.to_xyz(end_fp)
            out_fp = self.msmep_data_dir / f'results_pair{i}_msmep'


            cmd = f"create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.001 -sig 1 -mr 1 -nc node3d -preopt 0 -climb 0 -nimg 12 -name {out_fp} -min_ends 1"

            new_template = template.copy()
            new_template[-1] = cmd

            with open(self.submissions_dir / f'submission_{i}.sh', 'w+') as f:
                f.write("\n".join(new_template))
                
    def run_msmeps(self):
        ## submit all jobs
        all_jobs = list(self.submissions_dir.glob("*.sh"))


        for job in all_jobs:
            if self.verbose: print('\t',job)

            command = open(job).read().splitlines()[-1]
            out_fp = Path(command.split()[-3])
            if not out_fp.exists():
                if self.use_slurm:
                    with self.remember_cwd():
                        os.chdir(self.submissions_dir)
                        if self.verbose: print(f'\t\tsubmitting {job}')
                        _ = subprocess.run(['sbatch',f'{job}'], capture_output=True)


                else:
                    if self.verbose: print(f'\t\trunning {job}')
                    out = subprocess.run(command.split(), capture_output=True)
                    with open(out_fp.parent / f'out_{out_fp.name}', 'w+') as fout:
                        fout.write(out.stdout.decode("utf-8"))
            else:
                if self.verbose: print("\t\t\talready done")

    def _get_ind_mol(self, ref_list, mol):
        inds = np.where([a.is_isomorphic_to(b) for a,b in list(itertools.product([mol], ref_list))])[0]
        assert len(inds) == 1, "Too many matches. Something is bad."
        return inds[0]

    def _get_relevant_leaves(self, fp):
        adj_mat_fp = fp / 'adj_matrix.txt'
        adj_mat = np.loadtxt(adj_mat_fp)
        if adj_mat.size == 1:
            return [Chain.from_xyz(fp / f'node_0.xyz', ChainInputs(k=0.1, delta_k=0.09))]
        else:

            a = np.sum(adj_mat,axis=1)
            inds_leaves = np.where(a == 1)[0] 
            chains = [Chain.from_xyz(fp / f'node_{ind}.xyz',ChainInputs(k=0.1, delta_k=0.09, use_maxima_recyling=True)) for ind in inds_leaves]
            return chains

    def _get_relevant_edges(self, edge_dir, ind):
        relevant_keys = []
        for key, val in edge_dir.items():
            node_edges_str = key.split("-")
            node_edges = [int(v) for v in node_edges_str]
            if ind in node_edges:
                relevant_keys.append(key)
        return relevant_keys

    def _std_edge(self, edgelabel, ind):
        vals = edgelabel.split("-")
        if ind == int(vals[0]):
            return edgelabel
        else:
            return f"{vals[1]}-{vals[0]}"


    def create_rxn_network(self):
        msmep_paths = list(self.msmep_data_dir.glob("results*_msmep"))

        molecules = []
        edges = {}
        leaf_objects = {}
        for fp in msmep_paths:
            if self.verbose: print(f"\tDoing: {fp}. Len: {len(molecules)}")
            leaves = self._get_relevant_leaves(fp)
            for leaf in leaves:
                reactant = leaf[0].tdstructure.molecule_rp
                product = leaf[-1].tdstructure.molecule_rp
                out_leaf = leaf
                out_leaf_rev = out_leaf.copy()
                out_leaf_rev.nodes.reverse()

                es = len(out_leaf) == 12
                # es = out_leaf.is_elem_step()[0]

                if es:
                    if reactant not in molecules:
                        molecules.append(reactant)
                    if product not in molecules:
                        molecules.append(product)

                    ind_r = self._get_ind_mol(molecules, reactant)
                    ind_p = self._get_ind_mol(molecules, product)
                    eA = leaf.get_eA_chain()
                    edge_name = f'{ind_r}-{ind_p}'
                    rev_edge_name = f'{ind_p}-{ind_r}'
                    rev_eA = (leaf.energies.max() - leaf[-1].energy)*627.5

                    if edge_name in edges.keys():
                        edges[edge_name].append(eA)
                        leaf_objects[edge_name].append(out_leaf)
                    else:
                        edges[edge_name] = [eA]
                        leaf_objects[edge_name] = [out_leaf]

                    if rev_edge_name in edges.keys():
                        edges[rev_edge_name].append(rev_eA)
                        leaf_objects[rev_edge_name].append(out_leaf_rev)

                    else:
                        edges[rev_edge_name] = [rev_eA]
                        leaf_objects[rev_edge_name] = [out_leaf_rev]


        ind_start = self._get_ind_mol(molecules, leaves[0][0].tdstructure.molecule_rp)

        reactant = molecules[0]

        pot = Pot(root=reactant)

        for i, mol_to_add in enumerate(molecules):
            node_ind = i
            relevant_keys = self._get_relevant_edges(edges, node_ind)

            pot.graph.add_node(node_ind, molecule=mol_to_add, converged=False)
            rel_edges = self._get_relevant_edges(edges, node_ind)
            # print(rel_edges)
            for edgelabel in rel_edges:
                label = edgelabel
                vals = label.split("-")
                label_rev = f"{vals[1]}-{vals[0]}"
                pot.graph.add_edge(int(vals[1]), node_ind, reaction=f'eA ({edgelabel}): {np.min(edges[edgelabel])} || eA ({label_rev}):{np.min(edges[label_rev])}')


        self.leaf_objects = leaf_objects

        return pot
    
    
    def run_and_return_network(self):
        print("1... Creating Conformers with CREST")
        self.create_endpoint_conformers()
        print("2... Creating submission scripts")
        self.create_submission_scripts()
        print("3... Running NEB minimizations")
        self.run_msmeps()
        print("4... Creating reaction network")
        pot = self.create_rxn_network()
        print("Done!")
        return pot
        



