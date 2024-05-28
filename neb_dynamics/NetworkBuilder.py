from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.helper_functions import RMSD
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs

from scipy.signal import argrelmin
import tqdm

# from retropaths.reactions.pot import Pot
from neb_dynamics.pot import Pot

import numpy as np
import matplotlib.pyplot as plt
import contextlib
import os

from pathlib import Path
import subprocess
import itertools
from dataclasses import dataclass


@dataclass
class NetworkBuilder:

    data_dir: Path

    start: TDStructure = None
    end: TDStructure = None

    n_max_conformers: int = 10
    subsample_confs: bool = True
    conf_rmsd_cutoff: float = 0.5

    use_slurm: bool = False

    verbose: bool = True

    tolerate_kinks: bool = True

    CREST_temp: float = 298.15  # Kelvin
    CREST_ewin: float = 6.0  # kcal/mol

    chain_inputs: ChainInputs = ChainInputs()

    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)

        self.start_confs_dd = self.data_dir / "start_confs"
        self.start_confs_dd.mkdir(exist_ok=True)

        self.end_confs_dd = self.data_dir / "end_confs"
        self.end_confs_dd.mkdir(exist_ok=True)

        self.pairs_data_dir = self.data_dir / "pairs_to_do"
        self.pairs_data_dir.mkdir(exist_ok=True)

        self.msmep_data_dir = self.data_dir / "msmep_results"
        self.msmep_data_dir.mkdir(exist_ok=True)

        self.submissions_dir = self.data_dir / "submissions"
        self.submissions_dir.mkdir(exist_ok=True)

        self.leaf_objects = None

        if self.start is None:
            start_fp = self.start_confs_dd / "start.xyz"
            assert (
                start_fp.exists()
            ), f"Need to input a start geometry since {start_fp} does not exist"
            print(
                f"Warning: reading structure info from {start_fp}. Assuming charge=0 and spinmult=1"
            )
            self.start = TDStructure.from_xyz(start_fp)

        if self.end is None:
            end_fp = self.end_confs_dd / "end.xyz"
            assert (
                start_fp.exists()
            ), f"Need to input a start geometry since {end_fp} does not exist"
            print(
                f"Warning: reading structure info from {end_fp}. Assuming charge=0 and spinmult=1"
            )
            self.end = TDStructure.from_xyz(end_fp)

    @contextlib.contextmanager
    def remember_cwd(self):
        curdir = os.getcwd()
        try:
            yield
        finally:
            os.chdir(curdir)

    def sample_all_conformers(self, td: TDStructure, dd: Path, fn: str):
        confs_fp = dd / fn
        td.to_xyz(confs_fp)
        with self.remember_cwd():
            os.chdir(dd)

            results_conf_fp = dd / "crest_conformers.xyz"
            fps_confomers = list(dd.glob("crest_conf*.xyz"))
            fps_rotamers = list(dd.glob("crest_rot*.xyz"))

            # results_rots_fp = dd / 'crest_rotamers.xyz'

            # if results_conf_fp.exists() and results_rots_fp.exists():
            if len(fps_confomers) >= 1 and len(fps_rotamers) >= 1:
                if self.verbose:
                    print("\tConformers already computed.")
            else:
                if self.verbose:
                    print("\tRunning conformer sampling...")
                output = subprocess.run(
                    [
                        "crest",
                        f"{str(confs_fp.resolve())}",
                        f"-ewin {self.CREST_ewin}",
                        f"-temp {self.CREST_temp}",
                        "--gfn2",
                    ],
                    capture_output=True,
                )
                if self.verbose:
                    print(
                        f"\tWriting CREST output stream to {str((dd / 'crest_output.txt').resolve())}..."
                    )
                with open(dd / "crest_output.txt", "w+") as fout:
                    fout.write(output.stdout.decode("utf-8"))
                if self.verbose:
                    print("\tDone!")

                fps_confomers = list(dd.glob("crest_conf*.xyz"))
                fps_rotamers = list(dd.glob("crest_rot*.xyz"))

            conformers_trajs = []
            for conf_fp in fps_confomers:
                tr = Trajectory.from_xyz(conf_fp)
                print(f"\t\tCREST found {len(tr)} conformers")
                conformers_trajs.extend(tr.traj)
            for rot_fp in fps_rotamers:
                tr = Trajectory.from_xyz(rot_fp)
                if self.verbose:
                    print(f"\t\tCREST found {len(tr)} rotamers")
                conformers_trajs.extend(tr.traj)

            all_confs = Trajectory(traj=conformers_trajs)

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

        start_conformers = self.sample_all_conformers(
            td=self.start, dd=self.start_confs_dd, fn="start.xyz"
        )
        end_conformers = self.sample_all_conformers(
            td=self.end, dd=self.end_confs_dd, fn="end.xyz"
        )

        if self.subsample_confs:
            ### subselect to n conformers for each
            sub_start_confs = self.subselect_confomers(
                conformer_pool=start_conformers,
                n_max=self.n_max_conformers,
                rmsd_cutoff=self.conf_rmsd_cutoff,
            )
            sub_end_confs = self.subselect_confomers(
                conformer_pool=end_conformers,
                n_max=self.n_max_conformers,
                rmsd_cutoff=self.conf_rmsd_cutoff,
            )
            start_conformers = sub_start_confs
            end_conformers = sub_end_confs
        return start_conformers, end_conformers

    def create_pairs_of_structures(self, start_confs, end_confs):
        # generate all pairs of structures
        pairs_to_do = list(itertools.product(start_confs, end_confs))
        return pairs_to_do

    def create_submission_scripts(
        self,
        pairs_to_do=None,
        start_confs=None,
        end_confs=None,
        file_stem_name="pair",
        sub_id_name="",
    ):
        if start_confs is None and end_confs is None:
            start_confs, end_confs = self.create_endpoint_conformers()
        elif start_confs is None and end_confs is not None:
            start_confs, _ = self.create_endpoint_conformers()

        elif start_confs is not None and end_confs is None:
            _, end_confs = self.create_endpoint_conformers()

        pairs_to_do = self.create_pairs_of_structures(
            start_confs=start_confs, end_confs=end_confs
        )

        template = [
            "#!/bin/bash",
            "",
            "#SBATCH -t 12:00:00",
            "#SBATCH -J nebjan_test",
            "#SBATCH --qos=gpu_short",
            "#SBATCH --gres=gpu:1",
            "",
            "work_dir=$PWD",
            "",
            "cd $SCRATCH",
            "",
            "# Load modules",
            "ml TeraChem",
            "source /home/jdep/.bashrc",
            "source activate neb",
            "export OMP_NUM_THREADS=1",
            "# Run the job",
            "create_msmep_from_endpoints.py ",
        ]

        for i, (start, end) in enumerate(pairs_to_do):

            if self.verbose:
                print(f"\t***Creating pair {i} submission")
            start_fp = self.pairs_data_dir / f"start_{file_stem_name}_{i}.xyz"
            end_fp = self.pairs_data_dir / f"end_{file_stem_name}_{i}.xyz"
            start.to_xyz(start_fp)
            end.to_xyz(end_fp)
            out_fp = self.msmep_data_dir / f"results_{file_stem_name}{i}_msmep"

            cmd = f"create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.001 -sig {int(self.chain_inputs.skip_identical_graphs)} -mr {int(self.chain_inputs.use_maxima_recyling)} -nc node3d -preopt 0 -climb 0 -nimg 12 -min_ends 1 -es_ft 0.5 -name {out_fp}"

            new_template = template.copy()
            new_template[-1] = cmd

            with open(
                self.submissions_dir / f"submission_{sub_id_name}{i}.sh", "w+"
            ) as f:
                f.write("\n".join(new_template))

    def run_msmeps(self):
        ## submit all jobs
        all_jobs = list(self.submissions_dir.glob("*.sh"))

        for job in all_jobs:
            if self.verbose:
                print("\t", job)

            command = open(job).read().splitlines()[-1]
            out_fp = Path(command.split()[-1])
            if not out_fp.exists():
                if self.use_slurm:
                    with self.remember_cwd():
                        os.chdir(self.submissions_dir)
                        if self.verbose:
                            print(f"\t\tsubmitting {job}")
                        _ = subprocess.run(["sbatch", f"{job}"], capture_output=True)

                else:
                    if self.verbose:
                        print(f"\t\trunning {job}")
                    out = subprocess.run(command.split(), capture_output=True)
                    if self.verbose:
                        print(f"\t\t\twriting stdout in {out_fp.parent.resolve()}")
                    with open(out_fp.parent / f"out_{out_fp.name}", "w+") as fout:
                        fout.write(out.stdout.decode("utf-8"))
                        fout.write(out.stderr.decode("utf-8"))
                    with open(out_fp.parent / f"stderr_{out_fp.name}", "w+") as fout:
                        fout.write(out.stderr.decode("utf-8"))
            else:
                if self.verbose:
                    print("\t\t\talready done")

    def _get_ind_mol(self, ref_list, mol):
        inds = np.where(
            [a.is_isomorphic_to(b) for a, b in list(itertools.product([mol], ref_list))]
        )[0]
        assert len(inds) >= 1, "No matches found. Network cannot be constructed."
        return inds[0]

    def _get_ind_td(self, ref_list, td):
        inds = np.where(
            [a.is_identical(b) for a, b in list(itertools.product([td], ref_list))]
        )[0]
        assert len(inds) >= 1, "No matches found. Network cannot be constructed."
        return inds[0]

    def _get_relevant_leaves(self, fp):
        adj_mat_fp = fp / "adj_matrix.txt"
        adj_mat = np.loadtxt(adj_mat_fp)
        if adj_mat.size == 1:
            return [
                Chain.from_xyz(fp / "node_0.xyz", ChainInputs(k=0.1, delta_k=0.09))
            ]
        else:

            a = np.sum(adj_mat, axis=1)
            inds_leaves = np.where(a == 1)[0]
            chains = [
                Chain.from_xyz(
                    fp / f"node_{ind}.xyz",
                    ChainInputs(k=0.1, delta_k=0.09, use_maxima_recyling=True),
                )
                for ind in inds_leaves
            ]
            return chains

    def _get_relevant_conformers(self, node_ind: int):
        all_conformers = []

        for key in self.leaf_objects.keys():

            pull_reactants = False
            pull_products = False
            vals = key.split("-")
            if node_ind == int(vals[0]):
                pull_reactants = True

            elif node_ind == int(vals[1]):
                pull_products = True

            if pull_reactants:
                for chain in self.leaf_objects[key]:
                    all_conformers.append(chain[0])

            elif pull_products:
                for chain in self.leaf_objects[key]:
                    all_conformers.append(chain[-1])

        return all_conformers

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

    def _energies_fail(self, chain: Chain):
        try:
            chain.gradients
            return False
        except:
            return True

    def create_rxn_network(self):
        msmep_paths = list(self.msmep_data_dir.glob("results*_msmep"))

        molecules = []
        structures = []  # list of Node3D
        edges = {}
        leaf_objects = {}
        for fp in tqdm.tqdm(msmep_paths):
            if self.verbose:
                print(f"\tDoing: {fp}. Len: {len(molecules)}")

            leaves = self._get_relevant_leaves(fp)
            for leaf in leaves:
                # reactant = leaf[0].tdstructure.molecule_rp
                # product = leaf[-1].tdstructure.molecule_rp
                reactant = leaf[0]
                product = leaf[-1]
                out_leaf = leaf
                out_leaf_rev = out_leaf.copy()
                out_leaf_rev.nodes.reverse()
                if self._energies_fail(out_leaf):
                    continue

                if self.tolerate_kinks:
                    es = True
                else:
                    n_minima = len(argrelmin(out_leaf.energies)[0])

                    es = n_minima == 0
                # es = out_leaf.is_elem_step()[0]
                # es = len(out_leaf) == 12

                if es:
                    # if reactant not in molecules:
                    #     molecules.append(reactant)
                    # if product not in molecules:
                    #     molecules.append(product)
                    if reactant not in structures:
                        structures.append(reactant)
                    if product not in structures:
                        structures.append(product)

                    # ind_r = self._get_ind_mol(molecules, reactant)
                    # ind_p = self._get_ind_mol(molecules, product)
                    ind_r = self._get_ind_td(ref_list=structures, td=reactant)
                    ind_p = self._get_ind_td(ref_list=structures, td=product)
                    eA = leaf.get_eA_chain()
                    edge_name = f"{ind_r}-{ind_p}"
                    rev_edge_name = f"{ind_p}-{ind_r}"
                    rev_eA = (leaf.energies.max() - leaf[-1].energy) * 627.5

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

        self.leaf_objects = leaf_objects

        # ind_start = self._get_ind_mol(molecules, leaves[0][0].tdstructure.molecule_rp)
        ind_start = self._get_ind_td(structures, leaves[0][0])

        # reactant = molecules[0]
        reactant = structures[0].tdstructure.molecule_rp

        pot = Pot(root=reactant)

        # for i, mol_to_add in enumerate(molecules):
        for i, mol_to_add in enumerate(structures):
            node_ind = i
            if self.verbose:
                print(f"Adding node {node_ind}")
            relevant_keys = self._get_relevant_edges(edges, node_ind)
            relevant_conformers = self._get_relevant_conformers(node_ind)
            if self.verbose:
                print(f"\tIt had {len(relevant_conformers)} conformers")

            rel_edges = self._get_relevant_edges(edges, node_ind)
            pot.graph.add_node(
                node_ind,
                molecule=mol_to_add.tdstructure.molecule_rp,
                converged=False,
                td=mol_to_add.tdstructure,
                node_energy=mol_to_add.energy,
            )
            # node_energies=[conformer.energy for conformer in relevant_conformers],
            # conformers=relevant_conformers)

            # print(rel_edges)
            for edgelabel in rel_edges:
                label = edgelabel
                vals = label.split("-")
                label_rev = f"{vals[1]}-{vals[0]}"
                pot.graph.add_edge(
                    int(vals[1]),
                    node_ind,
                    reaction=f"eA ({edgelabel}): {np.min(edges[edgelabel])}",
                    list_of_nebs=leaf_objects[label],
                    barrier=np.min(edges[edgelabel]),
                    exp_neg_barrier=np.exp(-np.min(edges[edgelabel])),
                )
                pot.graph.add_edge(
                    node_ind,
                    int(vals[1]),
                    reaction=f"eA ({label_rev}):{np.min(edges[label_rev])}",
                    list_of_nebs=leaf_objects[label_rev],
                    barrier=np.min(edges[label_rev]),
                    exp_neg_barrier=np.exp(-np.min(edges[label_rev])),
                )

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


@dataclass
class ReactionData:
    data: dict

    def get_all_TS(self, reaction_key: str):
        return Trajectory([c.get_ts_guess() for c in self.data[reaction_key]])

    def get_all_paths(self, reaction_key: str, barrier_thre=200):
        paths = []
        for c in self.data[reaction_key]:
            if c.get_eA_chain() <= barrier_thre:
                paths.append(c)
        return paths

    def plot_all_paths(self, reaction_key: str, barrier_thre=200):
        s = 6
        f, ax = plt.subplots(figsize=(1.61 * s, s))
        fs = 18
        paths = self.get_all_paths(reaction_key, barrier_thre)
        for c in paths:
            plt.plot(c.path_length, c.energies, "-", alpha=1)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.ylabel("Energies (Hartree)", fontsize=fs)
        plt.xlabel("Path length", fontsize=fs)
        plt.show()
