import contextlib
import itertools
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.signal import argrelmin

from neb_dynamics.chain import Chain
from neb_dynamics.helper_functions import RMSD, pairwise
from neb_dynamics.inputs import ChainInputs, NetworkInputs
from neb_dynamics.pot import Pot
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.nodes.nodehelpers import molecule_to_structure, is_identical
from neb_dynamics.molecule import Molecule
from neb_dynamics.engines import Engine, QCOPEngine

from qcio import Structure, ProgramArgs

from typing import Union

from rxnmapper import RXNMapper


@dataclass
class NetworkBuilder:

    data_dir: Path

    start: Union[StructureNode, str, None] = None
    end: Union[StructureNode, str, None] = None

    charge: int = 0
    spinmult: int = 1

    engine: Engine = QCOPEngine(program='crest',
                                program_args=ProgramArgs(
                                    model={'method': 'gfn2'}, keywords={"calculation": {"level": [{"alpb": "h2o"}]}})
                                )
    network_inputs: NetworkInputs = NetworkInputs()
    chain_inputs: ChainInputs = ChainInputs()

    # def __post_init__(self):
    #     both_are_smi = isinstance(
    #         self.start, str) and isinstance(self.end, str)
    #     both_are_tds = isinstance(
    #         self.start, StructureNode) and isinstance(self.end, StructureNode)
    #     both_are_none = self.start is None and self.end is None

    #     assert both_are_smi or both_are_tds or both_are_none, 'Both inputs needs to be the same data type.'

    #     if both_are_smi:
    #         start_smi = self.start
    #         end_smi = self.end
    #         mapped_start, mapped_end = self.create_pairs_from_smiles(
    #             smi1=start_smi, smi2=end_smi)
    #         self.start = mapped_start
    #         self.end = mapped_end

    #         new_start_changed = mapped_start.structure.to_smiles() != start_smi
    #         new_end_changed = mapped_end.structure.to_smiles() != end_smi
    #         if new_start_changed or new_end_changed:
    #             print("Warning! Mapping endpoints indices changed smiles string.")
    #             print(
    #                 f"Input start: {start_smi}. Output start: {mapped_start.structure.to_smiles()}")
    #             print(
    #                 f"Input end: {end_smi}. Output end: {mapped_end.structure.to_smiles()}")

    #     self.data_dir.mkdir(exist_ok=True)

    #     self.start_confs_dd = self.data_dir / "start_confs"
    #     self.start_confs_dd.mkdir(exist_ok=True)

    #     self.end_confs_dd = self.data_dir / "end_confs"
    #     self.end_confs_dd.mkdir(exist_ok=True)

    #     self.pairs_data_dir = self.data_dir / "pairs_to_do"
    #     self.pairs_data_dir.mkdir(exist_ok=True)

    #     self.msmep_data_dir = self.data_dir / "msmep_results"
    #     self.msmep_data_dir.mkdir(exist_ok=True)

    #     self.submissions_dir = self.data_dir / "submissions"
    #     self.submissions_dir.mkdir(exist_ok=True)

    #     self.leaf_objects = None

    #     if self.start is None:
    #         start_fp = self.start_confs_dd / "start.xyz"
    #         assert (
    #             start_fp.exists()
    #         ), f"Need to input a start geometry since {start_fp} does not exist"
    #         self.start = StructureNode(structure=Structure.open(
    #             start_fp, charge=self.charge, multiplicity=self.spinmult))

    #     if self.end is None:
    #         end_fp = self.end_confs_dd / "end.xyz"
    #         assert (
    #             start_fp.exists()
    #         ), f"Need to input a start geometry since {end_fp} does not exist"
    #         self.end = StructureNode(structure=Structure.open(
    #             end_fp, charge=self.charge, multiplicity=self.spinmult))

    @ contextlib.contextmanager
    def remember_cwd(self):
        curdir = os.getcwd()
        try:
            yield
        finally:
            os.chdir(curdir)

    @ staticmethod
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

        start_conformers = self.start.sample_all_conformers(
            dd=self.start_confs_dd,
            fn="start.xyz",
            verbose=self.network_inputs.verbose,
            CREST_ewin=self.network_inputs.CREST_ewin,
            CREST_temp=self.network_inputs.CREST_temp
        )
        end_conformers = self.end.sample_all_conformers(
            dd=self.end_confs_dd,
            fn="end.xyz",
            verbose=self.network_inputs.verbose,
            CREST_ewin=self.network_inputs.CREST_ewin,
            CREST_temp=self.network_inputs.CREST_temp
        )

        if self.network_inputs.subsample_confs:
            # subselect to n conformers for each
            sub_start_confs = self.subselect_confomers(
                conformer_pool=start_conformers,
                n_max=self.network_inputs.n_max_conformers,
                rmsd_cutoff=self.network_inputs.conf_rmsd_cutoff,
            )
            sub_end_confs = self.subselect_confomers(
                conformer_pool=end_conformers,
                n_max=self.network_inputs.n_max_conformers,
                rmsd_cutoff=self.network_inputs.conf_rmsd_cutoff,
            )
            start_conformers = sub_start_confs
            end_conformers = sub_end_confs
        return start_conformers, end_conformers

    def create_pairs_from_smiles(self, smi1: str, smi2: str, charge=0, spinmult=1):
        rxnsmi = f"{smi1}>>{smi2}"
        rxn_mapper = RXNMapper()
        rxn = [rxnsmi]
        result = rxn_mapper.get_attention_guided_atom_maps(rxn)[0]
        mapped_smi = result['mapped_rxn']
        r_smi, p_smi = mapped_smi.split(">>")
        r = Molecule.from_mapped_smiles(r_smi)
        p = Molecule.from_mapped_smiles(p_smi)

        td_r, td_p = (
            StructureNode(structure=molecule_to_structure(
                r, charge=charge, spinmult=spinmult)),
            StructureNode(structure=molecule_to_structure(
                p, charge=charge, spinmult=spinmult))
        )
        return td_r, td_p

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

            if self.network_inputs.verbose:
                print(f"\t***Creating pair {i} submission")
            start_fp = self.pairs_data_dir / f"start_{file_stem_name}_{i}.xyz"
            end_fp = self.pairs_data_dir / f"end_{file_stem_name}_{i}.xyz"
            start.to_xyz(start_fp)
            end.to_xyz(end_fp)
            out_fp = self.msmep_data_dir / f"results_{file_stem_name}{i}_msmep"

            cmd = f"create_msmep_from_endpoints.py -st {start_fp} -en {end_fp} -tol 0.002 \
                -sig {int(self.chain_inputs.skip_identical_graphs)} -mr {int(self.chain_inputs.use_maxima_recyling)} \
                    -nc {self.chain_inputs.node_class.__repr__(self.chain_inputs.node_class)} -preopt 0 -climb 0 \
                        -nimg 12 -min_ends 1 -es_ft 0.03 -name {out_fp}"

            new_template = template.copy()
            new_template[-1] = cmd

            with open(
                self.submissions_dir / f"submission_{sub_id_name}{i}.sh", "w+"
            ) as f:
                f.write("\n".join(new_template))

    def run_msmeps(self):
        # submit all jobs
        all_jobs = list(self.submissions_dir.glob("*.sh"))

        for job in all_jobs:
            if self.network_inputs.verbose:
                print("\t", job)

            command = open(job).read().splitlines()[-1]
            out_fp = Path(command.split()[-1])
            if not out_fp.exists():
                if self.network_inputs.use_slurm:
                    with self.remember_cwd():
                        os.chdir(self.submissions_dir)
                        if self.network_inputs.verbose:
                            print(f"\t\tsubmitting {job}")
                        _ = subprocess.run(
                            ["sbatch", f"{job}"], capture_output=True)

                else:
                    if self.network_inputs.verbose:
                        print(f"\t\trunning {job}")
                    out = subprocess.run(command.split(), capture_output=True)
                    if self.network_inputs.verbose:
                        print(
                            f"\t\t\twriting stdout in {out_fp.parent.resolve()}")
                    with open(out_fp.parent / f"out_{out_fp.name}", "w+") as fout:
                        fout.write(out.stdout.decode("utf-8"))
                        fout.write(out.stderr.decode("utf-8"))
                    with open(out_fp.parent / f"stderr_{out_fp.name}", "w+") as fout:
                        fout.write(out.stderr.decode("utf-8"))
            else:
                if self.network_inputs.verbose:
                    print("\t\t\talready done")

    def _get_ind_mol(self, ref_list, mol):
        inds = np.where(
            [a.is_isomorphic_to(b) for a, b in list(
                itertools.product([mol], ref_list))]
        )[0]
        assert len(inds) >= 1, "No matches found. Network cannot be constructed."
        return inds[0]

    def _equality_function(self, a: StructureNode, b: StructureNode):

        return is_identical(a, b, fragment_rmsd_cutoff=self.chain_inputs.node_rms_thre, verbose=False,
                            kcal_mol_cutoff=self.chain_inputs.node_ene_thre)

    def _get_ind_td(self, ref_list, td):
        inds = np.where(
            [self._equality_function(a, b) for a, b in list(
                itertools.product([td], ref_list))]
        )[0]
        assert len(inds) >= 1, "No matches found. Network cannot be constructed."
        return inds[0]

    def _get_relevant_leaves(self, fp):
        adj_mat_fp = fp / "adj_matrix.txt"
        adj_mat = np.loadtxt(adj_mat_fp)
        if adj_mat.size == 1:
            return [
                Chain.from_xyz(fp / "node_0.xyz", self.chain_inputs)
            ]
        else:

            a = np.sum(adj_mat, axis=1)
            inds_leaves = np.where(a == 1)[0]
            chains = [
                Chain.from_xyz(
                    fp / f"node_{ind}.xyz",
                    self.chain_inputs,
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
        except Exception:
            return True

    def _load_network_data(self, msmep_paths: list[Path]):
        structures = []  # list of Node3D
        edges = {}
        leaf_objects = {}
        for fp in tqdm.tqdm(msmep_paths):
            if self.network_inputs.verbose:
                print(f"\tDoing: {fp}. Len: {len(structures)}")

            leaves = self._get_relevant_leaves(fp)
            for leaf in leaves:
                reactant = leaf[0]
                product = leaf[-1]
                out_leaf = leaf
                out_leaf_rev = out_leaf.copy()
                out_leaf_rev.nodes.reverse()
                if self._energies_fail(out_leaf):
                    if self.network_inputs.verbose:
                        print(
                            f"\t\t{fp} had a leaf with failed energies. Might result in disconnected nodes.")
                    continue

                if self.network_inputs.tolerate_kinks:
                    elementary_step = True
                else:
                    n_minima = len(argrelmin(out_leaf.energies)[0])

                    elementary_step = n_minima == 0
                if elementary_step:
                    print("len(structures)", len(structures))
                    reactant_comparison = all([not self._equality_function(
                        reactant, reference) for reference in structures])
                    product_comparison = all([not self._equality_function(
                        product, reference) for reference in structures])

                    print("reactant_comparison", reactant_comparison,
                          "product_comparison", product_comparison)

                    if reactant_comparison or len(structures) == 0:
                        structures.append(reactant)
                    if product_comparison or len(structures) == 1:
                        structures.append(product)

                    ind_r = self._get_ind_td(ref_list=structures, td=reactant)
                    ind_p = self._get_ind_td(ref_list=structures, td=product)
                    eA = leaf.get_eA_chain()
                    edge_name = f"{ind_r}-{ind_p}"
                    print(f"EDGE: {edge_name}")
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
        return structures, edges

    def _add_all_nodes(self, pot: Pot, structures: list):
        for i, mol_to_add in enumerate(structures):
            node_ind = i
            if self.network_inputs.verbose:
                print(f"Adding node {node_ind}")

            relevant_conformers = self._get_relevant_conformers(node_ind)
            if self.network_inputs.verbose:
                print(f"\tIt had {len(relevant_conformers)} conformers")

            pot.graph.add_node(
                node_ind,
                molecule=mol_to_add.graph,
                converged=False,
                td=mol_to_add,
                node_energy=mol_to_add.energy,
                node_energies=[
                    conformer.energy for conformer in relevant_conformers],
                conformers=relevant_conformers)

        return pot

    def _get_lowest_barrier_height_pair(self, edges: dict, edgelabel: str):
        all_barriers_fwd = edges[edgelabel]
        label = edgelabel
        vals = label.split("-")
        label_rev = f"{vals[1]}-{vals[0]}"
        all_barriers_rev = edges[label_rev]
        assert len(all_barriers_fwd) == len(all_barriers_rev), f"Inconsistency in number of barriers.\
              {len(all_barriers_fwd)=} {len(all_barriers_rev)=}"

        sum_barrier_fwd_and_back = [
            fwd+rev for fwd, rev in zip(all_barriers_fwd, all_barriers_rev)]
        lowest_barrier_ind = np.argmin(sum_barrier_fwd_and_back)
        return all_barriers_fwd[lowest_barrier_ind], all_barriers_rev[lowest_barrier_ind]

    def _add_all_edges(self, pot: Pot, structures: list, edges: dict):
        for i, _ in enumerate(structures):
            node_ind = i
            rel_edges = self._get_relevant_edges(edges, node_ind)
            for edgelabel in rel_edges:
                label = edgelabel
                vals = label.split("-")
                label_rev = f"{vals[1]}-{vals[0]}"
                lowest_barrier_height_fwd, \
                    lowest_barrier_height_rev = self._get_lowest_barrier_height_pair(
                        edges, edgelabel)

                if int(vals[1]) == node_ind:  # skipping to avoid self loops
                    continue

                if lowest_barrier_height_fwd <= self.network_inputs.maximum_barrier_height:

                    pot.graph.add_edge(
                        int(vals[1]),
                        node_ind,
                        reaction=f"eA ({edgelabel}): {np.min(edges[edgelabel])}",
                        list_of_nebs=self.leaf_objects[label],
                        # barrier=np.min(edges[edgelabel]),
                        barrier=lowest_barrier_height_fwd,
                        exp_neg_barrier=np.exp(-np.min(edges[edgelabel])),
                    )
                if lowest_barrier_height_rev <= self.network_inputs.maximum_barrier_height:
                    if int(vals[1]) == node_ind:
                        continue
                    pot.graph.add_edge(
                        node_ind,
                        int(vals[1]),
                        reaction=f"eA ({label_rev}):{np.min(edges[label_rev])}",
                        list_of_nebs=self.leaf_objects[label_rev],
                        # barrier=np.min(edges[label_rev]),
                        barrier=lowest_barrier_height_rev,
                        exp_neg_barrier=np.exp(-np.min(edges[label_rev])),
                    )
        return pot

    def create_rxn_network(self, file_pattern="results*_msmep"):
        msmep_paths = list(self.msmep_data_dir.glob(file_pattern))
        msmep_paths = [p for p in msmep_paths if p.is_dir()]
        structures, edges = self._load_network_data(msmep_paths=msmep_paths)
        pot = Pot.model_validate({'root': structures[0].graph})
        pot = self._add_all_nodes(pot, structures=structures)
        pot = self._add_all_edges(pot, structures=structures, edges=edges)
        return pot

    def path_to_keys(self, path_indices):
        pairs = list(pairwise(path_indices))
        labels = [f"{a}-{b}" for a, b in pairs]
        return labels

    def get_best_chain(self, list_of_chains):
        eAs = [c.get_eA_chain() for c in list_of_chains]
        return list_of_chains[np.argmin(eAs)]

    def calculate_barrier(self, chain):
        return (chain.energies.max() - chain[0].energy)*627.5

    def path_to_chain(self, path, leaf_objects):
        labels = self.path_to_keys(path)
        node_list = []
        for label in labels:
            node_list.extend(self.get_best_chain(leaf_objects[label]).nodes)
        c = Chain(node_list, ChainInputs())
        return c

    def path_to_list_of_chains(self, path, leaf_objects):
        labels = self.path_to_keys(path)
        c_list = []
        for label in labels:

            c = self.get_best_chain(leaf_objects[label])
            c_list.append(c)
        return c_list

    def get_lowest_barrier_chain(self, edge: str):
        edge_data = self.leaf_objects[edge]
        assert len(edge_data) >= 1, f"{edge} was not found in network."
        eAs = [c.get_eA_chain() for c in edge_data]
        best_ind = np.argmin(eAs)
        return edge_data[best_ind]

    def get_mechanism_mols(self, chain, elem_step_len=12):
        out_mols = [chain[0].graph]
        nsteps = int(len(chain)/elem_step_len)
        for ind in range(nsteps):
            r = chain[ind*elem_step_len].graph
            if r != out_mols[-1]:
                out_mols.append(r)

        p = chain[-1].graph
        if p != out_mols[-1]:
            out_mols.append(p)
        return out_mols

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
