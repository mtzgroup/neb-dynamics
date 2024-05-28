from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neb_dynamics.Chain import Chain
from neb_dynamics.helper_functions import qRMSD_distance
from neb_dynamics.Inputs import ChainInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.tdstructure import TDStructure


@dataclass
class MSMEPAnalyzer:
    parent_dir: Path
    msmep_root_name: str

    def get_relevant_chain(self, folder_name):
        data_dir = self.parent_dir / folder_name
        clean_chain = data_dir / f"{self.msmep_root_name}_msmep_clean.xyz"
        msmep_chain = data_dir / f"{self.msmep_root_name}_msmep.xyz"

        if clean_chain.exists():
            chain_to_use = Chain.from_xyz(clean_chain, ChainInputs())
        elif not clean_chain.exists() and msmep_chain.exists():
            chain_to_use = Chain.from_xyz(msmep_chain, ChainInputs())
        else:  # somehow the thing failed
            print(f"{folder_name} unavailable.")
            chain_to_use = None

        return chain_to_use

    def get_relevant_saddle_point(self, folder_name):
        data_dir = self.parent_dir / folder_name
        sp_fp = data_dir / "sp.xyz"
        sp = TDStructure.from_xyz(str(sp_fp))
        return sp

    def _distance_to_sp(self, chain: Chain, sp):
        ts_guess = chain.get_ts_guess()
        ts_guess = ts_guess.align_to_td(sp)

        return qRMSD_distance(ts_guess.coords, sp.coords)

    def get_relevant_leaves(self, folder_name):
        data_dir = self.parent_dir / folder_name
        fp = data_dir / f"{self.msmep_root_name}_msmep"
        adj_mat_fp = fp / "adj_matrix.txt"
        adj_mat = np.loadtxt(adj_mat_fp)
        if adj_mat.size == 1:
            return [
                Chain.from_xyz(fp / f"node_0.xyz", ChainInputs(k=0.1, delta_k=0.09))
            ]
        else:

            a = np.sum(adj_mat, axis=1)
            inds_leaves = np.where(a == 1)[0]
            chains = [
                Chain.from_xyz(fp / f"node_{ind}.xyz", ChainInputs(k=0.1, delta_k=0.09))
                for ind in inds_leaves
            ]
            return chains

    def get_relevant_leaves_nebs(self, folder_name):
        data_dir = self.parent_dir / folder_name
        fp = data_dir / f"{self.msmep_root_name}_msmep"
        adj_mat_fp = fp / "adj_matrix.txt"
        adj_mat = np.loadtxt(adj_mat_fp)
        if adj_mat.size == 1:
            return [NEB.read_from_disk(fp / f"node_0.xyz")]
        else:

            a = np.sum(adj_mat, axis=1)
            inds_leaves = np.where(a == 1)[0]
            nebs = [NEB.read_from_disk(fp / f"node_{ind}.xyz") for ind in inds_leaves]
            return nebs

    def get_relevant_tree(self, folder_name):
        data_dir = self.parent_dir / folder_name
        fp = data_dir / f"{self.msmep_root_name}_msmep"
        h = TreeNode.read_from_disk(fp)
        return h
