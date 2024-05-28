from dataclasses import dataclass
from functools import cached_property
from itertools import product
from multiprocessing.dummy import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from retropaths.abinitio.geodesic_interpolation.coord_utils import align_geom
from scipy.signal import argrelmin
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm.notebook import tqdm

from neb_dynamics.helper_functions import quaternionrmsd

# from neb_dynamics.trajectory import Trajectory
from retropaths.abinitio.trajectory import Trajectory


@dataclass
class ConformerAnalyzer:
    fp: Path = None

    @property
    def geo_traj_paths(self):
        traj_files = [x for x in self.fp.iterdir() if "-" in x.stem]
        traj_files = [x for x in self.fp.iterdir() if not "neb" in x.stem]
        trajs = [x for x in traj_files if str(x.resolve())[-4:] == ".xyz"]
        return trajs

    @property
    def neb_traj_paths(self):
        traj_files = [x for x in self.fp.iterdir() if "-" in x.stem]
        traj_files = [x for x in self.fp.iterdir() if "neb" in x.stem]
        trajs = [x for x in traj_files if str(x.resolve())[-4:] == ".xyz"]
        return trajs

    def _get_fp_energy(self, fp):
        traj = Trajectory.from_xyz(fp)
        try:
            energies = np.array(traj.energies)
        except:
            energies = np.nan
        return energies

    @cached_property
    def geo_energies(self):
        with Pool() as p:
            all_energies = list(
                tqdm(
                    p.imap(self._get_fp_energy, self.geo_traj_paths, chunksize=1),
                    total=len(self.geo_traj_paths),
                )
            )
        return all_energies

    @cached_property
    def neb_energies(self):
        with Pool() as p:
            all_energies = list(
                tqdm(
                    p.imap(self._get_fp_energy, self.neb_traj_paths, chunksize=1),
                    total=len(self.geo_traj_paths),
                )
            )
        return all_energies

    @property
    def geo_data(self):
        return self._get_data("geo")

    @property
    def neb_data(self):
        return self._get_data("neb")

    def _get_data(self, method: str):
        data = []

        if method == "geo":
            traj_paths = self.geo_traj_paths
            ens = self.geo_energies
        elif method == "neb":
            traj_paths = self.neb_traj_paths
            ens = self.neb_energies
        else:
            print("Alessio said Cianghini")

        for traj_fp, energies in zip(traj_paths, ens):

            fp = traj_fp
            traj_fp = str(traj_fp.stem)
            init, end = traj_fp.split("_")[1].split("-")

            tag = traj_fp.split("_")[2]

            row = [int(init), int(end), energies, str(fp), str(fp.parent), tag]
            data.append(row)

        return data

    def _init_conf_energy(self, row):
        return row["energies_neb"][0]

    def _end_conf_energy(self, row):
        return row["energies_neb"][-1]

    def _get_remap_dict(self, df, col):
        energies = []
        for conf in df[col].unique():
            sub = df[df[col] == conf]

            avg_conf_energy = sub[f"{col}_energy"].mean()
            energies.append((conf, avg_conf_energy))

        sorted_vals = sorted(energies, key=lambda x: x[1])

        remap = {}  # key: original ranking, value: actual ranking
        for i, (original_rank, _) in enumerate(sorted_vals):
            remap[original_rank] = i

        return remap

    def remap_conformer_rankings(self, df):

        remap_initial = self._get_remap_dict(df=df, col="initial")
        remap_end = self._get_remap_dict(df=df, col="end")

        def do_remap_initial(row):
            return remap_initial[row["initial"]]

        def do_remap_end_vals(row):
            return remap_end[row["end"]]

        df["remap_initial"] = df.apply(do_remap_initial, axis=1)
        df["remap_end"] = df.apply(do_remap_end_vals, axis=1)

        df = df.sort_values(by=["remap_initial", "remap_end"]).reset_index(drop=True)
        return df

    def _make_index_label_helper(self, row):
        label = str(row["remap_initial"]) + "->" + str(row["remap_end"])
        return label

    @property
    def neb_dataframe(self):
        df_neb = pd.DataFrame(
            self.neb_data,
            columns=[
                "initial",
                "end",
                "energies_neb",
                "fp_traj_neb",
                "data_dir",
                "tag",
            ],
        )
        df_neb = df_neb.sort_values(by=["initial", "end"]).reset_index(drop=True)
        return df_neb

    @property
    def geo_dataframe(self):
        df_geo = pd.DataFrame(
            self.geo_data,
            columns=[
                "initial",
                "end",
                "energies_geo",
                "fp_traj_geo",
                "data_dir",
                "tag",
            ],
        )
        df_geo = df_geo.sort_values(by=["initial", "end"]).reset_index(drop=True)
        return df_geo

    def calc_deltaE(self, row):
        maxE = row["energies_neb"].max()
        # deltaE = maxE - row['energies'][0]
        deltaE = maxE - self.global_min
        kcal_deltaE = deltaE * 627.5
        return kcal_deltaE

    def calc_deltaE_geo(self, row):
        maxE = row["energies_geo"].max()
        # deltaE = maxE - row['energies_geodesic'][0]
        deltaE = maxE - self.global_min
        kcal_deltaE = deltaE * 627.5
        return kcal_deltaE

    def get_distances_for_trajectory(self, fp):
        traj = Trajectory.from_xyz(fp)
        dist = []
        for i in range(len(traj)):
            if i == 0:
                continue
            start = traj[i - 1]
            end = traj[i]

            dist.append(quaternionrmsd(start.coords, end.coords))
        return np.array(dist)

    def calc_distances(self, row):
        fp = Path(row["fp_traj_neb"])
        return self.get_distances_for_trajectory(fp)

    def calc_work(self, row):
        work = row["energies_neb"].copy()
        work -= work[0]
        work *= 627.5

        works = np.abs(work[1:] * row["distances"])

        tot_work = works.sum()
        return tot_work

    def get_integrated_path_len_neb(self, row):
        fp = str(row["fp_traj_neb"])
        traj = Trajectory.from_xyz(fp)

        endpoint_vec = traj[-1].coords - traj[0].coords

        int_path_len = [0]
        for i, frame in enumerate(traj):
            if i == len(traj) - 1:
                continue
            next_frame = traj[i + 1]
            dist_vec = next_frame.coords - frame.coords
            proj = (
                np.tensordot(dist_vec, endpoint_vec)
                / np.tensordot(endpoint_vec, endpoint_vec)
            ) * endpoint_vec

            proj_dist = np.linalg.norm(proj)
            int_path_len.append(int_path_len[-1] + proj_dist)
        return int_path_len

    def get_integrated_path_len_geo(self, row):
        fp = Path(row["fp_traj_geo"])
        traj = Trajectory.from_xyz(fp)

        endpoint_vec = traj[-1].coords - traj[0].coords

        int_path_len = [0]
        for i, frame in enumerate(traj):
            if i == len(traj) - 1:
                continue
            next_frame = traj[i + 1]
            dist_vec = next_frame.coords - frame.coords
            proj = (
                np.tensordot(dist_vec, endpoint_vec)
                / np.tensordot(endpoint_vec, endpoint_vec)
            ) * endpoint_vec

            proj_dist = np.linalg.norm(proj)
            int_path_len.append(int_path_len[-1] + proj_dist)
        return int_path_len

    @property
    def dataframe(self):

        df_neb = self.neb_dataframe
        df_geo = self.geo_dataframe

        df = df_neb.merge(df_geo, how="outer")
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def reduce_to_best_instances(self, df):
        new_df = pd.DataFrame()
        for r, p in list(product(df["initial"].unique(), df["end"].unique())):
            sub = df[(df["initial"] == r) & (df["end"] == p)].reset_index()

            if len(sub) > 1:
                correct_ind = np.argmin(sub["deltaE"].values)
                new_df = pd.concat([new_df, sub.iloc[correct_ind, :]], axis=1)
            elif len(sub) < 1:
                print(f"conformer-{r} --> conformer-{p} has no interpolation")
            else:
                new_df = pd.concat([new_df, sub.T], axis=1)

        df_final = new_df.T.drop("level_0", axis=1).reset_index(drop=True)
        df_final = df_final.sort_values(by="deltaE").reset_index(drop=True)
        return df_final

    @cached_property
    def dataframe_refined(self):
        df = self.dataframe
        df["initial_energy"] = df.apply(self._init_conf_energy, axis=1)
        df["end_energy"] = df.apply(self._end_conf_energy, axis=1)
        df = self.remap_conformer_rankings(df)
        df["index"] = df.apply(self._make_index_label_helper, axis=1)

        self.global_min = df["initial_energy"].values.min()

        df["deltaE"] = df.apply(self.calc_deltaE, axis=1)
        df["deltaE_geo"] = df.apply(self.calc_deltaE_geo, axis=1)
        df["int_path_len_neb"] = df.apply(self.get_integrated_path_len_neb, axis=1)
        df["int_path_len_geo"] = df.apply(self.get_integrated_path_len_geo, axis=1)
        df = self.reduce_to_best_instances(df)
        df["tag"] = df["tag"].apply(int)

        return df.copy()

    def plot_single_graph(self, start_ind, end_ind, ind=0):
        fs = 20
        s = 5

        df = self.dataframe_refined
        f, ax = plt.subplots(figsize=(1.618 * s, s))

        row = df[
            (df["remap_initial"] == start_ind)
            & (df["remap_end"] == end_ind)
            & (df["tag"] == ind)
        ]
        ens = row["energies_neb"].values[0]
        ens = ens.copy()

        ens_geo = row["energies_geo"].values[0]
        ens_geo = ens_geo.copy()
        ens_geo -= ens[0]
        ens_geo *= 627.5

        # ens_cneb = row['energies_cneb'].values[0]
        # ens_cneb = ens_cneb.copy()
        # ens_cneb -= ens[0]
        # ens_cneb*= 627.5

        ens -= ens[0]
        ens *= 627.5

        plt.plot(row["int_path_len_neb"].values[0], ens, "o-", label="neb")
        plt.plot(
            row["int_path_len_geo"].values[0],
            ens_geo,
            "x--",
            color="gray",
            label="geodesic",
        )
        # plt.plot(ens_cneb, "^-", color='orange',label='cneb')

        plt.ylabel("Energy (kcal/mol)", fontsize=fs)
        plt.xlabel("Integrated path length", fontsize=fs)

        plt.text(
            0.05,
            0.90,
            s=f"Conformer {start_ind} --> {end_ind}",
            fontsize=0.9 * fs,
            transform=ax.transAxes,
        )

        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.legend(fontsize=fs)
        # plt.savefig(f"{start_ind}_{end_ind}.svg")
        plt.show()
        print(f"vmd {row['fp_traj_neb'].values[0]}")

    def plot_heatmap(self, n=120):
        df = self.dataframe_refined
        s = 10
        f, ax = plt.subplots(figsize=(s, n * s / 45))
        plt.imshow(
            np.vstack((df.iloc[:n]["energies_neb"].values - self.global_min) * 627.5)
        )
        plt.yticks(list(range(len(df.iloc[:n]["index"].values))))
        ax.set_yticklabels(df.iloc[:n]["index"].values)
        plt.colorbar()
        plt.show()

    def get_all_reactive_conformers(self):
        all_reactive_conformers = []
        en_thre = self.dataframe_refined["deltaE"].median()
        for i, t_fp in enumerate(self.dataframe_refined["fp_traj_neb"].values):
            print(f"{i}", end=" ")
            if self.dataframe_refined.iloc[i]["deltaE"] <= en_thre:
                # print(f"---- {i} under {en_thre} kcal/mol")
                val = self.get_reactive_conformers(t_fp)
                all_reactive_conformers.append(val)

        return all_reactive_conformers

    def get_reactive_conformers(self, traj_fp: str, buffer=2):
        reactive_mols = []

        traj = Trajectory.from_xyz(traj_fp)
        prev_td = traj[0]
        for i in range(1, len(traj.traj) - 1):
            current_td = traj[i]
            if current_td.molecule_rp == prev_td.molecule_rp:
                prev_td = current_td
                continue
            else:
                # print(f"a change has happened at frame {i}")
                reactive_mols.append(traj[i - buffer])
                reactive_mols.append(traj[i])
                prev_td = current_td

        reactant, product, intermediates = self.post_process_reactive_mols(
            traj, reactive_mols
        )
        if len(intermediates) > 0:
            # print("intermediates found!")
            return (reactant, product), intermediates
        else:
            return (reactant, product)

    def post_process_reactive_mols(self, traj, mol_list):
        reactant = None
        product = None
        intermediates = []
        for mol in mol_list:
            mol_relaxed = mol.xtb_geom_optimization()
            if mol_relaxed.molecule_rp == traj[0].molecule_rp:
                reactant = mol_relaxed
            elif mol_relaxed.molecule_rp == traj[-1].molecule_rp:
                product = mol_relaxed
            else:
                intermediates.append(mol_relaxed)
        return reactant, product, intermediates

    @cached_property
    def all_reactive_conformers(self):
        return self.get_all_reactive_conformers()

    def get_reactive_pairs(self):
        pairs = []
        for val in self.all_reactive_conformers:
            try:
                len(val[0])
                pairs.append(val[0])
            except:
                pairs.append(val)
        return pairs

    def score_with_n_clusters(self, n, X):
        km = KMeans(n)
        km.fit(X)
        return -km.score(X)

    def make_x_mat(self, vals, ind):
        all_coords = []
        for pair in vals:
            all_coords.append(pair[ind].coords.flatten())
        return np.array(all_coords)

    def make_out_val(self, max_val, vals, ind):
        X = self.make_x_mat(vals, ind)
        out = [self.score_with_n_clusters(i, X) for i in range(1, max_val)]
        return out

    def get_optim_cluster_number(self, out_arr):
        kn = KneeLocator(
            list(range(len(out_arr))), out_arr, curve="convex", direction="decreasing"
        )
        return kn.knee

    @property
    def reactive_reactants(self):
        return self._get_reactive_conformers_helper(0)

    @property
    def reactive_products(self):
        return self._get_reactive_conformers_helper(1)

    def _get_reactive_conformers_helper(self, ind=0):
        """
        if ind==0: reactant conformer
        ind==1: product conformer
        """
        pairs = self.get_reactive_pairs()
        aligned_rel_0 = [pairs[0][ind].coords]

        for i in range(1, len(pairs) - 1):
            ref_geom = aligned_rel_0[0]
            other_geom = pairs[i][ind].coords
            rmsd, new_coords = align_geom(ref_geom, other_geom)
            aligned_rel_0.append(new_coords)

        out = self.make_out_val(100, pairs, ind)
        optim = self.get_optim_cluster_number(out)
        km = KMeans(optim)
        X = self.make_x_mat(pairs, ind)
        km.fit(X)

        t = Trajectory.from_xyz(self.dataframe_refined.iloc[0]["fp_traj_neb"])
        td = t[0]
        clustered_structures = []
        closest_inds, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
        for ind in closest_inds:
            coords = X[ind]
            td_copy = td.copy()
            td_copy.update_coords(coords.reshape(td.coords.shape))
            clustered_structures.append(td_copy)

        return clustered_structures
