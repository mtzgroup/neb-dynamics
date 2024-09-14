from dataclasses import dataclass
from abc import ABC, abstractmethod
from neb_dynamics.elementarystep import ElemStepResults
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


@dataclass
class PathMinimizer(ABC):
    @abstractmethod
    def optimize_chain(self) -> ElemStepResults: ...

    def plot_opt_history(self, do_3d=False):

        s = 8
        fs = 18

        if do_3d:
            all_chains = self.chain_trajectory

            ens = np.array([c.energies - c.energies[0] for c in all_chains])
            all_integrated_path_lengths = np.array(
                [c.integrated_path_length for c in all_chains]
            )
            opt_step = np.array(list(range(len(all_chains))))
            s = 7
            fs = 18
            ax = plt.figure(figsize=(1.16 * s, s)).add_subplot(projection="3d")

            # Plot a sin curve using the x and y axes.
            x = opt_step
            ys = all_integrated_path_lengths
            zs = ens
            for i, (xind, y) in enumerate(zip(x, ys)):
                if i < len(ys) - 1:
                    ax.plot(
                        [xind] * len(y),
                        y,
                        "o-",
                        zs=zs[i],
                        color="gray",
                        markersize=3,
                        alpha=0.1,
                    )
                else:
                    ax.plot(
                        [xind] * len(y), y, "o-", zs=zs[i], color="blue", markersize=3
                    )
            ax.grid(False)

            ax.set_xlabel("optimization step", fontsize=fs)
            ax.set_ylabel("integrated path length", fontsize=fs)
            ax.set_zlabel("energy (hartrees)", fontsize=fs)

            # Customize the view angle so it's easier to see that the scatter points lie
            # on the plane y=0
            ax.view_init(elev=20.0, azim=-45)
            plt.tight_layout()
            plt.show()

        else:
            f, ax = plt.subplots(figsize=(1.16 * s, s))

            for i, chain in enumerate(self.chain_trajectory):
                if i == len(self.chain_trajectory) - 1:
                    plt.plot(
                        chain.integrated_path_length, chain.energies, "o-", alpha=1
                    )
                else:
                    plt.plot(
                        chain.integrated_path_length,
                        chain.energies,
                        "o-",
                        alpha=0.1,
                        color="gray",
                    )

            plt.xlabel("Integrated path length", fontsize=fs)

            plt.ylabel("Energy (kcal/mol)", fontsize=fs)
            plt.xticks(fontsize=fs)
            plt.yticks(fontsize=fs)
            plt.show()

    def write_to_disk(self, fp: Path, write_history=True):
        # write output chain
        self.chain_trajectory[-1].write_to_disk(fp)

        if write_history:
            out_folder = fp.resolve().parent / (fp.stem + "_history")

            if out_folder.exists():
                shutil.rmtree(out_folder)

            if not out_folder.exists():
                out_folder.mkdir()

            for i, chain in enumerate(self.chain_trajectory):
                fp = out_folder / f"traj_{i}.xyz"
                chain.write_to_disk(fp)
