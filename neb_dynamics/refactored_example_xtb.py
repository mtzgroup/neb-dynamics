from pathlib import Path

import matplotlib.pyplot as plt
from trajectory import Trajectory

from NEB_xtb import neb


def main():
    data_dir = Path("./example_cases/PDDA_geodesic.xyz")
    traj = Trajectory.from_xyz(data_dir)
    n = neb()

    original_chain_ens = [n.en_func(s) for s in traj]

    opt_chain, _ = n.optimize_chain(chain=traj, grad_func=n.grad_func, en_func=n.en_func, k=10, max_steps=1000)

    opt_chain_ens = [n.en_func(s) for s in opt_chain]
    plt.plot(list(range(len(original_chain_ens))), original_chain_ens, "x--", label="orig")
    plt.plot(list(range(len(original_chain_ens))), opt_chain_ens, "o", label="neb")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
