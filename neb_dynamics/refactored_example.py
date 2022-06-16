
from pathlib import Path
from platform import node

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from neb_dynamics.NEB import NEB, Chain, Node2D, Node3D, NoneConvergedException
from neb_dynamics.trajectory import Trajectory


def sorry_func(inp):
    
    x, y = inp
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def animate_func(neb_obj: NEB):
    n_nodes = len(neb_obj.initial_chain.nodes)
    en_func = neb_obj.initial_chain[0].en_func
    chain_traj = neb_obj.chain_trajectory
    plt.style.use("seaborn-pastel")

    figsize = 5

    f, ax = plt.subplots(figsize=(1.18 * figsize, figsize))
    x = np.linspace(start=-4, stop=4, num=n_nodes)
    y = x.reshape(-1, 1)

    # h = en_func(x, y)
    h = sorry_func([x, y])
    cs = plt.contourf(x, x, h)
    _ = f.colorbar(cs, ax=ax)
    (line,) = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(chain):

        x = chain[:, 0]
        y = chain[:, 1]
        line.set_data(x, y)
        return (line,)

    _ = FuncAnimation(fig=f, func=animate, frames=np.array([chain.coordinates for chain in chain_traj]), blit=True, repeat_delay=1000, interval=200)
    plt.show()


def plot_func(neb_obj: NEB):

    en_func = neb_obj.initial_chain[0].en_func
    orig_chain = neb_obj.initial_chain
    new_chain = neb_obj.chain_trajectory[-1]

    min_val = -4
    max_val = 4
    num = 10
    fig = 10
    f, _ = plt.subplots(figsize=(1.18 * fig, fig))
    x = np.linspace(start=min_val, stop=max_val, num=num)
    y = x.reshape(-1, 1)

    h = sorry_func([x, y])
    cs = plt.contourf(x, x, h)
    _ = f.colorbar(cs)
    plt.plot(
        [(node.coords[0]) for node in orig_chain],
        [(node.coords[1]) for node in orig_chain],
        "^--",
        c="white",
        label="original",
    )

    points_x = [node.coords[0] for node in new_chain]
    points_y = [node.coords[1] for node in new_chain]
    # plt.plot([toy_potential_2(point) for point in new_chain])
    plt.plot(points_x, points_y, "o--", c="white", label="NEB")
    # psave(new_chain, "new_chain.p")
    plt.show()


def plot_2D(neb_obj: NEB):
    opt_chain = neb_obj.chain_trajectory[-1]
    ens = np.array([node.energy for node in opt_chain])
    # ens = ens*627.5


    orig_ens = np.array([node.energy for node in neb_obj.initial_chain])
    # orig_ens = orig_ens*627.5
    orig_ens = orig_ens - ens[0]

    ens = ens-ens[0]


    plt.plot(list(range(len(ens))), ens, "o--", label="last chain")
    plt.plot(orig_ens, "*", label='original')
    plt.legend()
    plt.show()

def main():
    # nimages = 30
    # end_point = (3.00002182, 1.99995542)
    # start_point = (-3.77928812, -3.28320392)


    # coords = np.linspace(start_point, end_point, nimages)
    # chain = Chain.from_list_of_coords(k=0, list_of_coords=coords, node_class=Node2D)
    # n = NEB(initial_chain=chain, max_steps=1000, grad_thre=0.01, mag_grad_thre=1)
    # try: 
    #     n.optimize_chain()
    # except NoneConvergedException as e:
    #     print(e.obj.chain_trajectory[-1].gradients)
    #     animate_func(e.obj)
    #     plot_func(e.obj)
        

    # plot_2D(n)
    # plot_func(n)
    # animate_func(n)

    fp = Path("./example_cases/DA_geodesic_opt.xyz")
    # fp = Path("./example_cases/PDDA_geodesic.xyz")

    traj = Trajectory.from_xyz(fp)
    coords = traj.to_list()
    chain = Chain.from_list_of_coords(k=10, list_of_coords=coords, node_class=Node3D)
    n = NEB(initial_chain=chain, grad_thre=0.0001, en_thre=0.0001, mag_grad_thre=.1, max_steps=100)

    try: 
        n.optimize_chain()
        plot_2D(n)
    except NoneConvergedException as e:
        plot_2D(e.obj)
        print(e.obj.chain_trajectory[-1].gradients)




    


if __name__ == "__main__":
    main()