import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Chain import Chain
from neb_dynamics.Node2d import Node2D, Node2D_2, Node2D_ITM, Node2D_LEPS, Node2D_Flower
from neb_dynamics.Node import AlessioError
from neb_dynamics.TS_PRFO import TS_PRFO
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.potential_functions import (
    sorry_func_0,
    sorry_func_1,
    sorry_func_2,
    sorry_func_3,
    flower_func
)
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs


np.random.seed(0)
sfs = [sorry_func_0, sorry_func_1, sorry_func_2, sorry_func_3, flower_func]
index = 4
nodes = [Node2D, Node2D_2, Node2D_ITM, Node2D_LEPS, Node2D_Flower ]
min_sizes = [-4, -2, -2, 0.5, -5.2]
max_sizes = [4, 2, 2, 4, 5.2]

presets = {
    "pot_func": sfs[index],
    "node": nodes[index],
    "min_size": min_sizes[index],
    "max_size": max_sizes[index],
}


def sorry_func(inp):
    return sfs[index](inp)

# s=5

def animate_func(neb_obj: NEB):
    n_nodes = len(neb_obj.initial_chain.nodes)
    en_func = neb_obj.initial_chain[0].en_func
    chain_traj = neb_obj.chain_trajectory
    plt.style.use("seaborn-pastel")

    figsize = 5

    f, ax = plt.subplots(figsize=(1.618 * figsize, figsize))

    # min_val = -s
    # max_val = s

    min_val = presets["min_size"]
    max_val = presets["max_size"]

    x = np.linspace(start=min_val, stop=max_val, num=1000)
    y = x.reshape(-1, 1)

    h = sorry_func([x, y])
    cs = plt.contourf(x, x, h)
    _ = f.colorbar(cs, ax=ax)
    arrows = [
        ax.arrow(0, 0, 0, 0, head_width=0.05, facecolor="black") for _ in range(n_nodes)
    ]
    (line,) = ax.plot([], [], "o--", lw=3)

    def animate(chain):

        x = chain.coordinates[:, 0]
        y = chain.coordinates[:, 1]

        for arrow, (x_i, y_i), (dx_i, dy_i) in zip(
            arrows, chain.coordinates, chain.gradients
        ):
            arrow.set_data(x=x_i, y=y_i, dx=-1 * dx_i, dy=-1 * dy_i)

        line.set_data(x, y)
        return (x for x in arrows)

    anim = FuncAnimation(
        fig=f,
        func=animate,
        frames=chain_traj,
        blit=True,
        repeat_delay=1000,
        interval=200,
    )
    # anim.save(f'pot{ind_f}_super_trippy.gif')
    plt.show()

def plot_ethan(chain):
    size = 8

    new_chain = chain
    # max_val = s
    min_val = presets["min_size"]
    max_val = presets["max_size"]
    num = 10
    fig = 10
    f, _ = plt.subplots(figsize=(1.18 * fig, fig))
    x = np.linspace(start=min_val, stop=max_val, num=1000)
    y = x.reshape(-1, 1)

    h = sorry_func([x, y])
    cs = plt.contourf(x, x, h)
    _ = f.colorbar(cs)
    # plt.plot(
    #     [(node.coords[0]) for node in orig_chain],
    #     [(node.coords[1]) for node in orig_chain],
    #     "^--",
    #     c="white",
    #     label="original",
    #     ms=9,
    # )

    points_x = [node.coords[0] for node in new_chain]
    points_y = [node.coords[1] for node in new_chain]
    # plt.plot([toy_potential_2(point) for point in new_chain])
    plt.plot(points_x, points_y, "o--", c="white", label="NEB", ms=9)
    # psave(new_chain, "new_chain.p")
    plt.show()

def plot_func(neb_obj: NEB):
    size = 8

    en_func = neb_obj.initial_chain[0].en_func
    orig_chain = neb_obj.initial_chain
    new_chain = neb_obj.chain_trajectory[-1]

    # min_val = -s
    # max_val = s
    min_val = presets["min_size"]
    max_val = presets["max_size"]
    num = 10
    fig = 10
    f, _ = plt.subplots(figsize=(1.18 * fig, fig))
    x = np.linspace(start=min_val, stop=max_val, num=1000)
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
        ms=9,
    )

    points_x = [node.coords[0] for node in new_chain]
    points_y = [node.coords[1] for node in new_chain]
    # plt.plot([toy_potential_2(point) for point in new_chain])
    plt.plot(points_x, points_y, "o--", c="white", label="NEB", ms=9)
    # psave(new_chain, "new_chain.p")
    plt.show()


def plot_2D(neb_obj: NEB):
    opt_chain = neb_obj.chain_trajectory[-1]
    ens = np.array([node.energy for node in opt_chain])
    # ens = ens*627.5

    orig_ens = np.array([node.energy for node in neb_obj.initial_chain])
    # orig_ens = orig_ens*627.5
    orig_ens = orig_ens - ens[0]

    ens = ens - ens[0]

    print(f"{opt_chain.integrated_path_length=}")
    plt.plot(opt_chain.integrated_path_length, ens, "o--", label="last chain")
    plt.plot(
        np.linspace(0, opt_chain.integrated_path_length[-1], len(orig_ens)),
        orig_ens,
        "*",
        label="original",
    )
    plt.legend()
    plt.show()


def main():
    nimages = 10

    ### node 2d
    # end_point = (3.00002182, 1.99995542)
    # end_point = (2.129, 2.224)
    # start_point = (-3.77928812, -3.28320392)

    ### node 2d - 2
    # start_point = (-1, 1)
    # end_point = (1, 1)
    # end_point = (1, -1)
    # end_point = (1.01, -1.01)

    # ## node 2d - ITM
    # start_point = (0, 0)
    # end_point = (1, 0)
    # end_point = (-1, -1)
    # end_point = (.5, -.5)

    # ## node 2d - LEPS
    # start_point = [0.74200203, 4]
    # end_point = [4, 0.74200311]
    
    ### node 2d - flower
    start_point = [-2.59807434, -1.499999  ]
    # end_point = [5, .000001]
    end_point = [2.5980755 , 1.49999912]
    
    
    coords = np.linspace(start_point, end_point, nimages)
    coords[1:-1] += np.random.normal(scale=.2, size=coords[1:-1].shape)
    # coords[1:-1] -= np.random.normal(scale=.15)
    # coords[5]+= np.array([0,.2])
    
    
    
    

    # coords = np.linspace(start_point, (-1.2, 1), 15)
    # coords = np.append(coords, np.linspace(end_point, (1.2, -1), 15), axis=0)

    # ks = np.array([0.1, 0.1, 10, 10, 10, 10, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1]).reshape(-1,1)
    # ks = np.array([1]*(len(coords)-2)).reshape(-1,1)
    # kval = .01
    
    
    
    
    ks = .01
    cni = ChainInputs(
        k=ks,
        node_class=presets["node"],
        delta_k=0,
        step_size=.5,
        do_parallel=False,
        use_geodesic_interpolation=False,
    )
    gii = GIInputs(nimages=nimages)
    nbi = NEBInputs(tol=.1, v=1, max_steps=1000, climb=False)
    chain = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni)
    m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii)
    h_root_node, out_chain = m.find_mep_multistep(input_chain=chain)
    # out_chain.plot_chain()
    h_root_node.draw()
    # plot_func(h[0])
    [animate_func(obj) for obj in h_root_node.get_optimization_history()]
    plot_ethan(out_chain)
    
    
    # frankenstein = h[0]
    # frankenstein.chain_trajectory.extend(h[1][0].chain_trajectory)
    # frankenstein.chain_trajectory.extend(h[2][0].chain_trajectory)
    # frankenstein.chain_trajectory.append(out_chain)
    # animate_func(frankenstein)
    # # out_chain.plot_chain()
    # print(f'{coords=}')
    # print(f'{h[0].optimized.coordinates=}')
    # print(f'{h[1][0].optimized.coordinates=}')
    # print(f'{h[2][0].optimized.coordinates=}')
    # print(f"{out_chain.coordinates=}")
    # plot_func(frankenstein)
    

if __name__ == "__main__":
    main()
