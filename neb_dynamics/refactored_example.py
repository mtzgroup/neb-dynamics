import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from neb_dynamics.NEB import NEB, NoneConvergedException
from chain import Chain
from neb_dynamics.nodes.Node2d import (
    Node2D,
    Node2D_2,
    Node2D_ITM,
    Node2D_LEPS,
    Node2D_Flower,
)
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from nodes.node import AlessioError
from itertools import product
from TS.TS_PRFO import TS_PRFO
from neb_dynamics.potential_functions import (
    sorry_func_0,
    sorry_func_1,
    sorry_func_2,
    sorry_func_3,
    flower_func,
)
from neb_dynamics.Inputs import ChainInputs, NEBInputs

sfs = [sorry_func_0, sorry_func_1, sorry_func_2, sorry_func_3, flower_func]
index = 1
nodes = [Node2D, Node2D_2, Node2D_ITM, Node2D_LEPS, Node2D_Flower]
min_sizes = [-4, -2, -2, 0.5, -4]
max_sizes = [4, 2, 2, 4, 4]

presets = {
    "pot_func": sfs[index],
    "node": nodes[index],
    "min_size": min_sizes[index],
    "max_size": max_sizes[index],
}


def sorry_func(inp):
    return sfs[index](inp)


s = 4


def animate_func(neb_obj: NEB):
    n_nodes = len(neb_obj.initial_chain.nodes)
    en_func = neb_obj.initial_chain[0].en_func
    chain_traj = neb_obj.chain_trajectory
    # plt.style.use("seaborn-pastel")

    figsize = 5

    f, ax = plt.subplots(figsize=(1.618 * figsize, figsize))

    min_val = -s
    max_val = s

    min_val = presets["min_size"]
    max_val = presets["max_size"]

    gridsize = 100
    x = np.linspace(start=min_val, stop=max_val, num=gridsize)
    h_flat_ref = np.array([presets["node"].en_func_arr(pair) for pair in product(x, x)])
    h = h_flat_ref.reshape(gridsize, gridsize).T
    cs = plt.contourf(x, x, h)
    _ = f.colorbar(cs, ax=ax)

    arrows = [  # these are the total arrows
        ax.arrow(0, 0, 0, 0, head_width=0.05, facecolor="black") for _ in range(n_nodes)
    ]

    arrows2 = [  # these are the gradient arrows
        ax.arrow(0, 0, 0, 0, head_width=0.05, facecolor="red", color="red")
        for _ in range(n_nodes)
    ]

    arrows3 = [  # these are the tangets arrows
        ax.arrow(0, 0, 0, 0, head_width=0.05, facecolor="red", color="blue")
        for _ in range(n_nodes)
    ]

    arrows4 = [  # these are the tangets arrows
        ax.arrow(0, 0, 0, 0, head_width=0.05, facecolor="red", color="gray")
        for _ in range(n_nodes)
    ]

    (line,) = ax.plot([], [], "o--", lw=1)

    def animate(chain):

        x = chain.coordinates[:, 0]
        y = chain.coordinates[:, 1]

        for arrow, (x_i, y_i), (dx_i, dy_i) in zip(
            arrows, chain.coordinates, chain.gradients
        ):
            arrow.set_data(x=x_i, y=y_i, dx=-1 * dx_i, dy=-1 * dy_i)

        for arrow2, (x_i, y_i), (dx_i, dy_i) in zip(
            arrows2, chain.coordinates, [node.gradient for node in chain.nodes]
        ):
            arrow2.set_data(x=x_i, y=y_i, dx=-1 * dx_i, dy=-1 * dy_i)

        tans = [
            chain._create_tangent_path(*triplet) for triplet in chain.iter_triplets()
        ]
        tans_unit = [tan / np.linalg.norm(tan) for tan in tans]

        for arrow3, (x_i, y_i), (dx_i, dy_i) in zip(
            arrows3, chain.coordinates[1:-1], tans_unit
        ):
            arrow3.set_data(x=x_i, y=y_i, dx=1 * dx_i, dy=1 * dy_i)

        for arrow4, (x_i, y_i), (dx_i, dy_i) in zip(
            arrows4,
            chain.coordinates[1:-1],
            [
                chain.get_force_spring_nudged(
                    prev_node, current_node, next_node, unit_tan
                )
                for (prev_node, current_node, next_node), unit_tan in zip(
                    chain.iter_triplets(), tans_unit
                )
            ],
        ):
            arrow4.set_data(x=x_i, y=y_i, dx=1 * dx_i, dy=1 * dy_i)

        line.set_data(x, y)
        # all_arrows = arrows + arrows2 + arrows3 + arrows4
        # all_arrows = arrows + arrows2 + arrows4

        return (x for x in arrows)
        # return (x for x in all_arrows)

    anim = FuncAnimation(
        fig=f,
        func=animate,
        frames=chain_traj,
        blit=True,
        repeat_delay=1000,
        interval=200,
    )
    # anim.save(f'flower_nimages_{n_nodes}_k_{neb_obj.initial_chain.parameters.k}.gif')
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
    fig = 10
    f, _ = plt.subplots(figsize=(1.18 * fig, fig))

    gridsize = 100
    x = np.linspace(start=min_val, stop=max_val, num=gridsize)
    y = x.reshape(-1, 1)
    h_flat_ref = np.array(
        [presets["node_class"].en_func_arr(pair) for pair in product(x, x)]
    )
    h = h_flat_ref.reshape(gridsize, gridsize).T
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
    nimages = 15

    ### node 2d
    # end_point = (3.00002182, 1.99995542)
    # end_point = (2.129, 2.224)
    # start_point = (-3.77928812, -3.28320392)

    ### node 2d - 2
    start_point = (-1, 1)
    # end_point = (1, 1)
    # end_point = (1, -1)
    end_point = (1.01, -1.01)

    # ## node 2d - ITM
    # start_point = (0, 0)
    # end_point = (1, 0)
    # end_point = (-1, -1)
    # end_point = (.5, -.5)

    # ## node 2d - LEPS
    # start_point = [0.74200203, 4]
    # end_point = [4, 0.74200311]

    # ## node 2d - flower
    # start_point = [-2.59807434, -1.499999  ]
    # end_point = [2.5980755 , 1.49999912]

    coords = np.linspace(start_point, end_point, nimages)
    # coords[1:-1]+= [-1,1]
    # coords = np.linspace(start_point, (-1.2, 1), 15)
    # coords = np.append(coords, np.linspace(end_point, (1.2, -1), 15), axis=0)

    # ks = np.array([0.1, 0.1, 10, 10, 10, 10, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1]).reshape(-1,1)
    # ks = np.array([1]*(len(coords)-2)).reshape(-1,1)
    # kval = .01
    ks = 0.1
    cni = ChainInputs(k=ks, node_class=presets["node"], delta_k=0, do_parallel=False)
    nbi = NEBInputs(tol=0.1, barrier_thre=5, v=True, max_steps=500, climb=False)
    chain = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni)
    opt = VelocityProjectedOptimizer(timestep=0.01, activation_tol=0.1)
    n = NEB(initial_chain=chain, parameters=nbi, optimizer=opt)

    try:
        n.optimize_chain()

        print(f"{n.optimized.coordinates=}")
        animate_func(n)
        n.plot_convergence_metrics()
        # plot_func(n)
        # # plot_2D(n)

        # node_ind = np.argmax(n.optimized.energies)
        # node = n.optimized[node_ind]
        # print(f"Finding TS using node_index {node_ind} as guess >> {node.coords}")

        # tsopt = TS_PRFO(initial_node=node, dr=1, max_step_size=1)
        # print(f"SP = {tsopt.ts.coords}")

    except AlessioError as e:
        print(e.message)
    except NoneConvergedException as e:
        print(e.obj.chain_trajectory[-1].gradients)
        # plot_2D(e.obj)
        animate_func(e.obj)
        plot_func(e.obj)


if __name__ == "__main__":
    main()
