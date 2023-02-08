import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Chain import Chain
from neb_dynamics.Node2d import Node2D, Node2D_2, Node2D_ITM, Node2D_LEPS
from neb_dynamics.Node import AlessioError
from neb_dynamics.TS_PRFO import TS_PRFO
from neb_dynamics.potential_functions import (
    sorry_func_0,
    sorry_func_1,
    sorry_func_2,
    sorry_func_3,
)
from neb_dynamics.Inputs import ChainInputs, NEBInputs

sfs = [sorry_func_0, sorry_func_1, sorry_func_2, sorry_func_3]
index = 0
nodes = [Node2D, Node2D_2, Node2D_ITM, Node2D_LEPS]
min_sizes = [-4, -2, -2, 0.5]
max_sizes = [4, 2, 2, 4]

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
    plt.style.use("seaborn-pastel")

    figsize = 5

    f, ax = plt.subplots(figsize=(1.618 * figsize, figsize))

    min_val = -s
    max_val = s

    min_val = presets["min_size"]
    max_val = presets["max_size"]

    x = np.linspace(start=min_val, stop=max_val, num=n_nodes)
    y = x.reshape(-1, 1)

    # h = en_func(x, y)
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
    nimages = 5

    ### node 2d
    # end_point = (3.00002182, 1.99995542)
    end_point = (2.129, 2.224)
    start_point = (-3.77928812, -3.28320392)

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
    coords = np.linspace(start_point, end_point, nimages)
    # coords[5]+= np.array([0,.2])

    # coords = np.linspace(start_point, (-1.2, 1), 15)
    # coords = np.append(coords, np.linspace(end_point, (1.2, -1), 15), axis=0)

    # ks = np.array([0.1, 0.1, 10, 10, 10, 10, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1]).reshape(-1,1)
    # ks = np.array([1]*(len(coords)-2)).reshape(-1,1)
    # kval = .01
    ks = 0.1
    cni = ChainInputs(k=ks, node_class=presets["node"], delta_k=0, step_size=0.1, do_parallel=False)
    nbi = NEBInputs(tol=0.01)
    chain = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni)
    n = NEB(initial_chain=chain, parameters=nbi)

    try:
        n.optimize_chain()

        print(f"{n.optimized.coordinates=}")
        animate_func(n)
        plot_func(n)
        # plot_2D(n)

        node_ind = np.argmax(n.optimized.energies)
        node = n.optimized[node_ind]
        print(f"Finding TS using node_index {node_ind} as guess >> {node.coords}")

        tsopt = TS_PRFO(initial_node=node)
        print(f"SP = {tsopt.ts.coords}")

    except AlessioError as e:
        print(e.message)
    except NoneConvergedException as e:
        print(e.obj.chain_trajectory[-1].gradients)
        # plot_2D(e.obj)
        animate_func(e.obj)
        plot_func(e.obj)


#     fp = Path("../example_cases/DA_geodesic_opt.xyz")
#     fp = Path("./example_cases/debug_geodesic_claisen.xyz")
# # # # # # #     fp = Path(f"../example_cases/neb_DA_k0.1.xyz")
# # # # # # # #     # fp = Path("../example_cases/pdda_traj_xtb_optmized_geo_att2.xyz")
# # # # # # # #     # fp = Path("./example_cases/PDDA_geodesic.xyz")

#     kval = .001
#     traj = Trajectory.from_xyz(fp)
#     chain = Chain.from_traj(traj=traj,k=kval, node_class=Node3D, delta_k=0, step_size=0.37)


#     # n = NEB(initial_chain=chain, grad_thre_per_atom=0.1, en_thre=0.0001, mag_grad_thre=100, max_steps=1000, redistribute=False, remove_folding=False,
#     # climb=True)0
#     n = NEB(initial_chain=chain, grad_thre_per_atom=0.0016, max_steps=2000, redistribute=False, remove_folding=False,
#     climb=False, vv_force_thre=0.0)

#     try:
#         n.optimize_chain()
#         plot_2D(n)


#         # n.write_to_disk(fp.parent / f"cneb_DA_k{kval}.xyz")
#         # n.write_to_disk(fp.parent / f"neb_CR_k{kval}_neb.xyz")
#         # n.write_to_disk(fp.parent / f"neb_PDDA_k{kval}.xyz")


#         opt_chain = n.optimized
#         all_dists = []
#         for i in range(len(opt_chain)-2):
#             c0 = opt_chain[i].coords
#             c1 = opt_chain[i+1].coords


#             dist = np.linalg.norm(c1 - c0)
#             all_dists.append(dist)

#         plt.bar(x=list(range(len(all_dists))), height=all_dists)
#         plt.show()


# #         # now make it climb

# #         # n = NEB(initial_chain=opt_chain, grad_thre=0.01, max_steps=2000, redistribute=False, remove_folding=False,
# #         #   climb=True, vv_force_thre=0) # i.e. dont do VV

# #         # n.climb_chain(opt_chain)

# #         # plot_2D(n)
# #         # n.write_to_disk(fp.parent / f"neb_DA_k{kval}_cneb.xyz")


#     except AlessioError as e:
#         print(e.message)

#     except NoneConvergedException as e:
#         plot_2D(e.obj)


if __name__ == "__main__":
    main()
