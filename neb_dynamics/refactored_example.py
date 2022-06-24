
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from pathlib import Path
from platform import node
from xmlrpc.client import FastMarshaller

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from regex import F
from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from neb_dynamics.NEB import NEB, AlessioError, Chain, Node2D, Node3D, NoneConvergedException
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
    (line,) = ax.plot([], [], 'o--', lw=3)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(chain):

        x = chain[:, 0]
        y = chain[:, 1]
        line.set_data(x, y)
        return (line,)

    anim = FuncAnimation(fig=f, func=animate, frames=np.array([chain.coordinates for chain in chain_traj]), blit=True, repeat_delay=1000, interval=200)
    # anim.save('c-neb.gif')
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
    # nimages = 100
    # end_point = (3.00002182, 1.99995542)
    # # end_point = (-2.8, 3.11)
    # start_point = (-3.77928812, -3.28320392)



    # coords = np.linspace(start_point, end_point, nimages)
    # # ks = np.array([0.1, 0.1, 10, 10, 10, 10, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1]).reshape(-1,1)
    # # ks = np.array([1]*(len(coords)-2)).reshape(-1,1)
    # ks = 1
    # chain = Chain.from_list_of_coords(k=ks, list_of_coords=coords, node_class=Node2D, delta_k=0.9)
    # n = NEB(initial_chain=chain, max_steps=1000, grad_thre=1,en_thre=1e-2, mag_grad_thre=1000 ,redistribute=False,
    # climb=False)
    # try: 
    #     n.optimize_chain()
    #     animate_func(n)
    #     plot_func(n)
    #     plot_2D(n)

    # except AlessioError as e:
    #     print(e.message)
    # except NoneConvergedException as e:
    #     print(e.obj.chain_trajectory[-1].gradients)
    #     # plot_2D(e.obj)
    #     animate_func(e.obj)
    #     plot_func(e.obj)
        

 

    fp = Path("../example_cases/DA_geodesic_opt.xyz")
# #     # fp = Path("../example_cases/debug_geodesic_claisen.xyz")
# #     fp = Path(f"../example_cases/neb_DA_k0.1.xyz")
# # #     # fp = Path("../example_cases/pdda_traj_xtb_optmized_geo_att2.xyz")
# # #     # fp = Path("./example_cases/PDDA_geodesic.xyz")

    kval = 1
    traj = Trajectory.from_xyz(fp)
    coords = traj.to_list()
    chain = Chain.from_list_of_coords(k=kval, list_of_coords=coords, node_class=Node3D, delta_k=0.9)
    # chain[6].do_climb = True


    n = NEB(initial_chain=chain, grad_thre=0.0001, en_thre=0.0001, mag_grad_thre=100, max_steps=1000, redistribute=False, remove_folding=False,
    climb=True)

    try:
        n.optimize_chain()
        plot_2D(n)


        
        # n.write_to_disk(fp.parent / f"neb_DA_k{kval}_redist_and_defolded.xyz")
        # n.write_to_disk(fp.parent / f"neb_DA_k{kval}_cneb.xyz")
        # n.write_to_disk(fp.parent / f"neb_PDDA_k{kval}.xyz")


        opt_chain = n.optimized
        all_dists = []
        for i in range(len(opt_chain)-2):
            c0 = opt_chain[i].coords
            c1 = opt_chain[i+1].coords

            

            dist = np.linalg.norm(c1 - c0)
            all_dists.append(dist)

        plt.bar(x=list(range(len(all_dists))), height=all_dists)
        plt.show()

    except AlessioError as e:
        print(e.message)

    except NoneConvergedException as e:
        plot_2D(e.obj)


if __name__ == "__main__":
    main()
