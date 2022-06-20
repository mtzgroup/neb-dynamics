
from http.client import NETWORK_AUTHENTICATION_REQUIRED
from pathlib import Path
from platform import node

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from neb_dynamics.ALS import BOHR_TO_ANGSTROMS

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
    # nimages = 10
    # end_point = (3.00002182, 1.99995542)
    # start_point = (-3.77928812, -3.28320392)


    # coords = np.linspace(start_point, end_point, nimages)
    # chain = Chain.from_list_of_coords(k=1, list_of_coords=coords, node_class=Node2D)
    # n = NEB(initial_chain=chain, max_steps=1, grad_thre=0.01, mag_grad_thre=1)
    # try: 
    #     n.optimize_chain()
    # except NoneConvergedException as e:
    #     print(e.obj.chain_trajectory[-1].gradients)
    #     animate_func(e.obj)
    #     plot_func(e.obj)
        

    # plot_2D(n)
    # plot_func(n)
    # animate_func(n)

    fp = Path("../example_cases/DA_geodesic_opt.xyz")
#     # fp = Path("../example_cases/pdda_traj_xtb_optmized_geo_att2.xyz")
#     # fp = Path("./example_cases/PDDA_geodesic.xyz")


    kval = 0.01
    traj = Trajectory.from_xyz(fp)
    coords = traj.to_list()
    chain = Chain.from_list_of_coords(k=kval, list_of_coords=coords, node_class=Node3D)
    n = NEB(initial_chain=chain, grad_thre=0.0001, en_thre=0.0001, mag_grad_thre=1, max_steps=1000, redistribute=False, remove_folding=False)

    try:
        n.optimize_chain()
        plot_2D(n)


        
        # n.write_to_disk(fp.parent / f"neb_DA_k{kval}_redist_and_defolded.xyz")
        # n.write_to_disk(fp.parent / f"neb_DA_k{kval}.xyz")
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

    except NoneConvergedException as e:
        plot_2D(e.obj)

#     # nsteps = 1
#     # chain_previous = n.initial_chain.copy()
    
#     # chain=chain_previous

#     # chain_grads = []
#     # for prev_node, current_node, next_node in chain.iter_triplets():

#     #     vec_tan_path = chain._create_tangent_path(prev_node, current_node, next_node)
#     #     unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)


#     #     # print(f"\n\n\n{unit_tan_path=}\n\n\n")


#     #     pe_grad = current_node.gradient

#     #     # print(f"\n\n\n{pe_grad=}\n\n\n")


#     #     pe_grad_nudged_const = current_node.dot_function(pe_grad, unit_tan_path)
#     #     pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path


#     #     # print(f"\n\n\n{pe_grad_nudged=}\n\n\n")


#     #     force_spring = chain._get_spring_force(
#     #         prev_node=prev_node,
#     #         current_node=current_node,
#     #         next_node=next_node,
#     #     )


#     #     # print(f"\n\n\n{force_spring=}\n\n\n")


#     #     force_spring_nudged_const = current_node.dot_function(
#     #         force_spring, unit_tan_path
#     #     )

#     #     # print(f"\n\n\n{force_spring_nudged_const=}\n\n\n")

#     #     force_spring_nudged = force_spring - force_spring_nudged_const * unit_tan_path
#     #     # force_spring_nudged = force_spring_nudged_const * unit_tan_path
#     #     # print(f"\n\n\n{force_spring_nudged=}\n\n\n")

#     #     # print(f"\n\n\n{force_spring_nudged=}\n\n\n")





#     #     # ANTI-KINK FORCE
#     #     vec_2_to_1 = next_node.coords - current_node.coords
#     #     vec_1_to_0 = current_node.coords - prev_node.coords
#     #     cos_phi = current_node.dot_function(vec_2_to_1, vec_1_to_0) / (
#     #         np.linalg.norm(vec_2_to_1) * np.linalg.norm(vec_1_to_0)
#     #     )

#     #     f_phi = 0.5 * (1 + np.cos(np.pi * cos_phi))


#     #     anti_kinking_grad = (f_phi * (force_spring_nudged))
#     #     # print(f"\n\n\n{anti_kinking_grad=}\n\n\n")



#     #     pe_and_spring_grads = -1 * (-1 * pe_grad_nudged +  force_spring_nudged)


#     #     grad = pe_and_spring_grads + anti_kinking_grad

#     #     # print(f"\n\n\n{grad=}\n\n\n")

#     #     # ------- def correct up to here
        

#     #     chain_grads.append(grad)
        
#     # # print(f"\n\n\n{chain.gradients=}\n\n\n")
#     # # print(f"\n\n\n{chain_grads=}\n\n\n")
#     # # print(f"\n\n\n{chain.displacements=}\n\n\n")
#     # # print(f"\n\n\n{chain_grads[1]=}\n\n\n")

#     # coords = chain.coordinates

#     # new_chain_coordinates = (
#     #         coords - (chain.gradients*BOHR_TO_ANGSTROMS) * chain.displacements
#     #     )


#     # # print(f"\n\n\n{chain.coordinates[1]=}\n\n\n")
#     # # print(f"\n\n\n{chain.gradients[1]=}\n\n\n")
#     # # print(f"\n\n\n{chain.displacements[1]=}\n\n\n")


#     # print(f"\n\n\n{new_chain_coordinates[1]}\n\n\n")

    
#     # # new_chain = chain.copy()
#     # # for node, new_coords in zip(new_chain.nodes, new_chain_coordinates):
#     # #     node.update_coords(new_coords)

#     # # print(f"{chain.gradients[1]=}")


#     # # print(f"{new_chain.coordinates[1]=}")

 




if __name__ == "__main__":
    main()
