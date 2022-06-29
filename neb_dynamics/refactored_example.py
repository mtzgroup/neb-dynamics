

from pathlib import Path
from platform import node
from tracemalloc import start
from matplotlib.patches import FancyArrow


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from neb_dynamics.constants import BOHR_TO_ANGSTROMS

from neb_dynamics.NEB import NEB, AlessioError, Chain, Node2D, Node2D_2, Node2D_ITM, Node3D, NoneConvergedException
from neb_dynamics.trajectory import Trajectory


s=2

# def sorry_func(inp):
    
#     x, y = inp
#     return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def sorry_func(inp): # https://theory.cm.utexas.edu/henkelman/pubs/sheppard11_1769.pdf
    x, y = inp
    A = np.cos(np.pi*x) 
    B = np.cos(np.pi*y) 
    C = np.pi*np.exp(-np.pi*x**2)
    D = (np.exp(-np.pi*(y - 0.8)**2))  + np.exp(-np.pi*(y+0.8)**2)
    return A + B + C*D


# def sorry_func(inp):
#     x, y = inp
#     Ax = 1
#     Ay = 1
#     return -1*(Ax*np.cos(2*np.pi*x) + Ay*np.cos(2*np.pi*y))



def animate_func(neb_obj: NEB):
    n_nodes = len(neb_obj.initial_chain.nodes)
    en_func = neb_obj.initial_chain[0].en_func
    chain_traj = neb_obj.chain_trajectory
    plt.style.use("seaborn-pastel")

    figsize = 5

    f, ax = plt.subplots(figsize=(1.18 * figsize, figsize))


    min_val = -s
    max_val = s

    # min_val = -.5
    # max_val = 1.5

    x = np.linspace(start=min_val, stop=max_val, num=n_nodes)
    y = x.reshape(-1, 1)

    # h = en_func(x, y)
    h = sorry_func([x, y])
    cs = plt.contourf(x, x, h)
    _ = f.colorbar(cs, ax=ax)
    arrows = [ax.arrow(0,0,0,0,head_width=.05, facecolor='black') for _ in range(n_nodes)]
    (line,)  = ax.plot([], [], 'o--', lw=3)

    def animate(chain):

        x = chain.coordinates[:, 0]
        y = chain.coordinates[:, 1]
        
        for arrow, (x_i, y_i), (dx_i, dy_i) in zip(arrows, chain.coordinates, chain.gradients):
            arrow.set_data(x=x_i, y=y_i, dx=-1*dx_i, dy=-1*dy_i)

        line.set_data(x, y)
        return (x for x in arrows)

    anim = FuncAnimation(fig=f, func=animate, frames=chain_traj, blit=True, repeat_delay=1000, interval=200)
    # anim.save('cool_arrows.gif')
    plt.show()


def plot_func(neb_obj: NEB):

    en_func = neb_obj.initial_chain[0].en_func
    orig_chain = neb_obj.initial_chain
    new_chain = neb_obj.chain_trajectory[-1]

    min_val = -s
    max_val = s
    # min_val = -.5
    # max_val = 1.5
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
    nimages = 70

    # ### node 2d
    # end_point = (3.00002182, 1.99995542)
    # end_point = (2.129, 2.224)
    # start_point = (-3.77928812, -3.28320392)

    # ### node 2d - 2
    start_point = (-1, 1)
    # end_point = (1, 1)
    # end_point = (-1, -1)
    end_point = (1, -1)

    ### node 2d - ITM
    # start_point = (0, 0)
    # end_point = (1, 0)
    # end_point = (-1, -1)
    # end_point = (.5, -.5)

    coords = np.linspace(start_point, end_point, nimages)
    # coords[5]+= np.array([0,.2])
    


    # coords = np.linspace(start_point, (-1.2, 1), 15)
    # coords = np.append(coords, np.linspace(end_point, (1.2, -1), 15), axis=0)


    # ks = np.array([0.1, 0.1, 10, 10, 10, 10, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1]).reshape(-1,1)
    # ks = np.array([1]*(len(coords)-2)).reshape(-1,1)
    # kval = .01
    ks = .1
    chain = Chain.from_list_of_coords(k=ks, list_of_coords=coords, node_class=Node2D_2, delta_k=0, step_size=.01)
    n = NEB(initial_chain=chain, max_steps=500, grad_thre=.1, climb=False, vv_force_thre=0.0)#,en_thre=1e-2, mag_grad_thre=1000 ,redistribute=False,
    try: 
        n.optimize_chain()
        animate_func(n)
        plot_func(n)
        plot_2D(n)

    except AlessioError as e:
        print(e.message)
    except NoneConvergedException as e:
        print(e.obj.chain_trajectory[-1].gradients)
        # plot_2D(e.obj)
        animate_func(e.obj)
        plot_func(e.obj)
        

 

    # fp = Path("../example_cases/DA_geodesic_opt.xyz")
#     fp = Path("./example_cases/debug_geodesic_claisen.xyz")
# # # # # #     fp = Path(f"../example_cases/neb_DA_k0.1.xyz")
# # # # # # #     # fp = Path("../example_cases/pdda_traj_xtb_optmized_geo_att2.xyz")
# # # # # # #     # fp = Path("./example_cases/PDDA_geodesic.xyz")

#     kval = .001 
#     traj = Trajectory.from_xyz(fp)
#     coords = traj.to_list()
#     chain = Chain.from_list_of_coords(k=kval, list_of_coords=coords, node_class=Node3D, delta_k=0, step_size=0.01, velocity=np.zeros_like([struct.coords for struct in traj]))
#     # chain[6].do_climb = True


#     # n = NEB(initial_chain=chain, grad_thre=0.1, en_thre=0.0001, mag_grad_thre=100, max_steps=1000, redistribute=False, remove_folding=False,
#     # climb=True)
#     n = NEB(initial_chain=chain, grad_thre=0.03, max_steps=2000, redistribute=False, remove_folding=False,
#     climb=False, vv_force_thre=.3)

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



#         # now make it climb
        
#         # n = NEB(initial_chain=opt_chain, grad_thre=0.01, max_steps=2000, redistribute=False, remove_folding=False,
#         #   climb=True, vv_force_thre=0) # i.e. dont do VV
        
#         # n.climb_chain(opt_chain)

#         # plot_2D(n)
#         # n.write_to_disk(fp.parent / f"neb_DA_k{kval}_cneb.xyz")



        
#     except AlessioError as e:
#         print(e.message)

#     except NoneConvergedException as e:
#         plot_2D(e.obj)


if __name__ == "__main__":
    main()
