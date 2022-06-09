
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from neb_dynamics.NEB import Chain, NEB


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
    h = en_func([x, y])
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
    new_chain = neb_obj.optimized

    min_val = -4
    max_val = 4
    num = 10
    fig = 10
    f, _ = plt.subplots(figsize=(1.18 * fig, fig))
    x = np.linspace(start=min_val, stop=max_val, num=num)
    y = x.reshape(-1, 1)

    h = en_func([x, y])
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


def main():
    nimages = 10
    end_point = (3.00002182, 1.99995542)
    start_point = (-3.77928812, -3.28320392)



    coords = np.linspace(start_point, end_point, nimages)
    chain = Chain.from_list_of_coords(k=10, list_of_coords=coords,
                                      grad_func=toy_grad_2, en_func=toy_potential_2,
                                      dot_func=np.dot)

    chain = Chain.from_list_of_coords(k=10, 
                                      list_of_coords=tds,
                                      grad_func=toy_grad_2, 
                                      en_func=toy_potential_2,
                                      dot_func=np.dot)

    print(f"{coords=}")

    n = NEB(initial_chain=chain)

    n.optimize_chain()

    plot_func(n)
    animate_func(n)


if __name__ == "__main__":
    main()
