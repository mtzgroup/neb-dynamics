from NEB import neb
import numpy as np
import matplotlib.pyplot as plt



def main():
    nimages = 15
    end_point = (3.00002182, 1.99995542)
    start_point = (-3.77928812, -3.28320392)
    
    
    def toy_potential_2(inp):
        x, y = inp
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2



    def toy_grad_2(inp):
        x, y = inp
        dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
        return np.array([dx, dy])


    
    
    
    chain = np.linspace(start_point, end_point, nimages)
    new_chain, chain_traj = neb().optimize_chain(
            chain=chain,
            grad_func=toy_grad_2,
            en_func=toy_potential_2,
            k=10,
            max_steps=1000,
            grad_thre=0.01
    )




    min_val = -4
    max_val = 4
    num = 10
    fig = 10
    f, ax = plt.subplots(figsize=(1.18 * fig, fig))
    x = np.linspace(start=min_val, stop=max_val, num=num)
    y = x.reshape(-1, 1)


    h = toy_potential_2([x, y])
    cs = plt.contourf(x, x, h)
    cbar = f.colorbar(cs)
    plt.plot(
        [(point[0]) for point in chain],
        [(point[1]) for point in chain],
        "^--",
        c="white",
        label="original",
    )

    points_x = [point[0] for point in new_chain]
    points_y = [point[1] for point in new_chain]
    plt.plot(points_x, points_y, "o--", c="white", label="NEB")
    plt.show()
if __name__=="__main__":
    main()