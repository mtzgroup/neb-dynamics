from pathlib import Path
import numpy as np
from neb_dynamics.NEB import Dimer, Node2D, NEB, Chain, Node3D

import matplotlib.pyplot as plt

from neb_dynamics.trajectory import Trajectory



s=4
import random
random.seed(70)

def en_func(inp):
    
    x, y = inp
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def plot_dimer(dimer_obj: Dimer,c='gray'):

    min_val = -s
    max_val = s
    # min_val = -.5
    # max_val = 1.5
    num = 10
    fig = 10
    f, _ = plt.subplots(figsize=(1.18 * fig, fig))
    x = np.linspace(start=min_val, stop=max_val, num=num)
    y = x.reshape(-1, 1)

    h = en_func([x, y])
    cs = plt.contourf(x, x, h)
    _ = f.colorbar(cs)

    r1,r2 = dimer_obj.optimized_dimer
    unit_dir = dimer_obj.get_unit_dir(dimer_obj.optimized_dimer)
    
    x1,y1 = r1.coords
    x2,y2 = r2.coords
    x_mid, y_mid = r1.coords + dimer_obj.delta_r*unit_dir
    
    plt.plot([x1,x_mid,x2],[y1,y_mid,y2], 'o--', color=c)
    plt.show()

def main():
    nimages = 6
    end_point = (2.129, 2.224)
    start_point = (-3.77928812, -3.28320392)
    coords = np.linspace(start_point, end_point, nimages)
    ks = 6
    chain = Chain.from_list_of_coords(k=ks, list_of_coords=coords, node_class=Node2D, delta_k=0, step_size=.01)
    n = NEB(initial_chain=chain, max_steps=500, grad_thre_per_atom=1, climb=False, vv_force_thre=0.0)
    n.optimize_chain()
    node_for_dimer = n.optimized[4]

    d = Dimer(initial_node=node_for_dimer, delta_r=0.1, d_theta=0.0001,step_size=0.001)
    d.find_ts()
    print("init:",d.initial_node.coords)
    print("end:", d.ts_node.coords)

    plot_dimer(d)

    # fp = Path("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement/traj_13-6_neb.xyz")
    # traj = Trajectory.from_xyz(fp)
    # chain = Chain.from_traj(traj=traj, k=0.1, delta_k=0, step_size=0.1, node_class=Node3D)
    # node_for_dimer = chain[13]
    # d = Dimer(initial_node=node_for_dimer, delta_r=1, d_theta=0.01, step_size=0.37)
    # d.find_ts()

    # d.ts_node.tdstructure.write_to_disk("ts.xyz")

if __name__=="__main__":
    main()