# -*- coding: utf-8 -*-
from neb_dynamics.NEB import Node2D, Chain, NEB
import numpy as np
import matplotlib.pyplot as plt

s=4
def plot_func(neb_obj: NEB):

    # en_func = neb_obj.initial_chain[0].en_func
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


def en_func(inp):
    
    x, y = inp
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


nimages = 6
end_point = (2.129, 2.224)
start_point = (-3.77928812, -3.28320392)
coords = np.linspace(start_point, end_point, nimages)

ks = 6
chain = Chain.from_list_of_coords(k=ks, list_of_coords=coords, node_class=Node2D, delta_k=0, step_size=.01)
n = NEB(initial_chain=chain, max_steps=500, grad_thre_per_atom=1, climb=False, vv_force_thre=0.0)#,en_thre=1e-2, mag_grad_thre=1000 ,redistribute=False,

n.optimize_chain()

s=4
def plot_func(neb_obj: NEB):

    # en_func = neb_obj.initial_chain[0].en_func
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


plot_func(n)

plot_2D(n)

node_for_dimer = n.optimized[4]

# # Create random unit vector and make dimer

vec = node_for_dimer.pair_of_coordinates

dim = 2 # dimension of vector
random_vec = np.random.rand(2)
random_unit_vec = random_vec / np.linalg.norm(random_vec)

# +
delta_r = 0.3 # distance between dimer images
r1 = vec - delta_r*random_unit_vec
r2 = vec + delta_r*random_unit_vec

dimer = (r1, r2)


# -

def get_dimer_energy(dimer):
    r1,r2 = dimer
    return en_func(r1) + en_func(r2)
get_dimer_energy((r1,r2))


def grad_func(inp):
        x, y = inp
        dx = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
        dy = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
        return np.array([dx, dy])


def force_func(r_vec):
    return - grad_func(r_vec)


# +
def force_perp(r_vec, unit_dir):
    force_r_vec = force_func(r_vec)
    return force_r_vec - np.dot(force_r_vec, unit_dir)*unit_dir

print(force_perp(r1, random_unit_vec))
print(force_perp(r2, random_unit_vec))


# -

def plot_grad_arrows(dimer, grads):
    r1,r2 = dimer
    x1,y1 = r1
    x2,y2 = r2
    
    
    dx1,dy1 = grads[0]
    dx2,dy2 = grads[1]
    
    plt.arrow(x=x1,y=y1, dx=dx1, dy=dy1, head_width=0.2)
    plt.arrow(x=x2,y=y2, dx=dx2, dy=dy2, head_width=0.2)


def get_unit_dir(dimer):
    r1, r2 = dimer
    return (r2 - r1)/np.linalg.norm(r2-r1)


plot_grad_arrows((r1,r2), [force_perp(r1, random_unit_vec), force_perp(r2, random_unit_vec)])


# +
def get_dimer_force_perp(dimer):
    r1, r2 = dimer
    unit_dir= get_unit_dir(dimer)
    
    f_r1 = force_perp(r1, unit_dir=unit_dir)
    f_r2 = force_perp(r2, unit_dir=unit_dir)
    return f_r2 - f_r1

get_dimer_force_perp((r1,r2))

# -

dimer_force_perp = get_dimer_force_perp((r1,r2))
theta_rot = dimer_force_perp / np.linalg.norm(dimer_force_perp)

unit_dir = get_unit_dir((r1,r2))
def update_img(r_vec, d_theta, unit_dir, theta_rot, delta_r):
    return r_vec + (unit_dir*np.cos(d_theta) + theta_rot*np.sin(d_theta))*delta_r
update_img(r_vec=r2, d_theta=.1, unit_dir=unit_dir, theta_rot=theta_rot, delta_r=delta_r)


def update_dimer(dimer, delta_r, d_theta=0.5):
    r1, r2a = dimer
    unit_dir = get_unit_dir(dimer)
    midpoint = r2 - unit_dir*delta_r
    
    dimer_force_perp = get_dimer_force_perp(dimer)
    theta_rot = dimer_force_perp / np.linalg.norm(dimer_force_perp)
    r2_prime = update_img(r_vec=midpoint, d_theta=d_theta, unit_dir=unit_dir, theta_rot=theta_rot, delta_r=delta_r)
    
    new_dir = (r2_prime - midpoint)
    new_unit_dir = new_dir / np.linalg.norm(new_dir)
    
    r1_prime = r2_prime - 2*delta_r*new_unit_dir
    
    return (r1_prime, r2_prime)
update_dimer((r1,r2), delta_r)


# # Plotting

def plot_dimer(dimer,c='gray'):
    r1,r2 = dimer
    unit_dir = get_unit_dir(dimer)
    
    x1,y1 = r1
    x2,y2 = r2
    x_mid, y_mid = r1 + delta_r*unit_dir
    
    dimer_force_perp = get_dimer_force_perp(dimer)
    theta_rot = dimer_force_perp / np.linalg.norm(dimer_force_perp)
    
    d_theta=1
    # print(f"{r2=}")
    r2_prime = update_img(r_vec=(x_mid, y_mid), d_theta=d_theta, unit_dir=unit_dir, theta_rot=theta_rot, delta_r=delta_r)

    # print(f"{r2_prime=}")
    dir_r2 = r2_prime - r2
    # print(f"{dir_r2=}")
    plt.plot([x1,x_mid,x2],[y1,y_mid,y2], 'o--', color=c)
    plt.arrow(x=r2[0], y=r2[1], dx=dir_r2[0], dy=dir_r2[1])
    


# +
def rotate_dimer(dimer):
    dt=0.001

    dimer_0 = dimer
    en_0 = get_dimer_energy(dimer_0)
    dimer_1 = update_dimer(dimer, delta_r, d_theta=dt)
    en_1 = get_dimer_energy(dimer_1)
    n_counts = 0
    while np.abs(en_1 - en_0) > 1e-7 and n_counts < 10000:

        dimer_0 = dimer_1
        en_0 = get_dimer_energy(dimer_0)
        dimer_1 = update_dimer(dimer_0, delta_r, d_theta=dt)
        en_1 = get_dimer_energy(dimer_1)
        n_counts+=1
        
    if np.abs(en_1 - en_0) <= 1e-7: print(f"Rotation converged in {n_counts} steps!")
    else: print(f"Rotation did not converge. Final |∆E|: {np.abs(en_1 - en_0)}")
    return dimer_1

rotate_dimer(dimer)


# +
def get_climb_force(dimer):
    r1, r2 = dimer
    unit_path = get_unit_dir(dimer)
    
    
    f_r1 = force_func(r1)
    f_r2 = force_func(r2)
    F_R = f_r1 + f_r2
    
    
    f_parallel_r1 = np.dot(f_r1, unit_path)*unit_path
    f_parallel_r2 = np.dot(f_r2, unit_path)*unit_path
    F_Par = f_parallel_r1 + f_parallel_r2
    

    return F_R - 2*F_Par
    
    
get_climb_force(rotated_dimer)
# -

rotated_dimer = rotate_dimer(dimer)
unit_path = get_unit_dir(rotated_dimer)
plot_grad_arrows(rotated_dimer, [f_climb(r1, unit_path),f_climb(r2, unit_path)])


# +
def translate_dimer(dimer, step=0.001):
    dimer_0 = dimer
    en_0 = get_dimer_energy(dimer_0)
    
    
    r1,r2 = dimer_0
    force = get_climb_force(dimer_0)
    r2_prime = r2 + step*force
    r1_prime = r1 + step*force
    
    
    dimer_1 = (r1_prime, r2_prime)
    en_1 = get_dimer_energy(dimer_1)
    n_counts = 0
    while np.abs(en_1 - en_0) > 1e-7 and n_counts < 10000:

        r1,r2 = dimer_1
        dimer_0 = dimer_1
        
        
        en_0 = get_dimer_energy(dimer_0)
        force = get_climb_force(dimer_0)
        r2_prime = r2 + step*force
        r1_prime = r1 + step*force


        dimer_1 = (r1_prime, r2_prime)
        en_1 = get_dimer_energy(dimer_1)
        n_counts+=1
    
    
    
    if np.abs(en_1 - en_0) <= 1e-7: print(f"Translation converged in {n_counts} steps!")
    return (r1_prime, r2_prime)

translate_dimer(rotated_dimer)

# +
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

# dimer0
dimer = (r1,r2)
print(f"init_dimer_en: {get_dimer_energy(dimer)}")
plot_dimer(dimer,c='yellow')

rotated_dimer = rotate_dimer(dimer)
print(f"rotated_dimer_en: {get_dimer_energy(rotated_dimer)}")
plot_dimer(rotated_dimer)


trans_dimer = translate_dimer(rotated_dimer)
print(f"translated_dimer_en: {get_dimer_energy(trans_dimer)}")
plot_dimer(trans_dimer, c='green')
# # dimer1
# dimer1 = update_dimer(dimer, delta_r, d_theta=0.1)
# print(f"dimer_1_en: {get_dimer_energy(dimer1)}")
# plot_dimer(dimer1)


# # dimer2
# dimer2 = update_dimer(dimer1, delta_r, d_theta=0.1)
# print(f"dimer_2_en: {get_dimer_energy(dimer2)}")
# plot_dimer(dimer2, c='red')

# # points = [r1,r2,vec]
# # for x,y in points:
# #     plt.scatter(x,y)
# -

final_unit_dir = get_unit_dir(trans_dimer)
trans_dimer[0] + delta_r*final_unit_dir


