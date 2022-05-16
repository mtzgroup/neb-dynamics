import numpy as np
import matplotlib.pyplot as plt


# # Define potentials


def coulomb(r, d, r0, alpha):
    return (d / 2) * ((3 / 2) * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))


coulomb(d=4.746, r=1, r0=0.742, alpha=1.942)


def exchange(r, d, r0, alpha):
    return (d / 4) * (np.exp(-2 * alpha * (r - r0)) - 6 * np.exp(-alpha * (r - r0)))


exchange(d=4.746, r=1,  r0=0.742, alpha=1.942)


# +
# plt.plot([coulomb(d=4.746, r=x, r0=0.742, alpha=1.942) for x in list(range(10))])
# -


def potential(r_ab, r_bc, a=0.05, b=0.30, c=0.05, d_ab=4.746, d_bc=4.746, d_ac=3.445, r0=0.742, alpha=1.942):
    Q_AB = coulomb(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
    Q_BC = coulomb(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
    Q_AC = coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

    J_AB = exchange(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
    J_BC = exchange(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
    J_AC = exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

    result_Qs = (Q_AB / (1 + a)) + (Q_BC / (1 + b)) + (Q_AC / (1 + c))
    result_Js_1 = ((J_AB**2) / ((1 + a) ** 2)) + ((J_BC**2) / ((1 + b) ** 2)) + ((J_AC**2) / ((1 + c) ** 2))
    result_Js_2 = ((J_AB * J_BC) / ((1 + a) * (1 + b))) + ((J_AC * J_BC) / ((1 + c) * (1 + b))) + ((J_AB * J_AC) / ((1 + a) * (1 + c)))
    result_Js = result_Js_1 - result_Js_2

    result = result_Qs - (result_Js) ** (1 / 2)
    return result


potential(1, 1)


def toy_potential(x, y, height=1):
    result = -(x**2) + -(y**2) + 10
    return np.where(result < 0, 0, result)


def toy_grad(x, y):
    if toy_potential(x, y) == 0:
        return [0, 0]
    dx = -2 * x
    dy = -2 * y
    return np.array([dx, dy])


def toy_potential_2(inp):
    x,y = inp
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


# + tags=[]
def toy_grad_2(inp):
    x, y = inp
    dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    return np.array([dx, dy])


# -

def toy_potential_3(x, y):
    return -np.absolute(np.sin(x) * np.cos(y) * np.exp(np.absolute(1 - (np.sqrt(x**2 + y**2) / np.pi))))


def toy_grad_3(x, y):
    alpha = np.cos(y) * np.sin(x)
    beta = 1 - (np.sqrt(x**2 + y**2) / np.sqrt(np.pi))

    dx = -(
        (alpha * np.exp(np.abs(beta)) * (np.cos(x) * np.cos(y) * np.exp(np.abs(beta)) - (x * beta * alpha * np.exp(np.abs(beta))) / (np.sqrt(np.pi) * np.sqrt(x**2 + y**2) * np.abs(beta))))
        / (alpha * np.exp(np.abs(beta)))
    )

    dy = -(
        (alpha * np.exp(np.abs(beta)) * (-1 * np.sin(x) * np.sin(y) * np.exp(np.abs(beta)) - (x * beta * alpha * np.exp(np.abs(beta))) / (np.sqrt(np.pi) * np.sqrt(x**2 + y**2) * np.abs(beta))))
        / (alpha * np.exp(np.abs(beta)))
    )

    return np.array([dx, dy])


# # Get minima

def toy_potential_2_foo(inp):
    x,y = inp
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


from scipy.optimize import fmin
fmin(toy_potential_2_foo, np.array([4, 4]))


# # NEB code

def spring_grad_neb(view, k=0.1, ideal_distance=0.5, en_func=toy_potential, grad_func=toy_grad):

    # pe_grad = 0
    neighs = view[[0,2]]
    # neighs = [view[2]]
    x,y = view[1]

    
    vec_tan_path = neighs[1] - neighs[0]
    # vec_tan_path = view[2] - view[1]
    unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)
    

    pe_grad = grad_func(view[1])
    pe_grad_nudged_const = np.dot(pe_grad, unit_tan_path)
    pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path

    grads_neighs = []
    force_springs = []
    
    for neigh in neighs:
        neigh_x, neigh_y = neigh
        dist_x = np.abs(neigh_x - x)
        dist_y = np.abs(neigh_y - y)

        force_x = -k * (dist_x - ideal_distance)
        force_y = -k * (dist_y - ideal_distance)
        # print(f"\t{force_x=} {force_y=}")

        if neigh_x > x:
            force_x *= -1

        if neigh_y > y:
            force_y *= -1

        force_spring = np.array([force_x, force_y])
        force_springs.append(force_spring)
        
        
        force_spring_nudged_const = np.dot(force_spring, unit_tan_path)
        force_spring_nudged = force_spring - force_spring_nudged_const * unit_tan_path

        grads_neighs.append(force_spring_nudged)

    # print(f"\t{grads_neighs=}")

    tot_grads_neighs = np.sum(grads_neighs, axis=0)
    
    
    
    
    ### ANTI-KINK FORCE
    force_springs = np.sum(force_springs, axis=0)
    
    vec_2_to_1 = view[2] - view[1]
    vec_1_to_0 = view[1] - view[0]
    cos_phi = np.dot(vec_2_to_1,vec_1_to_0)/(np.linalg.norm(vec_2_to_1)*np.linalg.norm(vec_1_to_0))
    # print(f"{cos_phi=}")
    f_phi = 0.5*(1 + np.cos(np.pi*cos_phi))
    
    proj_force_springs = force_springs - np.dot(force_springs, unit_tan_path)*unit_tan_path
    

    # print(f"{tot_grads_neighs}")
    # return pe_grad_nudged - tot_grads_neighs 
    return (pe_grad_nudged - tot_grads_neighs)  + f_phi*(proj_force_springs)


def spring_grad(view, k=0.1, ideal_distance=0.5, en_func=toy_potential, grad_func=toy_grad):


    # pe_grad = 0
    neighs = view[[0,2]]
    x,y = view[1]



    pe_grad = grad_func(x, y)
    

    grads_neighs = []
    for neigh in neighs:
        neigh_x, neigh_y = neigh
        dist_x = np.abs(neigh_x - x)
        dist_y = np.abs(neigh_y - y)

        force_x = -k * (dist_x - ideal_distance)
        force_y = -k * (dist_y - ideal_distance)
        # print(f"\t{force_x=} {force_y=}")

        if neigh_x > x:
            force_x *= -1

        if neigh_y > y:
            force_y *= -1

        force_spring = np.array([force_x, force_y])
        
        grads_neighs.append(force_spring)

    # print(f"\t{grads_neighs=}")

    tot_grads_neighs = np.sum(grads_neighs, axis=0)

    # print(f"{tot_grads_neighs}")
    return tot_grads_neighs - pe_grad


def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def update_points(chain, grad, dr):
    new_chain = []
    for point in chain:
        p_current = point

        p_x, p_y = p_current
        grad_x, grad_y = grad(p_x, p_y)
        p_new = (p_x + (-grad_x * dr), p_y + (-grad_y * dr))

        p_current = p_new
        new_chain.append(p_current)

    return new_chain


def update_points_spring(chain, dr, en_func, grad_func, k=1, ideal_dist=0.5):
    chain_copy = np.zeros_like(chain)
    chain_copy[0] = chain[0]
    chain_copy[-1] = chain[-1]
    
    for i in range(1, len(chain) - 1):
        view = chain[i-1:i+2]
        p_x, p_y = chain[i]
        
        grad_x, grad_y = spring_grad(view, k=k, ideal_distance=ideal_dist, grad_func=grad_func, en_func=en_func)
        
        # print(f"{grad_x=} {grad_y=}")
        p_new = np.array((p_x + (-1*grad_x * dr), p_y + (-1*grad_y * dr)))
        
        chain_copy[i] = p_new
    

    return chain_copy


# + tags=[]
def ArmijoLineSearch(f, xk, pk, gfk, phi0, alpha0, rho=0.5, c1=1e-4):
    """Minimize over alpha, the function ``f(xₖ + αpₖ)``.
    α > 0 is assumed to be a descent direction.
    
    Parameters
    --------------------
    f : callable
        Function to be minimized.
    xk : array
        Current point.
    gfk : array
        Gradient of `f` at point `xk`.
    phi0 : float
        Value of `f` at point `xk`.
    alpha0 : scalar
        Value of `alpha` at the start of the optimization.
    rho : float, optional
        Value of alpha shrinkage factor.
    c1 : float, optional
        Value to control stopping criterion.
    
    Returns
    --------------------
    alpha : scalar
        Value of `alpha` at the end of the optimization.
    phi : float
        Value of `f` at the new point `x_{k+1}`.
    """
    derphi0 = np.dot(gfk, pk)
    phi_a0 = f(xk + alpha0*pk)
#     print(f"{phi_a0=}")
#     print(f"\t{phi0 + c1*alpha0*derphi0}")
    
    while not phi_a0 <= phi0 + c1*alpha0*derphi0:
        alpha0 = alpha0 * rho
        phi_a0 = f(xk + alpha0*pk)
    
    return alpha0, phi_a0


# -

def update_points_spring_neb(chain, dr, en_func, grad_func, k=1, ideal_dist=0.5):
    
    chain_copy = np.zeros_like(chain)
    chain_copy[0] = chain[0]
    chain_copy[-1] = chain[-1]
    
    for i in range(1, len(chain) - 1):
        view = chain[i-1:i+2]
        p_x, p_y = chain[i]
        
        grad_x, grad_y = spring_grad_neb(view, k=k, ideal_distance=ideal_dist, grad_func=grad_func, en_func=en_func)
        
        len_grad = np.linalg.norm(np.array([grad_x, grad_y]))
        
        # print(f"{grad_x=}")
        grad_x_scaled = grad_x/len_grad
        # print(f"{grad_x_scaled=}")

        # print(f"{grad_y=}")
        grad_y_scaled = grad_y/len_grad
        # print(f"{grad_y_scaled=}")
        
        # print(f"\t{np.linalg.norm([grad_x_scaled,grad_y_scaled])}")
        
        
        dr, _ = ArmijoLineSearch(f=en_func, xk=chain[i], gfk=np.array([grad_x, grad_y]), 
                                 phi0=en_func(chain[i]), alpha0=.1, pk=-1*np.array([grad_x, grad_y]))
        
        # print(f"{dr=}")
        # print(f"{grad_x=} {grad_y=}")
        p_new = chain[i] - dr*np.array([grad_x, grad_y])
        # p_new = np.array((p_x + grad_x_scaled, p_y + grad_y_scaled ))
        
        chain_copy[i] = p_new
    

    return chain_copy


def _check_en_converged(chain_prev, chain_new, en_func, en_thre):
    for i in range(1, len(chain_prev) - 1):
        node_prev = chain_prev[i]
        node_new = chain_new[i]

        delta_e = np.abs(en_func(node_new) - en_func(node_prev))
        if delta_e > en_thre:
            # print(f"{delta_e} > {en_thre}")
            return False
    return True


def _check_grad_converged(chain_prev, chain_new, grad_func, grad_thre):
    for i in range(1, len(chain_prev) - 1):  # don't consider the endpoints, they're fixed
        # print(f"---->{i}")
        node_prev = chain_prev[i]
        node_new = chain_new[i]

        delta_grad_x, delta_grad_y = np.abs(grad_func(node_new) - grad_func(node_prev))
        # print(f"{delta_grad_x=} {delta_grad_y=}")

        if (delta_grad_x > grad_thre) or (delta_grad_y > grad_thre):
            # print(f"\t{delta_grad_x} > {grad_thre}")
            # print(f"\t{delta_grad_y} > {grad_thre}")

            return False
    return True


def optimize_chain(chain, en_func, grad_func, update_func, k=0.1, grad_thre=0.01, en_thre=0.01, max_steps=1000):
    chain_previous = chain.copy()
    
    dist_id = np.linalg.norm(np.array(chain[-1]) - np.array(chain[0]))/len(chain)
    # dist_id=0.5

    # print(f"{chain_previous=}\n")
    chain_traj = []
    chain_optimized = False

    nsteps = 0

    while nsteps < max_steps and not chain_optimized:
        new_chain = update_func(chain=chain_previous, dr=dr, k=k, ideal_dist=dist_id, en_func=en_func, grad_func=grad_func)
        en_converged = _check_en_converged(chain_prev=chain_previous, chain_new=new_chain, en_func=en_func, en_thre=en_thre)
        grad_converged = _check_grad_converged(chain_prev=chain_previous, chain_new=new_chain, grad_func=grad_func, grad_thre=grad_thre)

        if en_converged and grad_converged:
            chain_optimized = True

        chain_traj.append(new_chain)
            
        chain_previous = new_chain.copy()
        nsteps += 1
        
        

    if chain_optimized:
        print("Chain converged!")
    else:
        print("Chain did not converge...")

    # print(f"final chain (PEB): {new_chain}")
    # for i in range(len(new_chain) - 1):
    #     print(f"dist {i}-{i+1}: {dist(new_chain[i], new_chain[i+1])}")
    return new_chain, chain_traj

# + tags=[]
np.random.seed(1)
fs = 14
# vars for sim
nsteps = 1000
dr = .01
k = 10

nimages = 15

end_point = (3.00002182, 1.99995542)
start_point = (-3.77928812, -3.28320392)
dist_id = np.linalg.norm(np.array(end_point) - np.array(start_point))/nimages

en_func = toy_potential_2
grad_func = toy_grad_2


# set up plot for potential
min_val = -4
max_val = 4
num = 10
fig = 10
f, ax = plt.subplots(figsize=(1.18 * fig, fig))
x = np.linspace(start=min_val, stop=max_val, num=num)
y = x.reshape(-1, 1)


h = en_func([x,y])
cs = plt.contourf(x, x, h)
cbar = f.colorbar(cs)


# # set up points
# chain = np.sort([np.random.uniform(-1, 1, size=2) for n in range(nimages)])
# chain = np.linspace((min_val, max_val), (max_val, min_val), nimages)
# chain = np.linspace((min_val, min_val), (max_val, max_val), nimages)
chain = np.linspace(start_point, end_point , nimages)
# chain = np.linspace((-5, -3), (5, 3), nimages)
# chain = [(-2,-.1),(0,2),(2,.1)]
# print(chain)


plt.plot([(point[0]) for point in chain], [(point[1]) for point in chain], "^--", c="white", label="original")


# opt_chain_peb, _ = optimize_chain(chain=chain, en_func=en_func, grad_func=grad_func, en_thre=0.01, grad_thre=0.01, update_func=update_points_spring, max_steps=nsteps,k=k)
# print(f"{opt_chain_peb=}")
# points_x = [point[0] for point in opt_chain_peb]
# points_y = [point[1] for point in opt_chain_peb]
# plt.plot(points_x, points_y, "*--", c="white", label="PEB")


opt_chain_neb, _ = optimize_chain(chain=chain, en_func=en_func, grad_func=grad_func, en_thre=0.01, grad_thre=0.01, update_func=update_points_spring_neb, max_steps=nsteps,k=k)
print(f"{opt_chain_neb=}")
points_x = [point[0] for point in opt_chain_neb]
points_y = [point[1] for point in opt_chain_neb]
plt.plot(points_x, points_y, "o--", c="white", label="NEB")


plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()


# +
# # set up plot for potential

# num = 10
# fig = 10
# f, ax = plt.subplots(figsize=(1.18 * fig, fig))
# x = np.linspace(start=min_val, stop=max_val, num=num)
# y = x.reshape(-1, 1)


# # h = en_func(x, y)
# h = en_func(x, y)
# cs = plt.contourf(x, x, h)
# cbar = f.colorbar(cs)
# points_x = [point[0] for point in opt_chain_neb]
# points_y = [point[1] for point in opt_chain_neb]
# plt.plot(points_x, points_y, "o--", c="white", label="neb")


# points_x = [point[0] for point in opt_chain_peb]
# points_y = [point[1] for point in opt_chain_peb]
# plt.plot(points_x, points_y, "x--", c="white", label="peb")

# points_x = [point[0] for point in chain]
# points_y = [point[1] for point in chain]
# plt.plot(points_x, points_y, "*--", c="white", label="original")

# plt.legend()
# print(opt_chain_neb)
# plt.show()
# +
# ens = [en_func(x[0], x[1]) for x in opt_chain_neb]
# ens_peb = [en_func(x[0], x[1]) for x in opt_chain_peb]
# ens_orig = [en_func(x[0], x[1]) for x in chain]
# plt.plot(ens, 'o',label="neb")
# plt.plot(ens_peb, 'o--',label="peb")
# plt.plot(ens_orig, label="orig")
# plt.legend()
# -

# # Anim

chain = np.linspace(start_point, end_point , nimages)

chain

# + tags=[]
final_chain, all_chains = optimize_chain(chain=chain, en_func=en_func, grad_func=grad_func, en_thre=0.01, grad_thre=0.01, update_func=update_points_spring_neb, max_steps=nsteps,k=k)
# final_chain_peb, all_chains_peb = optimize_chain(chain=chain, en_func=en_func, grad_func=grad_func, en_thre=0.01, grad_thre=0.01, update_func=update_points_spring, max_steps=nsteps,k=k)

# +
# %matplotlib widget

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

figsize=5

fig, ax = plt.subplots(figsize=(1.18 * figsize, figsize))
x = np.linspace(start=-4, stop=4, num=num)
y = x.reshape(-1, 1)


# h = en_func(x, y)
h = en_func([x, y])
cs = plt.contourf(x, x, h)
cbar = f.colorbar(cs, ax=ax)
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,

def animate(chain):
    
    x = chain[:, 0]
    y = chain[:, 1]
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig=fig, func=animate, frames=all_chains, blit=True, repeat_delay=1000, interval=500)
# anim.save('sine_wave.gif', writer='imagemagick')

# +
c = all_chains[3]
x_foo = c[:,0]
y_foo = c[:,1]

plt.plot(x_foo, y_foo)
# -


