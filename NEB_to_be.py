# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt


# # Define potentials


def coulomb(r, d, r0, alpha):
    return (d / 2) * ((3 / 2) * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))


coulomb(d=4.746, r=1, r0=0.742, alpha=1.942)


def exchange(r, d, r0, alpha):
    return (d / 4) * (np.exp(-2 * alpha * (r - r0)) - 6 * np.exp(-alpha * (r - r0)))


exchange(d=4.746, r=1, r0=0.742, alpha=1.942)


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
    return dx, dy


def toy_potential_2(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def toy_grad_2(x, y):
    dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    return dx, dy


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

    return dx, dy


def spring_grad_neb(x, y, neighs, k=0.1, ideal_distance=0.5, en_func=toy_potential, grad_func=toy_grad):

    # pe_grad = 0

    if len(neighs) == 1:
        vec_tan_path = neighs[0] - np.array([x, y])

    elif len(neighs) == 2:
        vec_tan_path = np.array(neighs[1]) - np.array(neighs[0])
    else:
        raise ValueError("Wtf are you doing.")

    unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)
    # print(f"{unit_tan_path=}")

    pe_grad = grad_func(x, y)
    pe_grad_nudged_const = np.dot(pe_grad, unit_tan_path)
    pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tan_path

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
        force_spring_nudged_const = np.dot(force_spring, unit_tan_path**2)
        force_spring_nudged = force_spring - force_spring_nudged_const * unit_tan_path

        grads_neighs.append(force_spring_nudged)

    # print(f"\t{grads_neighs=}")

    tot_grads_neighs = np.sum(grads_neighs, axis=0)

    # print(f"{tot_grads_neighs}")
    return tot_grads_neighs - pe_grad_nudged


def spring_grad(x, y, neighs, k=0.1, ideal_distance=0.5, en_func=toy_potential, grad_func=toy_grad):

    # pe_grad = 0

    if len(neighs) == 1:
        vec_tan_path = neighs[0] - np.array([x, y])

    elif len(neighs) == 2:
        vec_tan_path = np.array(neighs[1]) - np.array(neighs[0])
    else:
        raise ValueError("Wtf are you doing.")

    unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)
    # print(f"{unit_tan_path=}")

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
    new_chain = [chain[0]]

    for i in range(1, len(chain) - 1):

        point = chain[i]
        p_current = point

        p_x, p_y = p_current
        if i == 0:
            neighs = [chain[1]]
        elif i == len(chain) - 1:
            neighs = [chain[-2]]

        else:
            neighs = [chain[i - 1], chain[i + 1]]
        grad_x, grad_y = spring_grad(p_x, p_y, neighs=neighs, k=k, ideal_distance=ideal_dist, grad_func=grad_func, en_func=en_func)
        # print(f"{grad_x=} {grad_y=}")
        p_new = (p_x + (grad_x * dr), p_y + (grad_y * dr))

        p_current = p_new
        new_chain.append(p_current)
    new_chain.append(chain[-1])

    return new_chain


def update_points_spring_neb(chain, dr, en_func, grad_func, k=1, ideal_dist=0.5):
    new_chain = [chain[0]]

    for i in range(1, len(chain) - 1):

        point = chain[i]
        p_current = point

        p_x, p_y = p_current
        if i == 0:
            neighs = [chain[1]]
        elif i == len(chain) - 1:
            neighs = [chain[-2]]

        else:
            neighs = [chain[i - 1], chain[i + 1]]
        grad_x, grad_y = spring_grad_neb(p_x, p_y, neighs=neighs, k=k, ideal_distance=ideal_dist, grad_func=grad_func, en_func=en_func)
        # print(f"{grad_x=} {grad_y=}")
        p_new = (p_x + (grad_x * dr), p_y + (grad_y * dr))

        p_current = p_new
        new_chain.append(p_current)
    new_chain.append(chain[-1])

    return new_chain


def _check_en_converged(chain_prev, chain_new, en_func, en_thre):
    for i in range(1, len(chain_prev) - 1):
        node_prev = chain_prev[i]
        node_new = chain_new[i]

        delta_e = np.abs(en_func(node_new[0], node_new[1]) - en_func(node_prev[0], node_prev[1]))
        if delta_e > en_thre:
            # print(f"{delta_e} > {en_thre}")
            return False
    return True


def _check_grad_converged(chain_prev, chain_new, grad_func, grad_thre):
    for i in range(1, len(chain_prev) - 1):  # don't consider the endpoints, they're fixed
        # print(f"---->{i}")
        node_prev = chain_prev[i]
        node_new = chain_new[i]

        delta_grad_x, delta_grad_y = np.abs(grad_func(node_new[0], node_new[1]) - en_func(node_prev[0], node_prev[1]))
        # print(f"{delta_grad_x=} {delta_grad_y=}")

        if (delta_grad_x > grad_thre) or (delta_grad_y > grad_thre):
            # print(f"\t{delta_grad_x} > {grad_thre}")
            # print(f"\t{delta_grad_y} > {grad_thre}")

            return False
    return True


def optimize_chain(chain, en_func, grad_func, update_func, k=0.1, dist_id=0.1, grad_thre=0.01, en_thre=0.01, max_steps=1000):
    chain_previous = chain.copy()

    # print(f"{chain_previous=}\n")

    chain_optimized = False

    nsteps = 0

    while nsteps < max_steps and not chain_optimized:
        new_chain = update_func(chain=chain_previous, dr=dr, k=k, ideal_dist=dist_id, en_func=en_func, grad_func=grad_func)
        en_converged = _check_en_converged(chain_prev=chain_previous, chain_new=new_chain, en_func=en_func, en_thre=en_thre)
        grad_converged = _check_grad_converged(chain_prev=chain_previous, chain_new=new_chain, grad_func=grad_func, grad_thre=grad_thre)

        if en_converged and grad_converged:
            chain_optimized = True

        chain_previous = new_chain.copy()
        nsteps += 1

    if chain_optimized:
        print("Chain converged!")
    else:
        print("Chain did not converge...")

    print(f"final chain (PEB): {new_chain}")
    for i in range(len(new_chain) - 1):
        print(f"dist {i}-{i+1}: {dist(new_chain[i], new_chain[i+1])}")
    return new_chain


# ### NB:
# For top left to bottom right, k=25, idealdist=10

# + tags=[]
np.random.seed(1)
fs = 14
# vars for sim
nsteps = 100
dr = 0.01
k = 30
dist_id = 0.5

nimages = 21

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


h = en_func(x, y)
cs = plt.contourf(x, x, h)
cbar = f.colorbar(cs)


# # set up points
# chain = np.sort([np.random.uniform(-1, 1, size=2) for n in range(nimages)])
# chain = np.linspace((min_val, max_val), (max_val, min_val), nimages)
# chain = np.linspace((min_val, min_val), (max_val, max_val), nimages)
chain = np.linspace((-3.7933036307483574, -3.103697226077475), (3, 2), nimages)
# chain = np.linspace((-1.3, 0), (1.5, -3), nimages)
# chain = [(-2,-.1),(0,2),(2,.1)]
print(chain)


plt.plot([(point[0]) for point in chain], [(point[1]) for point in chain], "^--", c="white", label="original")


opt_chain_peb = optimize_chain(chain=chain, en_func=en_func, grad_func=grad_func, en_thre=0.01, grad_thre=0.01, update_func=update_points_spring, max_steps=nsteps)
print(f"{opt_chain_peb=}")
points_x = [point[0] for point in opt_chain_peb]
points_y = [point[1] for point in opt_chain_peb]
plt.plot(points_x, points_y, "*--", c="white", label="PEB")


opt_chain_neb = optimize_chain(chain=chain, en_func=en_func, grad_func=grad_func, en_thre=0.01, grad_thre=0.01, update_func=update_points_spring_neb, max_steps=nsteps)
print(f"{opt_chain_neb=}")
points_x = [point[0] for point in opt_chain_neb]
points_y = [point[1] for point in opt_chain_neb]
plt.plot(points_x, points_y, "o--", c="white", label="NEB")


plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()


# +
# set up plot for potential
min_val = -4
max_val = 4
num = 10
fig = 10
f, ax = plt.subplots(figsize=(1.18 * fig, fig))
x = np.linspace(start=min_val, stop=max_val, num=num)
y = x.reshape(-1, 1)


# h = en_func(x, y)
h = en_func(x, y)
cs = plt.contourf(x, x, h)
cbar = f.colorbar(cs)
points_x = [point[0] for point in opt_chain_neb]
points_y = [point[1] for point in opt_chain_neb]
plt.plot(points_x, points_y, "o--", c="white", label="neb")


points_x = [point[0] for point in opt_chain_peb]
points_y = [point[1] for point in opt_chain_peb]
plt.plot(points_x, points_y, "x--", c="white", label="peb")

points_x = [point[0] for point in chain]
points_y = [point[1] for point in chain]
plt.plot(points_x, points_y, "*--", c="white", label="original")

plt.legend()
print(opt_chain_neb)
plt.show()
# -
ens = [en_func(x[0], x[1]) for x in opt_chain_neb]
ens_peb = [en_func(x[0], x[1]) for x in opt_chain_peb]
ens_orig = [en_func(x[0], x[1]) for x in chain]
plt.plot(ens, label="neb")
plt.plot(ens_peb, label="peb")
plt.plot(ens_orig, label="orig")
plt.legend()
