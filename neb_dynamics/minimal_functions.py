# +
import matplotlib.pyplot as plt
import numpy as np


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


def toy_potential_2(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def toy_grad_2(x, y):
    dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    return dx, dy


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


def spring_grad_neb0(array_thing, k=0.1, ideal_distance=0.5, en_func=toy_potential_2, grad_func=toy_grad_2):
    x, y = array_thing[1]
    neighs = array_thing[0], array_thing[2]
    print(f"fuck {x=} {y=} {neighs=}")
    vec_tan_path = np.array(neighs[1]) - np.array(neighs[0])

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


def spring_grad_neb(array_thing, k=0.1, ideal_distance=0.5, en_func=toy_potential_2, grad_func=toy_grad_2):
    x, y = array_thing[1]
    neighs = array_thing[0], array_thing[2]
    print(f"fuck {x=} {y=} {neighs=}")

    vec_tan_path = np.array(neighs[1]) - np.array(neighs[0])
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


def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# +
def optimize_chain(chain, en_func, grad_func, update_func, k=0.1, dist_id=0.1, grad_thre=0.01, en_thre=0.01, max_steps=1000):
    chain_previous = chain.copy()

    # print(f"{chain_previous=}\n")

    chain_optimized = False
    chains = [chain]
    nsteps = 0

    while nsteps < max_steps and not chain_optimized:
        new_chain = update_func(chain=chain_previous, dr=dr, k=k, ideal_dist=dist_id, en_func=en_func, grad_func=grad_func)
        en_converged = _check_en_converged(chain_prev=chain_previous, chain_new=new_chain, en_func=en_func, en_thre=en_thre)
        grad_converged = _check_grad_converged(chain_prev=chain_previous, chain_new=new_chain, grad_func=grad_func, grad_thre=grad_thre)

        if en_converged and grad_converged:
            chain_optimized = True

        chain_previous = new_chain.copy()
        chains.append(chain_previous)
        nsteps += 1

    #     if chain_optimized:
    #         print("Chain converged!")
    #     else:
    #         print("Chain did not converge...")

    #     print(f"final chain (PEB): {new_chain}")
    #     for i in range(len(new_chain) - 1):
    #         print(f"dist {i}-{i+1}: {dist(new_chain[i], new_chain[i+1])}")
    return new_chain, chains


# +
what_we_want = np.array(
    [
        [-3.79330363, -3.10369723],
        [-3.4229274984442957, -2.889390340896632],
        [-3.1332456396951365, -2.567674909080191],
        [-2.8298142767474572, -2.264260828745875],
        [-2.514111420990047, -1.9771807848887033],
        [-2.1876150838116954, -1.7044674625036926],
        [-1.8518032766011916, -1.4441535465858613],
        [-1.508154010747324, -1.194271722130227],
        [-1.1581452976388817, -0.9528546741318074],
        [-0.8032551486646543, -0.7179350875856201],
        [-0.44496157521343094, -0.48754564748668333],
        [-0.08474258867400043, -0.25971903883001485],
        [0.2759237995648485, -0.03248794661063219],
        [0.6355595781143266, 0.19611494417644687],
        [0.9926867355856437, 0.4280569485362039],
        [1.3458272605900123, 0.6653053814736222],
        [1.6935031417386428, 0.9098275579936833],
        [2.034236367642746, 1.1635907931013691],
        [2.366548926913533, 1.4285624018016625],
        [2.6889628081622132, 1.7067096990995454],
        [3.0, 2.0],
    ]
)

nimages = 21

chain = np.linspace((-3.7933036307483574, -3.103697226077475), (3, 2), nimages)
dr = 0.01
k = 30
ideal_dist = 0.5
dist_id = 0.5
en_func = toy_potential_2
grad_func = toy_grad_2

for i in range(1, len(chain) - 1):
    array_thing = chain[i - 1 : i + 2]
    p_x, p_y = chain[i]
    print(f"{i=} -> {array_thing}")
    grad_x, grad_y = spring_grad_neb(array_thing, k=k, ideal_distance=ideal_dist, grad_func=grad_func, en_func=en_func)
    grad_x0, grad_y0 = spring_grad_neb0(array_thing, k=k, ideal_distance=ideal_dist, grad_func=grad_func, en_func=en_func)
    print(f"          {grad_x0=}, {grad_y0=}, {grad_x=}, {grad_y=}")
    chain[i] = (p_x + (grad_x * dr), p_y + (grad_y * dr))
print(chain)


# +
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
# print(chain)


plt.plot([(point[0]) for point in chain], [(point[1]) for point in chain], "^--", c="white", label="original")


opt_chain_neb, chains = optimize_chain(chain=chain, en_func=en_func, grad_func=grad_func, en_thre=0.01, grad_thre=0.01, update_func=update_points_spring_neb, max_steps=nsteps)
# print(f"{opt_chain_neb=}")
points_x = [point[0] for point in opt_chain_neb]
points_y = [point[1] for point in opt_chain_neb]
plt.plot(points_x, points_y, "o--", c="white", label="NEB")


plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()
# -

chains
