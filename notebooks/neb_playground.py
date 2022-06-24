# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from neb_dynamics.trajectory import Trajectory
from matplotlib.animation import FuncAnimation
from neb_dynamics.helper_functions import pairwise

# from NEB import neb
from neb_dynamics.NEB import Chain, Node3D, NEB, Node2D
# from neb_dynamics.ALS_xtb import ArmijoLineSearch
from neb_dynamics.ALS import ArmijoLineSearch
from neb_dynamics.trajectory import Trajectory
from pathlib import Path
import matplotlib.pyplot as plt

# fp = Path("../example_cases/DA_geodesic_opt.xyz")
fp = Path("../example_cases/debug_geodesic_claisen.xyz")
# fp = Path("../example_cases/neb_DA_k0.1.xyz")
traj = Trajectory.from_xyz(fp)
chain = Chain.from_traj(traj, k=.1, delta_k=0.09)

plt.plot(chain.energies, 'o-')

# # c-NEB

n = NEB(initial_chain=chain, mag_grad_thre=100, grad_thre=0.00001, en_thre=0.00001, climb=True)

# + tags=[]
n.optimize_chain()
# -

chain_ens = chain.energies
opt_ens = n.optimized.energies

# + tags=[]
chain_ref = Chain.from_traj(traj, k=1, delta_k=0)
n_ref = NEB(initial_chain=chain_ref, mag_grad_thre=10000, grad_thre=0.0001, en_thre=0.0001, climb=False)
n_ref.optimize_chain()
# -

ref_ens = n_ref.optimized.energies

# +
# n.write_to_disk(Path("variable_springs_DA_neb.xyz"))
# n.write_to_disk(Path("variable_springs_claisen_neb.xyz"))
# -

# plt.plot((chain_ens-chain_ens[0])*627.5, '-', label='GI')
plt.plot((chain_ens-chain_ens[0])*627.5, '-', label='NEB (k=0.1)')
plt.plot((opt_ens-opt_ens[0])*627.5, 'o--', label='NEB (k$_{max}$=1')
plt.plot((ref_ens-ref_ens[0])*627.5, 'o--', label='NEB (k=1)')
plt.legend()

cneb_traj = Trajectory([node.tdstructure for node in n.optimized.nodes])
cneb_traj.write_trajectory(Path("./cneb_claisen.xyz"))

# # Redistribution

# +
n = NEB(initial_chain=chain)
direction = np.array(
    [
        next_node.coords - current_node.coords
        for current_node, next_node in pairwise(chain)
    ]
)
distances = np.linalg.norm(direction, axis=1)
tot_dist = np.sum(distances)
cumsum = np.cumsum(distances)  # cumulative sum
cumsum = np.insert(cumsum, 0, 0)


cum=cumsum
distributed_chain = []
for num in np.linspace(0, tot_dist, len(chain)):

    for ii, ((cum_sum_init, node_start), (cum_sum_end, node_end)) in enumerate(pairwise(zip(cum, chain))):

            if cum_sum_init <= num < cum_sum_end:
                direction = node_end.coords - node_start.coords
                percentage = (num - cum_sum_init) / (cum_sum_end - cum_sum_init)

                new_node = node_start.copy()
                new_coords = node_start.coords + (direction * percentage)
                new_node = new_node.update_coords(new_coords)
    print(f"{new_node}")

    distributed_chain.append(new_node)

distributed_chain[0] = chain[0]
distributed_chain[-1] = chain[-1]


# -

dist_chain = Chain(distributed_chain, k=0.1)

plt.plot(chain.energies, 'o--')
plt.plot(dist_chain.energies, "o")

dist_traj = Trajectory([n.tdstructure for n in dist_chain])

dist_traj.write_trajectory(Path("./bs.xyz"))

coulomb(d=4.746, r=1, r0=0.742, alpha=1.942)


def exchange(r, d, r0, alpha):
    return (d / 4) * (np.exp(-2 * alpha * (r - r0)) - 6 * np.exp(-alpha * (r - r0)))


exchange(d=4.746, r=1, r0=0.742, alpha=1.942)


# +
# plt.plot([coulomb(d=4.746, r=x, r0=0.742, alpha=1.942) for x in list(range(10))])
# -


def potential(
    inp,
    a=0.05,
    b=0.30,
    c=0.05,
    d_ab=4.746,
    d_bc=4.746,
    d_ac=3.445,
    r0=0.742,
    alpha=1.942,
):
    r_ab, r_bc = inp

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


def dQ_dr(d, alpha, r, r0):
    return (d / 2) * ((3 / 2) * (-2 * alpha * np.exp(-2 * alpha * (r - r0))) + alpha * np.exp(-alpha * (r - r0)))


def dJ_dr(d, alpha, r, r0):
    return (d / 4) * (np.exp(-2 * alpha * (r - r0)) * (-2 * alpha) + 6 * alpha * np.exp(-alpha * (r - r0)))


def grad_x(
    inp,
    a=0.05,
    b=0.30,
    c=0.05,
    d_ab=4.746,
    d_bc=4.746,
    d_ac=3.445,
    r0=0.742,
    alpha=1.942,
):
    r_ab, r_bc = inp

    ealpha_x = np.exp(alpha * (r0 - r_ab))
    neg_ealpha_x = np.exp(alpha * (r_ab - r0))
    ealpha_y = np.exp(alpha * (r0 - r_bc))
    neg_ealpha_y = np.exp(alpha * (r_bc - r0))

    e2alpha_x = np.exp(2 * alpha * (r0 - r_ab))
    e2alpha_y = np.exp(2 * alpha * (r0 - r_bc))

    aDenom = 1 / (1 + a)
    bDenom = 1 / (1 + b)

    Qconst = coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
    Jconst = exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

    cDenom = 1 / (1 + c)

    d = d_ab

    dx = (
        0.25
        * aDenom**2
        * alpha
        * d
        * ealpha_x
        * (
            -2 * (1 + a) * (-1 + 3 * ealpha_x)
            + ((-3 + ealpha_x) * (2 * d * ealpha_x * (-6 + ealpha_x) - (1 + a) * d * ealpha_y * (-6 + ealpha_y) * bDenom - 4 * (1 + a) * Jconst * cDenom))
            / (
                np.sqrt(
                    (
                        ((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2)
                        + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2
                        - d**2 * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc)) * (-1 + 6 * neg_ealpha_x) * (-1 + 6 * neg_ealpha_y) * aDenom * bDenom
                    )
                    - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom
                    - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom)
                    + 16 * Jconst**2 * cDenom**2
                )
            )
        )
    )

    return dx


def grad(inp):
    return np.array([grad_x(inp), grad_y(inp)])


def grad_y(
    inp,
    a=0.05,
    b=0.30,
    c=0.05,
    d_ab=4.746,
    d_bc=4.746,
    d_ac=3.445,
    r0=0.742,
    alpha=1.942,
):
    r_ab, r_bc = inp

    ealpha_x = np.exp(alpha * (r0 - r_ab))
    neg_ealpha_x = np.exp(alpha * (r_ab - r0))
    ealpha_y = np.exp(alpha * (r0 - r_bc))
    neg_ealpha_y = np.exp(alpha * (r_bc - r0))

    e2alpha_x = np.exp(2 * alpha * (r0 - r_ab))
    e2alpha_y = np.exp(2 * alpha * (r0 - r_bc))

    aDenom = 1 / (1 + a)
    bDenom = 1 / (1 + b)

    Qconst = coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)
    Jconst = exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

    cDenom = 1 / (1 + c)

    d = d_bc

    dy = (
        0.25
        * bDenom**2
        * alpha
        * d
        * ealpha_y
        * (
            -2 * (1 + b) * (-1 + 3 * ealpha_y)
            + ((-3 + ealpha_y) * (2 * d * ealpha_y * (-6 + ealpha_y) - (1 + b) * d * ealpha_x * (-6 + ealpha_x) * aDenom - 4 * (1 + b) * Jconst * cDenom))
            / (
                np.sqrt(
                    (
                        ((d**2 * e2alpha_x) * (-6 + ealpha_x) ** 2 * aDenom**2)
                        + (d**2 * e2alpha_y * (-6 + ealpha_y) ** 2) * bDenom**2
                        - d**2 * np.exp(-2 * alpha * (-2 * r0 + r_ab + r_bc)) * (-1 + 6 * neg_ealpha_x) * (-1 + 6 * neg_ealpha_y) * aDenom * bDenom
                    )
                    - 4 * d * ealpha_x * (-6 + ealpha_x) * Jconst * aDenom * cDenom
                    - 4 * d * ealpha_y * (-6 + ealpha_y * Jconst * bDenom * cDenom)
                    + 16 * Jconst**2 * cDenom**2
                )
            )
        )
    )

    return dy


# +
# def grad(
#     inp,
#     a=0.05,
#     b=0.30,
#     c=0.05,
#     d_ab=4.746,
#     d_bc=4.746,
#     d_ac=3.445,
#     r0=0.742,
#     alpha=1.942,
# ):
#     r_ab,r_bc = inp

#     Q_AB = coulomb(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
#     Q_BC = coulomb(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
#     Q_AC = coulomb(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

#     J_AB = exchange(r=r_ab, d=d_ab, r0=r0, alpha=alpha)
#     J_BC = exchange(r=r_bc, d=d_bc, r0=r0, alpha=alpha)
#     J_AC = exchange(r=d_ac, d=d_ac, r0=r0, alpha=alpha)

#     dQ_dAB = dQ_dr(d=d_ab, alpha=alpha, r=r_ab, r0=r0)
#     dQ_dBC = dQ_dr(d=d_bc, alpha=alpha, r=r_bc, r0=r0)

#     dJ_dAB = dJ_dr(d=d_ab, alpha=alpha, r=r_ab, r0=r0)
#     dJ_dBC = dQ_dr(d=d_bc, alpha=alpha, r=r_bc, r0=r0)

#     result_Js_1 = (
#         ((J_AB**2) / ((1 + a) ** 2))
# + ((J_BC**2) / ((1 + b) ** 2))



# + ((J_AC**2) / ((1 + c) ** 2))
#     )
#     result_Js_2 = (
#         ((J_AB * J_BC) / ((1 + a) * (1 + b)))
# + ((J_AC * J_BC) / ((1 + c) * (1 + b)))



# + ((J_AB * J_AC) / ((1 + a) * (1 + c)))
#     )
#     result_Js = result_Js_1 - result_Js_2

#     Cj = J_AC**2/((1+c)**2)


#     dx = dQ_dAB*(1/(1+a)) - 0.5*((result_Js)**(-.5))*(2*J_AB*dJ_dAB*(1/((1+a)**2)) - (J_BC/((1+a)*(1+b)))*dJ_dAB - dJ_dAB*Cj/(1+a))
#     dy = dQ_dBC*(1/(1+b)) - 0.5*((result_Js)**(-.5))*(2*J_BC*dJ_dBC*(1/((1+b)**2)) - (J_AB/((1+a)*(1+b)))*dJ_dBC - dJ_dBC*Cj/(1+b))

#     return np.array([dx, dy])
# -

potential([1, 1])

grad([0, 0])


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
    x, y = inp
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


# + tags=[]
def toy_grad_2(inp):
    x, y = inp
    dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    return np.array([dx, dy])


# -


def toy_potential_3(inp):
    x, y = inp
    return -np.absolute(np.sin(x) * np.cos(y) * np.exp(np.absolute(1 - (np.sqrt(x**2 + y**2) / np.pi))))


def toy_grad_3(inp):
    x, y = inp
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


from scipy.optimize import fmin

fmin(toy_potential_2, np.array([4, 4]))


fmin(toy_potential_3, np.array([2, 0]))

fmin(potential, np.array([0.74200203, 4]))

# + tags=[]
np.random.seed(1)
fs = 14
# vars for sim
nsteps = 1000
k = 10
n = neb()

nimages = 15


end_point = [4, 0.74200311]
start_point = [0.74200203, 4]
# end_point = (3.00002182, 1.99995542)
# start_point = (-3.77928812, -3.28320392)
# end_point = [1.26258079e+00, -2.11266342e-06]
# start_point = [-1.71238558, -3.41884257]


en_func = potential
grad_func = grad


# # set up points
# chain = np.sort([np.random.uniform(-1, 1, size=2) for n in range(nimages)])
# chain = np.linspace((min_val, max_val), (max_val, min_val), nimages)
# chain = np.linspace((min_val, min_val), (max_val, max_val), nimages)
chain = np.linspace(start_point, end_point, nimages)
# chain = np.linspace((-5, -3), (5, 3), nimages)
# chain = [(-2,-.1),(0,2),(2,.1)]
# print(chain)

fs = 19
figsize = 10
fig, ax = plt.subplots(figsize=(1.18 * figsize, figsize))
plt.style.use("seaborn-pastel")


plt.plot(
    [(point[0]) for point in chain],
    [(point[1]) for point in chain],
    "^--",
    c="white",
    label="original",
)


opt_chain_neb, _ = n.optimize_chain(chain=chain, en_func=en_func, grad_func=grad_func, en_thre=0.001, grad_thre=0.001, max_steps=nsteps, k=10)
print(f"{opt_chain_neb=}")
points_x = [point[0] for point in opt_chain_neb]
points_y = [point[1] for point in opt_chain_neb]
plt.plot(points_x, points_y, "o--", c="white", label="NEB")


# opt_chain_neb_k0, opt_chain_traj = n.optimize_chain(
#     chain=chain,
#     en_func=en_func,
#     grad_func=grad_func,
#     en_thre=0.01,
#     grad_thre=0.01,
#     max_steps=nsteps,
#     k=0
# )
# points_x = [point[0] for point in opt_chain_neb]
# points_y = [point[1] for point in opt_chain_neb]
# plt.plot(points_x, points_y, "o--", c="red", label="k=0")


# opt_chain_neb_k20, opt_chain_traj = n.optimize_chain(
#     chain=chain,
#     en_func=en_func,
#     grad_func=grad_func,
#     en_thre=0.01,
#     grad_thre=0.01,
#     max_steps=nsteps,
#     k=20
# )
# points_x = [point[0] for point in opt_chain_neb]
# points_y = [point[1] for point in opt_chain_neb]
# plt.plot(points_x, points_y, "o--", c="blue", label="k=20")

plt.legend(fontsize=fs)

plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)


# set up plot for potential


x = np.linspace(start=0.5, stop=4, num=num)
xx, yy = np.meshgrid(x, x)
# y = x.reshape(-1, 1)


# h = en_func(x, y)
h = en_func([xx, yy])
cs = plt.contourf(xx, yy, h)
cbar = f.colorbar(cs, ax=ax)
cbar.ax.tick_params(labelsize=fs)


plt.show()


# +
# # %matplotlib widget
# set up plot for potential
min_val = 1
max_val = 4
num = 100
fig = 10
f, ax = plt.subplots(figsize=(1.18 * fig, fig))
x = np.linspace(start=min_val, stop=max_val, num=num)
y = np.linspace(start=min_val, stop=max_val, num=num)

xx, yy = np.meshgrid(x, y)

h = potential([xx, yy])
cs = plt.contourf(xx, yy, h)
cbar = f.colorbar(cs)
plt.show()
# -

ens = [en_func(x) for x in opt_chain_neb]
# ens_peb = [en_func(x[0], x[1]) for x in opt_chain_peb]
ens_orig = [en_func(x) for x in chain]
plt.plot(ens, "o", label="neb")
# plt.plot(ens_peb, 'o--',label="peb")
plt.plot(ens_orig, label="orig")
plt.legend()
