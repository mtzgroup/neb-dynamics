from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from nodes.node import Node
from numpy.typing import NDArray

from neb_dynamics.chain import Chain
from neb_dynamics.geodesic_interpolation.coord_utils import align_geom
from neb_dynamics.helper_functions import get_mass
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.engine import Engine


def _distance_to_chain(chain1: Chain, chain2: Chain) -> float:
    """
    calculates the distance between two chains of same length
    by summing the RMSD for nodes with shared indices, then
    normalizing by chain length.
    """
    distances = []

    for node1, node2 in zip(chain1.nodes, chain2.nodes):
        if node1.coords.shape[0] > 2:
            dist, _ = align_geom(refgeom=node1.coords, geom=node2.coords)
        else:
            dist = np.linalg.norm(node1.coords - node2.coords)
        distances.append(dist)

    return sum(distances) / len(chain1)


def _tangent_correlations(chain: Chain, other_chain: Chain) -> float:
    """
    returns the vector correlation of unit tangents of two different chains.
    """
    chain1_vec = np.array(chain.unit_tangents).flatten()
    chain2_vec = np.array(other_chain.unit_tangents).flatten()
    projector = np.dot(chain1_vec, chain2_vec)
    normalization = np.dot(chain1_vec, chain1_vec)

    return projector / normalization


def _gperp_correlation(chain: Chain, other_chain: Chain):
    gp_chain = get_g_perps(chain)
    gp_other_chain = get_g_perps(other_chain)
    dp = np.dot(gp_chain.flatten(), gp_other_chain.flatten())
    normalization = np.linalg.norm(gp_chain) * np.linalg.norm(gp_other_chain)
    return dp / normalization


def _gradient_correlation(chain: Chain, other_chain: Chain):
    chain1_vec = np.array(compute_NEB_gradient(chain=chain)).flatten()
    chain1_vec = chain1_vec / np.linalg.norm(chain1_vec)

    chain2_vec = np.array(compute_NEB_gradient(other_chain)).flatten()
    chain2_vec = chain2_vec / np.linalg.norm(chain2_vec)

    projector = np.dot(chain1_vec, chain2_vec)
    normalization = np.dot(chain1_vec, chain1_vec)

    return projector / normalization


def _get_mass_weighed_coords(chain: Chain):
    coords = chain.coordinates
    symbols = chain.symbols

    weights = np.array([np.sqrt(get_mass(s)) for s in symbols])
    weights = weights / sum(weights)
    mass_weighed_coords = coords * weights.reshape(-1, 1)
    return mass_weighed_coords


def iter_triplets(chain: Chain) -> list[list[Node]]:
    for i in range(1, len(chain.nodes) - 1):
        yield chain.nodes[i - 1: i + 2]


def neighs_grad_func(chain: Chain, prev_node: Node, current_node: Node, next_node: Node):
    vec_tan_path = _create_tangent_path(
        prev_node=prev_node, current_node=current_node, next_node=next_node
    )
    unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

    pe_grad = current_node.gradient

    # remove rotations and translations
    if pe_grad.shape[1] >= 3:  # if we have at least 3 atoms
        pe_grad[0, :] = 0  # this atom cannot move
        pe_grad[1, :2] = 0  # this atom can only move in a line
        pe_grad[2, :1] = 0  # this atom can only move in a plane

    if current_node.do_climb:
        pe_along_path_const = np.dot(
            pe_grad.flatten(), unit_tan_path.flatten())
        pe_along_path = pe_along_path_const * unit_tan_path

        climbing_grad = 2 * pe_along_path

        pe_grads_nudged = pe_grad - climbing_grad

        zero = np.zeros_like(pe_grad)
        spring_forces_nudged = zero

    else:
        pe_grads_nudged = get_nudged_pe_grad(
            unit_tangent=unit_tan_path, gradient=pe_grad
        )
        spring_forces_nudged = get_force_spring_nudged(
            chain=chain,
            prev_node=prev_node,
            current_node=current_node,
            next_node=next_node,
            unit_tan_path=unit_tan_path,
        )

    return pe_grads_nudged, spring_forces_nudged


def pe_grads_spring_forces_nudged(chain: Chain):
    pe_grads_nudged = []
    spring_forces_nudged = []

    for prev_node, current_node, next_node in iter_triplets(chain):
        pe_grad_nudged, spring_force_nudged = neighs_grad_func(
            chain=chain,
            prev_node=prev_node,
            current_node=current_node,
            next_node=next_node,
        )

        pe_grads_nudged.append(pe_grad_nudged)
        spring_forces_nudged.append(spring_force_nudged)

    pe_grads_nudged = np.array(pe_grads_nudged)
    spring_forces_nudged = np.array(spring_forces_nudged)
    return pe_grads_nudged, spring_forces_nudged


def get_g_perps(chain: Chain) -> NDArray:
    """
    computes the perpendicular gradients of a chain.
    """
    pe_grads_nudged, _ = pe_grads_spring_forces_nudged(chain=chain)
    zero = np.zeros_like(pe_grads_nudged[0])
    grads = np.insert(pe_grads_nudged, 0, zero, axis=0)
    grads = np.insert(grads, len(grads), zero, axis=0)

    return grads


def _k_between_nodes(
    node0: Node, node1: Node, e_ref: float, k_max: float, e_max: float,
    parameters: ChainInputs
):
    e_i = max(node1.energy, node0.energy)
    if e_i > e_ref:
        new_k = k_max - parameters.delta_k * \
            ((e_max - e_i) / (e_max - e_ref))
    elif e_i <= e_ref:
        new_k = k_max - parameters.delta_k
    return new_k


def compute_NEB_gradient(chain: Chain) -> NDArray:
    """
    will return the sum of the perpendicular gradient
    and the spring gradient
    """
    pe_grads_nudged, spring_forces_nudged = pe_grads_spring_forces_nudged(
        chain)

    grads = (
        pe_grads_nudged - spring_forces_nudged
    )

    # endpoints have 0 gradient because we freeze them
    zero = np.zeros_like(grads[0])
    grads = np.insert(grads, 0, zero, axis=0)
    grads = np.insert(grads, len(grads), zero, axis=0)
    return grads


def _create_tangent_path(prev_node: Node, current_node: Node, next_node: Node):
    en_2 = next_node.energy
    en_1 = current_node.energy
    en_0 = prev_node.energy
    if en_2 > en_1 and en_1 > en_0:
        return next_node.coords - current_node.coords
    elif en_2 < en_1 and en_1 < en_0:
        return current_node.coords - prev_node.coords

    else:
        deltaV_max = max(np.abs(en_2 - en_1), np.abs(en_0 - en_1))
        deltaV_min = min(np.abs(en_2 - en_1), np.abs(en_0 - en_1))

        tau_plus = next_node.coords - current_node.coords
        tau_minus = current_node.coords - prev_node.coords
        if en_2 > en_0:
            tan_vec = deltaV_max * tau_plus + deltaV_min * tau_minus
        elif en_2 < en_0:
            tan_vec = deltaV_min * tau_plus + deltaV_max * tau_minus

        else:
            return 0.5 * (tau_minus + tau_plus)
            # raise ValueError(
            #     f"Energies adjacent to current node are identical. {en_2=} {en_0=}"
            # )

        return tan_vec


def get_force_spring_nudged(
    chain: Chain,
    prev_node: Node,
    current_node: Node,
    next_node: Node,
    unit_tan_path: np.array,
):
    parameters = chain.parameters
    k_max = (
        max(parameters.k)
        if hasattr(parameters.k, "__iter__")
        else parameters.k
    )
    e_ref = max(chain.nodes[0].energy,
                chain.nodes[len(chain.nodes) - 1].energy)
    e_max = max(chain.energies)

    k01 = _k_between_nodes(
        node0=prev_node,
        node1=current_node,
        e_ref=e_ref,
        k_max=k_max,
        e_max=e_max,
        parameters=parameters
    )

    k12 = _k_between_nodes(
        node0=current_node,
        node1=next_node,
        e_ref=e_ref,
        k_max=k_max,
        e_max=e_max,
        parameters=parameters
    )

    force_spring = k12 * np.linalg.norm(
        next_node.coords - current_node.coords
    ) - k01 * np.linalg.norm(current_node.coords - prev_node.coords)
    return force_spring * unit_tan_path


def _select_split_method(self, conditions: dict, irc_results, concavity_results):
    all_conditions_met = all([val for key, val in conditions.items()])
    if all_conditions_met:
        return None

    if conditions["concavity"] is False:  # prioritize the minima condition
        return "minima"
    elif conditions["irc"] is False:
        return "maxima"


def get_nudged_pe_grad(unit_tangent: np.array, gradient: np.array):
    """
    Returns the component of the gradient that acts perpendicular to the path tangent
    """
    pe_grad = gradient
    pe_grad_nudged_const = np.dot(pe_grad.flatten(), unit_tangent.flatten())
    pe_grad_nudged = pe_grad - pe_grad_nudged_const * unit_tangent
    return pe_grad_nudged


def animate_chain_trajectory(
    chain_traj, min_y=-100, max_y=100,
    max_x=1.1, min_x=-0.1, norm_path_len=True
):

    figsize = 5
    fig, ax = plt.subplots(figsize=(1.618 * figsize, figsize))

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    (line,) = ax.plot([], [], "o--", lw=3)

    def animate(chain):
        if norm_path_len:
            x = chain.integrated_path_length
        else:
            x = chain.path_length

        y = chain.energies_kcalmol
        line.set_data(x, y)
        line.set_color("skyblue")
        return

    ani = FuncAnimation(
        fig, animate, frames=chain_traj)
    return HTML(ani.to_jshtml())

