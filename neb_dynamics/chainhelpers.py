from __future__ import annotations
from IPython.display import display, HTML
import base64
import io

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
from qcop.exceptions import ExternalProgramError
from qcio.view import generate_structure_viewer_html
from scipy.signal import argrelextrema

from neb_dynamics.chain import Chain
from neb_dynamics.nodes.node import StructureNode, Node, XYNode
from neb_dynamics.geodesic_interpolation.coord_utils import align_geom
from neb_dynamics.geodesic_interpolation.geodesic import (
    run_geodesic_py,
    run_geodesic_get_smoother,
)
from neb_dynamics.errors import ElectronicStructureError
from neb_dynamics.helper_functions import get_mass
from neb_dynamics.inputs import ChainInputs, GIInputs
from neb_dynamics.helper_functions import (
    linear_distance,
    qRMSD_distance,
)
from ipywidgets import IntSlider, interact


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


def _get_ind_minima(chain):
    ind_minima = argrelextrema(chain.energies, np.less, order=1)[0]
    return ind_minima


def _get_ind_maxima(chain):
    maxima_indices = argrelextrema(chain.energies, np.greater, order=1)[0]
    if len(maxima_indices) > 1:
        ind_maxima = maxima_indices[0]
    else:
        ind_maxima = int(maxima_indices)
    return ind_maxima


def _get_mass_weights(chain: Chain, normalize_weights=True):
    symbols = chain[0].symbols

    weights = np.array([np.sqrt(get_mass(s)) for s in symbols])
    if normalize_weights:
        weights = weights / sum(weights)
    return weights


def _get_mass_weighed_coords(chain: Chain):
    weights = _get_mass_weights(chain)
    coords = np.array([node.coords for node in chain])
    mass_weighed_coords = coords * weights.reshape(-1, 1)
    return mass_weighed_coords


def iter_triplets(chain: Chain) -> list[list[Node]]:
    for i in range(1, len(chain.nodes) - 1):
        yield chain.nodes[i - 1: i + 2]


def neighs_grad_func(
    chain: Chain, prev_node: Node, current_node: Node, next_node: Node
):
    vec_tan_path = _create_tangent_path(
        prev_node=prev_node, current_node=current_node, next_node=next_node
    )
    unit_tan_path = vec_tan_path / np.linalg.norm(vec_tan_path)

    pe_grad = current_node.gradient

    # remove rotations and translations
    # if we have at least 3 atoms
    if len(pe_grad.shape) > 1 and pe_grad.shape[1] >= 3:
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
    node0: Node,
    node1: Node,
    e_ref: float,
    k_max: float,
    e_max: float,
    parameters: ChainInputs,
):
    e_i = max(node1.energy, node0.energy)
    if e_i > e_ref:
        new_k = k_max - parameters.delta_k * ((e_max - e_i) / (e_max - e_ref))
    elif e_i <= e_ref:
        new_k = k_max - parameters.delta_k
    return new_k


def compute_NEB_gradient(chain: Chain) -> NDArray:
    """
    will return the sum of the perpendicular gradient
    and the spring gradient
    """
    pe_grads_nudged, spring_forces_nudged = pe_grads_spring_forces_nudged(
        chain=chain)

    grads = pe_grads_nudged - spring_forces_nudged

    # endpoints have 0 gradient because we freeze them
    if len(grads) == 0:
        print(
            f"WTAF: \n\n{chain.gradients=}\n\n{pe_grads_nudged=}\n\n{spring_forces_nudged=}"
        )
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
    k_max = max(parameters.k) if hasattr(
        parameters.k, "__iter__") else parameters.k
    e_ref = max(chain.nodes[0].energy,
                chain.nodes[len(chain.nodes) - 1].energy)
    e_max = max(chain.energies)

    k01 = _k_between_nodes(
        node0=prev_node,
        node1=current_node,
        e_ref=e_ref,
        k_max=k_max,
        e_max=e_max,
        parameters=parameters,
    )

    k12 = _k_between_nodes(
        node0=current_node,
        node1=next_node,
        e_ref=e_ref,
        k_max=k_max,
        e_max=e_max,
        parameters=parameters,
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


def gi_path_to_chain(
    xyz_coords: np.array, parameters: ChainInputs, symbols: list, charge=0, spinmult=1
):
    from qcio import Structure

    new_nodes = []
    for new_coords in xyz_coords:
        struct = Structure(
            geometry=new_coords, symbols=symbols, charge=charge, multiplicity=spinmult
        )
        node = StructureNode(structure=struct)
        new_nodes.append(node)
    chain_copy = Chain(nodes=new_nodes, parameters=parameters)
    return chain_copy


def run_geodesic(chain: Union[Chain, List[StructureNode]], chain_inputs=None, **kwargs):
    if isinstance(chain, list) and chain_inputs is None:
        print(
            "Warning! You input a list of nodes to interpolate and no ChainInputs. Will use defaults ChainInputs"
        )
        chain_inputs = ChainInputs()
    elif isinstance(chain, Chain):
        chain_inputs = chain.parameters
    coords = np.array([node.coords for node in chain])
    xyz_coords = run_geodesic_py((chain[0].symbols, coords), **kwargs)
    charge = chain[0].structure.charge
    spinmult = chain[0].structure.multiplicity
    chain_copy = gi_path_to_chain(
        xyz_coords=xyz_coords,
        parameters=chain_inputs.copy(),
        symbols=chain[0].symbols,
        charge=charge,
        spinmult=spinmult,
    )
    return chain_copy


def calculate_geodesic_distance(
    node1: StructureNode, node2: StructureNode, nimages=12, nudge=0.1
):
    from neb_dynamics.helper_functions import RMSD

    rmsd_dist, _ = RMSD(node1.coords, node2.coords)
    nudge = min(rmsd_dist, nudge)
    smoother = run_geodesic_get_smoother(
        input_object=[node1.symbols, [node1.coords, node2.coords]],
        nudge=nudge,
        nimages=nimages,
    )
    return smoother.length


def calculate_geodesic_xtb_barrier(
    node1: StructureNode, node2: StructureNode, nimages=12, nudge=0.1
):
    from neb_dynamics.engines.ase import ASEEngine
    from xtb.ase.calculator import XTB
    calc = XTB()
    eng = ASEEngine(calculator=calc)

    gi = run_geodesic([node1, node2], nimages=nimages, nudge=nudge)
    try:
        eng.compute_energies(gi)
    except ElectronicStructureError:
        return np.inf

    return max(gi.energies_kcalmol)


def _update_cache(self, chain: Chain, gradients: NDArray, energies: NDArray) -> None:
    """
    will update the `_cached_energy` and `_cached_gradient` attributes in the chain
    nodes based on the input `gradients` and `energies`
    """
    from neb_dynamics.fakeoutputs import FakeQCIOOutput, FakeQCIOResults

    for node, grad, ene in zip(chain, gradients, energies):
        res = FakeQCIOResults(energy=ene, gradient=grad)
        outp = FakeQCIOOutput(results=res)
        node._cached_result = outp
        node._cached_energy = ene
        node._cached_gradient = grad


def create_friction_optimal_gi(
    chain: Chain, gi_inputs: GIInputs, chain_inputs: ChainInputs()
):
    from neb_dynamics.engines.qcop import QCOPEngine
    print("GI: Optimizing friction parameter")
    eng = QCOPEngine()
    frics = [0.0001, 0.001, 0.01, 0.1, 1]
    all_gis = [
        run_geodesic(
            chain=chain,
            nimages=gi_inputs.nimages,
            friction=fric,
            nudge=gi_inputs.nudge,
            chain_inputs=chain_inputs,
            **gi_inputs.extra_kwds,
        )
        for fric in frics
    ]
    eAs = []
    for gi in all_gis:
        try:
            _ = eng.compute_energies(gi)
            eAs.append(gi.get_eA_chain())
        except ExternalProgramError:
            eAs.append(10000000)
    ind_best = np.argmin(eAs)
    gi = all_gis[ind_best]
    _reset_cache(gi)
    print(f"GI: Chose friction: {frics[ind_best]}")
    return gi


def _calculate_chain_distances(chain_traj: List[Chain]):
    distances = [None]  # None for the first chain
    for i, chain in enumerate(chain_traj):
        if i == 0:
            continue

        prev_chain = chain_traj[i - 1]
        dist = prev_chain._distance_to_chain(chain)
        distances.append(dist)
    return np.array(distances)


def _reset_node_convergence(chain) -> None:
    """
    sets each node in chain  to `node.converged = False`
    """
    for node in chain:
        node.converged = False


def _reset_cache(chain) -> None:
    """
    sets each node in chain  to `node.converged = False`
    """
    for node in chain:
        node._cached_energy = None
        node._cached_gradient = None
        node._cached_result = None


def extend_by_n_frames(list_obj: List, n: int = 2):
    """
    will return the same chain where each node has been duplicated by n
    """
    orig_list = list_obj
    new_list = []
    for value in orig_list:
        new_list.extend([value] * n)

    return new_list


def _animate_structure_list(structure_list):
    """
    animates a list of qcio structure objects
    """
    structure_html = generate_structure_viewer_html(structure_list)
    return display(HTML(structure_html))


def animate_chain_trajectory(
    chain_traj, min_y=-100, max_y=100, max_x=1.1, min_x=-0.1, norm_path_len=True
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

    ani = FuncAnimation(fig, animate, frames=chain_traj)
    return HTML(ani.to_jshtml())


def generate_neb_plot(
    chain: List[Node],
    ind_node,
    figsize=(6.4, 4.8),
    grid=True,
    markersize=20,
    title="Energies across chain",
) -> str:
    """
    generate plot of chain
    """
    try:
        energies = _energies_kcalmol(chain)
    except:
        print("Cannot plot energies.")
        return ""

    fig, ax1 = plt.subplots(figsize=figsize)
    color = "tab:blue"
    ax1.set_xlabel("Path length")
    ax1.set_ylabel("Relative energies (kcal/mol)", color=color)
    markercolors = ["green"] * len(chain)
    markersizes = [markersize] * len(chain)

    markercolors[ind_node] = "gold"
    markersizes[ind_node] = markersize + 50

    ax1.plot(path_length(chain=chain), energies, color="green")
    ax1.scatter(
        path_length(chain=chain),
        energies,
        label="Energy",
        marker="o",
        color=markercolors,
        s=markersizes,
    )

    ax1.tick_params(axis="y", labelcolor=color)
    plt.title(title, pad=20)
    ax1.legend(loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)  # Close the figure to avoid duplicate plots
    return image_base64


def _path_len_dist_func(coords1, coords2, molecular_system: bool = False):
    if molecular_system:
        return qRMSD_distance(coords1, coords2)
    else:
        return linear_distance(coords1, coords2)


def path_length(chain: List[Node]) -> np.array:
    if isinstance(chain[0], StructureNode):
        molecular_system = True
    elif isinstance(chain[0], XYNode):
        molecular_system = False
    else:
        raise NotImplementedError(
            f"Cannot compute path length for node type: {type(chain[0])}"
        )

    if molecular_system:
        coords = _get_mass_weighed_coords(chain)
    else:
        coords = np.array([node.coords for node in chain])

    cum_sums = [0]
    for i, frame_coords in enumerate(coords):
        if i == len(coords) - 1:
            continue
        next_frame = coords[i + 1]
        distance = _path_len_dist_func(
            frame_coords, next_frame, molecular_system=molecular_system
        )
        cum_sums.append(cum_sums[-1] + distance)

    cum_sums = np.array(cum_sums)
    path_len = cum_sums
    return np.array(path_len)


def _energies_kcalmol(chain: List[Node]):
    """
    returns the relative energies of the list of nodes
    in kcal/mol, where the first energy will be set to 0.
    """
    enes = np.array([node.energy for node in chain])
    return (enes - enes[0]) * 627.5


def build_correlation_matrix(node_list, ts_vector):
    node_list = [(node.coords.flatten() - ts_vector) for node in node_list]
    a = node_list[0]
    mat = np.zeros(shape=(len(a), len(a)))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            products = sum([vec[i]*vec[j]
                           for vec in node_list]) / len(node_list)
            sums = sum([vec[i] for vec in node_list])*sum([vec[j]
                                                           for vec in node_list]) / (len(node_list)**2)
            mat[i, j] = products - sums
    return mat


def get_rxn_coordinate(c: Chain):
    mat = build_correlation_matrix(
        c, ts_vector=c.get_ts_node().coords.flatten())
    evals, evecs = np.linalg.eigh(mat)
    print(evals)
    return evecs[:, -1]


def get_projections(c: Chain, eigvec):
    ind_ts = c.energies.argmax()
    # ind_ts = 0

    all_dists = []
    for i, node in enumerate(c):
        displacement = c[i].coords.flatten() - c[ind_ts].coords.flatten()
        all_dists.append(np.dot(displacement, eigvec))
    # plt.plot(all_dists)
    return all_dists


def visualize_chain(chain: List[StructureNode]):
    """
    returns an interactive visualizer for a chain
    """

    def wrap(frame):
        final_html = []
        image_base64 = generate_neb_plot(chain, ind_node=frame)
        structure_html = generate_structure_viewer_html(chain[frame].structure)
        img_html = (
            f'<img src="data:image/png;base64,{image_base64}" alt="Energy Optimization by '
            'Cycle" style="width: 100%; max-width: 600px;">'
        )

        final_html.append(
            f"""
        <div style="text-align: center;">
            <div style="display: flex; align-items: center; justify-content: space-around;">
                <div style="text-align: center; margin-right: 20px; flex: 1;">
                    <div style="display: inline-block; text-align: center;">
                        {structure_html}
                    </div>
                </div>
                <div style="text-align: center; margin-left: 20px; flex: 1;">
                    {img_html}
                </div>
            </div>
        </div>
                """
        )

        return HTML("".join(final_html))

    return interact(
        wrap,
        frame=IntSlider(
            min=0, max=len(chain) - 1, step=1, description="Trajectory frames"
        ),
    )


def plot_opt_history(chain_trajectory: List[Chain], do_3d=False):

    s = 8
    fs = 18

    if do_3d:
        all_chains = chain_trajectory

        ens = np.array([c.energies - c.energies[0] for c in all_chains])
        all_integrated_path_lengths = np.array(
            [c.integrated_path_length for c in all_chains]
        )
        opt_step = np.array(list(range(len(all_chains))))
        s = 7
        fs = 18
        ax = plt.figure(figsize=(1.16 * s, s)).add_subplot(projection="3d")

        # Plot a sin curve using the x and y axes.
        x = opt_step
        ys = all_integrated_path_lengths
        zs = ens
        for i, (xind, y) in enumerate(zip(x, ys)):
            if i < len(ys) - 1:
                ax.plot(
                    [xind] * len(y),
                    y,
                    "o-",
                    zs=zs[i],
                    color="gray",
                    markersize=3,
                    alpha=0.1,
                )
            else:
                ax.plot([xind] * len(y), y, "o-", zs=zs[i],
                        color="blue", markersize=3)
        ax.grid(False)

        ax.set_xlabel("optimization step", fontsize=fs)
        ax.set_ylabel("integrated path length", fontsize=fs)
        ax.set_zlabel("energy (hartrees)", fontsize=fs)

        # Customize the view angle so it's easier to see that the scatter points lie
        # on the plane y=0
        ax.view_init(elev=20.0, azim=-45)
        plt.tight_layout()
        plt.show()

    else:
        f, ax = plt.subplots(figsize=(1.16 * s, s))

        for i, chain in enumerate(chain_trajectory):
            if i == len(chain_trajectory) - 1:
                plt.plot(chain.integrated_path_length,
                         chain.energies, "o-", alpha=1)
            else:
                plt.plot(
                    chain.integrated_path_length,
                    chain.energies,
                    "o-",
                    alpha=0.1,
                    color="gray",
                )

        plt.xlabel("Integrated path length", fontsize=fs)

        plt.ylabel("Energy (kcal/mol)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.show()
