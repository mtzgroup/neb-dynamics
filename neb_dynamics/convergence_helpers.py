import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import NEBInputs


def _check_en_converged(chain_prev: Chain, chain_new: Chain, threshold: float) -> bool:
    differences = np.abs(chain_new.energies - chain_prev.energies)
    indices_converged = np.where(differences <= threshold)
    return indices_converged[0], differences


def _check_springgrad_converged(chain: Chain, threshold: float) -> bool:
    import neb_dynamics.chainhelpers as ch
    bools = []
    grad_norms_components = []
    pe_grads_nudged, spring_forces_nudged = ch.pe_grads_spring_forces_nudged(
        chain=chain)

    for grad in spring_forces_nudged:
        grad = np.array(grad)
        grad_norm = np.dot(grad.flatten(), grad.flatten()) / len(grad)
        grad_norms_components.append(grad_norm)
        bools.append(grad_norm < threshold)

    return np.where(bools), grad_norms_components


def _check_gperps_converged(chain: Chain, threshold: float) -> bool:
    import neb_dynamics.chainhelpers as ch
    bools = []
    max_grad_components = []
    pe_grads_nudged, spring_forces_nudged = ch.pe_grads_spring_forces_nudged(
        chain=chain)
    for grad in pe_grads_nudged:
        max_grad = np.amax(np.abs(grad))
        max_grad_components.append(max_grad)
        bools.append(max_grad < threshold)

    return np.where(bools), max_grad_components


def _check_barrier_height_conv(chain_prev: Chain, chain_new: Chain, threshold: float):
    prev_eA = chain_prev.get_eA_chain()
    new_eA = chain_new.get_eA_chain()

    delta_eA = np.abs(new_eA - prev_eA)
    return delta_eA <= threshold


def _check_rms_grad_converged(chain: Chain, threshold: float) -> Tuple[NDArray, NDArray]:
    """
    returns two arrays. first array are the indices of converged nodes.
    second array is the RMS values of the perpendicular gradients.
    """
    bools = []
    rms_gperps = []

    for rms_gp, rms_gradient in zip(chain.rms_gperps, chain.rms_gradients):
        # rms_gperps.append(rms_gp)

        # # the boolens are used exclusively for deciding to freeze nodes or not
        # # I want the spring forces to affect whether a node is frozen, but not
        # # to affect the total chain's convergence.
        # rms_grad_converged = rms_gradient <= threshold

        # 01072025: Maybe the spring forces should affect, so we don't cheat it by
        # having bunch of nodes near minima
        rms_grad_converged = rms_gradient <= threshold
        rms_gperps.append(rms_gradient)

        bools.append(rms_grad_converged)

    return np.where(bools), rms_gperps


def chain_converged(chain_prev: Chain, chain_new: Chain, parameters: NEBInputs) -> bool:
    import neb_dynamics.chainhelpers as ch
    fraction_freeze = 1/10

    rms_grad_conv_ind, rms_gperps = _check_rms_grad_converged(
        chain_new, threshold=parameters.rms_grad_thre*fraction_freeze)

    gperp_conv_ind, gperps = _check_gperps_converged(
        chain_new, threshold=parameters.rms_grad_thre*fraction_freeze)

    ts_triplet_gspring = chain_new.ts_triplet_gspring_infnorm
    grad_converged_indices, springgrads = _check_springgrad_converged(chain=chain_new,
                                                                      threshold=parameters.
                                                                      ts_spring_thre*fraction_freeze)
    g_perps = ch.get_g_perps(chain_new)

    converged_nodes_indices = np.intersect1d(
        grad_converged_indices, rms_grad_conv_ind
    )

    converged_nodes_indices = np.intersect1d(
        gperp_conv_ind, converged_nodes_indices
    )

    ind_ts_node = chain_new.energies.argmax()
    # never freeze TS node
    converged_nodes_indices = converged_nodes_indices[converged_nodes_indices != ind_ts_node]
    # print(f"{len(converged_nodes_indices)}=")
    if chain_new.parameters.node_freezing:
        _update_node_convergence(
            chain=chain_new, indices=converged_nodes_indices, prev_chain=chain_prev)
        _copy_node_information_to_converged(
            new_chain=chain_new, old_chain=chain_prev)

    barrier_height_converged = _check_barrier_height_conv(
        chain_prev=chain_prev, chain_new=chain_new, threshold=parameters.barrier_thre)
    ind_ts_guess = np.argmax(chain_new.energies)
    ts_guess_grad = np.amax(np.abs(g_perps[ind_ts_guess]))
    criteria_converged = [
        np.amax(rms_gperps) <= parameters.max_rms_grad_thre,
        sum(rms_gperps)/len(chain_new) <= parameters.rms_grad_thre,
        ts_guess_grad <= parameters.ts_grad_thre,
        ts_triplet_gspring <= parameters.ts_spring_thre,
        max(springgrads) <= parameters.ts_spring_thre,
        barrier_height_converged]

    CRITERIA_NAMES = ["MAX(RMS_GPERP)", "MEAN(RMS_GPERP)",
                      "TS_GRAD", "TS_SPRING", "INFNORM_SPRING", "BARRIER_HEIGHT"]
    print(f"\n{list(zip(CRITERIA_NAMES, criteria_converged))}\n")

    converged = sum(criteria_converged) == len(criteria_converged)

    return converged


def _update_node_convergence(chain: Chain, indices: np.array, prev_chain: Chain) -> None:
    endpoints_indices = [0, len(chain)-1]
    for i, (node, prev_node) in enumerate(zip(chain, prev_chain)):
        if i in indices or i in endpoints_indices:
            if prev_node._cached_gradient is not None:
                # print(f"node{i} is frozen with _cached res: {prev_node._cached_result}")
                node.converged = True
                node._cached_gradient = prev_node._cached_gradient
                node._cached_energy = prev_node._cached_energy
        else:
            node.converged = False


def _copy_node_information_to_converged(new_chain: Chain, old_chain: Chain) -> None:
    for new_node, old_node in zip(new_chain.nodes, old_chain.nodes):
        if old_node.converged:
            new_node._cached_gradient = old_node._cached_gradient
            new_node._cached_energy = old_node._cached_energy
