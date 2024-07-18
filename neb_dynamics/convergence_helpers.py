import numpy as np
from chain import Chain
from neb_dynamics.inputs import NEBInputs


def _check_en_converged(chain_prev: Chain, chain_new: Chain, threshold: float) -> bool:
    differences = np.abs(chain_new.energies - chain_prev.energies)
    indices_converged = np.where(differences <= threshold)
    return indices_converged[0], differences


def _check_grad_converged(chain: Chain, threshold: float) -> bool:
    bools = []
    max_grad_components = []
    gradients = chain.gradients
    for grad in gradients:
        max_grad = np.amax(np.abs(grad))
        max_grad_components.append(max_grad)
        bools.append(max_grad < threshold)

    return np.where(bools), max_grad_components


def _check_barrier_height_conv(chain_prev: Chain, chain_new: Chain, threshold: float):
    prev_eA = chain_prev.get_eA_chain()
    new_eA = chain_new.get_eA_chain()

    delta_eA = np.abs(new_eA - prev_eA)
    return delta_eA <= threshold


def _check_rms_grad_converged(chain: Chain, threshold: float):
    bools = []
    rms_gperps = []

    for rms_gp, rms_gradient in zip(chain.rms_gperps, chain.rms_gradients):
        rms_gperps.append(rms_gp)

        # the boolens are used exclusively for deciding to freeze nodes or not
        # I want the spring forces to affect whether a node is frozen, but not
        # to affect the total chain's convergence.
        rms_grad_converged = rms_gradient <= threshold
        bools.append(rms_grad_converged)

    return np.where(bools), rms_gperps


def chain_converged(chain_prev: Chain, chain_new: Chain, parameters: NEBInputs) -> bool:
    import neb_dynamics.chainhelpers as ch
    rms_grad_conv_ind, rms_gperps = _check_rms_grad_converged(
        chain_new, threshold=parameters.rms_grad_thre)
    ts_triplet_gspring = chain_new.ts_triplet_gspring_infnorm
    en_converged_indices, en_deltas = _check_en_converged(
        chain_prev=chain_prev, chain_new=chain_new,
        threshold=parameters.en_thre
    )

    grad_conv_ind, max_grad_components = _check_grad_converged(
        chain=chain_new, threshold=parameters.grad_thre)

    converged_nodes_indices = np.intersect1d(
        en_converged_indices, rms_grad_conv_ind
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
    ts_guess_grad = np.amax(np.abs(ch.get_g_perps(chain_new)[ind_ts_guess]))

    criteria_converged = [
        max(rms_gperps) <= parameters.max_rms_grad_thre,
        sum(rms_gperps)/len(chain_new) <= parameters.rms_grad_thre,
        ts_guess_grad <= parameters.ts_grad_thre,
        ts_triplet_gspring <= parameters.ts_spring_thre,
        barrier_height_converged]

    converged = sum(criteria_converged) >= 5

    return converged


def _update_node_convergence(chain: Chain, indices: np.array, prev_chain: Chain) -> None:
    for i, (node, prev_node) in enumerate(zip(chain, prev_chain)):
        if i in indices:
            if prev_node._cached_result is not None:
                node.converged = True
                node._cached_result = prev_node._cached_result
        else:
            node.converged = False


def _copy_node_information_to_converged(new_chain: Chain, old_chain: Chain) -> None:
    for new_node, old_node in zip(new_chain.nodes, old_chain.nodes):
        if old_node.converged:
            new_node._cached_result = old_node._cached_result
