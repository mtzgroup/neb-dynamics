import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import NEBInputs

# Try to use rich for pretty printing, fall back to regular print
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _console = Console()
    _rich_available = True
except ImportError:
    _console = None
    _rich_available = False

# Try to use rich for pretty printing
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    _rich_available = True
    _console = Console()
except ImportError:
    _rich_available = False
    _console = None

# Rich imports for pretty printing
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    _console = Console()
    _rich_available = True
except ImportError:
    _console = None
    _rich_available = False


def _check_en_converged(energies_prev: NDArray, energies_new: NDArray, threshold: float) -> Tuple[NDArray, NDArray]:
    differences = np.abs(energies_new - energies_prev)
    indices_converged = np.where(differences <= threshold)
    return indices_converged[0], differences


def _check_springgrad_converged(spring_forces: NDArray, threshold: float) -> Tuple[NDArray, NDArray]:
    bools = []
    grad_norms_components = []

    for grad in spring_forces:
        N = len(grad)
        grad = np.array(grad)
        grad_norm = np.dot(grad.flatten(), grad.flatten()) / N
        grad_norms_components.append(grad_norm)
        bools.append(grad_norm < threshold)

    return np.where(bools), grad_norms_components


def _check_gperps_converged(pe_grads: NDArray, threshold: float) -> Tuple[NDArray, NDArray]:
    bools = []
    max_grad_components = []

    for grad in pe_grads:
        max_grad = np.amax(np.abs(grad))
        max_grad_components.append(max_grad)
        bools.append(max_grad < threshold)

    return np.where(bools), max_grad_components


def _check_barrier_height_conv(chain_prev: Chain, chain_new: Chain, threshold: float):
    prev_eA = chain_prev.get_eA_chain()
    new_eA = chain_new.get_eA_chain()

    delta_eA = np.abs(new_eA - prev_eA)
    return delta_eA <= threshold


def _check_rms_grad_converged(gradients: NDArray, threshold: float) -> Tuple[NDArray, NDArray]:
    """
    Returns two arrays. First array contains the indices of converged nodes.
    Second array contains the RMS values of the gradients.
    """
    bools = []
    rms_gperps = []

    for grad in gradients:
        rms_gradient = np.sqrt(np.mean(np.square(grad)))
        rms_grad_converged = rms_gradient <= threshold
        rms_gperps.append(rms_gradient)
        bools.append(rms_grad_converged)

    return np.where(bools), rms_gperps


def chain_converged(chain_prev: Chain, chain_new: Chain, parameters: NEBInputs) -> bool:
    import neb_dynamics.chainhelpers as ch
    fraction_freeze = 0.5


    grad = chain_new.gradients
    springgrads = chain_new.springgradients
    gperps = chain_new.gperps
    if chain_new.parameters.frozen_atom_indices:
        not_frozen_atoms = list(set(list(range(len(grad)))) - set(chain_new.parameters.frozen_atom_indices))
        grad = [g[not_frozen_atoms] for g in grad]
        springgrads = [g[not_frozen_atoms] for g in springgrads]
        gperps = [g[not_frozen_atoms] for g in gperps]


    rms_grad_conv_ind, rms_gperps = _check_rms_grad_converged(
        grad, threshold=parameters.rms_grad_thre*fraction_freeze)

    gperp_conv_ind, gperps = _check_gperps_converged(
        gperps, threshold=parameters.rms_grad_thre*fraction_freeze)

    ts_triplet_gspring = chain_new.ts_triplet_gspring_infnorm
    grad_converged_indices, springgrads = _check_springgrad_converged(springgrads,
                                                                      threshold=parameters.
                                                                      ts_spring_thre*fraction_freeze)

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
    if ind_ts_guess == 0 or ind_ts_guess == len(chain_new)-1:
        if _rich_available:
            _console.print("[yellow]⚠ Warning: TS guess is at the end of the chain. This might indicate a problem with the initial guess.[/yellow]")
        else:
            print("Warning: TS guess is at the end of the chain. This might indicate a problem with the initial guess.")
        ts_guess_grad = 0
    else:
        ts_guess_grad = np.amax(np.abs(gperps[ind_ts_guess]))
    criteria_converged = [
        max(rms_gperps) <= parameters.max_rms_grad_thre,
        sum(rms_gperps)/len(chain_new) <= parameters.rms_grad_thre,
        ts_guess_grad <= parameters.ts_grad_thre,
        ts_triplet_gspring <= parameters.ts_spring_thre,
        max(springgrads) <= parameters.ts_spring_thre,
        barrier_height_converged]

    CRITERIA_NAMES = ["MAX(RMS_GPERP)", "MEAN(RMS_GPERP)",
                      "TS_GRAD", "TS_SPRING", "INFNORM_SPRING", "BARRIER_HEIGHT"]

    # Print convergence criteria as a nice table
    if _rich_available:
        table = Table(title="[bold]Convergence Criteria[/bold]", box=box.ROUNDED)
        table.add_column("Criterion", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Threshold", style="dim")

        for name, converged in zip(CRITERIA_NAMES, criteria_converged):
            status = "[bold green]✓[/bold green]" if converged else "[bold red]✗[/bold red]"
            # Get threshold value for display
            if name == "MAX(RMS_GPERP)":
                thresh = f"{parameters.max_rms_grad_thre}"
            elif name == "MEAN(RMS_GPERP)":
                thresh = f"{parameters.rms_grad_thre}"
            elif name == "TS_GRAD":
                thresh = f"{parameters.ts_grad_thre}"
            elif name == "TS_SPRING":
                thresh = f"{parameters.ts_spring_thre}"
            elif name == "INFNORM_SPRING":
                thresh = f"{parameters.ts_spring_thre}"
            else:
                thresh = f"{parameters.barrier_thre}"
            table.add_row(name, status, thresh)

        _console.print(table)
    else:
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
