"""
this whole module needs to be revamped and integrated with the qcio results objects probably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from neb_dynamics.chain import Chain
from neb_dynamics.nodes.node import Node
from neb_dynamics.nodes.nodehelpers import is_identical
import numpy as np
from neb_dynamics.engines import Engine
from neb_dynamics.nodes.nodehelpers import _is_connectivity_identical, is_identical


@dataclass
class ElemStepResults:
    """
    Object to build report on minimization from elementary step checks.
    """

    is_elem_step: bool
    is_concave: bool
    splitting_criterion: str
    minimization_results: List[Node]
    number_grad_calls: int


@dataclass
class ConcavityResults:
    """
    Stores results on concavity checks (i.e. whether chain has a "dip" that could be\
        a new minimum)
    """

    is_concave: bool
    minimization_results: list[Node]
    number_grad_calls: int

    @property
    def is_not_concave(self):
        return not self.is_concave


@dataclass
class IRCResults:
    """Stores results on (pseudo)IRC checks"""

    found_reactant: Node
    found_product: Node
    number_grad_calls: int


def check_if_elem_step(inp_chain: Chain, engine: Engine) -> ElemStepResults:
    """Calculates whether an input chain is an elementary step.

    Args:
        inp_chain (Chain): input chain to check.

    Returns:
        ElemStepResults: object containing report on chain.
    """
    n_geom_opt_grad_calls = 0
    chain = inp_chain.copy()
    if len(inp_chain) <= 1:
        return ElemStepResults(
            is_elem_step=True,
            is_concave=True,
            splitting_criterion=None,
            minimization_results=None,
            number_grad_calls=0,
        )

    concavity_results = _chain_is_concave(chain=inp_chain, engine=engine)
    n_geom_opt_grad_calls += concavity_results.number_grad_calls

    if concavity_results.is_not_concave:
        return ElemStepResults(
            is_elem_step=False,
            is_concave=concavity_results.is_concave,
            splitting_criterion="minima",
            minimization_results=concavity_results.minimization_results,
            number_grad_calls=n_geom_opt_grad_calls,
        )

    crude_irc_passed, ngc_approx_elem_step = is_approx_elem_step(
        chain=inp_chain, engine=engine
    )
    n_geom_opt_grad_calls += ngc_approx_elem_step

    if crude_irc_passed:
        return ElemStepResults(
            is_elem_step=True,
            is_concave=concavity_results.is_concave,
            splitting_criterion=None,
            minimization_results=[inp_chain[0], inp_chain[-1]],
            number_grad_calls=n_geom_opt_grad_calls,
        )

    pseu_irc_results = pseudo_irc(chain=inp_chain, engine=engine)
    n_geom_opt_grad_calls += pseu_irc_results.number_grad_calls

    found_r = is_identical(
        pseu_irc_results.found_reactant,
        chain[0],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
    )
    found_p = is_identical(
        pseu_irc_results.found_product,
        chain[-1],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
    )
    minimizing_gives_endpoints = found_r and found_p

    elem_step = True if minimizing_gives_endpoints else False

    return ElemStepResults(
        is_elem_step=elem_step,
        is_concave=concavity_results.is_concave,
        splitting_criterion="maxima",
        minimization_results=[
            pseu_irc_results.found_reactant,
            pseu_irc_results.found_product,
        ],
        number_grad_calls=n_geom_opt_grad_calls,
    )


def is_approx_elem_step(
    chain: Chain, engine: Engine, slope_thresh=0.1
) -> Tuple[bool, int]:
    """Will do at most 50 steepest descent steps  on geometries neighboring the transition state guess
    and check whether they are approaching the chain endpoints. If function returns False, the geoms
    will be fully optimized.

    Args:
        chain (Chain): chain to check on
        slope_thresh (float, optional): Steepest descent optimization will stop when the slope
        of the distances of the minimized geometry to the target endpoint is >= threshold.
        Defaults to 0.1.

    Returns:
        (bool, int): whether chain seems to be an elementary step, number grad calls it took to do this check

    """
    if chain.energies_are_monotonic:
        return True, 0

    arg_max = np.argmax(chain.energies)

    try:
        r_passes_opt, r_traj = _converges_to_an_endpoints(
            chain=chain,
            engine=engine,
            node_index=(arg_max - 1),
            direction=-1,
            slope_thresh=slope_thresh,
        )
        p_passes_opt, p_traj = _converges_to_an_endpoints(
            chain=chain,
            engine=engine,
            node_index=(arg_max + 1),
            direction=+1,
            slope_thresh=slope_thresh,
        )
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print(f"Error in geometry optimization: {e}. Pretending this is an elem step.")
        return True, 0
    nodes_have_graph = chain.nodes[0].has_molecular_graph
    # if we have molecular graphs to work with, make sure the connectivities are
    # isomorphic to each other. Otherwise, we will decide only based on distance.
    # (which is bad!!)
    if nodes_have_graph:
        r_passes = r_passes_opt and _is_connectivity_identical(r_traj[-1], chain[0])
        p_passes = p_passes_opt and _is_connectivity_identical(p_traj[-1], chain[-1])
    else:
        r_passes = r_passes_opt
        p_passes = p_passes_opt

    n_grad_calls = len(r_traj) + len(p_traj)
    if r_passes and p_passes:
        return True, n_grad_calls
    else:
        return False, n_grad_calls


def _converges_to_an_endpoints(
    chain, node_index, direction, engine: Engine, slope_thresh: float, max_grad_calls=50
) -> Tuple[bool, List[Node]]:
    """helper function to `is_approx_elem_step`. Actually carries out the minimizations.

    Args:
        chain (_type_): chain with reference geometries.
        node_index (_type_): index of geometry to minimize.
        slope_thresh (float, optional): Threshold for exiting out of minimization early.. Defaults to 0.01.
        direction (int, optional): Direction minimization should be going towards if elem step. -1 refers to
        reactant. +1 refers to product.
        max_grad_calls (int, optional): Maximum number of steepest descent calls until exits out of check.
        Defaults to 50.

    Returns:
        Tuple[bool, List[Node]]: boolean describing whether geometry is minimizing in correct direction, and list of
        nodes containing minimization trajectory.
    """
    done = False
    total_traj = [chain[node_index]]
    while not done:
        try:
            traj = engine.steepest_descent(node=total_traj[-1], max_steps=5)
            total_traj.extend(traj)
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(
                f"Error in geometry optimization: {e}. Pretending this is an elem step."
            )
            return True

        distances = [
            _distances_to_refs(ref1=chain[0], ref2=chain[-1], raw_node=n)
            for n in total_traj
        ]

        slopes_to_ref1 = distances[-1][0] - distances[0][0]
        slopes_to_ref2 = distances[-1][1] - distances[0][1]

        slope1_conv = abs(slopes_to_ref1) / slope_thresh > 1
        slope2_conv = abs(slopes_to_ref2) / slope_thresh > 1

        slopes_to_ref1, slopes_to_ref2

        done = slope1_conv and slope2_conv
        if len(total_traj) - 1 >= max_grad_calls:
            done = True
    if direction == -1:
        return slopes_to_ref1 < 0 and slopes_to_ref2 > 0, total_traj
    elif direction == 1:
        return slopes_to_ref1 > 0 and slopes_to_ref2 < 0, total_traj


def _distances_to_refs(ref1: Node, ref2: Node, raw_node: Node) -> List[float]:
    """
    Computes distances of `raw_node` to `ref1` and `ref2`.
    """
    dist_to_ref1 = np.linalg.norm(raw_node.coords - ref1.coords)
    dist_to_ref2 = np.linalg.norm(raw_node.coords - ref2.coords)
    return [dist_to_ref1, dist_to_ref2]


def _run_geom_opt(node: Node, engine: Engine):
    """
    will run a check on whether the Engine has implemented the
    geometry optimization function. If not, it will just run Steepest
    Descent.
    """
    try:
        opt_traj = engine.compute_geometry_optimization(node)
    except AttributeError:
        opt_traj = engine.steepest_descent(node, max_steps=500, ss=0.001)

    return opt_traj


def _chain_is_concave(chain: Chain, engine: Engine) -> ConcavityResults:
    """
    will assess+categorize the presence of minima on the chain.
    """
    import neb_dynamics.chainhelpers as ch

    n_grad_calls = 0
    ind_minima = ch._get_ind_minima(chain=chain)
    minima_present = len(ind_minima) != 0
    opt_results = []
    if minima_present:
        minimas_is_r_or_p = []
        try:
            for i in ind_minima:
                opt_traj = _run_geom_opt(chain[i], engine=engine)
                n_grad_calls += len(opt_traj)
                opt = opt_traj[-1]
                opt_results.append(opt)
                is_r = is_identical(
                    opt,
                    chain[0],
                    fragment_rmsd_cutoff=chain.parameters.node_rms_thre,
                    kcal_mol_cutoff=chain.parameters.node_ene_thre,
                )
                is_p = is_identical(
                    opt,
                    chain[-1],
                    fragment_rmsd_cutoff=chain.parameters.node_rms_thre,
                    kcal_mol_cutoff=chain.parameters.node_ene_thre,
                )
                minimas_is_r_or_p.append(is_r or is_p)
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(
                f"Error in geometry optimization: {e}. Pretending this is an elem step."
            )

            return ConcavityResults(
                is_concave=True,
                minimization_results=[chain[0], chain[-1]],
                number_grad_calls=n_grad_calls,
            )

        if all(minimas_is_r_or_p):
            return ConcavityResults(
                is_concave=True,
                minimization_results=[chain[0], chain[-1]],
                number_grad_calls=n_grad_calls,
            )
        else:
            return ConcavityResults(
                is_concave=False,
                minimization_results=opt_results,
                number_grad_calls=n_grad_calls,
            )
    else:
        return ConcavityResults(
            is_concave=True,
            minimization_results=[chain[0], chain[-1]],
            number_grad_calls=n_grad_calls,
        )


def pseudo_irc(chain: Chain, engine: Engine):
    n_grad_calls = 0
    arg_max = np.argmax(chain.energies)

    if arg_max == len(chain) - 1 or arg_max == 0:  # monotonically changing function,
        return IRCResults(
            found_reactant=chain[0],
            found_product=chain[len(chain) - 1],
            number_grad_calls=n_grad_calls,
        )

    try:

        candidate_r = chain[arg_max - 1]
        candidate_p = chain[arg_max + 1]

        r_traj = _run_geom_opt(candidate_r, engine=engine)
        r = r_traj[-1]
        n_grad_calls += len(r_traj)

        p_traj = _run_geom_opt(candidate_p, engine=engine)
        n_grad_calls += len(p_traj)
        p = p_traj[-1]

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print(f"Error in geometry optimization: {e}. Pretending this is an elem step.")
        return IRCResults(
            found_reactant=chain[0],
            found_product=chain[len(chain) - 1],
            number_grad_calls=n_grad_calls,
        )

    return IRCResults(found_reactant=r, found_product=p, number_grad_calls=n_grad_calls)
