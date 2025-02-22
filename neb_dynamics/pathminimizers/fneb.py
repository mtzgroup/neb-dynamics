from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from neb_dynamics.geodesic_interpolation.geodesic import Geodesic

import neb_dynamics.chainhelpers as ch
from neb_dynamics.chain import Chain
from neb_dynamics.engines.engine import Engine
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.helper_functions import RMSD, get_maxene_node
from neb_dynamics.pathminimizers.pathminimizer import PathMinimizer
from neb_dynamics.optimizers.optimizer import Optimizer
from neb_dynamics.elementarystep import check_if_elem_step, ElemStepResults
from neb_dynamics.inputs import RunInputs

import traceback

import sys

GI_NIMG = 30
MINIMGS = 3


DISTANCE_METRICS = ["GEODESIC", "RMSD", "LINEAR"]
IS_ELEM_STEP = ElemStepResults(
    is_elem_step=True,
    is_concave=True,
    splitting_criterion=None,
    minimization_results=None,
    number_grad_calls=0,)


KCAL_MOL_CUTOFF = 5.0
GRAD_TOL = 0.01  # Hartree/Bohr
TANGERR_TOL = 0.3  # Bohr
MAXTANG_NTRIES = 5


@dataclass
class FreezingNEB(PathMinimizer):
    initial_chain: Chain
    engine: Engine
    optimizer: Optimizer
    chain_trajectory: list[Chain] = field(default_factory=list)
    parameters: SimpleNamespace = None

    def __post_init__(self):

        if self.parameters is None:
            ri = RunInputs(path_min_method='FNEB')
            self.parameters = ri.path_min_inputs

        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0

    def _distance_function(self, node1: StructureNode, node2: StructureNode):
        if self.parameters.distance_metric.upper() == "RMSD":
            return RMSD(node1.coords, node2.coords)[0]
        elif self.parameters.distance_metric.upper() == "GEODESIC":
            return ch.calculate_geodesic_distance(node1, node2)
        elif self.parameters.distance_metric.upper() == "LINEAR":
            return np.linalg.norm(node1.coords - node2.coords)
        elif self.parameters.distance_metric.upper() == "XTBGI":
            return abs(ch.calculate_geodesic_xtb_barrier(node1, node2))
            # xtbeng = QCOPEngine()
            # enes = xtbeng.compute_energies([node1, node2])
            # print((enes[1] - enes[0])*627.5)
            # return abs(enes[1] - enes[0])*627.5
        else:
            raise ValueError(
                f"Invalid distance metric: {self.parameters.distance_metric}. Use one of {DISTANCE_METRICS}"
            )

    def optimize_chain(
        self,
    ):
        """
        will run freezing string on chain.
        dr --> the requested path resolution in Bohr
        """
        # print(f"START: {self.grad_calls_made}")
        chain = self.initial_chain.copy()
        ch._reset_cache(chain=chain)
        self.optimizer.g_old = None
        chain.nodes = [
            chain.nodes[0],
            chain.nodes[-1],
        ]  # need to make sure I only use the endpoints
        self.engine.compute_energies(chain)
        self.grad_calls_made += 2
        chain.nodes[0].converged = True
        chain.nodes[1].converged = True
        self.chain_trajectory = [chain]

        converged = False
        nsteps = 0
        last_grown_ind = 0

        while not converged and nsteps < self.parameters.max_grow_iter:
            print(f"STEP{nsteps}")
            # grow nodes
            print("\tgrowing nodes")

            # grown_chain = self.grow_nodes(
            #     chain, dr=self.parameters.path_resolution
            # )
            grown_chain, node_ind, ind_ts_gi = self.grow_nodes_maxene(
                chain, last_grown_ind=last_grown_ind, nimg=GI_NIMG)

            no_growth = len(grown_chain) == len(chain)
            # no_barrier_change = abs(grown_chain.get_eA_chain(
            # ) - chain.get_eA_chain()) < KCAL_MOL_CUTOFF
            no_barrier_change = abs(grown_chain.get_eA_chain(
            ) - chain.get_eA_chain()) < self.parameters.barrier_thre

            if no_growth or no_barrier_change:
                print(
                    f"Converged!\n\tNo growth?: {no_growth}\n\tNo barrier change?: {no_barrier_change}")
                converged = True
                break
            else:

                self.chain_trajectory.append(grown_chain.copy())

                # minimize nodes
                print("\tminimizing nodes")
                # min_chain = self.minimize_nodes(
                #     chain=grown_chain, d0=d0, min_two_nodes=add_two_nodes
                # )
                min_chain = self.minimize_node_maxene(
                    chain=grown_chain, node_ind=node_ind,
                    ind_ts_gi=ind_ts_gi)
                self.chain_trajectory.append(min_chain.copy())
                last_grown_ind = node_ind
                print(f"LAST GROWN INDEX WAS: {last_grown_ind}")

                # check convergence
                # print("\tchecking convergence")
                # converged, inner_bead_distance = self.chain_converged(min_chain)
                # dr = self.parameters.path_resolution
                # dr = self._distance_function(
                #     min_chain[0], min_chain[-1]) / self.parameters.min_images

                # smoother = run_geodesic_get_smoother(
                #     input_object=[min_chain[0].symbols, [node.coords for node in min_chain]])
                # dr = smoother.length / self.parameters.min_images
                # print(f"---> current dr: {dr}")
                # if inner_bead_distance <= self.parameters.early_stop_scaling * dr:
                #     new_params = SimpleNamespace(**self.parameters.__dict__)
                #     new_params.early_stop_scaling = 0.0
                #     self.parameters = new_params

                #     if self.parameters.do_elem_step_checks:
                #         elem_step_results = check_if_elem_step(
                #             inp_chain=min_chain, engine=self.engine
                #         )
                #     else:
                #         print("Not doing Elem step check. Pretending elem step.")
                #         elem_step_results = IS_ELEM_STEP
                #     if not elem_step_results.is_elem_step:
                #         print("Stopping early because chain is multistep.")
                #         self.optimized = min_chain
                #         return elem_step_results
                chain = min_chain.copy()
                nsteps += 1

        self.optimized = self.chain_trajectory[-1]
        print(f"Converged? {converged}")
        if self.parameters.do_elem_step_checks:
            short_chain = Chain.model_validate(
                {"nodes": [chain[0], chain.get_ts_node(), chain[-1]], "parameters": chain.parameters})
            elem_step_results = check_if_elem_step(
                short_chain, engine=self.engine)
            # elem_step_results = check_if_elem_step(chain, engine=self.engine)
            self.geom_grad_calls_made += elem_step_results.number_grad_calls
        else:
            elem_step_results = IS_ELEM_STEP
        return elem_step_results

    def _min_node_maxene(
        self,
        raw_chain: Chain,
        max_iter: int,
        ind_ts_gi: int,
        ind_node: int = 0,

    ):
        """
        ind_node: index of node to minimize
        """
        converged = False
        nsteps = 1  # we already made one gradient call when growing the node.

        nimg1 = ind_ts_gi + 1
        nimg2 = GI_NIMG - ind_ts_gi

        while not converged:
            if nsteps >= max_iter:
                converged = True
            try:

                node_to_opt = raw_chain[ind_node]
                print([node.converged for node in raw_chain])
                assert not node_to_opt.converged, "Trying to minimize a node that was already converged!"
                sys.stdout.flush()
                prev_iter_ene = node_to_opt.energy
                if self.parameters.verbosity > 1:
                    print(f"{prev_iter_ene=}")

                # output = ch.calculate_geodesic_tangent(
                #     list_of_nodes=raw_chain, ref_node_ind=ind_node, nimages=nimg1, nimages2=nimg2,
                #     tangent_err=self.parameters.tang_err,
                #     max_ntries=self.parameters.max_tang_tries)

                output = ch.calculate_geodesic_tangent(
                    list_of_nodes=[
                        raw_chain[ind_node-1], raw_chain[ind_node], raw_chain[ind_node+1]], ref_node_ind=1,
                    nimages=nimg1, nimages2=nimg2,
                    tangent_err=self.parameters.tang_err,
                    max_ntries=self.parameters.max_tang_tries)

                prev_node, curr_node, next_node = output[0]
                d1, d2 = output[1]
                # print(f"{d1=} || {d2=}")

                dtot = d1+d2
                nimg1 = max(int((d1/dtot)*GI_NIMG), MINIMGS)
                nimg2 = max(GI_NIMG - nimg1, MINIMGS)

                tangent = ((node_to_opt.coords - prev_node.coords) +
                           (next_node.coords - node_to_opt.coords)) / 2

                unit_tan = tangent / np.linalg.norm(tangent)

                # grad1 = self.engine.compute_gradients([node_to_opt])
                grad1 = node_to_opt.gradient

                gperp1 = ch.get_nudged_pe_grad(
                    unit_tangent=unit_tan, gradient=grad1)

                gperp1[0, :] = 0  # this atom cannot move
                gperp1[1, :2] = 0  # this atom can only move in a line
                gperp1[2, :1] = 0  # this atom can only move in a plane

                grad_inf_norm = np.amax(abs(gperp1))
                print("MIN: ", grad_inf_norm)
                # if grad_inf_norm <= GRAD_TOL:
                if grad_inf_norm <= self.parameters.grad_tol:
                    converged = True
                    break

                # add a spring force
                kconst = raw_chain.parameters.k

                # this uses the RMSD distance between neighboring nodes
                # fspring = kconst * np.linalg.norm(
                #     raw_chain[ind_node+1].coords -
                #     node_to_opt.coords
                # ) - kconst * np.linalg.norm(node_to_opt.coords - raw_chain[ind_node-1].coords)

                # this uses the geodesic distance between neighboring nodes
                fspring = kconst * d2 - kconst * d1

                gperp1 -= fspring*unit_tan
                print(f"***{fspring=}")

                out_chain = self.optimizer.optimize_step(
                    chain=Chain.model_validate({"nodes": [node_to_opt]}),
                    chain_gradients=np.array([gperp1]))

                # direction = -1 * gperp1 * ss
                # direction_scaled = direction.copy()
                # for i_atom, vector in enumerate(direction):
                #     length = np.linalg.norm(vector)
                #     if length > max_atom_displacement:
                #         if len(direction.shape) > 1:
                #             direction_scaled[i_atom, :] = (
                #                 vector / length
                #             ) * max_atom_displacement
                #         else:
                #             direction_scaled[i_atom] = (
                #                 vector / length
                #             ) * max_atom_displacement

                # new_node1_coords = node_to_opt.coords + direction_scaled
                # new_node1 = node_to_opt.update_coords(new_node1_coords)
                new_node1 = out_chain.nodes[0]
                self.engine.compute_energies([new_node1])
                self.grad_calls_made += 1
                # print(f"PROG: {self.grad_calls_made}")

                raw_chain.nodes[ind_node] = new_node1
                self.chain_trajectory.append(raw_chain.copy())
                nsteps += 1

            except Exception:
                print(traceback.format_exc())
                return raw_chain
        print(f"\t converged in {nsteps}")
        raw_chain.nodes[ind_node].converged = True
        return raw_chain

    def minimize_node_maxene(self, chain: Chain, node_ind: int, ind_ts_gi: int):
        raw_chain = chain.copy()

        chain_opt = self._min_node_maxene(
            raw_chain,
            ind_node=node_ind,
            max_iter=self.parameters.max_min_iter,
            ind_ts_gi=ind_ts_gi
        )
        self.engine.g_old = None  # reset the conjugate gradient memory
        return chain_opt

    def _get_innermost_nodes_inds(self, chain: Chain):
        if len(chain) == 2:
            return 0, 1

        ind_node2 = int(len(chain) / 2)
        ind_node1 = ind_node2 - 1

        if len(chain) % 2 != 0:  # chain is an odd length
            ind_node3 = ind_node2 + 1
            d02 = self._distance_function(chain[ind_node3], chain[ind_node2])
            d12 = self._distance_function(chain[ind_node1], chain[ind_node2])
            if d02 > d12:
                return ind_node2, ind_node3  # because now the 'rightmost' node is ind_node3

        return ind_node1, ind_node2

    def _get_innermost_nodes(self, chain: Chain):
        """
        returns a chain object with the two innermost nodes
        """
        ind_node1, ind_node2 = self._get_innermost_nodes_inds(chain)
        out_chain = chain.copy()
        out_chain.nodes = [chain[ind_node1], chain[ind_node2]]

        return out_chain

    def grow_nodes_maxene(self, chain: Chain, last_grown_ind: int = 0, nimg: int = 20):
        """
        will return a chain with 1 new node which is the highest energy interpolated
        node between the last node added and its nearest neighbors.

        If no node is added, will return the input chain.
        """
        if last_grown_ind == 0:  # initial case
            gi = ch.run_geodesic([chain[0], chain[-1]], nimages=nimg, nudge=0)
            # eng = RunInputs(program='xtb').engine
            # eng.compute_energies(gi)
            # deleteme

            grown_chain = chain.copy()
            maxnode_data = get_maxene_node(gi, engine=self.engine)
            ind_max = maxnode_data['index']
            self.grad_calls_made += maxnode_data['grad_calls']

            if ind_max == 0 or ind_max == len(chain)-1:
                print("No TS found between endpoints. Returning input chain.")
                return chain, 1, ind_max

            # node = gi[ind_max]
            node = maxnode_data['node']
            # node._cached_energy = None
            # node._cached_gradient = None
            # self.engine.compute_energies([node])
            # self.grad_calls_made += 1
            grown_chain.nodes = [chain[0], node, chain[-1]]
            new_ind = 1

        else:
            gi1 = ch.run_geodesic([chain[last_grown_ind-1],
                                  chain[last_grown_ind]],
                                  nimages=nimg, nudge=0)
            gi2 = ch.run_geodesic([chain[last_grown_ind],
                                  chain[last_grown_ind+1]],
                                  nimages=nimg, nudge=0)

            # eng = RunInputs(program='xtb').engine
            # eng.compute_energies(gi1)
            # eng.compute_energies(gi2)

            # deltaEs = [gi1.get_eA_chain(), gi2.get_eA_chain()]
            maxnode1_data = get_maxene_node(gi1, engine=self.engine)
            maxnode2_data = get_maxene_node(gi2, engine=self.engine)

            self.grad_calls_made += maxnode1_data['grad_calls']
            self.grad_calls_made += maxnode2_data['grad_calls']

            deltaEs = [(maxnode1_data['node'].energy -
                        chain[last_grown_ind].energy)*627.5,
                       (maxnode2_data['node'].energy -
                        chain[last_grown_ind].energy)*627.5
                       ]

            # ind_max_left = gi1.energies.argmax()
            # ind_max_right = gi2.energies.argmax()
            ind_max_left = maxnode1_data['index']
            ind_max_right = maxnode2_data['index']
            # if all([dE < KCAL_MOL_CUTOFF for dE in deltaEs]):

            if all([dE < self.parameters.barrier_thre for dE in deltaEs]):
                left_side_converged = right_side_converged = True
                ind_max = ind_max_left  # this is a random choice

            else:

                left_side_converged = (
                    ind_max_left == len(gi1)-1 or ind_max_left == 0
                )

                right_side_converged = (
                    ind_max_right == 0 or ind_max_right == 0
                )

            if not left_side_converged and not right_side_converged:
                print("Two potential directions found. Choosing highest ascent")
                # left = gi1.energies.max()
                # right = gi2.energies.max()
                left = maxnode1_data['node'].energy
                right = maxnode2_data['node'].energy
                if left > right:
                    right_side_converged = True
                    ind_max = ind_max_left
                else:
                    left_side_converged = True
                    ind_max = ind_max_right

            if left_side_converged and right_side_converged:
                print("TS guess found. Returning input chain.")
                ind_max = ind_max_left  # arbitrary choice
                return chain, last_grown_ind, ind_max

            elif left_side_converged and not right_side_converged:
                print("Growing rightwards...")
                grown_chain = chain.copy()
                # node = gi2[gi2.energies.argmax()]
                node = maxnode2_data['node']
                # node._cached_energy = None
                # node._cached_gradient = None
                # self.engine.compute_energies([node])
                # self.grad_calls_made += 1

                grown_chain.nodes.insert(
                    last_grown_ind+1, node)

                new_ind = last_grown_ind+1
                ind_max = ind_max_right

            elif not left_side_converged and right_side_converged:
                print("Growing leftwards...")
                grown_chain = chain.copy()
                # node = gi1[gi1.energies.argmax()]
                node = maxnode1_data['node']
                # node._cached_energy = None
                # node._cached_gradient = None
                # self.engine.compute_energies([node])
                # self.grad_calls_made += 1

                grown_chain.nodes.insert(
                    last_grown_ind, node)

                new_ind = last_grown_ind
                ind_max = ind_max_left

        return grown_chain, new_ind, ind_max

    def _get_closest_node_ind(self, smoother_obj, reference):
        smallest_dist = 1e10
        ind = None
        for i, geom in enumerate(smoother_obj.path):
            dist, _ = RMSD(geom, reference)
            if dist < smallest_dist:
                smallest_dist = dist
                ind = i
        # print(smallest_dist)
        return ind

    def _select_node_at_dist(
        self,
        reference_node: StructureNode,
        chain: Chain,
        dist: float,
        direction: int,
        smoother: Geodesic = None,
        dist_err: float = 0.1,
    ):
        """
        will iterate through chain and select the node that is 'dist' away up to 'dist_err'

        dist --> the requested distance where you want the nodes
        direction --> whether to go forward (1) or backwards (-1). if forward, will pick the requested node
                    to the first node. if backwards, will pick the requested node to the last node

        """
        reference_index = self._get_closest_node_ind(
            smoother_obj=smoother, reference=reference_node.coords)
        # input_chain = chain.copy()
        if direction == -1:
            subset_chain = chain.nodes[:(reference_index+1)]
            enum_start = 1
        else:
            subset_chain = chain.nodes[reference_index:]
            enum_start = 0
        # print(f"REFERENCE INDEX: {reference_index} || {len(subset_chain)}")

        # start_node = input_chain[0]
        best_node = None
        best_dist_err = 10000.0

        eng = QCOPEngine()
        enes = eng.compute_energies(subset_chain)
        enes = enes - enes[0]

        for i, _ in enumerate(subset_chain, start=enum_start):
            if direction == -1:
                start = (reference_index - i) + 1
                end = (reference_index) + 1
                # node = subset_chain[start-1]
                node = chain[start-1]
                # subnode = subset_chain[i-1]

            elif direction == 1:
                start = reference_index + 1
                end = (reference_index + i) + 1
                # node = subset_chain[end - 1]

                node = chain[end-1]
                # subnode = subset_chain[i]

            if start == end or start == 0 or end == 0:
                continue

            if self.parameters.distance_metric.upper() == "GEODESIC":
                smoother.compute_disps(start=start, end=end)
                curr_dist = smoother.length
            # elif self.parameters.distance_metric.upper() == "XTBGI":
            #     curr_dist = abs(subnode.energy -
            #                     subset_chain[reference_index].energy)*627.5
            else:
                curr_dist = self._distance_function(
                    node1=reference_node, node2=node)
            curr_dist_err = np.abs(curr_dist - dist)
            if self.parameters.verbosity > 1:
                print(f"\t{curr_dist_err=} vs {dist_err=} || {direction=}")

            if curr_dist_err <= dist_err and curr_dist_err < best_dist_err:
                if self.parameters.distance_metric.upper() == 'XTBGI' and best_node is not None:
                    new_dist, _ = RMSD(node.coords, reference_node.coords)
                    best_dist, _ = RMSD(
                        best_node.coords, reference_node.coords)
                    if best_dist < new_dist:
                        continue

                best_node = node
                best_dist_err = curr_dist_err

                break

        return best_node

    def chain_converged(self, chain: Chain):
        dr = self.parameters.path_resolution
        node1, node2 = self._get_innermost_nodes(chain)
        dist = self._distance_function(node1, node2)
        print(f"distance between innermost nodes {dist}")
        if dist <= dr + (self.parameters.dist_err * self.parameters.path_resolution):
            result = True
        else:
            result = False

        return result, dist
