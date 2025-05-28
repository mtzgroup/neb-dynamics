from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

# from neb_dynamics.geodesic_interpolation.geodesic import Geodesic
from neb_dynamics.geodesic_interpolation2.morsegeodesic import MorseGeodesic

import neb_dynamics.chainhelpers as ch
from neb_dynamics.chain import Chain
from neb_dynamics.engines.engine import Engine
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.helper_functions import RMSD, get_maxene_node
from neb_dynamics.pathminimizers.pathminimizer import PathMinimizer
from neb_dynamics.optimizers.optimizer import Optimizer
from neb_dynamics.elementarystep import check_if_elem_step, ElemStepResults
from neb_dynamics.inputs import RunInputs

import traceback

import sys

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
    gi_inputs: SimpleNamespace = None

    def __post_init__(self):

        if self.parameters is None:
            ri = RunInputs(path_min_method='FNEB')
            self.parameters = ri.path_min_inputs
        if self.gi_inputs is None:
            ri = RunInputs(path_min_method='FNEB')
            self.gi_inputs = ri.gi_inputs

        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0

    def _distance_function(self, node1: StructureNode, node2: StructureNode):
        if self.parameters.distance_metric.upper() == "RMSD":
            return RMSD(node1.coords, node2.coords)[0]
        elif self.parameters.distance_metric.upper() == "GEODESIC":
            return min([ch.calculate_geodesic_distance(node1, node2, nudge=0.01, nimages=self.gi_inputs.nimages) for i in range(5)])
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
        np.random.seed(0)
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

        d0 = self._distance_function(chain[0], chain[1])
        # +1 so that if only one image is requested, node is placed at 50% of path
        dr = d0 / (self.parameters.min_images+1)

        converged = False
        nsteps = 0
        last_grown_ind = 0

        while not converged and nsteps < self.parameters.max_grow_iter:
            print(f"STEP{nsteps}")
            # grow nodes
            print("\tgrowing nodes")

            if self.parameters.todd_way:
                print(f"gonna grow at {last_grown_ind}")
                grown_chain, tangents, idx_grown, dr = self.grow_nodes(
                    chain, dr=dr, indices=(last_grown_ind, last_grown_ind+1)
                )
                self.chain_trajectory.append(grown_chain.copy())

                # this section will stop opt if grown nodes are too low in energy
                node0_done = grown_chain[idx_grown[0]
                                         ].energy < grown_chain.get_ts_node().energy
                node1_done = True
                if idx_grown[1] is not None:
                    node1_done = grown_chain[idx_grown[1]
                                             ].energy < grown_chain.get_ts_node().energy

                if node0_done and node1_done:
                    if nsteps == 0:  # if this is the first step, we need to make sure that the grown nodes are higher than both endpoints
                        result = self.grow_nodes_maxene(
                            chain, last_grown_ind=last_grown_ind, nimg=self.gi_inputs.nimages, nudge=self.gi_inputs.nudge)
                        grown_chain, node_ind, ind_ts_gi = result[0], result[1], result[2]
                        print(
                            "Initial grown node was lower than endpoints. Using max-ene growth.")
                        self.chain_trajectory.append(grown_chain.copy())
                    else:
                        print(
                            "Grown nodes are lower in energy than TS guess. Stopping optimization.")
                        converged = True
                        break

                min_chain = self.minimize_nodes(
                    chain=grown_chain, node_tangents=tangents, dr=dr, idx_grown=idx_grown
                )
                # check convergence
                print("\tchecking convergence")
                converged, last_grown_ind = self.chain_converged(
                    min_chain, dr, indices=idx_grown)

                # last_grown_ind = idx_grown[0]
                print(f"LAST GROWN INDEX WAS: {last_grown_ind}")

                self.optimizer.g_old = None
            else:
                result = self.grow_nodes_maxene(
                    chain, last_grown_ind=last_grown_ind, nimg=self.gi_inputs.nimages, nudge=self.gi_inputs.nudge)
                grown_chain, node_ind, ind_ts_gi = result[0], result[1], result[2]
                # if self.parameters.use_dqdr:
                #     smoother = result[3]
                # else:
                #     smoother = None

                smoother = None

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
                    min_chain = self.minimize_node_maxene(
                        chain=grown_chain, node_ind=node_ind,
                        ind_ts_gi=ind_ts_gi, smoother=smoother)

                    self.chain_trajectory.append(min_chain.copy())
                    last_grown_ind = node_ind
                    print(f"LAST GROWN INDEX WAS: {last_grown_ind}")

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
        ind_ts_gi: int,
        ind_node: int = 0,
        smoother: MorseGeodesic = None

    ):
        """
        ind_node: index of node to minimize
        """
        converged = False
        max_iter = self.parameters.max_min_iter
        nsteps = 1  # we already made one gradient call when growing the node.

        nimg1 = ind_ts_gi + 1
        nimg2 = self.gi_inputs.nimages - ind_ts_gi
        fwd_tang_old = None
        back_tang_old = None

        init_d1 = self._distance_function(
            raw_chain[ind_node], raw_chain[ind_node-1])
        init_d2 = self._distance_function(
            raw_chain[ind_node], raw_chain[ind_node+1])

        gi1, smoother1 = ch.run_geodesic(
            [raw_chain[ind_node], raw_chain[ind_node-1]], nimages=nimg1, return_smoother=True)

        gi2, smoother2 = ch.run_geodesic(
            [raw_chain[ind_node], raw_chain[ind_node+1]], nimages=nimg2, return_smoother=True)

        d1, d2 = smoother1.length, smoother2.length
        dtot = d1+d2

        # smoother1._compute_disps(start=1, end=2)
        # sm1_len = smoother1.length
        sm1_len = sum(smoother1.segment_lengths[:2])
        d1_neighbor = sm1_len

        # smoother2._compute_disps(start=1, end=2)
        # sm2_len = smoother2.length
        sm2_len = sum(smoother2.segment_lengths[:2])
        d2_neighbor = sm2_len

        if self.parameters.tangent == 'geodesic':
            print("nimg1: ", nimg1, ' nimg2: ', nimg2)

            nimg1 = max(int((d1/dtot)*self.gi_inputs.nimages), MINIMGS)
            nimg2 = max(self.gi_inputs.nimages - nimg1, MINIMGS)

            fwd_tang = gi2[1].coords.flatten() - \
                gi2[0].coords.flatten()
            fwd_tang /= np.linalg.norm(fwd_tang)

            back_tang = gi1[1].coords.flatten() - \
                gi1[0].coords.flatten()
            back_tang /= np.linalg.norm(back_tang)
            back_tang *= -1  # we constructed this geodesic backwards, so need to flip it

        else:
            # linear tangent
            back_tang = raw_chain[ind_node].coords.flatten(
            ) - raw_chain[ind_node-1].coords.flatten()

            back_tang /= np.linalg.norm(back_tang)
            fwd_tang = raw_chain[ind_node+1].coords.flatten() - \
                raw_chain[ind_node].coords.flatten()
            fwd_tang /= np.linalg.norm(fwd_tang)

        # mix tangents
        if fwd_tang_old is not None and back_tang_old is not None:
            print("mixing tangents with alpha: ",
                  self.parameters.tangent_alpha)
            fwd_tang = self.parameters.tangent_alpha*fwd_tang + \
                (1-self.parameters.tangent_alpha)*fwd_tang_old
            back_tang = self.parameters.tangent_alpha*back_tang + \
                (1-self.parameters.tangent_alpha)*back_tang_old

        while not converged:
            curr_d1 = self._distance_function(
                raw_chain[ind_node], raw_chain[ind_node-1])
            curr_d2 = self._distance_function(raw_chain[ind_node],
                                              raw_chain[ind_node+1])
            print(
                f"Current d1: {curr_d1} || Current d2: {curr_d2} || {init_d1=} || {init_d2=}")
            if curr_d1 <= 0.75*init_d1 or curr_d2 <= 0.75*init_d2:
                # if curr_d1 <= 0.5*init_d1 or curr_d2 <= 0.5*init_d2:
                print("Node fell more than 25% to one direction. Stopping minimization.")
                converged = True
                break

            if nsteps >= max_iter:
                converged = True
                break
            try:

                node_to_opt = raw_chain[ind_node]
                # grad1 = node_to_opt.gradient
                # should already be cached, doing this so it spits out the biased gradienet
                # in cases where it should be biased
                grad1 = self.engine.compute_gradients([node_to_opt])[0]
                print([node.converged for node in raw_chain])
                assert not node_to_opt.converged, "Trying to minimize a node that was already converged!"
                sys.stdout.flush()
                prev_iter_ene = node_to_opt.energy
                # gi1, smoother1 = ch.run_geodesic(
                #     [raw_chain[ind_node], raw_chain[ind_node-1]], nimages=nimg1, return_smoother=True)

                # gi2, smoother2 = ch.run_geodesic(
                #     [raw_chain[ind_node], raw_chain[ind_node+1]], nimages=nimg2, return_smoother=True)

                # d1, d2 = smoother1.length, smoother2.length
                # dtot = d1+d2

                # smoother1.compute_disps(start=1, end=2)
                # d1_neighbor = smoother1.length

                # smoother2.compute_disps(start=1, end=2)
                # d2_neighbor = smoother2.length

                if self.parameters.verbosity > 1:
                    print(f"{prev_iter_ene=}")
                # if self.parameters.tangent == 'geodesic':
                #     print("nimg1: ", nimg1, ' nimg2: ', nimg2)

                #     nimg1 = max(int((d1/dtot)*self.gi_inputs.nimages), MINIMGS)
                #     nimg2 = max(self.gi_inputs.nimages - nimg1, MINIMGS)

                #     fwd_tang = gi2[1].coords.flatten() - \
                #         gi2[0].coords.flatten()
                #     fwd_tang /= np.linalg.norm(fwd_tang)

                #     back_tang = gi1[1].coords.flatten() - \
                #         gi1[0].coords.flatten()
                #     back_tang /= np.linalg.norm(back_tang)
                #     back_tang *= -1  # we constructed this geodesic backwards, so need to flip it

                # else:
                #     # linear tangent
                #     back_tang = raw_chain[ind_node].coords.flatten(
                #     ) - raw_chain[ind_node-1].coords.flatten()

                #     back_tang /= np.linalg.norm(back_tang)
                #     fwd_tang = raw_chain[ind_node+1].coords.flatten() - \
                #         raw_chain[ind_node].coords.flatten()
                #     fwd_tang /= np.linalg.norm(fwd_tang)

                # # mix tangents
                # if fwd_tang_old is not None and back_tang_old is not None:
                #     print("mixing tangents with alpha: ",
                #           self.parameters.tangent_alpha)
                #     fwd_tang = self.parameters.tangent_alpha*fwd_tang + \
                #         (1-self.parameters.tangent_alpha)*fwd_tang_old
                #     back_tang = self.parameters.tangent_alpha*back_tang + \
                #         (1-self.parameters.tangent_alpha)*back_tang_old

                # for i in range(MAX_MIN_STEPS):
                grad = grad1.copy().flatten()
                tang = (back_tang + fwd_tang)/2.
                proj_tang = np.dot(grad, tang)
                proj_grad_tang = proj_tang*tang
                # proj_back = np.dot(grad, back_tang)

                # proj_grad_back = proj_back*back_tang
                # gperp_internal = grad - proj_grad_back

                # proj_fwd = np.dot(gperp_internal, fwd_tang)
                # proj_grad_fwd = proj_fwd*fwd_tang

                # gperp_internal = gperp_internal - proj_grad_fwd

                # grad_final = gperp_internal + 2.0 * proj_grad_fwd \
                #     + 2.0*proj_grad_back

                # gperp1 = grad_final.reshape(grad1.shape)
                gperp_internal = grad - proj_grad_tang
                gperp1 = gperp_internal.reshape(grad1.shape)

                gperp1 = gperp1-gperp1[0, :]

                # if grad_inf_norm <= GRAD_TOL:
                grad_inf_norm = np.amax(abs(gperp1))
                print("MIN: ", grad_inf_norm)
                if grad_inf_norm <= self.parameters.grad_tol:
                    converged = True
                    break

                # add a spring force
                kconst = raw_chain.parameters.k * \
                    len(gperp1)  # scaled by number of atoms

                # this uses the geodesic distance between neighboring nodes
                # print(f"{d1=} {d2=}")
                # if d1 < d2:
                #     print("\t moving to d2, along fwd tang")
                #     grad_spring = -(kconst * d2)*fwd_tang
                # else:
                #     print("\t moving to d1, along back tang")
                #     grad_spring = -(kconst * d1)*(back_tang)

                print(f"{d1_neighbor=} {d2_neighbor=}")
                if d1_neighbor < d2_neighbor:
                    print("\t moving to d1, along back tang")
                    grad_spring = -(kconst * d1_neighbor)*(back_tang)
                else:
                    print("\t moving to d2, along fwd tang")
                    grad_spring = -(kconst * d2_neighbor)*fwd_tang

                grad_spring = grad_spring.reshape(gperp1.shape)
                gperp1 += grad_spring
                # gperp1 -= fspring*unit_tan
                print(f"***{np.linalg.norm(grad_spring)=}")

                out_chain = self.optimizer.optimize_step(
                    chain=Chain.model_validate({"nodes": [node_to_opt]}),
                    chain_gradients=np.array([gperp1]))

                new_node1 = out_chain.nodes[0]
                self.engine.compute_energies([new_node1])
                self.grad_calls_made += 1

                raw_chain.nodes[ind_node] = new_node1
                self.chain_trajectory.append(raw_chain.copy())
                fwd_tang_old = fwd_tang
                back_tang_old = back_tang
                nsteps += 1

            except Exception:
                print(traceback.format_exc())
                return raw_chain
        print(f"\t converged in {nsteps}")
        raw_chain.nodes[ind_node].converged = True
        return raw_chain

    def minimize_nodes(self, chain: Chain, node_tangents: list, dr, idx_grown: tuple):
        raw_chain = chain.copy()
        idx1, idx2 = idx_grown
        # return raw_chain
        chain_opt1 = self._min_node(
            raw_chain,
            tangent=node_tangents[0],
            ind_node=idx1,
            dr=dr

        )
        if idx2 is not None:
            chain_opt2 = self._min_node(
                chain_opt1,
                tangent=node_tangents[1],
                ind_node=idx2,
                dr=dr

            )
        else:
            chain_opt2 = chain_opt1

        return chain_opt2

    def _min_node(
        self,
        raw_chain: Chain,
        tangent: np.array,
        dr: float,
        ind_node: int,
    ):
        """
        ind_node: index of the node to minimze. 0 if you want the optimize the leftmost inner node. 1
        if you want the rightmost inner node.
        """
        # node1_ind, node2_ind = self._get_innermost_nodes_inds(raw_chain)
        node1_ind, node2_ind = ind_node, ind_node+1
        node1 = raw_chain[node1_ind]
        node2 = raw_chain[node2_ind]
        converged = False
        max_iter = self.parameters.max_min_iter
        nsteps = 1  # we already made one gradient call when growing the node.

        # init_d1d2 = self._distance_function(node1, node2)
        init_d1 = self._distance_function(
            raw_chain[ind_node], raw_chain[ind_node-1])
        init_d2 = self._distance_function(raw_chain[ind_node],
                                          raw_chain[ind_node+1])

        while not converged:
            try:

                node1 = raw_chain[node1_ind]
                node2 = raw_chain[node2_ind]

                curr_d1 = self._distance_function(
                    raw_chain[ind_node], raw_chain[ind_node-1])
                curr_d2 = self._distance_function(raw_chain[ind_node],
                                                  raw_chain[ind_node+1])
                print(
                    f"Current d1: {curr_d1} || Current d2: {curr_d2} || {init_d1=} || {init_d2=}")
                if curr_d1 <= 0.75*init_d1 or curr_d2 <= 0.75*init_d2:
                    print(
                        "Node fell more than 25% to one direction. Stopping minimization.")
                    converged = True
                    break

                # curr_d1d2 = self._distance_function(node1, node2)
                # print(
                #     f"Current d1d2: {curr_d1d2} || init d1d2: {init_d1d2} || dr: {dr}")

                # if curr_d1d2 > (init_d1d2+.5*dr):
                #     print("Distance between nodes is growing. Stopping minimization.")
                #     converged = True
                #     break

                if nsteps >= max_iter:
                    converged = True
                    break

                node_to_opt = node1
                node_to_opt_ind = ind_node

                prev_iter_ene = node_to_opt.energy
                if self.parameters.verbosity > 1:
                    print(f"{prev_iter_ene=}")

                if tangent is None:
                    tangent = node2.coords - node1.coords
                unit_tan = tangent / np.linalg.norm(tangent)

                # grad1 = self.engine.compute_gradients([node_to_opt])
                grad1 = node_to_opt.gradient
                gperp1 = ch.get_nudged_pe_grad(
                    unit_tangent=unit_tan, gradient=grad1)

                direction = gperp1
                out_chain = self.optimizer.optimize_step(
                    chain=Chain.model_validate({"nodes": [node_to_opt]}),
                    chain_gradients=np.array([direction]))

                new_node1 = out_chain.nodes[0]
                self.engine.compute_energies([new_node1])
                self.grad_calls_made += 1

                prev_node_ind = node_to_opt_ind
                if ind_node == 0:
                    prev_node_ind -= 1
                    # opp_node = node2
                elif ind_node == 1:
                    prev_node_ind += 1
                    # opp_node = node1

                raw_chain.nodes[node_to_opt_ind] = new_node1
                self.chain_trajectory.append(raw_chain.copy())
                nsteps += 1

                grad_inf_norm = np.amax(abs(gperp1))
                print("MIN: ", grad_inf_norm)
                if grad_inf_norm <= self.parameters.grad_tol:
                    converged = True
                    break

            except Exception:
                print(traceback.format_exc())
                return raw_chain
        print(f"\t converged in {nsteps}")
        return raw_chain

    def minimize_node_maxene(self, chain: Chain, node_ind: int, ind_ts_gi: int, smoother: MorseGeodesic):
        raw_chain = chain.copy()
        # print("SMOOTHER IS: ", smoother)
        chain_opt = self._min_node_maxene(
            raw_chain,
            ind_node=node_ind,
            ind_ts_gi=ind_ts_gi,
            smoother=smoother
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

    def grow_nodes(self, chain: Chain, dr: float, indices: tuple = None):
        sub_chain = [chain[indices[0]], chain[indices[1]]]

        found_nodes = False
        add_two_nodes = True
        nalready_grown = len(chain)-2
        nimg_to_grow = self.parameters.min_images+2 - nalready_grown
        # nimg_todo = max([nimg, 3])

        d0 = self._distance_function(sub_chain[0], sub_chain[1])
        # dr_new = d0 / (nimg_todo-1)
        # dr = max([dr, dr_new])

        # nimg = self.parameters.min_images+2
        # nimg = 20
        nimg = self.gi_inputs.nimages

        final_node1 = None
        final_node1_tan = None
        final_node2 = None
        final_node2_tan = None

        if (d0 <= 2 * dr) or nimg_to_grow == 3:
            add_two_nodes = False
            dr = d0 / 2

        while not found_nodes:
            if self.parameters.tangent.upper() == "LINEAR":
                node1, node2 = sub_chain[0].coords, sub_chain[1].coords
                direction = (node2 - node1) / np.linalg.norm((node2 - node1))
                new_node1 = node1 + direction * dr
                new_node2 = node2 - direction * dr
                final_node1 = sub_chain[0].update_coords(new_node1)
                final_node2 = sub_chain[1].update_coords(new_node2)
                found_nodes = True

            elif self.parameters.tangent.upper() == "GEODESIC":
                if self.parameters.verbosity > 1:
                    print(f"\t\t***trying with {nimg=}")
                if self.parameters.naive_grow:

                    interpolated = ch.run_geodesic(
                        chain=sub_chain, nimages=nimg)
                    final_node1 = interpolated[1]
                    if add_two_nodes:
                        final_node2 = interpolated[-2]

                    found_nodes = True
                    sys.stdout.flush()
                else:

                    smoother = ch.sample_shortest_geodesic(
                        sub_chain, nsamples=5, nimages=nimg)
                    interpolated = ch.gi_path_to_nodes(
                        xyz_coords=smoother.path,
                        symbols=sub_chain[0].structure.symbols,
                        charge=sub_chain[0].structure.charge,
                        spinmult=sub_chain[0].structure.multiplicity,
                    )
                    # dr = smoother.length / (nimg_todo-1)
                    print("\t\tlength_smoother: ", smoother.length, "dr: ", dr)
                    sys.stdout.flush()

                    if not final_node1 or not final_node2:
                        node1, tan1 = self._select_node_at_dist(
                            chain=interpolated,
                            dist=dr,
                            direction=1,
                            dist_err=self.parameters.dist_err
                            * dr,
                            smoother=smoother,
                        )
                        if node1:
                            final_node1 = node1
                            final_node1_tan = tan1

                        if add_two_nodes:
                            node2, tan2 = self._select_node_at_dist(
                                chain=interpolated,
                                dist=dr,
                                direction=-1,
                                dist_err=self.parameters.dist_err
                                * dr,
                                smoother=smoother,
                            )

                            if node2:
                                final_node2 = node2
                                final_node2_tan = tan2

                    if add_two_nodes and (final_node1 and final_node2):
                        found_nodes = True
                    elif not add_two_nodes and final_node1:
                        found_nodes = True
                    else:
                        nimg += 10
                        self.gi_inputs.nimages = nimg
            else:
                raise ValueError(
                    f"Invalid tangent type: {self.parameters.tangent}. Use one of LINEAR or GEODESIC"
                )

        if add_two_nodes:
            self.engine.compute_energies([final_node2, final_node1])
            self.grad_calls_made += 2
        else:
            self.engine.compute_energies([final_node1])
            self.grad_calls_made += 1
        self.grad_calls_made += 2

        grown_chain = chain.copy()
        # insert_index = int(len(grown_chain) / 2)
        insert_index = indices[1]
        idx2 = None
        if add_two_nodes:
            grown_chain.nodes.insert(insert_index, final_node2)
            idx2 = insert_index + 1
        grown_chain.nodes.insert(insert_index, final_node1)
        idx1 = insert_index

        return grown_chain, [final_node1_tan, final_node2_tan], (idx1, idx2), dr

    def grow_nodes_maxene(self, chain: Chain, last_grown_ind: int = 0, nimg: int = 20, nudge=0.1):
        """
        will return a chain with 1 new node which is the highest energy interpolated
        node between the last node added and its nearest neighbors.

        If no node is added, will return the input chain.
        """
        smoother = None
        if last_grown_ind == 0:  # initial case
            smoother = ch.sample_shortest_geodesic([chain[0], chain[-1]],
                                                   nimages=nimg, nudge=nudge, nsamples=5)
            gi = ch.gi_path_to_nodes(
                xyz_coords=smoother.path,
                symbols=chain[0].symbols,
                charge=chain[0].structure.charge,
                spinmult=chain[0].structure.multiplicity,
            )
            # eng = RunInputs(program='xtb').engine
            # eng.compute_energies(gi)
            print("LENGI: ", len(gi))

            grown_chain = chain.copy()
            if self.parameters.use_xtb_grow:
                print("using xtb to select max energy node")
                maxnode_data = get_maxene_node(gi, engine=RunInputs().engine)
                maxnode_data['node']._cached_energy = None
                maxnode_data['node']._cached_gradient = None
                self.engine.compute_energies([maxnode_data['node']])
                self.grad_calls_made += 1
            else:
                maxnode_data = get_maxene_node(gi, engine=self.engine)
                self.grad_calls_made += maxnode_data['grad_calls']

            ind_max = maxnode_data['index']
            node = maxnode_data['node']
            barrier_climb_kcal = (node._cached_energy -
                                  chain.energies.max())*627.5
            skip_growth = barrier_climb_kcal <= self.parameters.barrier_thre

            if ind_max == 0 or ind_max == len(chain)-1:
                print("No TS found between endpoints. Returning input chain.")
                return chain, 1, ind_max
            elif skip_growth:
                print("Barrier climb is too low. Returning input chain.")
                return chain, chain.energies.argmax(), ind_max

            # node = gi[ind_max]
            # node._cached_energy = None
            # node._cached_gradient = None
            # self.engine.compute_energies([node])
            # self.grad_calls_made += 1
            grown_chain.nodes = [chain[0], node, chain[-1]]
            new_ind = 1

        else:
            smoother1 = ch.sample_shortest_geodesic([chain[last_grown_ind-1],
                                                     chain[last_grown_ind]],
                                                    nimages=nimg, nudge=nudge, nsamples=5)
            gi1 = ch.gi_path_to_nodes(
                xyz_coords=smoother1.path,
                symbols=chain[0].symbols,
                charge=chain[0].structure.charge,
                spinmult=chain[0].structure.multiplicity,
            )
            smoother2 = ch.sample_shortest_geodesic([chain[last_grown_ind],
                                                     chain[last_grown_ind+1]],
                                                    nimages=nimg, nudge=nudge, nsamples=5)
            gi2 = ch.gi_path_to_nodes(
                xyz_coords=smoother2.path,
                symbols=chain[0].symbols,
                charge=chain[0].structure.charge,
                spinmult=chain[0].structure.multiplicity,
            )

            # eng = RunInputs(program='xtb').engine
            # eng.compute_energies(gi1)
            # eng.compute_energies(gi2)

            # deltaEs = [gi1.get_eA_chain(), gi2.get_eA_chain()]
            if self.parameters.use_xtb_grow:
                print("using xtb to select max energy node")
                maxnode1_data = get_maxene_node(gi1, engine=RunInputs().engine)
                maxnode1_data['node']._cached_energy = None
                maxnode1_data['node']._cached_gradient = None
                self.engine.compute_energies([maxnode1_data['node']])

                maxnode2_data = get_maxene_node(gi2, engine=RunInputs().engine)
                maxnode2_data['node']._cached_energy = None
                maxnode2_data['node']._cached_gradient = None
                self.engine.compute_energies([maxnode2_data['node']])

                self.grad_calls_made += 2
            else:

                maxnode1_data = get_maxene_node(gi1, engine=self.engine)
                maxnode2_data = get_maxene_node(gi2, engine=self.engine)

                self.grad_calls_made += maxnode1_data['grad_calls']
                self.grad_calls_made += maxnode2_data['grad_calls']

            # deltaEs = [(maxnode1_data['node'].energy -
            #             chain[last_grown_ind].energy)*627.5,
            #            (maxnode2_data['node'].energy -
            #             chain[last_grown_ind].energy)*627.5
            #            ]
            deltaEs = [(maxnode1_data['node'].energy -
                        chain.energies.max())*627.5,
                       (maxnode2_data['node'].energy -
                        chain.energies.max())*627.5
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
                    smoother = smoother1
                    smoother = None  # CHANGEME
                else:
                    left_side_converged = True
                    ind_max = ind_max_right
                    smoother = smoother2
                    smoother = None

            if left_side_converged and right_side_converged:
                print("TS guess found. Returning input chain.")
                ind_max = ind_max_left  # arbitrary choice
                return chain, last_grown_ind, ind_max, smoother

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
                smoother = smoother2
                smoother = None

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
                smoother = smoother1
                smoother = None

        return grown_chain, new_ind, ind_max, smoother

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
        chain: Chain,
        dist: float,
        direction: int,
        smoother: MorseGeodesic = None,
        dist_err: float = 0.1,
    ):
        """
        will iterate through chain and select the node that is 'dist' away up to 'dist_err'

        dist - -> the requested distance where you want the nodes
        direction - -> whether to go forward(1) or backwards(-1). if forward, will pick the requested node
                    to the first node. if backwards, will pick the requested node to the last node

        """
        input_chain = [node.copy() for node in chain]
        if direction == -1:
            input_chain.reverse()

        start_node = input_chain[0]
        best_node = None
        best_dist_err = 10000.0
        best_node_tangent = None
        for i, node in enumerate(input_chain[1:-1], start=1):
            if self.parameters.distance_metric.upper() == "GEODESIC":
                if direction == -1:
                    start = len(smoother.path) - i
                    end = -1

                elif direction == 1:
                    start = 1
                    end = i + 1

                # smoother._compute_disps(start=start, end=end)
                # curr_dist = smoother.length
                curr_dist = smoother.segment_lengths[start-1:end-1].sum()
            else:
                curr_dist = self._distance_function(
                    node1=start_node, node2=node)
            curr_dist_err = np.abs(curr_dist - dist)
            if self.parameters.verbosity > 1:
                print(f"\t{curr_dist_err=} vs {dist_err=} || {direction=}")

            if curr_dist_err <= dist_err and curr_dist_err < best_dist_err:
                best_node = node
                best_dist_err = curr_dist_err
                prev_node = input_chain[i - 1]
                next_node = input_chain[i + 1]
                if self.parameters.tangent == "geodesic":
                    # raise NotImplementedError("Not done yet.")
                    # self.engine.compute_energies([prev_node, node, next_node])
                    # self.grad_calls_made += 3
                    tau_plus = next_node.coords - node.coords
                    tau_minus = node.coords - prev_node.coords
                    best_node_tangent = (tau_plus + tau_minus) / 2
                    # best_node_tangent = ch._create_tangent_path(
                    #     prev_node=prev_node,
                    #     current_node=node,
                    #     next_node=next_node,
                    # )
                else:

                    best_node_tangent = None

                # break

        return best_node, best_node_tangent

    def _check_nodes_converged(
        self, node, prev_node, opposite_node, prev_iter_ene: float, d0
    ):
        curr_iter_ene = node.energy
        d = self._distance_function(node, opposite_node)
        s = self._distance_function(node, prev_node)
        # print(d, d0, s, d0 + 0.5*s, curr_iter_ene, prev_iter_ene)
        distance_exceeded = d > d0 + 0.5 * s
        energy_exceeded = curr_iter_ene > prev_iter_ene
        if self.parameters.verbosity > 1:
            print(f"{distance_exceeded=} {energy_exceeded=}")
        return distance_exceeded or energy_exceeded

    def chain_converged(self, chain: Chain, dr: float, indices):
        # node1, node2 = self._get_innermost_nodes(chain)
        node1_ind, node2_ind = indices
        print("INDICES: ", indices)
        node2_ind = node1_ind + 1

        node1 = chain[node1_ind]
        node2 = chain[node2_ind]

        next_grow_ind = node1_ind

        if indices[1] is None:
            node2_1 = chain[node1_ind + 1]
            node2_2 = chain[node1_ind - 1]

            dist_1 = self._distance_function(node1, node2_1)
            enedist_1 = abs(node1.energy - node2_1.energy)
            dist_2 = self._distance_function(node1, node2_2)
            enedist_2 = abs(node1.energy - node2_2.energy)
            print(f"distances: {dist_1} {dist_2}")
            # if dist_1 > dist_2:
            if enedist_1 > enedist_2:
                dist = dist_1
            else:
                next_grow_ind = node1_ind - 1
                dist = dist_2
        else:
            dist = self._distance_function(node1, node2)
        print(f"distance between innermost nodes {dist}")
        # if dist <= dr + (self.parameters.dist_err * dr):
        if dist <= dr:
            result = True
        else:
            result = False

        return result, next_grow_ind
