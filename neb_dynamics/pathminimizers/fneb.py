from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

import neb_dynamics.chainhelpers as ch
from neb_dynamics.chain import Chain
from neb_dynamics.engines.engine import Engine
from neb_dynamics.helper_functions import RMSD
from neb_dynamics.pathminimizers.pathminimizer import PathMinimizer
from neb_dynamics.elementarystep import check_if_elem_step

import traceback

import sys


@dataclass
class FreezingNEB(PathMinimizer):
    initial_chain: Chain
    engine: Engine
    chain_trajectory: list[Chain] = field(default_factory=list)
    fneb_kwds: dict = field(default_factory=dict)

    def __post_init__(self):
        fneb_kwds_defaults = {
            "stepsize": 0.5,
            "ngradcalls": 3,
            "max_cycles": 500,
            "path_resolution": 1 / 10,  # BOHR,
            "max_atom_displacement": 0.1,
            "early_stop_scaling": 3,
            "use_geodesic_tangent": False,
            "dist_err": 0.5,
        }
        for key, val in self.fneb_kwds.items():
            fneb_kwds_defaults[key] = val
        self.fneb_kwds = fneb_kwds_defaults

        self.parameters = SimpleNamespace(**self.fneb_kwds)
        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0

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
        chain.nodes = [
            chain.nodes[0],
            chain.nodes[-1],
        ]  # need to make sure I only use the endpoints
        self.engine.compute_energies(chain)
        # print(chain.energies, self.engine)
        self.grad_calls_made += 2
        # print(f"PROG: {self.grad_calls_made}")
        self.chain_trajectory = [chain]

        node1, node2 = self._get_innermost_nodes(chain)
        d0, _ = RMSD(node1.coords, node2.coords)

        self.dinitial = d0
        converged = False
        nsteps = 0

        while not converged and nsteps < self.parameters.max_cycles:
            print(f"STEP{nsteps}")
            # grow nodes
            print("\tgrowing nodes")
            node1, node2 = self._get_innermost_nodes(chain)
            d0, _ = RMSD(node1.coords, node2.coords)
            grown_chain, node_tangents = self.grow_nodes(
                chain, dr=self.parameters.path_resolution * d0
            )

            # minimize nodes
            print("\tminimizing nodes")
            min_chain = self.minimize_nodes(
                chain=grown_chain, node_tangents=node_tangents, d0=d0
            )
            self.chain_trajectory.append(min_chain)

            # check convergence
            print("\tchecking convergence")
            converged, inner_bead_distance = self.chain_converged(min_chain)
            dr = self.parameters.path_resolution * d0
            if inner_bead_distance <= self.parameters.early_stop_scaling * dr:
                new_params = SimpleNamespace(**self.fneb_kwds)
                new_params.early_stop_scaling = 0.0
                self.parameters = new_params

                elem_step_results = check_if_elem_step(
                    inp_chain=min_chain, engine=self.engine
                )
                if not elem_step_results.is_elem_step:
                    print("Stopping early because chain is multistep.")
                    self.optimized = min_chain
                    return elem_step_results
            chain = min_chain.copy()
            nsteps += 1

        self.optimized = self.chain_trajectory[-1]
        print(f"Converged? {converged}")
        elem_step_results = check_if_elem_step(chain, engine=self.engine)
        self.geom_grad_calls_made += elem_step_results.number_grad_calls
        return elem_step_results

    def _check_nodes_converged(
        self, node, prev_node, opposite_node, prev_iter_ene: float, d0
    ):
        curr_iter_ene = node.energy
        d, _ = RMSD(node.coords, opposite_node.coords)
        s, _ = RMSD(node.coords, prev_node.coords)
        # print(d, d0, s, d0 + 0.5*s, curr_iter_ene, prev_iter_ene)
        distance_exceeded = d > d0 + 0.5 * s
        energy_exceeded = curr_iter_ene > prev_iter_ene
        print(f"{distance_exceeded=} {energy_exceeded=}")
        return distance_exceeded or energy_exceeded

    def _min_node(
        self,
        raw_chain: Chain,
        tangent: np.array,
        ngradcalls: int,
        ss: float,
        d0: float,
        ind_node: int = 0,
        max_atom_displacement=0.1,  # BOHR
    ):
        """
        ind_node: 0 or 1. 0 if you want the optimize the leftmost inner node. 1
        if you want the rightmost inner node.
        """
        node1_ind, node2_ind = self._get_innermost_nodes_inds(raw_chain)
        node1 = raw_chain[node1_ind]
        node2 = raw_chain[node2_ind]
        converged = False
        nsteps = 1  # we already made one gradient call when growing the node.

        while not converged and nsteps < ngradcalls:
            try:
                node1_ind, node2_ind = self._get_innermost_nodes_inds(raw_chain)
                node1 = raw_chain[node1_ind]
                node2 = raw_chain[node2_ind]

                node_to_opt = [node1, node2][ind_node]
                node_to_opt_ind = [node1_ind, node2_ind][ind_node]

                prev_iter_ene = node_to_opt.energy
                print(f"{prev_iter_ene=}")

                if tangent is None:
                    tangent = node2.coords - node1.coords
                unit_tan = tangent / np.linalg.norm(tangent)

                # grad1 = self.engine.compute_gradients([node_to_opt])
                grad1 = node_to_opt.gradient
                gperp1 = ch.get_nudged_pe_grad(unit_tangent=unit_tan, gradient=grad1)

                direction = -1 * gperp1 * ss
                direction_scaled = direction.copy()
                for i_atom, vector in enumerate(direction):
                    length = np.linalg.norm(vector)
                    if length > max_atom_displacement:
                        direction_scaled[i_atom, :] = (
                            vector / length
                        ) * max_atom_displacement

                new_node1_coords = node_to_opt.coords + direction_scaled
                new_node1 = node_to_opt.update_coords(new_node1_coords)
                self.engine.compute_energies([new_node1])
                # print(f"PROG: {self.grad_calls_made}")
                self.grad_calls_made += 1

                converged = self._check_nodes_converged(
                    node=new_node1,
                    prev_node=raw_chain[node1_ind - 1],
                    opposite_node=node2,
                    prev_iter_ene=prev_iter_ene,
                    d0=d0,
                )

                raw_chain.nodes[node_to_opt_ind] = new_node1
                nsteps += 1

            except Exception:
                print(traceback.format_exc())
                return raw_chain

        return raw_chain

    def minimize_nodes(self, chain: Chain, node_tangents: list, d0):
        raw_chain = chain.copy()
        # return raw_chain
        chain_opt1 = self._min_node(
            raw_chain,
            tangent=node_tangents[0],
            ind_node=0,
            ngradcalls=self.parameters.ngradcalls,
            ss=self.parameters.stepsize,
            max_atom_displacement=self.parameters.max_atom_displacement,
            d0=d0,
        )
        chain_opt2 = self._min_node(
            chain_opt1,
            tangent=node_tangents[1],
            ind_node=1,
            ngradcalls=self.parameters.ngradcalls,
            ss=self.parameters.stepsize,
            max_atom_displacement=self.parameters.max_atom_displacement,
            d0=d0,
        )

        return chain_opt2

    def _get_innermost_nodes_inds(self, chain: Chain):
        if len(chain) == 2:
            return 0, 1

        ind_node2 = int(len(chain) / 2)
        ind_node1 = ind_node2 - 1
        return ind_node1, ind_node2

    def _get_innermost_nodes(self, chain: Chain):
        """
        returns a chain object with the two innermost nodes
        """
        ind_node1, ind_node2 = self._get_innermost_nodes_inds(chain)
        out_chain = chain.copy()
        out_chain.nodes = [chain[ind_node1], chain[ind_node2]]

        return out_chain

    def grow_nodes(self, chain: Chain, dr: float):
        sub_chain = self._get_innermost_nodes(chain)

        sweep = True
        found_nodes = False
        nimg = 10
        # interpolated = ch.run_geodesic(sub_chain, nimages=nimg, sweep=sweep)
        # interpolation_1 = interpolated.copy()
        # interpolation_1.nodes = interpolation_1.nodes[:2]
        # interpolation_2 = interpolated.copy()
        # interpolation_2.nodes = interpolation_2.nodes[-2:]

        final_node1 = None
        final_node1_tan = None
        final_node2 = None
        final_node2_tan = None

        while not found_nodes:
            print(f"\t\t***trying with {nimg=}")
            interpolated = ch.run_geodesic(sub_chain, nimages=nimg, sweep=sweep)
            sys.stdout.flush()

            if not final_node1:
                # # interpolation_1.nodes = interpolation_1.nodes[
                # #     : int(len(interpolation_1) / 2)
                # # ]
                # interpolation_1.nodes = interpolation_1.nodes[:2]
                # interpolation_1 = ch.run_geodesic(
                #     interpolation_1, nimages=nimg, sweep=sweep
                # )
                node1, tan1 = self._select_node_at_dist(
                    # chain=interpolation_1,
                    chain=interpolated,
                    dist=dr,
                    direction=1,
                    dist_err=self.parameters.dist_err
                    * self.parameters.path_resolution
                    * self.dinitial,
                )
            if not final_node2:
                # interpolation_2.nodes = interpolation_2.nodes[
                #     : int(len(interpolation_2) / 2)
                # ]
                # interpolation_2.nodes = interpolation_2.nodes[-2:]
                # interpolation_2 = ch.run_geodesic(
                #     interpolation_2, nimages=nimg, sweep=sweep
                # )
                node2, tan2 = self._select_node_at_dist(
                    # chain=interpolation_2,
                    chain=interpolated,
                    dist=dr,
                    direction=-1,
                    dist_err=self.parameters.dist_err
                    * self.parameters.path_resolution
                    * self.dinitial,
                )
            if node1:
                final_node1 = node1
                final_node1_tan = tan1

            if node2:
                final_node2 = node2
                final_node2_tan = tan2

            if final_node1 and final_node2:
                found_nodes = True
            else:
                nimg += 5

        grown_chain = chain.copy()
        insert_index = int(len(grown_chain) / 2)
        grown_chain.nodes.insert(insert_index, final_node2)
        grown_chain.nodes.insert(insert_index, final_node1)

        return grown_chain, [final_node1_tan, final_node2_tan]

    def _select_node_at_dist(
        self, chain: Chain, dist: float, direction: int, dist_err: float = 0.1
    ):
        """
        will iterate through chain and select the node that is 'dist' away up to 'dist_err'

        dist --> the requested distance where you want the nodes
        direction --> whether to go forward (1) or backwards (-1). if forward, will pick the requested node
                    to the first node. if backwards, will pick the requested node to the last node


        write recuersively midpoint geodesic finding node thing so that you keep recycling onlky half the grodesic trajectory every time you remake it
        """
        input_chain = chain.copy()
        if direction == -1:
            input_chain.nodes.reverse()

        start_node = input_chain[0]
        best_node = None
        best_dist_err = 10000.0
        best_node_tangent = None
        for i, node in enumerate(input_chain.nodes[1:-1], start=1):
            curr_dist, _ = RMSD(start_node.coords, node.coords)
            curr_dist_err = np.abs(curr_dist - dist)
            print(f"\t{curr_dist_err=} vs {dist_err=}")
            if curr_dist_err <= dist_err and curr_dist_err < best_dist_err:
                best_node = node
                best_dist_err = curr_dist_err
                prev_node = input_chain.nodes[i - 1]
                next_node = input_chain.nodes[i + 1]
                if self.parameters.use_geodesic_tangent:
                    self.engine.compute_energies([prev_node, node, next_node])
                    self.grad_calls_made += 3
                    # print(f"PROG: {self.grad_calls_made}")
                    best_node_tangent = ch._create_tangent_path(
                        prev_node=prev_node,
                        current_node=node,
                        next_node=next_node,
                    )
                else:
                    self.engine.compute_energies([node])
                    # print(f"PROG: {self.grad_calls_made}")
                    self.grad_calls_made += 1
                    best_node_tangent = None

                break

        return best_node, best_node_tangent

    def chain_converged(self, chain: Chain):
        dr = self.parameters.path_resolution * self.dinitial
        node1, node2 = self._get_innermost_nodes(chain)
        dist, _ = RMSD(node1.coords, node2.coords)
        print(f"distance between innermost nodes {dist}")
        if dist <= dr:
            result = True
        else:
            result = False

        return result, dist
