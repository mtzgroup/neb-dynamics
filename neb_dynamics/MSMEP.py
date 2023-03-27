from dataclasses import dataclass

import numpy as np
from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
from retropaths.helper_functions import pairwise

from neb_dynamics.Chain import Chain
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Inputs import NEBInputs, ChainInputs, GIInputs
from neb_dynamics.helper_functions import _get_ind_minima
from neb_dynamics.TreeNode import TreeNode

@dataclass
class MSMEP:

    neb_inputs: NEBInputs = NEBInputs()
    chain_inputs: ChainInputs = ChainInputs()
    gi_inputs: GIInputs = GIInputs()

    # electronic structure params
    charge: int = 0
    spinmult: int = 1

    recycle_chain: bool = False

    def create_endpoints_from_rxn_name(self, rxn_name, reactions_object):
        rxn = reactions_object[rxn_name]
        root = TDStructure.from_rxn_name(rxn_name, reactions_object)

        c3d_list = root.get_changes_in_3d(rxn)

        root = root.pseudoalign(c3d_list)
        root.gum_mm_optimization()
        root_opt = root.xtb_geom_optimization()

        target = root.copy()
        target.apply_changed3d_list(c3d_list)
        target.gum_mm_optimization()
        target_opt = target.xtb_geom_optimization()

        if not root_opt.molecule_rp == root.molecule_rp:
            raise ValueError(
                "Pseudoaligned start molecule was not a minimum at this level of theory. Exiting."
            )

        if not root_opt.molecule_rp == root.molecule_rp:
            raise ValueError(
                "Product molecule was not a minimum at this level of theory. Exiting."
            )

        return root_opt, target_opt

    def find_mep_multistep(self, input_chain):
        root_neb_obj, chain = self.get_neb_chain(input_chain=input_chain)
        # history = [root_neb_obj]
        history = TreeNode(data=root_neb_obj, children=[])
        if not chain:
            return None, None
        if self.is_elem_step(chain):
            return history, chain
        else:
            sequence_of_chains = self.make_sequence_of_chains(chain)
            elem_steps = []
            for i, chain_frag in enumerate(sequence_of_chains):
                print(f"On chain {i+1} of {len(sequence_of_chains)}...")
                out_history, chain = self.find_mep_multistep(chain_frag)
                if (
                    chain
                ):  # << TODO:!!!! i remove None's by this. think about this later...
                    elem_steps.append(chain)
                    history.children.append(out_history)
                # history.append(out_history)

            stitched_elem_steps = self.stitch_elem_steps(elem_steps)
            return (
                history,
                stitched_elem_steps,
            )  # the first 'None' will hold the DataTree that holds all NEB objects

    def _create_interpolation(self, chain: Chain):

        if chain.parameters.use_geodesic_interpolation:
            traj = Trajectory(
                [node.tdstructure for node in chain],
                charge=self.charge,
                spinmult=self.spinmult,
            )

            gi = traj.run_geodesic(
                nimages=self.gi_inputs.nimages,
                friction=self.gi_inputs.friction,
                nudge=self.gi_inputs.nudge,
                **self.gi_inputs.extra_kwds,
            )
            interpolation = Chain.from_traj(traj=gi, parameters=self.chain_inputs)

        else:  # do a linear interpolation using numpy
            start_point = chain[0].coords
            end_point = chain[-1].coords
            coords = np.linspace(start_point, end_point, self.gi_inputs.nimages)
            coords[1:-1] += np.random.normal(scale=0.05)

            interpolation = Chain.from_list_of_coords(
                list_of_coords=coords, parameters=self.chain_inputs
            )

        return interpolation

    def get_neb_chain(self, input_chain: Chain):
        if input_chain[0].is_identical(input_chain[-1]):
            print("Endpoints are identical. Returning nothing")
            return None, None

        if len(input_chain) < self.gi_inputs.nimages:
            interpolation = self._create_interpolation(input_chain)
        else:
            interpolation = input_chain

        n = NEB(initial_chain=interpolation, parameters=self.neb_inputs)
        try:
            print("Running NEB calculation...")
            n.optimize_chain()
            out_chain = n.optimized

        except NoneConvergedException:
            print(
                "\nWarning! A chain did not converge. Returning an unoptimized chain..."
            )
            out_chain = n.chain_trajectory[-1]

        return n, out_chain

    def _chain_is_concave(self, chain):
        ind_minima = _get_ind_minima(chain)
        return len(ind_minima) == 0

    def _approx_irc(self, chain):
        arg_max = np.argmax(chain.energies)
        candidate_r = chain[arg_max - 1]
        candidate_p = chain[arg_max + 1]
        r = candidate_r.do_geometry_optimization()
        p = candidate_p.do_geometry_optimization()
        return r.is_identical(chain[0]) and p.is_identical(chain[-1])

    def is_elem_step(self, chain):
        if len(chain) <= 1:
            return True

        conditions = []
        is_concave = self._chain_is_concave(chain)
        conditions.append(is_concave)
        if is_concave:  # i.e. we only have one maxima
            minimizing_gives_endpoints = self._approx_irc(chain)
            conditions.append(minimizing_gives_endpoints)
        return all(conditions)

    def _make_chain_frag(self, chain: Chain, pair_of_inds):
        start, end = pair_of_inds
        chain_frag = chain.copy()
        chain_frag.nodes = chain[start : end + 1]
        opt_start = chain[start].do_geometry_optimization()
        opt_end = chain[end].do_geometry_optimization()

        chain_frag.insert(0, opt_start)
        chain_frag.append(opt_end)

        return chain_frag

    def _make_chain_pair(self, chain: Chain, pair_of_inds):
        start, end = pair_of_inds
        start_opt = chain[start].do_geometry_optimization()
        end_opt = chain[end].do_geometry_optimization()

        opt_start = chain[start].do_geometry_optimization()
        opt_end = chain[end].do_geometry_optimization()
        chain_frag = Chain(nodes=[start_opt, end_opt], parameters=chain.parameters)

        return chain_frag

    def make_sequence_of_chains(self, chain):
        all_inds = [0]
        ind_minima = _get_ind_minima(chain)
        all_inds.extend(ind_minima)
        all_inds.append(len(chain) - 1)

        pairs_inds = list(pairwise(all_inds))

        chains = []
        for ind_pair in pairs_inds:
            if self.recycle_chain:
                chains.append(self._make_chain_frag(chain, ind_pair))
            else:
                chains.append(self._make_chain_pair(chain, ind_pair))

        return chains

    def stitch_elem_steps(self, list_of_chains):
        out_list_of_chains = [chain for chain in list_of_chains if chain is not None]
        return Chain.from_list_of_chains(
            out_list_of_chains, parameters=self.chain_inputs
        )
