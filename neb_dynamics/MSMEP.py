from dataclasses import dataclass

from pathlib import Path
import sys
import numpy as np
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.helper_functions import pairwise

from neb_dynamics.Chain import Chain
from neb_dynamics.NEB import NEB, NoneConvergedException
from neb_dynamics.Inputs import NEBInputs, ChainInputs, GIInputs
from neb_dynamics.helper_functions import _get_ind_minima, _get_ind_maxima, create_friction_optimal_gi
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.Optimizer import Optimizer
from neb_dynamics.NEB_TCDLF import NEB_TCDLF

@dataclass
class MSMEP:

    neb_inputs: NEBInputs
    chain_inputs: ChainInputs 
    gi_inputs: GIInputs 
    
    optimizer: Optimizer
    
    # other parameters
    skip_identical_graphs: bool = True
    _use_dlf_as_backend: bool = False

    # electronic structure params
    charge: int = 0
    spinmult: int = 1

    def create_endpoints_from_rxn_name(self, rxn_name, reactions_object):
        rxn = reactions_object[rxn_name]
        root = TDStructure.from_rxn_name(rxn_name, reactions_object)

        c3d_list = root.get_changes_in_3d(rxn)

        root = root.pseudoalign(c3d_list)
        root.gum_mm_optimization()
        root_opt = root.xtb_geom_optimization()

        target = root.copy()
        target.apply_changed3d_list(c3d_list)
        target_opt = target.xtb_geom_optimization()

        if not root_opt.molecule_rp.is_bond_isomorphic_to(root.molecule_rp):
            raise ValueError(
                "Pseudoaligned start molecule was not a minimum at this level of theory. Exiting."
            )

        if not target_opt.molecule_rp.is_bond_isomorphic_to(target.molecule_rp):
            raise ValueError(
                "Product molecule was not a minimum at this level of theory. Exiting."
            )

        return root_opt, target_opt

    def find_mep_multistep(self, input_chain, tree_node_index=0):
        
        if input_chain[0].is_a_molecule:
            if input_chain[0]._is_connectivity_identical(input_chain[-1]) and self.skip_identical_graphs:
                print("Endpoints are identical. Returning nothing")
                return TreeNode(data=None,children=[],index=tree_node_index), None
        else:
            if input_chain[0].is_identical(input_chain[-1]):
                print("Endpoints are identical. Returning nothing")
                return TreeNode(data=None,children=[],index=tree_node_index), None
        
        root_neb_obj, chain, elem_step_results = self.get_neb_chain(input_chain=input_chain)
        history = TreeNode(data=root_neb_obj, children=[], index=tree_node_index)
        
        elem_step, split_method, minimization_results = elem_step_results
        
        if elem_step:
            return history, chain
       
        else:
            sequence_of_chains = self.make_sequence_of_chains(chain,split_method, minimization_results)
            print(f"Splitting chains based on: {split_method}")
            elem_steps = []
            new_tree_node_index = tree_node_index + 1
            for i, chain_frag in enumerate(sequence_of_chains, start=1):
                print(f"On chain {i} of {len(sequence_of_chains)}...")
                out_history, chain = self.find_mep_multistep(chain_frag, tree_node_index=new_tree_node_index)
                
                # add the outputs
                elem_steps.append(chain)
                history.children.append(out_history)
                
                # increment the node indices
                new_tree_node_index = out_history.max_index+1
            stitched_elem_steps = self.stitch_elem_steps(elem_steps)
            return history, stitched_elem_steps

    def _create_interpolation(self, chain: Chain):

        if chain.parameters.use_geodesic_interpolation:
            traj = chain.to_trajectory()
            if chain.parameters.friction_optimal_gi:
                gi = create_friction_optimal_gi(traj, self.gi_inputs)
            else:
                gi = traj.run_geodesic(
                    nimages=self.gi_inputs.nimages,
                    friction=self.gi_inputs.friction,
                    nudge=self.gi_inputs.nudge,
                    **self.gi_inputs.extra_kwds,
                )
            
            interpolation = Chain.from_traj(traj=gi, parameters=self.chain_inputs)
            interpolation._zero_velocity()

        else:  # do a linear interpolation using numpy
            start_point = chain[0].coords
            end_point = chain[-1].coords
            coords = np.linspace(start_point, end_point, self.gi_inputs.nimages)
            coords[1:-1] += np.random.normal(scale=0.00)

            interpolation = Chain.from_list_of_coords(
                list_of_coords=coords, parameters=self.chain_inputs
            )

        return interpolation

    def get_neb_chain(self, input_chain: Chain):
        
        if len(input_chain) != self.gi_inputs.nimages:
            interpolation = self._create_interpolation(input_chain)
        else:
            interpolation = input_chain
            
        print("Running NEB calculation...")
        if not self._use_dlf_as_backend:
            print("Using in-house NEB optimizer")
            sys.stdout.flush()
            
            n = NEB(initial_chain=interpolation, parameters=self.neb_inputs, optimizer=self.optimizer)
            try:
                elem_step_results = n.optimize_chain()
                out_chain = n.optimized

            except NoneConvergedException:
                print(
                    "\nWarning! A chain did not converge. Returning an unoptimized chain..."
                )
                out_chain = n.chain_trajectory[-1]
                elem_step_results = out_chain.is_elem_step()
                
                
                
            # except Exception as e:
            #     print(e)
            #     print("Warning! Electronic structure error. Aborting.")
            #     out_chain = n.chain_trajectory[-1]
            #     elem_step_results = out_chain.is_elem_step()
                

        else:
            print("Using DL-Find as NEB optimizer")
            sys.stdout.flush()
            dlf_inputs = self.neb_inputs.copy()
            dlf_inputs.tol = self.neb_inputs.early_stop_force_thre
            dlf_inputs.en_thre = dlf_inputs.tol
            dlf_inputs.grad_thre = dlf_inputs.tol
            dlf_inputs.rms_grad_thre = dlf_inputs.tol * (2/3)
            
            # dlf_inputs.max_steps = int(self.neb_inputs.max_steps / 5)
            
            total_chain_traj = [interpolation]
            
            # do neb until early stop
            n = NEB_TCDLF(initial_chain=total_chain_traj[-1], parameters=dlf_inputs)
            n.optimize_chain()
            total_chain_traj.extend(n.chain_trajectory)
            
            if not n.optimized.is_elem_step()[0]:
                print("Chain is not an elementary step...splitting")
            else:
                print("Chain appears to be an elem step, continuing...")
                n = NEB_TCDLF(initial_chain=total_chain_traj[-1], parameters=self.neb_inputs)
                n.optimize_chain()
                total_chain_traj.extend(n.chain_trajectory)
                print(f"Converged? {n.converged}")
            
            n.chain_trajectory = total_chain_traj
            out_chain = total_chain_traj[-1]
            elem_step_results = out_chain.is_elem_step()
            
        
        
        return n, out_chain, elem_step_results
    

    # def _make_chain_frag(self, chain: Chain, pair_of_inds):
    def _make_chain_frag(self, chain: Chain, geom_pair):
        # start, end = pair_of_inds
        opt_start, opt_end = geom_pair
        # chain_frag = chain.copy()
        chain_frag = Chain(nodes=[opt_start, opt_end], parameters=chain.parameters)
        # opt_start = chain[start].do_geometry_optimization()
        # opt_end = chain[end].do_geometry_optimization()

        # chain_frag.insert(0, opt_start)
        # chain_frag.append(opt_end)

        return chain_frag

    def _make_chain_pair(self, chain: Chain, pair_of_inds):
        start, end = pair_of_inds
        start_opt = chain[start].do_geometry_optimization()
        end_opt = chain[end].do_geometry_optimization()

        chain_frag = Chain(nodes=[start_opt, end_opt], parameters=chain.parameters)

        return chain_frag


    def _do_minima_based_split(self, chain, minimization_results):
        all_geometries = [chain[0]]
        all_geometries.extend(minimization_results)
        all_geometries.append(chain[-1])
        # all_inds = [0]
        # ind_minima = _get_ind_minima(chain)
        # all_inds.extend(ind_minima)
        # all_inds.append(len(chain) - 1)

        # pairs_inds = list(pairwise(all_inds))
        pairs_geoms = list(pairwise(all_geometries))

        chains = []
        # for ind_pair in pairs_inds:
            # chains.append(self._make_chain_frag(chain, ind_pair))
        for geom_pair in pairs_geoms:
            chains.append(self._make_chain_frag(chain, geom_pair))

        return chains

    def _do_maxima_based_split(self, chain: Chain, minimization_results):
        
        
        ind_maxima = _get_ind_maxima(chain)
        # r, p = chain._approx_irc(index=ind_maxima)
        r, p = minimization_results
        chains_list = []
        
        # add the input start, to R
        nodes = [chain[0], r]
        chain_frag = chain.copy()
        chain_frag.nodes = nodes
        chains_list.append(chain_frag)

        # add the r to p, passing through the maxima
        nodes2 = [r, chain[ind_maxima], p]
        chain_frag2 = chain.copy()
        chain_frag2.nodes = nodes2
        chains_list.append(chain_frag2)

        # add the p to final chain
        nodes3 = [p, chain[len(chain)-1]]
        chain_frag3 = chain.copy()
        chain_frag3.nodes = nodes3
        chains_list.append(chain_frag3)
        
        
        return chains_list

    def make_sequence_of_chains(self, chain, split_method, minimization_results):
        if split_method == 'minima':
            chains = self._do_minima_based_split(chain, minimization_results)

        elif split_method == 'maxima':
            chains = self._do_maxima_based_split(chain, minimization_results)

        return chains


    def stitch_elem_steps(self, list_of_chains):
        out_list_of_chains = [chain for chain in list_of_chains if chain is not None]
        return Chain.from_list_of_chains(
            out_list_of_chains, parameters=self.chain_inputs
        )
        

        
    
    
    



