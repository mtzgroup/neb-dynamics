import multiprocessing as mp
from dataclasses import dataclass
from typing import List

import numpy as np
from neb_dynamics.helper_functions import RMSD

AVAIL_DIST_FUNC_NAMES = ['per_node', 'simp_frechet']

@dataclass
class ChainBiaser:
    reference_chains: list # list of Chain objects
    
    amplitude: float = 1.0
    sigma: float = 1.0
    
    distance_func: str = "simp_frechet"
    
    def node_wise_distance(self, chain):
        
        ### this one only works if the number of nodes is identical
        if self.distance_func == 'per_node':
            tot_dist = 0
            for reference_chain in self.reference_chains:
                for n1, n2 in zip(chain.coordinates[1:-1], reference_chain.coordinates[1:-1]):
                    if chain[0].is_a_molecule:
                        tot_dist += RMSD(n1, n2)[0]
                    else:
                        tot_dist += np.linalg.norm(abs(n1 - n2))
            
        
        
        #### this one is a simplified frechet
        elif self.distance_func == 'simp_frechet':
            tot_dist = sum([self._simp_frechet_helper(coords) for coords in chain.coordinates[1:-1]]) 
        else:
            raise ValueError(f"Invalid distance func name: {self.distance_func}. Available are: {AVAIL_DIST_FUNC_NAMES}")
            
        return tot_dist / len(chain)
    
    def _simp_frechet_helper(self, coords):
        node_distances = []
        for reference_chain in self.reference_chains:
            ref_coords = reference_chain.coordinates[1:-1]
            for ref_coord in ref_coords:
                if len(ref_coord.shape) == 1: # 2d potentials, just an xy coordinate
                    dist = np.linalg.norm(coords - ref_coord)
                else:
                    dist = RMSD(coords, ref_coord)[0]
                
                
                node_distances.append(dist)
        return min(node_distances)
    
    def path_bias(self, distance):
        return self.amplitude*np.exp(-distance**2 / (2*self.sigma**2))
    
    def chain_bias(self, chain):
        dist_to_chain = self.node_wise_distance(chain)
        return self.path_bias(dist_to_chain)
    
    def grad_node_bias(self, chain, node, ind_node, dr=.1):
        grads = []
        
        if node.is_a_molecule:
            directions = ['x','y','z']
            n_atoms = len(node.coords)
            shape = n_atoms, len(directions)
        else:
            directions = ['x','y']
            n_atoms = 1
            shape = 2
        
        for i in range(n_atoms):
            for j, _ in enumerate(directions):
                disp_vector = np.zeros(len(directions)*n_atoms)
                disp_vector[i+j] += dr
            
                displaced_coord_flat = node.coords.flatten()+disp_vector
                displaced_coord = displaced_coord_flat.reshape(n_atoms, len(directions))
                node_disp_direction = node.update_coords(displaced_coord)
                fake_chain = chain.copy()
                fake_chain.nodes[ind_node] = node_disp_direction 

                grad_direction = self.chain_bias(fake_chain) - self.chain_bias(chain)
                grads.append(grad_direction)
                
        grad_node = np.array(grads).reshape(shape) / dr
        return grad_node
    
    def grad_chain_bias(self, chain):
        all_grads = []
        for ind_node, node in enumerate(chain[1:-1], start=1):
            grad_node = self.grad_node_bias(chain=chain, node=node, ind_node=ind_node)
            all_grads.append(grad_node)
        return np.array(all_grads)
    