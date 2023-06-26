import multiprocessing as mp
from dataclasses import dataclass
from typing import List

import numpy as np

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
                node_distances.append(np.linalg.norm(coords - ref_coord))
        return min(node_distances)
    
    def path_bias(self, distance):
        return self.amplitude*np.exp(-distance**2 / (2*self.sigma**2))
    
    def chain_bias(self, chain):
        dist_to_chain = self.node_wise_distance(chain)
        return self.path_bias(dist_to_chain)
    
    def grad_node_bias(self, chain, node, ind_node, dr=.1):
        if node.is_a_molecule:
            raise NotImplementedError
        
        grads = []
        directions = ['x','y']
        for i, _ in enumerate(directions):
        
            disp_vector = np.zeros(len(directions))
            disp_vector[i] += dr
        
            node_disp_direction = node.update_coords(node.coords+disp_vector)
            fake_chain = chain.copy()
            fake_chain.nodes[ind_node] = node_disp_direction 

            grad_direction = self.chain_bias(fake_chain) - self.chain_bias(chain)
            grads.append(grad_direction)
        grad_node = np.array(grads) / dr

        return grad_node
    
    def grad_chain_bias(self, chain):
        all_grads = []
        for ind_node, node in enumerate(chain):
            grad_node = self.grad_node_bias(chain=chain, node=node, ind_node=ind_node)
            all_grads.append(grad_node)
        return np.array(all_grads)
    