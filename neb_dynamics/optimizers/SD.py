from __future__ import annotations


from dataclasses import dataclass
from neb_dynamics.Optimizer import Optimizer
from neb_dynamics.optimizers import ALS
from neb_dynamics.Chain import Chain
import numpy as np

@dataclass
class SteepestDescent(Optimizer):
    step_size_per_atom: float = 0.01
    
    
    def optimize_step(self, chain, chain_gradients):
        atomn = chain[0].coords.shape[0]
        
        max_disp = self.step_size_per_atom*atomn*len(chain)
        
        scaling = 1
        if np.linalg.norm(chain_gradients) > max_disp: # if step size is too large
            scaling = (1/(np.linalg.norm(chain_gradients)))*max_disp
        
        new_chain_coordinates = chain.coordinates - (chain_gradients * scaling)
        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = Chain(new_nodes, parameters=chain.parameters)
        return new_chain
        
        
        
        
        
