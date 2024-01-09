from __future__ import annotations


from dataclasses import dataclass, field
from typing import Callable
from neb_dynamics.Optimizer import Optimizer
from neb_dynamics.optimizers import ALS
from neb_dynamics.Chain import Chain
import numpy as np

@dataclass
class VelocityProjectedOptimizer(Optimizer):
    # als_max_steps: int = 3
    # step_size_per_atom: float = 1.0
    timestep: float = 1.0
    activation_tol: float = 0.1
    # min_step_size: float = 0.33
    # alpha: float = 0.01
    # beta: float = None
    
    # def __post_init__(self):
    #     if self.beta is None:
    #         beta = (self.min_step_size / self.step_size)**(1/self.als_max_steps)
    #         self.beta = beta
    
    
    def optimize_step(self, chain, chain_gradients):
        prev_velocity = chain.velocity
        atomn = chain[0].coords.shape[0]
        max_disp = self.timestep*atomn*len(chain)
        new_force = -(chain_gradients) 
        new_force_unit = new_force / np.linalg.norm(new_force) 
        timestep = self.timestep #self.step_size_per_atom*atomn*len(chain)
        if np.linalg.norm(new_force) < self.activation_tol:
            if not np.all(prev_velocity.flatten()==0):
                orig_shape = new_force.shape
                prev_velocity_flat = prev_velocity.flatten()
                projection = np.dot(prev_velocity_flat, new_force_unit.flatten())
                vj_flat = projection*new_force_unit.flatten()
                vj = vj_flat.reshape(orig_shape)
                # print(f'\n{projection}')
                if  projection < 0: 
                    # print(f"\nproj={projection} Reset!")
                    vj = np.zeros_like(new_force) 
                
                force = timestep*vj
                
            else:
                vj = np.zeros_like(new_force)
                force = timestep*new_force
        else:
            vj = np.zeros_like(new_force)
            force = timestep*new_force
            
        scaling = 1
        if np.linalg.norm(force) > max_disp: # if step size is too large
            scaling = (1/(np.linalg.norm(force)))*max_disp
        
        

        new_chain_coordinates = chain.coordinates + force*scaling
        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = Chain(new_nodes, parameters=chain.parameters)
        new_vel = vj + timestep*force*scaling
        new_chain.velocity = new_vel
        
        return new_chain
        