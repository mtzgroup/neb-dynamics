from __future__ import annotations


from dataclasses import dataclass, field
from typing import Callable
from neb_dynamics.Optimizer import Optimizer
from neb_dynamics.optimizers import ALS
from neb_dynamics.Chain import Chain
import numpy as np

@dataclass
class VelocityProjectedOptimizer(Optimizer):
    als_max_steps: int = 3
    step_size: float = 1.0
    min_step_size: float = 0.33
    alpha: float = 0.01
    beta: float = None
    
    def __post_init__(self):
        if self.beta is None:
            beta = (self.min_step_size / self.step_size)**(1/self.als_max_steps)
            self.beta = beta
    
    
    def optimize_step(self, chain, chain_gradients):
        prev_velocity = chain.velocity
        step = ALS.ArmijoLineSearch(
            chain=chain,
            t=self.step_size,
            alpha=self.alpha,
            beta=self.beta,
            grad=chain_gradients,
            max_steps=self.als_max_steps)
         
        # step = 1
         
        new_force = -(chain_gradients)    
        
        orig_shape = new_force.shape
        new_force_flat = new_force.flatten()
        prev_velocity_flat = prev_velocity.flatten()
        projection = np.dot(new_force_flat, prev_velocity_flat)
        if  projection < 0: 
            new_vel = np.zeros_like(new_force)
        else:
            new_vel_flat =  projection*new_force_flat   
            new_vel = new_vel_flat.reshape(orig_shape)
            
        force = new_vel + new_force
        
        scaling = 1
        if np.linalg.norm(force*step) > self.step_size*len(chain): # if step size is too large
            scaling = (1/(np.linalg.norm(force*step)))*self.step_size*len(chain)
        
        

        new_chain_coordinates = chain.coordinates + force*step*scaling
        new_nodes = []
        for node, new_coords in zip(chain.nodes, new_chain_coordinates):

            new_nodes.append(node.update_coords(new_coords))

        new_chain = Chain(new_nodes, parameters=chain.parameters)
        new_chain.velocity = force*step*scaling
        
        return new_chain
        