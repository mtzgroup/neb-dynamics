from dataclasses import dataclass
import numpy as np

@dataclass
class neb:
    
    def optimize_chain(
        chain,
        grad_func,
        en_func,
        k,

        en_thre = 0.01,
        grad_thre = 0.01,
        max_steps = 1000):


