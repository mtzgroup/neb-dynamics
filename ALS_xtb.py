import numpy as np

from retropaths.abinitio.tdstructure import TDStructure

def ArmijoLineSearch(struct, grad, t, alpha, beta, f):
    
    max_steps = 10
    count=0
    
    try:
        struct_prime = TDStructure.from_coords_symbs(
            coords=struct.coords_bohr - t*grad,
            symbs=struct.symbols,
            tot_charge=struct.charge,
            tot_spinmult=struct.spinmult)
    
        en_struct_prime = f(struct_prime)
    except:
        t*= 0.1
        struct_prime = TDStructure.from_coords_symbs(
            coords=struct.coords_bohr - t*grad,
            symbs=struct.symbols,
            tot_charge=struct.charge,
            tot_spinmult=struct.spinmult)
    
        en_struct_prime = f(struct_prime)

    en_struct = f(struct)

    condition = (en_struct - (en_struct_prime + alpha*t*(np.linalg.norm(grad)**2) ) < 0)
    
        
    while condition and count<max_steps:
        t *= beta
        count+=1
        
        try:
            struct_prime = TDStructure.from_coords_symbs(
                coords=struct.coords_bohr - t*grad,
                symbs=struct.symbols,
                tot_charge=struct.charge,
                tot_spinmult=struct.spinmult)
        
            en_struct_prime = f(struct_prime)
        except:
            t*= 0.5
            struct_prime = TDStructure.from_coords_symbs(
                coords=struct.coords_bohr - t*grad,
                symbs=struct.symbols,
                tot_charge=struct.charge,
                tot_spinmult=struct.spinmult)
        
            en_struct_prime = f(struct_prime)

        en_struct = f(struct)

        condition = (en_struct - (en_struct_prime + alpha*t*(np.linalg.norm(grad)**2) ) < 0)
        
        print(f"{t=} {count=}")
    return t