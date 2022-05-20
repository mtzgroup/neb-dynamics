import numpy as np

from retropaths.abinitio.tdstructure import TDStructure

def ArmijoLineSearch(struct, grad, t, alpha, beta, f):
    # """Minimize over alpha, the function ``f(xₖ + αpₖ)``.
    # α > 0 is assumed to be a descent direction.

    # Parameters
    # --------------------
    # f : callable
    #     Function to be minimized.
    # xk : array
    #     Current point.
    # gfk : array
    #     Gradient of `f` at point `xk`.
    # phi0 : float
    #     Value of `f` at point `xk`.
    # alpha0 : scalar
    #     Value of `alpha` at the start of the optimization.
    # rho : float, optional
    #     Value of alpha shrinkage factor.
    # c1 : float, optional
    #     Value to control stopping criterion.

    # Returns
    # --------------------
    # alpha : scalar
    #     Value of `alpha` at the end of the optimization.
    # phi : float
    #     Value of `f` at the new point `x_{k+1}`.
    # """
    # derphi0 = np.tensordot(gfk, pk)
    # new_coords = xk.coords + alpha0 * pk
    # new_xk = TDStructure.from_coords_symbs(coords=new_coords, symbs=xk.symbols, tot_charge=xk.charge, tot_spinmult=xk.spinmult)
    # phi_a0 = f(new_xk)
    
    # # print(f"{phi_a0=} {phi0=} {c1=} {alpha0=} {derphi0=}")
    # # print(f"da shit {phi_a0 <= phi0 + c1 * alpha0 * derphi0}")
    # while not phi_a0 <= phi0 + c1 * alpha0 * derphi0:
    #     print(f"{phi_a0=} > {phi0 + c1 * alpha0 * derphi0}")
    #     alpha0 = alpha0 * rho
    #     new_coords = new_xk.coords + alpha0 * pk
    #     new_xk = TDStructure.from_coords_symbs(coords=new_coords, symbs=new_xk.symbols, tot_charge=new_xk.charge, tot_spinmult=new_xk.spinmult)
    #     phi_a0 = f(new_xk)


    # return alpha0, phi_a0
    max_steps = 10
    count=0
    
    try:
        struct_prime = TDStructure.from_coords_symbs(
            coords=struct.coords - t*grad,
            symbs=struct.symbols,
            tot_charge=struct.charge,
            tot_spinmult=struct.spinmult)
    
        en_struct_prime = f(struct_prime)
    except:
        t*= 0.1
        struct_prime = TDStructure.from_coords_symbs(
            coords=struct.coords - t*grad,
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
                coords=struct.coords - t*grad,
                symbs=struct.symbols,
                tot_charge=struct.charge,
                tot_spinmult=struct.spinmult)
        
            en_struct_prime = f(struct_prime)
        except:
            t*= 0.5
            struct_prime = TDStructure.from_coords_symbs(
                coords=struct.coords - t*grad,
                symbs=struct.symbols,
                tot_charge=struct.charge,
                tot_spinmult=struct.spinmult)
        
            en_struct_prime = f(struct_prime)

        en_struct = f(struct)

        condition = (en_struct - (en_struct_prime + alpha*t*(np.linalg.norm(grad)**2) ) < 0)
        
        print(f"{t=} {count=}")
    return t