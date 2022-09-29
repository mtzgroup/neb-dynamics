from neb_dynamics.NEB import Chain, Node2D
import numpy as np
import matplotlib.pyplot as plt


# # Define Plots and such

def newton_raph_step(inp):
    grad = grad_func(inp)
    h = hess(inp)
    EVALS, EVECS = np.linalg.eigh(h)
    EVECS = EVECS.T # so that EVECS[0] is actually the 0th eigenvector

    fis = [np.dot(EVECS[i], grad) for i, _ in enumerate(grad)]


    his = [-1*fis[i]*EVECS[i]/EVALS[i] for i, _ in enumerate(grad)]
    h_mod = his[0]+his[1]
    return h_mod
newton_raph_step((-2,-1))


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


# # Let's build an approximate Hessian

from retropaths.abinitio.tdstructure import TDStructure
from neb_dynamics.NEB import Node3D
from pathlib import Path
import numpy as np

fp = Path("/Users/janestrada/wtf_is_up_with_cneb/ts_cneb_40_nodes.xyz")
td = TDStructure.from_fp(fp)

td_node = td.copy()
td_node.update_coords(td_node.coords_bohr)
node = Node3D(tdstructure=td_node)

node.hessian

td

grad = td.gradient_xtb()
grad


# +
def hessian(node):
    dr = 0.01 # displacement vector, Angstroms
    numatoms = node.tdstructure.atomn
    approx_hess = []
    for n in range(numatoms):
    # for n in range(2):
        grad_n = []
        for coord_ind, coord_name in enumerate(['dx', 'dy', 'dz']):
            # print(f"doing atom #{n} | {coord_name}")

            coords = np.array(node.coords, dtype='float64')

            # print(coord_ind, coords[n, coord_ind])
            coords[n, coord_ind] = coords[n, coord_ind] + dr
            # print(coord_ind, coords[n, coord_ind])

            node2 = node.copy()
            node2 = node2.update_coords(coords)

            delta_grad = node2.gradient - node.gradient 
            # print(delta_grad)
            grad_n.extend(delta_grad)


        approx_hess.append(grad_n)
    approx_hess = np.array(approx_hess)
    approx_hess = approx_hess.reshape(3*numatoms, 3*numatoms)
    
    assert check_symmetric(approx_hess, rtol=1e-3, atol=1e-3), 'Hessian not symmetric for some reason'
    return approx_hess

hessian(node)
# -



dr = 0.01 # displacement vector, Angstroms
numatoms = td.atomn
approx_hess = []
for n in range(numatoms):
# for n in range(2):
    grad_n = []
    for coord_ind, coord_name in enumerate(['dx', 'dy', 'dz']):
        print(f"doing atom #{n} | {coord_name}")

        coords = np.array(td.coords, dtype='float64')
        
        # print(coord_ind, coords[n, coord_ind])
        coords[n, coord_ind] = coords[n, coord_ind] + dr
        # print(coord_ind, coords[n, coord_ind])

        td2 = td.copy()
        td2.update_coords(coords)

        delta_grad = td2.gradient_xtb() - td.gradient_xtb() 
        # print(delta_grad)
        grad_n.extend(delta_grad)

        
    approx_hess.append(grad_n)
approx_hess = np.array(approx_hess)
approx_hess = approx_hess.reshape(3*numatoms, 3*numatoms)
print(approx_hess.shape)
check_symmetric(approx_hess, rtol=1e-3, atol=1e-3)

evals, evecs = np.linalg.eigh(approx_hess)

evals

evecs[:, 0]


