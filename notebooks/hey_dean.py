from neb_dynamics.NEB import Node2D, TS_PRFO, Node3D, Chain
from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
import numpy as np

node_att = Node2D(pair_of_coordinates=(-2,1))
tsopt = TS_PRFO(initial_node=node_att, max_nsteps=10000, dr=0.1, grad_thre=1e-8)

print(tsopt.ts.coords), tsopt.plot_path()

# +
# numatoms = 1
# # for n in range(numatoms):
# approx_hess = []
# # for n in range(1):
# grad_n = []
# for coord_ind, coord_name in enumerate(['dx', 'dy']):
#     print(f"doing atom #{n} | {coord_name}")
    
#     coords = np.array(node_att.coords, dtype='float64')
#     print(coords, coord_ind, coords[coord_ind])
#     coords[coord_ind] = coords[coord_ind] + 1
#     print(coords)

#     node2 = node_att.copy()
#     node2  = node2.update_coords(coords)

#     delta_grad = node2.gradient - node_att.gradient 
#     print(delta_grad)
#     grad_n.extend(delta_grad)
# approx_hess.append(grad_n)

# approx_hess = np.array(approx_hess)
# approx_hess
# -

# # Now with 3D molecules...

# ### Claisen Rearrangement

# +
# td.write_to_disk("claise_ts_guess.xyz")

# +
# fp = Path("/Users/janestrada/wtf_is_up_with_cneb/ts_cneb_40_nodes.xyz")
fp = Path("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-NEB/traj_0-0_neb.xyz")

chain = Chain.from_xyz(fp)
# -


chain[9]

chain.plot_chain()


# +
# td = TDStructure.from_fp(fp)
# td
# -

def change_to_bohr(td):
    td_node = td.copy()
    td_node.update_coords(td_node.coords_bohr)
    return td_node


def change_to_angstroms(td):
    td_node = td.copy()
    td_node.update_coords(td_node.coords*BOHR_TO_ANGSTROMS)
    return td_node


# +
# td_node = td.copy()
# td_node.update_coords(td_node.coords_bohr)
# node = Node3D(tdstructure=td_node)
# -

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

# +
# # def heydean(inp):
# inp = node

# dr = 0.01 # displacement vector, Bohr
# numatoms = inp.tdstructure.atomn
# approx_hess = []
# for n in range(numatoms):
# # for n in range(2):
#     grad_n = []
#     for coord_ind, coord_name in enumerate(['dx', 'dy', 'dz']):
#         coords = np.array(inp.coords, dtype='float64')
#         coords[n, coord_ind] = coords[n, coord_ind] + dr

#         node2 = inp.copy()
#         node2 = node2.update_coords(coords)

#         delta_grad = (node2.gradient - inp.gradient ) / dr
#         grad_n.append(delta_grad.flatten())

#     approx_hess.extend(grad_n)
# approx_hess = np.array(approx_hess)
# approx_hess


# # approx_hess_sym = 0.5*(approx_hess + approx_hess.T)
# # assert check_symmetric(approx_hess_sym, rtol=1e-3, atol=1e-3), 'Hessian not symmetric for some reason'

# +
# approx_hess[0, :]

# +
# node = chain[9]
# -



# +
# TDStructure(change_to_angstroms(node.tdstructure).molecule_obmol)
# -

tsopt = TS_PRFO(initial_node=chain[9], max_nsteps=3000, dr=.05, grad_thre=1e-5)

tsopt.ts

new_td = tsopt.ts.tdstructure

new_coords = new_td.coords*BOHR_TO_ANGSTROMS
new_td.update_coords(new_coords)

new_td

# +
# new_td.write_to_disk("claisen_ts_2.xyz")
# -

# # Diels Alder


