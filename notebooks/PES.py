# +
from neb_dynamics.treenode import TreeNode
from neb_dynamics.tdstructure import TDStructure
import numpy as np

from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons/structures/Wittig/ASNEB_03_NOSIG_NOMR/")


# +
def steepest_descent_freeze_nodes(td: TDStructure, node_indices, tol=0.0001, ss=1):
    grad = td.gradient_xtb()
    grad[node_indices] = [0,0,0]
    count = 0
    while np.amax(abs(grad)) >= tol and count < 5000:
        try:
            td = td.displace_by_dr(ss*(-1)*grad)
            grad = td.gradient_xtb()
            grad[node_indices] = [0,0,0]

            count += 1
            # print(count)
        except:
            return td
    # print(np.amax(abs(grad)))
    return td
        
        
    
    
# -

td = steepest_descent_freeze_nodes(td=h.output_chain[4].tdstructure, node_indices=[0, 7])

np.amax(abs(td.gradient_xtb()))

PO_indices = [0, 7]
CC_indices = [1, 6]


def compute_dist(td, ind_pair):
    dist_vec = td.coords[ind_pair][0] - td.coords[ind_pair][1]
    dist = np.sqrt(dist_vec@dist_vec)/2
    return dist


import matplotlib.pyplot as plt


@np.vectorize
def get_displaced_td(scaling1, scaling2):
    td = h.output_chain[0].tdstructure
    dr = np.zeros_like(td.coords)
    
    ind_pair1 = PO_indices
    ind_pair2 = CC_indices

    vec1 = td.coords[ind_pair1[0]] - td.coords[ind_pair1[1]] 
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    curr_dist1 = np.sqrt(vec1@vec1) / 2
    target_dist1 = scaling1
    diff1 = target_dist1 - curr_dist1
    
    
    vec2 = td.coords[ind_pair2[0]] - td.coords[ind_pair2[1]] 
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    curr_dist2 = np.sqrt(vec2@vec2) / 2
    target_dist2 = scaling2
    diff2 = target_dist2 - curr_dist2
    
    dr[ind_pair1[0]] = (unit_vec1*diff1)
    dr[ind_pair1[1]] = (-1*unit_vec1*diff1)
    
    dr[ind_pair2[0]] = (unit_vec2*diff2)
    dr[ind_pair2[1]] = (-1*unit_vec2*diff2)

    

    new_td = td.displace_by_dr(dr)
    return new_td

np.amax(h.output_chain[0].tdstructure.gradient_xtb())


@np.vectorize
def do_opts_get_ens(td):
    td_opt = steepest_descent_freeze_nodes(td, node_indices=PO_indices+CC_indices)
    return td_opt.energy_xtb()


x = np.arange(0.6, 2.2, step=0.1)
y = np.arange(0.6, 2.2, step=0.1)

xv, yv = np.meshgrid(x, y)

td_mat = get_displaced_td(xv, yv)

mat_ens = do_opts_get_ens(td_mat)

mat_ens_scaled = (mat_ens - min(mat_ens.flatten()))*627.5

start_ind = 0
plt.contourf(xv[start_ind:,start_ind:], yv[start_ind:,start_ind:], mat_ens_scaled[start_ind:,start_ind:], vmax=80)
plt.plot([compute_dist(td.tdstructure, PO_indices) for td in h.output_chain],
         [compute_dist(td.tdstructure, CC_indices) for td in h.output_chain],'o-')
plt.colorbar()
# plt.xlim(0.80, 0.9)
# plt.ylim(0.60, 1.8)

PO_DISTS_CHAIN = [compute_dist(td.tdstructure, PO_indices) for td in h.output_chain]
CC_DISTS_CHAIN = [compute_dist(td.tdstructure, CC_indices) for td in h.output_chain]

PO_DISTS_CHAIN

CC_DISTS_CHAIN

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, yv, mat_ens_scaled)
ax.view_init(10, 60)
ax.scatter3D(PO_DISTS_CHAIN, CC_DISTS_CHAIN, h.output_chain.energies,
            'o-',s=300, color='yellow')

from neb_dynamics.trajectory import Trajectory

Trajectory(td_mat[:, )

steepest_descent_freeze_nodes(td_mat[1,1],[])


