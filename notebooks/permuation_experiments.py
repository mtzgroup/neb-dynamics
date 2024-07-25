# +
from neb_dynamics.treenode import TreeNode
from neb_dynamics.Inputs import ChainInputs, GIInputs
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from nodes.node3d import Node3D
from neb_dynamics.nodes.Node3D_TC_Local import Node3D_TC_Local
from neb_dynamics.nodes.Node3D_TC_TCPB import Node3D_TC_TCPB

from neb_dynamics.NEB_TCDLF import NEB_TCDLF
from chain import Chain
from neb_dynamics.Inputs import NEBInputs
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.helper_functions import create_friction_optimal_gi
from neb_dynamics.optimizers.SD import SteepestDescent

from pathlib import Path
import numpy as np
from neb_dynamics.NEB import NEB

from neb_dynamics.NEB import NoneConvergedException
from neb_dynamics.optimizers.BFGS import BFGS
from neb_dynamics.optimizers.Linesearch import Linesearch
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer


# -

def animate_chain_trajectory(chain_traj, min_y=-100, max_y=100,
                             max_x=1.1, min_x=-0.1, norm_path_len=True):
    # %matplotlib notebook
    import matplotlib.pyplot as plt
    import matplotlib.animation
    import numpy as np


    figsize = 5
    s=4

    fig, ax = plt.subplots(figsize=(1.618 * figsize, figsize))

    ax.set_xlim(min_x,max_x)
    ax.set_ylim(min_y, max_y)

    (line,) = ax.plot([], [], "o--", lw=3)


    def animate(chain):
            if norm_path_len:
                x = chain.integrated_path_length
            else:
                x = chain.path_length

            y = chain.energies_kcalmol



            color = 'lightblue'

            line.set_data(x, y)
            line.set_color("skyblue")




            return

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=chain_traj)


    from IPython.display import HTML
    return HTML(ani.to_jshtml())

h = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Claisen-Rearrangement-Aromatic/yesxtb_preopt_02132024/")
h2 = TreeNode.read_from_disk("/home/jdep/T3D_data/msmep_draft/comparisons_dft/structures/Claisen-Rearrangement-Aromatic/start_opt_2024_msmep/")

print([len(obj.chain_trajectory) for obj in h.get_optimization_history()])
sum([len(obj.chain_trajectory) for obj in h.get_optimization_history()])


print([len(obj.chain_trajectory) for obj in h2.get_optimization_history()])
sum([len(obj.chain_trajectory) for obj in h2.get_optimization_history()])


import matplotlib.pyplot as plt

ct = []
for obj in h2.get_optimization_history():
    ct.extend(obj.chain_trajectory)

animate_chain_trajectory(ct)

plt.plot(h2.output_chain.path_length, h2.output_chain.energies,'o-', label='GI-seeds')
plt.plot(h.output_chain.path_length, h.output_chain.energies,'o-', label='xtb-seeds')
plt.legend()

h.output_chain[0].is_identical(h2.output_chain[0])

tr = Trajectory.from_xyz("/home/jdep/T3D_data/msmep_draft/permutation_experiments/claisen_original.xyz")

nbi = NEBInputs(tol=0.001*BOHR_TO_ANGSTROMS, early_stop_force_thre=0.01*BOHR_TO_ANGSTROMS,
                early_stop_chain_rms_thre=1, v=True, max_steps=500)
# cni = ChainInputs(k=0.1, delta_k=0.09,node_class=Node3D_TC,node_freezing=True)
cni = ChainInputs(k=0.1, delta_k=0.09,node_class=Node3D,node_freezing=True)
gii = GIInputs(nimages=12)
optimizer = BFGS(bfgs_flush_steps=50, bfgs_flush_thre=0.80, use_linesearch=False,
                 step_size=3,
                 min_step_size= 0.1,
                 activation_tol=.1
            )

m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=optimizer)


def sort_td(td):
    coords = td.coords
    symbols = td.symbols
    sorted_inds = np.argsort(symbols)
    coords_sorted = coords[sorted_inds]
    symbols_sorted = symbols[sorted_inds]

    td_sorted = TDStructure.from_coords_symbols(coords_sorted, symbols_sorted, tot_charge=td.charge,
                                            tot_spinmult=td.spinmult)
    td_sorted.update_tc_parameters(td)
    return td_sorted


# +
# tsg = TDStructure.from_xyz("/home/jdep/tsg_elim_w_alk_shift.xyz")

# +
# tsg.tc_model_method = 'wb97xd3'
# tsg.tc_model_basis = 'def2-svp'

# +
# ts = tsg.tc_geom_optimization('ts')
# -

def shuffle_element_td(td, elem, n_to_shuffle=10000000000):
    inds_element = np.nonzero(td.symbols == elem)[0][:n_to_shuffle]
    np.random.seed(0)
    permutation_element = np.random.permutation(inds_element)
    inds_original = np.arange(len(td.symbols))
    inds_permuted = inds_original.copy()
    inds_permuted[inds_element] = permutation_element
    coords_permuted = td.coords[inds_permuted]
    td_shuffled = TDStructure.from_coords_symbols(coords_permuted, td.symbols, tot_charge=td.charge,
                                            tot_spinmult=td.spinmult)
    td_shuffled.update_tc_parameters(td)
    return td_shuffled


td_sorted = sort_td(tr[0])

non_permuted_end =  sort_td(tr[-1])

permuted_start = shuffle_element_td(sort_td(tr[0]),'C', n_to_shuffle=3)

tr_permuted = Trajectory([permuted_start, non_permuted_end]).run_geodesic(nimages=12)

chain = Chain.from_traj(tr_permuted, cni)

h, out = m.find_mep_multistep(chain)

# +
# h.write_to_disk(Path("/home/jdep/T3D_data/msmep_draft/permutation_experiments/claisen_permuted_2C_seed0/"))
# -


