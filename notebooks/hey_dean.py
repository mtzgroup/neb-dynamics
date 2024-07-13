# # BFGS

import numpy as np
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.nodes.Node2d import Node2D_Flower, Node2D
from neb_dynamics.nodes.Node3D_TC import Node3D_TC
from nodes.node3d import Node3D
from neb_dynamics.optimizers.LBFGS import LBFGS
from itertools import product
import matplotlib.pyplot as plt

from neb_dynamics.optimizers.BFGS import BFGS


# +
def func(x):
    x, y = x
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

def grad_func(x):
    x, y = x
    dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    return np.array([dx, dy])



# +
#### get energies for countourplot
gridsize = 100
min_val = -4
max_val = 4
# min_val = -.05
# max_val = .05

x = np.linspace(start=min_val, stop=max_val, num=gridsize)
y = x.reshape(-1, 1)

# +
ind = 1

the_noise = [-1,1]

noises_bool = [
    True,
    False

]




start_points = [
     [-2.59807434, -1.499999  ],
    [-3.77931026, -3.283186  ]
]

end_points = [
    [2.5980755 , 1.49999912],
    [2.99999996, 1.99999999]

]
tols = [
    0.1,
    0.05,

]

step_sizes = [
    1,
    1
]


k_values = [
    1,#.05,
    50

]



nodes = [Node2D_Flower, Node2D]
node_to_use = nodes[ind]
start_point = start_points[ind]
end_point = end_points[ind]
tol = tols[ind]

ss = step_sizes[ind]
ks = k_values[ind]
do_noise = noises_bool[ind]
# -

h_flat_ref = np.array([node_to_use.en_func_arr(pair) for pair in product(x,x)])
h_ref = h_flat_ref.reshape(gridsize,gridsize).T

# +
max_steps=2000


# x_prev = np.array([1,1])
x_prev = np.array([1,-2.5])
# x_prev = np.array([0.1,3]) # almost a TS.... weird
# x_prev = np.array([-1.5,1])
# x_prev = np.array([-1,-1])


ss = 1
m_ss = .01

# opt = BFGS(x0=x_prev, grad_func=grad_func, en_func=func, step_size=ss, min_step_size=m_ss, bfgs_flush_steps=1000000000, bfgs_flush_thre=-20)
# opt = LBFGS(func=func,func_args={'x_0':x_prev}, grad_func=grad_func, m=100)
opt = BFGS(x0=x_prev, grad_func=grad_func, en_func=func, step_size=ss, min_step_size=m_ss, bfgs_flush_steps=max_steps, bfgs_flush_thre=.9)
# opt = BFGS(x0=x_prev, grad_func=grad_func, en_func=func, step_size=ss, min_step_size=m_ss, bfgs_flush_steps=1, bfgs_flush_thre=0)

# opt.optimize()

traj = []
for i in range(max_steps):
    x_prime = opt.bfgs_step(x_prev)
    traj.append(x_prime)
    if np.linalg.norm(x_prime - x_prev) < .01 and np.linalg.norm(grad_func(x_prime)) < .01:
        print(f"converged in {i} steps")
        break
    # print(opt.bfgs_hess)
    x_prev = x_prime


# +
fig = 8
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
# x = np.linspace(start=min_val, stop=max_val, num=1000)
# y = x.reshape(-1, 1)

cs = ax.contourf(x, x, h_ref, cmap="Greys",alpha=.9)
# cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

# plt.plot([p[0] for p in opt.x_traj],[p[1] for p in opt.x_traj],'o-')
plt.plot([p[0] for p in traj],[p[1] for p in traj],'o-')
# plt.plot([p[0] for p in opt.x_traj],[p[1] for p in opt.x_traj],'o-')
# plot_chain(n_ref.initial_chain, c='orange',label='initial guess')
# plot_chain(n_ref.chain_trajectory[-1], c='green',linestyle='-',label=f'NEB({nimages} nodes)')
# plot_chain(n_ref_long.chain_trajectory[-1], c='gold',linestyle='-',label=f'NEB({nimages_long} nodes)', marker='*', ms=12)
# plot_chain(out_chain, c='red',marker='o',linestyle='-',label='AS-NEB')

# plt.legend(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
# plt.savefig(f"/home/jdep/T3D_data/msmep_draft/figures/results_2D_potential_ind{ind}.svg")
plt.show()
# -

# # MEP Nitrophenyk

from retropaths.abinitio.tdstructure import TDStructure
from retropaths.abinitio.trajectory import Trajectory
import matplotlib.pyplot as plt
from neb_dynamics.NEB import NEB
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import NEBInputs, ChainInputs
import warnings
warnings.filterwarnings('ignore')

start = TDStructure.from_xyz("./dean_nitrophenyl/start.xyz")
end = TDStructure.from_xyz("./dean_nitrophenyl/end.xyz")

start_opt = start.xtb_geom_optimization()
# gi = Trajectory([end, start]).run_geodesic(nimages=15)
gi = Trajectory([end, start_opt]).run_geodesic(nimages=15)

plt.plot(gi.energies_xtb(), 'o-')

# +
initial_guess = Chain.from_traj(gi, parameters=ChainInputs(k=0.0001))

n = NEB(initial_guess, parameters=NEBInputs(grad_thre=0.001, rms_grad_thre=0.0005,en_thre=0.0005, v=True))
# -

n.optimized.to_trajectory().write_trajectory("./dean_nitrophenyl/neb_opt.xyz")

n.plot_opt_history(do_3d=False)

n.chain_trajectory[-1].plot_chain()

n.optimized.plot_chain()

n.optimized.to_trajectory()

# +
# start.tc_model_basis = "6-31gs"
# start.tc_model_method = "wpbe"
# start.tc_kwds = {
#     "rc_w":0.3,
#     "precision": "mixed",
#     "hhtda":"yes",
#     "hhtdasinglets":4,
#     "cistarget":1,
#     "charge":-2,
#     "spinmult":1,
#     "cisguessvecs":16,
#     "cisnumstates":4,
#     'cistarget':1,
#     "cismaxiter":1000
# }

# +
# def gradient_tc_x(self):
#     atomic_input = self._prepare_input(method="grad")
#     future_result = self.tc_client.compute(
#         atomic_input, engine="terachem_fe", queue=None
#     )
#     result = future_result.get()
#     return result

# +
# r = gradient_tc_x(start)

# +
# print(r.error.error_message)
# -

# # PRFO Shit

# +
from TS.TS_PRFO import TS_PRFO
from neb_dynamics.nodes.Node2d import Node2D
from nodes.node3d import Node3D
from neb_dynamics.Chain import Chain
from retropaths.abinitio.tdstructure import TDStructure
from pathlib import Path
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
import numpy as np
from IPython.core.display import HTML

HTML('<script src="//d3js.org/d3.v3.min.js"></script>')
# -

node_att = Node2D(pair_of_coordinates=(2,3))
tsopt = TS_PRFO(initial_node=node_att, max_nsteps=10000, dr=0.1, grad_thre=1e-8)

node_att.do_geometry_optimization()

tsopt.ts.do_geometry_optimization()

tsopt.ts.do_geometry_optimization()

node_att = Node2D(pair_of_coordinates=[0.086, 2.8842547])
node_att.do_geometry_optimization()

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


chain.plot_chain()

from retropaths.abinitio.trajectory import Trajectory
import matplotlib.pyplot as plt

t = Trajectory.from_xyz(fp)
t.draw();

plt.plot(t.energies_xtb(),'o--')


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

# # Wittig

from retropaths.molecules.molecule import Molecule

# +
# mol_1 = Molecule.from_smiles("C(=O)(C)C")
# mol_1.draw()

# +
# mol_2 = Molecule.from_smiles("C=P(c1ccccc1)(c2ccccc2)c3ccccc3")
# mol_2.draw()
# -

# mol = Molecule.from_smiles("C(=O)(C)C.C=P(c1ccccc1)(c2ccccc2)c3ccccc3")
mol = Molecule.from_smiles("C(=O)(C)C.C=P(C)(C)C")
# mol2 = Molecule.from_smiles("CC(=C)C.c1ccc(cc1)P(=O)(c2ccccc2)c3ccccc3")
mol2 = Molecule.from_smiles("CC(=C)C.CP(=O)(C)C")

mol.draw()

mol2.draw()

# +
# mol2_1 = Molecule.from_smiles("CC(=C)C")
# mol2_1.draw()

# +
# mol2_2 = Molecule.from_smiles("c1ccc(cc1)P(=O)(c2ccccc2)c3ccccc3")
# mol2_2.draw()
# -

# root_1 = TDStructure.from_RP(mol_1)
# root_2 = TDStructure.from_RP(mol_2)
# target_1 = TDStructure.from_RP(mol2_1)
# target_2 = TDStructure.from_RP(mol2_2)
root = TDStructure.from_RP(mol)
target = TDStructure.from_RP(mol2)
# root = TDStructure.from_fp(Path("../example_cases/wittig/root.xyz"))
# target = TDStructure.from_fp(Path("../example_cases/wittig/target-aligned_to-root.xyz"))
root = TDStructure.from_fp(Path("../example_cases/wittig/root_small.xyz"))
target = TDStructure.from_fp(Path("../example_cases/wittig/target_small-aligned_to-root_small.xyz"))

root.mm_optimization("mmff94")
target.mm_optimization('mmff94')

root_opt = root.xtb_geom_optimization()

root_opt

target_opt = target.xtb_geom_optimization()

target_opt

traj = Trajectory(traj=[root_opt, target_opt])
gi = traj.run_geodesic(nimages=15,friction=.01)

gi.draw();

# +
# gi.write_trajectory(Path("../example_cases/wittig_gi.xyz"))
# -

plt.plot(gi.energies, 'o--')

gi.draw();

c = Chain.from_traj(traj=gi,k=0.1,delta_k=0,step_size=0.33,node_class=Node3D)

from neb_dynamics.NEB import NEB

n = NEB(initial_chain=c,max_steps=500)
n.optimize_chain()

n2 = NEB(initial_chain=n.chain_trajectory[-1],max_steps=500)
n2.optimize_chain()

n2.chain_trajectory[-1].plot_chain()

n3 = NEB(initial_chain=n2.chain_trajectory[-1],max_steps=500)
n3.optimize_chain()

n3.chain_trajectory[-1].plot_chain()

# +
# n2.write_to_disk("../example_cases/wittig_neb.xyz")
# -

n4 = NEB(initial_chain=n3.chain_trajectory[-1],max_steps=5000)
n4.optimize_chain()

c_temp = n3.chain_trajectory[-1]

c_temp[0].tdstructure.xtb_geom_optimization()

c_temp[0].tdstructure.xtb_geom_optimization()

c_temp[9].tdstructure.xtb_geom_optimization()

t = Trajectory([n.tdstructure for n in c_temp])

t.write_trajectory("../example_cases/wittig/c_temp.xyz")


c_temp2 = n4.chain_trajectory[-1]

c_temp2[5].tdstructure.xtb_geom_optimization()

# # Let's find this silly mech

n3.chain_trajectory[-1].plot_chain() # n3 was after 1500 steps

n3.write_to_disk("../example_cases/wittig/root_neb_chain.xyz")

# ### path to first minima

# +
c_to_split = n3.chain_trajectory[-1]

frag1 = Trajectory([c_to_split[0].tdstructure.xtb_geom_optimization(), c_to_split[5].tdstructure.xtb_geom_optimization()])
geo1 = frag1.run_geodesic(nimages=15, friction=0.01)
chain1 = Chain.from_traj(traj=geo1,k=0.1,delta_k=0,step_size=0.33,node_class=Node3D)

n_frag1 = NEB(initial_chain=chain1,max_steps=1500)
n_frag1.optimize_chain()
# -

n_frag1.chain_trajectory[-1].plot_chain()

n_frag1.write_to_disk("../example_cases/wittig/frag1_neb.xyz")

# ### path to first first minima (lmao)

# +
c_to_split2 = n_frag1.chain_trajectory[-1]

frag1_1 = Trajectory([c_to_split2[0].tdstructure.xtb_geom_optimization(), c_to_split2[8].tdstructure.xtb_geom_optimization()])
geo1_1 = frag1_1.run_geodesic(nimages=15, friction=0.01)
chain1_1 = Chain.from_traj(traj=geo1_1,k=0.1,delta_k=0,step_size=0.33,node_class=Node3D)
# -

n_frag1_1 = NEB(initial_chain=chain1_1,max_steps=1500)
n_frag1_1.optimize_chain()

n_frag1_1_continued = NEB(initial_chain=n_frag1_1.chain_trajectory[-1], max_steps=1500)
n_frag1_1_continued.optimize_chain()


