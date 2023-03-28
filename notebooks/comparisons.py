from pathlib import Path
from retropaths.abinitio.trajectory import Trajectory
from retropaths.abinitio.tdstructure import TDStructure
import numpy as np
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs, NEBInputs, GIInputs
from neb_dynamics.NEB import NEB
from neb_dynamics.Node2d import Node2D_Flower, Node2D
from neb_dynamics.Node3D_TC import Node3D_TC
from itertools import product
import matplotlib.pyplot as plt

from neb_dynamics.MSMEP import MSMEP
from IPython.core.display import HTML
HTML('<script src="//d3js.org/d3.v3.min.js"></script>')

# # Variables

NIMAGES = 15


# # Helper Functions

def get_eA(chain_energies):
    return max(chain_energies) - chain_energies[0]


# +
def plot_chain(chain,linestyle='--',ax=None, marker='o',**kwds):
    if ax:
        ax.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)
    else:
        plt.plot(chain.coordinates[:,0],chain.coordinates[:,1],linestyle=linestyle,marker=marker,**kwds)

        
def plot_neb(neb,linestyle='--',marker='o',ax=None,**kwds):
    plot_chain(chain=neb.chain_trajectory[-1],linestyle='-',marker=marker,ax=ax,**kwds)
    plot_chain(chain=neb.initial_chain,linestyle='--',marker=marker,ax=ax,**kwds)


# -

# # 2D potentials

# +
ind = 0

noises_bool = [
    False,#True,
    False

]




start_points = [
     [-2.59807434, -1.499999  ],
    [-3.77931026, -3.283186  ]
]

end_points = [
    [0,3],#[2.5980755 , 1.49999912],
    [2.99999996, 1.99999999]

]
tols = [
    0.1,
    .5,

]

step_sizes = [
    .1,
    1

]


k_values = [
    .05,
    1

]



nodes = [Node2D_Flower, Node2D]
node_to_use = nodes[ind]
start_point = start_points[ind]
end_point = end_points[ind]
tol = tols[ind]

ss = step_sizes[ind]
ks = k_values[ind]
do_noise = noises_bool[ind]

# +
nimages = NIMAGES
np.random.seed(0)



coords = np.linspace(start_point, end_point, nimages)
if do_noise:
    coords[1:-1] += [-1,1] # i.e. good initial guess
cni_ref = ChainInputs(
    k=ks,
    node_class=node_to_use,
    delta_k=0,
    step_size=ss,
    do_parallel=False,
    use_geodesic_interpolation=False,
)
gii = GIInputs(nimages=nimages)
nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, stopping_threshold=0)
chain_ref = Chain.from_list_of_coords(list_of_coords=coords, parameters=cni_ref)
# -

n_ref = NEB(initial_chain=chain_ref,parameters=nbi )
n_ref.optimize_chain()

gii = GIInputs(nimages=nimages)
nbi_msmep = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, stopping_threshold=3)
m = MSMEP(neb_inputs=nbi_msmep,chain_inputs=cni_ref, gi_inputs=gii,split_method='minima',recycle_chain=True, root_early_stopping=True)
history, out_chain = m.find_mep_multistep(chain_ref)

# +
nimages_long = len(out_chain)

coords_long = np.linspace(start_point, end_point, nimages_long)
if do_noise:
    coords_long[1:-1] += [-1,1] # i.e. good initial guess
gii = GIInputs(nimages=nimages_long)
nbi = NEBInputs(tol=tol, v=1, max_steps=4000, climb=False, stopping_threshold=0)
chain_ref_long = Chain.from_list_of_coords(list_of_coords=coords_long, parameters=cni_ref)

n_ref_long = NEB(initial_chain=chain_ref_long,parameters=nbi )
n_ref_long.optimize_chain()
# -

#### get energies for countourplot
gridsize = 100
# min_val = -5.3
# max_val = 5.3
min_val = -4
max_val = 4
x = np.linspace(start=min_val, stop=max_val, num=gridsize)
y = x.reshape(-1, 1)

h_flat_ref = np.array([node_to_use.en_func_arr(pair) for pair in product(x,x)])
h_ref = h_flat_ref.reshape(gridsize,gridsize).T

# +
fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
f, ax = plt.subplots(figsize=(1.3 * fig, fig),ncols=1)
# x = np.linspace(start=min_val, stop=max_val, num=1000)
# y = x.reshape(-1, 1)

# cs = ax[0].contourf(x, x, h_ref, cmap="Greys",alpha=.8)
cs = ax.contourf(x, x, h_ref,alpha=1)
_ = f.colorbar(cs)

plot_chain(n_ref.initial_chain, c='orange',label='initial guess')
plot_chain(n_ref.chain_trajectory[-1], c='skyblue',linestyle='-',label='neb(short)')
plot_chain(out_chain, c='red',marker='o',linestyle='-',label='as-neb')
plot_chain(n_ref_long.chain_trajectory[-1], c='yellow',linestyle='-',label='neb(long)')
plt.legend()
plt.show()
# -

n_steps_orig_neb = len(n_ref.chain_trajectory)
n_steps_msmep = sum([len(obj.chain_trajectory) for obj in history.get_optimization_history()]) 
n_steps_long_neb = len(n_ref_long.chain_trajectory)

fig = 8
min_val = -5.3
max_val = 5.3
fs = 18
plt.figure(figsize=(1.16*fig,fig))
plt.bar(x=["AS-NEB","NEB","NEB(many nodes)"],
       height=[n_steps_msmep, n_steps_orig_neb, n_steps_long_neb])
plt.yticks(fontsize=fs)
plt.ylabel("Number of optimization steps",fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

import pandas as pd

with open("/home/jdep/T3D_data/msmep_draft/comparisons/dlfind/Wittig/scr.initial_guess.1/nebinfo") as f:
    data_str = f.read().splitlines()

data_raw = [line.split() for line in data_str[1:]]

data = []
for row in data_raw:
    new_row = [float(x) for x in row]
    data.append(new_row)

df = pd.DataFrame(data,columns=['path_len','energy','work'])

df.plot('path_len','energy',kind='line')

# # Make endpoints

out_dir = Path("/home/jdep/T3D_data/msmep_draft/comparisons/structures")

import retropaths.helper_functions  as hf
reactions = hf.pload("/home/jdep/retropaths/data/reactions.p")
m = MSMEP()

r, p = m.create_endpoints_from_rxn_name("Aza-Grob-Fragmentation-X-Bromine", reactions)

# +
rxn_name = "Aza-Grob-Fragmentation-X-Bromine"
reactions_object = reactions

rxn = reactions_object[rxn_name]
root = TDStructure.from_rxn_name(rxn_name, reactions_object)

c3d_list = root.get_changes_in_3d(rxn)

root = root.pseudoalign(c3d_list)
root.gum_mm_optimization()
root = root.xtb_geom_optimization()

target = root.copy()
target.apply_changed3d_list(c3d_list)
target.mm_optimization("gaff", steps=5000)
target.mm_optimization("uff", steps=5000)
# target = target.xtb_geom_optimization()


# -

target

#### extract templates with 1 reactant and no charges
single_mol_reactions = []
for rn in reactions:
    rxn = reactions[rn]
    n_reactants = len(rxn.reactants.force_smiles().split("."))
    if n_reactants == 1:
        if rxn.reactants.charge == 0:
            single_mol_reactions.append(rn)

#### create endpoints
failed = []
for rn in single_mol_reactions:
    try:
        rn_output_dir = out_dir / rn
        if rn_output_dir.exists():
            print(f"already did {rn}")
            continue
        rn_output_dir.mkdir(parents=False, exist_ok=True)
        r, p = m.create_endpoints_from_rxn_name(rn, reactions)
        r_opt, p_opt = r.xtb_geom_optimization(), p.xtb_geom_optimization()
        
        
        r_opt.to_xyz(rn_output_dir / "start.xyz")
        p_opt.to_xyz(rn_output_dir / "end.xyz")
    except:
        print(f"{rn} did not work. Check it out.")
        failed.append(rn)

# +
#### create gis
failed = []

frics = [0.001,0.01, 0.1, 1]

for rn in single_mol_reactions:
    # try:
    rn_output_dir = out_dir / rn
    output_file = rn_output_dir / "initial_guess.xyz"
    if output_file.exists():
        print(f'{output_file} exists')
        continue
    else:
        
        start_fp = rn_output_dir/"start.xyz"
        end_fp = rn_output_dir/"end.xyz"
        start = TDStructure.from_xyz(str(start_fp.resolve()))
        end = TDStructure.from_xyz(str(end_fp.resolve()))
        trajs = []
        for fric in frics:
            traj = Trajectory([start, end]).run_geodesic(nimages=NIMAGES, friction=fric)
            traj.write_trajectory(rn_output_dir / f'gi_fric{fric}.xyz')
            trajs.append(traj)

        eAs = [get_eA(t.energies_xtb()) for t in trajs]
        best_gi_ind = np.argmin(eAs)
        best_gi = trajs[best_gi_ind]
        print(f"Best {rn} gi had friction {frics[best_gi_ind]}")
        best_gi.write_trajectory(output_file)

            
                
        
        
#     except:
#         print(f"{rn} did not work. Check it out.")
#         failed.append(rn)
# -

# # write scripts for sbatching


