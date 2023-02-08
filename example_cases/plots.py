# +
import matplotlib.pyplot as plt
import numpy as np

from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs
from neb_dynamics.NEB import NEB
# -


# # Claisen

# +
traj_tol01 = Chain.from_xyz("./claisen/cr_MSMEP_tol_01.xyz")
traj_tol0045 = Chain.from_xyz("./claisen/cr_MSMEP_tol_0045.xyz")
traj_tol0045_hf = Chain.from_xyz("./claisen/cr_MSMEP_tol_0045_hydrogen_fix.xyz")

# reference = Chain.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-NEB_v3/traj_0-0_0_neb.xyz")
# reference = Chain.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_0-0_0_cneb.xyz")
reference = Chain.from_xyz("/home/jdep/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_0-0_0_cneb.xyz")

# +
s = 8
fs = 18

ens_tol01 = traj_tol01.energies
ens_tol0045  = traj_tol0045.energies
ens_tol0045_hf  = traj_tol0045_hf.energies
ens_reference = reference.energies

f,ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(reference.integrated_path_length, (ens_reference - ens_reference[0])*627.5,'o-',label="reference cneb")
# plt.plot(traj_tol01.integrated_path_length, (ens_tol01-ens_reference[0])*627.5,'o-',label="MSMEP tol=0.01 no-hydrogen-fix")
plt.plot(traj_tol0045.integrated_path_length, (ens_tol0045 - ens_reference[0])*627.5,'o-',label="MSMEP tol=0.0045 no-hydrogen-fix")
plt.plot(traj_tol0045_hf.integrated_path_length, (ens_tol0045_hf - ens_reference[0])*627.5,'o-',label="MSMEP tol=0.0045 hydrogen-fix")

plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (kcal/mol) ['reference' neb is 0]",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# -



# # Wittig

params = ChainInputs()
root_neb = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/wittig_root_neb.xyz", parameters=params)
dep1_0 = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/wittig_depth_1_ind0.xyz", parameters=params)
dep1_1 = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/wittig_depth_1_ind1.xyz", parameters=params)
# dep2_0 = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/wittig_depth_2_ind0.xyz", parameters=params)
# dep3_0 = Chain.from_xyz("/home/jdep/T3D_data/msmep_draft/wittig_depth_3_ind0.xyz", parameters=params)

# +
s = 8
fs = 18

shift = .1
space = 0.02

f,ax = plt.subplots(figsize=(1.8*s, s))


plt.plot(root_neb.integrated_path_length,root_neb.energies, 'o--', label='root')

p_split = .525
plt.plot(dep1_0.integrated_path_length*(p_split-space),dep1_0.energies-shift, 'o-',label='depth1', color='orange')
plt.plot((dep1_1.integrated_path_length*((1-p_split)+space))+p_split,dep1_1.energies-shift, 'o-', color='orange')


# plt.plot(dep2_0.integrated_path_length*.5,dep2_0.energies-2*shift, 'o-',label='depth2', color='red')

# plt.plot(dep3_0.integrated_path_length*.49,dep3_0.energies-3*shift, 'o-',label='depth2', color='black')



plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (Hartrees)",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()
# -

from dataclasses import dataclass



HistoryTree.from_history_list(history2)

dep2_0[-1].energy

# +
# t_witt = Chain.from_xyz("./wittig/auto_extracted_att2.xyz")
# t_witt = Chain.from_xyz("../example_mep.xyz")
# -

t_witt.to_trajectory().draw();

# +
s = 8
fs = 18

ens_witt = t_witt.energies

f,ax = plt.subplots(figsize=(1.8*s, s))

plt.plot(t_witt.integrated_path_length, (ens_witt - ens_witt[0])*627.5,'o-',label="extracted tol=0.01")

plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# -

traj_witt = t_witt.to_trajectory()

traj_witt.draw()

start, intermediate, end = traj_witt[0], traj_witt[14], traj_witt[-1]

start_opt = start.tc_geom_optimization()
int_opt = intermediate.tc_geom_optimization()
end_opt = end.tc_geom_optimization()

pair_start_to_int = Trajectory([start_opt, int_opt])
gi_start_to_int = pair_start_to_int.run_geodesic(nimages=15, sweep=False)

pair_int_to_end = Trajectory([int_opt, end_opt])
gi_int_to_end = pair_int_to_end.run_geodesic(nimages=15, sweep=False)

from neb_dynamics.Inputs import NEBInputs, ChainInputs

ChainInputs(

NEBInputs(v=True, 
neb_start_to_int = 

# # Ugi

# t_ugi = Chain.from_xyz("./ugi/auto_extracted_0.xyz")
t_ugi = Chain.from_xyz("/home/jdep/neb_dynamics/example_cases/ugi/msmep_tol0.01_max_2000.xyz")

# +
s = 8
fs = 18

ens_ugi = t_ugi.energies

f,ax = plt.subplots(figsize=(2*s, s))

plt.plot(t_ugi.integrated_path_length, (ens_ugi - ens_ugi[0])*627.5,'o-',label="extracted tol=0.01")

plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (kcal/mol)",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# -


