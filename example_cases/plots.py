import matplotlib.pyplot as plt
import numpy as np
from retropaths.abinitio.trajectory import Trajectory
from neb_dynamics.Chain import Chain


# # Claisen

# +
traj_tol01 = Chain.from_xyz("./claisen/cr_MSMEP_tol_01.xyz")
traj_tol0045 = Chain.from_xyz("./claisen/cr_MSMEP_tol_0045.xyz")
traj_tol0045_hf = Chain.from_xyz("./claisen/cr_MSMEP_tol_0045_hydrogen_fix.xyz")

# reference = Chain.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-NEB_v3/traj_0-0_0_neb.xyz")
reference = Chain.from_xyz("/Users/janestrada/T3D_data/template_rxns/Claisen-Rearrangement-cNEB_v3/traj_0-0_0_cneb.xyz")
# -

# ### **TODO**: Need to regenerate an NEB run for 0-0_0 with the tol 0.0045 so that i can see the multi-minima nature of it

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

t = Trajectory([n.tdstructure for n in traj_tol0045])
t.draw();

t[14].to_xyz("frame14.xyz")

t[15].to_xyz("frame15.xyz")

# # Wittig

t_witt = Chain.from_xyz("./wittig/auto_extracted_att2.xyz")

# +
s = 8
fs = 18

ens_witt = t_witt.energies

f,ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(t_witt.integrated_path_length, (ens_witt - ens_witt[0])*627.5,'o-',label="extracted tol=0.01")

plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (kcal/mol) ['reference' neb is 0]",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# -

# # Ugi

t_ugi = Chain.from_xyz("./ugi/auto_extracted_0.xyz")

# +
s = 8
fs = 18

ens_ugi = t_ugi.energies

f,ax = plt.subplots(figsize=(1.16*s, s))

plt.plot(t_ugi.integrated_path_length, (ens_ugi - ens_ugi[0])*627.5,'o-',label="extracted tol=0.01")

plt.legend(fontsize=fs)
plt.xlabel("Integrated path length",fontsize=fs)
plt.ylabel("Energy (kcal/mol) ['reference' neb is 0]",fontsize=fs)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.show()

# -


