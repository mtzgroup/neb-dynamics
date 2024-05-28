# # Trajectory object
# A list of TDStructures is a trajectory. 
# The `Trajectory` object enables us to do two main things:
# * parallel computation of energies and gradients enabled by ChemCloud
# * interpolation of images using geodesic interpolation

# As an example, let's create endpoints that correspond to an ethene pi-bond rotation

from neb_dynamics.tdstructure import TDStructure

start = TDStructure.from_smiles("C=C")

start.coords, start.symbols

end = start.copy()
end_coords = end.coords
end_coords_swapped = end_coords[[0,1,3,2,4,5],:] # this is a pi-bond rotation
# end_coords_swapped = end_coords[[0,1,4,3,2,5],:] # this is a crazy hydrogen swap
end = end.update_coords(end_coords_swapped) 

# Now let's create a trajectory object that contains only our endpoints

from neb_dynamics.trajectory import Trajectory

traj = Trajectory([start, end])

len(traj)

traj_interpolated = traj.run_geodesic(nimages=10)

len(traj_interpolated)

traj_interpolated.draw();

# You should be seeing a pi bond rotation! Let's look at the energies of this path. 

import matplotlib.pyplot as plt

energies = traj_interpolated.energies_xtb() # will output relative energies to first geometry
energies_notscaled = [td.energy_xtb() for td in traj_interpolated]

fs=18
plt.plot(energies,'o-')
plt.xlabel("image number",fontsize=fs)
plt.ylabel("relative energy (kcal/mol)",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.show()

fs=18
plt.plot(energies_notscaled,'o-')
plt.xlabel("image number",fontsize=fs)
plt.ylabel("energy (Hartree)",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.show()
