# Running our first NEB calculation. Naturally, we are going to calculate a pi-bond torsion.

# # 1. Create initial guess
#
# ### We will create endpoints as we did the previous tutorials.

from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory

start = TDStructure.from_smiles("C=C")
end = start.copy()

# we can also load in a structure from a file path with:
# start = TDStructure.from_xyz("/path/to/file.xyz")
end_coords = end.coords
end_coords_swapped = end_coords[[0,1,3,2,4,5],:] # We have swapped the indices of two hydrogens
end = end.update_coords(end_coords_swapped)
# we can also load in a structure from a file path with:
# end = TDStructure.from_xyz("/path/to/file.xyz")

# ### Now we need to optimize them to our desired level of theory

start_opt = start.xtb_geom_optimization()
# start_opt = start.tc_geom_optimization() # use this one if you want to run TeraChem in ChemCloud
end_opt = end.xtb_geom_optimization()
# end_opt = end.tc_geom_optimization() # use this one if you want to run TeraChem in ChemCloud

# ### Now we need to interpolate between our endpoints

tr = Trajectory([start_opt, end_opt]).run_geodesic(nimages=10)

# ### Finally, let's create our Chain object

from chain import Chain
from neb_dynamics.Inputs import ChainInputs

cni = ChainInputs(k=0.1, delta_k=0.09)
initial_chain = Chain.from_traj(tr, parameters=cni)

# # 2. Let's set up our NEB optimization

from neb_dynamics.NEB import NEB
from neb_dynamics.Inputs import NEBInputs

# NEB

# The NEB object takes in 3 parameters:
# * initial_chain: our initial guess chain object to optimize
# * parameters: an NEBInputs object containing the parameters for NEB optimization and convergence
# * optimizer: the optimizer to use. We suggest you use `VelocityProjectedOptimizer` from
#             `neb_dynamics.optimizer.VPO`

nbi = NEBInputs(v=True, tol=0.0001, ts_spring_thre=0.0001, climb=True)

from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer

opt = VelocityProjectedOptimizer(timestep=0.5)

n = NEB(initial_chain=initial_chain, parameters=nbi, optimizer=opt)

_ = n.optimize_chain()

n.plot_opt_history(1)

print(n.optimized.get_ts_guess().coords)
