# # Node3D
#
# We now understand what a TDStructure is, so we can transition into explaining the NEB node object. 
# A `Node3D` is a wrapper for a TDstructure. During the course of a NEB optimization it will cache computed energies and gradients, and will hold whether a given node has been `converged` (i.e. should it be frozen)
#
# We have two main Node3D objects: `Node3D` and `Node3D_TC`. The first one uses **GFN2-XTB** to compute energies and gradients. The latter uses **TeraChem** in [ChemCloud](https://mtzgroup.github.io/chemcloud-client/tutorial/compute/)
#

from neb_dynamics.nodes.Node3D import Node3D
from neb_dynamics.tdstructure import TDStructure

td = TDStructure.from_smiles("COCO")

node = Node3D(td)

print(node._cached_gradient)

node.gradient

print(node._cached_gradient)

# The node object also has two geometry optimization functions:
#  * `.do_geometry_optimization()`
#  * `.do_geom_opt_trajectory()`
#  
# The former will only return the final optimized structure, whereas the latter will return a `Trajectory` object containing the intermediate images of the optimization. The latter can be useful when recycling relaxation frames in NEB, as we will discuss later
#
# <span style="color:red">N.B. this will likely change with upcoming refator</span>.

node_opt = node.do_geometry_optimization()

node_opt_traj = node.do_geom_opt_trajectory()

node_opt_traj.draw();

# # Chain
# A `Chain` is a wrapper for a list of Node objects. It does the heavy lifting of computing the projected gradients and the spring forces NEB. 
#
# It is composed of two things:
#    * nodes --> list containing Node objects
#    * parameters --> instance of `ChainInputs` which contains all necessary parameters 

from neb_dynamics.Inputs import ChainInputs

# ?ChainInputs

# You can instantiate a chain either from a list of nodes, or from a trajectory object.
#
# Let's borrow the pi-bond rotation example from the previous tutorial

# +
from neb_dynamics.trajectory import Trajectory

start = TDStructure.from_smiles("C=C")
end = start.copy()
end_coords = end.coords
end_coords_swapped = end_coords[[0,1,3,2,4,5],:] # this is a pi-bond rotation
end = end.update_coords(end_coords_swapped) 

tr = Trajectory([start, end]).run_geodesic(nimages=10)

# +
from neb_dynamics.Chain import Chain

nodes = [Node3D(td) for td in tr] 
parameters = ChainInputs() # using defaults
chain_1 = Chain(nodes=nodes, parameters=parameters)
# -

chain_1.plot_chain()

chain_2 = Chain.from_traj(traj=tr, parameters=parameters)

chain_2.plot_chain()

# you can also convert any chain object back to a trajectory for easy viewing!
chain_1.to_trajectory().draw();


