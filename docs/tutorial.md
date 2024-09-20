# tutorial
This tutorial references [ChainInputs](https://mtzgroup.github.io/neb-dynamics/inputs), [NEBInputs](inputs.md),[Chain](https://mtzgroup.github.io/neb-dynamics/chain/), [Trajectory](https://mtzgroup.github.io/neb-dynamics/trajectory/), [TDStructure](https://mtzgroup.github.io/neb-dynamics/tdstructure/), [NEB](https://mtzgroup.github.io/neb-dynamics/neb/) and [VelocityProjectedOptimizer](https://mtzgroup.github.io/neb-dynamics/vpo)


# Running our first NEB calculation. We are going to calculate a pi-bond torsion.
## 1. Create initial guess
### We will create geometrics using QCIO, then instantiate our nodes
```python
from qcio import Structure
from neb_dynamics import StructureNode

start = Structure.from_smiles("C=C")
# we can also load in a structure from a file path with:
# start = Structure.open("/path/to/file.xyz")

start_node = StructureNode(structure=start)

end_node = start_node.copy()
end_coords = end_node.coords
end_coords_swapped = end_coords[[0,1,3,2,4,5],:] # We have swapped the indices of two hydrogens
end_node = end_node.update_coords(end_coords_swapped)
```

### Now we need to optimize them to our desired level of theory
```python
from neb_dynamics import QCOPEngine, ASEEngine
from xtb.ase.calculator import XTB

calc = XTB()
eng = ASEEngine(calculator=calc) #QCOPEngine() # Also have ASEEngine if you want to run with ASE calculators
# QCOPEngine will default to using XTB

start_opt = eng.compute_geometry_optimization(start_node)
end_opt = eng.compute_geometry_optimization(end_node)
```
### Now we need to interpolate between our endpoints
```python
import neb_dynamics.chainhelpers as ch
from neb_dynamics.inputs import ChainInputs

cni = ChainInputs(k=0.1, delta_k=0.09) # energy weighted spring constants
# see: https://pubs.acs.org/doi/10.1021/acs.jctc.1c00462
initial_chain = ch.run_geodesic([start_node, end_node], chain_inputs=cni, nimages=15)
```
## 2. Let's set up our NEB optimization
### The NEB object takes in 4 parameters:
 * `initial_chain`: our initial guess chain object to optimize
 * `parameters`: an NEBInputs object containing the parameters for NEB optimization and convergence
 * `optimizer`: the optimizer to use. We suggest you use `VelocityProjectedOptimizer` from
            `neb_dynamics.optimizer.VPO`
 * `engine`: defines the electronic structure engine to be used
```python
from neb_dynamics.neb import NEB
from neb_dynamics.inputs import NEBInputs

nbi = NEBInputs(v=True)

from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer

opt = VelocityProjectedOptimizer(timestep=0.5)

n = NEB(initial_chain=initial_chain, parameters=nbi, optimizer=opt, engine=eng)

elem_step_results = n.optimize_chain()

n.plot_opt_history(1)

ch.visualize_chain(n.optimized)
```