# Key Concepts

This section explains the fundamental concepts behind NEB Dynamics.

## Nodes and Chains

### Nodes

A **Node** represents a single molecular structure (geometry) along a reaction path. The main node types are:

- **`StructureNode`**: Represents a molecular structure with geometry, symbols, charge, and multiplicity. Stores cached energy and gradient data.
- **`XYNode`**: A simplified node for 2D coordinates (useful for model systems).

Nodes are containers for molecular structures that also hold computational results:
```python
from neb_dynamics import StructureNode
from qcio import Structure

# Create a node from a structure
structure = Structure.from_smiles("C=O")
node = StructureNode(structure=structure)

# Update coordinates
new_coords = node.coords + 0.1  # Slight displacement
new_node = node.update_coords(new_coords)
```

### Chains

A **Chain** is a collection of Nodes representing a reaction pathway. It contains:
- A list of Node objects (images along the path)
- Chain parameters (spring constants, interpolation settings)

```python
from neb_dynamics import Chain, ChainInputs
from neb_dynamics.nodes import StructureNode
from qcio import Structure

# Create a chain from nodes
nodes = [StructureNode(structure=Structure.from_smiles(s)) for s in ["C=O", "C[O]"]]
chain = Chain(nodes=nodes, parameters=ChainInputs())

# Access chain properties
print(chain.energies)       # Energy of each node
print(chain.gradients)      # Gradient of each node
print(chain.coordinates)    # Atomic coordinates
print(chain.path_length)    # Integrated path length
```

## The Nudged Elastic Band Method

### How NEB Works

NEB finds the minimum energy path by:
1. Creating a chain of images between reactant and product
2. Optimizing images simultaneously using spring forces (to keep them spaced) and perpendicular forces (to push toward the MEP)

### Key Equations

The total force on each image consists of two components:

**Spring Force (parallel to path):**
```
F_spring = k * (|R_i+1 - R_i| - d) * t
```
where `t` is the tangent direction.

**Perpendicular Gradient (perpendicular to path):**
```
F_perp = -∇E + (∇E · t) * t
```

The "nudging" removes the component of the spring force perpendicular to the path, ensuring images don't climb barriers.

### Climbing Image NEB

Enable with `NEBInputs(climb=True)` - the image with highest energy climbs to the saddle point.

```python
nbi = NEBInputs(climb=True)
```

## Geodesic Interpolation

Geodesic interpolation provides better initial path guesses than linear interpolation, especially for reactions involving bond breaking/forming.

### How It Works

1. Calculate pairwise distances between all atoms
2. Find a path that minimizes the sum of geodesic distances
3. Generate images along this path

```python
import neb_dynamics.chainhelpers as ch
from neb_dynamics.inputs import ChainInputs, GIInputs

# Geodesic interpolation
gi_inputs = GIInputs(nimages=15, friction=0.001, nudge=0.1)
initial_chain = ch.run_geodesic([start_node, end_node],
                                 chain_inputs=ChainInputs(),
                                 nimages=15)
```

### Parameters

- **`nimages`**: Number of images in the chain
- **`friction`**: Controls penalty for large pairwise distances (lower = more flexible)
- **`nudge`**: Nudge parameter for path optimization

## Multi-Step MEP (MSMEP)

MSMEP automatically handles complex reactions with multiple elementary steps:

1. Run NEB on initial path
2. Check if the path represents a single elementary step
3. If not, split the path at intermediate minima
4. Recursively minimize each segment
5. Build a tree of all pathways found

```python
from neb_dynamics import MSMEP, RunInputs

# Run recursive minimization
inputs = RunInputs(path_min_method='NEB')
m = MSMEP(inputs=inputs)

# This will automatically split if needed
history = m.run_recursive_minimize(initial_chain)
```

## Convergence Criteria

NEB Dynamics uses multiple convergence criteria:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `en_thre` | Energy difference threshold (Ha) | 1e-4 |
| `rms_grad_thre` | RMS perpendicular gradient threshold | 0.02 Ha/Bohr |
| `max_rms_grad_thre` | Maximum RMS gradient threshold | 0.05 Ha/Bohr |
| `ts_grad_thre` | Transition state gradient threshold | 0.05 Ha/Bohr |
| `ts_spring_thre` | TS spring force threshold | 0.02 Ha/Bohr |
| `barrier_thre` | Barrier height change threshold (kcal/mol) | 0.1 |
| `max_steps` | Maximum optimization steps | 500 |

## Visualization

NEB Dynamics provides several visualization tools:

```python
import neb_dynamics.chainhelpers as ch

# Plot energy profile
chain.plot_chain()

# Plot optimization history
n.plot_opt_history(1)

# Plot convergence metrics
n.plot_convergence_metrics()

# Visualize chain in 3D
ch.visualize_chain(chain)
```
