# API Reference

This section provides detailed API documentation for the main classes and functions.

## Core Classes

### NEB

Main class for Nudged Elastic Band calculations.

```python
from neb_dynamics.neb import NEB
```

::: neb_dynamics.neb.NEB

### MSMEP

Multi-Step Minimum Energy Path calculator for handling complex reactions.

```python
from neb_dynamics import MSMEP
```

::: neb_dynamics.msmep.MSMEP

### Chain

Container for a pathway consisting of multiple images.

```python
from neb_dynamics import Chain
```

::: neb_dynamics.chain.Chain

### StructureNode

A node containing a molecular structure.

```python
from neb_dynamics import StructureNode
```

::: neb_dynamics.nodes.node.StructureNode

## Input Classes

### NEBInputs

Configuration for NEB optimization.

```python
from neb_dynamics.inputs import NEBInputs
```

::: neb_dynamics.inputs.NEBInputs

### ChainInputs

Configuration for chain behavior.

```python
from neb_dynamics.inputs import ChainInputs
```

::: neb_dynamics.inputs.ChainInputs

### GIInputs

Configuration for geodesic interpolation.

```python
from neb_dynamics.inputs import GIInputs
```

::: neb_dynamics.inputs.GIInputs

### RunInputs

Complete configuration for MSMEP calculations.

```python
from neb_dynamics.inputs import RunInputs
```

::: neb_dynamics.inputs.RunInputs

## Engines

### Engine (Abstract Base)

Base class for all engines.

```python
from neb_dynamics.engines import Engine
```

::: neb_dynamics.engines.engine.Engine

### QCOPEngine

Engine using QCOP for electronic structure calculations.

```python
from neb_dynamics.engines import QCOPEngine
```

::: neb_dynamics.engines.qcop.QCOPEngine

### ASEEngine

Engine using ASE calculators.

```python
from neb_dynamics.engines import ASEEngine
```

::: neb_dynamics.engines.ase.ASEEngine

## Optimizers

### Optimizer (Abstract Base)

Base class for optimizers.

```python
from neb_dynamics.optimizers import Optimizer
```

### VelocityProjectedOptimizer

VPO optimizer with velocity projection.

```python
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
```

### ConjugateGradient

Conjugate gradient optimizer.

```python
from neb_dynamics.optimizers.cg import ConjugateGradient
```

### LBFGS

Limited-memory BFGS optimizer.

```python
from neb_dynamics.optimizers.lbfgs import LBFGS

## Helper Functions

### chainhelpers

Utility functions for chain manipulation and visualization.

```python
import neb_dynamics.chainhelpers as ch
```

**Common functions:**

- `ch.run_geodesic()` - Create chain using geodesic interpolation
- `ch.visualize_chain()` - Visualize chain in 3D
- `ch.compute_NEB_gradient()` - Calculate NEB gradient
- `ch.get_g_perps()` - Get perpendicular gradients
- `ch._get_ind_minima()` - Find indices of minima in chain
