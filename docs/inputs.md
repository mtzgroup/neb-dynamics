# Input Parameters

NEB Dynamics uses several input classes to configure calculations. This guide covers all available options.

## NEBInputs

Controls Nudged Elastic Band optimization parameters.

```python
from neb_dynamics.inputs import NEBInputs

nbi = NEBInputs(
    # Convergence thresholds
    tol=1e-4,                    # Overall tolerance (Hartrees)
    en_thre=1e-4,                # Energy difference threshold
    rms_grad_thre=0.02,          # RMS perpendicular gradient (Ha/Bohr)
    max_rms_grad_thre=0.05,      # Maximum RMS gradient (Ha/Bohr)
    ts_grad_thre=0.05,           # Transition state gradient threshold
    ts_spring_thre=0.02,         # TS spring force threshold
    barrier_thre=0.1,            # Barrier height change (kcal/mol)

    # NEB options
    climb=False,                 # Enable climbing image NEB
    use_geodesic_tangent=False,  # Use geodesic tangents

    # Early stopping
    early_stop_force_thre=0.0,   # Early stop check threshold

    # Step control
    negative_steps_thre=10,      # Steps before halving step size
    max_steps=500,               # Maximum optimization steps

    # Other options
    skip_identical_graphs=True,   # Skip if endpoints have same graph
    do_elem_step_checks=True,    # Check for elementary steps
    v=True,                      # Verbose output
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `climb` | bool | False | Enable climbing image NEB to find exact saddle point |
| `en_thre` | float | 1e-4 | Energy difference threshold (Hartrees) |
| `rms_grad_thre` | float | 0.02 | RMS perpendicular gradient threshold (Ha/Bohr) |
| `max_rms_grad_thre` | float | 0.05 | Maximum RMS gradient threshold (Ha/Bohr) |
| `ts_grad_thre` | float | 0.05 | Transition state gradient infinity norm (Ha/Bohr) |
| `ts_spring_thre` | float | 0.02 | TS spring force infinity norm (Ha/Bohr) |
| `barrier_thre` | float | 0.1 | Barrier height change threshold (kcal/mol) |
| `early_stop_force_thre` | float | 0.0 | Early stop check (0 = disabled) |
| `negative_steps_thre` | int | 10 | Steps with poor gradient correlation before halving step size |
| `max_steps` | int | 500 | Maximum optimization steps |
| `skip_identical_graphs` | bool | True | Skip minimization if endpoints have identical molecular graphs |
| `do_elem_step_checks` | bool | False | Check if path is elementary step during optimization |
| `use_geodesic_tangent` | bool | False | Use geodesic-based tangents instead of standard |
| `v` | bool | True | Verbose output |

## ChainInputs

Controls the chain of images and path interpolation.

```python
from neb_dynamics.inputs import ChainInputs

ci = ChainInputs(
    # Spring constants
    k=0.1,           # Maximum spring constant
    delta_k=0.09,    # Energy-weighted spring constant parameter

    # Parallel computation
    do_parallel=True, # Compute gradients in parallel

    # Interpolation
    use_geodesic_interpolation=True,  # Use geodesic interpolation
    friction_optimal_gi=True,         # Optimize friction parameter

    # Node freezing
    node_freezing=True,    # Enable node freezing
    node_rms_thre=5.0,     # RMS threshold for freezing (Bohr)
    node_ene_thre=5.0,     # Energy threshold for freezing (kcal/mol)

    # Frozen atoms
    frozen_atom_indices="",  # Space-separated indices of frozen atoms
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | float | 0.1 | Maximum spring constant for NEB |
| `delta_k` | float | 0.09 | Parameter for energy-weighted spring constants |
| `do_parallel` | bool | True | Compute gradients in parallel |
| `use_geodesic_interpolation` | bool | True | Use geodesic interpolation for initial path |
| `friction_optimal_gi` | bool | True | Optimize friction parameter for GI |
| `node_freezing` | bool | True | Freeze nodes that have converged |
| `node_rms_thre` | float | 5.0 | RMS coordinate change threshold for freezing (Bohr) |
| `node_ene_thre` | float | 5.0 | Energy change threshold for freezing (kcal/mol) |
| `frozen_atom_indices` | str | "" | Indices of atoms to freeze during optimization |

### Energy-Weighted Spring Constants

The energy-weighted spring constant is calculated as:
```
k_i = k * exp(-delta_k * (E_i - E_min) / (E_max - E_min))
```

This keeps images more tightly spaced near the barrier where energies change rapidly.

## GIInputs

Controls geodesic interpolation for generating initial path guesses.

```python
from neb_dynamics.inputs import GIInputs

gi = GIInputs(
    nimages=10,           # Number of images
    friction=0.001,       # Friction parameter
    nudge=0.1,            # Nudge parameter
    align=True,           # Align structures before interpolation
    extra_kwds={},        # Additional keywords
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nimages` | int | 10 | Number of images in the chain |
| `friction` | float | 0.001 | Friction parameter (controls path smoothness) |
| `nudge` | float | 0.1 | Nudge parameter for path optimization |
| `align` | bool | True | Align structures using RMSD before interpolation |
| `extra_kwds` | dict | {} | Additional geodesic interpolation options |

### Tips for Geodesic Interpolation

- **Higher `nudge` values** produce shorter paths but may be less smooth
- **Lower `friction` values** allow more flexibility in atomic movements
- Run multiple interpolations with different parameters and select the shortest path

## RunInputs

Complete input class for running MSMEP calculations.

```python
from neb_dynamics.inputs import RunInputs
from qcio import ProgramArgs

ri = RunInputs(
    # Engine settings
    engine_name="qcop",      # "qcop", "ase", or "chemcloud"
    program="xtb",           # Electronic structure program

    # Path minimization
    path_min_method="NEB",   # "NEB", "FNEB", or "MLPGI"

    # Program arguments
    program_kwds=ProgramArgs(
        model={"method": "GFN2xTB", "basis": "GFN2xTB"},
        keywords={}
    ),

    # Custom inputs
    path_min_inputs=NEBInputs().__dict__,
    chain_inputs=ChainInputs().__dict__,
    gi_inputs=GIInputs().__dict__,

    # Optimizer
    optimizer_kwds={"timestep": 0.5},
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine_name` | str | "qcop" | Engine to use: "qcop", "ase", or "chemcloud" |
| `program` | str | "xtb" | Electronic structure program |
| `path_min_method` | str | "NEB" | Path minimization method: "NEB", "FNEB", "MLPGI" |
| `program_kwds` | ProgramArgs | None | Program-specific arguments |
| `path_min_inputs` | dict | {} | NEBInputs as dictionary |
| `chain_inputs` | dict | {} | ChainInputs as dictionary |
| `gi_inputs` | dict | {} | GIInputs as dictionary |
| `optimizer_kwds` | dict | {"timestep": 0.5} | Optimizer keyword arguments |

### Loading/Saving Inputs

```python
# Save inputs to TOML file
ri.save("my_inputs.toml")

# Load inputs from TOML file
ri = RunInputs.open("my_inputs.toml")
```

## NetworkInputs

Settings for building reaction networks.

```python
from neb_dynamics.inputs import NetworkInputs

ni = NetworkInputs(
    # Conformer generation
    n_max_conformers=10,      # Maximum conformers per endpoint
    subsample_confs=True,      # Subsample conformers
    conf_rmsd_cutoff=0.5,     # RMSD cutoff for new conformer (Angstroms)

    # Network settings
    network_nodes_are_conformers=False,  # Each conformer is a node
    maximum_barrier_height=1000,         # Max barrier (kcal/mol)
    tolerate_kinks=True,                  # Include paths with kinks

    # Computing
    use_slurm=False,           # Submit to SLURM queue

    # CREST settings
    CREST_temp=298.15,         # Temperature (K)
    CREST_ewin=6.0,            # Energy window (kcal/mol)
)
```

## Path Minimization Methods

### Standard NEB

The standard Nudged Elastic Band method with spring forces and perpendicular gradients.

```python
nbi = NEBInputs()
```

### Freezing NEB (FNEB)

Grow the chain dynamically, freezing images once they converge.

```python
from neb_dynamics.inputs import RunInputs

ri = RunInputs(
    path_min_method="FNEB",
    path_min_inputs={
        "max_min_iter": 100,
        "max_grow_iter": 20,
        "grad_tol": 0.05,
        "barrier_thre": 5,
    }
)
```

### MLP Geodesic (MLPGI)

Using machine learning potentials with geodesic interpolation.

```python
ri = RunInputs(
    path_min_method="MLPGI",
)
```
