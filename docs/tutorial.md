# Tutorial

This tutorial walks through using NEB Dynamics for finding reaction pathways.

## Prerequisites

Install NEB Dynamics and dependencies:

```bash
pip install "git+https://github.com/mtzgroup/neb-dynamics.git"
```

**Note:** These tutorials use ChemCloud for electronic structure calculations. You'll need:
1. Sign up at https://chemcloud.mtzlab.com/signup
2. Configure authentication:

```bash
# Option 1: Run setup_profile() - writes credentials to ~/.chemcloud/credentials
python -c "from chemcloud import setup_profile; setup_profile()"

# Option 2: Use environment variables (for memory-only auth)
export CHEMCLOUD_USERNAME=your_email@chemcloud.com
export CHEMCLOUD_PASSWORD=your_password
```

## Tutorial 1: Simple NEB Calculation

Calculate an SN2 reaction: bromide substitution by hydroxide.

### Step 1: Create Initial and Final Structures

```python
from qcio import Structure
from neb_dynamics import StructureNode

# Define structures using embedded XYZ data
start_xyz = """17
Frame 0
 C          0.885440409184        2.102076768875        0.627821743488
 C          0.841612815857        1.115651130676       -0.247040778399
 C         -0.420807182789        0.556873917580       -0.838685691357
 O         -0.430870711803        0.798001468182       -2.239025592804
 C         -0.522329509258       -0.964021146297       -0.644988238811
 C         -0.715307652950       -1.349323630333        0.785972416401
 C          0.097422197461       -2.135773420334        1.466876506805
 H         -0.267592728138        1.735007524490       -2.391090631485
 H         -1.386946558952       -1.296320080757       -1.227010726929
 H          0.993258416653       -2.552916049957        1.031124353409
 H         -0.008914750069        2.554719924927        1.029638171196
 H          1.820721507072        2.502734899521        0.982666194439
 H          1.748645901680        0.692936420441       -0.660514891148
 H         -1.296261906624        1.040292620659       -0.372635990381
 H          0.376222431660       -1.434682250023       -1.046772599220
 H         -1.611668467522       -0.956238925457        1.253052473068
 H         -0.102624341846       -2.409019231796        2.490613460541
"""

end_xyz = """17
Frame 29
 C          0.576232671738        0.921668708324        0.825146675110
 C          0.811691701412        1.315650582314       -0.631796598434
 C         -0.290672153234        0.930290937424       -1.571661710739
 O         -0.139186233282        0.584553778172       -2.709024667740
 C         -1.166546344757       -1.861497998238       -0.220005810261
 C         -0.774998486042       -1.183089017868        0.841198801994
 C          0.589584112167       -0.601971745491        1.032947182655
 H          0.880913138390        2.407624959946       -0.707833826542
 H         -2.171884059906       -2.235154390335       -0.311172336340
 H          1.294490098953       -1.047748923302        0.330671131611
 H         -0.374946832657        1.333065629005        1.164748311043
 H          1.358177542686        1.381788730621        1.429385662079
 H          1.752718806267        0.898995816708       -0.990877032280
 H         -1.302031993866        1.039528012276       -1.137167930603
 H         -0.504405736923       -2.070925474167       -1.043461441994
 H         -1.469262003899       -0.993297398090        1.650624632835
 H          0.930125653744       -0.819482147694        2.048279047012
"""

# Load structures from XYZ strings
start = Structure.from_xyz(start_xyz)
end = Structure.from_xyz(end_xyz)

# Create nodes
start_node = StructureNode(structure=start)
end_node = StructureNode(structure=end)
```

### Step 2: Optimize Endpoints

```python
from neb_dynamics.engines.qcop import QCOPEngine

# Using ChemCloud for electronic structure
eng = QCOPEngine(compute_program="chemcloud")

# Optimize geometries
start_opt = eng.compute_geometry_optimization(start_node)
end_opt = eng.compute_geometry_optimization(end_node)

# Get final optimized structures
start_node = start_opt[-1]
end_node = end_opt[-1]
```

### Step 3: Create Initial Chain with Geodesic Interpolation

```python
import neb_dynamics.chainhelpers as ch
from neb_dynamics.inputs import ChainInputs
from neb_dynamics import Chain

# Set up chain inputs
cni = ChainInputs(k=0.1, delta_k=0.09)

# Create a Chain object first
chain = Chain.model_validate({
    'nodes': [start_node, end_node],
    'parameters': cni
})

# Generate initial path using geodesic interpolation
initial_chain = ch.run_geodesic(chain, nimages=15)
```

### Step 4: Run NEB Optimization

```python
from neb_dynamics.neb import NEB
from neb_dynamics.inputs import NEBInputs
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics.optimizers.cg import ConjugateGradient

# Set up NEB parameters
nbi = NEBInputs(v=True)

# Choose optimizer
# Velocity Projected Optimizer (recommended)
opt = VelocityProjectedOptimizer(timestep=0.5)

# Or Conjugate Gradient
# opt = ConjugateGradient(timestep=0.5)

# Create NEB object
n = NEB(
    initial_chain=initial_chain,
    parameters=nbi,
    optimizer=opt,
    engine=eng
)

# Run optimization
elem_step_results = n.optimize_chain()
```

### Step 5: Visualize Results

```python
# Plot optimization history
n.plot_opt_history(1)

# Plot energy profile
n.optimized.plot_chain()

# Visualize chain in 3D
ch.visualize_chain(n.optimized)
```

## Tutorial 2: Using Climbing Image NEB

Climbing image NEB finds the exact saddle point:

```python
# Enable climbing image
nbi = NEBInputs(climb=True)

n = NEB(
    initial_chain=initial_chain,
    parameters=nbi,
    optimizer=VelocityProjectedOptimizer(timestep=0.5),
    engine=eng
)

results = n.optimize_chain()
```

## Tutorial 3: Multi-Step MEP (MSMEP)

For complex reactions with multiple elementary steps, use MSMEP:

```python
from qcio import Structure
from neb_dynamics import MSMEP, RunInputs, StructureNode, ChainInputs
from neb_dynamics.engines.qcop import QCOPEngine
import neb_dynamics.chainhelpers as ch

# Set up engine with ChemCloud
eng = QCOPEngine(compute_program="chemcloud")

# Set up inputs - use engine_name="chemcloud" to run on ChemCloud
ri = RunInputs(
    engine_name="chemcloud",
    program="xtb",  # Program to use on ChemCloud
    path_min_method="NEB",
)

m = MSMEP(inputs=ri)

# Create structures using embedded XYZ data
start_xyz = """17
Frame 0
 C          0.885440409184        2.102076768875        0.627821743488
 C          0.841612815857        1.115651130676       -0.247040778399
 C         -0.420807182789        0.556873917580       -0.838685691357
 O         -0.430870711803        0.798001468182       -2.239025592804
 C         -0.522329509258       -0.964021146297       -0.644988238811
 C         -0.715307652950       -1.349323630333        0.785972416401
 C          0.097422197461       -2.135773420334        1.466876506805
 H         -0.267592728138        1.735007524490       -2.391090631485
 H         -1.386946558952       -1.296320080757       -1.227010726929
 H          0.993258416653       -2.552916049957        1.031124353409
 H         -0.008914750069        2.554719924927        1.029638171196
 H          1.820721507072        2.502734899521        0.982666194439
 H          1.748645901680        0.692936420441       -0.660514891148
 H         -1.296261906624        1.040292620659       -0.372635990381
 H          0.376222431660       -1.434682250023       -1.046772599220
 H         -1.611668467522       -0.956238925457        1.253052473068
 H         -0.102624341846       -2.409019231796        2.490613460541
"""

end_xyz = """17
Frame 29
 C          0.576232671738        0.921668708324        0.825146675110
 C          0.811691701412        1.315650582314       -0.631796598434
 C         -0.290672153234        0.930290937424       -1.571661710739
 O         -0.139186233282        0.584553778172       -2.709024667740
 C         -1.166546344757       -1.861497998238       -0.220005810261
 C         -0.774998486042       -1.183089017868        0.841198801994
 C          0.589584112167       -0.601971745491        1.032947182655
 H          0.880913138390        2.407624959946       -0.707833826542
 H         -2.171884059906       -2.235154390335       -0.311172336340
 H          1.294490098953       -1.047748923302        0.330671131611
 H         -0.374946832657        1.333065629005        1.164748311043
 H          1.358177542686        1.381788730621        1.429385662079
 H          1.752718806267        0.898995816708       -0.990877032280
 H         -1.302031993866        1.039528012276       -1.137167930603
 H         -0.504405736923       -2.070925474167       -1.043461441994
 H         -1.469262003899       -0.993297398090        1.650624632835
 H          0.930125653744       -0.819482147694        2.048279047012
"""

# Load structures from XYZ strings
start = Structure.from_xyz(start_xyz)
end = Structure.from_xyz(end_xyz)

start_node = StructureNode(structure=start)
end_node = StructureNode(structure=end)

# Optimize endpoints
start_opt = eng.compute_geometry_optimization(start_node)
start_node = start_opt[-1]

end_opt = eng.compute_geometry_optimization(end_node)
end_node = end_opt[-1]

# Create initial chain using Chain.model_validate()
chain = Chain.model_validate({
    'nodes': [start_node, end_node],
    'parameters': ChainInputs(k=0.1, delta_k=0.09)
})
initial_chain = ch.run_geodesic(chain, nimages=20)

# Run recursive minimization
# This will automatically split the path if needed
history = m.run_recursive_minimize(initial_chain)

# Visualize final path
ch.visualize_chain(history.data.optimized)
```

## Tutorial 4: Custom Convergence Criteria

Adjust convergence thresholds for your needs:

```python
# Tight convergence for high-accuracy calculations
nbi_tight = NEBInputs(
    en_thre=1e-5,           # Tighter energy threshold
    rms_grad_thre=0.002,     # Tighter gradient threshold
    max_steps=1000,          # More steps allowed
)

# Loose convergence for quick screening
nbi_loose = NEBInputs(
    en_thre=1e-3,
    rms_grad_thre=0.1,
    max_steps=100,
)
```

## Tutorial 6: Loading and Saving Calculations

Save intermediate and final results:

```python
# Save chain to disk
n.optimized.write_to_disk("final_chain.xyz", write_qcio=True)

# Save complete NEB history
import os
os.makedirs("neb_history", exist_ok=True)
for i, chain in enumerate(n.chain_trajectory):
    chain.write_to_disk(f"neb_history/traj_{i}.xyz")

# Load chain from xyz
from neb_dynamics import Chain, ChainInputs
loaded_chain = Chain.from_xyz("final_chain.xyz", parameters=ChainInputs())
```

## Common Issues and Solutions

### Problem: Chain doesn't converge

**Solutions:**
- Increase `max_steps`
- Reduce timestep in optimizer
- Check that endpoints are properly optimized
- Try different `k` and `delta_k` values in ChainInputs

### Problem: Images bunch up at endpoints

**Solutions:**
- Enable node freezing (`node_freezing=True` in ChainInputs)
- Use energy-weighted spring constants (higher `delta_k`)

### Problem: Path has "kinks"

**Solutions:**
- Increase number of images (`nimages`)
- Use geodesic interpolation instead of linear
- Enable climbing image NEB

### Problem: Electronic structure errors

**Solutions:**
- Check that structures are chemically reasonable
- Try different electronic structure method
- Enable `do_elem_step_checks` for better diagnostics

## Example Input File

Create a TOML input file for batch processing:

```toml
[RunInputs]
engine_name = "qcop"
program = "xtb"
path_min_method = "NEB"

[program_kwds]
[program_kwds.model]
method = "GFN2xTB"
basis = "GFN2xTB"

[chain_inputs]
k = 0.1
delta_k = 0.09

[gi_inputs]
nimages = 15
friction = 0.001
nudge = 0.1

[path_min_inputs]
v = true
max_steps = 500

[optimizer_kwds]
timestep = 0.5
```

Then run with ChemCloud by setting up authentication:

```bash
# Option 1: Run setup_profile() - writes credentials to ~/.chemcloud/credentials
python -c "from chemcloud import setup_profile; setup_profile()"

# Option 2: Use environment variables (for memory-only auth)
export CHEMCLOUD_USERNAME=your_email@chemcloud.com
export CHEMCLOUD_PASSWORD=your_password
```

Load and use:

```python
from neb_dynamics.inputs import RunInputs
from neb_dynamics import MSMEP

inputs = RunInputs.open("my_input.toml")
m = MSMEP(inputs=inputs)
history = m.run_recursive_minimize(initial_chain)
```
