# Electronic Structure Engines

NEB Dynamics supports multiple electronic structure engines through a common interface. Engines handle energy and gradient calculations for molecular structures.

## Available Engines

| Engine | Description | Use Case |
|--------|-------------|----------|
| `QCOPEngine` | Uses QCOP to interface with quantum chemistry codes | Production calculations with ChemCloud, XTB, ORCA, etc. |
| `ASEEngine` | Interfaces with ASE calculators | ML potentials, custom methods |
| `LEPSEngine` | Simple LEPS potential | Testing, model systems |
| `ThreeWellEngine` | Three-well potential | Testing, model systems |

## ChemCloud Setup

ChemCloud is the recommended way to run electronic structure calculations. You'll need:

1. Sign up at https://chemcloud.mtzlab.com/signup
2. Configure authentication (choose one option):

```bash
# Option 1: Run setup_profile() - writes credentials to ~/.chemcloud/credentials
python -c "from chemcloud import setup_profile; setup_profile()"

# Option 2: Use environment variables (for memory-only auth)
export CHEMCLOUD_USERNAME=your_email@chemcloud.com
export CHEMCLOUD_PASSWORD=your_password

# Option 3: Custom server (if using a different domain)
export CHEMCLOUD_DOMAIN="https://your-server-url.com"
```

## QCOPEngine

The QCOPEngine interfaces with various quantum chemistry programs through QCOP.

### Basic Usage

```python
from neb_dynamics.engines.qcop import QCOPEngine

# Using ChemCloud (recommended)
eng = QCOPEngine(compute_program="chemcloud")

# Default: uses XTB
eng = QCOPEngine()

# Or specify program arguments
from qcio import ProgramArgs

args = ProgramArgs(
    model={"method": "GFN2xTB", "basis": "GFN2xTB"},
    keywords={"threads": 4}
)
eng = QCOPEngine(program_args=args, program="xtb")
```

### Features

- **Geometry Optimization**: `eng.compute_geometry_optimization(node)`
- **Energy Calculation**: `eng.compute_energies(chain)`
- **Gradient Calculation**: `eng.compute_gradients(chain)`
- **Supports external programs**: XTB, ORCA, TeraChem, Psi4, etc.

### Supported Programs

```python
# ChemCloud (recommended)
eng = QCOPEngine(compute_program="chemcloud")

# XTB (default, requires local installation)
eng = QCOPEngine(program="xtb")

# ORCA
eng = QCOPEngine(program="orca")

# TeraChem
eng = QCOPEngine(program="terachem")
```

## ASEEngine

The ASEEngine interfaces with ASE (Atomic Simulation Environment) calculators, enabling use of machine learning potentials and other methods.

### Basic Usage

```python
from neb_dynamics.engines.ase import ASEEngine
from mace.calculators import MACECalculator

# Load MACE potential
calc = MACECalculator(model="mace-medium", device="cuda")
eng = ASEEngine(calculator=calc)

# Now run NEB as usual
n = NEB(initial_chain=initial_chain, parameters=nbi, optimizer=opt, engine=eng)
```

### ASE Optimizers

```python
from neb_dynamics.engines.ase import ASEEngine

eng = ASEEngine(
    calculator=calc,
    geometry_optimizer="LBFGSLineSearch"  # Default
)
```

Available optimizers: `LBFGS`, `BFGS`, `FIRE`, `LBFGSLineSearch`, `MDMin`

## Engine Interface

All engines implement the following interface:

```python
class Engine:
    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        """Compute gradients for each node in the chain"""
        ...

    def compute_energies(self, chain: Union[Chain, List]) -> NDArray:
        """Compute energies for each node in the chain"""
        ...

    def compute_geometry_optimization(self, node: Node) -> List[Node]:
        """Optimize a single node geometry"""
        ...
```

## Using with Chains

Engines work with Chain objects to compute properties:

```python
# Create engine
eng = QCOPEngine(compute_program="chemcloud")

# Compute energies (also computes gradients internally)
energies = eng.compute_energies(chain)

# Compute gradients explicitly
gradients = eng.compute_gradients(chain)

# Optimize a single structure
optimized_node = eng.compute_geometry_optimization(start_node)
trajectory = eng.compute_geometry_optimization(start_node)  # Returns full trajectory
final_structure = trajectory[-1]
```

## Choosing an Engine

### Use QCOPEngine with ChemCloud when:
- Running production calculations
- Don't want to install local quantum chemistry software
- Need reliable cloud computing

### Use ASEEngine when:
- Using machine learning potentials (MACE, NequIP, etc.)
- Need custom ASE calculators
- Have GPU access for ML potentials
