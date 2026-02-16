# NEB Dynamics

Automated minimum energy path (MEP) calculations using the Nudged Elastic Band (NEB) method and its variants.

Docs: https://github.com/mtzgroup/neb-dynamics/tree/master](https://mtzgroup.github.io/neb-dynamics/

## Features

- **NEB Calculations**: Find transition states and reaction paths
- **MSMEP**: Automatic pathway splitting for complex reactions
- **Multiple Engines**: Support for ChemCloud, XTB, ORCA, and ASE-based calculators
- **Geodesic Interpolation**: Better initial path guesses

## Installation

```bash
pip install "git+https://github.com/mtzgroup/neb-dynamics.git"
```

## Quick Start

```python
from neb_dynamics import NEB, NEBInputs, ChainInputs, StructureNode
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.optimizers.cg import ConjugateGradient
import neb_dynamics.chainhelpers as ch
from neb_dynamics import Chain
from qcio import Structure

# Load structures
start = Structure.from_xyz("start.xyz")
end = Structure.from_xyz("end.xyz")

# Set up engine using ChemCloud
eng = QCOPEngine(compute_program="chemcloud")

# Optimize endpoints and create chain
start_node = StructureNode(structure=start)
end_node = StructureNode(structure=end)
start_opt = eng.compute_geometry_optimization(start_node)
end_opt = eng.compute_geometry_optimization(end_node)

chain = Chain.model_validate({
    'nodes': [start_opt[-1], end_opt[-1]],
    'parameters': ChainInputs(k=0.1, delta_k=0.09)
})
initial_chain = ch.run_geodesic(chain, nimages=15)

# Run NEB
n = NEB(
    initial_chain=initial_chain,
    parameters=NEBInputs(v=True),
    optimizer=ConjugateGradient(timestep=0.5),
    engine=eng
)
results = n.optimize_chain()
```

## Documentation

Full documentation is available at: https://mtzgroup.github.io/neb-dynamics/

## ChemCloud Setup

NEB Dynamics uses ChemCloud for electronic structure calculations. Sign up at https://chemcloud.mtzlab.com/signup, then configure authentication:

```bash
# Option 1: Run setup_profile() - writes credentials to ~/.chemcloud/credentials
python -c "from chemcloud import setup_profile; setup_profile()"

# Option 2: Use environment variables
export CHEMCLOUD_USERNAME=your_email@chemcloud.com
export CHEMCLOUD_PASSWORD=your_password
```

## CLI Usage

```bash
# Run NEB calculation
mepd run --start start.xyz --end end.xyz --inputs inputs.toml

# Optimize transition state
mepd ts ts_guess.xyz --inputs inputs.toml

# Create default input file
mepd make-default-inputs --name inputs.toml
```

## Maintainers

For questions, contact:
- [Jan](mailto:jdep@stanford.edu)
- [Alessio](mailto:alevale@stanford.edu)

## Tips

- If using XTB locally, set `OMP_NUM_THREADS=1` to speed up calculations
- Use climbing image NEB (`NEBInputs(climb=True)`) for more accurate transition states
- For complex reactions, use MSMEP with `run_recursive_minimize()` for automatic pathway splitting
