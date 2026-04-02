# NEB Dynamics

Automated minimum energy path (MEP) calculations using the Nudged Elastic Band (NEB) method and its variants.

Docs: https://mtzgroup.github.io/neb-dynamics/

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
If you have cloned the repository, `cd` to `examples/`.and run `run_oxycope.sh` or `run_wittig.sh` to see the code in action

```bash
# Run NEB calculation
mepd run --start start.xyz --end end.xyz --inputs inputs.toml

# Two-stage refinement (cheap discovery -> expensive refinement)
mepd run-refine examples/oxycope.xyz \
  -i expensive.toml \
  -ci cheap.toml \
  --recursive \
  --recycle-nodes \
  --name oxycope_refine

# Optimize transition state
mepd ts ts_guess.xyz --inputs inputs.toml

# Create default input file
mepd make-default-inputs --name inputs.toml
```

## MEPD Drive Quickstart

Local run from this repository:

```bash
# Install repo dependencies
uv sync

# Launch blank Drive session using an existing RunInputs TOML
uv run mepd drive --inputs examples/example_inputs.toml
```

SMILES bootstrap mode:

```bash
uv run mepd drive \
  --smiles "C=CC(O)CC=C" \
  --product-smiles "C=CC(=O)CC=C" \
  --environment "O" \
  --charge 0 \
  --multiplicity 1 \
  --inputs examples/example_inputs.toml \
  --name allylic_alcohol_drive
```

Resume an existing workspace:

```bash
uv run mepd drive --workspace ./allylic_alcohol_drive
```

Notes:

- You need a valid electronic-structure backend setup (for example ChemCloud credentials in `~/.chemcloud/credentials`, or local QC binaries if using local engines).
- If you do not pass `--inputs` at startup, provide an inputs TOML path in the Drive initializer UI before running initialization.
- The reaction-template `+` action in Drive and `mepd netgen-smiles` require the optional `retropaths` repository. Set `RETROPATHS_REPO=/path/to/retropaths` (or place it at `~/retropaths`).
- Detailed Drive options and remote SSH tunnel usage are documented in `docs/cli.md` under `drive`.

## Render Deploy

This repository includes a Render Blueprint in `render.yaml` and a startup wrapper in `scripts/start_render_drive.sh` for deploying `mepd drive` as a Render Web Service.

Default behavior:

- binds Drive to `0.0.0.0:$PORT`
- loads a bundled demo workspace from `examples/claisen`
- rewrites that workspace into `/tmp/mepd-drive` before startup
- uses `examples/example_inputs.toml` for follow-on actions inside the deployed app
- can hydrate `~/.chemcloud/credentials` from a Render secret at startup
- clones `retropaths` into `/opt/render/project/retropaths` during the Render build and points `RETROPATHS_REPO` there

To deploy on Render:

1. Push this repository to GitHub.
2. In Render, create a new Blueprint and point it at this repo.
3. Set either `CHEMCLOUD_CREDENTIALS_B64` or `CHEMCLOUD_CREDENTIALS_TOML` in the Render dashboard.
4. Deploy the Blueprint.

To reuse an existing local ChemCloud login without storing your password in Render, base64-encode your local credentials file and paste the result into `CHEMCLOUD_CREDENTIALS_B64`:

```bash
base64 < ~/.chemcloud/credentials
```

That env var is written back to `~/.chemcloud/credentials` inside the Render instance before `mepd drive` starts.

The default public URL will be `https://<service-name>.onrender.com`.

Note: the included Blueprint uses the free plan, which has an ephemeral filesystem. The Drive workspace under `/tmp/mepd-drive` will not survive service restarts or redeploys.

## Maintainers

For questions, contact:
- [Jan](mailto:jdep@stanford.edu)
- [Alessio](mailto:alevale@stanford.edu)

## Tips

- If using XTB locally, set `OMP_NUM_THREADS=1` to speed up calculations
- Use climbing image NEB (`NEBInputs(climb=True)`) for more accurate transition states
- For complex reactions, use MSMEP with `run_recursive_minimize()` for automatic pathway splitting
