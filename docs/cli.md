# Command Line Interface

NEB Dynamics provides a command-line interface (CLI) via the `mepd` command. This is installed automatically when you install the package.

## Installation

After installing NEB Dynamics, the CLI is available:

```bash
pip install "git+https://github.com/mtzgroup/neb-dynamics.git"
mepd --help
```

## Commands

### run

Run NEB or MSMEP calculations from the command line.

```bash
mepd run --start START.xyz --end END.xyz --inputs inputs.toml
```

**Options:**

| Option | Description |
|--------|-------------|
| `--start`, `-s` | Path to start structure file (XYZ or .qcio), or a SMILES string with `--use-smiles` |
| `--end`, `-e` | Path to end structure file, or SMILES with `--use-smiles` |
| `--geometries` | Path to file containing an approximate path (multiple structures) |
| `--inputs`, `-i` | Path to RunInputs TOML file |
| `--use-smiles` | Use SMILES strings for start/end instead of files |
| `--use-tsopt` | Run transition state optimization on each TS guess |
| `--minimize-ends` | Minimize endpoint geometries before NEB |
| `--recursive` | Run autosplitting MSMEP |
| `--name` | Custom name for output files |
| `--charge` | Molecular charge (default: 0) |
| `--multiplicity` | Spin multiplicity (default: 1) |
| `--create-irc` | Run IRC after TS optimization |
| `--use-bigchem` | Use ChemCloud for Hessian calculations in TS optimization |

**Example:**

```bash
# Basic NEB calculation
mepd run --start reactant.xyz --product.xyz --inputs inputs.toml

# Using SMILES
mepd run --start "CBr" --end "CO" --use-smiles --inputs inputs.toml

# Autosplitting MSMEP
mepd run --start start.xyz --end end.xyz --recursive --inputs inputs.toml

# With TS optimization and IRC
mepd run --start start.xyz --end end.xyz --use-tsopt --create-irc --inputs inputs.toml
```

---

### ts

Optimize a transition state structure.

```bash
mepd ts geometry.xyz --inputs inputs.toml
```

**Options:**

| Option | Description |
|--------|-------------|
| `geometry` | Path to geometry file to optimize (required) |
| `--inputs`, `-i` | Path to RunInputs TOML file |
| `--name` | Custom name for output files |
| `--charge` | Molecular charge (default: 0) |
| `--multiplicity` | Spin multiplicity (default: 1) |
| `--bigchem` | Use ChemCloud for TS optimization |

**Example:**

```bash
mepd ts ts_guess.xyz --inputs inputs.toml
mepd ts ts_guess.xyz --bigchem --name my_ts
```

---

### pseuirc

Compute pseudo-IRC path from a transition state.

```bash
mepd pseuirc ts_geometry.xyz --inputs inputs.toml
```

**Options:**

| Option | Description |
|--------|-------------|
| `geometry` | Path to TS geometry file (required) |
| `--inputs`, `-i` | Path to RunInputs TOML file |
| `--name` | Custom name for output files |
| `--charge` | Molecular charge (default: 0) |
| `--multiplicity` | Spin multiplicity (default: 1) |
| `--dr` | Displacement distance in Angstroms (default: 1.0) |

**Example:**

```bash
mepd pseuirc ts_guess.xyz --dr 0.5
```

---

### make-default-inputs

Create a default RunInputs TOML file.

```bash
mepd make-default-inputs --name inputs.toml
```

**Options:**

| Option | Description |
|--------|-------------|
| `--name` | Path to output TOML file |
| `--path-min-method`, `-pmm` | Path minimization method: `neb` or `fneb` (default: neb) |

---

### run-netgen

Run network generation for multiple conformer pairs.

```bash
mepd run-netgen --start reactants.xyz --end products.xyz --inputs inputs.toml
```

**Options:**

| Option | Description |
|--------|-------------|
| `--start` | Path to reactant conformers file |
| `--end` | Path to product conformers file |
| `--inputs`, `-i` | Path to RunInputs TOML file |
| `--name` | Custom name for output files |
| `--charge` | Molecular charge (default: 0) |
| `--multiplicity` | Spin multiplicity (default: 1) |
| `--max-pairs` | Maximum number of pairs to process (default: 500) |
| `--minimize-ends` | Minimize endpoint geometries before NEB |

---

### make-netgen-summary

Create a summary plot and network from network generation results.

```bash
mepd make-netgen-summary --directory ./results
```

**Options:**

| Option | Description |
|--------|-------------|
| `--directory` | Path to data directory containing network results |
| `--inputs`, `-i` | Path to RunInputs TOML file |
| `--name` | Name of pot and summary file (default: netgen) |
| `--verbose` | Enable verbose output |

---

### make-netgen-path

Create a path through a network.

```bash
mepd make-netgen-path --name network.json --inds 0 5 10
```

**Options:**

| Option | Description |
|--------|-------------|
| `--name` | Path to JSON file containing network object |
| `--inds` | Sequence of node indices to create path for |

## Input File Format

The CLI uses TOML input files. See [Input Parameters](inputs.md) for details.

**Example input file:**

```toml
[RunInputs]
engine_name = "qcop"
program = "xtb"
path_min_method = "NEB"

[program_kwds]
[program_kwds.model]
method = "GFN2xTB"

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

### MLPGI TOML Example

```toml
engine_name = "chemcloud"
program = "crest"
path_min_method = "mlpgi"

[path_min_inputs]
backend = "fairchem"
device = "cpu"
dtype = "float32"
model_path = "esen_sm_conserving_all.pt"
model_repo = "facebook/OMol25"
auto_download_model = true
```

If auto-download is enabled and the model repo is gated, authenticate first:

```bash
huggingface-cli login
```

To pre-download the checkpoint manually:

```bash
huggingface-cli download facebook/OMol25 checkpoints/esen_sm_conserving_all.pt --local-dir .
```
