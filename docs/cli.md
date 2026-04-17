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
| `--rst7-prmtop` | Required only when `--start` or `--end` is `.rst7`; used to convert rst7 endpoints to XYZ internally |

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

For `.rst7` endpoints, provide a prmtop:

```bash
mepd run --start react.rst7 --end prod.rst7 --rst7-prmtop ref.prmtop --inputs qmmm_inputs.toml
```

---

### run-refine

Run a two-stage refinement workflow:
1. run a cheap discovery MEP/MSMEP,
2. reoptimize discovered minima at an expensive level,
3. run expensive pair-path minimizations between adjacent refined minima.

```bash
mepd run-refine PATH.xyz --inputs expensive.toml --cheap-inputs cheap.toml
```

**Options:**

| Option | Description |
|--------|-------------|
| `--start`, `-s` | Path to start structure file (or SMILES with `--use-smiles`) |
| `--end`, `-e` | Path to end structure file (or SMILES with `--use-smiles`) |
| `geometries` | Path to approximate path file (same convention as `run`) |
| `--inputs`, `-i` | Required expensive RunInputs TOML |
| `--cheap-inputs`, `-ci` | Optional cheap RunInputs TOML (defaults to `--inputs`) |
| `--recursive` | Use recursive MSMEP in cheap stage and expensive pair stage |
| `--recycle-nodes` | Seed expensive pair runs with cheap-path nodes instead of fresh interpolation |
| `--minimize-ends` | Minimize endpoints at cheap level before discovery |
| `--name` | Prefix for output files |
| `--charge` | Molecular charge (default: 0) |
| `--multiplicity` | Spin multiplicity (default: 1) |

**Example:**

```bash
mepd run-refine examples/oxycope.xyz \
  -i expensive_chemcloud_terachem.toml \
  -ci cheap_chemcloud_crest.toml \
  --recursive \
  --recycle-nodes \
  --name oxycope_refine
```

**Outputs:**

- `<name>_cheap.xyz`
- `<name>_cheap_msmep/` (recursive cheap stage)
- `<name>_refined_minima.xyz`
- `<name>_refined_pairs/`
- `<name>_refined.xyz`

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

`mepd ts` now uses the engine-level TS protocol (`compute_transition_state`) when available, including `QMMMEngine`.

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
| `--path-min-method`, `-pmm` | Path minimization method: `neb`, `fneb`, `mlpgi`, or `neb-dlf` (default: neb) |

---

### toml-from-tcin

Create a QMMM `RunInputs` TOML from a TeraChem input file (including frozen atoms from `$constraints`).

```bash
mepd toml-from-tcin tc.in --output qmmm_inputs_s0min_frozen.toml
```

**Options:**

| Option | Description |
|--------|-------------|
| `tcin` | Path to TeraChem input file (`.in` / `tc.in`) |
| `--output`, `-o` | Output TOML path (default: `qmmm_inputs_from_tc.toml`) |
| `--compute-program` | QMMM backend (`chemcloud` or `qcop`, default: `chemcloud`) |
| `--queue` | Optional ChemCloud queue to write into TOML |

Notes:
- Parses `method`, `basis`, `charge`, `spinmult`, `run`, and `min_coordinates`.
- Parses additional TeraChem keywords into `[program_kwds.keywords]`.
- Parses `$constraints` lines of the form `atom N` into 0-based `chain_inputs.frozen_atom_indices`.

---

### visualize

Render an interactive browser-based viewer for saved NEB/MSMEP results.

```bash
mepd visualize path/to/result
```

`result` can be either:

- a NEB output `.xyz` with sibling `<stem>_history/` folder
- a Pot network `.json`
- a recursive MSMEP TreeNode folder containing `adj_matrix.txt`
- a drive workspace directory (for example `mepd-drive-1775282164/`)
- a drive `workspace.json` file

**Options:**

| Option | Description |
|--------|-------------|
| `result_path` | Path to result `.xyz`, network `.json`, TreeNode folder, or drive workspace |
| `--output`, `-o` | Output HTML path (default: `<name>_visualize.html`) |
| `--qminds-fp` | Path to `qmindices.dat` to visualize only selected atoms |
| `--atom-indices` | Comma/space-separated atom indices for subset visualization |
| `--charge` | Charge for reading serialized geometries (default: 0) |
| `--multiplicity` | Multiplicity for reading serialized geometries (default: 1) |
| `--no-open` | Do not automatically open browser |

**Example:**

```bash
mepd visualize mep_output_neb.xyz
mepd visualize mep_output_msmep --output msmep_view.html
mepd visualize mepd-drive-1775282164
mepd visualize mep_output_neb.xyz --qminds-fp qmindices.dat
mepd visualize mep_output_neb.xyz --atom-indices "12,13,14,15"
```

---

### netgen-smiles

Grow a retropaths reaction network from a root SMILES string, convert each node into a structure-bearing NEB network, minimize endpoints, queue recursive MSMEP runs, and write a live status page.

```bash
mepd netgen-smiles \
  --smiles "C=CC(O)CC=C" \
  --environment "O" \
  --reactions-fp /path/to/reactions.p \
  --inputs examples/example_inputs.toml \
  --name allylic_alcohol_water
```

**Options:**

| Option | Description |
|--------|-------------|
| `--smiles`, `-s` | Root reactant SMILES |
| `--environment`, `-e` | Environment SMILES (optional) |
| `--reactions-fp` | Path to the retropaths `reactions.p` library |
| `--inputs`, `-i` | Path minimization `RunInputs` TOML |
| `--name` | Workspace name / default output directory |
| `--directory`, `-d` | Existing or new workspace directory |
| `--timeout-seconds` | Retropaths growth timeout |
| `--max-nodes` | Maximum retropaths pot size |
| `--max-depth` | Maximum retropaths search depth |
| `--max-parallel-nebs` | Number of recursive NEBs to run concurrently |
| `--no-open` | Do not automatically open `status.html` |

**Example:**

```bash
uv run mepd netgen-smiles \
  --smiles "C=CC(O)CC=C" \
  --environment "O" \
  --reactions-fp /Users/janestrada/retropaths/data/reactions.p \
  --inputs examples/example_inputs.toml \
  --name allylic_alcohol_water \
  --max-nodes 10 \
  --max-parallel-nebs 4
```

**What it does:**

1. grows the retropaths pot from the input SMILES
   using the configured `reactions.p` library
2. converts each pot node into a `StructureNode` while preserving molecular graph atom indices
3. minimizes each endpoint structure once, caching the optimization result in the workspace
4. builds a persistent NEB queue
5. runs recursive MSMEP on queued pairs
6. reconstructs a completed-results NEB pot from the finished leaf chains

**Workspace outputs:**

- `workspace.json`
- `retropaths_pot.json`
- `neb_pot.json`
- `neb_queue.json`
- `neb_pot_annotated.json`
- `queue_runs/`
- `endpoint_optimizations/`
- `status.html`

Notes:
- completed NEB chains are treated as bidirectional when reconstructing the completed network
- each autosplit leaf is labeled as `ReactionName(step N)` in the completed NEB graph
- reverse edges inherit the same step label and use the reversed chain with the reverse barrier

---

### drive

Launch the interactive MEPD Drive web UI.

`drive` can start in three modes:

1. blank interactive mode (initialize from the browser)
2. SMILES-bootstrap mode (create workspace before the browser opens)
3. resume mode (load an existing workspace immediately)

**Requirements (local):**

1. Install project dependencies (from source checkout):
   ```bash
   uv sync
   ```
2. Have a usable RunInputs TOML (for example `examples/example_inputs.toml`).
3. Configure your electronic-structure backend:
   - ChemCloud-backed runs: valid credentials (for example `~/.chemcloud/credentials`)
   - Local QCOP runs: required backend program binaries (such as `xtb`, `crest`, or `terachem`) on `PATH`
4. Optional but recommended for reaction-template features:
   - a `retropaths` checkout at `~/retropaths` (default lookup) or
   - set `RETROPATHS_REPO` to your `retropaths` checkout path

**Examples:**

```bash
# Blank drive session; initialize from the browser UI
uv run mepd drive --inputs examples/example_inputs.toml

# Start drive from SMILES on the command line
uv run mepd drive \
  --smiles "C=CC(O)CC=C" \
  --product-smiles "C=CC(=O)CC=C" \
  --environment "O" \
  --charge 0 \
  --multiplicity 1 \
  --inputs examples/example_inputs.toml \
  --name allylic_alcohol_drive

# Re-open an existing workspace mid-run
uv run mepd drive --workspace ./allylic_alcohol_drive

# `--directory` also resumes automatically when it already contains workspace.json
uv run mepd drive --directory ./allylic_alcohol_drive

# Disable network-split overlay if you only want base pot/queue visualization
uv run mepd drive --workspace ./allylic_alcohol_drive --no-network-splits
```

**Options:**

| Option | Description |
|--------|-------------|
| `--inputs`, `-i` | Path minimization `RunInputs` TOML. Required for `--smiles`; optional when resuming an existing workspace |
| `--smiles`, `-s` | Root reactant SMILES to bootstrap a new drive workspace before opening the UI |
| `--product-smiles`, `--end` | Optional product/end SMILES used to add a target endpoint and queue the initial edge |
| `--environment`, `-e` | Optional environment SMILES for SMILES-based drive initialization |
| `--charge` | Total charge for SMILES-bootstrapped endpoint structures |
| `--multiplicity` | Spin multiplicity for SMILES-bootstrapped endpoint structures |
| `--name` | Run name / workspace name for SMILES-based drive initialization |
| `--workspace` | Existing workspace directory, `workspace.json`, `*_network.json`, or `*_request_manifest.json` to load on startup |
| `--reactions-fp` | Path to the retropaths `reactions.p` library |
| `--directory`, `-d` | Base directory for new drive workspaces, or an existing workspace directory to resume |
| `--host` | Host interface for the local drive server |
| `--port` | Port for the local drive server (`0` selects a free port) |
| `--ssh-login` | SSH target used to print a ready-made tunnel command |
| `--local-port` | Local laptop-side port for the printed SSH tunnel command |
| `--timeout-seconds` | Retropaths growth timeout |
| `--max-nodes` | Maximum retropaths pot size |
| `--max-depth` | Maximum retropaths search depth |
| `--max-parallel-nebs` | Number of autosplitting NEBs to run concurrently |
| `--network-splits/--no-network-splits` | Enable/disable the recursive autosplit overlay in Drive |
| `--no-open` | Do not automatically open the browser |

**Drive behavior notes:**

- If you launch `drive` with no workspace flags, you can still initialize entirely from the browser.
- The browser-side initializer accepts an inputs TOML path, reactions file, environment SMILES, and mode (`reactant-only` or `reactant-product`).
- If you did not pass `--inputs` at launch, you must provide an inputs TOML path in the initializer UI before initialization.
- `netgen-smiles` and the Drive reaction-template `+` action require the optional `retropaths` repository. If unavailable, Drive now returns a clear error describing how to set `RETROPATHS_REPO`.
- When an autosplitting NEB finishes, drive now shows a barrier and viewer link even when the completed result is attached to the reverse edge or when the original attempted pair was split into intermediate edges.
- Completed queue-pair viewers are cached after first materialization so state polling stays responsive after the first finished NEB.
- When multiple node minimizations are submitted and the loaded inputs use a ChemCloud-backed engine, drive submits them as one ChemCloud batch instead of serial single-node jobs.

**Drive Kinetics Section (Rate-Equation tab):**

The `Kinetics` panel in Drive runs a deterministic population model over the currently merged Drive graph.

- Data source:
  The model is built from the active Drive network snapshot (`neb_pot` plus optional network-split overlay).
- Which edges are used:
  Directed edges with a numeric `barrier` are converted to rates; edges without barriers are ignored.
- Suppressed edges:
  A near-zero barrier edge can be suppressed when its stored NEB chain indicates the source endpoint is already at the chain maximum (guard against suspicious zero barriers).
- Rate law:
  Each retained edge uses `k = (k_B T / h) * exp(-barrier_kcal / (R * T))`.
- Initial conditions:
  If the input box is empty, defaults are node `0 = 1.0` and all other nodes `= 0.0`.
  If provided, `Initial Conditions (JSON)` must be a JSON object like `{"0": 1.0, "4": 0.25}`.
  Missing nodes are filled as `0.0`.
- Time controls:
  `Kinetics Final Time` can be left blank; Drive then uses an automatic horizon `10 / max(rate_constant)` from the current graph.
  `Kinetics Max Steps` is the number of implicit-Euler steps used in the integration.
- Solver behavior:
  The simulation integrates a linear rate-equation system (continuous populations), clamps negatives to zero, and renormalizes to conserve total population.
- What the panel shows:
  Node/edge/suppressed-edge counts, default initial state, a population-vs-time plot (top 6 final-population nodes), and final populations.
- Time interpretation:
  The plot axis is currently labeled `Time (a.u.)`; treat it as relative/arbitrary time unless you have separately validated a physical calibration.
- Model type note:
  This is not a stochastic Gillespie trajectory; it is a deterministic ODE-style evolution of populations.
- API naming note:
  The preferred endpoint is `/api/run-kinetics`; `/api/run-kmc` is still accepted as a backward-compatible alias.

**Running drive on a remote server over SSH:**

Start the server on the remote machine and bind it to loopback:

```bash
uv run mepd drive \
  --workspace ./allylic_alcohol_drive \
  --host 127.0.0.1 \
  --port 8123 \
  --no-open
```

Then create an SSH tunnel from your laptop:

```bash
ssh -N -L 9000:127.0.0.1:8123 user@remote-host
```

Open the tunneled URL locally:

```text
http://127.0.0.1:9000/
```

You can also ask `drive` to print the tunnel command for you:

```bash
uv run mepd drive \
  --workspace ./allylic_alcohol_drive \
  --host 127.0.0.1 \
  --port 8123 \
  --ssh-login user@remote-host \
  --local-port 9000 \
  --no-open
```

Remote-use notes:

- Keep `--host 127.0.0.1` unless you intentionally want the web server exposed on the remote network.
- `--workspace` is the cleanest way to reconnect to a run that is already in progress.
- If you are starting from SMILES on the remote host, use the same `drive --smiles ... --inputs ...` command there, then tunnel to the printed local URL.

For a dedicated remote-run walkthrough, see [MEPD Drive (Remote)](drive_remote.md).

---

### status

Regenerate the live status page for a `netgen-smiles` workspace.

```bash
mepd status --directory ./allylic_alcohol_water
```

**Options:**

| Option | Description |
|--------|-------------|
| `--directory`, `-d` | Workspace directory containing `workspace.json` |
| `--output`, `-o` | Optional output path for the generated HTML |
| `--temperature` | Temperature in kelvin for the kinetics widget |
| `--initial-condition` | Repeatable `NODE=VALUE` override for initial concentrations |
| `--no-open` | Do not automatically open the browser |

**Examples:**

```bash
uv run mepd status -d ./allylic_alcohol_water

uv run mepd status \
  -d ./allylic_alcohol_water \
  --temperature 350 \
  --initial-condition 0=0.8 \
  --initial-condition 1=0.2
```

`status.html` includes:

- the original retropaths pot graph
- the completed-results NEB pot graph
- graph-delta counts
- queue status tables
- clickable per-edge NEB viewers
- a concentration-vs-time kinetics widget over the completed directed NEB graph

Kinetics defaults:
- all nodes start at `0.0 M`
- node `0` starts at `1.0 M`

Kinetics note:
- the status-page kinetics widget now solves the linear rate equations over the directed completed NEB graph
- `End Time` is the simulation horizon
- `Time Points` is the numerical resolution used by the ODE integrator
- the time axis is currently best interpreted as relative/arbitrary time, not a validated physical seconds scale

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
write_qcio = true
path_min_method = "mlpgi"

[path_min_inputs]
backend = "fairchem"
device = "cpu"
dtype = "float32"
model_path = "esen_sm_conserving_all.pt"
model_repo = "facebook/OMol25"
auto_download_model = true
```

`write_qcio` is a top-level optional flag and defaults to `false`. When enabled, saved CLI outputs also include cached `qcio.ProgramOutput` sidecars (`*.qcio`), which can take substantial disk space.

If auto-download is enabled and the model repo is gated, authenticate first:

```bash
huggingface-cli login
```

To pre-download the checkpoint manually:

```bash
huggingface-cli download facebook/OMol25 checkpoints/esen_sm_conserving_all.pt --local-dir .
```
