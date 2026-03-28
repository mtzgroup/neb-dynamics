# Input Parameters

This page documents the `RunInputs` TOML schema accepted on the `master` branch.
It is derived from `neb_dynamics/inputs.py` on this branch, so it reflects the
actual keys and defaults the CLI accepts today.

## Top-Level Layout

A typical inputs file looks like this:

```toml
engine_name = "chemcloud"
program = "xtb"
chemcloud_queue = "celery"
path_min_method = "NEB"

[path_min_inputs]
max_steps = 500
v = false

[chain_inputs]
k = 0.1
delta_k = 0.09
do_parallel = true
use_geodesic_interpolation = true
friction_optimal_gi = true
node_freezing = true
node_rms_thre = 5.0
node_ene_thre = 5.0
frozen_atom_indices = ""

[gi_inputs]
nimages = 10
friction = 0.001
nudge = 0.1
align = true

[gi_inputs.extra_kwds]

[program_kwds]
cmdline_args = []

[program_kwds.keywords]
threads = 1

[program_kwds.extras]

[program_kwds.files]

[program_kwds.model]
method = "GFN2xTB"
basis = "GFN2xTB"

[program_kwds.model.extras]

[optimizer_kwds]
name = "cg"
timestep = 0.5
```

## Top-Level Keys

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `engine_name` | `str` | `"chemcloud"` | Supported: `"qcop"`, `"chemcloud"`, `"ase"` |
| `program` | `str` | `"xtb"` | Common values: `"xtb"`, `"crest"`, `"terachem"`; `ase` currently only supports `"omol25"` |
| `chemcloud_queue` | `str \| null` | `null` | Queue passed to `QCOPEngine` when using `engine_name = "chemcloud"` |
| `path_min_method` | `str` | `"NEB"` | Supported: `"NEB"`, `"FNEB"`, `"MLPGI"` |
| `path_min_inputs` | `table` | branch-dependent defaults | See below |
| `chain_inputs` | `table` | `ChainInputs()` defaults | See below |
| `gi_inputs` | `table` | `GIInputs()` defaults | See below |
| `program_kwds` | `table` | inferred from `engine_name`/`program` | Parsed into `qcio.ProgramArgs` for `qcop` / `chemcloud` |
| `optimizer_kwds` | `table` | `{ name = "cg", timestep = 0.5 }` | See optimizer names below |

## `path_min_inputs` for `NEB`

These keys are accepted when `path_min_method = "NEB"`.

| Key | Type | Default |
| --- | --- | --- |
| `climb` | `bool` | `false` |
| `en_thre` | `float` | `1e-4` |
| `rms_grad_thre` | `float` | `0.02` |
| `max_rms_grad_thre` | `float` | `0.05` |
| `skip_identical_graphs` | `bool` | `true` |
| `ts_grad_thre` | `float` | `0.05` |
| `ts_spring_thre` | `float` | `0.02` |
| `barrier_thre` | `float` | `0.1` |
| `early_stop_force_thre` | `float` | `0.0` |
| `negative_steps_thre` | `int` | `5` |
| `positive_steps_thre` | `int` | `10` |
| `use_geodesic_tangent` | `bool` | `false` |
| `do_elem_step_checks` | `bool` | `false` |
| `max_steps` | `float` | `500` |
| `v` | `bool` | `false` |

Example:

```toml
[path_min_inputs]
climb = false
en_thre = 0.0001
rms_grad_thre = 0.02
max_rms_grad_thre = 0.05
skip_identical_graphs = true
ts_grad_thre = 0.05
ts_spring_thre = 0.02
barrier_thre = 0.1
early_stop_force_thre = 0.03
negative_steps_thre = 5
positive_steps_thre = 10
use_geodesic_tangent = false
do_elem_step_checks = false
max_steps = 500
v = false
```

## `path_min_inputs` for `FNEB`

When `path_min_method = "FNEB"`, the branch provides these defaults:

| Key | Default |
| --- | --- |
| `max_min_iter` | `100` |
| `max_grow_iter` | `20` |
| `verbosity` | `1` |
| `skip_identical_graphs` | `true` |
| `do_elem_step_checks` | `true` |
| `grad_tol` | `0.05` |
| `barrier_thre` | `5` |
| `tangent` | `"geodesic"` |
| `tangent_alpha` | `1.0` |
| `use_xtb_grow` | `true` |
| `distance_metric` | `"GEODESIC"` |
| `min_images` | `10` |
| `todd_way` | `true` |
| `dist_err` | `0.1` |

Any provided values override these defaults.

## `path_min_inputs` for `MLPGI`

`MLPGI` uses an empty default table on this branch. Keys are effectively
backend-specific passthrough.

## `chain_inputs`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `k` | `float` | `0.1` | NEB spring constant |
| `delta_k` | `float` | `0.09` | Energy-weighted spring parameter |
| `do_parallel` | `bool` | `true` | Parallel gradient/energy computation |
| `use_geodesic_interpolation` | `bool` | `true` | Use geodesic interpolation for initial path generation |
| `friction_optimal_gi` | `bool` | `true` | Optimize GI friction parameter |
| `node_freezing` | `bool` | `true` | Freeze converged nodes |
| `node_rms_thre` | `float` | `5.0` | Node RMS threshold |
| `node_ene_thre` | `float` | `5.0` | Node energy threshold |
| `frozen_atom_indices` | `str` | `""` | Space-separated atom indices |

Example:

```toml
[chain_inputs]
k = 0.1
delta_k = 0.09
do_parallel = true
use_geodesic_interpolation = true
friction_optimal_gi = true
node_freezing = true
node_rms_thre = 5.0
node_ene_thre = 5.0
frozen_atom_indices = ""
```

## `gi_inputs`

| Key | Type | Default |
| --- | --- | --- |
| `nimages` | `int` | `10` |
| `friction` | `float` | `0.001` |
| `nudge` | `float` | `0.1` |
| `align` | `bool` | `true` |
| `extra_kwds` | `table` | `{}` |

Example:

```toml
[gi_inputs]
nimages = 10
friction = 0.001
nudge = 0.1
align = true

[gi_inputs.extra_kwds]
```

## `program_kwds`

For `engine_name = "qcop"` or `"chemcloud"`, this section is parsed into a
`qcio.ProgramArgs` object.

### Structure

```toml
[program_kwds]
cmdline_args = []

[program_kwds.keywords]
threads = 1

[program_kwds.extras]

[program_kwds.files]

[program_kwds.model]
method = "GFN2xTB"
basis = "GFN2xTB"

[program_kwds.model.extras]
```

### Defaults

If `program_kwds` is omitted, defaults depend on `program`:

- `program = "xtb"`
  - if `crest` is available: method/basis `gfn2`, `threads = 1`, and `program`
    is internally changed to `"crest"`
  - otherwise: method/basis `GFN2xTB`
- `program` containing `"terachem"`
  - method `ub3lyp`
  - basis `3-21g`

## `optimizer_kwds`

`optimizer_kwds.name` selects the optimizer implementation.

Supported names:

- `cg`
- `conjugate_gradient`
- `vpo`
- `velocity_projected`
- `lbfgs`
- `adam`
- `amg`
- `adaptive_momentum`
- `fire`

Defaults:

```toml
[optimizer_kwds]
name = "cg"
timestep = 0.5
```

## QMMM TOML Inputs

Use `engine_name = "qmmm"` with a `[qmmm_inputs]` block.

```toml
engine_name = "qmmm"
program = "terachem"
chemcloud_queue = "gpu-a100"   # optional; overrides env queue

[program_kwds.model]
method = "b3lyp"
basis = "6-31g**"

[program_kwds.keywords]
dispersion = "yes"
maxit = 500

[qmmm_inputs]
qminds_fp = "qmindices.dat"      # required
prmtop_fp = "ref.prmtop"         # required
rst7_fp_react = "ref.rst7"       # required
tcin_fp = "tc.in"                # optional; if omitted, tc.in is generated from TOML
compute_program = "chemcloud"    # "chemcloud" or "qcop"
print_stdout = false             # stream QCOP stdout if true
```

### `qmmm_inputs` reference

| Key | Required | Description |
|-----|----------|-------------|
| `qminds_fp` | yes | Path to QM atom index file |
| `prmtop_fp` | yes | Path to AMBER topology |
| `rst7_fp_react` | yes | Reference rst7 used for coordinate replacement |
| `rst7_fp_prod` | no | Optional product reference rst7 |
| `tcin_fp` | no | TeraChem input file path |
| `tcin_text` | no | Inline `tc.in` contents |
| `compute_program` | no | `"chemcloud"` (default) or `"qcop"` |
| `chemcloud_queue` | no | Queue name (overrides env) |
| `print_stdout` | no | Stream stdout during compute calls |
| `debug_dump_inputs` | no | Dump bundled QM/MM inputs for debugging |
| `debug_dump_dir` | no | Directory for debug input dumps |

### Generated `tc.in`

If `tcin_fp`/`tcin_text` is not provided, `RunInputs` generates `tc.in` from TOML:

- QM/MM file references (`coordinates ref.rst7`, `qmindices qmindices.dat`, `prmtop ref.prmtop`)
- `charge`, `spinmult`, `run` type
- method/basis and extra keywords from `[program_kwds]`

### Frozen atoms for QM/MM NEB

Configure frozen atoms in `[chain_inputs]` using `frozen_atom_indices`.
Keep endpoint and NEB frozen regions consistent to avoid interior images becoming artificially lower than endpoints.

If you already have a TeraChem `.in` file (for example `s0min_3A_reactant.in`), you can generate a matching QMMM TOML with:

```bash
mepd toml-from-tcin s0min_3A_reactant.in --output qmmm_inputs_s0min_frozen.toml
```

This helper parses `$constraints` `atom N` lines and stores them as 0-based `frozen_atom_indices`.

## Minimal Examples

### ChemCloud xTB run

```toml
engine_name = "chemcloud"
program = "xtb"
chemcloud_queue = "celery"
path_min_method = "NEB"

[path_min_inputs]
max_steps = 250
v = true

[chain_inputs]
use_geodesic_interpolation = true

[gi_inputs]
nimages = 10

[optimizer_kwds]
name = "cg"
timestep = 0.5
```

### Local QCOP TeraChem run

```toml
engine_name = "qcop"
program = "terachem"
path_min_method = "NEB"

[path_min_inputs]
max_steps = 100
v = true

[program_kwds]
cmdline_args = []

[program_kwds.keywords]
gpus = 1
sphericalbasis = false

[program_kwds.model]
method = "b3lyp"
basis = "6-31g**"

[optimizer_kwds]
name = "cg"
timestep = 0.5
```

## Loading and Saving

```python
from neb_dynamics.inputs import RunInputs

ri = RunInputs.open("inputs.toml")
ri.save("copied_inputs.toml")
```

## CLI Notes

The CLI reads this file with:

```bash
mepd run --start start.xyz --end end.xyz --inputs inputs.toml
```

A few options are CLI flags, not TOML keys. For example:

- `--recursive`
- `--use-tsopt`
- `--create-irc`
- `--minimize-ends`

Those belong on the command line, not inside `inputs.toml`.
