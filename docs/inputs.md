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
| `path_min_method` | `str` | `"NEB"` | Supported: `"NEB"`, `"FNEB"`, `"MLPGI"`, `"NEB-DLF"` |
| `path_min_inputs` | `table` | branch-dependent defaults | See below |
| `chain_inputs` | `table` | `ChainInputs()` defaults | See below |
| `gi_inputs` | `table` | `GIInputs()` defaults | See below |
| `program_kwds` | `table` | inferred from `engine_name`/`program` | Parsed into `qcio.ProgramArgs` for `qcop` / `chemcloud` |
| `optimizer_kwds` | `table` | `{ name = "cg", timestep = 0.5 }` | See optimizer names below |

Notes:
- For `engine_name = "ase"` with `program = "omol25"`, the FAIR-Chem checkpoint for
  endpoint minimization and all ASE energy/gradient calls is resolved from:
  `program_kwds.model_path` (or `path_min_inputs.model_path`) and
  `program_kwds.device` (or `path_min_inputs.device`).
- If omitted, fallback defaults are `model_path = "/home/diptarka/fairchem/esen_sm_conserving_all.pt"`
  and `device = "cuda"`.

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

`MLPGI` defaults now map directly to the two-stage geodesic optimization
configuration described in:

- [Locating Ab Initio Transition States via Geodesic Construction on Machine-Learned Potential Energy Surfaces](https://doi.org/10.1021/acs.jctc.5c01221)
- [ArXiv full text with Table 1 / Section II](https://arxiv.org/html/2507.17968v2)

### MLPGI keys

| Key | Default | Units | Notes |
| --- | --- | --- | --- |
| `skip_identical_graphs` | `true` | — | MSMEP endpoint short-circuit |
| `do_elem_step_checks` | `true` | — | Run elementary-step checks |
| `v` | `false` | — | Verbose progress |
| `backend` | `"fairchem"` | — | MLP backend (`fairchem`, `mace`, etc.) |
| `model_path` | `"esen_sm_conserving_all.pt"` | path | Checkpoint path or repo filename |
| `auto_download_model` | `false` | — | Auto-fetch checkpoint when missing |
| `model_repo` | `"facebook/OMol25"` | repo id | Hugging Face repo used for download |
| `model_cache_dir` | `null` | path | Optional download cache directory |
| `hf_token` | `null` | token | Optional auth token for gated repos |
| `device` | `null` | — | Auto-selects CUDA/CPU when `null` |
| `dtype` | `"float32"` | — | Torch dtype (`float32`, `float64`) |
| `fire_stage1_iter` | `200` | iterations | Initial relaxation stage max iterations |
| `fire_stage2_iter` | `500` | iterations | Refinement stage max iterations |
| `fire_grad_tol` | `0.01` | eV/Angstrom | FIRE infinity-norm gradient tolerance |
| `variance_penalty_weight` | `0.0433641` | eV | Segment penalty weight (`beta = 1 kcal/mol`) |
| `fire_conv_window` | `20` | iterations | Convergence window size |
| `fire_conv_geolen_tol` | `0.25` | kcal/mol | Path-length span tolerance (converted to eV internally) |
| `fire_conv_erelpeak_tol` | `0.25` | kcal/mol | Barrier-span tolerance (converted to eV internally) |
| `refinement_step_interval` | `10` | iterations | Check interval for node insertion |
| `refinement_dynamic_threshold_fraction` | `0.1` | fraction | Refinement cutoff (`10%`) |
| `tangent_project` | `true` | — | Project tangent component from path-length gradient |
| `climb` | `true` | — | Enable climbing image in stage 2 |
| `alpha_climb` | `0.5` | — | Climbing force scaling factor |

### Compatibility aliases

To make paper-style input keys work directly in `[path_min_inputs]`, MLPGI also accepts:

| Alias key | Interpreted as | Conversion |
| --- | --- | --- |
| `beta` | `variance_penalty_weight` | kcal/mol -> eV |
| `tau_refine` | `refinement_step_interval` | direct |
| `cutoff` | `refinement_dynamic_threshold_fraction` | `%` if value > 1, else fraction |
| `convergence_window` | `fire_conv_window` | direct |
| `path_length_tolerance` | `fire_conv_geolen_tol` | kcal/mol -> eV |
| `barrier_height_tolerance` | `fire_conv_erelpeak_tol` | kcal/mol -> eV |

### Tuning guide (paper-aligned)

Use these defaults first. They were reported by the paper for robust behavior.

| If you need... | Primary knobs | Practical direction |
| --- | --- | --- |
| Better TS-region focus | `climb`, `alpha_climb` | Keep `climb=true`; increase `alpha_climb` cautiously. The paper warns large values (for example `>1`) can become unstable. |
| More aggressive refinement | `refinement_step_interval`, `refinement_dynamic_threshold_fraction` | Lower interval and/or increase cutoff fraction to insert nodes more readily near poorly fit segments. |
| Smoother spacing in energy | `variance_penalty_weight` (or `beta`) | Raise slightly to enforce more uniform energy-space segment lengths; lower if refinement over-constrains the path. |
| Stricter convergence | `fire_grad_tol`, `fire_conv_geolen_tol`, `fire_conv_erelpeak_tol`, `fire_conv_window` | Decrease tolerances and/or increase window. This improves stability checks but costs more iterations. |
| Faster runs | `fire_stage1_iter`, `fire_stage2_iter` | Reduce iteration caps for cheaper, less-refined guesses. |
| Better throughput | `device`, `dtype` | Use GPU when available; the paper notes geodesic optimization can involve hundreds of iterations over ~40 nodes/midpoints. |

Example:

```toml
path_min_method = "mlpgi"

[path_min_inputs]
backend = "fairchem"
model_path = "esen_sm_conserving_all.pt"
auto_download_model = true
device = "cuda"
dtype = "float32"

# Paper-style aliases (also accepted)
beta = 1.0
tau_refine = 10
cutoff = 10

# Or use canonical internal keys directly
fire_stage1_iter = 200
fire_stage2_iter = 500
fire_grad_tol = 0.01
alpha_climb = 0.5
```

## `path_min_inputs` for `NEB-DLF`

`NEB-DLF` runs TeraChem DL-Find NEB through `QCOPEngine` (`program = "terachem"`).

| Key | Default | Notes |
| --- | --- | --- |
| `nstep` | `200` | DL-Find max steps |
| `min_image` | `null` | If `null`, uses current chain image count |
| `min_nebk` | `0.01` | DL-Find NEB spring constant |
| `max_nebk` | `null` | Optional spring cap |
| `new_minimizer` | `"no"` | Keep DL-Find pathway in TeraChem |
| `skip_identical_graphs` | `true` | MSMEP endpoint short-circuit |
| `do_elem_step_checks` | `true` | Run autosplitting elementary-step checks |
| `collect_files` | `true` | Required to parse `nebpath.xyz`/`nebinfo` |
| `dlfind_keywords` | `{}` | Extra DL-Find/TeraChem keyword passthrough |
| `v` | `false` | Verbose progress |

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
