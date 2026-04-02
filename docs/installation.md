# Installation & Setup

This page covers:

- installing `neb-dynamics` with either `uv` or `pip`
- validating the CLI install
- configuring electronic-structure backends
- setting up the optional `retropaths` repository for network-growth features

## Prerequisites

- Python 3.10+
- Git
- A shell environment where `python`, `pip`, and/or `uv` are available

## Install With `uv` (recommended for development)

### 1. Clone the repository

```bash
git clone https://github.com/mtzgroup/neb-dynamics.git
cd neb-dynamics
```

### 2. Create/sync the environment

```bash
uv sync
```

### 3. Verify install

```bash
uv run mepd --help
uv run python -c "import neb_dynamics; print('neb_dynamics import OK')"
```

### 4. (Optional) Editable install workflow

If you want your active environment to track local source edits directly:

```bash
uv pip install -e .
```

## Install With `pip`

### Option A: Install from GitHub directly

```bash
pip install "git+https://github.com/mtzgroup/neb-dynamics.git"
```

### Option B: Install from a local clone

```bash
git clone https://github.com/mtzgroup/neb-dynamics.git
cd neb-dynamics
pip install .
```

Editable local install:

```bash
pip install -e .
```

Verify:

```bash
mepd --help
python -c "import neb_dynamics; print('neb_dynamics import OK')"
```

## Backend Setup (required to run calculations)

NEB Dynamics needs a configured backend through your `RunInputs` (for example ChemCloud/QCOP/other supported engines).

### ChemCloud setup

1. Sign up: <https://chemcloud.mtzlab.com/signup>
2. Configure auth using either:

```bash
# Writes credentials to ~/.chemcloud/credentials
python -c "from chemcloud import setup_profile; setup_profile()"
```

or environment variables:

```bash
export CHEMCLOUD_USERNAME=your_email@chemcloud.com
export CHEMCLOUD_PASSWORD=your_password
```

Optional custom server:

```bash
export CHEMCLOUD_DOMAIN="https://your-server-url.com"
```

## `retropaths` Setup (optional but required for network growth/template actions)

`retropaths` is required for:

- `mepd netgen-smiles`
- MEPD Drive reaction-template growth actions (`+` in the UI)

### 1. Clone `retropaths`

```bash
git clone https://github.com/mtzgroup/retropaths.git ~/retropaths
```

### 2. Point NEB Dynamics to the repo

NEB Dynamics checks `RETROPATHS_REPO` first, then `~/retropaths`.

```bash
export RETROPATHS_REPO=~/retropaths
```

### 3. Locate your reactions library

You typically pass a `reactions.p` file path in CLI/UI inputs:

```bash
ls "$RETROPATHS_REPO"/data/reactions.p
```

Use this file path with:

- `mepd netgen-smiles --reactions-fp ...`
- MEPD Drive `Reactions File` field

## Quick Sanity Checks

### CLI check

```bash
mepd --help
```

### Drive check (local)

```bash
uv run mepd drive --inputs examples/example_inputs.toml --no-open
```

If this starts and prints a URL, your installation path is healthy.

## Related Pages

- [CLI Reference](cli.md) for command-level usage
- [MEPD Drive Remote](drive_remote.md) for remote launch + SSH tunnel workflow
