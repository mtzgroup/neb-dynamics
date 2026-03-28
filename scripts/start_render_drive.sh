#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT_VALUE="${PORT:-10000}"
HOST_VALUE="${MEPD_DRIVE_HOST:-0.0.0.0}"
WORKDIR_VALUE="${MEPD_DRIVE_WORKDIR:-/tmp/mepd-drive}"
INPUTS_VALUE="${MEPD_DRIVE_INPUTS:-examples/charla_pr_data/example_inputs.toml}"
SMILES_VALUE="${MEPD_DRIVE_SMILES:-C=C.CCC=N}"
ENVIRONMENT_VALUE="${MEPD_DRIVE_ENVIRONMENT:-O}"
NAME_VALUE="${MEPD_DRIVE_NAME:-render-drive}"

mkdir -p "$WORKDIR_VALUE"

exec uv run mepd drive \
  --host "$HOST_VALUE" \
  --port "$PORT_VALUE" \
  --directory "$WORKDIR_VALUE" \
  --inputs "$INPUTS_VALUE" \
  --smiles "$SMILES_VALUE" \
  --environment "$ENVIRONMENT_VALUE" \
  --name "$NAME_VALUE" \
  --no-open
