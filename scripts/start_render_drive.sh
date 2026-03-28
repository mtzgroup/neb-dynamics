#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT_VALUE="${PORT:-10000}"
HOST_VALUE="${MEPD_DRIVE_HOST:-0.0.0.0}"
WORKDIR_VALUE="${MEPD_DRIVE_WORKDIR:-/tmp/mepd-drive}"
WORKSPACE_VALUE="${MEPD_DRIVE_WORKSPACE:-}"
INPUTS_VALUE="${MEPD_DRIVE_INPUTS:-examples/charla_pr_data/example_inputs.toml}"
SMILES_VALUE="${MEPD_DRIVE_SMILES:-C=C.CCC=N}"
ENVIRONMENT_VALUE="${MEPD_DRIVE_ENVIRONMENT:-O}"
NAME_VALUE="${MEPD_DRIVE_NAME:-render-drive}"
CREDENTIALS_DIR_VALUE="${CHEMCLOUD_BASE_DIRECTORY:-$HOME/.chemcloud}"
CREDENTIALS_FILE_VALUE="${CHEMCLOUD_CREDENTIALS_FILE:-credentials}"
CHEMCLOUD_CREDENTIALS_TOML_VALUE="${CHEMCLOUD_CREDENTIALS_TOML:-}"
CHEMCLOUD_CREDENTIALS_B64_VALUE="${CHEMCLOUD_CREDENTIALS_B64:-}"

mkdir -p "$WORKDIR_VALUE"

if [[ -n "$CHEMCLOUD_CREDENTIALS_B64_VALUE" || -n "$CHEMCLOUD_CREDENTIALS_TOML_VALUE" ]]; then
  mkdir -p "$CREDENTIALS_DIR_VALUE"
  CREDENTIALS_TARGET="$CREDENTIALS_DIR_VALUE/$CREDENTIALS_FILE_VALUE"
  if [[ -n "$CHEMCLOUD_CREDENTIALS_B64_VALUE" ]]; then
    printf '%s' "$CHEMCLOUD_CREDENTIALS_B64_VALUE" | base64 --decode > "$CREDENTIALS_TARGET"
  else
    printf '%s\n' "$CHEMCLOUD_CREDENTIALS_TOML_VALUE" > "$CREDENTIALS_TARGET"
  fi
fi

if [[ -n "$WORKSPACE_VALUE" ]]; then
  SOURCE_WORKSPACE="$WORKSPACE_VALUE"
  if [[ "$SOURCE_WORKSPACE" != /* ]]; then
    SOURCE_WORKSPACE="$ROOT_DIR/$SOURCE_WORKSPACE"
  fi
  SOURCE_WORKSPACE="$(cd "$(dirname "$SOURCE_WORKSPACE")" && pwd)/$(basename "$SOURCE_WORKSPACE")"
  TARGET_WORKSPACE="$WORKDIR_VALUE"
  rm -rf "$TARGET_WORKSPACE"
  mkdir -p "$(dirname "$TARGET_WORKSPACE")"
  cp -R "$SOURCE_WORKSPACE" "$TARGET_WORKSPACE"
  export RENDER_WORKSPACE_TARGET="$TARGET_WORKSPACE"
  export RENDER_WORKSPACE_INPUTS="$ROOT_DIR/$INPUTS_VALUE"
  uv run python - <<'PY'
import json
import os
from pathlib import Path

workspace_dir = Path(os.environ["RENDER_WORKSPACE_TARGET"]).resolve()
workspace_fp = workspace_dir / "workspace.json"
payload = json.loads(workspace_fp.read_text())
old_workdir = str(payload.get("workdir") or "").rstrip("/")
payload["workdir"] = str(workspace_dir)
payload["inputs_fp"] = os.environ["RENDER_WORKSPACE_INPUTS"]
workspace_fp.write_text(json.dumps(payload, indent=2, sort_keys=True))

queue_fp = workspace_dir / "neb_queue.json"
if queue_fp.exists() and old_workdir:
    queue_payload = json.loads(queue_fp.read_text())
    for item in queue_payload.get("items", []):
        for key in ("result_dir", "output_chain_xyz"):
            value = item.get(key)
            if isinstance(value, str) and value.startswith(old_workdir):
                item[key] = str(workspace_dir / Path(value).relative_to(old_workdir))
    queue_fp.write_text(json.dumps(queue_payload, indent=2, sort_keys=True))
PY
  exec uv run mepd drive \
    --host "$HOST_VALUE" \
    --port "$PORT_VALUE" \
    --workspace "$TARGET_WORKSPACE" \
    --no-open
fi

exec uv run mepd drive \
    --host "$HOST_VALUE" \
    --port "$PORT_VALUE" \
    --directory "$WORKDIR_VALUE" \
    --inputs "$INPUTS_VALUE" \
    --smiles "$SMILES_VALUE" \
    --environment "$ENVIRONMENT_VALUE" \
    --name "$NAME_VALUE" \
    --no-open
