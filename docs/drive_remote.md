# MEPD Drive (Remote)

This page documents the standard workflow for launching **MEPD Drive** on a remote machine and connecting from your laptop over SSH.

## When to use this

Use remote Drive when:

- your compute environment (engines/credentials/filesystem) lives on a server or cluster
- you want to keep long-running jobs on that remote machine
- you only need browser access locally

## 1. Launch Drive on the remote machine

On the remote host:

```bash
cd /path/to/neb-dynamics
git switch <branch-name>
uv run mepd drive \
  --workspace /path/to/mepd-drive/<run-folder> \
  --host 127.0.0.1 \
  --port 51113 \
  --no-open
```

Notes:

- `--host 127.0.0.1` keeps the server private to the remote machine
- fixed `--port` makes tunneling predictable
- use `--workspace` to resume an existing run

## 2. Create an SSH tunnel from your laptop

On your laptop:

```bash
ssh -N -L 51113:127.0.0.1:51113 <user>@<remote-host>
```

Keep this terminal open while using Drive.

## 3. Open Drive locally

In your local browser:

```text
http://127.0.0.1:51113/
```

## Alternate: ask Drive to print tunnel command

You can have `mepd drive` print a ready-made SSH tunnel command:

```bash
uv run mepd drive \
  --workspace /path/to/mepd-drive/<run-folder> \
  --host 127.0.0.1 \
  --port 51113 \
  --ssh-login <user>@<remote-host> \
  --local-port 51113 \
  --no-open
```

## Bootstrap a new remote run from CLI

If you do not already have a workspace:

```bash
uv run mepd drive \
  --smiles "C=CC(O)CC=C" \
  --product-smiles "C=CC(=O)CC=C" \
  --environment "O" \
  --inputs examples/example_inputs.toml \
  --name allylic_alcohol_drive \
  --host 127.0.0.1 \
  --port 51113 \
  --no-open
```

Then tunnel and open the same way.

## Troubleshooting

### Tunnel connects but page does not load

- confirm remote Drive is still running
- verify remote port and local tunnel port match
- check that remote launch used `--host 127.0.0.1`

### Port already in use

- choose a different port on both commands, for example `52222`

### Wrong branch/version in UI

- restart Drive after pulling/changing branches
- hard-refresh the browser after restart

### Existing workspace won’t load

- pass the workspace directory directly with `--workspace`
- ensure `workspace.json` exists inside that directory

## Related

- [Installation & Setup](installation.md)
- [CLI Reference](cli.md#drive)
