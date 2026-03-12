from __future__ import annotations
import base64
import contextlib
import io
import json
import logging
from dataclasses import dataclass
import networkx as nx
import numpy as np
from typing import List
from itertools import product
from neb_dynamics.pot import plot_results_from_pot_obj
from neb_dynamics.pot import Pot
from neb_dynamics.helper_functions import (
    compute_irc_chain,
    parse_terachem_input_file,
    parse_symbols_from_prmtop,
    rst7_to_coords_and_indices,
)
from neb_dynamics.inputs import NetworkInputs, ChainInputs
from neb_dynamics.NetworkBuilder import NetworkBuilder
from neb_dynamics.qcio_structure_helpers import read_multiple_structure_from_file
from neb_dynamics.nodes.nodehelpers import displace_by_dr
from neb_dynamics.msmep import MSMEP
from neb_dynamics.chain import Chain
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.neb import NEB
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.nodes.nodehelpers import (
    _is_connectivity_identical,
    is_identical,
)
from neb_dynamics.inputs import RunInputs
from neb_dynamics.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from neb_dynamics.geodesic_interpolation2.fileio import write_xyz

import typer
from typing_extensions import Annotated

import os
from openbabel import openbabel
from qcio import Structure, ProgramOutput
from qcio.view import generate_structure_viewer_html
from qcop.exceptions import ExternalProgramError
import sys
from pathlib import Path
import time
import traceback
from datetime import datetime
import webbrowser
import shutil
import tomli_w

from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.status import Status
from rich.table import Table
from rich.box import Box
from rich.text import Text
from rich.syntax import Syntax
from rich import box
from neb_dynamics.chainhelpers import generate_neb_plot
from neb_dynamics.retropaths_workflow import (
    RetropathsWorkspace,
    create_workspace,
    prepare_neb_workspace,
    run_netgen_smiles_workflow,
    summarize_queue,
    write_status_html,
)

# Custom theme for Claude Code-like styling
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "header": "bold magenta",
    "dim": "dim",
})


class _SuppressWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING


logging.getLogger().addFilter(_SuppressWarningFilter())


def _configure_cli_logging():
    """Keep noisy third-party loggers quiet for CLI runs."""
    logger_specs = (
        ("chemcloud", False),
        ("chemcloud.client", False),
        ("chemcloud.models", False),
        ("qccodec", False),
        ("qccodec.codec", False),
        ("geometric", True),
        ("geometric.nifty", True),
        ("neb_dynamics.geodesic_interpolation2", False),
        ("neb_dynamics.geodesic_interpolation2.morsegeodesic", False),
        ("httpx", False),
        ("httpcore", False),
    )
    for logger_name, disable in logger_specs:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        for handler in logger.handlers:
            handler.setLevel(logging.WARNING)
        logger.disabled = disable


_configure_cli_logging()


ob_log_handler = openbabel.OBMessageHandler()


def _parse_kmc_initial_condition_overrides(
    overrides: list[str] | None,
) -> dict[int, float] | None:
    if not overrides:
        return None

    parsed: dict[int, float] = {}
    for override in overrides:
        if "=" not in override:
            raise typer.BadParameter(
                f"Invalid --initial-condition '{override}'. Use NODE=VALUE, for example 0=1.0."
            )
        node_text, value_text = override.split("=", 1)
        try:
            node_index = int(node_text.strip())
            value = float(value_text.strip())
        except ValueError as exc:
            raise typer.BadParameter(
                f"Invalid --initial-condition '{override}'. Use NODE=VALUE, for example 0=1.0."
            ) from exc
        parsed[node_index] = value
    return parsed
ob_log_handler.SetOutputLevel(0)

app = typer.Typer(
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)

# CLI Banner
BANNER = """
[bold magenta]╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║                        [bold cyan]NEB[/bold cyan] [bold white]-[/bold white] [bold cyan]Dynamics[/bold cyan]                         ║
║        [dim]Reaction Path Optimization & Network Generation[/dim]        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝[/bold magenta]
"""


def print_banner():
    """Print the CLI banner."""
    console.print(BANNER)


# Global console instance for consistent styling
console = Console(theme=custom_theme)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """Show banner before running any command."""
    _configure_cli_logging()
    if ctx.invoked_subcommand is None:
        print_banner()


def create_progress():
    """Create a rich progress bar for long-running tasks."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _truncate_label(label: str, max_len: int) -> str:
    if len(label) <= max_len:
        return label
    if max_len <= 3:
        return label[:max_len]
    return label[: max_len - 3] + "..."


def _build_ascii_energy_profile(energies, labels, width: int = 60, height: int = 12):
    if len(energies) == 0:
        return "No energies to plot."
    if len(energies) != len(labels):
        raise ValueError("labels must be same length as energies")

    min_e = float(min(energies))
    max_e = float(max(energies))
    if max_e == min_e:
        max_e = min_e + 1e-6

    def _xpos(i):
        if len(energies) == 1:
            return 0
        return int(round(i * (width - 1) / (len(energies) - 1)))

    def _ypos(e):
        return int(round((e - min_e) * (height - 1) / (max_e - min_e)))

    grid = [[" " for _ in range(width)] for _ in range(height)]

    xs = [_xpos(i) for i in range(len(energies))]
    ys = [_ypos(e) for e in energies]

    for i in range(len(energies) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        if x0 == x1:
            continue
        step = 1 if x1 > x0 else -1
        for x in range(x0, x1 + step, step):
            t = (x - x0) / (x1 - x0)
            y = int(round(y0 + (y1 - y0) * t))
            row = height - 1 - y
            if 0 <= row < height and 0 <= x < width and grid[row][x] == " ":
                grid[row][x] = "-"

    for x, y in zip(xs, ys):
        row = height - 1 - y
        if 0 <= row < height and 0 <= x < width:
            grid[row][x] = "*"

    prefix_len = len(f"{max_e:7.2f} |")
    lines = []
    for r in range(height):
        y_val = max_e - (max_e - min_e) * (r / (height - 1))
        prefix = f"{y_val:7.2f} |"
        lines.append(prefix + "".join(grid[r]))

    lines.append(" " * prefix_len + "-" * width)

    label_line = [" " for _ in range(width)]
    max_label_len = 12
    truncated = False
    for x, label in zip(xs, labels):
        tlabel = _truncate_label(label, max_label_len)
        if tlabel != label:
            truncated = True
        start = max(0, min(width - len(tlabel), x - len(tlabel) // 2))
        for i, ch in enumerate(tlabel):
            label_line[start + i] = ch
    lines.append(" " * prefix_len + "".join(label_line))

    if truncated:
        lines.append("")
        lines.append("Full SMILES labels:")
        for i, label in enumerate(labels):
            lines.append(f"{i}: {label}")

    return "\n".join(lines)


def _ascii_profile_for_chain(chain: Chain):
    try:
        energies = chain.energies_kcalmol
    except Exception as exc:
        console.print(
            f"[yellow]⚠ Could not compute energy profile: {exc}[/yellow]")
        return

    # Keep x-axis labeling consistent with live NEB tables: node indices only.
    labels = [str(i) for i, _ in enumerate(chain.nodes)]

    plot = _build_ascii_energy_profile(energies, labels)
    console.print("\nASCII Reaction Profile (Energy vs Node)")
    console.print(plot, markup=False)


def _write_chain_with_nan_fallback(chain: Chain, fp: Path) -> None:
    """Write chain xyz plus energy/gradient sidecars; missing data is serialized as NaN."""
    fp = Path(fp)
    xyz_arr = chain.coordinates * BOHR_TO_ANGSTROMS
    write_xyz(filename=fp, atoms=chain.symbols, coords=xyz_arr)

    n_nodes = len(chain.nodes)
    n_atoms = chain.coordinates.shape[1] if n_nodes > 0 else 0

    ene_path = fp.parent / Path(f"{fp.stem}.energies")
    grad_path = fp.parent / Path(f"{fp.stem}.gradients")
    grad_shape_path = fp.parent / Path(f"{fp.stem}_grad_shapes.txt")

    energies = []
    missing_energies = False
    for node in chain.nodes:
        try:
            energies.append(float(node.energy))
        except Exception:
            missing_energies = True
            break
    if missing_energies:
        np.savetxt(ene_path, np.full(n_nodes, np.nan))
    else:
        np.savetxt(ene_path, np.array(energies))

    gradients = []
    missing_gradients = False
    for node in chain.nodes:
        try:
            gradients.append(np.array(node.gradient, dtype=float))
        except Exception:
            missing_gradients = True
            break
    if missing_gradients:
        np.savetxt(grad_path, np.full(n_nodes * n_atoms * 3, np.nan))
        np.savetxt(grad_shape_path, np.array([n_nodes, n_atoms, 3]))
    else:
        grad_arr = np.array(gradients, dtype=float)
        np.savetxt(grad_path, grad_arr.flatten())
        np.savetxt(grad_shape_path, np.array(grad_arr.shape))


def _write_chain_history_with_nan_fallback(chain_trajectory: list[Chain], fp: Path) -> None:
    out_folder = fp.resolve().parent / f"{fp.stem}_history"
    if out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    for i, chain in enumerate(chain_trajectory):
        _write_chain_with_nan_fallback(chain, out_folder / f"traj_{i}.xyz")


def _write_neb_results_with_history(neb_result, fp: Path) -> bool:
    """Write final chain plus history for non-recursive NEB runs."""
    if hasattr(neb_result, "write_to_disk"):
        try:
            neb_result.write_to_disk(fp, write_history=True)
            return True
        except Exception as exc:
            console.print(
                f"[yellow]⚠ Could not write full NEB history via write_to_disk: {exc}. "
                "Falling back to NaN-safe writer.[/yellow]"
            )

    chain_for_profile = None
    if getattr(neb_result, "chain_trajectory", None):
        chain_for_profile = neb_result.chain_trajectory[-1]
    elif getattr(neb_result, "optimized", None) is not None:
        chain_for_profile = neb_result.optimized

    if chain_for_profile is None:
        return False

    _write_chain_with_nan_fallback(chain_for_profile, fp)
    if getattr(neb_result, "chain_trajectory", None):
        _write_chain_history_with_nan_fallback(neb_result.chain_trajectory, fp)
    return True


@dataclass
class _VisualizationData:
    chain: Chain
    chain_trajectory: list[Chain] | None = None
    tree_layers: list[dict] | None = None


def _neb_chains_for_visualization(neb_obj) -> list[Chain]:
    if getattr(neb_obj, "chain_trajectory", None):
        return list(neb_obj.chain_trajectory)
    if getattr(neb_obj, "optimized", None) is not None:
        return [neb_obj.optimized]
    return []


def _collect_tree_layers_for_visualization(tree: TreeNode) -> list[dict]:
    layers: list[dict] = []
    by_depth: dict[int, list[dict]] = {}
    stack: list[tuple[TreeNode, int, int | None]] = [(tree, 0, None)]
    while stack:
        tree_node, depth, parent_index = stack.pop()
        if getattr(tree_node, "data", None):
            chains = _neb_chains_for_visualization(tree_node.data)
            if chains:
                by_depth.setdefault(depth, []).append(
                    {
                        "label": f"Node {tree_node.index}",
                        "node_index": int(tree_node.index),
                        "parent_index": parent_index,
                        "chains": chains,
                    }
                )
        for child in reversed(tree_node.children):
            stack.append((child, depth + 1, int(tree_node.index)))

    for depth in sorted(by_depth):
        layers.append({"depth": depth, "groups": by_depth[depth]})
    return layers


def _load_visualization_data(
    result_path: Path,
    charge: int = 0,
    multiplicity: int = 1,
) -> _VisualizationData:
    """Load visualization payload from NEB file, TreeNode folder, or plain chain xyz."""
    result_path = Path(result_path)
    if not result_path.exists():
        raise FileNotFoundError(f"Path does not exist: {result_path}")

    if result_path.is_dir():
        adj_matrix_fp = result_path / "adj_matrix.txt"
        if adj_matrix_fp.exists():
            tree = TreeNode.read_from_disk(
                folder_name=result_path, charge=charge, multiplicity=multiplicity
            )
            return _VisualizationData(
                chain=tree.output_chain,
                chain_trajectory=None,
                tree_layers=_collect_tree_layers_for_visualization(tree),
            )
        raise ValueError(
            "Directory input must be a TreeNode folder containing adj_matrix.txt."
        )

    history_folder = result_path.parent / f"{result_path.stem}_history"
    if history_folder.exists():
        neb = NEB.read_from_disk(
            fp=result_path,
            history_folder=history_folder,
            charge=charge,
            multiplicity=multiplicity,
        )
        if neb.chain_trajectory:
            return _VisualizationData(
                chain=neb.chain_trajectory[-1],
                chain_trajectory=list(neb.chain_trajectory),
            )
        if getattr(neb, "optimized", None) is not None:
            return _VisualizationData(chain=neb.optimized, chain_trajectory=None)
        raise ValueError("Loaded NEB object has no optimized chain or history.")

    try:
        chain = Chain.from_xyz(
            result_path, parameters=ChainInputs(), charge=charge, spinmult=multiplicity
        )
        return _VisualizationData(chain=chain, chain_trajectory=None)
    except Exception as exc:
        raise ValueError(
            "Could not detect serialized result type. For NEB provide the .xyz output "
            "that has a sibling '<stem>_history/' folder; for recursive MSMEP provide "
            "the TreeNode directory with adj_matrix.txt; or provide a valid chain xyz."
        ) from exc


def _load_chain_for_visualization(result_path: Path, charge: int = 0, multiplicity: int = 1) -> Chain:
    """Backward-compatible wrapper returning only the chain."""
    return _load_visualization_data(
        result_path=result_path, charge=charge, multiplicity=multiplicity
    ).chain


def _generate_opt_history_plot_b64(
    chain_trajectory: list[Chain], selected_index: int
) -> str:
    if not chain_trajectory:
        return ""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        selected_index = max(0, min(selected_index, len(chain_trajectory) - 1))
        for i, chain in enumerate(chain_trajectory):
            x = chain.integrated_path_length
            y = chain.energies_kcalmol
            if i == selected_index:
                ax.plot(x, y, "o-", color="tab:blue", alpha=1.0, linewidth=2.0)
            else:
                ax.plot(x, y, "o-", color="gray", alpha=0.2, linewidth=1.0)
        ax.set_xlabel("Integrated path length")
        ax.set_ylabel("Energy (kcal/mol)")
        ax.set_title("Optimization History")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
        return image_base64
    except Exception:
        return ""


def _build_chain_visualizer_html(
    chain: Chain,
    chain_trajectory: list[Chain] | None = None,
    tree_layers: list[dict] | None = None,
) -> str:
    def _chain_plot_payload(chain_obj: Chain) -> dict[str, list[float]]:
        try:
            x_vals = [float(v) for v in chain_obj.integrated_path_length]
            y_vals = [float(v) for v in chain_obj.energies_kcalmol]
        except Exception:
            x_vals = []
            y_vals = []
        return {"x": x_vals, "y": y_vals}

    def _serialize_chains(chains: list[Chain]) -> dict:
        chain_payload = []
        for chain_ind, chain_obj in enumerate(chains):
            frames = []
            plot_payload = _chain_plot_payload(chain_obj)
            for node in chain_obj.nodes:
                frames.append(
                    {
                        "xyz_b64": base64.b64encode(
                            node.structure.to_xyz().encode("utf-8")
                        ).decode("ascii"),
                    }
                )
            chain_payload.append(
                {
                    "index": chain_ind,
                    "frames": frames,
                    "plot": plot_payload,
                }
            )
        default_chain_index = max(len(chains) - 1, 0)
        return {
            "chains": chain_payload,
            "default_chain_index": default_chain_index,
        }

    if tree_layers:
        layers_payload = []
        for layer in tree_layers:
            groups_payload = []
            for group in layer["groups"]:
                groups_payload.append(
                    {
                        "label": group["label"],
                        "node_index": int(group["node_index"]),
                        "parent_index": (
                            int(group["parent_index"])
                            if group["parent_index"] is not None
                            else None
                        ),
                        "viz": _serialize_chains(group["chains"]),
                    }
                )
            layers_payload.append({"depth": layer["depth"], "groups": groups_payload})
        default_layer_index = max(len(layers_payload) - 1, 0)
        default_group_index = 0
        has_tree_layers = True
    else:
        chains_to_visualize = list(chain_trajectory) if chain_trajectory else [chain]
        layers_payload = [
            {
                "depth": 0,
                "groups": [
                    {
                        "label": "NEB",
                        "node_index": 0,
                        "parent_index": None,
                        "viz": _serialize_chains(chains_to_visualize),
                    }
                ],
            }
        ]
        default_layer_index = 0
        default_group_index = 0
        has_tree_layers = False

    payload = json.dumps(layers_payload)
    tree_panel_style = "" if has_tree_layers else "display:none;"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NEB Dynamics Visualizer</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 16px; }}
    .row {{ display: flex; gap: 16px; align-items: flex-start; }}
    .panel {{ flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    .controls {{ margin-bottom: 12px; }}
    img {{ width: 100%; max-width: 700px; }}
  </style>
</head>
<body>
  <h2>NEB Dynamics Interactive Viewer</h2>
  <div class="controls">
    <div id="treePanel" style="{tree_panel_style} margin-bottom: 12px;">
      <div style="font-weight: 600; margin-bottom: 6px;">Optimization Tree</div>
      <svg id="treeSvg" width="100%" height="220" viewBox="0 0 900 220" style="border: 1px solid #e6e6e6; border-radius: 8px; background: #fafafa;"></svg>
      <div style="font-size: 12px; color: #666; margin-top: 6px;">Click a node to load that NEB object's data.</div>
    </div>
    <label for="chainSelect">Chain: </label>
    <select id="chainSelect"></select>
    <br/>
    <label for="frameSlider">Frame: <span id="frameLabel">0</span></label><br/>
    <input id="frameSlider" type="range" min="0" max="0" value="0" step="1" style="width: min(720px, 90vw);" />
  </div>
  <div class="row">
    <div class="panel">
      <h3>Structure</h3>
      <iframe id="structureFrame" style="width: 100%; height: 520px; border: 0;" title="Structure viewer"></iframe>
    </div>
    <div class="panel">
      <h3>Energy Profile (Selected Chain)</h3>
      <div id="energyPlot" style="width: 100%; max-width: 700px;"></div>
      <div id="historyPanel" style="margin-top: 18px; display:none;">
        <h3>Optimization History</h3>
        <div id="historyPlot" style="width: 100%; max-width: 700px;"></div>
      </div>
    </div>
  </div>
  <script>
    const layers = {payload};
    function decodeB64UTF8(b64) {{
      return decodeURIComponent(escape(window.atob(b64)));
    }}
    function clamp(val, low, high) {{
      return Math.max(low, Math.min(high, val));
    }}
    function makeStructureSrcdoc(xyzB64) {{
      return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>
    html, body, #viewer {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: white;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    #status {{
      padding: 12px;
      color: #555;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div id="status">Loading 3D viewer...</div>
  <script>
    const xyz = decodeURIComponent(escape(atob("__XYZ_B64__")));
    function boot() {{
      const status = document.getElementById("status");
      const host = document.getElementById("viewer");
      status.remove();
      const viewer = $3Dmol.createViewer(host, {{ backgroundColor: "white" }});
      viewer.addModel(xyz, "xyz");
      viewer.setStyle({{}}, {{ stick: {{}}, sphere: {{ scale: 0.3 }} }});
      viewer.zoomTo();
      viewer.render();
    }}
    if (window.$3Dmol) {{
      boot();
    }} else {{
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/3dmol@2.5.3/build/3Dmol-min.js";
      script.onload = boot;
      script.onerror = () => {{
        const status = document.getElementById("status");
        status.textContent = "Failed to load 3Dmol.js";
      }};
      document.head.appendChild(script);
    }}
  <\\/script>
</body>
</html>`.replace("__XYZ_B64__", xyzB64);
    }}
    function renderPlot(containerId, traces, selectedTraceIndex = null, selectedPointIndex = null, title = "") {{
      const container = document.getElementById(containerId);
      if (!container) return;
      if (!traces || !traces.length) {{
        container.innerHTML = "<div style=\\"color:#666;font-size:13px;\\">No plot data available.</div>";
        return;
      }}

      const width = 700;
      const height = 420;
      const margin = {{ top: 28, right: 20, bottom: 44, left: 58 }};
      const xs = traces.flatMap((trace) => trace.x || []);
      const ys = traces.flatMap((trace) => trace.y || []);
      if (!xs.length || !ys.length) {{
        container.innerHTML = "<div style=\\"color:#666;font-size:13px;\\">No plot data available.</div>";
        return;
      }}

      let minX = Math.min(...xs);
      let maxX = Math.max(...xs);
      let minY = Math.min(...ys);
      let maxY = Math.max(...ys);
      if (minX === maxX) maxX = minX + 1;
      if (minY === maxY) maxY = minY + 1;
      const yPad = (maxY - minY) * 0.08;
      minY -= yPad;
      maxY += yPad;

      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const sx = (x) => margin.left + ((x - minX) / (maxX - minX)) * plotW;
      const sy = (y) => margin.top + (1 - (y - minY) / (maxY - minY)) * plotH;
      const polyline = (xVals, yVals) => xVals.map((x, i) => `${{sx(x)}},${{sy(yVals[i])}}`).join(" ");

      const ticks = 5;
      const gridLines = [];
      for (let i = 0; i <= ticks; i++) {{
        const t = i / ticks;
        const x = margin.left + t * plotW;
        const y = margin.top + t * plotH;
        const xv = minX + t * (maxX - minX);
        const yv = maxY - t * (maxY - minY);
        gridLines.push(`<line x1="${{x}}" y1="${{margin.top}}" x2="${{x}}" y2="${{height - margin.bottom}}" stroke="#eee" />`);
        gridLines.push(`<line x1="${{margin.left}}" y1="${{y}}" x2="${{width - margin.right}}" y2="${{y}}" stroke="#eee" />`);
        gridLines.push(`<text x="${{x}}" y="${{height - margin.bottom + 18}}" text-anchor="middle" font-size="11" fill="#666">${{xv.toFixed(2)}}</text>`);
        gridLines.push(`<text x="${{margin.left - 8}}" y="${{y + 4}}" text-anchor="end" font-size="11" fill="#666">${{yv.toFixed(2)}}</text>`);
      }}

      const series = traces.map((trace, idx) => {{
        const active = selectedTraceIndex === null ? true : idx === selectedTraceIndex;
        const lineColor = active ? "#18834a" : "#b7b7b7";
        const lineWidth = active ? 2.5 : 1.5;
        const opacity = active ? 1.0 : 0.45;
        const points = (trace.x || []).map((x, pointIdx) => {{
          const y = trace.y[pointIdx];
          const selected = idx === selectedTraceIndex && pointIdx === selectedPointIndex;
          const r = selected ? 7 : 4;
          const fill = selected ? "#f59e0b" : (active ? "#18834a" : "#9ca3af");
          const stroke = selected ? "#7a4b00" : "none";
          return `<circle cx="${{sx(x)}}" cy="${{sy(y)}}" r="${{r}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="1.5" />`;
        }}).join("");
        return `
          <polyline fill="none" stroke="${{lineColor}}" stroke-width="${{lineWidth}}" opacity="${{opacity}}" points="${{polyline(trace.x || [], trace.y || [])}}" />
          ${{points}}
        `;
      }}).join("");

      container.innerHTML = `
        <svg viewBox="0 0 ${{width}} ${{height}}" width="100%" role="img" aria-label="${{title || "Plot"}}">
          <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="white" rx="8" />
          ${{gridLines.join("")}}
          <line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="#444" />
          <line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="#444" />
          <text x="${{width / 2}}" y="18" text-anchor="middle" font-size="15" fill="#222">${{title}}</text>
          <text x="${{width / 2}}" y="${{height - 8}}" text-anchor="middle" font-size="12" fill="#444">Integrated path length</text>
          <text x="16" y="${{height / 2}}" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 16 ${{height / 2}})">Energy (kcal/mol)</text>
          ${{series}}
        </svg>
      `;
    }}
    let currentLayer = {default_layer_index};
    let currentGroup = {default_group_index};
    let currentChain = 0;
    function getCurrentLayer() {{
      return layers[currentLayer] || null;
    }}
    function getCurrentGroups() {{
      const layer = getCurrentLayer();
      return layer ? (layer.groups || []) : [];
    }}
    function getCurrentViz() {{
      const groups = getCurrentGroups();
      const group = groups[currentGroup];
      return group ? group.viz : null;
    }}
    function getCurrentFrames() {{
      const viz = getCurrentViz();
      if (!viz) return [];
      const chain = viz.chains[currentChain];
      return chain ? chain.frames : [];
    }}
    const treeNodeMap = {{}};
    function renderTreeGraph() {{
      const svg = document.getElementById("treeSvg");
      if (!svg || layers.length <= 1) return;
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      const width = 900;
      const height = 220;
      const topPad = 30;
      const bottomPad = 30;
      const leftPad = 40;
      const rightPad = 40;

      for (let layerIdx = 0; layerIdx < layers.length; layerIdx++) {{
        const layer = layers[layerIdx];
        const groups = layer.groups || [];
        const y = layers.length === 1
          ? height / 2
          : topPad + (layerIdx * (height - topPad - bottomPad) / (layers.length - 1));
        for (let groupIdx = 0; groupIdx < groups.length; groupIdx++) {{
          const g = groups[groupIdx];
          const x = groups.length === 1
            ? width / 2
            : leftPad + (groupIdx * (width - leftPad - rightPad) / (groups.length - 1));
          treeNodeMap[g.node_index] = {{
            layerIdx: layerIdx,
            groupIdx: groupIdx,
            x: x,
            y: y,
            parent: g.parent_index,
            label: g.label
          }};
        }}
      }}

      Object.values(treeNodeMap).forEach((node) => {{
        if (node.parent == null || !(node.parent in treeNodeMap)) return;
        const parent = treeNodeMap[node.parent];
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", String(parent.x));
        line.setAttribute("y1", String(parent.y));
        line.setAttribute("x2", String(node.x));
        line.setAttribute("y2", String(node.y));
        line.setAttribute("stroke", "#b7b7b7");
        line.setAttribute("stroke-width", "1.5");
        svg.appendChild(line);
      }});

      Object.entries(treeNodeMap).forEach(([nodeIndex, node]) => {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("data-node-index", nodeIndex);
        group.style.cursor = "pointer";

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", String(node.x));
        circle.setAttribute("cy", String(node.y));
        circle.setAttribute("r", "12");
        circle.setAttribute("fill", "#1f77b4");
        circle.setAttribute("stroke", "#0f4872");
        circle.setAttribute("stroke-width", "1");
        group.appendChild(circle);

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", String(node.x + 16));
        text.setAttribute("y", String(node.y + 4));
        text.setAttribute("font-size", "12");
        text.setAttribute("fill", "#222");
        text.textContent = node.label;
        group.appendChild(text);

        group.addEventListener("click", () => {{
          currentLayer = node.layerIdx;
          currentGroup = node.groupIdx;
          syncChainOptions();
          syncSlider(0);
          renderHistory();
          renderFrame(parseInt(slider.value, 10));
          updateTreeSelection();
        }});

        svg.appendChild(group);
      }});
      updateTreeSelection();
    }}
    function updateTreeSelection() {{
      const svg = document.getElementById("treeSvg");
      if (!svg || layers.length <= 1) return;
      const groups = getCurrentGroups();
      const selectedGroup = groups[currentGroup];
      const selectedNode = selectedGroup ? selectedGroup.node_index : null;
      svg.querySelectorAll("g[data-node-index]").forEach((elem) => {{
        const c = elem.querySelector("circle");
        if (!c) return;
        if (String(selectedNode) === elem.getAttribute("data-node-index")) {{
          c.setAttribute("fill", "#f59e0b");
          c.setAttribute("stroke", "#7a4b00");
          c.setAttribute("stroke-width", "2");
        }} else {{
          c.setAttribute("fill", "#1f77b4");
          c.setAttribute("stroke", "#0f4872");
          c.setAttribute("stroke-width", "1");
        }}
      }});
    }}
    function syncChainOptions() {{
      const chainSelect = document.getElementById("chainSelect");
      const viz = getCurrentViz();
      if (!viz || !viz.chains.length) {{
        chainSelect.innerHTML = "";
        chainSelect.disabled = true;
        currentChain = 0;
        return;
      }}
      const options = viz.chains
        .map((_, i) => `<option value="${{i}}">Chain ${{i}}</option>`)
        .join("");
      chainSelect.innerHTML = options;
      currentChain = clamp(viz.default_chain_index || 0, 0, viz.chains.length - 1);
      chainSelect.value = String(currentChain);
      chainSelect.disabled = viz.chains.length <= 1;
    }}
    function renderHistory() {{
      const panel = document.getElementById("historyPanel");
      const viz = getCurrentViz();
      if (!viz || !viz.chains || viz.chains.length <= 1) {{
        panel.style.display = "none";
        return;
      }}
      panel.style.display = "block";
      renderPlot(
        "historyPlot",
        viz.chains.map((chain, i) => ({{
          x: chain.plot ? chain.plot.x : [],
          y: chain.plot ? chain.plot.y : [],
          label: `Chain ${{i}}`,
        }})),
        currentChain,
        null,
        "Optimization History"
      );
    }}
    function syncSlider(frameIndex = 0) {{
      const slider = document.getElementById("frameSlider");
      const frames = getCurrentFrames();
      const maxFrame = Math.max(frames.length - 1, 0);
      slider.max = String(maxFrame);
      slider.value = String(clamp(frameIndex, 0, maxFrame));
    }}
    function renderFrame(i) {{
      const frames = getCurrentFrames();
      const frame = frames[i];
      if (!frame) return;
      document.getElementById("frameLabel").textContent = String(i);
      const frameEl = document.getElementById("structureFrame");
      frameEl.srcdoc = makeStructureSrcdoc(frame.xyz_b64);
      const viz = getCurrentViz();
      const chain = viz && viz.chains ? viz.chains[currentChain] : null;
      renderPlot(
        "energyPlot",
        [{{
          x: chain && chain.plot ? chain.plot.x : [],
          y: chain && chain.plot ? chain.plot.y : [],
        }}],
        0,
        i,
        "Energy Profile"
      );
    }}
    const slider = document.getElementById("frameSlider");
    const chainSelect = document.getElementById("chainSelect");
    chainSelect.addEventListener("change", (e) => {{
      currentChain = parseInt(e.target.value, 10) || 0;
      syncSlider(0);
      renderHistory();
      renderFrame(parseInt(slider.value, 10));
      updateTreeSelection();
    }});
    slider.addEventListener("input", (e) => renderFrame(parseInt(e.target.value, 10)));
    renderTreeGraph();
    syncChainOptions();
    syncSlider(0);
    renderHistory();
    renderFrame(parseInt(slider.value, 10));
  </script>
</body>
</html>
"""


def _parse_visualize_atom_indices(
    qminds_fp: str | None = None,
    atom_indices: str | None = None,
) -> list[int] | None:
    if qminds_fp and atom_indices:
        raise ValueError("Provide either --qminds-fp or --atom-indices, not both.")
    if qminds_fp:
        values = []
        for raw in Path(qminds_fp).read_text().splitlines():
            token = raw.strip()
            if token:
                values.append(int(token))
        return sorted(set(values))
    if atom_indices:
        tokens = atom_indices.replace(",", " ").split()
        return sorted(set(int(v) for v in tokens))
    return None


def _subset_chain_for_visualization(chain: Chain, atom_indices: list[int]) -> Chain:
    if len(chain.nodes) == 0:
        return chain
    n_atoms = len(chain.nodes[0].structure.symbols)
    bad = [i for i in atom_indices if i < 0 or i >= n_atoms]
    if bad:
        raise ValueError(f"Atom indices out of bounds for n_atoms={n_atoms}: {bad}")

    new_nodes = []
    for node in chain.nodes:
        struct = node.structure
        new_struct = Structure(
            geometry=np.array(struct.geometry)[atom_indices],
            symbols=[struct.symbols[i] for i in atom_indices],
            charge=struct.charge,
            multiplicity=struct.multiplicity,
        )
        new_node = StructureNode(structure=new_struct)
        new_node.has_molecular_graph = False
        new_node.graph = None
        new_node._cached_energy = node._cached_energy
        if node._cached_gradient is not None:
            try:
                grad_arr = np.array(node._cached_gradient)
                if grad_arr.ndim == 2 and grad_arr.shape[0] >= max(atom_indices) + 1:
                    new_node._cached_gradient = grad_arr[atom_indices]
                else:
                    new_node._cached_gradient = node._cached_gradient
            except Exception:
                new_node._cached_gradient = node._cached_gradient
        new_nodes.append(new_node)
    return Chain.model_validate({"nodes": new_nodes, "parameters": chain.parameters})


def _subset_chain_trajectory_for_visualization(
    chain_trajectory: list[Chain], atom_indices: list[int]
) -> list[Chain]:
    return [
        _subset_chain_for_visualization(chain=chain, atom_indices=atom_indices)
        for chain in chain_trajectory
    ]


def _subset_tree_layers_for_visualization(
    tree_layers: list[dict], atom_indices: list[int]
) -> list[dict]:
    subset_layers = []
    for layer in tree_layers:
        subset_groups = []
        for group in layer["groups"]:
            subset_groups.append(
                {
                    "label": group["label"],
                    "node_index": group["node_index"],
                    "parent_index": group["parent_index"],
                    "chains": _subset_chain_trajectory_for_visualization(
                        group["chains"], atom_indices
                    ),
                }
            )
        subset_layers.append({"depth": layer["depth"], "groups": subset_groups})
    return subset_layers
def _compute_ts_node(engine, ts_guess: StructureNode, bigchem: bool = False):
    """Run TS optimization through the engine and normalize to (StructureNode|None, ProgramOutput|None)."""
    try:
        if bigchem and hasattr(engine, "_compute_ts_result"):
            raw_out = engine._compute_ts_result(node=ts_guess, use_bigchem=True)
            if getattr(raw_out, "success", False):
                return StructureNode(structure=raw_out.return_result), raw_out
            return None, raw_out

        if hasattr(engine, "compute_transition_state"):
            raw_out = engine.compute_transition_state(node=ts_guess)
            if isinstance(raw_out, StructureNode):
                return raw_out, None
            if isinstance(raw_out, ProgramOutput):
                if getattr(raw_out, "success", False):
                    return StructureNode(structure=raw_out.return_result), raw_out
                return None, raw_out
            if getattr(raw_out, "success", False) and getattr(raw_out, "return_result", None) is not None:
                return StructureNode(structure=raw_out.return_result), raw_out
            return None, raw_out

        if hasattr(engine, "_compute_ts_result"):
            raw_out = engine._compute_ts_result(node=ts_guess, use_bigchem=bigchem)
            if getattr(raw_out, "success", False):
                return StructureNode(structure=raw_out.return_result), raw_out
            return None, raw_out

        raise AttributeError("Engine does not implement transition-state optimization.")
    except Exception as exc:
        program_output = getattr(exc, "program_output", None)
        if program_output is not None:
            return None, program_output
        raise


def _extract_minima_nodes(history: TreeNode) -> list[StructureNode]:
    """Extract minima candidates from recursive cheap-history leaves."""
    minima: list[StructureNode] = []
    for leaf in history.ordered_leaves:
        if not leaf.data or not leaf.data.chain_trajectory:
            continue
        final_chain = leaf.data.chain_trajectory[-1]
        if len(final_chain.nodes) == 0:
            continue
        minima.append(final_chain[0].copy())
        minima.append(final_chain[-1].copy())
    return minima


def _extract_minima_nodes_from_chain(chain: Chain) -> list[StructureNode]:
    """Extract endpoints + strict local minima from a single optimized chain."""
    if len(chain.nodes) == 0:
        return []
    if len(chain.nodes) <= 2:
        return [node.copy() for node in chain.nodes]
    energies = chain.energies
    minima_inds = {0, len(chain.nodes) - 1}
    for i in range(1, len(chain.nodes) - 1):
        if energies[i] < energies[i - 1] and energies[i] < energies[i + 1]:
            minima_inds.add(i)
    return [chain.nodes[i].copy() for i in sorted(minima_inds)]


def _dedupe_minima_nodes(nodes: list[StructureNode], chain_inputs: ChainInputs) -> list[StructureNode]:
    """Drop duplicate minima by geometry/graph identity."""
    unique_nodes: list[StructureNode] = []
    for node in nodes:
        duplicate = False
        for existing in unique_nodes:
            if is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False,
            ):
                duplicate = True
                break
        if not duplicate:
            unique_nodes.append(node.copy())
    return unique_nodes


def _dedupe_minima_and_sources(
    minima: list[StructureNode],
    sources: list[StructureNode],
    chain_inputs: ChainInputs,
) -> tuple[list[StructureNode], list[StructureNode]]:
    """Deduplicate minima while preserving same-index source mapping."""
    if len(minima) != len(sources):
        raise ValueError("minima and sources must have equal lengths.")

    unique_minima: list[StructureNode] = []
    unique_sources: list[StructureNode] = []
    for node, source in zip(minima, sources):
        duplicate = False
        for existing in unique_minima:
            if is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False,
            ):
                duplicate = True
                break
        if not duplicate:
            unique_minima.append(node.copy())
            unique_sources.append(source.copy())
    return unique_minima, unique_sources


def _clear_node_cached_properties(node: StructureNode) -> StructureNode:
    clean = node.copy()
    clean._cached_result = None
    clean._cached_energy = None
    clean._cached_gradient = None
    return clean


def _clear_chain_cached_properties(chain: Chain, parameters: ChainInputs) -> Chain:
    """Return chain copy with cached energies/gradients removed from every node."""
    clean_nodes = [_clear_node_cached_properties(node) for node in chain.nodes]
    return Chain.model_validate({"nodes": clean_nodes, "parameters": parameters})


def _find_matching_node_index(
    target: StructureNode,
    chain: Chain,
    chain_inputs: ChainInputs,
) -> int | None:
    try:
        dists = [np.linalg.norm(node.coords - target.coords)
                 for node in chain.nodes]
        if len(dists) == 0:
            return None
        closest_idx = int(np.argmin(dists))
        if dists[closest_idx] < 1e-8:
            return closest_idx
    except Exception:
        dists = []

    candidates: list[tuple[int, float]] = []
    for i, node in enumerate(chain.nodes):
        if is_identical(
            target,
            node,
            fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
            kcal_mol_cutoff=chain_inputs.node_ene_thre,
            verbose=False,
        ):
            dist = np.linalg.norm(node.coords - target.coords)
            candidates.append((i, dist))

    if candidates:
        return min(candidates, key=lambda t: t[1])[0]

    if len(dists) > 0:
        return int(np.argmin(dists))
    return None


def _build_recycled_pair_chain(
    cheap_output_chain: Chain,
    cheap_start_ref: StructureNode,
    cheap_end_ref: StructureNode,
    expensive_start: StructureNode,
    expensive_end: StructureNode,
    cheap_chain_inputs: ChainInputs,
    expensive_chain_inputs: ChainInputs,
    expected_nimages: int,
) -> Chain | None:
    start_idx = _find_matching_node_index(
        cheap_start_ref, cheap_output_chain, cheap_chain_inputs
    )
    end_idx = _find_matching_node_index(
        cheap_end_ref, cheap_output_chain, cheap_chain_inputs
    )
    if start_idx is None or end_idx is None or start_idx == end_idx:
        return None

    if start_idx < end_idx:
        segment_nodes = [node.copy()
                         for node in cheap_output_chain.nodes[start_idx:end_idx + 1]]
    else:
        segment_nodes = [
            node.copy() for node in cheap_output_chain.nodes[end_idx:start_idx + 1]][::-1]

    if len(segment_nodes) != expected_nimages:
        return None

    segment_nodes[0] = expensive_start.copy()
    segment_nodes[-1] = expensive_end.copy()
    recycled = Chain.model_validate(
        {"nodes": segment_nodes, "parameters": expensive_chain_inputs}
    )
    return _clear_chain_cached_properties(recycled, expensive_chain_inputs)


def _reverse_chain(chain: Chain) -> Chain:
    return Chain.model_validate(
        {"nodes": [node.copy() for node in chain.nodes[::-1]],
         "parameters": chain.parameters}
    )


def _concat_chains(chains: list[Chain], parameters: ChainInputs) -> Chain:
    if len(chains) == 0:
        raise ValueError("Cannot concatenate an empty list of chains.")
    nodes = []
    for i, chain in enumerate(chains):
        chain_nodes = chain.nodes if i == 0 else chain.nodes[1:]
        nodes.extend([node.copy() for node in chain_nodes])
    return Chain.model_validate({"nodes": nodes, "parameters": parameters})


def _section_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"value": obj}


def _flatten_params(data, prefix=""):
    if isinstance(data, dict):
        rows = []
        for key in sorted(data.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_params(data[key], next_prefix))
        return rows
    return [(prefix if prefix else "value", _format_param_value(data))]


def _format_param_value(value, max_str: int = 160):
    if isinstance(value, list):
        if len(value) > 20:
            head = ", ".join(str(v) for v in value[:8])
            tail = ", ".join(str(v) for v in value[-3:])
            return f"[{head}, ..., {tail}] (n={len(value)})"
        return str(value)
    s = str(value)
    if len(s) > max_str:
        return s[: max_str - 3] + "..."
    return s


def _render_runinputs(program_input: RunInputs):
    table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
    table.add_column("Section", style="bold cyan", no_wrap=True)
    table.add_column("Key", style="magenta")
    table.add_column("Value", style="white")

    sections = [
        ("General", {
            "engine_name": program_input.engine_name,
            "program": program_input.program,
            "path_min_method": program_input.path_min_method,
        }),
        ("QMMM", _section_dict(program_input.qmmm_inputs) if getattr(program_input, "qmmm_inputs", None) else {}),
        ("Path Minimizer", _section_dict(program_input.path_min_inputs)),
        ("Chain", _section_dict(program_input.chain_inputs)),
        ("GI", _section_dict(program_input.gi_inputs)),
        ("Program Args", _section_dict(program_input.program_kwds)),
        ("Optimizer", _section_dict(program_input.optimizer_kwds)),
    ]

    for section_name, section_data in sections:
        flat_rows = _flatten_params(section_data)
        for i, (key, value) in enumerate(flat_rows):
            table.add_row(section_name if i == 0 else "", key, value)

    console.print(
        Panel(
            table,
            title="[bold cyan]Input Parameters[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def _load_endpoint_structure(
    path: str,
    charge: int,
    multiplicity: int,
    rst7_prmtop_text: str | None = None,
) -> Structure:
    fp = Path(path)
    is_rst7 = fp.suffix.lower() == ".rst7"
    if is_rst7:
        if rst7_prmtop_text is None:
            raise ValueError(
                "RST7 endpoint conversion requires --rst7-prmtop."
            )
        coords, _ = rst7_to_coords_and_indices(fp.read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            symbols = parse_symbols_from_prmtop(rst7_prmtop_text)
        return Structure(
            geometry=coords * ANGSTROM_TO_BOHR,
            charge=charge,
            multiplicity=multiplicity,
            symbols=symbols,
        )
    return Structure.open(path)


@app.command("run")
def run(
        start: Annotated[str, typer.Option(
            help='path to start file, or a reactant smiles')] = None,
        end: Annotated[str, typer.Option(
            help='path to end file, or a product smiles')] = None,
        geometries:  Annotated[str, typer.Argument(help='file containing an approximate path between two endpoints. \
            Use this if you have precompted a path you want to use. Do not use this with smiles.')] = None,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to RunInputs toml file')] = None,
        use_smiles: bool = False,
        use_tsopt: Annotated[bool, typer.Option(
            help='whether to run a transition state optimization on each TS guess')] = False,
        minimize_ends: bool = False,
        recursive: bool = False,
        name: str = None,
        charge: int = 0,
        multiplicity: int = 1,
        rst7_prmtop: Annotated[str, typer.Option(
            "--rst7-prmtop",
            help="Path to AMBER prmtop used to map atomic symbols when converting rst7 endpoints.",
        )] = None,
        create_irc: Annotated[bool, typer.Option(
            help='whether to run and output an IRC chain. Need to set --use_tsopt also, otherwise\
                will attempt use the guess structure.')] = False,
        use_bigchem: Annotated[bool, typer.Option(
            help='whether to use chemcloud to compute hessian for ts opt and irc jobs')] = False):

    # Print header
    console.print(BANNER)

    table = Table(box=None, show_header=False)
    table.add_column(style="dim")
    table.add_row("[bold cyan]Command:[/bold cyan]", "[white]run[/white]")
    table.add_row("[bold cyan]Method:[/bold cyan]",
                  f"[yellow]{'recursive' if recursive else 'regular'}[/yellow]")
    table.add_row("[bold cyan]SMILES mode:[/bold cyan]",
                  f"[yellow]{use_smiles}[/yellow]")
    console.print(table)
    console.print()

    start_time = time.time()
    # load the structures
    if use_smiles:
        from neb_dynamics.nodes.nodehelpers import create_pairs_from_smiles
        from neb_dynamics.arbalign import align_structures

        console.print(
            "[yellow]⚠ WARNING:[/yellow] Using RXNMapper to create atomic mapping. Carefully check output to see how labels affected reaction path.")
        with console.status("[bold cyan]Creating structures from SMILES...[/bold cyan]") as status:
            start_structure, end_structure = create_pairs_from_smiles(
                smi1=start, smi2=end)

            console.print(
                "[cyan]Using arbalign to optimize index labelling for endpoints[/cyan]")
            end_structure = align_structures(
                start_structure, end_structure, distance_metric='RMSD')

        all_structs = [start_structure, end_structure]
    else:

        if geometries is not None:
            with console.status(f"[bold cyan]Loading geometries from {geometries}...[/bold cyan]"):
                try:
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=charge, spinmult=multiplicity)
                except ValueError:  # qcio does not allow an input for charge if file has a charge in it
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=None, spinmult=None)
        elif start is not None and end is not None:
            with console.status(f"[bold cyan]Loading structures...[/bold cyan]"):
                console.print(
                    f"[dim]Charge: {charge}, Multiplicity: {multiplicity}[/dim]")
                endpoint_paths = [Path(start), Path(end)]
                needs_rst7_prmtop = any(
                    fp.suffix.lower() == ".rst7" for fp in endpoint_paths
                )
                if needs_rst7_prmtop and rst7_prmtop is None:
                    console.print(
                        "[bold red]✗ ERROR:[/bold red] .rst7 endpoints require --rst7-prmtop."
                    )
                    raise typer.Exit(1)
                prmtop_text = Path(rst7_prmtop).read_text() if needs_rst7_prmtop else None
                start_ref = _load_endpoint_structure(
                    start,
                    charge=charge,
                    multiplicity=multiplicity,
                    rst7_prmtop_text=prmtop_text,
                )
                end_ref = _load_endpoint_structure(
                    end,
                    charge=charge,
                    multiplicity=multiplicity,
                    rst7_prmtop_text=prmtop_text,
                )

                if start_ref.charge != charge or start_ref.multiplicity != multiplicity:
                    console.print(
                        f"[yellow]⚠ WARNING:[/yellow] {start} has charge {start_ref.charge} and multiplicity {start_ref.multiplicity}. Using {charge} and {multiplicity} instead."
                    )
                    start_ref = Structure(geometry=start_ref.geometry,
                                          charge=charge,
                                          multiplicity=multiplicity,
                                          symbols=start_ref.symbols)
                if end_ref.charge != charge or end_ref.multiplicity != multiplicity:
                    console.print(
                        f"[yellow]⚠ WARNING:[/yellow] {end} has charge {end_ref.charge} and multiplicity {end_ref.multiplicity}. Using {charge} and {multiplicity} instead."
                    )
                    end_ref = Structure(geometry=end_ref.geometry,
                                        charge=charge,
                                        multiplicity=multiplicity,
                                        symbols=end_ref.symbols)

                all_structs = [start_ref, end_ref]
        else:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Either 'geometries' or 'start' and 'end' flags must be populated!")
            raise typer.Exit(1)

    # load the RunInputs
    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        if inputs is not None:
            program_input = RunInputs.open(inputs)
        else:
            program_input = RunInputs(program='xtb', engine_name='qcop')

    _render_runinputs(program_input)
    sys.stdout.flush()

    # minimize endpoints if requested
    all_nodes = [StructureNode(structure=s) for s in all_structs]
    if minimize_ends:
        console.print("[bold cyan]⟳ Minimizing input endpoints...[/bold cyan]")
        batch_optimizer = getattr(program_input.engine, "compute_geometry_optimizations", None)
        if callable(batch_optimizer):
            console.print("[dim]Submitting batched endpoint geometry optimizations...[/dim]")
            try:
                trajectories = batch_optimizer(
                    [all_nodes[0], all_nodes[-1]],
                    keywords={'coordsys': 'cart', 'maxiter': 500},
                )
                all_nodes[0] = trajectories[0][-1]
                all_nodes[-1] = trajectories[-1][-1]
            except TypeError:
                trajectories = batch_optimizer([all_nodes[0], all_nodes[-1]])
                all_nodes[0] = trajectories[0][-1]
                all_nodes[-1] = trajectories[-1][-1]
        else:
            console.print("[dim]Minimizing start endpoint...[/dim]")
            start_tr = program_input.engine.compute_geometry_optimization(
                all_nodes[0], keywords={'coordsys': 'cart', 'maxiter': 500})
            all_nodes[0] = start_tr[-1]
            console.print("[dim]Minimizing end endpoint...[/dim]")
            end_tr = program_input.engine.compute_geometry_optimization(
                all_nodes[-1], keywords={'coordsys': 'cart', 'maxiter': 500})
            all_nodes[-1] = end_tr[-1]
        console.print("[bold green]✓ Done![/bold green]")

    # create Chain
    console.print(f"[dim]Loading {len(all_nodes)} nodes...[/dim]")
    chain = Chain.model_validate({
        "nodes": all_nodes,
        "parameters": program_input.chain_inputs}
    )

    # create MSMEP object
    m = MSMEP(inputs=program_input)

    # Run the optimization
    chain_for_profile = None

    if recursive:
        if not program_input.path_min_inputs.do_elem_step_checks:
            console.print(
                "[yellow]⚠ WARNING: do_elem_step_checks is set to False. This may cause issues with recursive splitting.[/yellow]")
            console.print(
                "[yellow]Making it True to ensure proper functioning of recursive splitting.[/yellow]")
            program_input.path_min_inputs.do_elem_step_checks = True
        console.print(
            f"[bold magenta]▶ RUNNING AUTOSPLITTING {program_input.path_min_method}[/bold magenta]")
        history = m.run_recursive_minimize(chain)

        if not history.data:
            console.print("[bold red]✗ ERROR:[/bold red] Program did not run. Likely because your endpoints are conformers of the same molecular graph. Tighten the node_rms_thre and/or node_ene_thre parameters in chain_inputs and try again.")
            raise typer.Exit(1)

        leaves_nebs = [
            obj for obj in history.get_optimization_history() if obj]
        fp = Path("mep_output")
        if name is not None:
            name = Path(name)
            data_dir = Path(name).resolve().parent
            foldername = data_dir / name.stem
            filename = data_dir / (name.stem + ".xyz")

        else:
            data_dir = Path(os.getcwd())
            foldername = data_dir / f"{fp.stem}_msmep"
            filename = data_dir / f"{fp.stem}_msmep.xyz"

        end_time = time.time()
        history.output_chain.write_to_disk(filename)
        history.write_to_disk(foldername)
        chain_for_profile = history.output_chain

        if use_tsopt:
            for i, leaf in enumerate(history.ordered_leaves):
                if not leaf.data:
                    continue
                if not leaf.data.chain_trajectory:
                    console.print(
                        f"[yellow]⚠ Skipping TS optimization on leaf {i}: no chain trajectory.[/yellow]"
                    )
                    continue
                console.print(
                    f"[bold cyan]⟳ Running TS opt on leaf {i}...[/bold cyan]")
                try:
                    ts_node, tsres = _compute_ts_node(
                        engine=program_input.engine,
                        ts_guess=leaf.data.chain_trajectory[-1].get_ts_node(),
                    )
                except Exception:
                    console.print(
                        f"[yellow]⚠ TS optimization crashed on leaf {i}: {traceback.format_exc()}[/yellow]"
                    )
                    continue

                if tsres is not None and hasattr(tsres, "save"):
                    tsres.save(data_dir / (filename.stem+f"_tsres_{i}.qcio"))
                if ts_node is not None:
                    ts_node.structure.save(data_dir / (filename.stem+f"_ts_{i}.xyz"))
                    if create_irc:
                        try:
                            irc = compute_irc_chain(
                                ts_node=ts_node,
                                engine=program_input.engine,
                            )
                            irc.write_to_disk(
                                filename.stem+f"_tsres_{i}_IRC.xyz")

                        except Exception:
                            console.print(
                                f"[yellow]⚠ IRC failed: {traceback.format_exc()}[/yellow]")
                            console.print(
                                "[yellow]IRC failed. Continuing...[/yellow]")
                else:
                    console.print(
                        f"[yellow]⚠ TS optimization did not converge on leaf {i}...[/yellow]")

        tot_grad_calls = sum([obj.grad_calls_made for obj in leaves_nebs])
        geom_grad_calls = sum(
            [obj.geom_grad_calls_made for obj in leaves_nebs])
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {tot_grad_calls} gradient calls total.[/cyan]")
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {geom_grad_calls} gradient for geometry optimizations.[/cyan]")

    else:
        console.print(
            f"[bold magenta]▶ RUNNING REGULAR {program_input.path_min_method}[/bold magenta]")
        n, elem_step_results = m.run_minimize_chain(input_chain=chain)
        fp = Path("mep_output")
        data_dir = Path(os.getcwd())
        if name is not None:
            filename = data_dir / (name + ".xyz")

        else:
            filename = data_dir / f"{fp.stem}_neb.xyz"

        end_time = time.time()
        wrote_outputs = _write_neb_results_with_history(n, filename)
        if n.chain_trajectory:
            chain_for_profile = n.chain_trajectory[-1]
        elif n.optimized is not None:
            chain_for_profile = n.optimized

        if not wrote_outputs:
            console.print(
                "[yellow]⚠ Skipping output write/profile because path minimization did not produce an optimized chain.[/yellow]"
            )

        if use_tsopt and n.optimized is not None:
            console.print("[bold cyan]⟳ Running TS opt...[/bold cyan]")
            try:
                source_chain = n.chain_trajectory[-1] if n.chain_trajectory else n.optimized
                ts_node, tsres = _compute_ts_node(
                    engine=program_input.engine,
                    ts_guess=source_chain.get_ts_node(),
                )
            except Exception:
                console.print(
                    f"[yellow]⚠ TS optimization crashed: {traceback.format_exc()}[/yellow]"
                )
                ts_node, tsres = None, None
            if tsres is not None and hasattr(tsres, "save"):
                tsres.save(data_dir / (filename.stem+"_tsres.qcio"))
            if ts_node is not None:
                ts_node.structure.save(
                    data_dir / (filename.stem+"_ts.xyz"))

                if create_irc:
                    try:
                        irc = compute_irc_chain(
                            ts_node=ts_node, engine=program_input.engine
                        )
                        irc.write_to_disk(
                            filename.stem+"_tsres_IRC.xyz")

                    except Exception:
                        console.print(
                            f"[yellow]⚠ IRC failed: {traceback.format_exc()}[/yellow]")
                        console.print(
                            "[yellow]IRC failed. Continuing...[/yellow]")

            else:
                console.print("[yellow]⚠ TS optimization failed.[/yellow]")
        elif use_tsopt:
            console.print(
                "[yellow]⚠ Skipping TS optimization because path minimization did not converge.[/yellow]"
            )

        tot_grad_calls = n.grad_calls_made
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {tot_grad_calls} gradient calls total.[/cyan]")

    end_time = time.time()
    elapsed = end_time - start_time

    # Print summary panel
    summary = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    if elapsed > 60:
        summary.add_row(
            "⏱ Walltime:", f"[yellow]{elapsed/60:.1f} min[/yellow]")
    else:
        summary.add_row("⏱ Walltime:", f"[yellow]{elapsed:.1f} s[/yellow]")
    summary.add_row("📁 Output:", f"[cyan]{filename}[/cyan]")
    console.print(Panel(
        summary, title="[bold green]✓ Complete![/bold green]", border_style="green"))

    if chain_for_profile is not None:
        _ascii_profile_for_chain(chain_for_profile)


@app.command("run-refine")
def run_refine(
        start: Annotated[str, typer.Option(
            help='path to start file, or a reactant smiles')] = None,
        end: Annotated[str, typer.Option(
            help='path to end file, or a product smiles')] = None,
        geometries:  Annotated[str, typer.Argument(help='file containing an approximate path between two endpoints. \
            Use this if you have precompted a path you want to use. Do not use this with smiles.')] = None,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to expensive RunInputs toml file')] = None,
        cheap_inputs: Annotated[str, typer.Option("--cheap-inputs", "-ci",
                                                  help='optional path to cheaper RunInputs toml file for initial discovery')] = None,
        recycle_nodes: Annotated[bool, typer.Option(
            "--recycle-nodes",
            help="Reuse cheap-stage path nodes as initial guess for expensive pair refinement.",
        )] = False,
        use_smiles: bool = False,
        recursive: bool = False,
        minimize_ends: bool = False,
        name: str = None,
        charge: int = 0,
        multiplicity: int = 1,
        rst7_prmtop: Annotated[str, typer.Option(
            "--rst7-prmtop",
            help="Path to AMBER prmtop used to map atomic symbols when converting rst7 endpoints.",
        )] = None):
    """Two-stage refinement: cheap discovery -> expensive minima/path refinement."""
    console.print(BANNER)

    if inputs is None:
        console.print(
            "[bold red]✗ ERROR:[/bold red] --inputs/-i is required for run-refine."
        )
        raise typer.Exit(1)

    start_time = time.time()
    table = Table(box=None, show_header=False)
    table.add_column(style="dim")
    table.add_row("[bold cyan]Command:[/bold cyan]",
                  "[white]run-refine[/white]")
    table.add_row("[bold cyan]SMILES mode:[/bold cyan]",
                  f"[yellow]{use_smiles}[/yellow]")
    table.add_row("[bold cyan]Method:[/bold cyan]",
                  f"[yellow]{'recursive' if recursive else 'regular'}[/yellow]")
    table.add_row("[bold cyan]Cheap Inputs:[/bold cyan]",
                  f"[yellow]{cheap_inputs if cheap_inputs else inputs}[/yellow]")
    table.add_row("[bold cyan]Expensive Inputs:[/bold cyan]",
                  f"[yellow]{inputs}[/yellow]")
    table.add_row("[bold cyan]Recycle Nodes:[/bold cyan]",
                  f"[yellow]{recycle_nodes}[/yellow]")
    console.print(table)
    console.print()

    # load structures
    if use_smiles:
        from neb_dynamics.nodes.nodehelpers import create_pairs_from_smiles
        from neb_dynamics.arbalign import align_structures

        console.print(
            "[yellow]⚠ WARNING:[/yellow] Using RXNMapper to create atomic mapping. Carefully check output to see how labels affected reaction path.")
        with console.status("[bold cyan]Creating structures from SMILES...[/bold cyan]"):
            start_structure, end_structure = create_pairs_from_smiles(
                smi1=start, smi2=end)
            end_structure = align_structures(
                start_structure, end_structure, distance_metric='RMSD')
        all_structs = [start_structure, end_structure]
    else:
        if geometries is not None:
            with console.status(f"[bold cyan]Loading geometries from {geometries}...[/bold cyan]"):
                try:
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=charge, spinmult=multiplicity)
                except ValueError:
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=None, spinmult=None)
        elif start is not None and end is not None:
            with console.status(f"[bold cyan]Loading structures...[/bold cyan]"):
                endpoint_paths = [Path(start), Path(end)]
                needs_rst7_prmtop = any(
                    fp.suffix.lower() == ".rst7" for fp in endpoint_paths
                )
                if needs_rst7_prmtop and rst7_prmtop is None:
                    console.print(
                        "[bold red]✗ ERROR:[/bold red] .rst7 endpoints require --rst7-prmtop."
                    )
                    raise typer.Exit(1)
                prmtop_text = Path(rst7_prmtop).read_text() if needs_rst7_prmtop else None
                start_ref = _load_endpoint_structure(
                    start,
                    charge=charge,
                    multiplicity=multiplicity,
                    rst7_prmtop_text=prmtop_text,
                )
                end_ref = _load_endpoint_structure(
                    end,
                    charge=charge,
                    multiplicity=multiplicity,
                    rst7_prmtop_text=prmtop_text,
                )
                if start_ref.charge != charge or start_ref.multiplicity != multiplicity:
                    start_ref = Structure(
                        geometry=start_ref.geometry,
                        charge=charge,
                        multiplicity=multiplicity,
                        symbols=start_ref.symbols,
                    )
                if end_ref.charge != charge or end_ref.multiplicity != multiplicity:
                    end_ref = Structure(
                        geometry=end_ref.geometry,
                        charge=charge,
                        multiplicity=multiplicity,
                        symbols=end_ref.symbols,
                    )
                all_structs = [start_ref, end_ref]
        else:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Either 'geometries' or 'start' and 'end' flags must be populated!")
            raise typer.Exit(1)

    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        expensive_input = RunInputs.open(inputs)
        cheap_input = RunInputs.open(
            cheap_inputs) if cheap_inputs else RunInputs.open(inputs)

    console.print("[bold cyan]Cheap-level Inputs[/bold cyan]")
    _render_runinputs(cheap_input)
    console.print("[bold cyan]Expensive-level Inputs[/bold cyan]")
    _render_runinputs(expensive_input)

    all_nodes = [StructureNode(structure=s) for s in all_structs]
    if minimize_ends:
        console.print(
            "[bold cyan]⟳ Minimizing input endpoints at cheap level...[/bold cyan]")
        start_tr = cheap_input.engine.compute_geometry_optimization(
            all_nodes[0], keywords={'coordsys': 'cart', 'maxiter': 500})
        all_nodes[0] = start_tr[-1]
        end_tr = cheap_input.engine.compute_geometry_optimization(
            all_nodes[-1], keywords={'coordsys': 'cart', 'maxiter': 500})
        all_nodes[-1] = end_tr[-1]

    cheap_chain = Chain.model_validate(
        {"nodes": all_nodes, "parameters": cheap_input.chain_inputs}
    )
    cheap_msmep = MSMEP(inputs=cheap_input)

    console.print(
        f"[bold magenta]▶ CHEAP DISCOVERY RUN ({cheap_input.path_min_method})[/bold magenta]"
    )
    if recursive:
        if not cheap_input.path_min_inputs.do_elem_step_checks:
            console.print(
                "[yellow]⚠ WARNING: do_elem_step_checks is False with --recursive. Setting it to True.[/yellow]"
            )
            cheap_input.path_min_inputs.do_elem_step_checks = True
        cheap_history = cheap_msmep.run_recursive_minimize(cheap_chain)
        if not cheap_history.data:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Cheap run returned no valid history."
            )
            raise typer.Exit(1)
        cheap_output_chain = cheap_history.output_chain
        cheap_minima = _extract_minima_nodes(cheap_history)
    else:
        cheap_neb, _ = cheap_msmep.run_minimize_chain(cheap_chain)
        cheap_output_chain = cheap_neb.chain_trajectory[-1] if cheap_neb.chain_trajectory else cheap_neb.optimized
        if cheap_output_chain is None:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Cheap run produced no optimized chain."
            )
            raise typer.Exit(1)
        cheap_history = None
        cheap_minima = _extract_minima_nodes_from_chain(cheap_output_chain)

    base_name = name if name is not None else "mep_output"
    data_dir = Path(os.getcwd())
    cheap_chain_fp = data_dir / f"{base_name}_cheap.xyz"
    cheap_tree_dir = data_dir / f"{base_name}_cheap_msmep"
    cheap_output_chain.write_to_disk(cheap_chain_fp)
    if cheap_history is not None:
        cheap_history.write_to_disk(cheap_tree_dir)

    cheap_minima = _dedupe_minima_nodes(cheap_minima, cheap_input.chain_inputs)
    console.print(
        f"[cyan]Discovered {len(cheap_minima)} unique cheap minima (including endpoints).[/cyan]"
    )

    console.print(
        "[bold magenta]▶ REOPTIMIZING MINIMA AT EXPENSIVE LEVEL[/bold magenta]")
    refined_minima: list[StructureNode] = []
    refined_source_minima: list[StructureNode] = []
    dropped_count = 0
    kept_unoptimized_count = 0
    for i, node in enumerate(cheap_minima):
        optimized_successfully = False
        try:
            try:
                traj = expensive_input.engine.compute_geometry_optimization(
                    node, keywords={'coordsys': 'cart', 'maxiter': 500}
                )
            except TypeError:
                traj = expensive_input.engine.compute_geometry_optimization(
                    node)
            opt_node = traj[-1]
            optimized_successfully = True
        except Exception:
            console.print(
                f"[yellow]⚠ Failed to optimize minimum {i}; keeping cheap-level geometry for refinement.[/yellow]"
            )
            console.print(
                f"[yellow]Reason:[/yellow] {traceback.format_exc().strip()}"
            )
            opt_node = _clear_node_cached_properties(node)
            kept_unoptimized_count += 1

        if optimized_successfully and node.has_molecular_graph and opt_node.has_molecular_graph:
            same_connectivity = _is_connectivity_identical(
                node, opt_node, verbose=False
            )
            if not same_connectivity:
                dropped_count += 1
                continue
        refined_minima.append(opt_node)
        refined_source_minima.append(node.copy())

    refined_minima, refined_source_minima = _dedupe_minima_and_sources(
        refined_minima, refined_source_minima, expensive_input.chain_inputs
    )
    if len(refined_minima) < 2:
        console.print(
            "[bold red]✗ ERROR:[/bold red] Fewer than 2 minima remain after expensive-level refinement."
        )
        raise typer.Exit(1)

    refined_minima_chain = Chain.model_validate(
        {"nodes": refined_minima, "parameters": expensive_input.chain_inputs}
    )
    refined_minima_fp = data_dir / f"{base_name}_refined_minima.xyz"
    refined_minima_chain.write_to_disk(refined_minima_fp)
    console.print(
        f"[cyan]Retained {len(refined_minima)} minima, dropped {dropped_count} due to connectivity changes, kept {kept_unoptimized_count} without expensive optimization.[/cyan]"
    )

    console.print(
        "[bold magenta]▶ EXPENSIVE PAIRWISE PATH MINIMIZATION[/bold magenta]")
    pair_dir = data_dir / f"{base_name}_refined_pairs"
    pair_dir.mkdir(exist_ok=True)
    expensive_msmep = MSMEP(inputs=expensive_input)
    if recursive and not expensive_input.path_min_inputs.do_elem_step_checks:
        console.print(
            "[yellow]⚠ WARNING: expensive do_elem_step_checks is False with --recursive. Setting it to True.[/yellow]"
        )
        expensive_input.path_min_inputs.do_elem_step_checks = True

    pair_chains: list[Chain] = []
    pair_inds = list(zip(range(len(refined_minima) - 1),
                     range(1, len(refined_minima))))
    for i, j in pair_inds:
        endpoint_pair = Chain.model_validate(
            {"nodes": [refined_minima[i], refined_minima[j]],
                "parameters": expensive_input.chain_inputs}
        )
        pair = endpoint_pair
        if recycle_nodes:
            recycled_pair = _build_recycled_pair_chain(
                cheap_output_chain=cheap_output_chain,
                cheap_start_ref=refined_source_minima[i],
                cheap_end_ref=refined_source_minima[j],
                expensive_start=refined_minima[i],
                expensive_end=refined_minima[j],
                cheap_chain_inputs=cheap_input.chain_inputs,
                expensive_chain_inputs=expensive_input.chain_inputs,
                expected_nimages=expensive_input.gi_inputs.nimages,
            )
            if recycled_pair is not None:
                pair = recycled_pair
            else:
                console.print(
                    f"[yellow]⚠ Could not recycle cheap nodes for pair ({i}, {j}); using fresh interpolation.[/yellow]"
                )
        try:
            if recursive:
                pair_history = expensive_msmep.run_recursive_minimize(pair)
                if not pair_history.data:
                    console.print(
                        f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
                    )
                    continue
                out_chain = pair_history.output_chain
                pair_history.write_to_disk(pair_dir / f"pair_{i}_{j}_msmep")
            else:
                neb_obj, _ = expensive_msmep.run_minimize_chain(pair)
                out_chain = neb_obj.chain_trajectory[-1] if neb_obj.chain_trajectory else neb_obj.optimized
            if out_chain is None:
                console.print(
                    f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
                )
                continue
            pair_chains.append(out_chain)
            out_chain.write_to_disk(pair_dir / f"pair_{i}_{j}.xyz")
        except Exception:
            console.print(
                f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
            )
            continue

    if len(pair_chains) == 0:
        console.print(
            "[bold red]✗ ERROR:[/bold red] No expensive sequential pair paths converged."
        )
        raise typer.Exit(1)

    refined_chain = _concat_chains(
        pair_chains, expensive_input.chain_inputs)
    refined_chain_fp = data_dir / f"{base_name}_refined.xyz"
    refined_chain.write_to_disk(refined_chain_fp)

    elapsed = time.time() - start_time
    summary = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row(
        "⏱ Walltime:", f"[yellow]{elapsed/60:.1f} min[/yellow]" if elapsed > 60 else f"[yellow]{elapsed:.1f} s[/yellow]")
    summary.add_row("📁 Cheap chain:", f"[cyan]{cheap_chain_fp}[/cyan]")
    summary.add_row("📁 Refined minima:", f"[cyan]{refined_minima_fp}[/cyan]")
    summary.add_row("📁 Refined chain:", f"[cyan]{refined_chain_fp}[/cyan]")
    summary.add_row("🔗 Pair sequence:",
                    f"[white]{pair_inds}[/white]")
    console.print(Panel(
        summary, title="[bold green]✓ run-refine Complete![/bold green]", border_style="green"))

    _ascii_profile_for_chain(refined_chain)


@app.command("ts")
def ts(
    geometry: Annotated[str, typer.Argument(help='path to geometry file to optimize')],
    inputs: Annotated[str, typer.Option("--inputs", "-i",
                                        help='path to RunInputs toml file')] = None,
    name: str = None,
    charge: int = 0,
    multiplicity: int = 1,
    bigchem: bool = False
):
    console.print(BANNER)

    # create output names
    fp = Path(geometry)
    data_dir = Path(os.getcwd())

    if name is not None:
        results_name = data_dir / (name + ".qcio")
        filename = data_dir / (name + ".xyz")
    else:
        results_name = fp.stem + ".qcio"
        filename = fp.stem + ".xyz"

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')

    with console.status(f"[bold cyan]Optimizing transition state: {geometry}...[/bold cyan]") as status:
        sys.stdout.flush()
        try:
            struct = Structure.open(geometry)
            s_dict = struct.model_dump()
            s_dict["charge"], s_dict["multiplicity"] = charge, multiplicity
            struct = Structure(**s_dict)

            node = StructureNode(structure=struct)
            ts_node, output = _compute_ts_node(
                engine=program_input.engine,
                ts_guess=node,
                bigchem=bigchem,
            )

        except Exception:
            console.print(
                f"[bold red]✗ TS optimization failed:[/bold red] {traceback.format_exc()}"
            )
            raise typer.Exit(1)

    if output is not None and hasattr(output, "save"):
        output.save(results_name)
        console.print(f"[dim]Results: {results_name}[/dim]")
    if ts_node is None:
        console.print("[bold red]✗ TS optimization did not converge.[/bold red]")
        raise typer.Exit(1)
    ts_node.structure.save(filename)
    console.print(f"[bold green]✓ TS optimization complete![/bold green]")
    console.print(f"[dim]Geometry: {filename}[/dim]")


@app.command("pseuirc")
def pseuirc(geometry: Annotated[str, typer.Argument(help='path to geometry file to optimize')],
            inputs: Annotated[str, typer.Option("--inputs", "-i",
                                                help='path to RunInputs toml file')] = None,
            name: str = None,
            charge: int = 0,
            multiplicity: int = 1,
            dr: float = 1.0):
    console.print(BANNER)

    # create output names
    fp = Path(geometry)
    data_dir = Path(os.getcwd())

    if name is not None:
        results_name = data_dir / (name + ".qcio")
    else:
        results_name = Path(fp.stem + ".qcio")

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')

    with console.status(f"[bold cyan]Computing hessian...[/bold cyan]"):
        sys.stdout.flush()
        try:
            struct = Structure.open(geometry)
            s_dict = struct.model_dump()
            s_dict["charge"], s_dict["multiplicity"] = charge, multiplicity
            struct = Structure(**s_dict)

            node = StructureNode(structure=struct)
            hessres = program_input.engine._compute_hessian_result(node)

        except Exception as e:
            hessres = e.program_output

    hessres.save(results_name.parent / (results_name.stem+"_hessian.qcio"))

    console.print("[bold cyan]⟳ Minimizing TS(-)...[/bold cyan]")
    sys.stdout.flush()
    tsminus_raw = displace_by_dr(
        node=node, displacement=hessres.results.normal_modes_cartesian[0], dr=-dr)
    tsminus_res = program_input.engine._compute_geom_opt_result(
        tsminus_raw)
    tsminus_res.save(results_name.parent / (results_name.stem+"_minus.qcio"))

    console.print("[bold cyan]⟳ Minimizing TS(+)...[/bold cyan]")
    sys.stdout.flush()
    tsplus_raw = displace_by_dr(
        node=node, displacement=hessres.results.normal_modes_cartesian[0], dr=dr)
    tsplus_res = program_input.engine._compute_geom_opt_result(
        tsplus_raw)

    tsplus_res.save(results_name.parent / (results_name.stem+"_plus.qcio"))
    console.print(f"[bold green]✓ Pseudo-IRC complete![/bold green]")


@app.command("make-netgen-path")
def make_netgen_path(
    name: Annotated[Path, typer.Option("--name", help='path to json file containing network object')],
    inds: Annotated[List[int], typer.Option("--inds", help="sequence of node indices on network \
                                                               to create path for.")]):
    console.print(BANNER)
    name = Path(name)
    assert name.exists(), f"{name.resolve()} does not exist."
    pot = Pot.read_from_disk(name)
    if len(inds) > 2:
        p = inds
    else:
        console.print(
            "[cyan]Computing shortest path weighed by barrier heights[/cyan]")
        p = nx.shortest_path(pot.graph, weight='barrier',
                             source=inds[0], target=inds[1])
    console.print(f"[green]✓ Path:[/green] {p}")
    chain = pot.path_to_chain(path=p)
    chain.write_to_disk(
        name.parent / f"path_{'-'.join([str(a) for a in inds])}.xyz")
    console.print(f"[bold green]✓ Path written to disk![/bold green]")


@app.command("visualize")
def visualize(
    result_path: Annotated[str, typer.Argument(help="Path to a NEB result .xyz or TreeNode result folder")],
    output_html: Annotated[str, typer.Option("--output", "-o", help="Output HTML file path")] = None,
    qminds_fp: Annotated[str, typer.Option("--qminds-fp", help="Path to qmindices.dat for atom-subset visualization")] = None,
    atom_indices: Annotated[str, typer.Option("--atom-indices", help="Comma/space-separated atom indices (e.g. '1,2,3' or '1 2 3')")] = None,
    charge: Annotated[int, typer.Option(help="Charge used when reading serialized geometries")] = 0,
    multiplicity: Annotated[int, typer.Option(help="Spin multiplicity used when reading serialized geometries")] = 1,
    no_open: Annotated[bool, typer.Option("--no-open", help="Do not auto-open browser window")] = False,
):
    console.print(BANNER)
    src = Path(result_path).resolve()
    with console.status("[bold cyan]Loading result object...[/bold cyan]"):
        viz_data = _load_visualization_data(
            result_path=src,
            charge=charge,
            multiplicity=multiplicity,
        )
        selected = _parse_visualize_atom_indices(
            qminds_fp=qminds_fp, atom_indices=atom_indices
        )
        if selected is not None:
            viz_data.chain = _subset_chain_for_visualization(viz_data.chain, selected)
            if viz_data.chain_trajectory:
                viz_data.chain_trajectory = _subset_chain_trajectory_for_visualization(
                    viz_data.chain_trajectory, selected
                )
            if viz_data.tree_layers:
                viz_data.tree_layers = _subset_tree_layers_for_visualization(
                    viz_data.tree_layers, selected
                )
            console.print(
                f"[dim]Visualizing atom subset with {len(selected)} atoms.[/dim]"
            )

    with console.status("[bold cyan]Building interactive HTML...[/bold cyan]"):
        html = _build_chain_visualizer_html(
            chain=viz_data.chain,
            chain_trajectory=viz_data.chain_trajectory,
            tree_layers=viz_data.tree_layers,
        )

    if output_html is None:
        suffix = src.stem if src.is_file() else src.name
        out_fp = Path.cwd() / f"{suffix}_visualize.html"
    else:
        out_fp = Path(output_html).resolve()
    out_fp.write_text(html, encoding="utf-8")
    console.print(f"[bold green]✓ Visualization written:[/bold green] {out_fp}")

    if not no_open:
        webbrowser.open(out_fp.resolve().as_uri())
        console.print("[dim]Opened in default browser.[/dim]")


@app.command("make-default-inputs")
def make_default_inputs(
        name: Annotated[str, typer.Option(
            "--name", help='path to output toml file')] = None,
        path_min_method: Annotated[str, typer.Option("--path-min-method", "-pmm",
                                                     help='name of path minimization.\
                                                          Options are: [neb, fneb]')] = "neb"):
    console.print(BANNER)
    if name is None:
        name = Path(Path(os.getcwd()) / 'default_inputs')
    ri = RunInputs(path_min_method=path_min_method)
    out = Path(name)
    ri.save(out.parent / (out.stem+".toml"))
    console.print(
        f"[bold green]✓ Default inputs saved to:[/bold green] {out.parent / (out.stem+'.toml')}")


@app.command("netgen-smiles")
def netgen_smiles(
    smiles: Annotated[str, typer.Option("--smiles", "-s", help="Root reactant SMILES")] = None,
    inputs: Annotated[str, typer.Option("--inputs", "-i", help="Path minimization RunInputs TOML")] = None,
    reactions_fp: Annotated[str, typer.Option("--reactions-fp", help="Path to retropaths reactions.p file")] = None,
    environment: Annotated[str, typer.Option("--environment", "-e", help="Environment SMILES")] = "",
    name: Annotated[str, typer.Option("--name", help="Run name / default workspace folder name")] = None,
    directory: Annotated[str, typer.Option("--directory", "-d", help="Workspace directory")] = None,
    timeout_seconds: Annotated[int, typer.Option("--timeout-seconds", help="Retropaths growth timeout in seconds")] = 30,
    max_nodes: Annotated[int, typer.Option("--max-nodes", help="Retropaths maximum number of nodes")] = 40,
    max_depth: Annotated[int, typer.Option("--max-depth", help="Retropaths maximum search depth")] = 4,
    max_parallel_nebs: Annotated[int, typer.Option("--max-parallel-nebs", help="Number of recursive NEBs to run concurrently")] = 1,
    no_open: Annotated[bool, typer.Option("--no-open", help="Do not auto-open the generated status HTML")] = False,
):
    console.print(BANNER)
    if smiles is None:
        raise typer.BadParameter("--smiles is required.")
    if inputs is None:
        raise typer.BadParameter("--inputs/-i is required.")

    requested_dir = Path(directory).resolve() if directory else None
    if requested_dir is not None and (requested_dir / "workspace.json").exists():
        workspace = RetropathsWorkspace.read(requested_dir)
        workspace.max_parallel_nebs = max_parallel_nebs
        if reactions_fp is not None:
            workspace.reactions_fp = str(Path(reactions_fp).resolve())
        workspace.write()
    else:
        workspace = create_workspace(
            root_smiles=smiles,
            environment_smiles=environment,
            inputs_fp=inputs,
            reactions_fp=reactions_fp,
            name=name,
            directory=directory,
            timeout_seconds=timeout_seconds,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_parallel_nebs=max_parallel_nebs,
        )

    console.print(f"[bold cyan]Workspace:[/bold cyan] [white]{workspace.workdir}[/white]")
    console.print(f"[bold cyan]Root SMILES:[/bold cyan] [white]{workspace.root_smiles}[/white]")
    console.print(f"[bold cyan]Environment:[/bold cyan] [white]{workspace.environment_smiles or '(none)'}[/white]")
    console.print(f"[bold cyan]Reactions File:[/bold cyan] [white]{workspace.reactions_path}[/white]")

    with console.status("[bold cyan]Preparing retropaths cache, converted pot, and queue...[/bold cyan]"):
        prepare_neb_workspace(workspace)
        queue, _, status_fp = write_status_html(workspace)
    console.print(f"[dim]Initial status HTML: {status_fp}[/dim]")

    try:
        queue, _pot = run_netgen_smiles_workflow(
            workspace,
            progress=lambda msg: console.print(f"[cyan]{msg}[/cyan]"),
        )
    finally:
        queue, _, status_fp = write_status_html(workspace)

    counts = summarize_queue(queue)
    summary = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Workspace", workspace.workdir)
    summary.add_row("Queue Items", str(counts["items"]))
    summary.add_row("Completed", str(counts.get("completed", 0)))
    summary.add_row("Running", str(counts.get("running", 0)))
    summary.add_row("Pending", str(counts.get("pending", 0)))
    summary.add_row("Failed", str(counts.get("failed", 0)))
    summary.add_row("Incompatible", str(counts.get("incompatible", 0)))
    summary.add_row("Optimized Endpoints", str(sum(bool(_pot.graph.nodes[n].get("endpoint_optimized")) for n in _pot.graph.nodes)))
    summary.add_row("Status HTML", str(status_fp))
    console.print(Panel(summary, title="[bold green]✓ netgen-smiles Finished[/bold green]", border_style="green"))

    if not no_open:
        webbrowser.open(status_fp.resolve().as_uri())


@app.command("status")
def status_cmd(
    directory: Annotated[str, typer.Option("--directory", "-d", help="Workspace directory containing workspace.json")] = ".",
    output_html: Annotated[str, typer.Option("--output", "-o", help="Optional override path for generated status HTML")] = None,
    temperature: Annotated[float, typer.Option("--temperature", help="KMC temperature in kelvin for the status page")] = 298.15,
    initial_conditions: Annotated[List[str], typer.Option("--initial-condition", help="Override KMC initial conditions as NODE=VALUE. Repeatable.")] = None,
    no_open: Annotated[bool, typer.Option("--no-open", help="Do not auto-open browser window")] = False,
):
    console.print(BANNER)
    workspace_dir = Path(directory).resolve()
    workspace_fp = workspace_dir / "workspace.json"
    if not workspace_fp.exists():
        console.print(f"[bold red]✗ ERROR:[/bold red] No workspace.json found in {workspace_dir}")
        raise typer.Exit(1)

    workspace = RetropathsWorkspace.read(workspace_dir)
    kmc_initial_conditions = _parse_kmc_initial_condition_overrides(initial_conditions)
    queue, pot, status_fp = write_status_html(
        workspace,
        kmc_temperature_kelvin=temperature,
        kmc_initial_conditions=kmc_initial_conditions,
    )
    counts = summarize_queue(queue)

    if output_html is not None:
        out_fp = Path(output_html).resolve()
        out_fp.write_text(status_fp.read_text(encoding="utf-8"), encoding="utf-8")
        status_fp = out_fp

    summary = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Workspace", workspace.workdir)
    summary.add_row("Root", workspace.root_smiles)
    summary.add_row("Environment", workspace.environment_smiles or "(none)")
    summary.add_row("Queue Items", str(counts["items"]))
    summary.add_row("Completed", str(counts.get("completed", 0)))
    summary.add_row("Running", str(counts.get("running", 0)))
    summary.add_row("Pending", str(counts.get("pending", 0)))
    summary.add_row("Failed", str(counts.get("failed", 0)))
    summary.add_row("Incompatible", str(counts.get("incompatible", 0)))
    summary.add_row("Network Nodes", str(pot.graph.number_of_nodes()))
    summary.add_row("Network Edges", str(pot.graph.number_of_edges()))
    summary.add_row("Optimized Endpoints", str(sum(bool(pot.graph.nodes[n].get("endpoint_optimized")) for n in pot.graph.nodes)))
    summary.add_row("KMC Temperature (K)", f"{temperature:.2f}")
    summary.add_row("Status HTML", str(status_fp))
    console.print(Panel(summary, title="[bold cyan]Network Status[/bold cyan]", border_style="cyan"))

    if not no_open:
        webbrowser.open(status_fp.resolve().as_uri())


def _path_str_for_toml(path_text: str | None, source_dir: Path, output_dir: Path) -> str | None:
    if path_text is None:
        return None
    src = Path(path_text)
    if not src.is_absolute():
        src = (source_dir / src).resolve()
    try:
        return str(src.relative_to(output_dir))
    except ValueError:
        return str(src)


def _resolve_tcin_reference(path_text: str | None, source_dir: Path) -> str | None:
    if path_text is None:
        return None
    p = Path(path_text)
    if p.is_absolute():
        return str(p) if p.exists() else None

    direct = (source_dir / p)
    if direct.exists():
        return str(p)

    stem, suffix = p.stem, p.suffix
    candidates = sorted(source_dir.glob(f"{stem}*{suffix}"))
    if len(candidates) == 1:
        return candidates[0].name

    if "react" in stem.lower():
        react_candidates = sorted(source_dir.glob(f"*react*{suffix}"))
        if len(react_candidates) == 1:
            return react_candidates[0].name

    return None


@app.command("toml-from-tcin")
def toml_from_tcin(
    tcin: Annotated[str, typer.Argument(help="Path to TeraChem input file (.in/.tc.in)")],
    output: Annotated[str, typer.Option("--output", "-o", help="Output TOML file path")] = "qmmm_inputs_from_tc.toml",
    compute_program: Annotated[str, typer.Option("--compute-program", help="QMMM backend: chemcloud or qcop")] = "chemcloud",
    queue: Annotated[str, typer.Option("--queue", help="Optional ChemCloud queue to store in TOML")] = None,
):
    console.print(BANNER)
    tcin_fp = Path(tcin).resolve()
    if not tcin_fp.exists():
        raise typer.BadParameter(f"TeraChem input not found: {tcin_fp}")
    out_fp = Path(output).resolve()
    parsed = parse_terachem_input_file(tcin_fp)
    resolved_qminds = _resolve_tcin_reference(parsed["qmindices"], tcin_fp.parent)
    resolved_prmtop = _resolve_tcin_reference(parsed["prmtop"], tcin_fp.parent)
    resolved_coords = _resolve_tcin_reference(parsed["coordinates"], tcin_fp.parent)

    missing = []
    if resolved_qminds is None:
        missing.append(f"qmindices ({parsed['qmindices']})")
    if resolved_prmtop is None:
        missing.append(f"prmtop ({parsed['prmtop']})")
    if resolved_coords is None:
        missing.append(f"coordinates ({parsed['coordinates']})")
    if missing:
        raise typer.BadParameter(
            "Could not resolve required file references from tc.in: " + ", ".join(missing)
        )

    qmmm_inputs = {
        "qminds_fp": _path_str_for_toml(resolved_qminds, tcin_fp.parent, out_fp.parent),
        "prmtop_fp": _path_str_for_toml(resolved_prmtop, tcin_fp.parent, out_fp.parent),
        "rst7_fp_react": _path_str_for_toml(resolved_coords, tcin_fp.parent, out_fp.parent),
        "compute_program": compute_program,
        "charge": parsed["charge"],
        "spinmult": parsed["spinmult"],
        # NEB requires ene+grad evaluations; tc.in references are often minimization inputs.
        "run_type": "gradient",
        "min_coordinates": parsed["min_coordinates"],
    }
    qmmm_inputs = {k: v for k, v in qmmm_inputs.items() if v is not None}

    out = {
        "engine_name": "qmmm",
        "program": "terachem",
        "path_min_method": "neb",
        "program_kwds": {
            "model": {
                "method": parsed["method"],
                "basis": parsed["basis"],
            },
            "keywords": parsed["keywords"],
        },
        "qmmm_inputs": qmmm_inputs,
        "chain_inputs": {
            "frozen_atom_indices": parsed["frozen_atom_indices"],
            "use_geodesic_interpolation": False,
        },
    }
    if queue:
        out["chemcloud_queue"] = queue

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(tomli_w.dumps(out))

    n_frozen = len(parsed["frozen_atom_indices"])
    console.print(f"[bold green]✓ QMMM inputs TOML written:[/bold green] {out_fp}")
    console.print(f"[dim]Parsed {n_frozen} frozen atoms from $constraints.[/dim]")


@app.command("run-netgen")
def run_netgen(
        start: Annotated[str, typer.Option(
            help='path to start conformers data')] = None,
        end: Annotated[str, typer.Option(
            help='path to end conformers data')] = None,

        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to RunInputs toml file')] = None,

        name: str = None,
        charge: int = 0,
        multiplicity: int = 1,
        max_pairs: int = 500,
        minimize_ends: bool = False


):
    console.print(BANNER)

    start_time = time.time()

    # load the RunInputs
    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        if inputs is not None:
            program_input = RunInputs.open(inputs)
        else:
            program_input = RunInputs(program='xtb', engine_name='qcop')

    _render_runinputs(program_input)

    valid_suff = ['.qcio', '.xyz']
    assert (Path(start).suffix in valid_suff and Path(
        end).suffix in valid_suff), "Invalid file type. Make sure they are .qcio or .xyz files"

    # load the structures
    console.print(
        f"[dim]Loading structures: {Path(start).suffix} → {Path(end).suffix}[/dim]")
    if Path(start).suffix == ".qcio":
        start_structures = ProgramOutput.open(start).results.conformers
        start_nodes = [StructureNode(structure=s) for s in start_structures]
    elif Path(start).suffix == ".xyz":
        start_structures = read_multiple_structure_from_file(
            start, charge=charge, spinmult=multiplicity)
        if len(start_structures) != 1:
            start_nodes = [StructureNode(structure=s)
                           for s in start_structures]
        else:
            console.print(
                "[cyan]Only one structure in reactant file. Sampling using CREST![/cyan]")
            if minimize_ends:
                console.print(
                    "[bold cyan]⟳ Minimizing endpoints...[/bold cyan]")
                sys.stdout.flush()
                start_structure = program_input.engine.compute_geometry_optimization(
                    StructureNode(structure=start_structures[0]))[-1].structure
            else:
                start_structure = start_structures[0]

            console.print("[bold cyan]⟳ Sampling reactant...[/bold cyan]")
            sys.stdout.flush()
            try:
                start_conf_result = program_input.engine._compute_conf_result(
                    StructureNode(structure=start_structure))
                start_conf_result.save(Path(start).resolve(
                ).parent / (Path(start).stem + "_conformers.qcio"))

                start_nodes = [StructureNode(structure=s)
                               for s in start_conf_result.results.conformers]
                console.print("[bold green]✓ Done![/bold green]")

            except ExternalProgramError as e:
                console.print(f"[bold red]Error:[/bold red] {e.stdout}")

    if Path(end).suffix == ".qcio":
        end_structures = ProgramOutput.open(end).results.conformers
        end_nodes = [StructureNode(structure=s) for s in end_structures]
    elif Path(end).suffix == ".xyz":
        end_structures = read_multiple_structure_from_file(
            end, charge=charge, spinmult=multiplicity)

        if len(end_structures) != 1:
            end_nodes = [StructureNode(structure=s)
                         for s in end_structures]
        else:
            console.print(
                "[cyan]Only one structure in product file. Sampling using CREST![/cyan]")
            if minimize_ends:
                console.print(
                    "[bold cyan]⟳ Minimizing endpoints...[/bold cyan]")
                sys.stdout.flush()

                end_structure = program_input.engine.compute_geometry_optimization(
                    StructureNode(structure=end_structures[0]))[-1].structure
            else:
                end_structure = end_structures[0]

            console.print("[bold cyan]⟳ Sampling product...[/bold cyan]")
            sys.stdout.flush()
            end_conf_result = program_input.engine._compute_conf_result(
                StructureNode(structure=end_structure))
            end_conf_result.save(Path(end).resolve().parent /
                                 (Path(end).stem + "_conformers.qcio"))
            end_nodes = [StructureNode(structure=s)
                         for s in end_conf_result.results.conformers]
            console.print("[bold green]✓ Done![/bold green]")
            sys.stdout.flush()

    sys.stdout.flush()

    pairs = list(product(start_nodes, end_nodes))[:max_pairs]
    for i, (start_node, end_node) in enumerate(pairs):
        # create Chain
        chain = Chain.model_validate({
            "nodes": [start_node, end_node],
            "parameters": program_input.chain_inputs,
        })

        # define output names
        fp = Path("mep_output")
        if name is not None:
            data_dir = Path(name).resolve().parent
            foldername = data_dir / (name + f"_pair{i}")
            filename = data_dir / (name + f"_pair{i}.xyz")

        else:
            data_dir = Path(os.getcwd())
            foldername = data_dir / f"{fp.stem}_msmep_pair{i}"
            filename = data_dir / f"{fp.stem}_msmep_pair{i}.xyz"

        # create MSMEP object
        m = MSMEP(inputs=program_input)

        # Run the optimization

        console.print(
            f"[bold magenta]▶ Running autosplitting on pair {i+1}/{len(pairs)} ({program_input.path_min_method})[/bold magenta]")
        if filename.exists():
            console.print("[yellow]⚠ Already done. Skipping...[/yellow]")
            continue

        try:
            history = m.run_recursive_minimize(chain)

            history.output_chain.write_to_disk(filename)
            history.write_to_disk(foldername)
        except Exception:
            console.print(f"[bold red]✗ Failed on pair {i}[/bold red]")
            continue

    end_time = time.time()
    elapsed = end_time - start_time
    if elapsed > 60:
        time_str = f"{elapsed/60:.1f} min"
    else:
        time_str = f"{elapsed:.1f} s"
    console.print(
        f"[bold green]✓ Netgen complete![/bold green] [dim]Walltime: {time_str}[/dim]")


@app.command("make-netgen-summary")
def make_netgen_summary(
        directory: Annotated[str, typer.Option(
            "--directory", help='path to data directory')] = None,
        verbose: bool = False,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to RunInputs toml file')] = None,
        name: Annotated[str, typer.Option(
            "--name", help='name of pot and summary file')] = 'netgen'

):
    console.print(BANNER)

    if directory is None:
        directory = Path(os.getcwd())
    else:
        directory = Path(directory).resolve()
    if inputs is not None:
        ri = RunInputs.open(inputs)
        chain_inputs = ri.chain_inputs
    else:
        chain_inputs = ChainInputs()

    with console.status("[bold cyan]Building reaction network...[/bold cyan]"):
        nb = NetworkBuilder(data_dir=directory,
                            start=None, end=None,
                            network_inputs=NetworkInputs(verbose=verbose),
                            chain_inputs=chain_inputs)

        nb.msmep_data_dir = directory
        pot_fp = Path(directory / (name+".json"))
        if not pot_fp.exists():
            pot = nb.create_rxn_network(file_pattern="*")
            pot.write_to_disk(pot_fp)
        else:
            console.print(
                f"[yellow]⚠ {pot_fp} already exists. Loading...[/yellow]")
            pot = Pot.read_from_disk(pot_fp)

    with console.status("[bold cyan]Generating plots...[/bold cyan]"):
        plot_results_from_pot_obj(
            fp_out=(directory / f"{Path(name).stem+'.png'}"), pot=pot, include_pngs=True)
        plot_results_from_pot_obj(
            fp_out=(directory / f"{Path(name).stem+'.png'}"), pot=pot, include_pngs=False)

    # write nodes to xyz file
    nodes = [pot.graph.nodes[x]["td"] for x in pot.graph.nodes]
    chain = Chain.model_validate({"nodes": nodes})
    chain.write_to_disk(directory / 'nodes.xyz')

    console.print(f"[bold green]✓ Network summary complete![/bold green]")
    console.print(f"[dim]Network: {pot_fp}[/dim]")
    console.print(f"[dim]Nodes: {directory / 'nodes.xyz'}[/dim]")


if __name__ == "__main__":
    app()
