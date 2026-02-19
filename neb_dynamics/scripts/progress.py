"""Progress printing utilities for NEB Dynamics."""

import sys
import time
from typing import Optional

# Rich-based progress printer (used when rich is available)
_rich_available = False
try:
    from rich.console import Console
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.style import Style
    from rich.status import Status
    from rich import box

    _console = Console()
    _rich_available = True
except ImportError:
    _console = None

try:
    from qcinf import structure_to_smiles
except Exception:
    structure_to_smiles = None


class ProgressPrinter:
    """
    A class to handle progress printing with optional rich formatting.

    Can work in two modes:
    1. Rich mode (default): Uses rich library for fancy output
    2. Fallback mode: Simple text output for environments without rich
    """

    def __init__(self, use_rich: bool = True, update_interval: float = 0.1):
        self.use_rich = use_rich and _rich_available
        self.update_interval = update_interval
        self._last_print_time = 0.0
        self._current_task_id = None
        self._progress = None
        self._status = None
        self._status_active = False
        self._live = None
        self._last_ascii_plot = None
        self._last_caption = None
        self._last_status_message = None

        # Throttle updates to avoid too much output
        self._throttle = 0.5  # Only print every 0.5 seconds by default

    def start(self, description: str, total: Optional[int] = None):
        """Start a new progress task."""
        if self.use_rich:
            if self._progress is None:
                self._progress = Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=_console,
                )
                self._progress.start()

            self._current_task_id = self._progress.add_task(
                description,
                total=total,
                completed=0
            )
        else:
            sys.stdout.write(f"{description}\n")
            sys.stdout.flush()

    def update(self, advance: int = 1, **kwargs):
        """Update the current progress task."""
        if self.use_rich and self._progress and self._current_task_id is not None:
            self._progress.update(self._current_task_id,
                                  advance=advance, **kwargs)

    def set_description(self, description: str):
        """Update the task description."""
        if self.use_rich and self._progress and self._current_task_id is not None:
            self._progress.update(self._current_task_id,
                                  description=description)

    def stop(self):
        """Stop the current progress task."""
        if self.use_rich and self._progress:
            if self._current_task_id is not None:
                self._progress.remove_task(self._current_task_id)
                self._current_task_id = None

    def start_status(self, message: str):
        """Start a spinner status message."""
        if self.use_rich and self._live is not None:
            self._update_live_caption(message)
            return
        if self.use_rich:
            if self._status is None:
                self._status = Status(message, console=_console)
            else:
                self._status.update(message)
            if not self._status_active:
                self._status.start()
                self._status_active = True
        else:
            sys.stdout.write(f"{message}\n")
            sys.stdout.flush()

    def update_status(self, message: str):
        """Update the spinner status message."""
        if self.use_rich and self._live is not None:
            self._update_live_caption(message)
            return
        if self.use_rich:
            if self._status is None:
                _console.print(f"[dim]{message}[/dim]")
                return
            if not self._status_active:
                self._status.start()
                self._status_active = True
            self._status.update(message)
        elif not self.use_rich:
            sys.stdout.write(f"{message}\n")
            sys.stdout.flush()

    def stop_status(self):
        """Stop the spinner status message."""
        if self.use_rich and self._status and self._status_active:
            self._status.stop()
            self._status_active = False
        if self.use_rich and self._live is not None:
            # Ensure external rich prints (e.g., new-structure ASCII art)
            # are not obscured by an active Live renderer.
            self._live.stop()
            self._live = None

    def print_step(
        self,
        step: int,
        ts_grad: float,
        max_rms_grad: float,
        ts_triplet_gspring: float,
        nodes_frozen: int = 0,
        timestep: float = 0.0,
        grad_corr: str = "",
        force_update: bool = False
    ):
        """
        Print a formatted progress line for NEB optimization steps.

        This is the main method for displaying optimization progress.
        """
        current_time = time.time()

        # Throttle updates to avoid flooding output
        if not force_update and (current_time - self._last_print_time) < self._throttle:
            return

        self._last_print_time = current_time

        if self.use_rich:
            # Create a rich-formatted status line
            status_parts = [
                ("step ", "cyan"),
                (f"{step}", "bold white"),
                (" | ", "dim"),
                ("TS gperp: ", "yellow"),
                (f"{ts_grad:.4f}", "white"),
                (" | ", "dim"),
                ("max rms: ", "yellow"),
                (f"{max_rms_grad:.4f}", "white"),
                (" | ", "dim"),
                ("tspring: ", "yellow"),
                (f"{ts_triplet_gspring:.4f}", "white"),
            ]

            if nodes_frozen > 0 or timestep > 0:
                status_parts.extend([
                    (" | ", "dim"),
                    ("frozen: ", "magenta"),
                    (f"{nodes_frozen}", "white"),
                ])

            if timestep > 0:
                status_parts.extend([
                    (" | ", "dim"),
                    ("dt: ", "magenta"),
                    (f"{timestep:.3f}", "white"),
                ])

            if grad_corr:
                status_parts.extend([
                    (" | ", "dim"),
                    (grad_corr, "green"),
                ])

            # Print on same line using carriage return
            line = ""
            for text, style in status_parts:
                line += f"[{style}]{text}[/{style}]"

            # Use console.print with end="\n" to overwrite line
            _console.print(line, end="\n")
            sys.stdout.flush()
        else:
            # Simple text fallback
            line = f"step {step} | TS gperp: {ts_grad:.4f} | max rms: {max_rms_grad:.4f} | tspring: {ts_triplet_gspring:.4f}"
            if nodes_frozen > 0:
                line += f" | frozen: {nodes_frozen}"
            if timestep > 0:
                line += f" | dt: {timestep:.3f}"
            if grad_corr:
                line += f" | {grad_corr}"

            sys.stdout.write(f"\n{line}{' ' * 20}")
            sys.stdout.flush()

    def print_convergence(self, message: str = "Converged!"):
        """Print convergence message."""
        if self.use_rich and self._live is not None:
            self._live.stop()
            self._live = None
        if self.use_rich:
            _console.print(f"\n[bold green]✓[/bold green] {message}")
        else:
            print(f"\n{message}")

    def print_warning(self, message: str):
        """Print a warning message."""
        if self.use_rich and self._live is not None:
            self._live.stop()
            self._live = None
        if self.use_rich:
            _console.print(f"[yellow]⚠ {message}[/yellow]")
        else:
            print(f"WARNING: {message}")

    def print_error(self, message: str):
        """Print an error message."""
        if self.use_rich and self._live is not None:
            self._live.stop()
            self._live = None
        if self.use_rich:
            _console.print(f"[bold red]✗ {message}[/bold red]")
        else:
            print(f"ERROR: {message}")

    def print_chain_ascii(self, ascii_plot: str, caption: str, force_update: bool = False):
        """Render an ASCII chain profile as a rich table with a caption."""
        current_time = time.time()
        if not force_update and (current_time - self._last_print_time) < self._throttle:
            return

        if caption == self._last_caption and ascii_plot == self._last_ascii_plot:
            return

        self._last_print_time = current_time
        self._last_ascii_plot = ascii_plot
        self._last_caption = caption
        self._last_status_message = None

        if self.use_rich:
            if self._status_active:
                self._status.stop()
                self._status_active = False
            table = Table(show_header=False, box=box.SIMPLE, caption=caption)
            table.add_column("Chain")
            table.add_row(ascii_plot)

            if self._live is None:
                self._live = Live(
                    table,
                    console=_console,
                    refresh_per_second=4,
                    auto_refresh=False,
                )
                self._live.start()
                self._live.refresh()
            else:
                self._live.update(table)
                self._live.refresh()
        else:
            sys.stdout.write(f"\n{caption}\n{ascii_plot}\n")
            sys.stdout.flush()

    def _update_live_caption(self, message: str):
        """Update live table caption without repainting duplicate frames."""
        if self._live is None or self._last_ascii_plot is None:
            return
        if message == self._last_status_message:
            return
        self._last_status_message = message
        self._last_caption = message
        table = Table(show_header=False, box=box.SIMPLE, caption=message)
        table.add_column("Chain")
        table.add_row(self._last_ascii_plot)
        self._live.update(table)
        self._live.refresh()

    def preserve_chain_snapshot(self, note: Optional[str] = None):
        """Persist the current live chain table into scrollback, then reset live mode."""
        if self._last_ascii_plot is None:
            return

        caption = note if note else "Chain snapshot"

        if self.use_rich:
            table = Table(show_header=False, box=box.SIMPLE, caption=caption)
            table.add_column("Chain")
            table.add_row(self._last_ascii_plot)

            if self._live is not None:
                self._live.stop()
                self._live = None

            _console.print(table)
        else:
            sys.stdout.write(f"\n{caption}\n{self._last_ascii_plot}\n")
            sys.stdout.flush()

    def flush(self):
        """Flush stdout."""
        sys.stdout.flush()


# Global default instance
_default_printer: Optional[ProgressPrinter] = None


def get_progress_printer() -> ProgressPrinter:
    """Get or create the default progress printer instance."""
    global _default_printer
    if _default_printer is None:
        _default_printer = ProgressPrinter()
    return _default_printer


def print_neb_step(
    step: int,
    ts_grad: float,
    max_rms_grad: float,
    ts_triplet_gspring: float,
    nodes_frozen: int = 0,
    timestep: float = 0.0,
    grad_corr: str = "",
    force_update: bool = False
):
    """
    Convenience function to print NEB optimization step progress.

    Args:
        step: Current optimization step number
        ts_grad: Maximum tangential gradient at TS guess
        max_rms_grad: Maximum RMS gradient across chain
        ts_triplet_gspring: Triplet g-spring value
        nodes_frozen: Number of frozen nodes
        timestep: Current timestep
        grad_corr: Gradient correction indicator
        force_update: Force update even if throttled
    """
    printer = get_progress_printer()
    printer.print_step(
        step=step,
        ts_grad=ts_grad,
        max_rms_grad=max_rms_grad,
        ts_triplet_gspring=ts_triplet_gspring,
        nodes_frozen=nodes_frozen,
        timestep=timestep,
        grad_corr=grad_corr,
        force_update=force_update
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
    npoints = len(labels)
    # Keep x-axis labels readable by showing a sparse set of node indices.
    stride = max(1, npoints // 8)
    shown = set(range(0, npoints, stride))
    shown.add(npoints - 1)
    for i, x in enumerate(xs):
        if i not in shown:
            continue
        tlabel = str(labels[i])
        start = max(0, min(width - len(tlabel), x - len(tlabel) // 2))
        for j, ch in enumerate(tlabel):
            label_line[start + j] = ch
    lines.append(" " * prefix_len + "".join(label_line))
    lines.append(" " * prefix_len + "node index")

    return "\n".join(lines)


def _labels_for_chain(chain):
    return [str(i) for i, _ in enumerate(chain.nodes)]


def _endpoint_smiles_for_chain(chain):
    if structure_to_smiles is None or len(chain.nodes) == 0:
        return "N/A", "N/A"
    try:
        start = structure_to_smiles(chain.nodes[0].structure)
    except Exception:
        start = "N/A"
    try:
        end = structure_to_smiles(chain.nodes[-1].structure)
    except Exception:
        end = "N/A"
    return start, end


def _truncate_meta(text: str, max_len: int = 100) -> str:
    return _truncate_label(text, max_len)


def ascii_profile_for_chain(chain, width: int = 60, height: int = 12) -> str:
    try:
        energies = chain.energies_kcalmol
    except Exception:
        energies = chain.energies
    labels = _labels_for_chain(chain)
    plot = _build_ascii_energy_profile(energies, labels, width=width, height=height)
    start_smi, end_smi = _endpoint_smiles_for_chain(chain)
    meta = (
        f"start SMILES: {_truncate_meta(start_smi)}\n"
        f"end SMILES:   {_truncate_meta(end_smi)}\n"
    )
    return meta + plot


def format_neb_caption(
    step: int,
    ts_grad: float,
    max_rms_grad: float,
    ts_triplet_gspring: float,
    nodes_frozen: int = 0,
    timestep: float = 0.0,
    grad_corr: Optional[float] = None,
):
    parts = [
        f"step {step}",
        f"TS gperp: {ts_grad:.4f}",
        f"max rms: {max_rms_grad:.4f}",
        f"tspring: {ts_triplet_gspring:.4f}",
    ]
    if nodes_frozen > 0:
        parts.append(f"frozen: {nodes_frozen}")
    if timestep > 0:
        parts.append(f"dt: {timestep:.3f}")
    if grad_corr is not None and grad_corr != "":
        try:
            parts.append(f"grad corr: {float(grad_corr):.3f}")
        except Exception:
            parts.append(f"grad corr: {grad_corr}")
    return " | ".join(parts)


def print_chain_step(chain, caption: str, force_update: bool = False):
    printer = get_progress_printer()
    ascii_plot = ascii_profile_for_chain(chain)
    printer.print_chain_ascii(ascii_plot, caption, force_update=force_update)


def preserve_chain_snapshot(note: Optional[str] = None):
    printer = get_progress_printer()
    printer.preserve_chain_snapshot(note=note)


def start_status(message: str):
    """Convenience function to start a spinner status."""
    printer = get_progress_printer()
    printer.start_status(message)


def update_status(message: str):
    """Convenience function to update a spinner status."""
    printer = get_progress_printer()
    printer.update_status(message)


def stop_status():
    """Convenience function to stop a spinner status."""
    printer = get_progress_printer()
    printer.stop_status()
