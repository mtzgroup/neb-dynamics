"""Progress printing utilities for NEB Dynamics."""

import sys
import time
from typing import Optional

# Rich-based progress printer (used when rich is available)
_rich_available = False
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.style import Style
    from rich.status import Status

    _console = Console()
    _rich_available = True
except ImportError:
    _console = None


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
        if self.use_rich:
            _console.print(f"\n[bold green]✓[/bold green] {message}")
        else:
            print(f"\n{message}")

    def print_warning(self, message: str):
        """Print a warning message."""
        if self.use_rich:
            _console.print(f"[yellow]⚠ {message}[/yellow]")
        else:
            print(f"WARNING: {message}")

    def print_error(self, message: str):
        """Print an error message."""
        if self.use_rich:
            _console.print(f"[bold red]✗ {message}[/bold red]")
        else:
            print(f"ERROR: {message}")

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
