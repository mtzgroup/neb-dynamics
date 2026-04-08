from __future__ import annotations

import logging

from openbabel import openbabel
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "header": "bold magenta",
        "dim": "dim",
    }
)

BANNER = """
[bold magenta]╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║                        [bold cyan]NEB[/bold cyan] [bold white]-[/bold white] [bold cyan]Dynamics[/bold cyan]                         ║
║        [dim]Reaction Path Optimization & Network Generation[/dim]        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝[/bold magenta]
"""


class _SuppressWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING


logging.getLogger().addFilter(_SuppressWarningFilter())


def _configure_cli_logging() -> None:
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

openbabel.obErrorLog.SetOutputLevel(0)

console = Console(theme=custom_theme)


def print_banner() -> None:
    console.print(BANNER)


def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
