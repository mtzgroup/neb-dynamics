# mlp_geodesic/cli.py
"""
Command-line interface for Geodesic Path Optimization.

This script provides a user-friendly CLI to run the optimization,
parsing all hyperparameters and file paths. 
"""
import logging
import sys
import click
import torch
from pathlib import Path
from ase import Atoms
from ase.io import read
from typing import List

from optimizer import GeodesicOptimizer
from utils import write_xyz_with_energies, OptimizerConfig, create_frames_from_coords

application_log = logging.getLogger("geodesic")

class _SuppressWarningFilter(logging.Filter):
    """A logging filter to suppress specific log levels, e.g., warnings."""
    def filter(self, record):
        return record.levelno != logging.WARNING

@click.command()
@click.argument('input_xyz', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('output_xyz', type=click.Path(dir_okay=False, resolve_path=True))
@click.option('--backend', type=click.Choice(['mace', 'egret', 'fairchem']), default='fairchem', show_default=True, help='MLP backend.')
@click.option('--model-path', type=click.Path(exists=True, dir_okay=False, resolve_path=True), default=None, help='Path to ML model.')
@click.option('--dtype', type=click.Choice(['float32','float64']), default='float32', show_default=True, help='Calculation precision.')
@click.option('--device', default='cuda', show_default=True, help='Device (e.g., "cuda", "cpu").')
@click.option('--fire-stage1-iter', type=int, default=200, show_default=True, help='Max FIRE Stage 1 iterations.')
@click.option('--fire-stage2-iter', type=int, default=500, show_default=True, help='Max FIRE Stage 2 iterations.')
@click.option('--fire-grad-tol', type=float, default=1e-2, show_default=True, help='FIRE convergence gradient tolerance.')
@click.option('--variance-penalty-weight', type=float, default=0.0433641, show_default=True, help='beta: Path variance penalty weight (eV).')
@click.option('--fire-conv-window', type=int, default=20, show_default=True, help='Convergence check window size.')
@click.option('--fire-conv-geolen-tol', type=float, default=0.0108410, show_default=True, help='Path length span tolerance (eV).')
@click.option('--fire-conv-erelpeak-tol', type=float, default=0.0108410, show_default=True, help='Barrier height span tolerance (eV).')
@click.option('--refinement-step-interval', type=int, default=10, show_default=True, help='Frequency of Stage 2 midpoint refinement.')
@click.option('--refinement-dynamic-threshold-fraction', type=float, default=0.1, show_default=True, help='Fraction of segment length for dynamic refinement threshold.')
@click.option('--tangent-project/--no-tangent-project', default=True, show_default=True, help='Project path gradient perpendicular to the tangent.')
@click.option('--climb/--no-climb', default=True, show_default=True, help='Enable climbing image on the highest energy node.')
@click.option('--alpha-climb', type=float, default=0.5, show_default=True, help='Scaling factor for the climbing force in Stage 2.')
@click.option('--verbose', '-v', is_flag=True, help="Enable DEBUG logging for the logger.")

def main(**kwargs):
    """Optimizes a geodesic path using a two-stage FIRE optimizer with an MLP."""

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if not kwargs['verbose']:
        logging.getLogger().addFilter(_SuppressWarningFilter())
    
    application_log.setLevel(logging.DEBUG if kwargs['verbose'] else logging.INFO)
    application_log.info("Initializing Geodesic Path Optimization.")

    device = kwargs['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        application_log.warning('CUDA specified but unavailable; switching to CPU.')
        device = 'cpu'

    dtype = torch.float32 if kwargs['dtype'] == 'float32' else torch.float64
    application_log.info(f"Using device: {device}, Precision: {kwargs['dtype']}")

    backend = kwargs['backend']
    actual_model_path = kwargs['model_path']


    if backend == 'fairchem' and actual_model_path is None:
        default_fairchem_path = '/home/diptarka/fairchem/esen_sm_conserving_all.pt'
        application_log.warning(f"No --model-path for fairchem, attempting default: {default_fairchem_path}")
        actual_model_path = default_fairchem_path
 
        if not Path(actual_model_path).exists():
             err_msg = f"Default FAIRChem model not found: {actual_model_path}."
             application_log.error(err_msg)
             click.echo(f"Error: {err_msg}", err=True)
             sys.exit(1)

    elif actual_model_path is None and backend != 'fairchem':
        err_msg = f"--model-path is required for backend '{backend}'."
        application_log.error(err_msg)
        click.echo(f"Error: {err_msg}", err=True)
        sys.exit(1)

    application_log.info(f"Using {backend} model from: {actual_model_path}")

    try:
        initial_frames = read(kwargs['input_xyz'], ':')
        if len(initial_frames) < 3:
            raise ValueError('Input XYZ must contain at least 3 frames.')
        application_log.info(f"Read {len(initial_frames)} initial frames from {kwargs['input_xyz']}.")

    except Exception as e:
        application_log.exception(f"Error reading input file {kwargs['input_xyz']}.")
        click.echo(f"Error reading input file {kwargs['input_xyz']}: {e}", err=True)
        sys.exit(1)

    try:
        config = OptimizerConfig.from_cli_kwargs(kwargs)
        
        opt = GeodesicOptimizer(
            frames=initial_frames,
            backend=backend,
            model_path=actual_model_path,
            device=device,
            dtype=dtype,
            config=config
        )

        main_coords, main_E = opt.optimize()

        template_atoms = initial_frames[0].copy()
        out_frames = create_frames_from_coords(main_coords, template_atoms)

        write_xyz_with_energies(out_frames, main_E, kwargs['output_xyz'])
        msg = f"Optimized geodesic ({len(out_frames)} frames) saved to {kwargs['output_xyz']}"
        application_log.info(msg)
        click.echo(msg)
    except Exception as e:
        application_log.exception("An unexpected critical error occurred during optimization.")
        click.echo(f"An unexpected critical error occurred: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

