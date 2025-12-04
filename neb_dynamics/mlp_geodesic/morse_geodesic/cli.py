"""
Main command-line script for Morse-Geodesic Interpolation.

This script serves as the primary entry point for users wishing to perform
Morse-geodesic interpolation or smoothing of molecular reaction paths from
the command line. It takes an input XYZ file containing initial geometries,
processes them according to specified parameters, and outputs a smoothed path
in XYZ format.

The method optimizes a reaction path by minimizing its length in a metric space
defined by scaled redundant internal coordinates. This approach aims to avoid
discontinuity and convergence problems often found in conventional interpolation
methods by incorporating internal coordinate structure while operating in Cartesian
coordinates, thereby avoiding unfeasibility issues often encountered with pure
internal coordinate optimizations.

The underlying methodology is based on the principles described in:
"Geodesic interpolation for reaction pathways." 
Xiaolei Zhu, Keiran C. Thompson, and Todd J. Martínez.
J. Chem. Phys. 150, 164103 (2019).
https://doi.org/10.1063/1.5090303
"""

import logging
import click  
import numpy as np
import sys
import os
from typing import Optional

from fileio import read_xyz, write_xyz
from interpolation import redistribute
from morsegeodesic import MorseGeodesic
from config import MAIN_DEFAULTS, INTERPOLATION_DEFAULTS, LOGGING_LEVELS, LOGGING_FORMAT

# Set up a logger for this specific module.
logger = logging.getLogger(__name__)


# --- CLI Definition using Click ---
@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument("input_xyz", type=click.Path(exists=True, dir_okay=False, resolve_path=True))

@click.argument("output_xyz", type=click.Path(dir_okay=False, resolve_path=True))

@click.option(
    "--nimages",
    type=int,
    default=MAIN_DEFAULTS["nimages"],
    show_default=True,
    help="Target number of images (frames) in the final, smoothed path. Must be >= 2."
)

@click.option(
    "--tol",
    type=float,
    default=MAIN_DEFAULTS["tolerance"],
    show_default=True,
    help="Convergence tolerance for the path optimization.  This is based on the infinity norm of the path length gradient (|dL|_inf). Optimization stops when this value drops below the specified tolerance."
)

@click.option(
    "--maxiter",
    type=int,
    default=MAIN_DEFAULTS["max_iterations"],
    show_default=True,
    help="Maximum number of function evaluations for the SciPy least-squares optimizer during the path smoothing process."
)

@click.option(
    "--alpha", "--scaling", "morse_alpha",
    type=float,
    default=MAIN_DEFAULTS["morse_alpha"],
    show_default=True,
    help="The 'alpha' parameter for the Morse potential-like scaleri used in MorseGeodesic smoothing."
)

@click.option(
    "--friction",
    type=float,
    default=MAIN_DEFAULTS["friction"],
    show_default=True,
    help="Base friction term coefficient for regularization during optimization. A small positive value can help stabilize the optimization and prevent excessive deviations from an initial path."
)

@click.option(
    "--dist-cutoff", "distance_cutoff",
    type=float,
    default=MAIN_DEFAULTS["distance_cutoff"],
    show_default=True,
    help="Cut-off distance (Angstroms) for including atom pairs in the internal coordinates list."
)

@click.option(
    "--logging", "logging_level",
    type=click.Choice(list(LOGGING_LEVELS.keys()), case_sensitive=False),
    default=MAIN_DEFAULTS["logging_level"],
    show_default=True,
    help="Set the logging level for console output (e.g., DEBUG, INFO, WARNING, ERROR)."
)

@click.option(
    "--save-raw", "save_raw_filename",
    type=str,
    default=MAIN_DEFAULTS["save_raw_filename"],
    help="If a filename is specified, the raw path (after redistribution but before before Morse-Geodesic smoothing) will be saved to this XYZ file."
)

def main(**kwargs):
    """
    Main entry point for the Morse-geodesic interpolation command-line tool.
    This function is now decorated with @click commands.
    """
    # --- 1. Argument Parsing is handled by Click ---
    # The `kwargs` dictionary contains all parsed arguments.

    # --- 2. Logging Setup ---
    log_level_val = LOGGING_LEVELS.get(kwargs['logging_level'].upper(), logging.INFO)
    logging.basicConfig(format=LOGGING_FORMAT, level=log_level_val, force=True)
    logger.info(f"Logging level set to {kwargs['logging_level'].upper()}.")

    # --- 3. Input File Validation ---
    if not os.path.exists(kwargs['input_xyz']):
        logger.error(f"Input file not found: {kwargs['input_xyz']}")
        sys.exit(1)

    if kwargs['nimages'] < 2:
        logger.error(f"Target number of images (--nimages) must be at least 2. Received: {kwargs['nimages']}.")
        sys.exit(1)

    # --- 4. Read Input Geometries ---
    try:
        symbols, initial_geometries = read_xyz(kwargs['input_xyz'])
        logger.info(f"Successfully loaded {len(initial_geometries)} geometries with {len(symbols)} atoms each from '{kwargs['input_xyz']}'.")
    except (ValueError, FileNotFoundError, IOError) as e:
        logger.error(f"Error reading XYZ file '{kwargs['input_xyz']}': {e}")
        sys.exit(1)

    num_atoms = len(symbols)
    num_initial_frames = len(initial_geometries)

    if num_atoms < 3:
        logger.error(f"The system must contain at least 3 atoms. Found only {num_atoms} in '{kwargs['input_xyz']}'.")
        sys.exit(1)

    if num_initial_frames < 2:
        logger.error(f"At least two initial geometries are required. Found only {num_initial_frames} in '{kwargs['input_xyz']}'.")
        sys.exit(1)

    # --- 5. Redistribute Images to Target Number ---
    logger.info(f"Redistributing initial path of {num_initial_frames} frames to {kwargs['nimages']} target images...")
    # Tolerance for midpoint finding during redistribution is scaled from the main optimization tolerance.
    midpoint_tol_for_redistribute = kwargs['tol'] * INTERPOLATION_DEFAULTS["midpoint_tolerance_factor"]
    try:
        raw_path_geoms_list = redistribute(
            atoms=symbols,
            geoms=initial_geometries,
            nimages=kwargs['nimages'],
            tol=midpoint_tol_for_redistribute
        )
        raw_path = np.array(raw_path_geoms_list)
        logger.info(f"Path redistribution complete. Path now has {len(raw_path)} images.")
    except Exception as e:
        logger.error(f"An error occurred during path redistribution: {type(e).__name__}: {e}", exc_info=(log_level_val <= logging.DEBUG))
        sys.exit(1)

    # --- 6. Optionally Save Raw Redistributed Path ---
    if kwargs['save_raw_filename']:
        logger.info(f"Saving raw redistributed path to '{kwargs['save_raw_filename']}'...")
        try:
            write_xyz(kwargs['save_raw_filename'], symbols, raw_path)
            logger.info(f"Raw path saved successfully.")
        except (IOError, Exception) as e:
            logger.error(f"Could not write raw path file '{kwargs['save_raw_filename']}': {e}")

    # --- 7. Initialize MorseGeodesic Smoother ---
    logger.info("Initializing MorseGeodesic smoother object...")
    try:
        smoother = MorseGeodesic(
            atoms=symbols,
            path=raw_path,
            scaler=kwargs['morse_alpha'],
            threshold=kwargs['distance_cutoff'],
            log_level=log_level_val,
            friction=kwargs['friction']
        )
        logger.info("MorseGeodesic smoother initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing the MorseGeodesic object: {type(e).__name__}: {e}", exc_info=(log_level_val <= logging.DEBUG))
        sys.exit(1)

    # --- 8. Perform Path Smoothing ---
    logger.info(f"Starting path smoothing with tolerance={kwargs['tol']} and max_iter={kwargs['maxiter']}...")
    final_path: Optional[np.ndarray] = None
    try:
        final_path = smoother.smooth(
            tol=kwargs['tol'],
            max_iter=kwargs['maxiter'],
            friction=kwargs['friction']
        )
        logger.info("Path smoothing optimization finished.")
    except Exception as e:
        logger.error(f"An error occurred during path optimization: {type(e).__name__}: {e}", exc_info=(log_level_val <= logging.DEBUG))
        final_path = smoother.path
        logger.warning("Optimization failed. Attempting to save the path state before the error.")

    # --- 9. Save Final Path ---
    if final_path is not None and final_path.size > 0:
        logger.info(f"Saving final optimized path to file '{kwargs['output_xyz']}'...")
        try:
            write_xyz(kwargs['output_xyz'], symbols, final_path)
            logger.info(f"Final path successfully saved to '{kwargs['output_xyz']}'.")
        except (IOError, Exception) as e:
            logger.error(f"Could not write final path file '{kwargs['output_xyz']}': {e}")
            sys.exit(1)
    else:
        logger.error("Optimization did not produce a valid final path. Output file will not be written.")
        sys.exit(1)

    logger.info("Morse-geodesic interpolation/smoothing process finished successfully.")
    click.echo("Process finished successfully.")


if __name__ == "__main__":
    main()

