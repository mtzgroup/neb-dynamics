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
Xiaolei Zhu, John M. Herbert, J. Chem. Phys. 150, 164115 (2019);
https://doi.org/10.1063/1.5090303
"""

import logging
import argparse
import numpy as np
import sys
import os
from typing import Optional # For type hinting `final_path`

# Import local modules from the MorseGeodesic package
from .fileio import read_xyz, write_xyz
from .interpolation import redistribute
from .morsegeodesic import MorseGeodesic
from .config import MAIN_DEFAULTS, INTERPOLATION_DEFAULTS, LOGGING_LEVELS, LOGGING_FORMAT

# Set up a logger for this specific module.
# Messages will be prefixed with information about this module and function.
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main entry point for the Morse-geodesic interpolation command-line tool.

    This function orchestrates the entire process:
    1.  Parses command-line arguments provided by the user, which control
        various aspects of the interpolation and smoothing.
    2.  Sets up the logging system based on the user-specified logging level.
    3.  Reads the initial molecular geometries from an input XYZ file.
    4.  Performs basic validation of the input data (e.g., minimum number
        of atoms and frames required).
    5.  Redistributes the images (frames) in the path to achieve the target
        number of images specified by the user.
    6.  Optionally saves this "raw" redistributed path to a file if requested.
    7.  Initializes the `MorseGeodesic` smoother object with the redistributed
        path and relevant parameters.
    8.  Performs the core path smoothing optimization using the Morse-Geodesic method.
    9.  Saves the final, optimized path to an output XYZ file.

    Error handling and informative logging are included throughout these steps.
    """

    # --- 1. Argument Parsing ---
    # Setup the command-line argument parser with descriptions and default values.
    parser = argparse.ArgumentParser(
        description="Interpolates and smooths molecular reaction paths using the Morse-Geodesic method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help messages.
    )

    # Positional argument: input filename
    parser.add_argument(
        "filename",
        type=str,
        help="Input XYZ file containing initial molecular geometries. "
             "The file must contain at least 2 geometries (frames), and each geometry "
             "should have at least 3 atoms for meaningful interpolation."
    )

    # Optional arguments
    parser.add_argument(
        "--nimages",
        type=int,
        default=MAIN_DEFAULTS["nimages"],
        help="Target number of images (frames) in the final, smoothed path. Must be >= 2."
    )
    parser.add_argument(
        "--output", "-o",
        default=MAIN_DEFAULTS["output_filename"],
        type=str,
        help="Filename for the output XYZ file that will contain the final, optimized path."
    )
    parser.add_argument(
        "--tol",
        default=MAIN_DEFAULTS["tolerance"],
        type=float,
        help="Convergence tolerance for the path optimization. This is based on the "
             "infinity norm of the path length gradient (|dL|_inf). Optimization stops "
             "when this value drops below the specified tolerance."
    )
    parser.add_argument(
        "--maxiter",
        default=MAIN_DEFAULTS["max_iterations"],
        type=int,
        help="Maximum number of function evaluations allowed for the SciPy "
             "least-squares optimizer during the path smoothing process."
    )
    parser.add_argument(
        "--alpha", "--scaling", # Allow both --alpha and --scaling
        dest='morse_alpha',     # Store the value in args.morse_alpha
        default=MAIN_DEFAULTS["morse_alpha"],
        type=float,
        help="The 'alpha' parameter for the Morse potential-like scaler used in "
             "MorseGeodesic smoothing. Controls the 'stiffness' of the scaled internals."
    )
    parser.add_argument(
        "--friction",
        default=MAIN_DEFAULTS["friction"],
        type=float,
        help="Base friction term coefficient used for regularization during the optimization. "
             "A small positive value can help stabilize the optimization and prevent "
             "excessive deviations from an initial path."
    )
    parser.add_argument(
        "--dist-cutoff",
        dest='distance_cutoff', # Store the value in args.distance_cutoff
        default=MAIN_DEFAULTS["distance_cutoff"],
        type=float,
        help="Cut-off distance (in Angstroms) for including atom pairs in the "
             "internal coordinate list. This is used by the `get_bond_list` function "
             "within the MorseGeodesic class initialization."
    )
    parser.add_argument(
        "--logging",
        default=MAIN_DEFAULTS["logging_level"],
        choices=LOGGING_LEVELS.keys(), # Restrict choices to defined logging levels
        help="Set the logging level for console output (e.g., DEBUG, INFO, WARNING, ERROR)."
    )
    parser.add_argument(
        "--save-raw",
        dest='save_raw_filename', # Store the value in args.save_raw_filename
        default=MAIN_DEFAULTS["save_raw_filename"], # Default is None (do not save raw path).
        type=str, # Expects a filename string if provided.
        help="If a filename is specified, the raw path (after image redistribution but "
             "before Morse-Geodesic smoothing) will be saved to this XYZ file. "
             "If not specified, the raw path is not saved."
    )

    args = parser.parse_args() # Parse the command-line arguments.

    # --- 2. Logging Setup ---
    # Determine the integer logging level from the string argument (e.g., "INFO" -> logging.INFO).
    log_level_val = LOGGING_LEVELS.get(args.logging.upper(), logging.INFO) # Default to INFO if an invalid string is given.
    # Configure the root logger. `force=True` ensures this configuration takes precedence
    # even if other parts of a larger application have already configured logging.
    logging.basicConfig(format=LOGGING_FORMAT, level=log_level_val, force=True)
    logger.info(f"Logging level set to {args.logging.upper()}.")

    # --- 3. Input File Validation ---
    if not os.path.exists(args.filename):
        logger.error(f"Input file not found: {args.filename}")
        sys.exit(1) # Critical error, exit the program.

    if args.nimages < 2: # A path requires at least two images (start and end points).
        logger.error(f"Target number of images (--nimages) must be at least 2. Received: {args.nimages}.")
        sys.exit(1) # Critical error, exit.

    # --- 4. Read Input Geometries ---
    try:
        symbols, initial_geometries = read_xyz(args.filename)
        logger.info(f"Successfully loaded {len(initial_geometries)} geometries (frames) "
                    f"with {len(symbols)} atoms each from '{args.filename}'.")
    except ValueError as e: # Handles formatting errors in the XYZ file.
        logger.error(f"Error reading XYZ file '{args.filename}': {e}")
        sys.exit(1)
    except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard.
        logger.error(f"Input file '{args.filename}' not found during read operation (should have been caught earlier).")
        sys.exit(1)
    except IOError as e: # Handles other file reading issues.
        logger.error(f"IOError encountered while reading file '{args.filename}': {e}")
        sys.exit(1)

    # --- Validate Number of Atoms and Frames from Input ---
    num_atoms = len(symbols)
    num_initial_frames = len(initial_geometries)

    if num_atoms < 3: # Meaningful interpolation usually requires at least 3 atoms.
                      # (e.g., for defining angles, or for Kabsch alignment to be well-defined).
        logger.error(
            f"The molecular system must contain at least 3 atoms for meaningful interpolation. "
            f"Found only {num_atoms} atoms in '{args.filename}'."
        )
        sys.exit(1)

    if num_initial_frames < 2: # Need at least two points (start and end) to define a path.
        logger.error(
            f"At least two initial geometries (frames) are required in the input XYZ file "
            f"to define a path. Found only {num_initial_frames} frames in '{args.filename}'."
        )
        sys.exit(1)

    # --- 5. Redistribute Images to Target Number ---
    logger.info(f"Redistributing initial path of {num_initial_frames} frames to {args.nimages} target images...")
    # Tolerance for midpoint finding during redistribution is scaled from the main optimization tolerance.
    midpoint_tol_for_redistribute = args.tol * INTERPOLATION_DEFAULTS["midpoint_tolerance_factor"]
    try:
        raw_path_geoms_list = redistribute(
            atoms=symbols,
            geoms=initial_geometries, # Pass the list of np.ndarrays from read_xyz
            nimages=args.nimages,
            tol=midpoint_tol_for_redistribute
        )
        # Convert the list of geometries from redistribute into a single 3D NumPy array.
        raw_path = np.array(raw_path_geoms_list)
        logger.info(f"Path redistribution complete. Path now has {len(raw_path)} images.")
    except Exception as e: # Catch any error during the redistribution process.
        logger.error(
            f"An error occurred during path redistribution: {type(e).__name__}: {e}",
            exc_info=(log_level_val <= logging.DEBUG) # Show full traceback if logging level is DEBUG.
        )
        sys.exit(1) # Critical error, exit.

    # --- 6. Optionally Save Raw Redistributed Path ---
    if args.save_raw_filename:
        logger.info(f"Saving raw redistributed path (before smoothing) to '{args.save_raw_filename}'...")
        try:
            write_xyz(args.save_raw_filename, symbols, raw_path)
            logger.info(f"Raw path saved successfully.")
        except IOError as e:
            logger.error(f"Could not write raw path file '{args.save_raw_filename}': {e}")
            # This is not a critical error; continue with smoothing even if saving raw path fails.
        except Exception as e: # Catch other unexpected errors during write.
            logger.error(f"An unexpected error occurred while writing the raw path file "
                         f"'{args.save_raw_filename}': {e}")

    # --- 7. Initialize MorseGeodesic Smoother ---
    logger.info("Initializing MorseGeodesic smoother object...")
    try:
        smoother = MorseGeodesic(
            atoms=symbols,
            path=raw_path,                  # Pass the 3D NumPy array from redistribution.
            scaler=args.morse_alpha,        # Use Morse alpha from command line.
            threshold=args.distance_cutoff, # Use distance cutoff from command line.
            log_level=log_level_val,        # Pass the integer log level value.
            friction=args.friction          # Pass friction from command line.
        )
        logger.info("MorseGeodesic smoother initialized successfully.")
    except Exception as e: # Catch any error during smoother initialization.
        logger.error(
            f"Error initializing the MorseGeodesic object: {type(e).__name__}: {e}",
            exc_info=(log_level_val <= logging.DEBUG) # Show traceback for DEBUG level.
        )
        sys.exit(1) # Critical error, exit.

    # --- 8. Perform Path Smoothing ---
    logger.info(f"Starting path smoothing optimization with tolerance={args.tol} and max_iter={args.maxiter}...")
    final_path: Optional[np.ndarray] = None # Initialize to None.
    try:
        # Call the main smoothing method of the MorseGeodesic object.
        # This uses SciPy's least_squares internally by default.
        final_path = smoother.smooth(
            tol=args.tol,
            max_iter=args.maxiter,
            friction=args.friction # Pass the command-line friction to the smooth method.
            # `start` and `end` arguments for smoother.smooth() are not specified here,
            # so it will use its internal defaults (typically optimizing all interior images).
        )
        logger.info("Path smoothing optimization finished.")
    except Exception as e: # Catch any error during the optimization process.
        logger.error(
            f"An error occurred during path optimization: {type(e).__name__}: {e}",
            exc_info=(log_level_val <= logging.DEBUG) # Show traceback for DEBUG level.
        )
        final_path = smoother.path # Try to get the path state as it was before the error.
        logger.warning(
            "Optimization process failed or was interrupted. "
            "Attempting to save the path state as it was before the error."
        )

    # --- 9. Save Final Path ---
    if final_path is not None and final_path.size > 0: # Ensure path is valid before saving.
        logger.info(f"Saving final optimized path to file '{args.output}'...")
        try:
            write_xyz(args.output, symbols, final_path)
            logger.info(f"Final path successfully saved to '{args.output}'.")
        except IOError as e:
            logger.error(f"Could not write final path file '{args.output}': {e}")
            sys.exit(1) # Critical error, exit.
        except Exception as e: # Catch other unexpected errors.
            logger.error(f"An unexpected error occurred while writing the final path to "
                         f"'{args.output}': {e}")
            sys.exit(1)
    else: # Optimization did not produce a usable path.
        logger.error(
            "Optimization did not produce a valid final path (path is None or empty). "
            "Output file will not be written."
        )
        sys.exit(1) # Critical error, exit.

    logger.info("Morse-geodesic interpolation/smoothing process finished successfully.")


if __name__ == "__main__":
    # This block executes if the script is run directly (e.g., `python -m morsegeodesic.morsegeom ...`).
    main()

