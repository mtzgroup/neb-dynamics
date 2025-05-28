"""
File I/O utilities for reading and writing molecular geometries in XYZ format.

This module provides functions to parse XYZ files containing one or more
molecular structures and to write such structures back into the XYZ format.
It is tailored for the specific requirements of the MorseGeodesic interpolation package,
including handling of multi-frame files and ensuring atom symbol consistency.
"""

import numpy as np
from typing import List, Tuple, Union, Optional


def read_xyz(filename: str) -> Tuple[List[str], List[np.ndarray]]:
    """
    Read an XYZ file and return atom symbols and coordinates for each frame.

    This function expects a standard XYZ format where each frame consists of:
    1. A line with the number of atoms.
    2. A comment line.
    3. Subsequent lines with atom symbols and their X, Y, Z coordinates.

    It verifies that atom symbols are consistent across all frames if the file
    contains multiple frames (geometries).

    Args:
        filename (str): Path to the XYZ data file.

    Returns:
        Tuple[List[str], List[np.ndarray]]: A tuple containing:
            - atom_symbols (List[str]): A list of element symbols. These are taken
              from the first valid frame and are assumed to be consistent for all frames.
            - coords_list (List[np.ndarray]): A list of NumPy arrays. Each array
              has the shape (number_of_atoms, 3) and contains the Cartesian
              coordinates for one frame (geometry).

    Raises:
        FileNotFoundError: If the specified file `filename` does not exist.
        ValueError: If the file is empty, malformed (e.g., incorrect number of atoms,
                    non-numeric coordinates, insufficient columns per atom line), or if
                    atom symbols are inconsistent between frames in a multi-frame file.
        IOError: For other general file reading issues.
    """

    coords_list: List[np.ndarray] = []
    first_frame_atom_symbols: Optional[List[str]] = None
    frame_counter = 0  # To report frame number in case of errors

    try:
        with open(filename, 'r') as f:
            while True:
                frame_counter += 1

                # --- Read Number of Atoms ---
                line1 = f.readline()
                if not line1:  # End of file reached
                    break

                try:
                    num_atoms_current_frame = int(line1.strip())
                except ValueError:
                    raise ValueError(
                        f"Frame {frame_counter}: Expected an integer for the number of atoms on line 1, "
                        f"but got '{line1.strip()}'."
                    )

                if num_atoms_current_frame < 0:
                    raise ValueError(
                        f"Frame {frame_counter}: The number of atoms must be non-negative, "
                        f"but got {num_atoms_current_frame}."
                    )

                # --- Read Comment Line ---
                # The second line of each frame is a comment, which is read and discarded.
                _ = f.readline()

                # --- Read Atom Data for the Current Frame ---
                current_frame_symbols: List[str] = []
                current_frame_geom = np.zeros((num_atoms_current_frame, 3), dtype=float)

                for i in range(num_atoms_current_frame):
                    atom_line = f.readline()
                    if not atom_line:  # Premature end of file
                        raise ValueError(
                            f"Frame {frame_counter}: Unexpected end of file while reading atom {i + 1} "
                            f"of {num_atoms_current_frame} expected atoms."
                        )

                    parts = atom_line.split()
                    if len(parts) < 4:  # Expect symbol, X, Y, Z
                        raise ValueError(
                            f"Frame {frame_counter}, Atom line {i + 1}: Expected at least 4 columns "
                            f"(symbol, X, Y, Z), but got {len(parts)} parts: '{atom_line.strip()}'."
                        )

                    symbol = parts[0]
                    current_frame_symbols.append(symbol)

                    try:
                        # Convert coordinate string parts (parts[1], parts[2], parts[3]) to float.
                        current_frame_geom[i] = [float(x) for x in parts[1:4]]
                    except ValueError:
                        raise ValueError(
                            f"Frame {frame_counter}, Atom line {i + 1}: Cannot convert coordinates "
                            f"{parts[1:4]} to float values in line: '{atom_line.strip()}'."
                        )

                coords_list.append(current_frame_geom)

                # --- Check Atom Symbol Consistency Across Frames ---
                if first_frame_atom_symbols is None:
                    # This is the first valid frame; store its atom symbols.
                    first_frame_atom_symbols = current_frame_symbols
                elif current_frame_symbols != first_frame_atom_symbols:
                    # Atom symbols in the current frame differ from the first frame.
                    raise ValueError(
                        f"Atom symbols are inconsistent between frame 1 and frame {frame_counter}. "
                        "All frames in a multi-frame XYZ file must share the same atom order and symbols."
                    )

    except FileNotFoundError:
        # Re-raise FileNotFoundError specifically for clarity if the file isn't found.
        raise FileNotFoundError(f"The XYZ file was not found: {filename}")
    except Exception as e:
        # Handle other potential I/O or parsing errors not specifically ValueErrors.
        if not isinstance(e, ValueError):  # ValueErrors are already specific enough.
            raise IOError(f"An error occurred while reading the file {filename}: {e}") from e
        else:
            raise  # Re-raise the specific ValueError.

    # After attempting to read all frames, if coords_list is empty,
    # the file was effectively empty or malformed from the start.
    if not coords_list:
        raise ValueError(f"The XYZ file '{filename}' is empty or contains no valid geometry frames.")

    # This assignment is safe: if coords_list is non-empty,
    # first_frame_atom_symbols must have been set.
    final_atom_symbols = first_frame_atom_symbols if first_frame_atom_symbols is not None else []

    return final_atom_symbols, coords_list


def write_xyz(filename: str, atoms: List[str], coords: Union[List[np.ndarray], np.ndarray]) -> None:
    """
    Write atom symbols and coordinate data to an XYZ file.

    This function formats and writes molecular geometries, potentially across multiple
    frames, to a specified file in the standard XYZ format.

    Args:
        filename (str): Path to the output XYZ data file. The file will be overwritten
                        if it already exists.
        atoms (List[str]): A list of atom symbols (e.g., ['C', 'H', 'H', 'H']).
                           The order must correspond to the rows in the coordinate arrays.
        coords (Union[List[np.ndarray], np.ndarray]): Coordinate data.
            This can be:
            - A 3D NumPy array of shape (number_of_frames, number_of_atoms, 3).
            - A list of 2D NumPy arrays, where each 2D array is (number_of_atoms, 3).
            - A single 2D NumPy array (number_of_atoms, 3) for a single frame.

    Raises:
        IOError: If the file cannot be written (e.g., due to permissions).
        ValueError: If the coordinate data shape is inconsistent with the `atoms` list
                    (e.g., mismatch in the number of atoms), or if the input data
                    structure is otherwise invalid.
        RuntimeError: For other unexpected errors during the writing process.
    """

    num_atoms_from_symbols = len(atoms)
    coords_arr = np.asarray(coords, dtype=float)  # Ensure coords is a NumPy array of floats.
    processed_coords_list: List[np.ndarray]

    # --- Validate and Structure Input Coordinate Data ---
    if coords_arr.ndim == 3:
        # Input is a 3D array, assumed to be (nframes, natoms, 3).
        if coords_arr.shape[1] == num_atoms_from_symbols and coords_arr.shape[2] == 3:
            processed_coords_list = list(coords_arr)  # Convert to a list of 2D arrays for iteration.
        else:
            raise ValueError(
                f"For multi-frame input, coordinate shape {coords_arr.shape} is inconsistent with "
                f"the number of atoms ({num_atoms_from_symbols}) from the 'atoms' list or the "
                f"expected 3 spatial dimensions."
            )
    elif coords_arr.ndim == 2:
        # Input is a single 2D array, assumed to be (natoms, 3) for one frame.
        if coords_arr.shape[0] == num_atoms_from_symbols and coords_arr.shape[1] == 3:
            processed_coords_list = [coords_arr]  # Wrap in a list for uniform processing.
        else:
            raise ValueError(
                f"For single-frame input, coordinate shape {coords_arr.shape} is inconsistent with "
                f"the number of atoms ({num_atoms_from_symbols}) from the 'atoms' list or the "
                f"expected 3 spatial dimensions."
            )
    elif coords_arr.size == 0 and num_atoms_from_symbols > 0:
        # Atom symbols are provided, but no coordinate data.
        raise ValueError("Coordinate data is empty, but the 'atoms' list is not. Cannot write an XYZ file.")
    elif coords_arr.size == 0 and num_atoms_from_symbols == 0:
        # No atoms and no coordinates; this will write an "empty" XYZ frame (0 atoms).
        processed_coords_list = []
    else:
        # Other unsupported coordinate structures.
        raise ValueError(
            f"Invalid coordinate data structure. Expected a 3D NumPy array (nframes, natoms, 3), "
            f"a list of 2D arrays, or a single 2D array (natoms, 3). "
            f"Received an array with ndim={coords_arr.ndim} and shape={coords_arr.shape}."
        )

    try:
        with open(filename, 'w') as f:
            if not processed_coords_list and num_atoms_from_symbols == 0:
                # --- Handle Special Case: System with 0 Atoms ---
                # Write a standard XYZ representation for an empty system.
                f.write("0\n")
                f.write("Empty frame, 0 atoms\n")  # Standard comment for an empty frame.
                return  # Nothing more to write.

            # --- Write Each Frame to the File ---
            for frame_idx, frame_coords_np in enumerate(processed_coords_list):
                # Write the number of atoms for the current frame.
                f.write(f"{num_atoms_from_symbols}\n")

                # Write a standard comment line for the frame.
                f.write(f"Frame {frame_idx + 1}\n")

                # Write atom symbol and coordinates for each atom in the frame.
                for atom_idx in range(num_atoms_from_symbols):
                    atom_symbol = atoms[atom_idx]
                    x, y, z = frame_coords_np[atom_idx]
                    # Format: Symbol (left-aligned, 3 chars wide), X, Y, Z (float, specific precision).
                    f.write(f" {atom_symbol:<3s} {x:21.12f} {y:21.12f} {z:21.12f}\n")

    except IOError as e:  # Catch errors related to file writing operations (e.g., permissions).
        raise IOError(f"Could not write the XYZ file '{filename}': {e}") from e
    except Exception as e:
        # Catch any other unexpected errors during writing.
        if not isinstance(e, ValueError):  # Avoid re-wrapping ValueErrors already raised.
            raise RuntimeError(f"An unexpected error occurred during XYZ file writing: {e}") from e
        else:  # Re-raise specific ValueErrors.
            raise

