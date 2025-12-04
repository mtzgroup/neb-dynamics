# Morse-Geodesic Path Interpolation

This repository contains an implementation of the method described in the paper: "Geodesic interpolation for reaction pathways" by Xiaolei Zhu, Keiran C. Thompson, and Todd J. Martínez, *J. Chem. Phys.* 150, 164103 (2019) [DOI: 10.1063/1.5090303](https://doi.org/10.1063/1.5090303).

This tool generates a physically reasonable, smooth path between two molecular geometries (e.g., a reactant and a product). It is designed to create high-quality initial guesses for more expensive reaction path optimization methods like MLP geodesic construction, but also others like the Nudged Elastic Band (NEB).

This implementation is based on the [github repository](https://github.com/mtzgroup/geodesic-interpolate) associated with the original codebase. The principal difference lies in the use of sparse linear algebra for the least-squares optimization, which *significantly* accelerates the code. As a corollary, it is no longer necessary to use the `--sweep` mode of the original codebase, and support for that feature has been discontinued. The default convergence tolerance has also been tightened and the maximum number of optimization iterations increased.  

## Overview

This code formulates the path interpolation problem as a search for a geodesic (the shortest path) on a Riemannian manifold. The metric of this manifold is defined by a set of redundant internal coordinates (interatomic distances) that are scaled using a Morse-like potential function. This ensures that the generated path avoids unphysical structures like atomic collisions.

Key features:
1.  **Geodesic Formulation:** The path is optimized by minimizing its length in a Morse potential based metric space, leading to paths that avoid close contact.
2.  **Efficient & Sparse:** The use of `scipy.optimize.least_squares` with an analytically computed sparse Jacobian makes the optimization process very fast, even for large molecular systems.
3.  **Path Redistribution:** The tool can intelligently add or remove images from an initial guess path to achieve a desired, evenly-spaced number of frames before the final smoothing.

## Getting Started

### Install Dependencies
The code is designed to work with Python 3.12+ and requires the following packages. The versions listed are those used during development.

-   Python (3.12)
-   NumPy (2.2.6)
-   SciPy (1.15.2)
-   Click (8.2.1)

These packages can be installed using `pip` or `conda`:
```bash
pip install numpy scipy click
```

A copy of the development `conda` environment is also provided in `environment.yml`. This can be used to create an identical environment via:
```bash
conda env create -f environment.yml
```

## How to Run
The tool is executed from the command line via `cli.py`.

### Command Syntax
```bash
python /path/to/morse_geodesic/cli.py [OPTIONS] INPUT_XYZ OUTPUT_XYZ
```

#### Positional Arguments:
-   `INPUT_XYZ`: Path to an XYZ file containing the initial path (must have at least two frames: reactant and product).
-   `OUTPUT_XYZ`: Path where the final, smoothed path will be saved as a multi-frame XYZ file.

#### Key Optional Arguments:
-   `--nimages INT`: Target number of images in the final path. (Default: 17, odd number ideal for symmetric paths).
-   `--tol FLOAT`: Convergence tolerance for the optimization. (Default: 1e-5).
-   `--maxiter INT`: Maximum number of optimizer iterations. (Default: 200).

See all optional arguments by running:
```bash
python /path/to/morse_geodesic/cli.py --help
```

### Example
```bash
python cli.py \
    --nimages 23 \
    --tol 1e-4 \
    reactant_and_product.xyz \
    interpolated_path.xyz
```

## File Descriptions
-   **`cli.py`**: The command-line interface for running the interpolation. It parses user arguments and orchestrates the overall workflow.
-   **`morsegeodesic.py`**: Contains the core `MorseGeodesic` class. This class manages the path optimization by setting up and solving the least-squares problem to minimize the geodesic path length.
-   **`interpolation.py`**: Provides functions for path pre-processing, including the `redistribute` function to adjust the number of images in a path before smoothing.
-   **`coord_utils.py`**: A collection of utility functions for handling molecular coordinates. This includes path alignment (using the Kabsch algorithm), generation of internal coordinate lists, and the core `compute_wij` function that calculates scaled internal coordinates and their sparse Cartesian gradients.
-   **`fileio.py`**: Contains functions for reading and writing molecular structures in XYZ format.
-   **`config.py`**: Centralizes all default parameters for the CLI and internal algorithms.
-   **`tests`**: A directory of test cases.
