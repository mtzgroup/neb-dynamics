# MLP Geodesic Path Optimization

This repository contains the implementation for the paper: "Efficient Discovery of Transition States on Machine-Learned Potential Energy Surfaces via Geodesic Path Optimization".

This tool finds initial guess structures for transition states (TS) by optimizing a geodesic on a machine-learned (ML) potential energy surface (PES). This approach avoids expensive on-the-fly *ab initio* calculations during path optimization.

## Overview

This code optimizes a reaction path by minimizing the total path length on an ML PES, using the Fukui metric tensor. For elementary reactions, this geodesic path corresponds to the Minimum Energy Path (MEP), and its highest-energy point is a high-quality TS guess structure. This guess can be optimized to a stationary TS with P-RFO at the desired level of *ab initio* theory.

Key features:
1.  **Regularized Analytical Path Length:** The path is discretized, and the length of each segment (*s*<sub>k</sub>) is approximated with a regularized analytical formula based on a locally quadratic approximation to the energy.
2.  **MLP Backend:** All calculations use a pre-trained ML potential (MLP) like FAIRChem.
3.  **Two-Stage FIRE Optimization:**
    * **Stage 1 (Relaxation):** The initial path is rapidly relaxed to reduce total path length.
    * **Stage 2 (Refinement):** A climbing image is activated to precisely locate the highest-energy point, and new nodes are dynamically added to improve path resolution.
4.  **Energy-Based Node Spacing:** A penalty term in the loss function encourages nodes to be spaced evenly in energy, improving sampling throughout the path.

Consult the paper for furher details.

## Getting Started

### Install Dependencies and MLP backends.
The code has been implemented with Python (3.12), NumPy (2.2.6), SciPy (1.15.2) PyTorch (2.6.0), ASE (3.25.0), fairchem-core (2.2.0), mace-torch (0.3.13), click (8.2.1). These packages are needed for

-   **PyTorch:** For using the ML models. Install a CUDA-compatible version for GPU support.
-   **ASE (Atomic Simulation Environment):** Used for handling geometries and the FIRE optimizer.
-   **NumPy, SciPy, & Click:** For numerical operations and the CLI.
-   **FAIRChem and MACE:** MLP backends.

Most packages are readily installable with `pip` or `conda`, such as through:
```bash
pip install numpy scipy ase torch
```
#### FAIRChem
Follow the official installation instructions from the [FAIRChem repository](https://github.com/FAIR-Chem/fairchem).

#### MACE
Install `mace-torch` for MACE or Egret support:
```bash
pip install mace-torch
```

A copy of the virtual python environment used for code development is also provided (`environment.yml`), in case this is more convenient than installing packages.
This environment can be loaded via
```bash
conda env create -f environment.yml
```
Note that this particular environment cannot use mace.

## How to Run
The tool is executed from the command line.

### Command Syntax
```bash
python /path/to/codebase/cli.py [OPTIONS] INPUT_XYZ OUTPUT_XYZ
```

#### Positional Arguments:
-   `INPUT_XYZ`: Path to an XYZ file containing the initial reaction path (minimum 3 frames: reactant, guess, product). We reccomend this be obtained from Morse-geodesic (which is provided inside this repository), but any other guess can in principle be used.
-   `OUTPUT_XYZ`: Path to save the final optimized path as a multi-frame XYZ file.

#### Key Optional Arguments:
-   `--backend [mace|egret|fairchem]`: MLP backend to use. (Default: `fairchem`).
-   `--model-path FILE`: **Required.** Path to the pre-trained MLP model file.
-   `--device [cuda|cpu]`: Compute device. (Default: `cuda`).
-   `--climb / --no-climb`: Enable/disable the climbing image in Stage 2. (Default: `--climb`).
-   `-v, --verbose`: Enable detailed DEBUG-level logging.

See other optional arguments via
```bash
python /path/to/codebase/cli.py --help
```

### Example
```bash
python cli.py \
    --backend fairchem \
    --model-path /path/to/your/model.pt \
    --device cuda \
    initial_path.xyz \
    optimized_path.xyz
```

## File Descriptions
-   **`cli.py`**: The command-line interface for running the optimization.
-   **`optimizer.py`**: Contains the main `GeodesicOptimizer` class that orchestrates the optimization.
-   **`optimization_stages.py`**: Defines the optimization stages and includes the `_GeodesicCalculator` wrapper for compatibility with the ASE FIRE optimizer.
-   **`path_tools.py`**: The mathematical core, containing implementations for the geodesic segment length, gradient, tangent projection, and climbing image.
-   **`path_refinement.py`**: Handles the adaptive refinement logic for inserting new nodes into the path.
-   **`mlp_tools.py`**: Provides a unified interface for different MLP backends (MACE, FAIRChem).
-   **`utils.py`**: A collection of helpers, data classes (`OptimizerConfig`, `PathData`), I/O functions, and patches.
-   **`environment.yml`**: The virtual python environment used for development.
-   **`morse_geodesic`**: A copy of the Morse-geodesic code for generating initial paths. See README and tests inside for further details.
-   **`tests`**: A directory of test cases.
