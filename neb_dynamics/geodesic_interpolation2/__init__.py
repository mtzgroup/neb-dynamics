"""
Morse-Geodesic Interpolation Package.

This package provides primary classes and functions for path interpolation
and Morse-geodesic smoothing of molecular reaction paths. It allows users
to generate smooth and physically plausible pathways between molecular
configurations, which is crucial for studying chemical reactions and
conformational changes.

The core class is `MorseGeodesic`, which performs the path optimization.
Helper functions for path manipulation (`redistribute`, `mid_point`),
file I/O (`read_xyz`, `write_xyz`), and coordinate utilities (`align_path`,
`align_geom`) are also provided.
"""

# Make key components available at the package level for convenient import
# by users of this package. This defines the package's public API.
from .morsegeodesic import MorseGeodesic
from .interpolation import redistribute, mid_point
from .fileio import read_xyz, write_xyz
from .coord_utils import align_path, align_geom

# Define the public API of the package when 'from package import *' is used.
# This explicitly lists all names that will be imported, providing a clear
# and controlled interface for users.
__all__ = [
    'MorseGeodesic',    # Core class for Morse-geodesic path smoothing and optimization.
    'redistribute',     # Function to adjust the number of images (geometries) in a path.
    'mid_point',        # Function to find an optimal midpoint geometry between two given geometries.
    'read_xyz',         # Utility to read molecular geometries from standard XYZ files.
    'write_xyz',        # Utility to write molecular geometries to standard XYZ files.
    'align_path',       # Utility to align a sequence of geometries (an entire path) to minimize RMSD.
    'align_geom'        # Utility to align one geometry to a reference geometry using Kabsch algorithm.
]

