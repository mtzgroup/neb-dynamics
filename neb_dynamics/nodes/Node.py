from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from neb_dynamics.helper_functions import RMSD
from neb_dynamics.geodesic_interpolation.coord_utils import align_geom
from qcio.models.structure import Structure


@dataclass
class Node:
    structure: Structure

    converged: bool = False
    has_molecular_graph: bool = False
    symbols: np.array = None

    def __eq__(self, other: None) -> bool:
        return self.is_identical(other)

    def is_identical(self, other: Node,
                     *,
                     global_rmsd_cutoff: float = 20.0,
                     fragment_rmsd_cutoff: float = 0.5,
                     kcal_mol_cutoff: float = 1.0) -> bool:
        """
        computes whether two nodes are identical.
        if nodes have computable molecular graphs, will check that
        they are isomorphic to each other + that their distances are
        within a threshold (default: 0.5 Anstroms) and their enegies
        are within a windos (default: 1.0 kcal/mol).
        otherwise, will only check distances.
        """
        if self.has_molecular_graph:
            assert other.has_molecular_graph, "Both node objects must have computable molecular graphs."
            conditions = [self._is_connectivity_identical(
                other), self._is_conformer_identical(other)]
        else:
            conditions = [self._is_conformer_identical(other,
                                                       global_rmsd_cutoff=global_rmsd_cutoff,
                                                       fragment_rmsd_cutoff=fragment_rmsd_cutoff,
                                                       kcal_mol_cutoff=kcal_mol_cutoff)]

        return all(conditions)

    def _is_conformer_identical(self, other: Node,
                                *,
                                global_rmsd_cutoff: float = 20.0,
                                fragment_rmsd_cutoff: float = 0.5,
                                kcal_mol_cutoff: float = 1.0) -> bool:

        global_dist, aligned_geometry = align_geom(refgeom=other.structure.geometry, geom=self.structure.geometry)
        per_frag_dists = []
        if self.has_molecular_graph:
            self_frags = split_structure_into_frags(self.structure)
            other_frags = split_structure_into_frags(split_td_into_frags)
            for frag_self, frag_other in zip(self_frags, other_frags):
                aligned_frag_self = frag_self.align_to_td(frag_other)
                frag_dist = RMSD(aligned_frag_self.coords,
                                    frag_other.coords)[0]
                per_frag_dists.append(frag_dist)

        en_delta = np.abs((self.energy - other.energy) * 627.5)

        global_rmsd_identical = global_dist <= global_rmsd_cutoff
        fragment_rmsd_identical = max(
            per_frag_dists) <= fragment_rmsd_cutoff
        rmsd_identical = global_rmsd_identical and fragment_rmsd_identical
        energies_identical = en_delta < kcal_mol_cutoff
        # print(f"\nbarrier_to_conformer_rearr: {barrier} kcal/mol\n{en_delta=}\n")

        if rmsd_identical and energies_identical:  # and barrier_accessible:
            return True
        else:
            return False
