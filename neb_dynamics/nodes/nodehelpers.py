import numpy as np
from neb_dynamics.nodes.node import Node
from neb_dynamics.qcio_structure_helpers import split_structure_into_frags
from neb_dynamics.geodesic_interpolation.coord_utils import align_geom


def is_identical(self: Node, other: Node,
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
        conditions = [_is_connectivity_identical(self, other),
                      _is_conformer_identical(self, other,
                                              global_rmsd_cutoff=global_rmsd_cutoff,
                                              fragment_rmsd_cutoff=fragment_rmsd_cutoff,
                                              kcal_mol_cutoff=kcal_mol_cutoff)]
    else:
        conditions = [_is_conformer_identical(self, other,
                                              global_rmsd_cutoff=global_rmsd_cutoff,
                                              fragment_rmsd_cutoff=fragment_rmsd_cutoff,
                                              kcal_mol_cutoff=kcal_mol_cutoff)]

    return all(conditions)


def _is_conformer_identical(self: Node, other: Node,
                            *,
                            global_rmsd_cutoff: float = 20.0,
                            fragment_rmsd_cutoff: float = 0.5,
                            kcal_mol_cutoff: float = 1.0) -> bool:

    global_dist, aligned_geometry = align_geom(
        refgeom=other.structure.geometry, geom=self.structure.geometry)
    per_frag_dists = []
    if self.has_molecular_graph:
        if not _is_connectivity_identical(self, other):
            return False

        self_frags = split_structure_into_frags(self.structure)
        other_frags = split_structure_into_frags(other.structure)
        for frag_self, frag_other in zip(self_frags, other_frags):
            frag_dist, _ = align_geom(
                refgeom=frag_self.geometry, geom=frag_other.geometry)
            per_frag_dists.append(frag_dist)
    else:
        per_frag_dists.append(global_dist)

    en_delta = np.abs((self.energy - other.energy) * 627.5)

    global_rmsd_identical = global_dist <= global_rmsd_cutoff
    fragment_rmsd_identical = max(
        per_frag_dists) <= fragment_rmsd_cutoff
    rmsd_identical = global_rmsd_identical and fragment_rmsd_identical
    energies_identical = en_delta < kcal_mol_cutoff

    if rmsd_identical and energies_identical:
        return True
    else:
        return False


def _is_connectivity_identical(self: Node, other: Node) -> bool:
    """
    checks graphs of both nodes and returns whether they are isomorphic
    to each other.
    """
    # print("different graphs")
    connectivity_identical = self.graph.remove_Hs().is_bond_isomorphic_to(
        other.graph.remove_Hs()
    )
    return connectivity_identical
