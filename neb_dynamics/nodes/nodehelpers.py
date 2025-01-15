import numpy as np
from neb_dynamics.nodes.node import Node, XYNode
from neb_dynamics.qcio_structure_helpers import split_structure_into_frags
from neb_dynamics.geodesic_interpolation.coord_utils import align_geom
from neb_dynamics.molecule import Molecule
from neb_dynamics.qcio_structure_helpers import molecule_to_structure
import traceback
from rxnmapper import RXNMapper


def is_identical(
    self: Node,
    other: Node,
    *,
    global_rmsd_cutoff: float = 20.0,
    fragment_rmsd_cutoff: float = 1.0,
    kcal_mol_cutoff: float = 1.0,
) -> bool:
    """
    computes whether two nodes are identical.
    if nodes have computable molecular graphs, will check that
    they are isomorphic to each other + that their distances are
    within a threshold (default: 1.0 Bohr) and their enegies
    are within a windos (default: 1.0 kcal/mol).
    otherwise, will only check distances.
    """
    if self.has_molecular_graph:
        assert (
            other.has_molecular_graph
        ), "Both node objects must have computable molecular graphs."
        conditions = [
            _is_connectivity_identical(self, other),
            _is_conformer_identical(
                self,
                other,
                global_rmsd_cutoff=global_rmsd_cutoff,
                fragment_rmsd_cutoff=fragment_rmsd_cutoff,
                kcal_mol_cutoff=kcal_mol_cutoff,
            ),
        ]
    else:
        conditions = [
            _is_conformer_identical(
                self,
                other,
                global_rmsd_cutoff=global_rmsd_cutoff,
                fragment_rmsd_cutoff=fragment_rmsd_cutoff,
                kcal_mol_cutoff=kcal_mol_cutoff,
            )
        ]

    return all(conditions)


def _is_conformer_identical(
    self: Node,
    other: Node,
    *,
    global_rmsd_cutoff: float = 20.0,
    fragment_rmsd_cutoff: float = 0.5,
    kcal_mol_cutoff: float = 1.0,
) -> bool:

    print("\n\tVerifying if two geometries are identical.")
    if isinstance(self, XYNode):
        global_dist = np.linalg.norm(other.coords - self.coords)
    else:
        global_dist, aligned_geometry = align_geom(
            refgeom=other.coords, geom=self.coords
        )

    per_frag_dists = []
    if self.has_molecular_graph:
        if not _is_connectivity_identical(self, other):
            print("\t\tGraphs differed. Not identical.")
            return False
        print("\t\tGraphs identical. Checking distances...")
        self_frags = split_structure_into_frags(self.structure)
        other_frags = split_structure_into_frags(other.structure)
        for frag_self, frag_other in zip(self_frags, other_frags):
            frag_dist, _ = align_geom(
                refgeom=frag_self.geometry, geom=frag_other.geometry
            )
            per_frag_dists.append(frag_dist)
            print(f"\t\t\tfrag dist: {frag_dist}")
    else:
        per_frag_dists.append(global_dist)

    en_delta = np.abs((self.energy - other.energy) * 627.5)

    global_rmsd_identical = global_dist <= global_rmsd_cutoff
    fragment_rmsd_identical = max(per_frag_dists) <= fragment_rmsd_cutoff
    rmsd_identical = global_rmsd_identical and fragment_rmsd_identical
    energies_identical = en_delta < kcal_mol_cutoff

    print(f"\t\t{rmsd_identical=} {energies_identical=} {en_delta=}")
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
    try:
        smi1 = self.structure.to_smiles()
        smi2 = other.structure.to_smiles()
        stereochem_identical = smi1 == smi2
    except Exception as e:
        print(traceback.format_exc())
        print("Constructing smiles failed. Pretending this check succeeded.")
        stereochem_identical = True
    print(f"\t\tGraphs isomorphic to each other: {connectivity_identical}")
    print(f"\t\tStereochemical smiles identical: {stereochem_identical}")
    if not stereochem_identical:
        print(f"\t{smi1} || {smi2}")

    return connectivity_identical and stereochem_identical


def update_node_cache(node_list, results):
    """
    inplace update of cached results
    """
    for node, result in zip(node_list, results):
        node._cached_result = result
        node._cached_energy = result.results.energy
        node._cached_gradient = result.results.gradient


def create_pairs_from_smiles(smi1: str, smi2: str, spinmult=1):
    rxnsmi = f"{smi1}>>{smi2}"
    rxn_mapper = RXNMapper()
    rxn = [rxnsmi]
    result = rxn_mapper.get_attention_guided_atom_maps(rxn)[0]
    mapped_smi = result["mapped_rxn"]
    r_smi, p_smi = mapped_smi.split(">>")
    print(r_smi, p_smi)
    r = Molecule.from_mapped_smiles(r_smi)
    p = Molecule.from_mapped_smiles(p_smi)

    td_r, td_p = (
        molecule_to_structure(r, charge=r.charge, spinmult=spinmult),
        molecule_to_structure(p, charge=p.charge, spinmult=spinmult),
    )
    return td_r, td_p


def displace_by_dr(node: Node, displacement: np.array, dr: float = 0.1) -> Node:
    """returns a new node object that has been displaced along the input 'displacement' vector by 'dr'.

    Args:
        node (Node): Node to displace
        displacement (np.array): vector along which to displace
        dr (float, optional): Magnitude of displacement vector. Defaults to 0.1.
    """
    displacement = displacement / np.linalg.norm(displacement)
    new_coords = node.coords + dr*displacement
    return node.update_coords(new_coords)
