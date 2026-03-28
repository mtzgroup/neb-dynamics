from __future__ import annotations

import contextlib
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from openbabel import openbabel, pybel
from qcio import Structure

from neb_dynamics.constants import ANGSTROM_TO_BOHR
from neb_dynamics.elements import symbol_to_atomic_number
from neb_dynamics.molecule import Molecule
from neb_dynamics.nodes.node import StructureNode


def _retropaths_repo() -> Path:
    return Path(__file__).resolve().parents[3] / "retropaths"


def prepare_retropaths_imports() -> None:
    repo = _retropaths_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    if "xtb" not in sys.modules:
        xtb_stub = types.ModuleType("xtb")
        xtb_interface_stub = types.ModuleType("xtb.interface")
        xtb_interface_stub.Calculator = object
        xtb_interface_stub.XTBException = Exception
        xtb_utils_stub = types.ModuleType("xtb.utils")
        xtb_utils_stub.get_method = lambda *_args, **_kwargs: None
        xtb_libxtb_stub = types.ModuleType("xtb.libxtb")
        xtb_libxtb_stub.VERBOSITY_MUTED = 0
        xtb_ase_stub = types.ModuleType("xtb.ase")
        xtb_ase_calculator_stub = types.ModuleType("xtb.ase.calculator")
        xtb_ase_calculator_stub.XTB = object
        sys.modules["xtb"] = xtb_stub
        sys.modules["xtb.interface"] = xtb_interface_stub
        sys.modules["xtb.utils"] = xtb_utils_stub
        sys.modules["xtb.libxtb"] = xtb_libxtb_stub
        sys.modules["xtb.ase"] = xtb_ase_stub
        sys.modules["xtb.ase.calculator"] = xtb_ase_calculator_stub

    if "cairosvg" not in sys.modules:
        cairosvg_stub = types.ModuleType("cairosvg")
        cairosvg_stub.svg2png = lambda *_args, **_kwargs: None
        sys.modules["cairosvg"] = cairosvg_stub

    if "imgkit" not in sys.modules:
        imgkit_stub = types.ModuleType("imgkit")
        imgkit_stub.from_string = lambda *_args, **_kwargs: None
        sys.modules["imgkit"] = imgkit_stub


@lru_cache(maxsize=1)
def _reaction_library() -> Any:
    prepare_retropaths_imports()
    import retropaths.helper_functions as hf

    return hf.pload(_retropaths_repo() / "data" / "reactions.p")


@lru_cache(maxsize=1)
def _retropaths_molecule_class() -> Any:
    prepare_retropaths_imports()
    from retropaths.molecules.molecule import Molecule as RetropathsMolecule

    return RetropathsMolecule


def _molecule_key(molecule: Molecule | None) -> str:
    if molecule is None:
        return ""
    with contextlib.suppress(Exception):
        return str(molecule.smiles_from_multiple_molecules())
    with contextlib.suppress(Exception):
        return str(molecule.force_smiles())
    return ""


def _coerce_retropaths_molecule(molecule: Any, MoleculeCls: Any) -> Any | None:
    if molecule is None:
        return None
    if isinstance(molecule, MoleculeCls):
        return molecule.copy()
    smiles = _molecule_key(molecule)
    if smiles:
        with contextlib.suppress(Exception):
            return MoleculeCls.from_smiles(smiles)
    return None


def _dense_retropaths_molecule(molecule: Any, MoleculeCls: Any) -> Any | None:
    if molecule is None:
        return None
    dense = MoleculeCls()
    old_to_new = {
        old_index: new_index for new_index, old_index in enumerate(sorted(molecule.nodes))
    }
    for old_index in sorted(molecule.nodes):
        dense.add_node(old_to_new[old_index], **dict(molecule.nodes[old_index]))
    for atom1, atom2, attrs in molecule.edges(data=True):
        dense.add_edge(old_to_new[atom1], old_to_new[atom2], **dict(attrs))
    if hasattr(dense, "set_neighbors"):
        dense.set_neighbors()
    return dense


def _bond_order_number(label: str) -> float:
    return {
        "single": 1,
        "double": 2,
        "triple": 3,
        "aromatic": 1.5,
    }[str(label)]


def _retropaths_tdstructure_from_node(node: StructureNode, dense_graph: Any, TDStructure: Any) -> Any:
    obmol = openbabel.OBMol()
    obmol.SetTotalCharge(int(node.structure.charge))
    obmol.SetTotalSpinMultiplicity(int(node.structure.multiplicity))
    coords_angstrom = np.asarray(node.coords, dtype=float) / ANGSTROM_TO_BOHR

    for atom_index, graph_index in enumerate(sorted(dense_graph.nodes)):
        atom = openbabel.OBAtom()
        atom.SetVector(*coords_angstrom[atom_index])
        atom.SetAtomicNum(int(symbol_to_atomic_number(node.structure.symbols[atom_index])))
        atom.SetFormalCharge(int(dense_graph.nodes[graph_index].get("charge", 0)))
        obmol.AddAtom(atom)

    for atom1, atom2, attrs in dense_graph.edges(data=True):
        bond = openbabel.OBBond()
        bond.SetBegin(obmol.GetAtom(int(atom1) + 1))
        bond.SetEnd(obmol.GetAtom(int(atom2) + 1))
        bond_order = _bond_order_number(attrs["bond_order"])
        if bond_order == 1.5:
            bond.SetAromatic(True)
            bond.SetBondOrder(4)
        else:
            bond.SetBondOrder(int(bond_order))
        obmol.AddBond(bond)

    return TDStructure.from_obmol(obmol)


def _structure_node_from_retropaths_tdstructure(
    tdstructure: Any,
    *,
    target_graph: Molecule,
    charge: int,
    spinmult: int,
) -> StructureNode | None:
    coords_bohr = np.asarray(tdstructure.coords, dtype=float) * ANGSTROM_TO_BOHR
    if coords_bohr.shape[0] != len(target_graph.nodes):
        return None
    structure = Structure(
        geometry=coords_bohr,
        symbols=list(tdstructure.symbols),
        charge=charge,
        multiplicity=spinmult,
    )
    return StructureNode(
        structure=structure,
        graph=target_graph.copy(),
        converged=False,
    )


def _mm_optimize_tdstructure(
    tdstructure: Any,
    *,
    method: str = "uff",
    steps: int = 2000,
    energy_diff_threshold_kcalmol: float | None = 5.0,
    block_size: int = 10,
) -> Any:
    ff = openbabel.OBForceField.FindForceField(str(method).upper())
    if ff is None or not ff.Setup(tdstructure.molecule_obmol):
        pybel_mol = pybel.Molecule(tdstructure.molecule_obmol)
        pybel_mol.localopt(method, steps=steps)
        tdstructure.molecule_obmol = pybel_mol.OBMol
        return tdstructure

    max_steps = max(int(steps), 1)
    step_block = max(min(int(block_size), max_steps), 1)

    if energy_diff_threshold_kcalmol is None:
        ff.ConjugateGradients(max_steps)
        ff.GetCoordinates(tdstructure.molecule_obmol)
        return tdstructure

    ff.ConjugateGradientsInitialize(max_steps, 1.0e-6)
    previous_energy = float(ff.Energy())
    steps_taken = 0

    while steps_taken < max_steps:
        current_block = min(step_block, max_steps - steps_taken)
        ff.ConjugateGradientsTakeNSteps(current_block)
        steps_taken += current_block
        current_energy = float(ff.Energy())
        ff.GetCoordinates(tdstructure.molecule_obmol)
        if abs(previous_energy - current_energy) < float(energy_diff_threshold_kcalmol):
            break
        previous_energy = current_energy

    return tdstructure


def _endpoint_matches_template_reactants(node: StructureNode, rxn: Any, MoleculeCls: Any) -> bool:
    graph = getattr(node, "graph", None)
    dense_graph = _dense_retropaths_molecule(
        _coerce_retropaths_molecule(graph, MoleculeCls),
        MoleculeCls,
    )
    if dense_graph is None:
        return False
    with contextlib.suppress(Exception):
        return len(rxn.reactants.get_subgraph_isomorphisms_of(dense_graph)) > 0
    return False


def _pseudoalign_reactant_node(node: StructureNode, rxn: Any, MoleculeCls: Any) -> StructureNode | None:
    graph = getattr(node, "graph", None)
    dense_graph = _dense_retropaths_molecule(
        _coerce_retropaths_molecule(graph, MoleculeCls),
        MoleculeCls,
    )
    if dense_graph is None:
        return None

    prepare_retropaths_imports()
    from retropaths.abinitio.tdstructure import TDStructure

    source_tds = _retropaths_tdstructure_from_node(node, dense_graph, TDStructure)
    changes3d_list = source_tds.get_changes_in_3d(rxn)
    if not changes3d_list:
        return None
    changes3d = changes3d_list[0]
    if not getattr(changes3d, "forming", None):
        return None

    staged = source_tds.copy()
    staged.add_bonds_from_changes3d(changes3d)
    _mm_optimize_tdstructure(
        staged,
        method="uff",
        steps=2000,
        energy_diff_threshold_kcalmol=5.0,
        block_size=10,
    )
    staged.delete_formed_from_changes3d(changes3d)
    _mm_optimize_tdstructure(
        staged,
        method="uff",
        steps=2000,
        energy_diff_threshold_kcalmol=5.0,
        block_size=10,
    )

    return _structure_node_from_retropaths_tdstructure(
        staged,
        target_graph=graph,
        charge=int(node.structure.charge),
        spinmult=int(node.structure.multiplicity),
    )


def pseudoalign_reaction_pair(
    start: StructureNode,
    end: StructureNode,
    reaction_name: str | None,
) -> tuple[StructureNode, StructureNode]:
    reaction_name = str(reaction_name or "").strip()
    if not reaction_name:
        return start, end

    with contextlib.suppress(Exception):
        rxn = _reaction_library()[reaction_name]
        MoleculeCls = _retropaths_molecule_class()

        if _endpoint_matches_template_reactants(start, rxn, MoleculeCls):
            pseudoaligned = _pseudoalign_reactant_node(start, rxn, MoleculeCls)
            if pseudoaligned is not None:
                start = pseudoaligned
        elif _endpoint_matches_template_reactants(end, rxn, MoleculeCls):
            pseudoaligned = _pseudoalign_reactant_node(end, rxn, MoleculeCls)
            if pseudoaligned is not None:
                end = pseudoaligned

    return start, end
