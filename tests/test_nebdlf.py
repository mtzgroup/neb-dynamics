from types import SimpleNamespace

import numpy as np
from qcio import Structure
from qcio.models.inputs import ProgramArgs

from neb_dynamics.chain import Chain
from neb_dynamics.constants import ANGSTROM_TO_BOHR
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.pathminimizers.nebdlf import DLFindNEB


def _structure(x_angstrom: float) -> Structure:
    return Structure(
        geometry=np.array(
            [[x_angstrom, 0.0, 0.0], [x_angstrom, 0.0, 0.74]],
            dtype=float,
        )
        * ANGSTROM_TO_BOHR,
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def _chain(xs: list[float]) -> Chain:
    nodes = [StructureNode(structure=_structure(x)) for x in xs]
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def _xyz_from_structures(structures: list[Structure]) -> str:
    return "".join(struct.to_xyz().rstrip() + "\n" for struct in structures)


def _xyz_from_structures_with_blank_separators(structures: list[Structure]) -> str:
    chunks = [struct.to_xyz().strip() for struct in structures]
    return "\n\n\n".join(chunks) + "\n"


def _terachem_qcop_engine() -> QCOPEngine:
    return QCOPEngine(
        program="terachem",
        program_args=ProgramArgs(
            model={"method": "ub3lyp", "basis": "3-21g"},
            keywords={},
        ),
        compute_program="qcop",
    )


def test_nebdlf_parses_nebpath_and_nebinfo(monkeypatch):
    engine = _terachem_qcop_engine()
    input_chain = _chain([0.0, 1.0])
    output_structures = [_structure(0.0), _structure(0.5), _structure(1.0)]
    nebpath_xyz = _xyz_from_structures(output_structures)
    nebinfo_text = "\n".join(
        [
            "0 0 -1.000000",
            "0 1 -0.900000",
            "0 2 -0.800000",
            "1 0 -1.100000",
            "1 1 -0.950000",
            "1 2 -0.850000",
        ]
    )

    fake_output = SimpleNamespace(
        files={
            "scr/nebpath.xyz": nebpath_xyz,
            "scr/nebinfo": nebinfo_text,
        }
    )

    monkeypatch.setattr(
        engine,
        "compute_func",
        lambda *args, **kwargs: fake_output,
    )

    minimizer = DLFindNEB(
        initial_chain=input_chain,
        engine=engine,
        parameters={"do_elem_step_checks": False},
    )
    elem_step = minimizer.optimize_chain()

    assert elem_step.is_elem_step is True
    assert minimizer.optimized is not None
    assert len(minimizer.optimized) == 3
    np.testing.assert_allclose(
        minimizer.optimized.energies,
        np.array([-1.1, -0.95, -0.85]),
    )


def test_nebdlf_falls_back_to_engine_energies_when_nebinfo_missing(monkeypatch):
    engine = _terachem_qcop_engine()
    input_chain = _chain([0.0, 1.0])
    output_structures = [_structure(0.0), _structure(0.5), _structure(1.0)]
    nebpath_xyz = _xyz_from_structures(output_structures)
    fake_output = SimpleNamespace(files={"scr/nebpath.xyz": nebpath_xyz})

    monkeypatch.setattr(
        engine,
        "compute_func",
        lambda *args, **kwargs: fake_output,
    )

    call_counter = {"count": 0}

    def _fake_compute_energies(chain: Chain):
        call_counter["count"] += 1
        energies = np.array([-1.0, -0.9, -0.8], dtype=float)
        for node, ene in zip(chain, energies):
            node._cached_energy = float(ene)
        return energies

    monkeypatch.setattr(engine, "compute_energies", _fake_compute_energies)

    minimizer = DLFindNEB(
        initial_chain=input_chain,
        engine=engine,
        parameters={"do_elem_step_checks": False},
    )
    minimizer.optimize_chain()

    assert call_counter["count"] == 1
    assert minimizer.grad_calls_made == 3
    np.testing.assert_allclose(minimizer.optimized.energies, np.array([-1.0, -0.9, -0.8]))


def test_nebdlf_parses_nebpath_with_leading_blank_lines(monkeypatch):
    engine = _terachem_qcop_engine()
    input_chain = _chain([0.0, 1.0])
    output_structures = [_structure(0.0), _structure(0.5), _structure(1.0)]
    nebpath_xyz = "\n\n" + _xyz_from_structures_with_blank_separators(output_structures)
    fake_output = SimpleNamespace(files={"scr/nebpath.xyz": nebpath_xyz})

    monkeypatch.setattr(
        engine,
        "compute_func",
        lambda *args, **kwargs: fake_output,
    )

    def _fake_compute_energies(chain: Chain):
        energies = np.array([-1.0, -0.9, -0.8], dtype=float)
        for node, ene in zip(chain, energies):
            node._cached_energy = float(ene)
        return energies

    monkeypatch.setattr(engine, "compute_energies", _fake_compute_energies)

    minimizer = DLFindNEB(
        initial_chain=input_chain,
        engine=engine,
        parameters={"do_elem_step_checks": False},
    )
    minimizer.optimize_chain()

    assert minimizer.optimized is not None
    assert len(minimizer.optimized) == 3


def test_nebdlf_parses_path_geometry_fallback(monkeypatch):
    engine = _terachem_qcop_engine()
    input_chain = _chain([0.0, 0.5])
    path_geometry = """
 ***  Molecular Geometry (ANGS) ***
Type         X              Y              Z            Mass
   H   0.0000000000   0.0000000000   0.0000000000   1.0078250370
   H   0.0000000000   0.0000000000   0.7399999993   1.0078250370

 ***  Molecular Geometry (ANGS) ***
Type         X              Y              Z            Mass
   H   0.5000000000   0.0000000000   0.0000000000   1.0078250370
   H   0.5000000000   0.0000000000   0.7399999993   1.0078250370
"""
    fake_output = SimpleNamespace(files={"scr.path/path.geometry": path_geometry})

    monkeypatch.setattr(
        engine,
        "compute_func",
        lambda *args, **kwargs: fake_output,
    )

    def _fake_compute_energies(chain: Chain):
        energies = np.array([-1.0, -0.8], dtype=float)
        for node, ene in zip(chain, energies):
            node._cached_energy = float(ene)
        return energies

    monkeypatch.setattr(engine, "compute_energies", _fake_compute_energies)

    minimizer = DLFindNEB(
        initial_chain=input_chain,
        engine=engine,
        parameters={"do_elem_step_checks": False},
    )
    minimizer.optimize_chain()

    assert minimizer.optimized is not None
    assert len(minimizer.optimized) == 2


def test_nebdlf_reconstructs_path_from_neb_image_files(monkeypatch):
    engine = _terachem_qcop_engine()
    input_chain = _chain([0.0, 1.0, 2.0])
    neb_1_xyz = _xyz_from_structures([_structure(1.0)])
    optim_xyz = _xyz_from_structures([_structure(2.0)])
    fake_output = SimpleNamespace(
        files={
            "scr.path/neb_1.xyz": neb_1_xyz,
            "scr.path/optim.xyz": optim_xyz,
        }
    )

    monkeypatch.setattr(
        engine,
        "compute_func",
        lambda *args, **kwargs: fake_output,
    )

    def _fake_compute_energies(chain: Chain):
        energies = np.array([-1.0, -0.9, -0.8], dtype=float)
        for node, ene in zip(chain, energies):
            node._cached_energy = float(ene)
        return energies

    monkeypatch.setattr(engine, "compute_energies", _fake_compute_energies)

    minimizer = DLFindNEB(
        initial_chain=input_chain,
        engine=engine,
        parameters={"do_elem_step_checks": False},
    )
    minimizer.optimize_chain()

    assert minimizer.optimized is not None
    assert len(minimizer.optimized) == 3


def test_nebdlf_builds_minimal_path_when_only_one_geometry_present(monkeypatch):
    engine = _terachem_qcop_engine()
    input_chain = _chain([0.0, 1.0])
    path_geometry = """
 ***  Molecular Geometry (ANGS) ***
Type         X              Y              Z            Mass
   H   0.5000000000   0.0000000000   0.0000000000   1.0078250370
   H   0.5000000000   0.0000000000   0.7399999993   1.0078250370
"""
    fake_output = SimpleNamespace(files={"scr.path/path.geometry": path_geometry})

    monkeypatch.setattr(
        engine,
        "compute_func",
        lambda *args, **kwargs: fake_output,
    )

    def _fake_compute_energies(chain: Chain):
        energies = np.array([-1.0, -0.9, -0.8], dtype=float)
        for node, ene in zip(chain, energies):
            node._cached_energy = float(ene)
        return energies

    monkeypatch.setattr(engine, "compute_energies", _fake_compute_energies)

    minimizer = DLFindNEB(
        initial_chain=input_chain,
        engine=engine,
        parameters={"do_elem_step_checks": False},
    )
    minimizer.optimize_chain()

    assert minimizer.optimized is not None
    assert len(minimizer.optimized) == 3
