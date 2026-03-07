from pathlib import Path

import tomli

from neb_dynamics.helper_functions import parse_terachem_input_file
from neb_dynamics.scripts import main_cli


def test_parse_terachem_input_file_extracts_constraints(tmp_path):
    tcin = tmp_path / "s0min.in"
    tcin.write_text(
        "\n".join(
            [
                "prmtop ref.prmtop",
                "coordinates react.rst7",
                "qmindices qmindices.dat",
                "charge 0",
                "spinmult 1",
                "run minimize",
                "min_coordinates cartesian",
                "basis 6-31g**",
                "method b3lyp",
                "maxit 200",
                "$constraints",
                "atom 1",
                "atom 7",
                "$end",
                "",
            ]
        )
    )

    parsed = parse_terachem_input_file(tcin)
    assert parsed["method"] == "b3lyp"
    assert parsed["basis"] == "6-31g**"
    assert parsed["run_type"] == "minimize"
    assert parsed["frozen_atom_indices"] == [0, 6]
    assert parsed["keywords"]["maxit"] == "200"


def test_toml_from_tcin_writes_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tcin = tmp_path / "s0min_3A_reactant.in"
    tcin.write_text(
        "\n".join(
            [
                "prmtop 5p9i.prmtop",
                "coordinates react.rst7",
                "qmindices qmindices.dat",
                "charge 0",
                "spinmult 1",
                "run minimize",
                "min_coordinates cartesian",
                "basis 6-31g**",
                "method b3lyp",
                "threall 1e-14",
                "$constraints",
                "atom 1",
                "atom 2",
                "atom 10",
                "$end",
                "",
            ]
        )
    )
    (tmp_path / "5p9i.prmtop").write_text("x")
    (tmp_path / "qmindices.dat").write_text("0\n")
    (tmp_path / "react.rst7").write_text("x")

    out_fp = tmp_path / "qmmm_inputs_s0min_frozen.toml"
    main_cli.toml_from_tcin(
        tcin=str(tcin),
        output=str(out_fp),
        compute_program="chemcloud",
        queue="ethan",
    )

    data = tomli.loads(out_fp.read_text())
    assert data["engine_name"] == "qmmm"
    assert data["program"] == "terachem"
    assert data["chemcloud_queue"] == "ethan"
    assert data["program_kwds"]["model"]["method"] == "b3lyp"
    assert data["program_kwds"]["keywords"]["threall"] == "1e-14"
    assert data["qmmm_inputs"]["qminds_fp"] == "qmindices.dat"
    assert data["qmmm_inputs"]["prmtop_fp"] == "5p9i.prmtop"
    assert data["qmmm_inputs"]["rst7_fp_react"] == "react.rst7"
    assert data["chain_inputs"]["frozen_atom_indices"] == [0, 1, 9]
    assert data["qmmm_inputs"]["run_type"] == "gradient"


def test_toml_from_tcin_resolves_missing_coordinate_variant(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tcin = tmp_path / "s0min_3A_reactant.in"
    tcin.write_text(
        "\n".join(
            [
                "prmtop 5p9i_sphere_nobox.prmtop",
                "coordinates optim_reactant_closeCYS.rst7",
                "qmindices qmindices.dat",
                "charge 0",
                "spinmult 1",
                "run minimize",
                "basis 6-31g**",
                "method b3lyp",
                "$constraints",
                "atom 1",
                "$end",
                "",
            ]
        )
    )
    (tmp_path / "5p9i_sphere_nobox.prmtop").write_text("x")
    (tmp_path / "qmindices.dat").write_text("0\n")
    (tmp_path / "optim_reactant_closeCYS_3A.rst7").write_text("x")

    out_fp = tmp_path / "qmmm_inputs_from_tc.toml"
    main_cli.toml_from_tcin(
        tcin=str(tcin),
        output=str(out_fp),
        compute_program="chemcloud",
        queue=None,
    )
    data = tomli.loads(out_fp.read_text())
    assert data["qmmm_inputs"]["rst7_fp_react"] == "optim_reactant_closeCYS_3A.rst7"
