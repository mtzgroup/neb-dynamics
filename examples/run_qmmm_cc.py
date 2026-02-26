"""Legacy helper for local QMMM experiments.

Preferred workflow (CLI):
    neb-dynamics run --start react.xyz --end prod.xyz --inputs qmmm_inputs.toml
"""

from pathlib import Path

from neb_dynamics import MSMEP, StructureNode
from neb_dynamics.chain import Chain
from neb_dynamics.inputs import RunInputs
from qcio import Structure


def run_local_qmmm_example(inputs_fp: Path, start_fp: Path, end_fp: Path, output_fp: Path) -> None:
    """Run a non-recursive NEB from the same TOML schema used by the CLI."""
    run_inputs = RunInputs.open(inputs_fp)
    nodes = [StructureNode(structure=Structure.open(start_fp)), StructureNode(structure=Structure.open(end_fp))]
    chain = Chain.model_validate({"nodes": nodes, "parameters": run_inputs.chain_inputs})
    neb_result, _ = MSMEP(run_inputs).run_minimize_chain(chain)
    neb_result.write_to_disk(output_fp)


if __name__ == "__main__":
    run_local_qmmm_example(
        inputs_fp=Path("qmmm_inputs.toml"),
        start_fp=Path("./publication_launch/react2_opt.xyz"),
        end_fp=Path("./publication_launch/prod_opt_qmmm2.xyz"),
        output_fp=Path("./neb_qmmm_cli_style.xyz"),
    )
