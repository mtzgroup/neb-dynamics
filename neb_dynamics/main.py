from typing import List

from treenode import TreeNode
from models.minimizers import PathMinimizer, NodeType, MinimizerType
from elementarystep import check_if_elem_step
from .models.minimizers import NEB
from geodesic_interpolation.geodesic import run_geodesic_py
from .engines.qcop import QCOPEngine
from .optimizers.vpo import VelocityProjectedOptimizer
from .models.minimizers import EngineType


def run_msmep(minimizer: MinimizerType) -> TreeNode:
    """Run the MSMEP"""
    # Run first NEB
    minimizer.run()

    # Analyze the run
    results = check_if_elem_step(minimizer.trajectory[-1])

    parent = TreeNode(minimizer)

    if results.is_elem_step:
        return parent

    else:
        children = []
        # Split the path
        chains = minimizer.do_split()
        # Run NEB on each path
        for chain in chains:
            new_minimizer = minimizer.model_copy(
                deep=True, update={"trajectory": [chain]}
            )
            children.append(run_msmep(new_minimizer))

    # Add children to parent
    parent.children.extend(children)
    return parent


INTERPOLATORS = {
    "geodesic": run_geodesic_py,
}


def compute_energies_and_gradients(node: NodeType, engine: EngineType) -> None:
    """Compute Energies and Gradients for a Node"""
    # Compute Gradients
    engine.compute_gradients(node)

    # Compute Energies
    engine.compute_energies(node)


if __name__ == "__main__":
    from pathlib import Path

    # Interpolate chain between first and last node
    interp_params = {}
    algorithm = "geodesic"
    initial_structure: List[NodeType] = []
    chain = INTERPOLATORS[algorithm](initial_structure, **interp_params)

    # Instantiate the Engine
    engine_params = {}
    engine = QCOPEngine(**engine_params)

    # Instantiate the Optimizer
    optimizer_params = {}
    optimizer = VelocityProjectedOptimizer(**optimizer_params)

    # Instantiate the Minimizer
    neb_params = {}
    minimizer = NEB(
        trajectory=[chain], engine=engine, optimizer=optimizer, parameters=neb_params
    )
    root = run_msmep(minimizer)
    Path("output.json").write_text(root.model_dump_json())
