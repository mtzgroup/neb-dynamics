import importlib
from typing import TYPE_CHECKING

from neb_dynamics.nodes.node import Node, StructureNode
from neb_dynamics.chain import Chain
# from neb_dynamics.engines.qcop import QCOPEngine
# from neb_dynamics.engines.ase import ASEEngine
from neb_dynamics.msmep import MSMEP
from neb_dynamics.inputs import *

if TYPE_CHECKING:
    from neb_dynamics.retropaths_compat import (
        annotate_pot_with_neb_results,
        copy_graph_like_molecule,
        retropaths_pot_to_neb_pot,
        structure_node_from_graph_like_molecule,
    )
    from neb_dynamics.retropaths_queue import (
        RetropathsNEBQueue,
        annotate_pot_with_queue_results,
        build_retropaths_neb_queue,
        load_completed_queue_chains,
        pair_attempt_key,
        run_retropaths_neb_queue,
        structure_node_attempt_signature,
    )

_RETROPATHS_EXPORTS = {
    "annotate_pot_with_neb_results": ("neb_dynamics.retropaths_compat", "annotate_pot_with_neb_results"),
    "copy_graph_like_molecule": ("neb_dynamics.retropaths_compat", "copy_graph_like_molecule"),
    "retropaths_pot_to_neb_pot": ("neb_dynamics.retropaths_compat", "retropaths_pot_to_neb_pot"),
    "structure_node_from_graph_like_molecule": ("neb_dynamics.retropaths_compat", "structure_node_from_graph_like_molecule"),
    "RetropathsNEBQueue": ("neb_dynamics.retropaths_queue", "RetropathsNEBQueue"),
    "annotate_pot_with_queue_results": ("neb_dynamics.retropaths_queue", "annotate_pot_with_queue_results"),
    "build_retropaths_neb_queue": ("neb_dynamics.retropaths_queue", "build_retropaths_neb_queue"),
    "load_completed_queue_chains": ("neb_dynamics.retropaths_queue", "load_completed_queue_chains"),
    "pair_attempt_key": ("neb_dynamics.retropaths_queue", "pair_attempt_key"),
    "run_retropaths_neb_queue": ("neb_dynamics.retropaths_queue", "run_retropaths_neb_queue"),
    "structure_node_attempt_signature": ("neb_dynamics.retropaths_queue", "structure_node_attempt_signature"),
}


def __getattr__(name: str):
    if name in _RETROPATHS_EXPORTS:
        module_name, attr_name = _RETROPATHS_EXPORTS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
