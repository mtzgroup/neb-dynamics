from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from neb_dynamics.retropaths_compat import retropaths_pot_to_neb_pot
from neb_dynamics.retropaths_queue import build_retropaths_neb_queue


ROOT_SMILES = "C=CC(O)CC=C"
ENVIRONMENT_SMILES = "O"
RUN_NAME = "allylic_alcohol_water"
TIMEOUT_SECONDS = 30
MAX_NODES = 40
MAX_DEPTH = 4


def _retropaths_repo() -> Path:
    return Path(__file__).resolve().parents[3] / "retropaths"


def _cache_dir() -> Path:
    return Path(__file__).resolve().parent / "retropaths_cache"


def _retropaths_pot_cache_fp() -> Path:
    return _cache_dir() / f"{RUN_NAME}_retropaths_pot.json"


def _neb_pot_cache_fp() -> Path:
    return _cache_dir() / f"{RUN_NAME}_neb_pot.json"


def _neb_queue_cache_fp() -> Path:
    return _cache_dir() / f"{RUN_NAME}_neb_queue.json"


def _load_or_grow_retropaths_pot():
    repo = _retropaths_repo()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    if "cairosvg" not in sys.modules:
        cairosvg_stub = types.ModuleType("cairosvg")

        def _svg2png(*_args, **_kwargs):
            raise RuntimeError("cairosvg is not available in this environment")

        cairosvg_stub.svg2png = _svg2png
        sys.modules["cairosvg"] = cairosvg_stub

    if "imgkit" not in sys.modules:
        imgkit_stub = types.ModuleType("imgkit")

        def _from_string(*_args, **_kwargs):
            raise RuntimeError("imgkit is not available in this environment")

        imgkit_stub.from_string = _from_string
        sys.modules["imgkit"] = imgkit_stub

    import retropaths.helper_functions as hf
    from retropaths.molecules.molecule import Molecule
    from retropaths.reactions.pot import Pot

    cache_fp = _retropaths_pot_cache_fp()
    if cache_fp.exists():
        return Pot.from_json(cache_fp)

    library = hf.pload(repo / "data" / "reactions.p")
    root = Molecule.from_smiles(ROOT_SMILES)
    environment = Molecule.from_smiles(ENVIRONMENT_SMILES)
    pot = Pot(root=root, environment=environment, rxn_name=RUN_NAME)
    pot.run_with_timeout_and_error_catching(
        timeout_seconds_pot=TIMEOUT_SECONDS,
        library=library,
        name=RUN_NAME,
        maximum_number_of_nodes=MAX_NODES,
        max_depth=MAX_DEPTH,
    )
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    pot.to_json(cache_fp)
    return pot


def _summarize_retropaths_pot(pot) -> dict:
    return {
        "status": pot.status.name,
        "run_time": pot.run_time,
        "number_of_nodes": pot.number_of_nodes,
        "number_of_edges": pot.graph.number_of_edges(),
        "leaves": list(pot.leaves),
        "root_smiles": pot.root.force_smiles(),
        "environment_smiles": pot.environment.force_smiles(),
    }


def _summarize_neb_pot(pot) -> dict:
    return {
        "number_of_nodes": pot.number_of_nodes,
        "number_of_edges": pot.graph.number_of_edges(),
        "nodes_with_td": sum("td" in pot.graph.nodes[i] for i in pot.graph.nodes),
        "root_smiles": pot.root.force_smiles(),
    }


def main():
    retropaths_pot = _load_or_grow_retropaths_pot()
    neb_pot = retropaths_pot_to_neb_pot(retropaths_pot)
    neb_pot.write_to_disk(_neb_pot_cache_fp())
    queue = build_retropaths_neb_queue(neb_pot, queue_fp=_neb_queue_cache_fp())

    summary = {
        "retropaths_cache": str(_retropaths_pot_cache_fp()),
        "neb_cache": str(_neb_pot_cache_fp()),
        "queue_cache": str(_neb_queue_cache_fp()),
        "retropaths": _summarize_retropaths_pot(retropaths_pot),
        "neb": _summarize_neb_pot(neb_pot),
        "queue": {
            "items": len(queue.items),
            "pending": sum(item.status == "pending" for item in queue.items),
            "skipped_attempted": sum(
                item.status == "skipped_attempted" for item in queue.items
            ),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
