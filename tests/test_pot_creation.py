import json
from pathlib import Path

from neb_dynamics.molecule import Molecule
from neb_dynamics.pot import Pot


def test_pot_from_local_fixture_roundtrip():
    fixture_path = Path(__file__).parent / "data.json"
    with fixture_path.open() as f:
        loaded = json.load(f)

    pot = Pot.from_dict(loaded)
    dumped = pot.model_dump()
    pot_roundtrip = Pot.from_dict(dumped)

    for node_id in pot_roundtrip.graph.nodes:
        assert isinstance(pot_roundtrip.graph.nodes[node_id]["molecule"], Molecule)
