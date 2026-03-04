from types import SimpleNamespace

from neb_dynamics.scripts import progress


class _Node:
    def __init__(self, has_molecular_graph: bool):
        self.has_molecular_graph = has_molecular_graph
        self.graph = object() if has_molecular_graph else None
        self.structure = object()


class _Chain:
    def __init__(self, has_molecular_graph: bool):
        self.nodes = [_Node(has_molecular_graph), _Node(has_molecular_graph)]
        self.energies_kcalmol = [0.0, 1.0]


def test_ascii_profile_skips_smiles_for_graphless_nodes(monkeypatch):
    def _fail(_structure):
        raise AssertionError("structure_to_smiles should not be called")

    monkeypatch.setattr(progress, "structure_to_smiles", _fail)
    out = progress.ascii_profile_for_chain(_Chain(has_molecular_graph=False))

    assert "start SMILES: N/A" in out
    assert "end SMILES:   N/A" in out


def test_ascii_profile_uses_smiles_for_graph_nodes(monkeypatch):
    monkeypatch.setattr(progress, "structure_to_smiles", lambda _structure: "C")
    out = progress.ascii_profile_for_chain(_Chain(has_molecular_graph=True))

    assert "start SMILES: C" in out
    assert "end SMILES:   C" in out
