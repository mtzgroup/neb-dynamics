from types import SimpleNamespace

from neb_dynamics.elementarystep import _filter_new_structures
from neb_dynamics.nodes import nodehelpers


def test_is_identical_collects_connectivity_once(monkeypatch):
    calls: list[bool] = []

    def _fake_connectivity(self, other, verbose=True, collect_comparison=True):
        calls.append(collect_comparison)
        return True

    def _fake_conformer(
        self,
        other,
        global_rmsd_cutoff=20.0,
        fragment_rmsd_cutoff=0.5,
        kcal_mol_cutoff=1.0,
        verbose=True,
    ):
        # Mimic internal connectivity check without collecting a second row.
        return nodehelpers._is_connectivity_identical(
            self, other, verbose=False, collect_comparison=False
        )

    monkeypatch.setattr(nodehelpers, "_is_connectivity_identical", _fake_connectivity)
    monkeypatch.setattr(nodehelpers, "_is_conformer_identical", _fake_conformer)

    node_a = SimpleNamespace(has_molecular_graph=True)
    node_b = SimpleNamespace(has_molecular_graph=True)

    assert nodehelpers.is_identical(node_a, node_b, verbose=False) is True
    assert calls == [True, False]


def test_filter_new_structures_does_not_collect_comparisons(monkeypatch):
    calls: list[bool] = []

    def _fake_is_identical(
        self,
        other,
        *,
        global_rmsd_cutoff=20.0,
        fragment_rmsd_cutoff=1.0,
        kcal_mol_cutoff=1.0,
        verbose=True,
        collect_comparison=True,
    ):
        calls.append(collect_comparison)
        return False

    monkeypatch.setattr("neb_dynamics.elementarystep.is_identical", _fake_is_identical)

    chain = SimpleNamespace(
        parameters=SimpleNamespace(node_rms_thre=1.0, node_ene_thre=1.0)
    )
    nodes = [SimpleNamespace()]
    reactant = SimpleNamespace()
    product = SimpleNamespace()

    out = _filter_new_structures(nodes=nodes, reactant=reactant, product=product, chain=chain)
    assert out == nodes
    assert calls == [False, False]

