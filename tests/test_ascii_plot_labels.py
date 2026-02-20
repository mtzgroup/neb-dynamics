from neb_dynamics.scripts import main_cli


class _DummyChain:
    def __init__(self):
        self.nodes = [object(), object(), object(), object()]
        self.energies_kcalmol = [0.0, 1.5, 0.7, 2.0]


def test_final_ascii_profile_uses_node_indices(monkeypatch):
    captured = {}

    def _capture(energies, labels, width=60, height=12):
        captured["energies"] = list(energies)
        captured["labels"] = list(labels)
        return "plot"

    monkeypatch.setattr(main_cli, "_build_ascii_energy_profile", _capture)
    main_cli._ascii_profile_for_chain(_DummyChain())

    assert captured["labels"] == ["0", "1", "2", "3"]
    assert captured["energies"] == [0.0, 1.5, 0.7, 2.0]
