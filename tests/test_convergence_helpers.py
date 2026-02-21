from types import SimpleNamespace

import numpy as np

from neb_dynamics import convergence_helpers as conv


class _DummyChain:
    def __init__(self, n_nodes: int, ts_index: int):
        self._n = n_nodes
        self._energies = np.zeros(n_nodes)
        self._energies[ts_index] = 1.0
        self.nodes = [
            SimpleNamespace(converged=False, _cached_gradient=np.zeros((2, 3)), _cached_energy=0.0)
            for _ in range(n_nodes)
        ]
        self.parameters = SimpleNamespace(frozen_atom_indices="", node_freezing=False)

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return self._n

    @property
    def gradients(self):
        return [np.zeros((2, 3)) for _ in range(self._n)]

    @property
    def springgradients(self):
        return [np.zeros((2, 3)) for _ in range(self._n - 2)]

    @property
    def ts_triplet_gspring_infnorm(self):
        return 0.0

    @property
    def energies(self):
        return self._energies

    @property
    def gperps(self):
        raise AssertionError("chain_converged should use chainhelpers.get_g_perps, not chain.gperps")

    def get_eA_chain(self):
        return float(self._energies.max())


def test_chain_converged_uses_endpoint_padded_gperps_for_ts_index(monkeypatch):
    chain_prev = _DummyChain(n_nodes=15, ts_index=13)
    chain_new = _DummyChain(n_nodes=15, ts_index=13)

    # Endpoint-padded gperps shape: len == n_nodes.
    monkeypatch.setattr(
        "neb_dynamics.chainhelpers.get_g_perps",
        lambda chain: [np.zeros((2, 3)) for _ in range(len(chain))],
    )

    parameters = SimpleNamespace(
        rms_grad_thre=0.001,
        max_rms_grad_thre=0.01,
        ts_grad_thre=0.05,
        ts_spring_thre=0.02,
        barrier_thre=0.1,
    )

    # Regression: this used to raise IndexError when TS index was near end.
    out = conv.chain_converged(
        chain_prev=chain_prev,
        chain_new=chain_new,
        parameters=parameters,
        verbose=False,
    )
    assert isinstance(out, (bool, np.bool_))
