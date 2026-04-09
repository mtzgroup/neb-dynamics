from __future__ import annotations

import concurrent.futures
from contextlib import nullcontext

import networkx as nx

import neb_dynamics.isomorphism_tools as iso_tools


def _toy_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node(0, element="C", neighbors=4, charge=0)
    g.add_node(1, element="C", neighbors=4, charge=0)
    g.add_edge(0, 1, bond_order=1)
    return g


def test_subgraph_matcher_uses_timeout_on_main_thread(monkeypatch):
    g1 = _toy_graph()
    g2 = _toy_graph()
    seen = {"calls": 0}

    def _fake_timeout(*args, **kwargs):
        seen["calls"] += 1
        return nullcontext()

    monkeypatch.setattr(iso_tools, "timeout", _fake_timeout)

    matcher = iso_tools.SubGraphMatcher(g1, timeout_seconds=5)
    assert matcher.is_bond_isomorphic(g2) is True
    assert seen["calls"] >= 1


def test_subgraph_matcher_skips_signal_timeout_in_worker_thread(monkeypatch):
    g1 = _toy_graph()
    g2 = _toy_graph()

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("timeout() must not be used in worker thread")

    monkeypatch.setattr(iso_tools, "timeout", _should_not_be_called)

    def _run_in_worker() -> bool:
        matcher = iso_tools.SubGraphMatcher(g1, timeout_seconds=5)
        return matcher.is_bond_isomorphic(g2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_run_in_worker).result(timeout=5)
    assert result is True
