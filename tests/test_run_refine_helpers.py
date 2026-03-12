import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.TreeNode import TreeNode
from neb_dynamics.nodes.node import XYNode
from neb_dynamics.scripts.main_cli import (
    _build_recycled_pair_chain,
    _clear_chain_cached_properties,
    _concat_chains,
    _extract_minima_nodes,
    _extract_minima_nodes_from_chain,
)


def _xy_chain(coords: np.ndarray, energies: list[float]) -> Chain:
    nodes = []
    for c, e in zip(coords, energies):
        nodes.append(
            XYNode(
                structure=np.array(c, dtype=float),
                _cached_energy=float(e),
                _cached_gradient=np.zeros_like(np.array(c, dtype=float)),
            )
        )
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def test_extract_minima_nodes_from_chain_includes_endpoints_and_internal_minima():
    chain = _xy_chain(
        coords=np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]),
        energies=[0.0, 2.0, 1.0, 3.0, 0.0],
    )

    minima = _extract_minima_nodes_from_chain(chain)

    assert len(minima) == 3
    np.testing.assert_allclose(minima[0].coords, np.array([0.0]))
    np.testing.assert_allclose(minima[1].coords, np.array([2.0]))
    np.testing.assert_allclose(minima[2].coords, np.array([4.0]))


def test_extract_minima_nodes_from_recursive_history_uses_leaf_endpoints():
    leaf_chain = _xy_chain(
        coords=np.array([[0.0], [1.0], [2.0]]),
        energies=[0.0, 1.0, 0.0],
    )

    class _LeafData:
        chain_trajectory = [leaf_chain]

    root = TreeNode(data=None, children=[
        TreeNode(data=_LeafData(), children=[], index=1)
    ], index=0)

    minima = _extract_minima_nodes(root)

    assert len(minima) == 2
    np.testing.assert_allclose(minima[0].coords, np.array([0.0]))
    np.testing.assert_allclose(minima[1].coords, np.array([2.0]))


def test_concat_chains_avoids_duplicate_joint_node():
    chain_a = _xy_chain(
        coords=np.array([[0.0], [1.0], [2.0]]),
        energies=[0.0, 1.0, 2.0],
    )
    chain_b = _xy_chain(
        coords=np.array([[2.0], [3.0], [4.0]]),
        energies=[2.0, 1.0, 0.0],
    )

    combined = _concat_chains([chain_a, chain_b], ChainInputs())

    assert len(combined.nodes) == 5
    np.testing.assert_allclose(
        combined.coordinates.flatten(), np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    )


def test_clear_chain_cached_properties_removes_energies_and_gradients():
    chain = _xy_chain(
        coords=np.array([[0.0], [1.0], [2.0]]),
        energies=[0.0, 1.0, 2.0],
    )

    cleaned = _clear_chain_cached_properties(chain, ChainInputs())

    for node in cleaned.nodes:
        assert node._cached_energy is None
        assert node._cached_gradient is None
        assert node._cached_result is None


def test_build_recycled_pair_chain_reuses_segment_and_clears_cache():
    cheap_chain = _xy_chain(
        coords=np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]),
        energies=[0.0, 0.2, 0.4, 0.2, 0.0],
    )
    cheap_start_ref = cheap_chain.nodes[1].copy()
    cheap_end_ref = cheap_chain.nodes[3].copy()

    expensive_start = XYNode(
        structure=np.array([10.0]),
        _cached_energy=5.0,
        _cached_gradient=np.array([1.0]),
    )
    expensive_end = XYNode(
        structure=np.array([30.0]),
        _cached_energy=7.0,
        _cached_gradient=np.array([1.0]),
    )

    recycled = _build_recycled_pair_chain(
        cheap_output_chain=cheap_chain,
        cheap_start_ref=cheap_start_ref,
        cheap_end_ref=cheap_end_ref,
        expensive_start=expensive_start,
        expensive_end=expensive_end,
        cheap_chain_inputs=ChainInputs(),
        expensive_chain_inputs=ChainInputs(),
        expected_nimages=3,
    )

    assert recycled is not None
    assert len(recycled.nodes) == 3
    np.testing.assert_allclose(recycled.nodes[0].coords, np.array([10.0]))
    np.testing.assert_allclose(recycled.nodes[1].coords, np.array([2.0]))
    np.testing.assert_allclose(recycled.nodes[-1].coords, np.array([30.0]))
    for node in recycled.nodes:
        assert node._cached_energy is None
        assert node._cached_gradient is None
