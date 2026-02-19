import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import XYNode
from neb_dynamics.optimizers.adam import AdamOptimizer
from neb_dynamics.optimizers.cg import ConjugateGradient
from neb_dynamics.optimizers.lbfgs import LBFGS
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer


def _make_xy_chain(coords: np.ndarray, converged: list[bool] | None = None) -> Chain:
    if converged is None:
        converged = [False] * len(coords)
    nodes = [
        XYNode(structure=np.array(c, dtype=float), converged=is_conv)
        for c, is_conv in zip(coords, converged)
    ]
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def _quadratic_objective(chain: Chain) -> float:
    return 0.5 * float(np.sum(chain.coordinates**2))


def _quadratic_gradient(chain: Chain) -> np.ndarray:
    return chain.coordinates.copy()


def test_conjugate_gradient_respects_converged_nodes():
    chain = _make_xy_chain(
        np.array([[1.0, 0.0], [2.0, -1.0], [0.5, 0.5]]),
        converged=[False, True, False],
    )
    grads = np.array([[1.0, 0.0], [4.0, -3.0], [0.5, 0.5]])
    opt = ConjugateGradient(timestep=0.25)

    new_chain = opt.optimize_step(chain=chain, chain_gradients=grads)

    # Middle node is marked converged, so it should not move.
    np.testing.assert_allclose(new_chain.coordinates[1], chain.coordinates[1], atol=1e-12)
    # Unconverged nodes should move opposite to gradient.
    np.testing.assert_allclose(new_chain.coordinates[0], np.array([0.75, 0.0]), atol=1e-12)
    np.testing.assert_allclose(new_chain.coordinates[2], np.array([0.375, 0.375]), atol=1e-12)


def test_conjugate_gradient_recovers_after_gradient_shape_change():
    chain = _make_xy_chain(np.array([[1.0, 1.0], [0.5, -0.5]]))
    grads = np.array([[1.0, 1.0], [0.5, -0.5]])
    opt = ConjugateGradient(timestep=0.1)
    # Simulate stale optimizer state from a previous chain shape.
    opt.g_old = np.array([1.0, 2.0, 3.0])

    new_chain = opt.optimize_step(chain=chain, chain_gradients=grads)

    np.testing.assert_allclose(new_chain.coordinates, np.array([[0.9, 0.9], [0.45, -0.45]]), atol=1e-12)
    assert opt.g_old.shape == grads.shape


def test_vpo_uses_force_step_when_gradient_is_large():
    chain = _make_xy_chain(np.array([[1.0, -1.0], [0.5, 0.5]]))
    grads = np.array([[2.0, -2.0], [1.0, 1.0]])
    opt = VelocityProjectedOptimizer(timestep=0.2, activation_tol=0.01)

    new_chain = opt.optimize_step(chain=chain, chain_gradients=grads)

    expected_displacement = -0.2 * grads
    np.testing.assert_allclose(new_chain.coordinates, chain.coordinates + expected_displacement, atol=1e-12)
    np.testing.assert_allclose(new_chain.velocity, expected_displacement, atol=1e-12)


def test_lbfgs_first_step_matches_steepest_descent_direction():
    chain = _make_xy_chain(np.array([[1.0, 0.0], [0.0, -2.0]]))
    grads = np.array([[1.0, 0.0], [0.0, -2.0]])
    opt = LBFGS(timestep=0.5, history_size=3)

    new_chain = opt.optimize_step(chain=chain, chain_gradients=grads)

    np.testing.assert_allclose(new_chain.coordinates, np.array([[0.5, 0.0], [0.0, -1.0]]), atol=1e-12)


def test_lbfgs_falls_back_cleanly_when_curvature_is_poor():
    chain = _make_xy_chain(np.array([[1.0, 1.0], [2.0, -1.0]]))
    opt = LBFGS(timestep=0.1, history_size=3)

    # First step initializes state.
    chain = opt.optimize_step(chain=chain, chain_gradients=np.ones_like(chain.coordinates))
    # Same gradient on next step forces y_k ~ 0 and should trigger fallback/reset.
    next_chain = opt.optimize_step(chain=chain, chain_gradients=np.ones_like(chain.coordinates))

    assert len(opt.s_history) == 0
    assert len(opt.y_history) == 0
    np.testing.assert_allclose(next_chain.coordinates, chain.coordinates - 0.1, atol=1e-12)


def test_optimizer_steps_reduce_quadratic_objective():
    initial = _make_xy_chain(np.array([[2.0, -1.0], [1.5, 0.5], [-1.0, 1.0]]))
    optimizers = [
        ConjugateGradient(timestep=0.05),
        VelocityProjectedOptimizer(timestep=0.05, activation_tol=1e-8),
        LBFGS(timestep=0.2, history_size=5),
        AdamOptimizer(timestep=0.05),
    ]

    for opt in optimizers:
        chain = initial.model_copy(deep=True)
        start_obj = _quadratic_objective(chain)
        for _ in range(15):
            grads = _quadratic_gradient(chain)
            chain = opt.optimize_step(chain=chain, chain_gradients=grads)
        end_obj = _quadratic_objective(chain)
        assert end_obj < start_obj, f"{type(opt).__name__} failed to decrease objective."
