import numpy as np
from qcio import Structure

import neb_dynamics.chainhelpers as ch
from neb_dynamics.chain import Chain
from neb_dynamics.dynamics.chainbiaser import ChainBiaser
from neb_dynamics.engines.threewell import ThreeWellPotential
from neb_dynamics.engines.qcop import QCOPEngine
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.inputs import NEBInputs
from neb_dynamics.neb import NEB
from neb_dynamics.nodes.node import StructureNode, XYNode
from neb_dynamics.optimizers.cg import ConjugateGradient


def _structure(coords: np.ndarray) -> Structure:
    return Structure(
        geometry=np.array(coords, dtype=float),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def test_threewell_bias_uses_default_reference_chains():
    biaser = ChainBiaser(
        reference_chains=[
            [XYNode(structure=np.array([0.0, 0.0]))],
            [XYNode(structure=np.array([2.0, 0.0]))],
        ],
        amplitude=2.0,
        sigma=0.5,
    )
    engine = ThreeWellPotential(biaser=biaser)

    xy = np.array([0.0, 0.0])
    base_energy = (xy[0] ** 2 + xy[1] - 11) ** 2 + (xy[0] + xy[1] ** 2 - 7) ** 2
    biased_energy = engine._en_func(xy)

    assert biased_energy > base_energy
    np.testing.assert_allclose(
        biased_energy,
        base_energy + biaser.energy_node_bias(XYNode(structure=xy)),
    )


def test_qcop_bias_applied_once_across_multiple_reference_chains():
    node = StructureNode(structure=_structure([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]))
    node._cached_energy = -1.0
    node._cached_gradient = np.zeros((2, 3))

    chain = Chain.model_validate({"nodes": [node], "parameters": ChainInputs()})
    ref_a = [StructureNode(structure=_structure([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]))]
    ref_b = [StructureNode(structure=_structure([[0.2, 0.0, 0.0], [0.2, 0.0, 0.7]]))]
    biaser = ChainBiaser(reference_chains=[ref_a, ref_b], amplitude=1.5, sigma=0.8)
    engine = QCOPEngine(biaser=biaser)

    grads = engine.compute_gradients(chain)
    enes = engine.compute_energies(chain)

    np.testing.assert_allclose(grads[0], biaser.gradient_node_bias(node=node))
    np.testing.assert_allclose(enes[0], node.energy + biaser.energy_node_bias(node=node))


def test_compute_neb_gradient_supports_xy_chains():
    coords = np.array(
        [
            [-2.805118, 3.131312],
            [-1.5, 3.0],
            [0.0, 2.7],
            [1.6, 2.3],
            [3.0, 2.0],
        ]
    )
    chain = Chain.model_validate(
        {"nodes": [XYNode(structure=np.array(c, dtype=float)) for c in coords], "parameters": ChainInputs()}
    )
    engine = ThreeWellPotential()
    engine.compute_gradients(chain)
    engine.compute_energies(chain)

    grads = ch.compute_NEB_gradient(chain)

    assert grads.shape == coords.shape
    assert np.all(np.isfinite(grads))


def test_additional_pes_gradient_is_projected_out_along_tangent():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    nodes = [XYNode(structure=np.array(c, dtype=float)) for c in coords]
    for node in nodes:
        node._cached_energy = 0.0
        node._cached_gradient = np.zeros(2)
    chain = Chain.model_validate({"nodes": nodes, "parameters": ChainInputs(k=0.0, delta_k=0.0)})

    additional = np.zeros_like(coords)
    additional[1] = np.array([3.0, 0.0])

    grads = ch.compute_NEB_gradient(chain, additional_gradients=additional)

    np.testing.assert_allclose(grads, np.zeros_like(coords), atol=1e-12)


def test_neb_update_chain_accepts_chain_bias_for_xy_chain():
    coords = np.array(
        [
            [-2.805118, 3.131312],
            [-1.5, 3.0],
            [0.0, 2.7],
            [1.6, 2.3],
            [3.0, 2.0],
        ]
    )
    chain = Chain.model_validate(
        {
            "nodes": [XYNode(structure=np.array(c, dtype=float)) for c in coords],
            "parameters": ChainInputs(k=0.005, delta_k=0.002, use_geodesic_interpolation=False),
        }
    )
    reference_chain = chain.copy()
    biaser = ChainBiaser(reference_chains=[reference_chain], amplitude=0.01, sigma=3.0)
    neb = NEB(
        initial_chain=chain,
        parameters=NEBInputs(do_elem_step_checks=False, max_steps=5),
        optimizer=ConjugateGradient(timestep=1e-4),
        engine=ThreeWellPotential(),
        biaser=biaser,
    )

    updated = neb.update_chain(chain)

    assert np.all(np.isfinite(updated.coordinates))
    assert np.linalg.norm(updated.coordinates - chain.coordinates) > 0.0
