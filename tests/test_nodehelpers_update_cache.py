import numpy as np

from neb_dynamics.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from neb_dynamics.nodes.node import XYNode
from neb_dynamics.nodes.nodehelpers import update_node_cache


def test_update_node_cache_accepts_fake_qcio_output_without_input_data():
    node = XYNode(structure=np.array([0.0, 0.0], dtype=float))
    result = FakeQCIOOutput.model_validate(
        {
            "results": FakeQCIOResults.model_validate(
                {"energy": -1.234, "gradient": [0.1, -0.2]}
            )
        }
    )

    update_node_cache([node], [result])

    assert node._cached_result is result
    assert node._cached_energy == -1.234
    assert np.allclose(np.asarray(node._cached_gradient, dtype=float), np.array([0.1, -0.2]))
