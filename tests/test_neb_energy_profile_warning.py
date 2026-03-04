import numpy as np

from neb_dynamics.neb import _endpoint_energy_inversion_warning_text


def test_endpoint_energy_inversion_warning_triggers_for_qmmm():
    energies = np.array([0.020, 0.000, 0.005, 0.018])
    msg = _endpoint_energy_inversion_warning_text(
        energies=energies,
        is_qmmm_engine=True,
        frozen_atom_indices=[],
    )
    assert msg is not None
    assert "QM/MM" in msg
    assert "frozen atoms" in msg


def test_endpoint_energy_inversion_warning_not_triggered_when_ts_interior():
    energies = np.array([0.000, 0.020, 0.005, 0.001])
    msg = _endpoint_energy_inversion_warning_text(
        energies=energies,
        is_qmmm_engine=True,
        frozen_atom_indices=[0, 1],
    )
    assert msg is None
