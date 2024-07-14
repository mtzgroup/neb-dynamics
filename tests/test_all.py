from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
import numpy as np
from neb_dynamics.NEB import NEB
from neb_dynamics.Inputs import NEBInputs
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.optimizers.SD import SteepestDescent
from neb_dynamics.Chain import Chain
from neb_dynamics.Inputs import ChainInputs
from neb_dynamics.constants import BOHR_TO_ANGSTROMS


def test_tdstructure():
    td = TDStructure.from_smiles("C")
    td.tc_model_method = 'ub3lyp'
    td.tc_model_basis = '3-21gs'
    td.tc_kwds = {}

    ene_xtb = td.energy_xtb()
    # ene_tc = td.energy_tc()
    assert np.isclose(
        ene_xtb, -4.174962211746928), "XTB energy was different to reference."
    # assert np.isclose(ene_tc, -40.3015855774), "TC Chemcloud energy was different to reference"


def test_trajectory():
    coords = np.array([[1.07708884,  0.06051313,  0.04629178],
                      [2.41285238,  0.06051313,  0.04629178],
                       [0.51728556,  0.80914007,  0.59808046],
                       [0.51728556, -0.68811381, -0.50549689],
                       [2.97265566,  0.80914007,  0.59808046],
                       [2.97265566, -0.68811381, -0.50549689]])
    symbols = np.array(['C', 'C', 'H', 'H', 'H', 'H'], dtype='<U1')
    start = TDStructure.from_coords_symbols(coords, symbols)

    end = start.copy()
    end_coords = end.coords
    # this is a pi-bond rotation
    end_coords_swapped = end_coords[[0, 1, 3, 2, 4, 5], :]
    end = end.update_coords(end_coords_swapped)

    traj = Trajectory([start, end])
    traj_interpolated = traj.run_geodesic(nimages=10, sweep=False)
    barrier = max(traj_interpolated.energies_xtb())
    ref_barrier = 73  # there is a high variance in this barrier estimate.
    assert np.isclose(barrier, ref_barrier, atol=5,
                      rtol=5), f"GI barrier was off. ref={ref_barrier}. test={barrier}"


def test_neb():
    coords = np.array([[1.08685977,  0.06051313,  0.04629178],
                       [2.40308145,  0.06051313,  0.04629178],
                       [0.51561922,  0.79599799,  0.58839386],
                       [0.51561922, -0.67497173, -0.49581029],
                       [2.974322,  0.79599799,  0.58839386],
                       [2.974322, -0.67497173, -0.49581029]])

    symbols = np.array(['C', 'C', 'H', 'H', 'H', 'H'], dtype='<U1')
    start = TDStructure.from_coords_symbols(coords, symbols)
    end = start.copy()
    end_coords = end.coords
    # this is a pi-bond rotation
    end_coords_swapped = end_coords[[0, 1, 3, 2, 4, 5], :]
    end = end.update_coords(end_coords_swapped)

    tr = Trajectory([start, end]).run_geodesic(nimages=15)
    cni = ChainInputs(
        k=0.1,
        delta_k=0.09,
        do_parallel=True,
        node_freezing=True)

    tol = 0.0001
    nbi = NEBInputs(
        tol=tol * BOHR_TO_ANGSTROMS,
        barrier_thre=0.1,  # kcalmol,
        climb=False,

        rms_grad_thre=tol * BOHR_TO_ANGSTROMS,
        max_rms_grad_thre=tol * BOHR_TO_ANGSTROMS*2.5,
        ts_grad_thre=tol * BOHR_TO_ANGSTROMS,
        ts_spring_thre=tol * BOHR_TO_ANGSTROMS*3,

        v=1,
        max_steps=3000,
        early_stop_force_thre=0.0)
    initial_chain = Chain.from_traj(tr, parameters=cni)

    opt = VelocityProjectedOptimizer(timestep=0.5)
    n = NEB(initial_chain=initial_chain, parameters=nbi, optimizer=opt)

    _ = n.optimize_chain()
    assert n.optimized is not None, "Chain did not converge."
    barrier = n.optimized.get_eA_chain()
    ref_barrier = 83.32779475366846
    # atol and rtol assures that the same NEB run should be
    # within a kcal/mol
    # I've used VERY tight parameters in order to make this test numbers
    # low. Normal runs will have a barrier estimate varriance closer to 5kcal/mol
    assert np.isclose(
        barrier, ref_barrier, atol=.5, rtol=.5), f'NEB barrier was different. ref={ref_barrier}. test={barrier}'

    ts_guess = n.optimized.get_ts_guess()
    ref_ts_guess = np.array([[-7.03774981e-01, -4.20474533e-03,  1.29346690e-02],
                             [7.11319788e-01,  3.42021586e-04, -7.75091471e-03],
                             [-1.28292090e+00,  4.20407014e-01,  8.25902640e-01],
                             [-1.28587778e+00, -4.01021162e-01, -8.07049188e-01],
                             [1.27538590e+00,  8.10496554e-01, -4.47287007e-01],
                             [1.29344547e+00, -8.07823999e-01,  4.09841837e-01]])
    ref_tsg = TDStructure.from_coords_symbols(ref_ts_guess, symbols=symbols)
    aligned_tsg = ts_guess.align_to_td(ref_tsg)
    assert np.allclose(aligned_tsg.coords, ref_ts_guess, atol=0.1, rtol=0.1), \
        f"TS guess has a different geometry than reference. ref={ref_ts_guess}. test={aligned_tsg.coords}"


if __name__ == "__main__":
    # test_tdstructure()
    # test_trajectory()
    test_neb()
