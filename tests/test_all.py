from pathlib import Path

import numpy as np

from neb_dynamics.Chain import Chain
from neb_dynamics.constants import BOHR_TO_ANGSTROMS
from neb_dynamics.Inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.NEB import NEB
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory


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


def test_neb(test_data_dir: Path = Path("/home/jdep/neb_dynamics/tests")):
    tr = Trajectory.from_xyz(test_data_dir / "test_traj.xyz")

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
    symbols = tr.symbols

    opt = VelocityProjectedOptimizer(timestep=0.5)
    n = NEB(initial_chain=initial_chain, parameters=nbi, optimizer=opt)

    _ = n.optimize_chain()
    assert n.optimized is not None, "Chain did not converge."
    barrier = n.optimized.get_eA_chain()
    ref_barrier = 82.86363760397421
    print(barrier)
    # atol and rtol assures that the same NEB run should be
    # within a kcal/mol
    # I've used VERY tight parameters in order to make this test numbers
    # low. Normal runs will have a barrier estimate varriance closer to 5kcal/mol
    assert np.isclose(
        barrier, ref_barrier), f'NEB barrier was different. ref={ref_barrier}. test={barrier}'

    ts_guess = n.optimized.get_ts_guess()
    ref_ts_guess = np.array([[-7.03774981e-01, -4.20474533e-03,  1.29346690e-02],
                             [7.11319788e-01,  3.42021586e-04, -7.75091471e-03],
                             [-1.28292090e+00,  4.20407014e-01,  8.25902640e-01],
                             [-1.28587778e+00, -4.01021162e-01, -8.07049188e-01],
                             [1.27538590e+00,  8.10496554e-01, -4.47287007e-01],
                             [1.29344547e+00, -8.07823999e-01,  4.09841837e-01]])
    ref_tsg = TDStructure.from_coords_symbols(ref_ts_guess, symbols=symbols)
    aligned_tsg = ts_guess.align_to_td(ref_tsg)
    assert np.allclose(aligned_tsg.coords, ref_ts_guess, atol=0.5, rtol=0.5), \
        f"TS guess has a different geometry than reference. ref={ref_ts_guess}. test={aligned_tsg.coords}"


def test_msmep():
    coords = np.array([[9.36213367e-01,  7.59928266e-01, -9.42816347e-01],
                       [1.73549975e+00,  5.23206629e-01,  5.46603900e-01],
                       [-3.30550111e-03,  2.31729785e+00, -8.47421416e-01],
                       [-2.78141313e-01, -3.14382009e-01, -1.88677066e+00],
                       [2.21045453e+00,  1.03539947e+00, -2.22139261e+00],
                       [-1.77223560e+00, -6.57270784e-01,  2.89396984e+00],
                       [-1.73663677e+00, -1.18932471e+00,  1.48573588e+00],
                       [-2.45950581e+00, -7.73258073e-01,  6.12575009e-01],
                       [-7.01103435e-01, -2.25202354e+00,  1.24046812e+00],
                       [1.02463355e+00,  4.97990003e-01,  1.37153083e+00],
                       [2.35384432e+00, -3.74155335e-01,  5.30869282e-01],
                       [-3.20887510e-01,  2.64086433e+00, -1.83680405e+00],
                       [-8.80850553e-01,  2.15666343e+00, -2.23289208e-01],
                       [6.33042398e-01,  3.06954990e+00, -3.86069432e-01],
                       [-1.16202023e+00, -4.65744621e-01, -1.26738977e+00],
                       [-5.73862351e-01,  1.38603951e-01, -2.83008740e+00],
                       [1.86169467e-01, -1.28124151e+00, -2.08137103e+00],
                       [1.76270542e+00,  1.33669233e+00, -3.16631603e+00],
                       [2.88680187e+00,  1.80568382e+00, -1.85736351e+00],
                       [2.77317373e+00,  1.14588187e-01, -2.36501031e+00],
                       [-2.10012937e+00, -1.44655343e+00,  3.56788226e+00],
                       [-2.45580470e+00,  1.84556974e-01,  2.96737396e+00],
                       [-7.73579381e-01, -3.49597574e-01,  3.19703008e+00],
                       [-1.00671327e+00, -2.90521788e+00,  4.27075009e-01],
                       [2.16977805e-01, -1.74266599e+00,  9.34869204e-01],
                       [-4.94740426e-01, -2.82958969e+00,  2.13611841e+00]])

    symbols = np.array(['P', 'C', 'C', 'C', 'C', 'C', 'C', 'O', 'C', 'H', 'H', 'H', 'H',
                        'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                       dtype='<U1')

    start = TDStructure.from_coords_symbols(coords, symbols)

    end_coords = np.array([[-0.42718582,  0.20031172, -1.64880916],
                           [1.24564065,  0.26426699,  2.47485969],
                           [-0.47741102,  1.71570421, -0.62240075],
                           [-0.99357093,  0.71708118, -3.3148255],
                           [1.35514524, -0.18280879, -1.83200792],
                           [-1.16561892, -0.29981154,  2.41733394],
                           [0.28642142, -0.61693257,  2.23708559],
                           [-1.21221586, -0.91411914, -1.09683568],
                           [0.5652328, -1.99909903,  1.73582972],
                           [1.03381499,  1.25855173,  2.83448038],
                           [2.2877108,  0.02595162,  2.33601758],
                           [0.06700853,  2.53295105, -1.08373412],
                           [-1.51834952,  1.99659093, -0.47803498],
                           [-0.04002921,  1.4749508,  0.34445607],
                           [-2.03806731,  1.01143059, -3.23596566],
                           [-0.40571973,  1.5382459, -3.71138722],
                           [-0.9259007, -0.14343822, -3.97698307],
                           [1.90530438,  0.64595446, -2.26536374],
                           [1.74775878, -0.41238582, -0.84372877],
                           [1.45319671, -1.06414557, -2.4621738],
                           [-1.61554179, -0.98818519,  3.13181956],
                           [-1.68256071, -0.42625643,  1.46659614],
                           [-1.31116263,  0.71765225,  2.77282401],
                           [0.08674719, -2.13443361,  0.76653731],
                           [1.63338914, -2.17943707,  1.63738944],
                           [0.14596355, -2.73859046,  2.41702095]])
    end = TDStructure.from_coords_symbols(end_coords, symbols)

    tr = Trajectory([start, end]).run_geodesic(nimages=15)

    cni = ChainInputs(
        k=0.1,
        delta_k=0.09,
        do_parallel=True,
        node_freezing=True,
        skip_identical_graphs=True)

    tol = 0.001
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
        early_stop_force_thre=0.03*BOHR_TO_ANGSTROMS)
    initial_chain = Chain.from_traj(tr, parameters=cni)

    opt = VelocityProjectedOptimizer(timestep=0.5)
    gii = GIInputs(nimages=12)

    m = MSMEP(neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=opt)
    h = m.find_mep_multistep(initial_chain)
    assert len(h.ordered_leaves) == 2, f"MSMEP found incorrect number of elementary steps.\
          Found {len(h.ordered_leaves)}. Reference: 2"


if __name__ == "__main__":
    # test_tdstructure()
    # test_trajectory()
    test_neb()
    # test_msmep()
