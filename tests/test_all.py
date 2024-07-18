from pathlib import Path

import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics.constants import ANGSTROM_TO_BOHR
from neb_dynamics.inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.MSMEP import MSMEP
from neb_dynamics.neb import NEB
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.tdstructure import TDStructure
from neb_dynamics.trajectory import Trajectory
from neb_dynamics.engine import QCOPEngine

from qcio.models.inputs import ProgramInput


def test_engine():
    from neb_dynamics.chain import Chain
    from neb_dynamics.engine import QCOPEngine
    c = Chain.from_xyz(
        "/home/jdep/T3D_data/AutoMG_v0/msmep_results/results_pair149_msmep.xyz", parameters=ChainInputs())
    eng = QCOPEngine(program_input=ProgramInput(
        structure=c[0].structure, calctype='energy', model={'method': "GFN2xTB"}), program='xtb')
    geom0_before = c[0].structure.geometry
    geom1_before = c[1].structure.geometry
    eng.compute_gradients(c)
    geom0_after = c[0].structure.geometry
    geom1_after = c[1].structure.geometry

    assert np.array_equal(geom0_before and geom0_after) and np.array_equal(
        geom1_before, geom1_after), "Engine is overwriting structures of chain."


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
    # tr = Trajectory.from_xyz(test_data_dir / "test_traj.xyz")
    tol = 0.001
    cni = ChainInputs(
        k=0.01,
        delta_k=0.009,
        do_parallel=True,
        node_freezing=True)

    nbi = NEBInputs(
        tol=tol,  # * BOHR_TO_ANGSTROMS,
        barrier_thre=0.1,  # kcalmol,
        climb=False,

        rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
        max_rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS*2.5,
        ts_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
        ts_spring_thre=tol,  # * BOHR_TO_ANGSTROMS*3,

        v=1,
        max_steps=3000,
        early_stop_force_thre=0.0)  # *BOHR_TO_ANGSTROMS)
    initial_chain = Chain.from_xyz(
        test_data_dir / "test_traj.xyz", parameters=cni)

    # symbols = tr.symbols
    eng = QCOPEngine(program_input=ProgramInput(
        structure=initial_chain[0].structure,
        calctype='energy', model={'method': "GFN2xTB"}), program='xtb')

    opt = VelocityProjectedOptimizer(timestep=1.0)
    n = NEB(initial_chain=initial_chain, parameters=nbi, optimizer=opt,
            engine=eng)

    elem_step_output = n.optimize_chain()
    assert elem_step_output.is_elem_step, "Elementary step check failed."
    assert n.optimized is not None, "Chain did not converge."
    barrier = n.optimized.get_eA_chain()
    ref_barrier = 84.7500842709759
    assert np.isclose(
        barrier, ref_barrier), f'NEB barrier was different. ref={ref_barrier}. test={barrier}'

    ts_guess = n.optimized.get_ts_node()
    ref_ts_guess = np.array([[-1.24381420e+00, -8.99695564e-03,  1.71365711e-03],
                             [1.24190315e+00, -2.04919066e-03, -2.78637072e-02],
                             [-2.34107150e+00,  3.13314392e-01,  1.70711067e+00],
                             [-2.41487259e+00, -3.00865009e-01, -1.66025672e+00],
                             [2.39123943e+00,  1.69435382e+00, -2.10595863e-01],
                             [2.39827597e+00, -1.68798041e+00,  1.98658082e-01]])

    assert np.allclose(ts_guess.coords, ref_ts_guess), \
        f"TS guess has a different geometry than reference. ref={ref_ts_guess}. test= {ts_guess.coords=}"


def test_msmep(test_data_dir: Path = Path("/home/jdep/neb_dynamics/tests")):

    cni = ChainInputs(
        k=0.01,
        delta_k=0.009,
        do_parallel=True,
        node_freezing=True,
        friction_optimal_gi=False)

    tol = 0.001
    nbi = NEBInputs(
        tol=tol,  # * BOHR_TO_ANGSTROMS,
        barrier_thre=0.1,  # kcalmol,
        climb=False,

        rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
        max_rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS*2.5,
        ts_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
        ts_spring_thre=tol,  # * BOHR_TO_ANGSTROMS*3,

        v=1,
        max_steps=3000,
        early_stop_force_thre=0.03,
        skip_identical_graphs=True)  # *BOHR_TO_ANGSTROMS)
    initial_chain = Chain.from_xyz(
        test_data_dir/'traj_msmep.xyz', parameters=cni)
    opt = VelocityProjectedOptimizer(timestep=1.0)
    gii = GIInputs(nimages=12)

    eng = QCOPEngine(program_input=ProgramInput(
        structure=initial_chain[0].structure,
        calctype='energy', model={'method': "GFN2xTB"}), program='xtb')

    m = MSMEP(neb_inputs=nbi, chain_inputs=cni,
              gi_inputs=gii, optimizer=opt, engine=eng)
    h = m.find_mep_multistep(initial_chain)
    assert len(h.ordered_leaves) == 2, f"MSMEP found incorrect number of elementary steps.\
          Found {len(h.ordered_leaves)}. Reference: 2"


if __name__ == "__main__":
    # test_tdstructure()
    # test_trajectory()
    test_neb()
    test_msmep()
