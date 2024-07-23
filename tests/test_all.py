from pathlib import Path

import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.msmep import MSMEP
from neb_dynamics.neb import NEB
from neb_dynamics.optimizers.VPO import VelocityProjectedOptimizer
from neb_dynamics.engine import QCOPEngine, ThreeWellPotential, FlowerPotential
from neb_dynamics.nodes.node import XYNode

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
    n = NEB(initial_chain=initial_chain,
            parameters=nbi,
            optimizer=opt,
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

    tol = 0.002
    nbi = NEBInputs(
        tol=tol,  # * BOHR_TO_ANGSTROMS,
        barrier_thre=0.1,  # kcalmol,
        climb=False,

        rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
        max_rms_grad_thre=tol*2.5,  # * BOHR_TO_ANGSTROMS*2.5,
        ts_grad_thre=tol*2.5,  # * BOHR_TO_ANGSTROMS,
        ts_spring_thre=tol*1.5,  # * BOHR_TO_ANGSTROMS*3,

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
    h.write_to_disk("./test_msmep_results")
    assert len(h.ordered_leaves) == 2, f"MSMEP found incorrect number of elementary steps.\
          Found {len(h.ordered_leaves)}. Reference: 2"


def test_2d_neb():

    nimages = 15

    # inital guess ThreeWellPotential
    # end_point = (3.00002182, 1.99995542)
    # start_point = (-3.77928812, -3.28320392)

    # inital guess FlowerPotential
    start_point = [-2.59807434, -1.499999]
    end_point = [2.5980755, 1.49999912]

    coords = np.linspace(start_point, end_point, nimages)
    coords[1:-1] += [-1, 1]

    ks = 0.1
    cni = ChainInputs(k=ks, delta_k=0, node_class=XYNode)
    nbi = NEBInputs(tol=0.1, barrier_thre=5, v=True,
                    max_steps=500, climb=False)
    chain = Chain(nodes=[XYNode(structure=xy)
                  for xy in coords], parameters=cni)
    eng = FlowerPotential()  # ThreeWellPotential()
    opt = VelocityProjectedOptimizer(timestep=0.01)
    n = NEB(initial_chain=chain,
            parameters=nbi,
            optimizer=opt,
            engine=eng)
    n.optimize_chain()


if __name__ == "__main__":
    # test_tdstructure()
    # test_trajectory()
    # test_neb()
    # test_msmep()
    test_2d_neb()
