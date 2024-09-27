from pathlib import Path

import numpy as np

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs, GIInputs, NEBInputs
from neb_dynamics.msmep import MSMEP
from neb_dynamics.neb import NEB
from neb_dynamics.optimizers.vpo import VelocityProjectedOptimizer
from neb_dynamics.engines import QCOPEngine, ThreeWellPotential, FlowerPotential
from neb_dynamics.nodes.node import XYNode
from neb_dynamics.engines import Engine

from qcio.models.inputs import ProgramInput

import matplotlib.pyplot as plt
from itertools import product
from matplotlib.animation import FuncAnimation
from neb_dynamics.nodes.node import StructureNode
from qcio import Structure
from neb_dynamics.engines.ase import ASEEngine

from xtb.ase.calculator import XTB


def animate_func(neb_obj: NEB, engine: Engine):
    en_func = engine._en_func
    chain_traj = neb_obj.chain_trajectory
    # plt.style.use("seaborn-pastel")

    figsize = 5
    s = 8

    f, ax = plt.subplots(figsize=(1.618 * figsize, figsize))

    min_val = -s
    max_val = s

    min_val = -4
    max_val = 4

    gridsize = 100
    x = np.linspace(start=min_val, stop=max_val, num=gridsize)
    h_flat_ref = np.array([en_func(pair) for pair in product(x, x)])
    h = h_flat_ref.reshape(gridsize, gridsize).T
    cs = plt.contourf(x, x, h)
    _ = f.colorbar(cs, ax=ax)

    (line,) = ax.plot([], [], "o--", lw=1)

    def animate(chain):

        x = chain.coordinates[:, 0]
        y = chain.coordinates[:, 1]

        line.set_data(x, y)

        return (line,)
        # return (x for x in all_arrows)

    _ = FuncAnimation(
        fig=f,
        func=animate,
        frames=chain_traj,
        blit=True,
        repeat_delay=1000,
        interval=200,
    )
    # anim.save(f'flower_nimages_{n_nodes}_k_{neb_obj.initial_chain.parameters.k}.gif')
    plt.show()


def test_engine():
    from neb_dynamics.chain import Chain
    from neb_dynamics.engines import QCOPEngine

    c = Chain.from_xyz(
        "/home/jdep/neb_dynamics/tests/test_msmep_results/node_0.xyz",
        parameters=ChainInputs(),
    )
    eng = QCOPEngine(
        program_input=ProgramInput(
            structure=c[0].structure, calctype="energy", model={"method": "GFN2xTB"}
        ),
        program="xtb",
    )
    geom0_before = c[0].structure.geometry
    geom1_before = c[1].structure.geometry
    eng.compute_gradients(c)
    geom0_after = c[0].structure.geometry
    geom1_after = c[1].structure.geometry

    assert np.array_equal(geom0_before, geom0_after) and np.array_equal(
        geom1_before, geom1_after
    ), "Engine is overwriting structures of chain."


def test_neb(test_data_dir: Path = Path("/home/jdep/neb_dynamics/tests")):
    # tr = Trajectory.from_xyz(test_data_dir / "test_traj.xyz")
    tol = 0.001
    cni = ChainInputs(k=0.01, delta_k=0.009, do_parallel=True, node_freezing=True)

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
        early_stop_force_thre=0.0,
    )  # *BOHR_TO_ANGSTROMS)
    initial_chain = Chain.from_xyz(test_data_dir / "test_traj.xyz", parameters=cni)
    calc = XTB(method="GFN2-xTB")
    all_engs = [
        QCOPEngine(
            program_input=ProgramInput(
                structure=initial_chain[0].structure,
                calctype="energy",
                model={"method": "GFN2xTB"},
            ),
            program="xtb",
        ),
        ASEEngine(calculator=calc),
    ]
    for eng in all_engs:
        opt = VelocityProjectedOptimizer(timestep=1.0)
        n = NEB(initial_chain=initial_chain, parameters=nbi, optimizer=opt, engine=eng)

        elem_step_output = n.optimize_chain()
        assert elem_step_output.is_elem_step, "Elementary step check failed."
        assert n.optimized is not None, "Chain did not converge."
        barrier = n.optimized.get_eA_chain()
        ref_barrier = 84.7500842709759
        assert np.isclose(
            barrier, ref_barrier
        ), f"NEB barrier was different. ref={ref_barrier}. test={barrier}"

        ts_guess = n.optimized.get_ts_node()
        ref_ts_guess = np.array(
            [
                [-1.24381420e00, -8.99695564e-03, 1.71365711e-03],
                [1.24190315e00, -2.04919066e-03, -2.78637072e-02],
                [-2.34107150e00, 3.13314392e-01, 1.70711067e00],
                [-2.41487259e00, -3.00865009e-01, -1.66025672e00],
                [2.39123943e00, 1.69435382e00, -2.10595863e-01],
                [2.39827597e00, -1.68798041e00, 1.98658082e-01],
            ]
        )

        assert np.allclose(
            ts_guess.coords, ref_ts_guess, rtol=1e-4, atol=1e-4
        ), f"TS guess has a different geometry than reference. ref={ref_ts_guess}. test= {ts_guess.coords=}"


def test_msmep(test_data_dir: Path = Path("/home/jdep/neb_dynamics/tests")):

    cni = ChainInputs(
        k=0.01,
        delta_k=0.009,
        do_parallel=True,
        node_freezing=True,
        friction_optimal_gi=False,
        node_ene_thre=20,
        node_rms_thre=5.0,
    )

    tol = 0.002
    nbi = NEBInputs(
        tol=tol,  # * BOHR_TO_ANGSTROMS,
        barrier_thre=0.1,  # kcalmol,
        climb=False,
        rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
        max_rms_grad_thre=tol * 2.5,  # * BOHR_TO_ANGSTROMS*2.5,
        ts_grad_thre=tol * 2.5,  # * BOHR_TO_ANGSTROMS,
        ts_spring_thre=tol * 1.5,  # * BOHR_TO_ANGSTROMS*3,
        v=1,
        max_steps=3000,
        early_stop_force_thre=0.03,
        skip_identical_graphs=True,
    )  # *BOHR_TO_ANGSTROMS)
    initial_chain = Chain.from_xyz(test_data_dir / "traj_msmep.xyz", parameters=cni)
    opt = VelocityProjectedOptimizer(timestep=1.0)
    gii = GIInputs(nimages=12)

    eng = QCOPEngine(
        program_input=ProgramInput(
            structure=initial_chain[0].structure,
            calctype="energy",
            model={"method": "GFN2xTB"},
        ),
        program="xtb",
    )

    m = MSMEP(
        neb_inputs=nbi, chain_inputs=cni, gi_inputs=gii, optimizer=opt, engine=eng
    )
    h = m.run_recursive_minimize(initial_chain)

    h.write_to_disk("./test_msmep_results")
    assert (
        len(h.ordered_leaves) == 2
    ), f"MSMEP found incorrect number of elementary steps.\
          Found {len(h.ordered_leaves)}. Reference: 2"
    print("NGRAD CALLS: ", h.get_num_grad_calls())


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
    cni = ChainInputs(
        k=ks, delta_k=0, node_class=XYNode, use_geodesic_interpolation=False
    )
    nbi = NEBInputs(tol=0.1, barrier_thre=5, v=True, max_steps=500, climb=False)
    chain = Chain(nodes=[XYNode(structure=xy) for xy in coords], parameters=cni)
    eng = FlowerPotential()  # ThreeWellPotential()
    opt = VelocityProjectedOptimizer(timestep=0.01)
    m = MSMEP(engine=eng, neb_inputs=nbi, chain_inputs=cni, optimizer=opt)
    # n = NEB(initial_chain=chain,
    #         parameters=nbi,
    #         optimizer=opt,
    #         engine=eng)
    # n.optimize_chain()
    history = m.find_mep_multistep(chain)
    raise NotImplementedError()


def test_ASE_engine():

    calc = XTB(method="GFN2-xTB")
    eng = ASEEngine(calculator=calc)
    # eng = QCOPEngine()

    # struct = Structure.from_smiles("COCO")
    struct = Structure(
        geometry=np.array(
            [
                [-2.23478339, -1.02590026, -0.87809266],
                [-1.18195154, 1.30916802, -0.08954108],
                [1.4014656, 1.55697051, -0.74685569],
                [2.95526011, 0.00830625, 0.7779886],
                [-2.07669008, -1.24135384, -2.92870436],
                [-1.30974372, -2.60855384, 0.07735988],
                [-4.2376824, -1.04001041, -0.3703406],
                [1.94699549, 3.52022059, -0.40162623],
                [1.72030678, 1.15344489, -2.75327353],
                [3.01682314, -1.63229191, -0.03342495],
            ]
        ),
        symbols=["C", "O", "C", "O", "H", "H", "H", "H", "H", "H"],
    )

    chain = Chain([StructureNode(structure=struct) for i in range(10)])

    reference = np.array([-15.4514] * 10)

    vals = eng.compute_energies(chain)

    assert np.allclose(
        vals, reference
    ), f"ASE Calculator giving incorrect energies, \n{reference=}\n{vals=}"


def test_msmep_pygsm(test_data_dir: Path = Path("/home/jdep/neb_dynamics/tests")):

    cni = ChainInputs(
        k=0.01,
        delta_k=0.009,
        do_parallel=True,
        node_freezing=True,
        friction_optimal_gi=False,
        node_rms_thre=5.0,
    )

    tol = 0.002
    nbi = NEBInputs(
        tol=tol,  # * BOHR_TO_ANGSTROMS,
        barrier_thre=0.1,  # kcalmol,
        climb=False,
        rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
        max_rms_grad_thre=tol * 2.5,  # * BOHR_TO_ANGSTROMS*2.5,
        ts_grad_thre=tol * 2.5,  # * BOHR_TO_ANGSTROMS,
        ts_spring_thre=tol * 1.5,  # * BOHR_TO_ANGSTROMS*3,
        v=1,
        max_steps=3000,
        early_stop_force_thre=0.03,
        skip_identical_graphs=True,
        pygsm_kwds={
            "conv_tol": 0.005,  # Convergence tolerance for optimizing nodes
            "conv_Ediff": 100.0,  # Energy difference convergence of optimization.
            "conv_gmax": 100.0,  # Max grad rms threshold
            "max_gsm_iterations": 100,
            "max_opt_steps": 3,
        },
    )  # *BOHR_TO_ANGSTROMS)
    initial_chain = Chain.from_xyz(test_data_dir / "traj_msmep.xyz", parameters=cni)
    opt = VelocityProjectedOptimizer(timestep=1.0)
    gii = GIInputs(nimages=12)

    calc = XTB(method="GFN2-xTB")
    eng = ASEEngine(calculator=calc)
    # eng = QCOPEngine()

    m = MSMEP(
        neb_inputs=nbi,
        chain_inputs=cni,
        gi_inputs=gii,
        optimizer=opt,
        engine=eng,
        path_min_method="pygsm",
    )
    h = m.run_recursive_minimize(initial_chain)
    assert (
        len(h.ordered_leaves) == 2
    ), f"MSMEP found incorrect number of elementary steps.\
          Found {len(h.ordered_leaves)}. Reference: 2"


def test_msmep_fneb(test_data_dir: Path = Path("/home/jdep/neb_dynamics/tests")):

    cni = ChainInputs(
        k=0.01,
        delta_k=0.009,
        do_parallel=True,
        node_freezing=True,
        friction_optimal_gi=False,
        node_rms_thre=5.0,
    )

    tol = 0.002
    nbi = NEBInputs(
        tol=tol,  # * BOHR_TO_ANGSTROMS,
        barrier_thre=0.1,  # kcalmol,
        climb=False,
        rms_grad_thre=tol,  # * BOHR_TO_ANGSTROMS,
        max_rms_grad_thre=tol * 2.5,  # * BOHR_TO_ANGSTROMS*2.5,
        ts_grad_thre=tol * 2.5,  # * BOHR_TO_ANGSTROMS,
        ts_spring_thre=tol * 1.5,  # * BOHR_TO_ANGSTROMS*3,
        v=1,
        max_steps=3000,
        early_stop_force_thre=0.03,
        skip_identical_graphs=True,
        fneb_kwds={
            "stepsize": 0.1,
            "ngradcalls": 3,
            "max_cycles": 500,
            "path_resolution": 0.5,  # BOHR
        },
    )  # *BOHR_TO_ANGSTROMS)
    initial_chain = Chain.from_xyz(test_data_dir / "traj_msmep.xyz", parameters=cni)
    opt = VelocityProjectedOptimizer(timestep=1.0)
    gii = GIInputs(nimages=12)

    # calc = XTB(method="GFN2-xTB")
    # eng = ASEEngine(calculator=calc)
    eng = QCOPEngine()

    m = MSMEP(
        neb_inputs=nbi,
        chain_inputs=cni,
        gi_inputs=gii,
        optimizer=opt,
        engine=eng,
        path_min_method="fneb",
    )
    h = m.run_recursive_minimize(initial_chain)
    # this method finds a conformer-conformer rearrangement that changes
    # stereochemistry. This is a nice find, so this method should find 3
    # steps for this reaction.
    assert (
        len(h.ordered_leaves) == 3
    ), f"MSMEP found incorrect number of elementary steps.\
          Found {len(h.ordered_leaves)}. Reference: 3"
    print("NGRAD CALLS: ", h.get_num_grad_calls())


if __name__ == "__main__":
    # test_engine()
    # test_neb()
    # test_msmep()
    # test_msmep_pygsm()
    test_msmep_fneb()
    test_ASE_engine()
    # test_2d_neb()
