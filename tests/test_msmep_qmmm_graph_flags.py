import concurrent.futures
from types import SimpleNamespace

import numpy as np
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.msmep import MSMEP
import neb_dynamics.msmep as msmep_module
from neb_dynamics.nodes.node import StructureNode


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def test_qmmm_run_minimize_disables_molecular_graphs(monkeypatch):
    class QMMMEngine:
        def compute_gradients(self, chain):
            for node in chain:
                node._cached_gradient = np.zeros_like(node.coords)
            return np.array([node._cached_gradient for node in chain])

    inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(v=False),
        chain_inputs=ChainInputs(),
        gi_inputs=SimpleNamespace(nimages=2),
        optimizer_kwds={"name": "cg", "timestep": 0.1},
        engine=QMMMEngine(),
    )

    m = MSMEP(inputs=inputs)
    nodes = [StructureNode(structure=_structure_at_x(0.7)), StructureNode(structure=_structure_at_x(1.3))]
    chain = Chain.model_validate({"nodes": nodes, "parameters": inputs.chain_inputs})

    # Ensure the test starts with graphing enabled.
    assert all(node.has_molecular_graph for node in chain)

    class _FakeMinimizer:
        def __init__(self, initial_chain):
            self.initial_chain = initial_chain
            self.optimized = initial_chain
            self.chain_trajectory = [initial_chain]

        def optimize_chain(self):
            return SimpleNamespace(is_elem_step=True)

    monkeypatch.setattr(MSMEP, "_create_interpolation", lambda self, c: c)
    monkeypatch.setattr(MSMEP, "_construct_path_minimizer", lambda self, initial_chain: _FakeMinimizer(initial_chain))

    m.run_minimize_chain(chain)

    assert all(node.has_molecular_graph is False for node in chain)
    assert all(node.graph is None for node in chain)


def test_parallel_worker_forces_non_verbose_for_live_monitor_stability(monkeypatch):
    worker_inputs = SimpleNamespace(path_min_inputs=SimpleNamespace(v=True))
    captured = {}

    monkeypatch.setattr(
        msmep_module,
        "_clone_run_inputs_for_worker",
        lambda _run_inputs: worker_inputs,
    )

    class _FakeRunner:
        def __init__(self, inputs):
            self.inputs = inputs
            captured["init_v"] = self.inputs.path_min_inputs.v

        def _run_recursive_step(self, input_chain, tree_node_index):
            captured["step_v"] = self.inputs.path_min_inputs.v
            return ("history", [])

    monkeypatch.setattr(msmep_module, "MSMEP", _FakeRunner)

    out = msmep_module._parallel_recursive_step_worker(
        run_inputs=SimpleNamespace(),
        input_chain=object(),
        tree_node_index=4,
    )

    assert out == ("history", [])
    assert captured["init_v"] is False
    assert captured["step_v"] is False


def test_parallel_scheduler_marks_child_monitors_active_and_inactive(monkeypatch):
    inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(v=False),
        chain_inputs=ChainInputs(),
        gi_inputs=SimpleNamespace(nimages=2),
        optimizer_kwds={"name": "cg", "timestep": 0.1},
        engine=SimpleNamespace(),
    )
    m = MSMEP(inputs=inputs)
    nodes = [StructureNode(structure=_structure_at_x(0.1)), StructureNode(structure=_structure_at_x(0.9))]
    chain = Chain.model_validate({"nodes": nodes, "parameters": inputs.chain_inputs})

    root_data = SimpleNamespace(chain_trajectory=[chain], optimized=chain)
    root_history = msmep_module.TreeNode(data=root_data, children=[], index=0)

    monkeypatch.setattr(
        MSMEP,
        "_run_recursive_step",
        lambda self, input_chain, tree_node_index: (root_history, [chain, chain]),
    )

    def _fake_worker(run_inputs, input_chain, tree_node_index):
        data = SimpleNamespace(chain_trajectory=[chain], optimized=chain)
        return msmep_module.TreeNode(data=data, children=[], index=tree_node_index), []

    monkeypatch.setattr(msmep_module, "_parallel_recursive_step_worker", _fake_worker)

    class _FakePrinter:
        def __init__(self):
            self.active = []
            self.inactive = []
            self.path_updates = 0

        def clear_path_so_far(self):
            return None

        def mark_monitor_active(self, monitor_id):
            self.active.append(monitor_id)

        def mark_monitor_inactive(self, monitor_id):
            self.inactive.append(monitor_id)

        def update_path_so_far(self, chain, caption=""):
            self.path_updates += 1

    fake_printer = _FakePrinter()
    monkeypatch.setattr(msmep_module, "get_progress_printer", lambda: fake_printer)

    history = m.run_parallel_recursive_minimize(chain, max_workers=2)

    assert history is root_history
    assert set(fake_printer.active) >= {"branch-1", "branch-2"}
    assert set(fake_printer.inactive) >= {"branch-1", "branch-2"}
    assert fake_printer.path_updates >= 1


def test_parallel_scheduler_honors_explicit_worker_count_when_cpu_count_is_low(
    monkeypatch,
):
    inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(v=False),
        chain_inputs=ChainInputs(),
        gi_inputs=SimpleNamespace(nimages=2),
        optimizer_kwds={"name": "cg", "timestep": 0.1},
        engine=SimpleNamespace(),
    )
    m = MSMEP(inputs=inputs)
    nodes = [StructureNode(structure=_structure_at_x(0.2)), StructureNode(structure=_structure_at_x(1.2))]
    chain = Chain.model_validate({"nodes": nodes, "parameters": inputs.chain_inputs})

    root_data = SimpleNamespace(chain_trajectory=[chain], optimized=chain)
    root_history = msmep_module.TreeNode(data=root_data, children=[], index=0)

    monkeypatch.setattr(
        MSMEP,
        "_run_recursive_step",
        lambda self, input_chain, tree_node_index: (root_history, [chain, chain]),
    )

    def _fake_worker(run_inputs, input_chain, tree_node_index):
        data = SimpleNamespace(chain_trajectory=[chain], optimized=chain)
        return msmep_module.TreeNode(data=data, children=[], index=tree_node_index), []

    monkeypatch.setattr(msmep_module, "_parallel_recursive_step_worker", _fake_worker)
    monkeypatch.setattr(msmep_module.os, "cpu_count", lambda: 1)

    captured: dict[str, int] = {}

    class _FakeExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = int(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            fut = concurrent.futures.Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as exc:  # pragma: no cover
                fut.set_exception(exc)
            return fut

    monkeypatch.setattr(
        msmep_module.concurrent.futures, "ThreadPoolExecutor", _FakeExecutor
    )

    class _FakePrinter:
        def clear_path_so_far(self):
            return None

        def mark_monitor_active(self, monitor_id):
            return None

        def mark_monitor_inactive(self, monitor_id):
            return None

        def update_path_so_far(self, chain, caption=""):
            return None

    monkeypatch.setattr(msmep_module, "get_progress_printer", lambda: _FakePrinter())

    m.run_parallel_recursive_minimize(chain, max_workers=50)

    assert captured["max_workers"] == 50
