from types import SimpleNamespace

import numpy as np
from qcio import Structure

from neb_dynamics.chain import Chain
from neb_dynamics.inputs import ChainInputs
from neb_dynamics.nodes.node import StructureNode
from neb_dynamics.scripts import main_cli


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def _node_at_x(x: float, energy: float | None = None) -> StructureNode:
    node = StructureNode(structure=_structure_at_x(x))
    node.has_molecular_graph = False
    node.graph = None
    if energy is not None:
        node._cached_energy = float(energy)
        node._cached_gradient = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return node


def _chain_from_xs(xs: list[float], params: ChainInputs, energies: list[float] | None = None) -> Chain:
    if energies is None:
        nodes = [_node_at_x(x) for x in xs]
    else:
        nodes = [_node_at_x(x, e) for x, e in zip(xs, energies)]
    return Chain.model_validate({"nodes": nodes, "parameters": params})


class _FakeEngine:
    def __init__(self, shift: float):
        self.shift = shift

    def compute_geometry_optimization(self, node: StructureNode, keywords=None):
        moved = node.update_coords(node.coords + np.array([[self.shift, 0.0, 0.0]]))
        moved._cached_energy = float(np.linalg.norm(moved.coords[1] - moved.coords[0]))
        moved._cached_gradient = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        return [moved]


class _FakeMSMEP:
    expensive_pair_inputs: list[Chain] = []

    def __init__(self, inputs):
        self.inputs = inputs

    def run_minimize_chain(self, input_chain: Chain):
        if self.inputs.label == "cheap":
            cheap_chain = _chain_from_xs(
                [1.0, 10.0, 20.0, 30.0, 40.0],
                self.inputs.chain_inputs,
                energies=[0.0, 5.0, 1.0, 6.0, 0.0],
            )
            neb = SimpleNamespace(chain_trajectory=[cheap_chain], optimized=cheap_chain)
            return neb, None

        _FakeMSMEP.expensive_pair_inputs.append(input_chain.copy())
        out_chain = input_chain.copy()
        for node in out_chain.nodes:
            node._cached_energy = float(np.linalg.norm(node.coords[1] - node.coords[0]))
            node._cached_gradient = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        neb = SimpleNamespace(
            chain_trajectory=[out_chain], optimized=out_chain
        )
        return neb, None


def _make_fake_runinputs(label: str, shift: float, nimages: int = 3):
    return SimpleNamespace(
        label=label,
        engine_name="chemcloud",
        program="crest" if label == "cheap" else "terachem",
        path_min_method="NEB",
        chain_inputs=ChainInputs(),
        gi_inputs=SimpleNamespace(nimages=nimages),
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        engine=_FakeEngine(shift=shift),
        program_kwds={},
        optimizer_kwds={},
    )


def _run_refine_with_mocks(monkeypatch, tmp_path, recycle_nodes: bool):
    cheap_inputs = _make_fake_runinputs("cheap", shift=0.0, nimages=3)
    expensive_inputs = _make_fake_runinputs("expensive", shift=1.0, nimages=3)

    def _fake_open(path: str):
        return cheap_inputs if "cheap" in path else expensive_inputs

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(_fake_open))
    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: [_structure_at_x(1.0), _structure_at_x(40.0)],
    )
    monkeypatch.setattr(main_cli, "_ascii_profile_for_chain", lambda chain: None)
    _FakeMSMEP.expensive_pair_inputs = []

    monkeypatch.chdir(tmp_path)
    main_cli.run_refine(
        geometries="dummy.xyz",
        inputs="expensive.toml",
        cheap_inputs="cheap.toml",
        recursive=False,
        recycle_nodes=recycle_nodes,
        name="refine_case",
    )

    return list(_FakeMSMEP.expensive_pair_inputs)


def test_run_refine_nonrecursive_builds_adjacent_pair_paths(monkeypatch, tmp_path):
    pair_inputs = _run_refine_with_mocks(monkeypatch, tmp_path, recycle_nodes=False)

    assert len(pair_inputs) == 2
    assert len(pair_inputs[0].nodes) == 2
    assert len(pair_inputs[1].nodes) == 2
    assert (tmp_path / "refine_case_cheap.xyz").exists()
    assert (tmp_path / "refine_case_refined_minima.xyz").exists()
    assert (tmp_path / "refine_case_refined.xyz").exists()
    assert (tmp_path / "refine_case_refined_pairs" / "pair_0_1.xyz").exists()
    assert (tmp_path / "refine_case_refined_pairs" / "pair_1_2.xyz").exists()


def test_run_refine_recycle_nodes_seeds_expensive_pairs_from_cheap_chain(monkeypatch, tmp_path):
    pair_inputs = _run_refine_with_mocks(monkeypatch, tmp_path, recycle_nodes=True)

    assert len(pair_inputs) == 2
    assert len(pair_inputs[0].nodes) == 3
    assert len(pair_inputs[1].nodes) == 3

    bond_lengths0 = [
        float(np.linalg.norm(node.coords[1] - node.coords[0]))
        for node in pair_inputs[0].nodes
    ]
    bond_lengths1 = [
        float(np.linalg.norm(node.coords[1] - node.coords[0]))
        for node in pair_inputs[1].nodes
    ]
    assert bond_lengths0 == [1.0, 10.0, 20.0]
    assert bond_lengths1 == [20.0, 30.0, 40.0]
