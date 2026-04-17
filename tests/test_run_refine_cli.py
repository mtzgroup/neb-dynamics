from types import SimpleNamespace
from pathlib import Path

import numpy as np
from qcio import Structure
import pytest

from neb_dynamics.chain import Chain
from neb_dynamics.constants import ANGSTROM_TO_BOHR
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
        self.batch_call_sizes: list[int] = []
        self.batch_call_keywords: list[object] = []

    def compute_geometry_optimization(self, node: StructureNode, keywords=None):
        moved = node.update_coords(node.coords + np.array([[self.shift, 0.0, 0.0]]))
        moved._cached_energy = float(np.linalg.norm(moved.coords[1] - moved.coords[0]))
        moved._cached_gradient = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        return [moved]

    def compute_geometry_optimizations(self, nodes: list[StructureNode], keywords=None):
        self.batch_call_sizes.append(len(nodes))
        self.batch_call_keywords.append(keywords)
        return [
            self.compute_geometry_optimization(node=node, keywords=keywords)
            for node in nodes
        ]


class _FakeMSMEP:
    expensive_pair_inputs: list[Chain] = []
    expensive_parallel_pair_inputs: list[Chain] = []
    parallel_workers_seen: list[int] = []

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

    def run_recursive_minimize(self, input_chain: Chain):
        if self.inputs.label == "cheap":
            cheap_chain = _chain_from_xs(
                [1.0, 10.0, 20.0, 30.0, 40.0],
                self.inputs.chain_inputs,
                energies=[0.0, 5.0, 1.0, 6.0, 0.0],
            )

            class _FakeHistory:
                data = True
                output_chain = cheap_chain
                ordered_leaves = []

                def write_to_disk(self, _path):
                    return None

            return _FakeHistory()

        _FakeMSMEP.expensive_pair_inputs.append(input_chain.copy())

        class _FakeExpensiveHistory:
            data = True
            output_chain = input_chain.copy()

            def write_to_disk(self, _path):
                return None

        return _FakeExpensiveHistory()

    def run_parallel_recursive_minimize(self, input_chain: Chain, max_workers: int):
        _FakeMSMEP.expensive_parallel_pair_inputs.append(input_chain.copy())
        _FakeMSMEP.parallel_workers_seen.append(int(max_workers))

        class _FakeExpensiveParallelHistory:
            data = True
            output_chain = input_chain.copy()

            def write_to_disk(self, _path):
                return None

        return _FakeExpensiveParallelHistory()


def _fake_ts_irc_summary(source_workspace: str, refined_workspace: str, inputs_fp: str) -> dict:
    return {
        "source_workspace": source_workspace,
        "workspace": refined_workspace,
        "inputs_fp": inputs_fp,
        "edges_scanned": 1,
        "edges_with_chains": 1,
        "ts_guesses_attempted": 1,
        "ts_jobs_submitted": 1,
        "ts_converged": 1,
        "ts_failed": 0,
        "ts_failure_log": "",
        "ts_top_errors": [],
        "irc_jobs_submitted": 1,
        "irc_converged": 1,
        "irc_failed": 0,
        "irc_failure_log": "",
        "irc_top_errors": [],
        "added_nodes": 2,
        "added_edges": 1,
        "final_nodes": 2,
        "final_edges": 1,
        "queue_items": 1,
        "artifacts_dir": str(Path(refined_workspace) / "ts_irc_refinement"),
        "chemcloud_parallel": False,
        "use_bigchem": False,
    }


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
    _FakeMSMEP.expensive_parallel_pair_inputs = []
    _FakeMSMEP.parallel_workers_seen = []

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


def test_run_refine_network_splits_refines_only_best_path(monkeypatch, tmp_path):
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
    monkeypatch.setattr(
        main_cli,
        "_run_recursive_network_splits",
        lambda **kwargs: ([], tmp_path / "fake_network.json", tmp_path / "fake_manifest.json"),
    )
    monkeypatch.setattr(
        main_cli,
        "_load_best_path_chain_from_network_splits",
        lambda **kwargs: _chain_from_xs([1.0, 20.0, 40.0], cheap_inputs.chain_inputs, energies=[0.0, 1.0, 0.0]),
    )
    _FakeMSMEP.expensive_pair_inputs = []
    _FakeMSMEP.expensive_parallel_pair_inputs = []
    _FakeMSMEP.parallel_workers_seen = []

    monkeypatch.chdir(tmp_path)
    main_cli.run_refine(
        geometries="dummy.xyz",
        inputs="expensive.toml",
        cheap_inputs="cheap.toml",
        recursive=True,
        network_splits=True,
        recycle_nodes=False,
        name="refine_network_case",
    )

    pair_inputs = list(_FakeMSMEP.expensive_pair_inputs)
    assert len(pair_inputs) == 2
    pair_lengths = [
        [float(np.linalg.norm(node.coords[1] - node.coords[0])) for node in pair.nodes]
        for pair in pair_inputs
    ]
    assert pair_lengths == [[1.0, 20.0], [20.0, 40.0]]


def test_run_refine_chemcloud_uses_batched_minima_optimization(monkeypatch, tmp_path):
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
    _FakeMSMEP.expensive_parallel_pair_inputs = []
    _FakeMSMEP.parallel_workers_seen = []

    monkeypatch.chdir(tmp_path)
    main_cli.run_refine(
        geometries="dummy.xyz",
        inputs="expensive.toml",
        cheap_inputs="cheap.toml",
        recursive=False,
        recycle_nodes=False,
        name="refine_batch_case",
    )

    assert expensive_inputs.engine.batch_call_sizes == [3]


def test_refine_uses_precomputed_treenode_source(monkeypatch, tmp_path):
    expensive_inputs = _make_fake_runinputs("expensive", shift=1.0, nimages=3)

    monkeypatch.setattr(
        main_cli.RunInputs, "open", staticmethod(lambda _path: expensive_inputs)
    )
    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(main_cli, "_ascii_profile_for_chain", lambda chain: None)

    cheap_chain = _chain_from_xs(
        [1.0, 10.0, 20.0, 30.0, 40.0],
        ChainInputs(),
        energies=[0.0, 5.0, 1.0, 6.0, 0.0],
    )
    fake_tree = main_cli.TreeNode(
        data=SimpleNamespace(chain_trajectory=[cheap_chain], optimized=cheap_chain),
        children=[],
        index=0,
    )
    monkeypatch.setattr(
        main_cli,
        "_load_precomputed_refine_source",
        lambda **kwargs: fake_tree,
    )
    _FakeMSMEP.expensive_pair_inputs = []
    _FakeMSMEP.expensive_parallel_pair_inputs = []
    _FakeMSMEP.parallel_workers_seen = []

    monkeypatch.chdir(tmp_path)
    main_cli.refine(
        source="precomputed_source",
        inputs="expensive.toml",
        recursive=False,
        recycle_nodes=False,
        name="refine_tree_case",
    )

    pair_inputs = list(_FakeMSMEP.expensive_pair_inputs)
    assert len(pair_inputs) == 1
    assert len(pair_inputs[0].nodes) == 2
    assert (tmp_path / "refine_tree_case_cheap.xyz").exists()
    assert (tmp_path / "refine_tree_case_refined_minima.xyz").exists()
    assert (tmp_path / "refine_tree_case_refined.xyz").exists()


def test_refine_uses_precomputed_neb_source_with_recycled_pairs(monkeypatch, tmp_path):
    expensive_inputs = _make_fake_runinputs("expensive", shift=1.0, nimages=3)

    monkeypatch.setattr(
        main_cli.RunInputs, "open", staticmethod(lambda _path: expensive_inputs)
    )
    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(main_cli, "_ascii_profile_for_chain", lambda chain: None)

    cheap_chain = _chain_from_xs(
        [1.0, 10.0, 20.0, 30.0, 40.0],
        ChainInputs(),
        energies=[0.0, 5.0, 1.0, 6.0, 0.0],
    )
    fake_neb = object.__new__(main_cli.NEB)
    fake_neb.chain_trajectory = [cheap_chain]
    fake_neb.optimized = cheap_chain
    monkeypatch.setattr(
        main_cli,
        "_load_precomputed_refine_source",
        lambda **kwargs: fake_neb,
    )
    _FakeMSMEP.expensive_pair_inputs = []
    _FakeMSMEP.expensive_parallel_pair_inputs = []
    _FakeMSMEP.parallel_workers_seen = []

    monkeypatch.chdir(tmp_path)
    main_cli.refine(
        source="precomputed_neb.xyz",
        inputs="expensive.toml",
        recursive=False,
        recycle_nodes=True,
        name="refine_neb_case",
    )

    pair_inputs = list(_FakeMSMEP.expensive_pair_inputs)
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


def test_refine_uses_network_best_path_nodes_as_minima(monkeypatch, tmp_path):
    expensive_inputs = _make_fake_runinputs("expensive", shift=1.0, nimages=3)

    monkeypatch.setattr(
        main_cli.RunInputs, "open", staticmethod(lambda _path: expensive_inputs)
    )
    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(main_cli, "_ascii_profile_for_chain", lambda chain: None)

    best_path_chain = _chain_from_xs(
        [1.0, 10.0, 20.0, 30.0, 40.0],
        ChainInputs(),
        energies=[0.0, 5.0, 1.0, 6.0, 0.0],
    )
    monkeypatch.setattr(
        main_cli,
        "_load_precomputed_refine_source",
        lambda **kwargs: best_path_chain,
    )
    _FakeMSMEP.expensive_pair_inputs = []
    _FakeMSMEP.expensive_parallel_pair_inputs = []
    _FakeMSMEP.parallel_workers_seen = []

    monkeypatch.chdir(tmp_path)
    main_cli.refine(
        source="network.json",
        inputs="expensive.toml",
        recursive=False,
        recycle_nodes=False,
        name="refine_network_case",
    )

    pair_inputs = list(_FakeMSMEP.expensive_pair_inputs)
    assert len(pair_inputs) == 4
    assert [len(pair.nodes) for pair in pair_inputs] == [2, 2, 2, 2]


def test_refine_chemcloud_uses_batched_minima_optimization(monkeypatch, tmp_path):
    expensive_inputs = _make_fake_runinputs("expensive", shift=1.0, nimages=3)

    monkeypatch.setattr(
        main_cli.RunInputs, "open", staticmethod(lambda _path: expensive_inputs)
    )
    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(main_cli, "_ascii_profile_for_chain", lambda chain: None)

    cheap_chain = _chain_from_xs(
        [1.0, 10.0, 20.0, 30.0, 40.0],
        ChainInputs(),
        energies=[0.0, 5.0, 1.0, 6.0, 0.0],
    )
    fake_neb = object.__new__(main_cli.NEB)
    fake_neb.chain_trajectory = [cheap_chain]
    fake_neb.optimized = cheap_chain
    monkeypatch.setattr(
        main_cli,
        "_load_precomputed_refine_source",
        lambda **kwargs: fake_neb,
    )
    _FakeMSMEP.expensive_pair_inputs = []
    _FakeMSMEP.expensive_parallel_pair_inputs = []
    _FakeMSMEP.parallel_workers_seen = []

    monkeypatch.chdir(tmp_path)
    main_cli.refine(
        source="precomputed_neb.xyz",
        inputs="expensive.toml",
        recursive=False,
        recycle_nodes=False,
        name="refine_batch_case",
    )

    assert expensive_inputs.engine.batch_call_sizes == [3]


def test_refine_parallel_uses_parallel_recursive_minimizer(monkeypatch, tmp_path):
    expensive_inputs = _make_fake_runinputs("expensive", shift=1.0, nimages=3)

    monkeypatch.setattr(
        main_cli.RunInputs, "open", staticmethod(lambda _path: expensive_inputs)
    )
    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(main_cli, "_ascii_profile_for_chain", lambda chain: None)

    cheap_chain = _chain_from_xs(
        [1.0, 10.0, 20.0, 30.0, 40.0],
        ChainInputs(),
        energies=[0.0, 5.0, 1.0, 6.0, 0.0],
    )
    fake_neb = object.__new__(main_cli.NEB)
    fake_neb.chain_trajectory = [cheap_chain]
    fake_neb.optimized = cheap_chain
    monkeypatch.setattr(
        main_cli,
        "_load_precomputed_refine_source",
        lambda **kwargs: fake_neb,
    )
    _FakeMSMEP.expensive_pair_inputs = []
    _FakeMSMEP.expensive_parallel_pair_inputs = []
    _FakeMSMEP.parallel_workers_seen = []

    monkeypatch.chdir(tmp_path)
    main_cli.refine(
        source="precomputed_neb.xyz",
        inputs="expensive.toml",
        parallel=True,
        parallel_workers=7,
        recycle_nodes=False,
        name="refine_parallel_case",
    )

    assert len(_FakeMSMEP.expensive_pair_inputs) == 0
    assert len(_FakeMSMEP.expensive_parallel_pair_inputs) == 2
    assert _FakeMSMEP.parallel_workers_seen == [7, 7]


def test_refine_rejects_parallel_with_recursive(monkeypatch, tmp_path):
    expensive_inputs = _make_fake_runinputs("expensive", shift=1.0, nimages=3)
    monkeypatch.setattr(
        main_cli.RunInputs, "open", staticmethod(lambda _path: expensive_inputs)
    )

    with pytest.raises(main_cli.typer.BadParameter):
        main_cli.refine(
            source=str(tmp_path / "dummy_source"),
            inputs="expensive.toml",
            recursive=True,
            parallel=True,
        )


def test_load_precomputed_refine_source_reads_drive_workspace(tmp_path):
    workspace = main_cli.RetropathsWorkspace(
        workdir=str(tmp_path / "drive_workspace"),
        run_name="drive_workspace",
        root_smiles="CC",
        environment_smiles="",
        inputs_fp=str(tmp_path / "inputs.toml"),
    )
    workspace.write()

    out = main_cli._load_precomputed_refine_source(
        source=workspace.directory,
        charge=0,
        multiplicity=1,
    )
    assert isinstance(out, main_cli.RetropathsWorkspace)
    assert out.directory == workspace.directory


def test_refine_ts_irc_uses_workspace_source(monkeypatch, tmp_path):
    inputs_fp = tmp_path / "refine.toml"
    inputs_fp.write_text("engine_name='chemcloud'\n", encoding="utf-8")

    source_workspace = main_cli.RetropathsWorkspace(
        workdir=str(tmp_path / "source_workspace"),
        run_name="source_workspace",
        root_smiles="CC",
        environment_smiles="",
        inputs_fp=str(inputs_fp),
    )
    source_workspace.write()

    call_args = {}

    def _fake_refine_drive_workspace_network(
        workspace,
        *,
        refinement_inputs_fp,
        refined_workspace_dir=None,
        refined_run_name=None,
        use_bigchem=None,
        write_status_html_output=False,
        progress=None,
    ):
        call_args["workspace"] = workspace
        call_args["refinement_inputs_fp"] = Path(refinement_inputs_fp)
        call_args["refined_workspace_dir"] = refined_workspace_dir
        call_args["refined_run_name"] = refined_run_name
        call_args["use_bigchem"] = use_bigchem
        call_args["write_status_html_output"] = write_status_html_output
        return _fake_ts_irc_summary(
            source_workspace=str(workspace.directory),
            refined_workspace=str(tmp_path / "refined_workspace"),
            inputs_fp=str(refinement_inputs_fp),
        )

    monkeypatch.setattr(
        main_cli,
        "refine_drive_workspace_network",
        _fake_refine_drive_workspace_network,
    )

    main_cli.refine(
        source=str(source_workspace.directory),
        inputs=str(inputs_fp),
        mode="ts-irc",
        output_directory=str(tmp_path / "refined_workspace"),
    )

    assert isinstance(call_args["workspace"], main_cli.RetropathsWorkspace)
    assert call_args["workspace"].directory == source_workspace.directory
    assert call_args["refinement_inputs_fp"] == inputs_fp.resolve()
    assert call_args["refined_workspace_dir"] == (tmp_path / "refined_workspace").resolve()


def test_refine_ts_irc_builds_synthetic_workspace_from_chain_source(monkeypatch, tmp_path):
    inputs_fp = tmp_path / "refine.toml"
    inputs_fp.write_text("engine_name='chemcloud'\n", encoding="utf-8")

    chain = _chain_from_xs([1.0, 2.0, 3.0], ChainInputs(), energies=[0.0, 1.0, 0.0])
    observed = {}

    def _fake_refine_drive_workspace_network(
        workspace,
        *,
        refinement_inputs_fp,
        refined_workspace_dir=None,
        refined_run_name=None,
        use_bigchem=None,
        write_status_html_output=False,
        progress=None,
    ):
        observed["workspace"] = workspace
        observed["pot"] = main_cli.Pot.read_from_disk(workspace.neb_pot_fp)
        return _fake_ts_irc_summary(
            source_workspace=str(workspace.directory),
            refined_workspace=str(tmp_path / "refined_workspace"),
            inputs_fp=str(refinement_inputs_fp),
        )

    monkeypatch.setattr(
        main_cli,
        "refine_drive_workspace_network",
        _fake_refine_drive_workspace_network,
    )

    monkeypatch.chdir(tmp_path)
    main_cli.refine(
        source=chain,
        inputs=str(inputs_fp),
        mode="ts-irc",
    )

    workspace = observed["workspace"]
    pot = observed["pot"]
    assert isinstance(workspace, main_cli.RetropathsWorkspace)
    assert workspace.neb_pot_fp.exists()
    assert workspace.annotated_neb_pot_fp.exists()
    assert pot.graph.number_of_edges() == 1


def test_refine_ts_irc_preserves_network_splits_graph(monkeypatch, tmp_path):
    inputs_fp = tmp_path / "refine.toml"
    inputs_fp.write_text("engine_name='chemcloud'\n", encoding="utf-8")

    strict_inputs = ChainInputs(node_rms_thre=0.01, node_ene_thre=0.001)
    chain_a = _chain_from_xs([1.0, 2.0], strict_inputs, energies=[0.0, 1.0])
    chain_b = _chain_from_xs([2.0, 3.0], strict_inputs, energies=[0.0, 1.0])
    source_pot = main_cli._chains_to_refinement_pot(
        [chain_a, chain_b],
        source_label="network_source",
    )

    network_dir = tmp_path / "demo_network_splits"
    network_dir.mkdir()
    network_fp = network_dir / "demo_network.json"
    source_pot.write_to_disk(network_fp)

    observed = {}

    def _fake_refine_drive_workspace_network(
        workspace,
        *,
        refinement_inputs_fp,
        refined_workspace_dir=None,
        refined_run_name=None,
        use_bigchem=None,
        write_status_html_output=False,
        progress=None,
    ):
        observed["pot"] = main_cli.Pot.read_from_disk(workspace.neb_pot_fp)
        return _fake_ts_irc_summary(
            source_workspace=str(workspace.directory),
            refined_workspace=str(tmp_path / "refined_workspace"),
            inputs_fp=str(refinement_inputs_fp),
        )

    monkeypatch.setattr(
        main_cli,
        "refine_drive_workspace_network",
        _fake_refine_drive_workspace_network,
    )

    monkeypatch.chdir(tmp_path)
    main_cli.refine(
        source=str(network_dir),
        inputs=str(inputs_fp),
        mode="ts-irc",
    )

    assert observed["pot"].graph.number_of_edges() == 2


def test_load_precomputed_refine_source_reads_treenode_directory(monkeypatch, tmp_path):
    source_dir = tmp_path / "my_run_msmep"
    source_dir.mkdir()
    (source_dir / "adj_matrix.txt").write_text("1\n", encoding="utf-8")
    sentinel = object()

    monkeypatch.setattr(
        main_cli.TreeNode, "read_from_disk", staticmethod(lambda **kwargs: sentinel)
    )

    out = main_cli._load_precomputed_refine_source(
        source=source_dir,
        charge=0,
        multiplicity=1,
    )
    assert out is sentinel


def test_load_precomputed_refine_source_reads_neb_history(monkeypatch, tmp_path):
    source_fp = tmp_path / "mep_output_neb.xyz"
    source_fp.write_text("2\ncomment\nH 0 0 0\nH 0 0 1\n", encoding="utf-8")
    history_dir = tmp_path / "mep_output_neb_history"
    history_dir.mkdir()
    sentinel = object()

    monkeypatch.setattr(
        main_cli.NEB, "read_from_disk", staticmethod(lambda **kwargs: sentinel)
    )

    out = main_cli._load_precomputed_refine_source(
        source=source_fp,
        charge=0,
        multiplicity=1,
    )
    assert out is sentinel


def test_load_precomputed_refine_source_reads_network_json(monkeypatch, tmp_path):
    source_fp = tmp_path / "run_network.json"
    source_fp.write_text("{}", encoding="utf-8")
    sentinel_chain = _chain_from_xs([1.0, 2.0], ChainInputs(), energies=[0.0, 0.0])

    monkeypatch.setattr(
        main_cli,
        "_load_best_path_chain_from_network_json",
        lambda _fp: sentinel_chain,
    )

    out = main_cli._load_precomputed_refine_source(
        source=source_fp,
        charge=0,
        multiplicity=1,
    )
    assert out is sentinel_chain


def test_load_precomputed_refine_source_reads_network_splits_directory(monkeypatch, tmp_path):
    source_dir = tmp_path / "run_network_splits"
    source_dir.mkdir()
    network_fp = source_dir / "run_network.json"
    network_fp.write_text("{}", encoding="utf-8")
    sentinel_chain = _chain_from_xs([1.0, 2.0], ChainInputs(), energies=[0.0, 0.0])

    monkeypatch.setattr(
        main_cli,
        "_load_best_path_chain_from_network_json",
        lambda fp: sentinel_chain if fp == network_fp else None,
    )

    out = main_cli._load_precomputed_refine_source(
        source=source_dir,
        charge=0,
        multiplicity=1,
    )
    assert out is sentinel_chain


def test_load_endpoint_structure_converts_rst7_with_prmtop(tmp_path):
    rst7_fp = tmp_path / "start.rst7"
    rst7_fp.write_text(
        "TITLE\n"
        "2\n"
        " 0.0000000  0.0000000  0.0000000  1.0000000  0.0000000  0.0000000\n"
    )
    prmtop_text = (
        "%FLAG ATOMIC_NUMBER\n"
        "%FORMAT(10I8)\n"
        "       1       8\n"
        "%FLAG END\n"
    )

    struct = main_cli._load_endpoint_structure(
        path=str(rst7_fp),
        charge=0,
        multiplicity=1,
        rst7_prmtop_text=prmtop_text,
    )

    assert struct.symbols == ["H", "O"]
    assert np.isclose(struct.geometry[1][0], 1.0 * ANGSTROM_TO_BOHR)


def test_run_requires_prmtop_for_rst7_endpoints():
    with pytest.raises(main_cli.typer.Exit) as excinfo:
        main_cli.run(
            start="react.rst7",
            end="prod.xyz",
            inputs=None,
        )

    assert excinfo.value.exit_code == 1
