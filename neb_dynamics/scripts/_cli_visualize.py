from __future__ import annotations

import base64
import heapq
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VisualizationData:
    chain: Any
    chain_trajectory: list[Any] | None = None
    tree_layers: list[dict] | None = None
    network_pot: Any | None = None
    network_endpoint_hints: dict | None = None


@dataclass
class VisualizationDeps:
    Chain: Any
    ChainInputs: Any
    NEB: Any
    Path: Any
    Pot: Any
    Structure: Any
    StructureNode: Any
    TreeNode: Any
    is_connectivity_identical: Any
    nx: Any
    np: Any
    read_multiple_structure_from_file: Any
    reverse_chain: Any
    concat_chains: Any
    collect_tree_layers_for_visualization: Any | None = None
    match_network_endpoint_indices_by_connectivity: Any | None = None
    find_pot_root_node_index: Any | None = None
    find_pot_target_node_index: Any | None = None
    best_path_by_apparent_barrier: Any | None = None
    path_chain_from_pot: Any | None = None
    best_chain_for_directed_edge: Any | None = None
    load_network_endpoint_hints: Any | None = None
    load_network_endpoint_structures: Any | None = None


def _truncate_label(label: str, max_len: int) -> str:
    if len(label) <= max_len:
        return label
    if max_len <= 3:
        return label[:max_len]
    return label[: max_len - 3] + "..."


def _build_ascii_energy_profile(
    energies,
    labels,
    width: int = 60,
    height: int = 12,
) -> str:
    if len(energies) == 0:
        return "No energies to plot."
    if len(energies) != len(labels):
        raise ValueError("labels must be same length as energies")

    min_e = float(min(energies))
    max_e = float(max(energies))
    if max_e == min_e:
        max_e = min_e + 1e-6

    def _xpos(i):
        if len(energies) == 1:
            return 0
        return int(round(i * (width - 1) / (len(energies) - 1)))

    def _ypos(e):
        return int(round((e - min_e) * (height - 1) / (max_e - min_e)))

    grid = [[" " for _ in range(width)] for _ in range(height)]

    xs = [_xpos(i) for i in range(len(energies))]
    ys = [_ypos(e) for e in energies]

    for i in range(len(energies) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        if x0 == x1:
            continue
        step = 1 if x1 > x0 else -1
        for x in range(x0, x1 + step, step):
            t = (x - x0) / (x1 - x0)
            y = int(round(y0 + (y1 - y0) * t))
            row = height - 1 - y
            if 0 <= row < height and 0 <= x < width and grid[row][x] == " ":
                grid[row][x] = "-"

    for x, y in zip(xs, ys):
        row = height - 1 - y
        if 0 <= row < height and 0 <= x < width:
            grid[row][x] = "*"

    prefix_len = len(f"{max_e:7.2f} |")
    lines = []
    for r in range(height):
        y_val = max_e - (max_e - min_e) * (r / (height - 1))
        prefix = f"{y_val:7.2f} |"
        lines.append(prefix + "".join(grid[r]))

    lines.append(" " * prefix_len + "-" * width)

    label_line = [" " for _ in range(width)]
    max_label_len = 12
    truncated = False
    for x, label in zip(xs, labels):
        tlabel = _truncate_label(label, max_label_len)
        if tlabel != label:
            truncated = True
        start = max(0, min(width - len(tlabel), x - len(tlabel) // 2))
        for i, ch in enumerate(tlabel):
            label_line[start + i] = ch
    lines.append(" " * prefix_len + "".join(label_line))

    if truncated:
        lines.append("")
        lines.append("Full SMILES labels:")
        for i, label in enumerate(labels):
            lines.append(f"{i}: {label}")

    return "\n".join(lines)


def _ascii_profile_for_chain(chain, console) -> None:
    try:
        energies = chain.energies_kcalmol
    except Exception as exc:
        console.print(f"[yellow]⚠ Could not compute energy profile: {exc}[/yellow]")
        return

    labels = [str(i) for i, _ in enumerate(chain.nodes)]
    plot = _build_ascii_energy_profile(energies, labels)
    console.print("\nASCII Reaction Profile (Energy vs Node)")
    console.print(plot, markup=False)


def _neb_chains_for_visualization(neb_obj) -> list:
    if getattr(neb_obj, "chain_trajectory", None):
        return list(neb_obj.chain_trajectory)
    if getattr(neb_obj, "optimized", None) is not None:
        return [neb_obj.optimized]
    return []


def _collect_tree_layers_for_visualization(tree) -> list[dict]:
    layers: list[dict] = []
    by_depth: dict[int, list[dict]] = {}
    stack: list[tuple[Any, int, int | None]] = [(tree, 0, None)]
    while stack:
        tree_node, depth, parent_index = stack.pop()
        if getattr(tree_node, "data", None):
            chains = _neb_chains_for_visualization(tree_node.data)
            if chains:
                by_depth.setdefault(depth, []).append(
                    {
                        "label": f"Node {tree_node.index}",
                        "node_index": int(tree_node.index),
                        "parent_index": parent_index,
                        "chains": chains,
                    }
                )
        for child in reversed(tree_node.children):
            stack.append((child, depth + 1, int(tree_node.index)))

    for depth in sorted(by_depth):
        layers.append({"depth": depth, "groups": by_depth[depth]})
    return layers


def _chain_plot_payload(chain_obj) -> dict[str, list[float]]:
    try:
        x_vals = [float(v) for v in chain_obj.integrated_path_length]
        y_vals = [float(v) for v in chain_obj.energies_kcalmol]
    except Exception:
        x_vals = []
        y_vals = []
    return {"x": x_vals, "y": y_vals}


def _serialize_chains_for_visualization(chains: list) -> dict:
    chain_payload = []
    for chain_ind, chain_obj in enumerate(chains):
        frames = []
        plot_payload = _chain_plot_payload(chain_obj)
        for node in chain_obj.nodes:
            frames.append(
                {
                    "xyz_b64": base64.b64encode(
                        node.structure.to_xyz().encode("utf-8")
                    ).decode("ascii"),
                }
            )
        chain_payload.append(
            {
                "index": chain_ind,
                "frames": frames,
                "plot": plot_payload,
            }
        )
    default_chain_index = max(len(chains) - 1, 0)
    return {
        "chains": chain_payload,
        "default_chain_index": default_chain_index,
    }


def _node_distance(node_a, node_b) -> float:
    return float(((node_a.coords - node_b.coords) ** 2).sum() ** 0.5)


def _orient_chain_to_edge(chain, *, source_td, target_td, deps: VisualizationDeps):
    start_node = chain.nodes[0]
    end_node = chain.nodes[-1]

    forward_score = _node_distance(start_node, source_td) + _node_distance(
        end_node, target_td
    )
    reverse_score = _node_distance(start_node, target_td) + _node_distance(
        end_node, source_td
    )

    if reverse_score < forward_score:
        return deps.reverse_chain(chain)
    return chain.copy()


def _chain_peak_energy(chain, deps: VisualizationDeps) -> float:
    energies = deps.np.array(chain.energies, dtype=float)
    return float(deps.np.max(energies))


def _best_chain_between_nodes(
    pot,
    source: int,
    target: int,
    deps: VisualizationDeps,
):
    if pot.graph.has_edge(source, target):
        return _best_chain_for_directed_edge(pot, source, target, deps).copy()
    if pot.graph.has_edge(target, source):
        return deps.reverse_chain(
            _best_chain_for_directed_edge(pot, target, source, deps)
        )
    raise ValueError(f"No NEB chains found for pair {source}<->{target}.")


def _pair_neighbors(pot, node_idx: int) -> set[int]:
    neighbors = {int(v) for v in pot.graph.successors(int(node_idx))}
    neighbors.update(int(v) for v in pot.graph.predecessors(int(node_idx)))
    return neighbors


def _pair_barrier(pot, source: int, target: int) -> float:
    barriers = []
    if pot.graph.has_edge(source, target):
        barriers.append(float(pot.graph.edges[(source, target)].get("barrier", 0.0)))
    if pot.graph.has_edge(target, source):
        barriers.append(float(pot.graph.edges[(target, source)].get("barrier", 0.0)))
    if not barriers:
        return float("inf")
    return min(barriers)


def _pair_graph_shortest_path_length(
    pot,
    source: int,
    target: int,
    deps: VisualizationDeps,
) -> float:
    if int(source) == int(target):
        return 0.0

    frontier: list[tuple[float, int]] = [(0.0, int(source))]
    best_cost: dict[int, float] = {int(source): 0.0}
    while frontier:
        curr_cost, node_idx = heapq.heappop(frontier)
        if curr_cost > best_cost.get(node_idx, deps.np.inf):
            continue
        if node_idx == int(target):
            return curr_cost
        for nbr in _pair_neighbors(pot, node_idx):
            edge_cost = _pair_barrier(pot, node_idx, nbr)
            if not deps.np.isfinite(edge_cost):
                continue
            next_cost = curr_cost + edge_cost
            if next_cost < best_cost.get(nbr, deps.np.inf):
                best_cost[nbr] = next_cost
                heapq.heappush(frontier, (next_cost, int(nbr)))

    return deps.np.inf


def _best_chain_for_directed_edge(
    pot,
    source: int,
    target: int,
    deps: VisualizationDeps,
):
    source_td = pot.graph.nodes[source].get("td")
    target_td = pot.graph.nodes[target].get("td")
    forward = list(pot.graph.edges[(source, target)]["list_of_nebs"])
    if not forward or source_td is None or target_td is None:
        raise ValueError(f"No NEB chains found for edge {source}->{target}.")

    oriented = [
        _orient_chain_to_edge(
            chain,
            source_td=source_td,
            target_td=target_td,
            deps=deps,
        )
        for chain in forward
    ]
    peak_energies = [_chain_peak_energy(chain, deps) for chain in oriented]
    return oriented[int(deps.np.argmin(peak_energies))]


def _best_path_by_apparent_barrier(
    pot,
    root_idx: int,
    target_idx: int,
    deps: VisualizationDeps,
) -> tuple[list[int], float] | tuple[None, None]:
    if root_idx == target_idx:
        root_td = pot.graph.nodes[root_idx].get("td")
        if root_td is None:
            return None, None
        return [int(root_idx)], float(root_td.energy)

    frontier: list[tuple[float, int]] = []
    best_peak: dict[int, float] = {}
    predecessor: dict[int, int | None] = {int(root_idx): None}

    root_td = pot.graph.nodes[root_idx].get("td")
    if root_td is None:
        return None, None
    root_energy = float(root_td.energy)
    best_peak[int(root_idx)] = root_energy
    heapq.heappush(frontier, (root_energy, int(root_idx)))

    while frontier:
        curr_peak, node_idx = heapq.heappop(frontier)
        if curr_peak > best_peak.get(node_idx, deps.np.inf):
            continue
        if node_idx == int(target_idx):
            break

        for nbr in _pair_neighbors(pot, node_idx):
            try:
                best_chain = _best_chain_between_nodes(
                    pot, int(node_idx), int(nbr), deps
                )
                edge_peak = _chain_peak_energy(best_chain, deps)
            except Exception:
                continue
            next_peak = max(curr_peak, edge_peak)
            if next_peak < best_peak.get(nbr, deps.np.inf):
                best_peak[nbr] = next_peak
                predecessor[nbr] = int(node_idx)
                heapq.heappush(frontier, (next_peak, nbr))

    if int(target_idx) not in predecessor:
        return None, None

    path = [int(target_idx)]
    while predecessor[path[-1]] is not None:
        path.append(int(predecessor[path[-1]]))
    path.reverse()
    return path, best_peak[int(target_idx)]


def _find_pot_root_node_index(pot) -> int | None:
    for node_idx, data in pot.graph.nodes(data=True):
        if data.get("root"):
            return int(node_idx)
    return int(sorted(pot.graph.nodes)[0]) if pot.graph.nodes else None


def _load_network_endpoint_hints(network_json_fp: Path) -> dict | None:
    network_json_fp = Path(network_json_fp)
    candidates = []
    if network_json_fp.name.endswith("_network.json"):
        candidates.append(
            network_json_fp.with_name(
                network_json_fp.name.replace(
                    "_network.json", "_request_manifest.json"
                )
            )
        )
    candidates.append(
        network_json_fp.parent / f"{network_json_fp.stem}_request_manifest.json"
    )

    for manifest_fp in candidates:
        if not manifest_fp.exists():
            continue
        try:
            manifest = json.loads(manifest_fp.read_text())
        except Exception:
            continue
        requests = manifest.get("requests", [])
        if not requests:
            continue
        first = min(requests, key=lambda row: row.get("request_id", 10**9))
        if first.get("start_index") is None or first.get("end_index") is None:
            continue
        return {
            "root_index": int(first["start_index"]),
            "target_index": int(first["end_index"]),
            "manifest_path": str(manifest_fp),
        }
    return None


def _load_network_endpoint_structures(
    network_json_fp: Path,
    deps: VisualizationDeps,
):
    network_json_fp = Path(network_json_fp)

    base_name = network_json_fp.stem.replace("_network", "")
    search_roots = []
    for root in [network_json_fp.parent, *network_json_fp.parents]:
        if root not in search_roots:
            search_roots.append(root)

    name_variants = []
    for candidate in [
        base_name,
        re.sub(r"\d+$", "", base_name),
        re.sub(r"[_-]+\d+$", "", base_name),
    ]:
        value = candidate.strip("_-")
        if value and value not in name_variants:
            name_variants.append(value)

    start_candidates = []
    end_candidates = []
    for variant in name_variants:
        start_candidates.extend([f"start_{variant}.xyz", f"start_{variant}.rst7"])
        end_candidates.extend([f"end_{variant}.xyz", f"end_{variant}.rst7"])
    start_candidates.extend(["start.xyz", "start.rst7"])
    end_candidates.extend(["end.xyz", "end.rst7"])

    def _load_first(root_candidates, names):
        for root in root_candidates:
            for name in names:
                fp = root / name
                if fp.exists():
                    try:
                        struct = deps.read_multiple_structure_from_file(fp)[0]
                        return deps.StructureNode(structure=struct)
                    except Exception:
                        continue
        return None

    start_node = _load_first(search_roots, start_candidates)
    end_node = _load_first(search_roots, end_candidates)
    return start_node, end_node


def _match_network_endpoint_indices_by_connectivity(
    pot,
    start_node,
    end_node,
    deps: VisualizationDeps,
) -> dict | None:
    if start_node is None and end_node is None:
        return None

    matches = {"root_index": None, "target_index": None}
    for node_idx, data in pot.graph.nodes(data=True):
        td = data.get("td")
        if td is None:
            continue
        if start_node is not None and matches["root_index"] is None:
            try:
                if deps.is_connectivity_identical(
                    start_node,
                    td,
                    verbose=False,
                    collect_comparison=False,
                ):
                    matches["root_index"] = int(node_idx)
            except Exception:
                pass
        if end_node is not None and matches["target_index"] is None:
            try:
                if deps.is_connectivity_identical(
                    end_node,
                    td,
                    verbose=False,
                    collect_comparison=False,
                ):
                    matches["target_index"] = int(node_idx)
            except Exception:
                pass

    if matches["root_index"] is None and matches["target_index"] is None:
        return None
    return matches


def _find_pot_target_node_index(
    pot,
    deps: VisualizationDeps,
    target_idx_hint: int | None = None,
) -> int | None:
    if not pot.graph.nodes:
        return None

    if target_idx_hint is not None and int(target_idx_hint) in pot.graph.nodes:
        return int(target_idx_hint)

    for node_idx, data in pot.graph.nodes(data=True):
        if data.get("requested_target") or data.get("target"):
            return int(node_idx)

    if pot.target is not None and not pot.target.is_empty():
        target_smiles = pot.target.force_smiles()
        for node_idx, data in pot.graph.nodes(data=True):
            molecule = data.get("molecule")
            if molecule is not None and molecule.force_smiles() == target_smiles:
                return int(node_idx)

    root_idx = _find_pot_root_node_index(pot)
    if root_idx is None:
        return None

    sink_nodes = [
        int(node_idx)
        for node_idx in pot.graph.nodes
        if int(node_idx) != root_idx and pot.graph.out_degree(node_idx) == 0
    ]
    if not sink_nodes:
        sink_nodes = [
            int(node_idx)
            for node_idx in pot.graph.nodes
            if int(node_idx) != root_idx
        ]
    if not sink_nodes:
        return root_idx

    best_target = None
    best_cost = None
    for node_idx in sink_nodes:
        try:
            cost = float(
                deps.nx.shortest_path_length(
                    pot.graph,
                    source=root_idx,
                    target=node_idx,
                    weight="barrier",
                )
            )
        except Exception:
            continue
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_target = node_idx
    return best_target


def _path_chain_from_pot(pot, path: list[int], deps: VisualizationDeps):
    if not path:
        return None
    edge_chains = []
    for source, target in zip(path[:-1], path[1:]):
        edge_chains.append(_best_chain_between_nodes(pot, source, target, deps))
    if not edge_chains:
        node_td = pot.graph.nodes[path[0]].get("td")
        if node_td is None:
            return None
        return deps.Chain.model_validate(
            {"nodes": [node_td.copy()], "parameters": deps.ChainInputs()}
        )
    return deps.concat_chains(edge_chains, edge_chains[0].parameters)


def _parse_visualize_atom_indices(
    qminds_fp: str | None = None,
    atom_indices: str | None = None,
) -> list[int] | None:
    if qminds_fp and atom_indices:
        raise ValueError("Provide either --qminds-fp or --atom-indices, not both.")
    if qminds_fp:
        values = []
        for raw in Path(qminds_fp).read_text().splitlines():
            token = raw.strip()
            if token:
                values.append(int(token))
        return sorted(set(values))
    if atom_indices:
        tokens = atom_indices.replace(",", " ").split()
        return sorted(set(int(v) for v in tokens))
    return None


def _subset_chain_for_visualization(chain, atom_indices: list[int], deps: VisualizationDeps):
    if len(chain.nodes) == 0:
        return chain
    n_atoms = len(chain.nodes[0].structure.symbols)
    bad = [i for i in atom_indices if i < 0 or i >= n_atoms]
    if bad:
        raise ValueError(f"Atom indices out of bounds for n_atoms={n_atoms}: {bad}")

    new_nodes = []
    for node in chain.nodes:
        struct = node.structure
        new_struct = deps.Structure(
            geometry=deps.np.array(struct.geometry)[atom_indices],
            symbols=[struct.symbols[i] for i in atom_indices],
            charge=struct.charge,
            multiplicity=struct.multiplicity,
        )
        new_node = deps.StructureNode(structure=new_struct)
        new_node.has_molecular_graph = False
        new_node.graph = None
        new_node._cached_energy = node._cached_energy
        if node._cached_gradient is not None:
            try:
                grad_arr = deps.np.array(node._cached_gradient)
                if grad_arr.ndim == 2 and grad_arr.shape[0] >= max(atom_indices) + 1:
                    new_node._cached_gradient = grad_arr[atom_indices]
                else:
                    new_node._cached_gradient = node._cached_gradient
            except Exception:
                new_node._cached_gradient = node._cached_gradient
        new_nodes.append(new_node)
    return deps.Chain.model_validate({"nodes": new_nodes, "parameters": chain.parameters})


def _subset_chain_trajectory_for_visualization(
    chain_trajectory: list,
    atom_indices: list[int],
    deps: VisualizationDeps,
) -> list:
    return [
        _subset_chain_for_visualization(chain=chain, atom_indices=atom_indices, deps=deps)
        for chain in chain_trajectory
    ]


def _subset_tree_layers_for_visualization(
    tree_layers: list[dict],
    atom_indices: list[int],
    deps: VisualizationDeps,
) -> list[dict]:
    subset_layers = []
    for layer in tree_layers:
        subset_groups = []
        for group in layer["groups"]:
            subset_groups.append(
                {
                    "label": group["label"],
                    "node_index": group["node_index"],
                    "parent_index": group["parent_index"],
                    "chains": _subset_chain_trajectory_for_visualization(
                        group["chains"], atom_indices, deps
                    ),
                }
            )
        subset_layers.append({"depth": layer["depth"], "groups": subset_groups})
    return subset_layers


def _load_visualization_data(
    result_path: Path,
    deps: VisualizationDeps,
    charge: int = 0,
    multiplicity: int = 1,
) -> VisualizationData:
    result_path = Path(result_path)
    if not result_path.exists():
        raise FileNotFoundError(f"Path does not exist: {result_path}")

    collect_tree_layers = (
        deps.collect_tree_layers_for_visualization
        or _collect_tree_layers_for_visualization
    )
    load_network_endpoint_hints = (
        deps.load_network_endpoint_hints or _load_network_endpoint_hints
    )
    load_network_endpoint_structures = (
        deps.load_network_endpoint_structures
        or (lambda network_json_fp: _load_network_endpoint_structures(network_json_fp, deps))
    )
    match_endpoint_indices = (
        deps.match_network_endpoint_indices_by_connectivity
        or (lambda pot, start_node, end_node: _match_network_endpoint_indices_by_connectivity(pot, start_node, end_node, deps))
    )
    find_root = deps.find_pot_root_node_index or _find_pot_root_node_index
    find_target = (
        deps.find_pot_target_node_index
        or (lambda pot, target_idx_hint=None: _find_pot_target_node_index(pot, deps, target_idx_hint))
    )
    best_path = (
        deps.best_path_by_apparent_barrier
        or (lambda pot, root_idx, target_idx: _best_path_by_apparent_barrier(pot, root_idx, target_idx, deps))
    )
    path_chain_from_pot = (
        deps.path_chain_from_pot
        or (lambda pot, path: _path_chain_from_pot(pot, path, deps))
    )
    best_chain_for_directed_edge = (
        deps.best_chain_for_directed_edge
        or (lambda pot, source, target: _best_chain_for_directed_edge(pot, source, target, deps))
    )

    if result_path.is_dir():
        adj_matrix_fp = result_path / "adj_matrix.txt"
        if adj_matrix_fp.exists():
            tree = deps.TreeNode.read_from_disk(
                folder_name=result_path,
                charge=charge,
                multiplicity=multiplicity,
            )
            return VisualizationData(
                chain=tree.output_chain,
                chain_trajectory=None,
                tree_layers=collect_tree_layers(tree),
            )
        raise ValueError(
            "Directory input must be a TreeNode folder containing adj_matrix.txt."
        )

    if result_path.suffix.lower() == ".json":
        try:
            pot = deps.Pot.read_from_disk(result_path)
            endpoint_hints = load_network_endpoint_hints(result_path) or {}
            start_node, end_node = load_network_endpoint_structures(result_path)
            connectivity_hints = match_endpoint_indices(
                pot,
                start_node=start_node,
                end_node=end_node,
            )
            if connectivity_hints:
                endpoint_hints.update(
                    {k: v for k, v in connectivity_hints.items() if v is not None}
                )
            if not endpoint_hints:
                endpoint_hints = None
            root_idx = (
                int(endpoint_hints["root_index"])
                if endpoint_hints and endpoint_hints.get("root_index") is not None
                else find_root(pot)
            )
            target_idx = find_target(
                pot,
                target_idx_hint=(
                    int(endpoint_hints["target_index"])
                    if endpoint_hints and endpoint_hints.get("target_index") is not None
                    else None
                ),
            )
            path = []
            if (
                root_idx is not None
                and target_idx is not None
                and deps.nx.has_path(pot.graph, root_idx, target_idx)
            ):
                best_path_nodes, _ = best_path(
                    pot,
                    root_idx=root_idx,
                    target_idx=target_idx,
                )
                path = [int(v) for v in best_path_nodes] if best_path_nodes else []
            default_chain = path_chain_from_pot(pot, path)
            if default_chain is None:
                first_edge = next(iter(pot.graph.edges), None)
                if first_edge is not None:
                    default_chain = best_chain_for_directed_edge(
                        pot, int(first_edge[0]), int(first_edge[1])
                    ).copy()
                else:
                    root_td = (
                        pot.graph.nodes[root_idx].get("td")
                        if root_idx is not None
                        else None
                    )
                    if root_td is None:
                        raise ValueError(
                            "Pot network has no visualizable chains or node geometries."
                        )
                    default_chain = deps.Chain.model_validate(
                        {"nodes": [root_td.copy()], "parameters": deps.ChainInputs()}
                    )
            return VisualizationData(
                chain=default_chain,
                chain_trajectory=None,
                tree_layers=None,
                network_pot=pot,
                network_endpoint_hints=endpoint_hints,
            )
        except Exception as exc:
            raise ValueError(
                "Could not load network JSON for visualization."
            ) from exc

    history_folder = result_path.parent / f"{result_path.stem}_history"
    if history_folder.exists():
        neb = deps.NEB.read_from_disk(
            fp=result_path,
            history_folder=history_folder,
            charge=charge,
            multiplicity=multiplicity,
        )
        if neb.chain_trajectory:
            return VisualizationData(
                chain=neb.chain_trajectory[-1],
                chain_trajectory=list(neb.chain_trajectory),
            )
        if getattr(neb, "optimized", None) is not None:
            return VisualizationData(chain=neb.optimized, chain_trajectory=None)
        raise ValueError("Loaded NEB object has no optimized chain or history.")

    try:
        chain = deps.Chain.from_xyz(
            result_path,
            parameters=deps.ChainInputs(),
            charge=charge,
            spinmult=multiplicity,
        )
        return VisualizationData(chain=chain, chain_trajectory=None)
    except Exception as exc:
        raise ValueError(
            "Could not detect serialized result type. For network visualization provide a Pot .json; for NEB provide the .xyz output "
            "that has a sibling '<stem>_history/' folder; for recursive MSMEP provide "
            "the TreeNode directory with adj_matrix.txt; or provide a valid chain xyz."
        ) from exc


def _load_chain_for_visualization(
    result_path: Path,
    deps: VisualizationDeps,
    charge: int = 0,
    multiplicity: int = 1,
):
    return _load_visualization_data(
        result_path=result_path,
        deps=deps,
        charge=charge,
        multiplicity=multiplicity,
    ).chain


def _build_network_visualization_payload(
    pot,
    deps: VisualizationDeps,
    atom_indices: list[int] | None = None,
    endpoint_hints: dict | None = None,
) -> dict:
    graph_for_layout = pot.graph.to_undirected()
    layout = deps.nx.spring_layout(graph_for_layout, seed=7)
    find_root = deps.find_pot_root_node_index or _find_pot_root_node_index
    find_target = (
        deps.find_pot_target_node_index
        or (lambda pot, target_idx_hint=None: _find_pot_target_node_index(pot, deps, target_idx_hint))
    )
    best_path = (
        deps.best_path_by_apparent_barrier
        or (lambda pot, root_idx, target_idx: _best_path_by_apparent_barrier(pot, root_idx, target_idx, deps))
    )
    best_chain_for_directed_edge = (
        deps.best_chain_for_directed_edge
        or (lambda pot, source, target: _best_chain_for_directed_edge(pot, source, target, deps))
    )

    root_idx = (
        int(endpoint_hints["root_index"])
        if endpoint_hints and endpoint_hints.get("root_index") is not None
        else find_root(pot)
    )
    target_idx = find_target(
        pot,
        target_idx_hint=(
            int(endpoint_hints["target_index"])
            if endpoint_hints and endpoint_hints.get("target_index") is not None
            else None
        ),
    )

    highlighted_path = []
    highlighted_edges: set[tuple[int, int]] = set()
    apparent_barrier = None
    if root_idx is not None and target_idx is not None and deps.nx.has_path(
        pot.graph.to_undirected(), root_idx, target_idx
    ):
        best_path_nodes, best_path_peak = best_path(
            pot, root_idx=root_idx, target_idx=target_idx
        )
        highlighted_path = [int(v) for v in best_path_nodes] if best_path_nodes else []
        highlighted_edges = {
            (int(a), int(b))
            for a, b in zip(highlighted_path[:-1], highlighted_path[1:])
        }
        try:
            root_energy = float(pot.graph.nodes[root_idx]["td"].energy)
            if best_path_peak is not None:
                apparent_barrier = float((best_path_peak - root_energy) * 627.5)
        except Exception:
            apparent_barrier = None

    def _cost_to_target(node_idx: int) -> float:
        if target_idx is None:
            return deps.np.inf
        return _pair_graph_shortest_path_length(
            pot,
            int(node_idx),
            int(target_idx),
            deps,
        )

    def _preferred_pair_orientation(a: int, b: int) -> tuple[int, int]:
        if (a, b) in highlighted_edges:
            return a, b
        if (b, a) in highlighted_edges:
            return b, a
        a_cost = _cost_to_target(a)
        b_cost = _cost_to_target(b)
        if a_cost == b_cost:
            return (a, b) if (a, b) in pot.graph.edges else (b, a)
        return (a, b) if b_cost < a_cost else (b, a)

    nodes_payload = []
    for node_idx, coords in layout.items():
        nodes_payload.append(
            {
                "id": int(node_idx),
                "label": f"Node {int(node_idx)}",
                "x": float(coords[0]),
                "y": float(coords[1]),
                "is_root": int(node_idx) == root_idx,
                "is_target": int(node_idx) == target_idx,
            }
        )

    edges_payload = []
    seen_pairs: set[frozenset[int]] = set()
    for raw_source, raw_target in pot.graph.edges:
        pair_key = frozenset((int(raw_source), int(raw_target)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        source, target = _preferred_pair_orientation(int(raw_source), int(raw_target))
        try:
            best_chain = _best_chain_between_nodes(pot, source, target, deps)
        except Exception:
            continue
        try:
            forward_barrier = (
                float(pot.graph.edges[(source, target)].get("barrier", 0.0))
                if pot.graph.has_edge(source, target)
                else float(pot.graph.edges[(target, source)].get("barrier", 0.0))
            )
        except Exception:
            continue
        if atom_indices is not None:
            best_chain = _subset_chain_for_visualization(
                best_chain, atom_indices, deps
            )
        reverse_barrier = None
        if pot.graph.has_edge(target, source):
            reverse_barrier = float(
                pot.graph.edges[(target, source)].get("barrier", 0.0)
            )
        pair_sum = (
            forward_barrier + reverse_barrier
            if reverse_barrier is not None
            else forward_barrier
        )
        edges_payload.append(
            {
                "id": f"{source}->{target}",
                "source": source,
                "target": target,
                "label": f"{source}->{target}",
                "barrier": forward_barrier,
                "reverse_barrier": reverse_barrier,
                "pair_barrier_sum": pair_sum,
                "highlight": (source, target) in highlighted_edges,
                "viz": _serialize_chains_for_visualization([best_chain]),
            }
        )

    return {
        "nodes": nodes_payload,
        "edges": edges_payload,
        "root_index": root_idx,
        "target_index": target_idx,
        "highlighted_path": highlighted_path,
        "best_apparent_barrier": apparent_barrier,
    }


def _build_chain_visualizer_html(
    chain,
    chain_trajectory: list | None = None,
    tree_layers: list[dict] | None = None,
    network_payload: dict | None = None,
) -> str:
    if tree_layers:
        layers_payload = []
        for layer in tree_layers:
            groups_payload = []
            for group in layer["groups"]:
                groups_payload.append(
                    {
                        "label": group["label"],
                        "node_index": int(group["node_index"]),
                        "parent_index": (
                            int(group["parent_index"])
                            if group["parent_index"] is not None
                            else None
                        ),
                        "viz": _serialize_chains_for_visualization(group["chains"]),
                    }
                )
            layers_payload.append({"depth": layer["depth"], "groups": groups_payload})
        default_layer_index = max(len(layers_payload) - 1, 0)
        default_group_index = 0
        has_tree_layers = True
    else:
        chains_to_visualize = list(chain_trajectory) if chain_trajectory else [chain]
        layers_payload = [
            {
                "depth": 0,
                "groups": [
                    {
                        "label": "NEB",
                        "node_index": 0,
                        "parent_index": None,
                        "viz": _serialize_chains_for_visualization(chains_to_visualize),
                    }
                ],
            }
        ]
        default_layer_index = 0
        default_group_index = 0
        has_tree_layers = False

    payload = json.dumps(layers_payload)
    network_json = json.dumps(network_payload or {})
    tree_panel_style = "" if has_tree_layers else "display:none;"
    network_panel_style = "" if network_payload else "display:none;"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NEB Dynamics Visualizer</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 16px; }}
    .row {{ display: flex; gap: 16px; align-items: flex-start; }}
    .panel {{ flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    .controls {{ margin-bottom: 12px; }}
    img {{ width: 100%; max-width: 700px; }}
  </style>
</head>
<body>
  <h2>NEB Dynamics Interactive Viewer</h2>
  <div class="controls">
    <div id="treePanel" style="{tree_panel_style} margin-bottom: 12px;">
      <div style="font-weight: 600; margin-bottom: 6px;">Optimization Tree</div>
      <svg id="treeSvg" width="100%" height="220" viewBox="0 0 900 220" style="border: 1px solid #e6e6e6; border-radius: 8px; background: #fafafa;"></svg>
      <div style="font-size: 12px; color: #666; margin-top: 6px;">Click a node to load that NEB object's data.</div>
    </div>
    <div id="networkPanel" style="{network_panel_style} margin-bottom: 12px;">
      <div style="font-weight: 600; margin-bottom: 6px;">Reaction Network</div>
      <svg id="networkSvg" width="100%" height="320" viewBox="0 0 900 320" style="border: 1px solid #e6e6e6; border-radius: 8px; background: #fafafa;"></svg>
      <div id="networkInfo" style="font-size: 12px; color: #666; margin-top: 6px;">Click an edge to load the corresponding best NEB pair. The best overall path is highlighted in gold.</div>
    </div>
    <label for="chainSelect">Chain: </label>
    <select id="chainSelect"></select>
    <br/>
    <label for="frameSlider">Frame: <span id="frameLabel">0</span></label><br/>
    <input id="frameSlider" type="range" min="0" max="0" value="0" step="1" style="width: min(720px, 90vw);" />
  </div>
  <div class="row">
    <div class="panel">
      <h3>Structure</h3>
      <iframe id="structureFrame" style="width: 100%; height: 520px; border: 0;" title="Structure viewer"></iframe>
    </div>
    <div class="panel">
      <h3>Energy Profile (Selected Chain)</h3>
      <div id="energyPlot" style="width: 100%; max-width: 700px;"></div>
      <div id="historyPanel" style="margin-top: 18px; display:none;">
        <h3>Optimization History</h3>
        <div id="historyPlot" style="width: 100%; max-width: 700px;"></div>
      </div>
    </div>
  </div>
  <script>
    const layers = {payload};
    function decodeB64UTF8(b64) {{
      return decodeURIComponent(escape(window.atob(b64)));
    }}
    function clamp(val, low, high) {{
      return Math.max(low, Math.min(high, val));
    }}
    function makeStructureSrcdoc(xyzB64) {{
      return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>
    html, body, #viewer {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: white;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    #status {{
      padding: 12px;
      color: #555;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div id="status">Loading 3D viewer...</div>
  <script>
    const xyz = decodeURIComponent(escape(atob("__XYZ_B64__")));
    function boot() {{
      const status = document.getElementById("status");
      const host = document.getElementById("viewer");
      status.remove();
      const viewer = $3Dmol.createViewer(host, {{ backgroundColor: "white" }});
      viewer.addModel(xyz, "xyz");
      viewer.setStyle({{}}, {{ stick: {{}}, sphere: {{ scale: 0.3 }} }});
      viewer.zoomTo();
      viewer.render();
    }}
    if (window.$3Dmol) {{
      boot();
    }} else {{
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/3dmol@2.5.3/build/3Dmol-min.js";
      script.onload = boot;
      script.onerror = () => {{
        const status = document.getElementById("status");
        status.textContent = "Failed to load 3Dmol.js";
      }};
      document.head.appendChild(script);
    }}
  <\\/script>
</body>
</html>`.replace("__XYZ_B64__", xyzB64);
    }}
    const networkPayload = {network_json};
    function renderPlot(containerId, traces, selectedTraceIndex = null, selectedPointIndex = null, title = "") {{
      const container = document.getElementById(containerId);
      if (!container) return;
      if (!traces || !traces.length) {{
        container.innerHTML = "<div style=\\"color:#666;font-size:13px;\\">No plot data available.</div>";
        return;
      }}

      const width = 700;
      const height = 420;
      const margin = {{ top: 28, right: 20, bottom: 44, left: 58 }};
      const xs = traces.flatMap((trace) => trace.x || []);
      const ys = traces.flatMap((trace) => trace.y || []);
      if (!xs.length || !ys.length) {{
        container.innerHTML = "<div style=\\"color:#666;font-size:13px;\\">No plot data available.</div>";
        return;
      }}

      let minX = Math.min(...xs);
      let maxX = Math.max(...xs);
      let minY = Math.min(...ys);
      let maxY = Math.max(...ys);
      if (minX === maxX) maxX = minX + 1;
      if (minY === maxY) maxY = minY + 1;
      const yPad = (maxY - minY) * 0.08;
      minY -= yPad;
      maxY += yPad;

      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const sx = (x) => margin.left + ((x - minX) / (maxX - minX)) * plotW;
      const sy = (y) => margin.top + (1 - (y - minY) / (maxY - minY)) * plotH;
      const polyline = (xVals, yVals) => xVals.map((x, i) => `${{sx(x)}},${{sy(yVals[i])}}`).join(" ");

      const ticks = 5;
      const gridLines = [];
      for (let i = 0; i <= ticks; i++) {{
        const t = i / ticks;
        const x = margin.left + t * plotW;
        const y = margin.top + t * plotH;
        const xv = minX + t * (maxX - minX);
        const yv = maxY - t * (maxY - minY);
        gridLines.push(`<line x1="${{x}}" y1="${{margin.top}}" x2="${{x}}" y2="${{height - margin.bottom}}" stroke="#eee" />`);
        gridLines.push(`<line x1="${{margin.left}}" y1="${{y}}" x2="${{width - margin.right}}" y2="${{y}}" stroke="#eee" />`);
        gridLines.push(`<text x="${{x}}" y="${{height - margin.bottom + 18}}" text-anchor="middle" font-size="11" fill="#666">${{xv.toFixed(2)}}</text>`);
        gridLines.push(`<text x="${{margin.left - 8}}" y="${{y + 4}}" text-anchor="end" font-size="11" fill="#666">${{yv.toFixed(2)}}</text>`);
      }}

      const series = traces.map((trace, idx) => {{
        const active = selectedTraceIndex === null ? true : idx === selectedTraceIndex;
        const lineColor = active ? "#18834a" : "#b7b7b7";
        const lineWidth = active ? 2.5 : 1.5;
        const opacity = active ? 1.0 : 0.45;
        const points = (trace.x || []).map((x, pointIdx) => {{
          const y = trace.y[pointIdx];
          const selected = idx === selectedTraceIndex && pointIdx === selectedPointIndex;
          const r = selected ? 7 : 4;
          const fill = selected ? "#f59e0b" : (active ? "#18834a" : "#9ca3af");
          const stroke = selected ? "#7a4b00" : "none";
          return `<circle cx="${{sx(x)}}" cy="${{sy(y)}}" r="${{r}}" fill="${{fill}}" stroke="${{stroke}}" stroke-width="1.5" />`;
        }}).join("");
        return `
          <polyline fill="none" stroke="${{lineColor}}" stroke-width="${{lineWidth}}" opacity="${{opacity}}" points="${{polyline(trace.x || [], trace.y || [])}}" />
          ${{points}}
        `;
      }}).join("");

      container.innerHTML = `
        <svg viewBox="0 0 ${{width}} ${{height}}" width="100%" role="img" aria-label="${{title || "Plot"}}">
          <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="white" rx="8" />
          ${{gridLines.join("")}}
          <line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="#444" />
          <line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="#444" />
          <text x="${{width / 2}}" y="18" text-anchor="middle" font-size="15" fill="#222">${{title}}</text>
          <text x="${{width / 2}}" y="${{height - 8}}" text-anchor="middle" font-size="12" fill="#444">Integrated path length</text>
          <text x="16" y="${{height / 2}}" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 16 ${{height / 2}})">Energy (kcal/mol)</text>
          ${{series}}
        </svg>
      `;
    }}
    let currentLayer = {default_layer_index};
    let currentGroup = {default_group_index};
    let currentChain = 0;
    let currentNetworkEdgeId = null;
    function getCurrentLayer() {{
      return layers[currentLayer] || null;
    }}
    function getCurrentGroups() {{
      const layer = getCurrentLayer();
      return layer ? (layer.groups || []) : [];
    }}
    function getCurrentViz() {{
      const groups = getCurrentGroups();
      const group = groups[currentGroup];
      return group ? group.viz : null;
    }}
    function getCurrentNetworkEdge() {{
      if (!networkPayload || !networkPayload.edges || currentNetworkEdgeId === null) return null;
      return networkPayload.edges.find((edge) => edge.id === currentNetworkEdgeId) || null;
    }}
    function getCurrentFrames() {{
      const networkEdge = getCurrentNetworkEdge();
      if (networkEdge) {{
        const edgeChain = networkEdge.viz && networkEdge.viz.chains ? networkEdge.viz.chains[currentChain] : null;
        return edgeChain ? edgeChain.frames : [];
      }}
      const viz = getCurrentViz();
      if (!viz) return [];
      const chain = viz.chains[currentChain];
      return chain ? chain.frames : [];
    }}
    const treeNodeMap = {{}};
    function renderTreeGraph() {{
      const svg = document.getElementById("treeSvg");
      if (!svg || layers.length <= 1) return;
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      const width = 900;
      const height = 220;
      const topPad = 30;
      const bottomPad = 30;
      const leftPad = 40;
      const rightPad = 40;

      for (let layerIdx = 0; layerIdx < layers.length; layerIdx++) {{
        const layer = layers[layerIdx];
        const groups = layer.groups || [];
        const y = layers.length === 1
          ? height / 2
          : topPad + (layerIdx * (height - topPad - bottomPad) / (layers.length - 1));
        for (let groupIdx = 0; groupIdx < groups.length; groupIdx++) {{
          const g = groups[groupIdx];
          const x = groups.length === 1
            ? width / 2
            : leftPad + (groupIdx * (width - leftPad - rightPad) / (groups.length - 1));
          treeNodeMap[g.node_index] = {{
            layerIdx: layerIdx,
            groupIdx: groupIdx,
            x: x,
            y: y,
            parent: g.parent_index,
            label: g.label
          }};
        }}
      }}

      Object.values(treeNodeMap).forEach((node) => {{
        if (node.parent == null || !(node.parent in treeNodeMap)) return;
        const parent = treeNodeMap[node.parent];
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", String(parent.x));
        line.setAttribute("y1", String(parent.y));
        line.setAttribute("x2", String(node.x));
        line.setAttribute("y2", String(node.y));
        line.setAttribute("stroke", "#b7b7b7");
        line.setAttribute("stroke-width", "1.5");
        svg.appendChild(line);
      }});

      Object.entries(treeNodeMap).forEach(([nodeIndex, node]) => {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("data-node-index", nodeIndex);
        group.style.cursor = "pointer";

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", String(node.x));
        circle.setAttribute("cy", String(node.y));
        circle.setAttribute("r", "12");
        circle.setAttribute("fill", "#1f77b4");
        circle.setAttribute("stroke", "#0f4872");
        circle.setAttribute("stroke-width", "1");
        group.appendChild(circle);

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", String(node.x + 16));
        text.setAttribute("y", String(node.y + 4));
        text.setAttribute("font-size", "12");
        text.setAttribute("fill", "#222");
        text.textContent = node.label;
        group.appendChild(text);

        group.addEventListener("click", () => {{
          currentLayer = node.layerIdx;
          currentGroup = node.groupIdx;
          syncChainOptions();
          syncSlider(0);
          renderHistory();
          renderFrame(parseInt(slider.value, 10));
          updateTreeSelection();
        }});

        svg.appendChild(group);
      }});
      updateTreeSelection();
    }}
    function renderNetworkGraph() {{
      const svg = document.getElementById("networkSvg");
      if (!svg || !networkPayload || !networkPayload.nodes || !networkPayload.nodes.length) return;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const width = 900;
      const height = 320;
      const pad = 34;
      const xs = networkPayload.nodes.map((node) => node.x);
      const ys = networkPayload.nodes.map((node) => node.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const scaleX = (value) => pad + ((value - minX) / ((maxX - minX) || 1)) * (width - 2 * pad);
      const scaleY = (value) => pad + ((value - minY) / ((maxY - minY) || 1)) * (height - 2 * pad);
      const nodeMap = {{}};
      networkPayload.nodes.forEach((node) => {{
        nodeMap[node.id] = {{
          x: scaleX(node.x),
          y: scaleY(node.y),
          label: node.label,
          is_root: node.is_root,
          is_target: node.is_target,
        }};
      }});
      networkPayload.edges.forEach((edge) => {{
        const src = nodeMap[edge.source];
        const dst = nodeMap[edge.target];
        if (!src || !dst) return;
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.setAttribute("data-edge-id", edge.id);
        group.style.cursor = "pointer";

        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", String(src.x));
        line.setAttribute("y1", String(src.y));
        line.setAttribute("x2", String(dst.x));
        line.setAttribute("y2", String(dst.y));
        line.setAttribute("stroke", edge.highlight ? "#f59e0b" : "#94a3b8");
        line.setAttribute("stroke-width", edge.highlight ? "4" : "2.25");
        line.setAttribute("opacity", edge.highlight ? "0.95" : "0.85");
        group.appendChild(line);

        const hit = document.createElementNS("http://www.w3.org/2000/svg", "line");
        hit.setAttribute("x1", String(src.x));
        hit.setAttribute("y1", String(src.y));
        hit.setAttribute("x2", String(dst.x));
        hit.setAttribute("y2", String(dst.y));
        hit.setAttribute("stroke", "transparent");
        hit.setAttribute("stroke-width", "14");
        group.appendChild(hit);

        const midX = (src.x + dst.x) / 2;
        const midY = (src.y + dst.y) / 2;
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", String(midX));
        label.setAttribute("y", String(midY - 8));
        label.setAttribute("text-anchor", "middle");
        label.setAttribute("font-size", "11");
        label.setAttribute("fill", "#334155");
        label.textContent = `${{edge.source}}→${{edge.target}}`;
        group.appendChild(label);

        group.addEventListener("click", () => {{
          currentNetworkEdgeId = edge.id;
          currentChain = 0;
          syncChainOptions();
          syncSlider(0);
          renderHistory();
          renderFrame(parseInt(slider.value, 10));
          updateNetworkSelection();
        }});
        svg.appendChild(group);
      }});

      networkPayload.nodes.forEach((node) => {{
        const data = nodeMap[node.id];
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", String(data.x));
        circle.setAttribute("cy", String(data.y));
        circle.setAttribute("r", data.is_root || data.is_target ? "13" : "11");
        circle.setAttribute("fill", data.is_root ? "#2563eb" : (data.is_target ? "#059669" : "#ffffff"));
        circle.setAttribute("stroke", "#1f2937");
        circle.setAttribute("stroke-width", "1.5");
        group.appendChild(circle);
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", String(data.x));
        text.setAttribute("y", String(data.y + 4));
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("font-size", "11");
        text.setAttribute("fill", data.is_root || data.is_target ? "#ffffff" : "#111827");
        text.textContent = String(node.id);
        group.appendChild(text);
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", String(data.x));
        label.setAttribute("y", String(data.y + 24));
        label.setAttribute("text-anchor", "middle");
        label.setAttribute("font-size", "11");
        label.setAttribute("fill", "#334155");
        label.textContent = data.label;
        group.appendChild(label);
        svg.appendChild(group);
      }});
      updateNetworkSelection();
    }}
    function updateNetworkSelection() {{
      const svg = document.getElementById("networkSvg");
      const info = document.getElementById("networkInfo");
      if (!svg || !networkPayload || !networkPayload.edges) return;
      const selected = getCurrentNetworkEdge();
      svg.querySelectorAll("g[data-edge-id]").forEach((elem) => {{
        const edge = networkPayload.edges.find((item) => item.id === elem.getAttribute("data-edge-id"));
        const line = elem.querySelector("line");
        if (!line || !edge) return;
        const selectedEdge = selected && selected.id === edge.id;
        line.setAttribute("stroke", selectedEdge ? "#dc2626" : (edge.highlight ? "#f59e0b" : "#94a3b8"));
        line.setAttribute("stroke-width", selectedEdge ? "5" : (edge.highlight ? "4" : "2.25"));
      }});
      if (!info) return;
      if (!selected) {{
        info.textContent = "Click an edge to load the corresponding best NEB pair. The best overall path is highlighted in gold.";
        return;
      }}
      const reverse = selected.reverse_barrier === null || selected.reverse_barrier === undefined
        ? "n/a"
        : selected.reverse_barrier.toFixed(2);
      info.textContent = `Selected edge ${{selected.source}}→${{selected.target}} | forward barrier: ${{selected.barrier.toFixed(2)}} kcal/mol | reverse barrier: ${{reverse}} kcal/mol | pair sum: ${{selected.pair_barrier_sum.toFixed(2)}} kcal/mol`;
    }}
    function updateTreeSelection() {{
      const svg = document.getElementById("treeSvg");
      if (!svg || layers.length <= 1) return;
      const groups = getCurrentGroups();
      const selectedGroup = groups[currentGroup];
      const selectedNode = selectedGroup ? selectedGroup.node_index : null;
      svg.querySelectorAll("g[data-node-index]").forEach((elem) => {{
        const c = elem.querySelector("circle");
        if (!c) return;
        if (String(selectedNode) === elem.getAttribute("data-node-index")) {{
          c.setAttribute("fill", "#f59e0b");
          c.setAttribute("stroke", "#7a4b00");
          c.setAttribute("stroke-width", "2");
        }} else {{
          c.setAttribute("fill", "#1f77b4");
          c.setAttribute("stroke", "#0f4872");
          c.setAttribute("stroke-width", "1");
        }}
      }});
    }}
    function syncChainOptions() {{
      const chainSelect = document.getElementById("chainSelect");
      const networkEdge = getCurrentNetworkEdge();
      if (networkEdge) {{
        const options = (networkEdge.viz && networkEdge.viz.chains ? networkEdge.viz.chains : [])
          .map((_, i) => `<option value="${{i}}">Chain ${{i}}</option>`)
          .join("");
        chainSelect.innerHTML = options;
        currentChain = 0;
        chainSelect.value = String(currentChain);
        chainSelect.disabled = true;
        return;
      }}
      const viz = getCurrentViz();
      if (!viz || !viz.chains.length) {{
        chainSelect.innerHTML = "";
        chainSelect.disabled = true;
        currentChain = 0;
        return;
      }}
      const options = viz.chains
        .map((_, i) => `<option value="${{i}}">Chain ${{i}}</option>`)
        .join("");
      chainSelect.innerHTML = options;
      currentChain = clamp(viz.default_chain_index || 0, 0, viz.chains.length - 1);
      chainSelect.value = String(currentChain);
      chainSelect.disabled = viz.chains.length <= 1;
    }}
    function renderHistory() {{
      if (getCurrentNetworkEdge()) {{
        document.getElementById("historyPanel").style.display = "none";
        return;
      }}
      const panel = document.getElementById("historyPanel");
      const viz = getCurrentViz();
      if (!viz || !viz.chains || viz.chains.length <= 1) {{
        panel.style.display = "none";
        return;
      }}
      panel.style.display = "block";
      renderPlot(
        "historyPlot",
        viz.chains.map((chain, i) => ({{
          x: chain.plot ? chain.plot.x : [],
          y: chain.plot ? chain.plot.y : [],
          label: `Chain ${{i}}`,
        }})),
        currentChain,
        null,
        "Optimization History"
      );
    }}
    function syncSlider(frameIndex = 0) {{
      const slider = document.getElementById("frameSlider");
      const frames = getCurrentFrames();
      const maxFrame = Math.max(frames.length - 1, 0);
      slider.max = String(maxFrame);
      slider.value = String(clamp(frameIndex, 0, maxFrame));
    }}
    function renderFrame(i) {{
      const frames = getCurrentFrames();
      const frame = frames[i];
      if (!frame) return;
      document.getElementById("frameLabel").textContent = String(i);
      const frameEl = document.getElementById("structureFrame");
      frameEl.srcdoc = makeStructureSrcdoc(frame.xyz_b64);
      const networkEdge = getCurrentNetworkEdge();
      const viz = networkEdge ? networkEdge.viz : getCurrentViz();
      const chain = viz && viz.chains ? viz.chains[currentChain] : null;
      renderPlot(
        "energyPlot",
        [{{
          x: chain && chain.plot ? chain.plot.x : [],
          y: chain && chain.plot ? chain.plot.y : [],
        }}],
        0,
        i,
        networkEdge ? `Edge ${{networkEdge.source}}→${{networkEdge.target}}` : "Energy Profile"
      );
    }}
    const slider = document.getElementById("frameSlider");
    const chainSelect = document.getElementById("chainSelect");
    chainSelect.addEventListener("change", (e) => {{
      currentChain = parseInt(e.target.value, 10) || 0;
      syncSlider(0);
      renderHistory();
      renderFrame(parseInt(slider.value, 10));
      updateTreeSelection();
    }});
    slider.addEventListener("input", (e) => renderFrame(parseInt(e.target.value, 10)));
    renderTreeGraph();
    if (networkPayload && networkPayload.edges && networkPayload.edges.length) {{
      const highlighted = networkPayload.edges.find((edge) => edge.highlight);
      currentNetworkEdgeId = highlighted ? highlighted.id : networkPayload.edges[0].id;
      renderNetworkGraph();
    }}
    syncChainOptions();
    syncSlider(0);
    renderHistory();
    renderFrame(parseInt(slider.value, 10));
  </script>
</body>
</html>
"""
