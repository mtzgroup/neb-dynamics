from __future__ import annotations

import math

import numpy as np

from neb_dynamics.pot import Pot

K_BOLTZMANN = 1.380649e-23
PLANCK = 6.62607015e-34
R_GAS_KCAL = 0.00198720425864083


def default_initial_conditions(pot: Pot) -> dict[int, float]:
    return {
        int(node_index): (1.0 if int(node_index) == 0 else 0.0)
        for node_index in sorted(pot.graph.nodes)
    }


def normalize_initial_conditions(
    pot: Pot,
    initial_conditions: dict[int, float] | None = None,
) -> dict[int, float]:
    values = default_initial_conditions(pot)
    if initial_conditions:
        for key, value in initial_conditions.items():
            node_index = int(key)
            if node_index in values:
                values[node_index] = float(value)
    return values


def node_label(pot: Pot, node_index: int) -> str:
    node_attrs = pot.graph.nodes[node_index]
    molecule = node_attrs.get("molecule")
    if molecule is not None:
        try:
            smiles = molecule.force_smiles()
            if smiles:
                return f"{node_index}: {smiles}"
        except Exception:
            pass
    td = node_attrs.get("td")
    graph = getattr(td, "graph", None)
    if graph is not None:
        try:
            smiles = graph.force_smiles()
            if smiles:
                return f"{node_index}: {smiles}"
        except Exception:
            pass
    return str(node_index)


def edge_rate_constant(barrier_kcal: float, temperature_kelvin: float) -> float:
    if temperature_kelvin <= 0:
        raise ValueError("temperature_kelvin must be positive")
    prefactor = (K_BOLTZMANN * temperature_kelvin) / PLANCK
    exponent = -float(barrier_kcal) / (R_GAS_KCAL * temperature_kelvin)
    return prefactor * math.exp(exponent)


def _default_end_time(rate_constants: list[float]) -> float:
    positive = [float(rate) for rate in rate_constants if float(rate) > 0.0]
    if not positive:
        return 1.0
    return 10.0 / max(positive)


def build_kmc_payload(
    pot: Pot,
    temperature_kelvin: float = 298.15,
    initial_conditions: dict[int, float] | None = None,
) -> dict:
    normalized_initial = normalize_initial_conditions(
        pot=pot, initial_conditions=initial_conditions
    )
    nodes = [
        {
            "id": int(node_index),
            "label": node_label(pot, int(node_index)),
            "initial": float(normalized_initial.get(int(node_index), 0.0)),
        }
        for node_index in sorted(pot.graph.nodes)
    ]

    edges = []
    for source_node, target_node in sorted(pot.graph.edges):
        edge_attrs = pot.graph.edges[(source_node, target_node)]
        barrier = edge_attrs.get("barrier")
        if barrier is None:
            continue
        rate_constant = edge_rate_constant(
            barrier_kcal=float(barrier),
            temperature_kelvin=temperature_kelvin,
        )
        edges.append(
            {
                "source": int(source_node),
                "target": int(target_node),
                "barrier": float(barrier),
                "rate_constant": rate_constant,
                "reaction": str(edge_attrs.get("reaction") or ""),
            }
        )

    return {
        "temperature_kelvin": float(temperature_kelvin),
        "nodes": nodes,
        "edges": edges,
        "initial_conditions": normalized_initial,
        "default_end_time": _default_end_time(
            [edge["rate_constant"] for edge in edges]
        ),
    }


def _rate_matrix(payload: dict) -> tuple[list[int], np.ndarray]:
    node_ids = [int(node["id"]) for node in payload["nodes"]]
    index_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    matrix = np.zeros((len(node_ids), len(node_ids)), dtype=float)

    for edge in payload["edges"]:
        source_node = int(edge["source"])
        target_node = int(edge["target"])
        rate = float(edge["rate_constant"])
        if rate <= 0.0:
            continue
        source_idx = index_map[source_node]
        target_idx = index_map[target_node]
        matrix[target_idx, source_idx] += rate
        matrix[source_idx, source_idx] -= rate

    return node_ids, matrix


def _rk4_step(state: np.ndarray, dt: float, matrix: np.ndarray) -> np.ndarray:
    k1 = matrix @ state
    k2 = matrix @ (state + 0.5 * dt * k1)
    k3 = matrix @ (state + 0.5 * dt * k2)
    k4 = matrix @ (state + dt * k3)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[next_state < 0.0] = 0.0
    total = next_state.sum()
    if total > 0.0:
        reference_total = state.sum()
        if reference_total > 0.0:
            next_state *= reference_total / total
    return next_state


def simulate_kmc(
    pot: Pot,
    temperature_kelvin: float = 298.15,
    initial_conditions: dict[int, float] | None = None,
    max_steps: int = 200,
    final_time: float | None = None,
    seed: int | None = None,
) -> dict:
    del seed
    payload = build_kmc_payload(
        pot=pot,
        temperature_kelvin=temperature_kelvin,
        initial_conditions=initial_conditions,
    )
    node_ids, matrix = _rate_matrix(payload)
    state = np.array(
        [float(payload["initial_conditions"].get(node_id, 0.0)) for node_id in node_ids],
        dtype=float,
    )
    n_steps = max(int(max_steps), 1)
    end_time = float(final_time if final_time is not None else payload["default_end_time"])
    if end_time < 0.0:
        raise ValueError("final_time must be non-negative")
    dt = end_time / n_steps if n_steps > 0 else 0.0

    history = [
        {
            "time": 0.0,
            "event": None,
            "populations": {node_id: float(state[idx]) for idx, node_id in enumerate(node_ids)},
        }
    ]
    time = 0.0
    for _ in range(n_steps):
        if dt > 0.0:
            state = _rk4_step(state=state, dt=dt, matrix=matrix)
        time += dt
        history.append(
            {
                "time": time,
                "event": None,
                "populations": {node_id: float(state[idx]) for idx, node_id in enumerate(node_ids)},
            }
        )

    return {
        "temperature_kelvin": float(temperature_kelvin),
        "max_steps": n_steps,
        "final_time": end_time,
        "seed": None,
        "history": history,
        "final_populations": history[-1]["populations"],
    }
