from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import differential_evolution

from v2i.generator import optimization_bounds
from v2i.simulation import simulate


PARAM_ORDER = ["green_extension_sec", "queue_jump_sec", "activation_window_sec"]


def _objective(vec: np.ndarray, data) -> float:
    params = {k: float(v) for k, v in zip(PARAM_ORDER, vec)}
    base_metrics = simulate(data, priority_enabled=False, priority_params=params)
    prio_metrics = simulate(data, priority_enabled=True, priority_params=params)

    delay_gain = base_metrics["tram_avg_delay_sec"] - prio_metrics["tram_avg_delay_sec"]
    throughput_drop = base_metrics["intersection_throughput_veh_per_h"] - prio_metrics["intersection_throughput_veh_per_h"]

    penalty = max(0.0, throughput_drop) * 0.04
    score = -(delay_gain - penalty)
    return score


def optimize_priority_params(data) -> Dict[str, float]:
    bounds_map = optimization_bounds()
    bounds = [bounds_map[k] for k in PARAM_ORDER]

    result = differential_evolution(
        func=lambda x: _objective(x, data),
        bounds=bounds,
        seed=42,
        maxiter=25,
        polish=True,
    )
    return {k: float(v) for k, v in zip(PARAM_ORDER, result.x)}
