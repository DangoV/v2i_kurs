from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import differential_evolution

from v2i.generator import optimization_bounds
from v2i.simulation import simulate


ORDER = ["eta_threshold_sec", "green_extension_sec", "alpha_tram"]


def _objective(vec: np.ndarray, data: Dict) -> float:
    params = {
        "eta_threshold_sec": float(vec[0]),
        "green_extension_sec": float(vec[1]),
        "max_extension_sec": 15.0,
        "alpha_tram": float(vec[2]),
        "beta_cars": float(1 - vec[2]),
    }
    baseline = simulate(data, priority_enabled=False, params=params)
    priority = simulate(data, priority_enabled=True, params=params)

    tram_gain = baseline["tram_avg_delay_sec"] - priority["tram_avg_delay_sec"]
    cars_penalty = priority["cars_delay_penalty_sec"] / 300.0
    throughput_penalty = max(0.0, baseline["intersection_throughput_veh_per_h"] - priority["intersection_throughput_veh_per_h"]) / 90.0

    return -(params["alpha_tram"] * tram_gain - params["beta_cars"] * (cars_penalty + throughput_penalty))


def optimize_priority_params(data: Dict) -> Dict[str, float]:
    b = optimization_bounds()
    bounds = [b[k] for k in ORDER]
    res = differential_evolution(lambda x: _objective(x, data), bounds=bounds, seed=42, maxiter=30)
    best = {k: float(v) for k, v in zip(ORDER, res.x)}
    best["max_extension_sec"] = 15.0
    best["beta_cars"] = float(1 - best["alpha_tram"])
    return best
