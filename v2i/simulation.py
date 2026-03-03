from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _profile_lookup(signal_delay_profile: pd.DataFrame) -> Dict[tuple[int, str], tuple[float, float]]:
    return {
        (int(r.intersection_id), str(r.time_bucket)): (float(r.delay_factor), float(r.pedestrian_load))
        for r in signal_delay_profile.itertuples(index=False)
    }


def simulate(
    data: Dict[str, pd.DataFrame],
    priority_enabled: bool,
    priority_params: Dict[str, float],
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    intersections = data["intersections"].copy()
    tram_path = data["tram_path"].copy()
    profile = _profile_lookup(data["signal_delay_profile"])

    inter_map = intersections.set_index("intersection_id").to_dict(orient="index")

    tram_delays = []
    total_capacity = []
    priority_activations = 0

    for row in tram_path.itertuples(index=False):
        inter = inter_map[int(row.intersection_id)]
        delay_factor, ped_load = profile[(int(row.intersection_id), str(row.time_bucket))]

        cycle = float(inter["base_cycle_sec"])
        base_wait = cycle * 0.45 * delay_factor
        noise = rng.normal(0, 4.0)

        wait = max(0.0, base_wait + noise + 12.0 * ped_load)
        capacity = float(inter["base_capacity_veh_h"])

        if priority_enabled:
            if row.arrival_sec % cycle <= priority_params["activation_window_sec"]:
                priority_activations += 1
                wait = max(0.0, wait - priority_params["green_extension_sec"] - 0.5 * priority_params["queue_jump_sec"])
                capacity *= max(0.7, 1.0 - priority_params["green_extension_sec"] / 180.0)

        tram_delays.append(wait)
        total_capacity.append(capacity)

    tram_delays_arr = np.array(tram_delays, dtype=float)
    capacity_arr = np.array(total_capacity, dtype=float)

    return {
        "tram_avg_delay_sec": float(tram_delays_arr.mean()),
        "tram_p95_delay_sec": float(np.percentile(tram_delays_arr, 95)),
        "intersection_throughput_veh_per_h": float(capacity_arr.mean()),
        "priority_activations": int(priority_activations),
        "samples": int(len(tram_delays_arr)),
    }


def compare_scenarios(
    data: Dict[str, pd.DataFrame],
    baseline_params: Dict[str, float],
    priority_params: Dict[str, float],
) -> pd.DataFrame:
    base = simulate(data, priority_enabled=False, priority_params=baseline_params)
    v2i = simulate(data, priority_enabled=True, priority_params=priority_params)

    throughput_change = (
        (v2i["intersection_throughput_veh_per_h"] - base["intersection_throughput_veh_per_h"])
        / base["intersection_throughput_veh_per_h"]
        * 100
    )

    summary = pd.DataFrame(
        [
            {"scenario": "baseline", **base, "throughput_change_pct": 0.0},
            {"scenario": "v2i_priority", **v2i, "throughput_change_pct": throughput_change},
        ]
    )
    return summary
