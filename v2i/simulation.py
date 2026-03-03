from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _crossing_events(data: Dict[str, pd.DataFrame], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inter = data["intersections"].copy()
    sched = data["tram_schedule"].copy()

    avg_speed_ms = 22 / 3.6
    rows = []
    for s in sched.itertuples(index=False):
        for r in inter.itertuples(index=False):
            travel_t = r.distance_from_start_m / avg_speed_ms
            arrival = s.start_sec + travel_t + rng.normal(0, 8)
            rows.append(
                {
                    "tram_id": int(s.tram_id),
                    "intersection_id": int(r.intersection_id),
                    "arrival_sec": max(0.0, arrival),
                    "cycle_sec": float(r.cycle_sec),
                    "tram_green_sec": float(r.tram_green_sec),
                    "base_capacity_veh_h": float(r.base_capacity_veh_h),
                }
            )
    events = pd.DataFrame(rows)
    traffic_mean = data["traffic"].groupby("intersection_id", as_index=False)["traffic_level"].mean()
    events = events.merge(traffic_mean, on="intersection_id", how="left")
    return events


def simulate(data: Dict[str, pd.DataFrame], priority_enabled: bool, params: Dict[str, float], seed: int = 42) -> Dict[str, float]:
    events = _crossing_events(data, seed=seed)
    rng = np.random.default_rng(seed)

    v2i = data["v2i_log"]
    packet_loss_rate = float(v2i["packet_loss_flag"].mean())

    total_delay = []
    total_delay_cars = []
    stops = 0
    priority_ok = 0
    priority_req = 0
    capacities = []

    for e in events.itertuples(index=False):
        phase = e.arrival_sec % e.cycle_sec
        red_remaining = max(0.0, e.cycle_sec - phase) if phase > e.tram_green_sec else 0.0
        webster = (e.cycle_sec * (1 - e.tram_green_sec / e.cycle_sec) ** 2) / max(0.15, (2 * (1 - min(0.95, e.traffic_level))))
        baseline_delay = max(0.0, 0.45 * red_remaining + 0.25 * webster + rng.normal(0, 2.5))

        delay = baseline_delay
        cap = e.base_capacity_veh_h

        if priority_enabled:
            eta_to_stopline = max(0.0, red_remaining)
            priority_req += int(eta_to_stopline <= params["eta_threshold_sec"])
            if eta_to_stopline <= params["eta_threshold_sec"] and rng.random() > packet_loss_rate:
                extension = min(params["green_extension_sec"], params["max_extension_sec"])
                delay = max(0.0, baseline_delay - extension)
                cap *= max(0.88, 1 - extension / 220)
                total_delay_cars.append(extension * e.traffic_level * 0.7)
                priority_ok += 1

        total_delay.append(delay)
        capacities.append(cap)
        stops += int(delay > 3)

    arr = np.array(total_delay)
    total_route_length_km = float(data["intersections"]["distance_from_start_m"].max()) / 1000.0
    tram_count = data["tram_schedule"].shape[0]
    base_run_sec = total_route_length_km / 22.0 * 3600.0
    avg_delay_per_tram = float(arr.sum() / max(1, tram_count))
    avg_speed_kmph = total_route_length_km / max(1e-6, (base_run_sec + avg_delay_per_tram) / 3600.0)

    return {
        "tram_avg_delay_sec": float(arr.mean()),
        "tram_p95_delay_sec": float(np.percentile(arr, 95)),
        "intersection_throughput_veh_per_h": float(np.mean(capacities)),
        "avg_speed_kmph": float(avg_speed_kmph),
        "stops_per_tram": float(stops / max(1, tram_count)),
        "priority_requests": int(priority_req),
        "priority_success_rate": float(priority_ok / max(1, priority_req)),
        "cars_delay_penalty_sec": float(np.sum(total_delay_cars) if total_delay_cars else 0.0),
    }


def compare_scenarios(data: Dict[str, pd.DataFrame], params: Dict[str, float]) -> pd.DataFrame:
    base_params = params.copy()
    base = simulate(data, priority_enabled=False, params=base_params)
    prio = simulate(data, priority_enabled=True, params=params)
    prio["delay_reduction_pct"] = (base["tram_avg_delay_sec"] - prio["tram_avg_delay_sec"]) / max(1e-6, base["tram_avg_delay_sec"]) * 100
    prio["throughput_change_pct"] = (
        (prio["intersection_throughput_veh_per_h"] - base["intersection_throughput_veh_per_h"])
        / base["intersection_throughput_veh_per_h"]
        * 100
    )
    base["delay_reduction_pct"] = 0.0
    base["throughput_change_pct"] = 0.0
    return pd.DataFrame([
        {"scenario": "baseline", **base},
        {"scenario": "v2i_priority", **prio},
    ])


def sensitivity_analysis(data: Dict[str, pd.DataFrame], params: Dict[str, float]) -> pd.DataFrame:
    scenarios = []
    for traffic_scale in [0.4, 0.7, 1.0]:
        d = {k: v.copy() for k, v in data.items()}
        d["traffic"]["traffic_level"] = np.clip(d["traffic"]["traffic_level"] * (0.7 + traffic_scale / 1.2), 0.3, 1.0)
        m = simulate(d, priority_enabled=True, params=params)
        scenarios.append({"factor": "traffic_level", "value": traffic_scale, **m})

    for ext in [5, 10, 15, 20]:
        p = params.copy()
        p["green_extension_sec"] = ext
        p["max_extension_sec"] = min(15, ext)
        m = simulate(data, priority_enabled=True, params=p)
        scenarios.append({"factor": "green_extension_sec", "value": ext, **m})

    for headway in [300, 600, 900]:
        d = {k: v.copy() for k, v in data.items()}
        t = d["tram_schedule"].copy()
        starts = np.arange(0, 7200, headway)
        d["tram_schedule"] = pd.DataFrame({"tram_id": np.arange(1, len(starts) + 1), "start_sec": starts, "line_id": 3, "planned_headway_sec": headway})
        m = simulate(d, priority_enabled=True, params=params)
        scenarios.append({"factor": "headway_sec", "value": headway, **m})

    for loss in [0.0, 0.03, 0.05, 0.10]:
        d = {k: v.copy() for k, v in data.items()}
        n = d["v2i_log"].shape[0]
        flags = np.zeros(n, dtype=int)
        flags[: int(n * loss)] = 1
        d["v2i_log"]["packet_loss_flag"] = flags
        m = simulate(d, priority_enabled=True, params=params)
        scenarios.append({"factor": "packet_loss", "value": loss, **m})

    return pd.DataFrame(scenarios)
