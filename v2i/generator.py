from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class GenerationConfig:
    intersections: int = 8
    trams_per_day: int = 180
    day_seconds: int = 24 * 3600
    seed: int = 42


def _time_bucket(sec_of_day: int) -> str:
    hour = sec_of_day // 3600
    if 7 <= hour < 10 or 16 <= hour < 19:
        return "peak"
    if 10 <= hour < 16:
        return "day"
    return "offpeak"


def generate_synthetic_data(config: GenerationConfig) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)

    intersections_df = pd.DataFrame(
        {
            "intersection_id": np.arange(config.intersections),
            "x": rng.integers(0, 4, size=config.intersections),
            "y": rng.integers(0, 4, size=config.intersections),
            "base_cycle_sec": rng.integers(70, 120, size=config.intersections),
            "base_green_tram_sec": rng.integers(10, 22, size=config.intersections),
            "base_green_cars_sec": rng.integers(25, 55, size=config.intersections),
            "base_capacity_veh_h": rng.integers(900, 1800, size=config.intersections),
        }
    ).sort_values("intersection_id")

    trip_start = np.sort(rng.integers(5 * 3600, 23 * 3600, size=config.trams_per_day))
    route_choice = rng.integers(0, config.intersections, size=config.trams_per_day)
    planned_headway = np.clip(rng.normal(7 * 60, 100, size=config.trams_per_day), 240, 780)

    tram_schedule_df = pd.DataFrame(
        {
            "tram_id": np.arange(config.trams_per_day),
            "start_sec": trip_start,
            "entry_intersection": route_choice,
            "planned_headway_sec": planned_headway.round(0).astype(int),
            "line_id": rng.choice([3, 6, 9, 11], size=config.trams_per_day),
        }
    )

    records = []
    for _, row in intersections_df.iterrows():
        for bucket, factor in [("peak", 1.25), ("day", 1.0), ("offpeak", 0.75)]:
            records.append(
                {
                    "intersection_id": int(row["intersection_id"]),
                    "time_bucket": bucket,
                    "delay_factor": factor + rng.normal(0, 0.06),
                    "pedestrian_load": float(np.clip(rng.normal(0.5, 0.2), 0.1, 1.0)),
                }
            )
    signal_delay_df = pd.DataFrame(records)

    travel_records = []
    for _, trip in tram_schedule_df.iterrows():
        base_path = [(trip["entry_intersection"] + i) % config.intersections for i in range(4)]
        t = int(trip["start_sec"])
        for step, inter_id in enumerate(base_path):
            t += int(np.clip(rng.normal(150, 30), 70, 280))
            travel_records.append(
                {
                    "tram_id": int(trip["tram_id"]),
                    "step_idx": step,
                    "intersection_id": int(inter_id),
                    "arrival_sec": t,
                    "time_bucket": _time_bucket(t),
                }
            )
    tram_path_df = pd.DataFrame(travel_records)

    return {
        "intersections": intersections_df,
        "tram_schedule": tram_schedule_df,
        "signal_delay_profile": signal_delay_df,
        "tram_path": tram_path_df,
    }


def default_priority_params() -> Dict[str, float]:
    return {
        "green_extension_sec": 12.0,
        "queue_jump_sec": 6.0,
        "activation_window_sec": 35.0,
    }


def optimization_bounds() -> Dict[str, Tuple[float, float]]:
    return {
        "green_extension_sec": (2.0, 30.0),
        "queue_jump_sec": (0.0, 20.0),
        "activation_window_sec": (8.0, 80.0),
    }
