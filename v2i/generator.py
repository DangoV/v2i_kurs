from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class GenerationConfig:
    intersections: int = 8
    simulation_seconds: int = 7200  # 2 hours
    headway_sec: int = 600  # 10 minutes
    route_km: float = 7.0
    avg_speed_kmph: float = 22.0
    seed: int = 42


def default_priority_params() -> Dict[str, float]:
    return {
        "eta_threshold_sec": 20.0,
        "green_extension_sec": 12.0,
        "max_extension_sec": 15.0,
        "alpha_tram": 0.7,
        "beta_cars": 0.3,
    }


def optimization_bounds() -> Dict[str, tuple[float, float]]:
    return {
        "eta_threshold_sec": (8.0, 35.0),
        "green_extension_sec": (5.0, 15.0),
        "alpha_tram": (0.55, 0.95),
    }


def _build_intersections(rng: np.random.Generator, cfg: GenerationConfig) -> pd.DataFrame:
    segment_m = rng.integers(500, 901, size=cfg.intersections)
    distance_m = np.cumsum(segment_m)
    return pd.DataFrame(
        {
            "intersection_id": np.arange(1, cfg.intersections + 1),
            "distance_from_start_m": distance_m,
            "cycle_sec": rng.integers(84, 98, size=cfg.intersections),
            "tram_green_sec": rng.integers(25, 36, size=cfg.intersections),
            "min_phase_sec": rng.integers(18, 24, size=cfg.intersections),
            "base_capacity_veh_h": rng.integers(950, 1800, size=cfg.intersections),
        }
    )


def _build_tram_schedule(cfg: GenerationConfig) -> pd.DataFrame:
    starts = np.arange(0, cfg.simulation_seconds, cfg.headway_sec)
    return pd.DataFrame(
        {
            "tram_id": np.arange(1, len(starts) + 1),
            "start_sec": starts,
            "line_id": 3,
            "planned_headway_sec": cfg.headway_sec,
        }
    )


def _build_traffic_profile(rng: np.random.Generator, intersections: pd.DataFrame, cfg: GenerationConfig) -> pd.DataFrame:
    ts = np.arange(cfg.simulation_seconds)
    rows = []
    for i in intersections["intersection_id"]:
        base = 0.55 + 0.08 * np.sin(2 * np.pi * ts / cfg.simulation_seconds)
        peak = ((ts > 1200) & (ts < 3000)) | ((ts > 4800) & (ts < 6600))
        lvl = np.clip(base + peak * 0.22 + rng.normal(0, 0.05, size=len(ts)), 0.3, 1.0)
        q_len = np.clip(20 + 80 * lvl + rng.normal(0, 6, size=len(ts)), 5, 140)
        veh = np.clip(8 + 35 * lvl + rng.normal(0, 4, size=len(ts)), 1, 80)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp_sec": ts,
                    "intersection_id": i,
                    "vehicle_count_nearby": veh.round().astype(int),
                    "traffic_level": lvl,
                    "queue_length_m": q_len,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _build_signal_log(intersections: pd.DataFrame, cfg: GenerationConfig) -> pd.DataFrame:
    rows = []
    for r in intersections.itertuples(index=False):
        for t in range(cfg.simulation_seconds):
            phase_t = t % int(r.cycle_sec)
            is_tram_green = int(phase_t < int(r.tram_green_sec))
            rows.append(
                {
                    "timestamp_sec": t,
                    "intersection_id": int(r.intersection_id),
                    "phase_id": "tram_green" if is_tram_green else "cross_green",
                    "phase_duration_sec": int(r.tram_green_sec if is_tram_green else r.cycle_sec - r.tram_green_sec),
                    "remaining_time_sec": int((r.tram_green_sec - phase_t) if is_tram_green else (r.cycle_sec - phase_t)),
                    "priority_activated": 0,
                }
            )
    return pd.DataFrame(rows)


def _build_v2i_log(rng: np.random.Generator, cfg: GenerationConfig) -> pd.DataFrame:
    ts = np.arange(cfg.simulation_seconds)
    latency = np.clip(rng.normal(55, 18, size=len(ts)), 20, 110)
    packet_loss = (rng.random(size=len(ts)) < 0.05).astype(int)
    rssi = np.clip(rng.normal(-63, 7, size=len(ts)), -85, -42)
    return pd.DataFrame(
        {
            "timestamp_sec": ts,
            "latency_ms": latency,
            "packet_loss_flag": packet_loss,
            "rssi_dbm": rssi,
        }
    )


def _build_tram_position(
    rng: np.random.Generator,
    schedule: pd.DataFrame,
    intersections: pd.DataFrame,
    cfg: GenerationConfig,
) -> pd.DataFrame:
    rows = []
    route_m = float(intersections["distance_from_start_m"].max() + 500)
    avg_speed_ms = cfg.avg_speed_kmph / 3.6
    for s in schedule.itertuples(index=False):
        for t in range(s.start_sec, cfg.simulation_seconds):
            elapsed = t - s.start_sec
            if elapsed < 0:
                continue
            speed = np.clip(avg_speed_ms + rng.normal(0, 0.7), 1.8, 10.0)
            distance = min(route_m, elapsed * speed)
            rows.append(
                {
                    "timestamp_sec": t,
                    "tram_id": int(s.tram_id),
                    "speed_kmph": speed * 3.6,
                    "acceleration_mps2": float(rng.normal(0, 0.25)),
                    "distance_from_start_m": distance,
                    "distance_to_intersection_m": max(0.0, intersections["distance_from_start_m"].min() - distance),
                }
            )
            if distance >= route_m:
                break
    return pd.DataFrame(rows)


def generate_synthetic_data(config: GenerationConfig) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    intersections = _build_intersections(rng, config)
    tram_schedule = _build_tram_schedule(config)
    traffic = _build_traffic_profile(rng, intersections, config)
    signal_log = _build_signal_log(intersections, config)
    v2i_log = _build_v2i_log(rng, config)
    tram_position = _build_tram_position(rng, tram_schedule, intersections, config)

    return {
        "intersections": intersections,
        "tram_schedule": tram_schedule,
        "traffic": traffic,
        "signal_log": signal_log,
        "v2i_log": v2i_log,
        "tram_position": tram_position,
    }
