from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_training_table(data: Dict[str, pd.DataFrame], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    intersections = data["intersections"].set_index("intersection_id")
    traffic = data["traffic"].groupby("intersection_id", as_index=False)["traffic_level"].mean()

    rows = []
    for tram in data["tram_schedule"].itertuples(index=False):
        for inter in intersections.itertuples():
            distance = inter.distance_from_start_m
            speed = np.clip(rng.normal(22, 2.2), 14, 32)
            traffic_level = float(traffic.loc[traffic["intersection_id"] == inter.Index, "traffic_level"].iloc[0])
            eta = distance / (speed / 3.6)
            actual = eta * (1 + 0.35 * traffic_level) + rng.normal(0, 4)
            stop_flag = int(actual - eta > 10)
            rows.append(
                {
                    "timestamp_sec": int(tram.start_sec + eta),
                    "intersection_id": int(inter.Index),
                    "speed_kmph": speed,
                    "distance_m": distance,
                    "traffic_level": traffic_level,
                    "phase_ratio": inter.tram_green_sec / inter.cycle_sec,
                    "eta_target_sec": max(1.0, actual),
                    "stop_flag": stop_flag,
                }
            )
    return pd.DataFrame(rows).sort_values("timestamp_sec")


def train_sklearn_delay_model(df: pd.DataFrame) -> Dict[str, float]:
    y = df["eta_target_sec"]
    x = df[["intersection_id", "speed_kmph", "distance_m", "traffic_level", "phase_ratio"]]

    split_idx = int(len(df) * 0.7)
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }


def train_torch_delay_model(df: pd.DataFrame, epochs: int = 40) -> Dict[str, float] | None:
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    x = df[["intersection_id", "speed_kmph", "distance_m", "traffic_level", "phase_ratio"]].to_numpy(dtype=np.float32)
    y = df[["eta_target_sec"]].to_numpy(dtype=np.float32)

    split = int(len(df) * 0.7)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    xm, xs = x_train.mean(0), x_train.std(0) + 1e-6
    x_train = (x_train - xm) / xs
    x_test = (x_test - xm) / xs

    xtr, ytr = torch.tensor(x_train), torch.tensor(y_train)
    xte, yte = torch.tensor(x_test), torch.tensor(y_test)

    model = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xtr), ytr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = model(xte)
        mae = torch.mean(torch.abs(pred - yte)).item()
        rmse = torch.sqrt(torch.mean((pred - yte) ** 2)).item()
    return {"mae": float(mae), "rmse": float(rmse), "epochs": int(epochs)}
