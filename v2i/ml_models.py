from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from v2i.simulation import _profile_lookup


def build_training_table(data: Dict[str, pd.DataFrame], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    intersections = data["intersections"].set_index("intersection_id").to_dict(orient="index")
    profile = _profile_lookup(data["signal_delay_profile"])

    rows = []
    for row in data["tram_path"].itertuples(index=False):
        inter = intersections[int(row.intersection_id)]
        delay_factor, ped = profile[(int(row.intersection_id), str(row.time_bucket))]
        synthetic_delay = (
            inter["base_cycle_sec"] * 0.42 * delay_factor
            + 13.0 * ped
            + rng.normal(0, 4.5)
        )
        rows.append(
            {
                "intersection_id": int(row.intersection_id),
                "time_bucket": str(row.time_bucket),
                "base_cycle_sec": float(inter["base_cycle_sec"]),
                "base_capacity_veh_h": float(inter["base_capacity_veh_h"]),
                "pedestrian_load": float(ped),
                "delay_factor": float(delay_factor),
                "delay_sec": max(0.0, synthetic_delay),
            }
        )
    return pd.DataFrame(rows)


def train_sklearn_delay_model(df: pd.DataFrame) -> Dict[str, float]:
    y = df["delay_sec"]
    x = df.drop(columns=["delay_sec"])

    cat_cols = ["time_bucket"]
    num_cols = [c for c in x.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    model = Pipeline(
        steps=[
            ("pre", pre),
            ("rf", RandomForestRegressor(n_estimators=180, random_state=42)),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def train_torch_delay_model(df: pd.DataFrame, epochs: int = 60) -> Dict[str, float] | None:
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    x = df[["intersection_id", "base_cycle_sec", "base_capacity_veh_h", "pedestrian_load", "delay_factor"]].values
    y = df[["delay_sec"]].values

    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(x_t.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(x_t)
        mae = torch.mean(torch.abs(pred - y_t)).item()
    return {"mae": float(mae), "epochs": int(epochs)}
