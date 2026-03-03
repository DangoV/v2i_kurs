from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd


TABLES = ["intersections", "tram_schedule", "signal_delay_profile", "tram_path"]


def save_all_formats(data: Dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        df.to_csv(out_path / f"{name}.csv", index=False)
        df.to_json(out_path / f"{name}.json", orient="records", force_ascii=False, indent=2)

    sqlite_path = out_path / "v2i_data.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        for name, df in data.items():
            df.to_sql(name, conn, if_exists="replace", index=False)

    meta = {"tables": list(data.keys()), "sqlite": str(sqlite_path.name)}
    (out_path / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_from_csv_dir(in_dir: str | Path) -> Dict[str, pd.DataFrame]:
    in_path = Path(in_dir)
    return {name: pd.read_csv(in_path / f"{name}.csv") for name in TABLES}
