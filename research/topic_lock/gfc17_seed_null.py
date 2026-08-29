from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 168
NUM_WINDOWS = 20
DATASET = "autogluon/fev_datasets"
CONFIG = "proenfo_gfc17"
REFERENCE_SEED = 20260829

FEATURE_BASE = [
    "hour", "dow", "month", "doy",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168", "lag_336",
    "roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168",
]
UNSAFE_WITHIN_HORIZON = {
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_48",
    "roll_mean_24", "roll_std_24", "roll_mean_168", "roll_std_168",
}


def safe_feature_columns(weather: bool = True) -> list[str]:
    cols = [c for c in FEATURE_BASE if c not in UNSAFE_WITHIN_HORIZON]
    if weather:
        cols += ["weather", "weather_sq", "weather_hour"]
    return cols


def exact_cut_indices(n: int, horizon: int = HORIZON, num_windows: int = NUM_WINDOWS) -> list[int]:
    first = n - horizon - (num_windows - 1) * horizon
    if first < 1:
        raise ValueError(f"series too short: n={n}")
    return [first + i * horizon for i in range(num_windows)]


def _load_fev_long() -> pd.DataFrame:
    from datasets import load_dataset
    raw = load_dataset(DATASET, CONFIG, split="train").to_pandas()
    if raw.empty:
        raise RuntimeError("FEV GFC17 dataset is empty")
    if not isinstance(raw.iloc[0]["timestamp"], (list, tuple, np.ndarray)):
        df = raw.copy()
    else:
        parts = []
        for _, row in raw.iterrows():
            ts = list(row["timestamp"]); n = len(ts); part = pd.DataFrame({"timestamp": ts})
            for col in raw.columns:
                if col == "timestamp": continue
                value = row[col]
                if isinstance(value, np.ndarray): value = value.tolist()
                if isinstance(value, (list, tuple)) and len(value) == n: part[col] = list(value)
                else: part[col] = [value] * n
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
    required = {"id", "timestamp", "target", "airtemperature"}
    missing = required - set(df.columns)
    if missing: raise RuntimeError(f"FEV GFC17 missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["id", "timestamp"]).reset_index(drop=True)


def _supervised(g: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame({"timestamp": pd.to_datetime(g["timestamp"]), "target": pd.to_numeric(g["target"], errors="coerce")})
    ts=x["timestamp"]; x["hour"]=ts.dt.hour; x["dow"]=ts.dt.dayofweek; x["month"]=ts.dt.month; x["doy"]=ts.dt.dayofyear
    x["hour_sin"]=np.sin(2*np.pi*x["hour"]/24); x["hour_cos"]=np.cos(2*np.pi*x["hour"]/24)
    x["dow_sin"]=np.sin(2*np.pi*x["dow"]/7); x["dow_cos"]=np.cos(2*np.pi*x["dow"]/7)
    x["doy_sin"]=np.sin(2*np.pi*x["doy"]/365.25); x["doy_cos"]=np.cos(2*np.pi*x["doy"]/365.25)
    y=x["target"].astype(float)
    for lag in (1,2,3,24,48,168,336): x[f"lag_{lag}"]=y.shift(lag)
    for win in (24,168):
        x[f"roll_mean_{win}"]=y.shift(1).rolling(win).mean(); x[f"roll_std_{win}"]=y.shift(1).rolling(win).std()
    x["weather"]=pd.to_numeric(g["airtemperature"],errors="coerce").to_numpy(); x["weather_sq"]=x["weather"]**2; x["weather_hour"]=x["weather"]*x["hour_sin"]
    return x


def model_factory(name: str, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    if name=="HGB": return HistGradientBoostingRegressor(learning_rate=0.06,max_iter=300,max_leaf_nodes=31,l2_regularization=1.0,random_state=seed)
    if name=="RF": return RandomForestRegressor(n_estimators=250,min_samples_leaf=3,n_jobs=-1,random_state=seed)
    if name=="ET": return ExtraTreesRegressor(n_estimators=250,min_samples_leaf=2,n_jobs=-1,random_state=seed)
    raise ValueError(name)


def main() -> None:
    raise SystemExit("This public execution repo uses this module as a library for gdex_operational_gfs.py")

if __name__ == "__main__": main()
