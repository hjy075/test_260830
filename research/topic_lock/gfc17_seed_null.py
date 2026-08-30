from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 168
NUM_WINDOWS = 20
DATASET = "autogluon/fev_datasets"
CONFIG = "proenfo_gfc17"
REFERENCE_SEED = 20260829
DEFAULT_MARGIN_THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)

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
    local_path = os.getenv("FEV_GFC17_PARQUET")
    if local_path:
        raw = pd.read_parquet(local_path)
    else:
        from datasets import load_dataset

        raw = load_dataset(DATASET, CONFIG, split="train").to_pandas()
    if raw.empty:
        raise RuntimeError("FEV GFC17 dataset is empty")
    if not isinstance(raw.iloc[0]["timestamp"], (list, tuple, np.ndarray)):
        df = raw.copy()
    else:
        parts = []
        for _, row in raw.iterrows():
            ts = list(row["timestamp"])
            n = len(ts)
            part = pd.DataFrame({"timestamp": ts})
            for col in raw.columns:
                if col == "timestamp":
                    continue
                value = row[col]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, (list, tuple)) and len(value) == n:
                    part[col] = list(value)
                else:
                    part[col] = [value] * n
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
    required = {"id", "timestamp", "target", "airtemperature"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"FEV GFC17 missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["id", "timestamp"]).reset_index(drop=True)


def _supervised(g: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame({
        "timestamp": pd.to_datetime(g["timestamp"]),
        "target": pd.to_numeric(g["target"], errors="coerce"),
    })
    ts = x["timestamp"]
    x["hour"] = ts.dt.hour
    x["dow"] = ts.dt.dayofweek
    x["month"] = ts.dt.month
    x["doy"] = ts.dt.dayofyear
    x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24)
    x["dow_sin"] = np.sin(2 * np.pi * x["dow"] / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x["dow"] / 7)
    x["doy_sin"] = np.sin(2 * np.pi * x["doy"] / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * x["doy"] / 365.25)
    y = x["target"].astype(float)
    for lag in (1, 2, 3, 24, 48, 168, 336):
        x[f"lag_{lag}"] = y.shift(lag)
    for win in (24, 168):
        x[f"roll_mean_{win}"] = y.shift(1).rolling(win).mean()
        x[f"roll_std_{win}"] = y.shift(1).rolling(win).std()
    x["weather"] = pd.to_numeric(g["airtemperature"], errors="coerce").to_numpy()
    x["weather_sq"] = x["weather"] ** 2
    x["weather_hour"] = x["weather"] * x["hour_sin"]
    return x


def model_factory(name: str, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

    if name == "HGB":
        return HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=seed
        )
    if name == "ET":
        return ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    raise ValueError(name)


def run_oracle_seed(seed: int) -> pd.DataFrame:
    df = _load_fev_long()
    rows = []
    cols = safe_feature_columns(weather=True)
    for sid, g in df.groupby("id", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        sup = _supervised(g)
        cuts = exact_cut_indices(len(g))
        for w, cut in enumerate(cuts, 1):
            train = sup.iloc[:cut]
            test = sup.iloc[cut:cut + HORIZON]
            good = train[cols + ["target"]].dropna().index
            Xtr = train.loc[good, cols]
            ytr = train.loc[good, "target"].astype(float)
            Xte = test[cols]
            if Xte.isna().any().any():
                raise RuntimeError(f"oracle features contain NA: id={sid} window={w}")
            truth = test["target"].astype(float).to_numpy()
            for name in ("HGB", "RF", "ET"):
                model = model_factory(name, seed)
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                rows.append({
                    "zone": str(sid),
                    "window": w,
                    "model": name,
                    "mae": float(np.mean(np.abs(truth - pred))),
                    "seed": int(seed),
                    "test_start": str(test.iloc[0]["timestamp"]),
                })
    out = pd.DataFrame(rows)
    expected = df["id"].nunique() * NUM_WINDOWS * 3
    if len(out) != expected:
        raise RuntimeError(f"incomplete seed panel: got {len(out)}, expected {expected}")
    return out


def _order(row: pd.Series, models: list[str]) -> tuple[str, ...]:
    return tuple(sorted(models, key=lambda m: (float(row[m]), m)))


def compare_seed_rankings(
    reference: pd.DataFrame,
    alternate: pd.DataFrame,
    reference_seed: int,
    alternate_seed: int,
) -> pd.DataFrame:
    ref = reference.pivot_table(index=["zone", "window"], columns="model", values="mae")
    alt = alternate.pivot_table(index=["zone", "window"], columns="model", values="mae")
    common = ref.index.intersection(alt.index)
    models = sorted(set(ref.columns).intersection(alt.columns))
    if len(models) < 2:
        raise ValueError("need at least two common models")
    rows = []
    for idx in common:
        rr, aa = ref.loc[idx], alt.loc[idx]
        ro, ao = _order(rr, models), _order(aa, models)
        ref_values = sorted(float(rr[m]) for m in models)
        rows.append({
            "zone": idx[0],
            "window": int(idx[1]),
            "reference_seed": int(reference_seed),
            "alternate_seed": int(alternate_seed),
            "reference_rank": ">".join(ro),
            "alternate_rank": ">".join(ao),
            "rank_changed": ro != ao,
            "reference_winner": ro[0],
            "alternate_winner": ao[0],
            "winner_changed": ro[0] != ao[0],
            "reference_margin": ref_values[1] - ref_values[0],
        })
    return pd.DataFrame(rows)


def summarize_seed_null(
    detail: pd.DataFrame,
    thresholds=DEFAULT_MARGIN_THRESHOLDS,
) -> pd.DataFrame:
    rows = []
    for seed, group in detail.groupby("alternate_seed", sort=True):
        for threshold in thresholds:
            threshold = float(threshold)
            eligible = group if threshold <= 0 else group[group["reference_margin"] >= threshold]
            rows.append({
                "alternate_seed": int(seed),
                "margin_threshold": threshold,
                "rank_change_rate": float(eligible["rank_changed"].mean()) if len(eligible) else np.nan,
                "winner_change_rate": float(eligible["winner_changed"].mean()) if len(eligible) else np.nan,
                "n_cells": int(len(eligible)),
            })
    return pd.DataFrame(rows)


def aggregate_seed_files(
    input_dir: Path,
    reference_seed: int = REFERENCE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(input_dir.rglob("seed_*.csv"))
    panels = {}
    for path in files:
        df = pd.read_csv(path)
        if df.empty or "seed" not in df.columns:
            continue
        seed = int(df["seed"].iloc[0])
        panels[seed] = df
    if reference_seed not in panels:
        raise RuntimeError(f"reference seed {reference_seed} not found in {input_dir}")

    comparisons = []
    for seed, panel in sorted(panels.items()):
        if seed == reference_seed:
            continue
        comparisons.append(compare_seed_rankings(panels[reference_seed], panel, reference_seed, seed))
    if not comparisons:
        raise RuntimeError("no alternate seed panels found")

    detail = pd.concat(comparisons, ignore_index=True)
    by_seed = summarize_seed_null(detail)
    quantile_rows = []
    for threshold, group in by_seed.groupby("margin_threshold", sort=True):
        quantile_rows.append({
            "margin_threshold": float(threshold),
            "alternate_seed_count": int(group["alternate_seed"].nunique()),
            "winner_null_p95": float(group["winner_change_rate"].quantile(0.95)),
            "rank_null_p95": float(group["rank_change_rate"].quantile(0.95)),
            "winner_null_max": float(group["winner_change_rate"].max()),
            "rank_null_max": float(group["rank_change_rate"].max()),
            "min_cells": int(group["n_cells"].min()),
        })
    quantiles = pd.DataFrame(quantile_rows)
    return detail, by_seed, quantiles


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--outdir", required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--input-dir", required=True)
    agg.add_argument("--outdir", required=True)
    args = p.parse_args()

    if args.command == "run":
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        scores = run_oracle_seed(args.seed)
        scores.to_csv(outdir / f"seed_{args.seed}.csv", index=False)
        (outdir / f"seed_{args.seed}_metadata.json").write_text(json.dumps({
            "seed": args.seed,
            "dataset": DATASET,
            "config": CONFIG,
            "horizon": HORIZON,
            "num_windows": NUM_WINDOWS,
        }, indent=2), encoding="utf-8")
    else:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        detail, by_seed, quantiles = aggregate_seed_files(Path(args.input_dir))
        detail.to_csv(outdir / "seed_null_detail.csv", index=False)
        by_seed.to_csv(outdir / "seed_null_by_seed_margin.csv", index=False)
        quantiles.to_csv(outdir / "seed_null_quantiles.csv", index=False)
        report = {
            "alternate_seed_count": int(by_seed["alternate_seed"].nunique()),
            "eligible_for_final_g3": int(by_seed["alternate_seed"].nunique()) >= 20,
            "thresholds": quantiles.to_dict(orient="records"),
        }
        (outdir / "seed_null_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
