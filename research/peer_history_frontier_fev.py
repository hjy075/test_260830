from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .peer_history_frontier import _pairwise_scaled, _standardize_reference


def _sequence_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(list(value))


def prepare_fev_rossmann_weekly_frame(
    frame: pd.DataFrame,
    *,
    history_weeks: int = 24,
    eval_weeks: int = 8,
) -> dict:
    required = {"timestamp", "Sales"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FEV weekly frame missing columns: {sorted(missing)}")

    total_needed = history_weeks + eval_weeks
    eligible = []
    for _, row in frame.iterrows():
        ts = pd.to_datetime(_sequence_array(row["timestamp"]))
        sales = _sequence_array(row["Sales"]).astype(float)
        if len(ts) < total_needed or len(sales) < total_needed:
            continue
        hist_sales = sales[-total_needed:-eval_weeks]
        eval_sales = sales[-eval_weeks:]
        if not np.isfinite(hist_sales).all() or not np.isfinite(eval_sales).all():
            continue
        if np.any(hist_sales <= 0) or np.any(eval_sales <= 0):
            continue
        eligible.append((row, ts, hist_sales, eval_sales))

    if not eligible:
        raise ValueError("No eligible FEV weekly rows after history/evaluation filtering")

    first_ts = eligible[0][1]
    hist_start = pd.Timestamp(first_ts[-total_needed])
    hist_end = pd.Timestamp(first_ts[-eval_weeks - 1])
    eval_start = pd.Timestamp(first_ts[-eval_weeks])
    eval_end = pd.Timestamp(first_ts[-1])

    ids = []
    perf = []
    y_eval = []
    context_rows = []

    for row, _, hist_sales, eval_sales in eligible:
        store_id = row.get("Store", row.get("id"))
        ids.append(int(float(store_id)))
        perf.append(np.log(hist_sales))
        y_eval.append(float(np.mean(np.log(eval_sales))))

        ctx = {}
        comp = pd.to_numeric(pd.Series([row.get("CompetitionDistance", np.nan)]), errors="coerce").iloc[0]
        ctx["log_competition_distance"] = np.log1p(comp) if pd.notna(comp) and comp >= 0 else np.nan

        cy = row.get("CompetitionOpenSinceYear", np.nan)
        cm = row.get("CompetitionOpenSinceMonth", np.nan)
        try:
            ctx["competition_age_months"] = max(
                0.0,
                (hist_end.year - float(cy)) * 12 + (hist_end.month - float(cm)),
            )
        except (TypeError, ValueError):
            ctx["competition_age_months"] = np.nan

        promo2 = row.get("Promo2", 0.0)
        ctx["promo2"] = float(promo2) if pd.notna(promo2) else 0.0
        py = row.get("Promo2SinceYear", np.nan)
        pw = row.get("Promo2SinceWeek", np.nan)
        try:
            promo_date = pd.Timestamp.fromisocalendar(int(float(py)), int(float(pw)), 1)
            ctx["promo2_age_weeks"] = max(0.0, (hist_end - promo_date).days / 7.0)
        except (TypeError, ValueError):
            ctx["promo2_age_weeks"] = np.nan

        for name, out_name in [
            ("Promo", "promo_rate"),
            ("Open", "open_rate"),
            ("SchoolHoliday", "school_holiday_rate"),
        ]:
            if name in frame.columns:
                seq = _sequence_array(row[name]).astype(float)
                ctx[out_name] = float(np.nanmean(seq[-total_needed:-eval_weeks]))
            else:
                ctx[out_name] = np.nan

        ctx["StoreType"] = str(row.get("StoreType", "missing"))
        ctx["Assortment"] = str(row.get("Assortment", "missing"))
        ctx["PromoInterval"] = str(row.get("PromoInterval", "missing"))
        context_rows.append(ctx)

    context = pd.DataFrame(context_rows, index=ids)
    numeric_cols = [
        "log_competition_distance",
        "competition_age_months",
        "promo2",
        "promo2_age_weeks",
        "promo_rate",
        "open_rate",
        "school_holiday_rate",
    ]
    numeric = context[numeric_cols].astype(float).copy()
    for col in numeric.columns:
        med = numeric[col].median()
        numeric[col] = numeric[col].fillna(0.0 if pd.isna(med) else med)
    cats = pd.get_dummies(
        context[["StoreType", "Assortment", "PromoInterval"]],
        drop_first=False,
        dtype=float,
    )
    context_matrix = pd.concat([numeric, cats], axis=1)
    context_z, _, _ = _standardize_reference(context_matrix.to_numpy(dtype=float))
    context_dist, _ = _pairwise_scaled(context_z, context_z)

    return {
        "ids": np.asarray(ids),
        "perf_raw": np.asarray(perf, dtype=float),
        "y_eval": np.asarray(y_eval, dtype=float),
        "context_dist": context_dist,
        "context_feature_count": int(context_matrix.shape[1]),
        "date_range": {
            "history_start": str(hist_start.date()),
            "history_end": str(hist_end.date()),
            "eval_start": str(eval_start.date()),
            "eval_end": str(eval_end.date()),
        },
    }


def prepare_fev_rossmann_weekly(
    parquet_path: str | Path,
    *,
    history_weeks: int = 24,
    eval_weeks: int = 8,
) -> dict:
    frame = pd.read_parquet(parquet_path)
    return prepare_fev_rossmann_weekly_frame(
        frame,
        history_weeks=history_weeks,
        eval_weeks=eval_weeks,
    )
