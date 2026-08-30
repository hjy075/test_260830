from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
LEVEL_SHARES = (0.0, 0.25, 0.50, 0.75, 1.0)
DELTAS = (0.05, 0.10, 0.15, 0.20)
CONTAM_FRACS = (0.0, 0.25, 0.50, 0.75, 1.0)


def center_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - np.nanmean(x, axis=1, keepdims=True)


def _standardize_reference(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=float)
    mu = np.nanmean(reference, axis=0)
    sd = np.nanstd(reference, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (reference - mu) / sd, mu, sd


def _standardize_target(target: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(target, dtype=float) - mu) / sd


def _distance_scale(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    mask = np.isfinite(d) & (d > 0)
    return float(np.median(d[mask])) if mask.any() else 1.0


def _pairwise_scaled(target_z: np.ndarray, donor_z: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    denom = np.sqrt(max(1, donor_z.shape[1]))
    d = pairwise_distances(target_z, donor_z, metric="euclidean") / denom
    if scale is None:
        scale = _distance_scale(d)
    return d / scale, float(scale)


def compose_distance(
    context_dist: np.ndarray,
    level_dist: np.ndarray,
    shape_dist: np.ndarray,
    *,
    alpha: float,
    level_share: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= level_share <= 1.0:
        raise ValueError("level_share must be in [0, 1]")
    history = level_share * level_dist + (1.0 - level_share) * shape_dist
    return (1.0 - alpha) * context_dist + alpha * history


def make_config_grid(
    alphas: Iterable[float] = ALPHAS,
    level_shares: Iterable[float] = LEVEL_SHARES,
    *,
    include_raw_reference: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = [
        {
            "config_id": "context_only",
            "strategy": "context_only",
            "alpha": 0.0,
            "level_share": np.nan,
        }
    ]
    for alpha in alphas:
        alpha = float(alpha)
        if alpha <= 0:
            continue
        for level_share in level_shares:
            level_share = float(level_share)
            if level_share == 1.0:
                strategy = "level_only"
            elif level_share == 0.0:
                strategy = "shape_only"
            else:
                strategy = "level_shape_hybrid"
            rows.append(
                {
                    "config_id": f"decomp_a{alpha:.2f}_l{level_share:.2f}",
                    "strategy": strategy,
                    "alpha": alpha,
                    "level_share": level_share,
                }
            )
        if include_raw_reference:
            rows.append(
                {
                    "config_id": f"raw_a{alpha:.2f}",
                    "strategy": "raw_trajectory_reference",
                    "alpha": alpha,
                    "level_share": np.nan,
                }
            )
    return pd.DataFrame(rows).drop_duplicates("config_id").reset_index(drop=True)


def diagnostic_retention_auc(contam_frac: np.ndarray, recovery: np.ndarray) -> float:
    q = np.asarray(contam_frac, dtype=float)
    r = np.asarray(recovery, dtype=float)
    order = np.argsort(q)
    q = q[order]
    r = r[order]
    if len(q) < 2 or float(q[-1] - q[0]) <= 0:
        return float(np.nanmean(r))
    return float(np.trapezoid(r, q) / (q[-1] - q[0]))


def diagnostic_error_auc(contam_frac: np.ndarray, recovery: np.ndarray) -> float:
    q = np.asarray(contam_frac, dtype=float)
    r = np.asarray(recovery, dtype=float)
    order = np.argsort(q)
    q = q[order]
    err = np.abs(r[order] - 1.0)
    if len(q) < 2 or float(q[-1] - q[0]) <= 0:
        return float(np.nanmean(err))
    return float(np.trapezoid(err, q) / (q[-1] - q[0]))


def pareto_frontier(
    frame: pd.DataFrame,
    x: str = "comparability_gain_pct",
    y: str = "diagnostic_error_auc",
) -> pd.DataFrame:
    """Keep configs that are not dominated when x is maximized and y minimized."""
    if frame.empty:
        return frame.copy()
    vals = frame[[x, y]].to_numpy(dtype=float)
    keep = np.ones(len(frame), dtype=bool)
    for i, point in enumerate(vals):
        dominates_i = (
            (vals[:, 0] >= point[0])
            & (vals[:, 1] <= point[1])
            & ((vals[:, 0] > point[0]) | (vals[:, 1] < point[1]))
        )
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return frame.loc[keep].sort_values([x, y], ascending=[True, True]).reset_index(drop=True)


def _topk_peers(distance_row: np.ndarray, self_idx: int, k: int) -> np.ndarray:
    d = np.asarray(distance_row, dtype=float).copy()
    d[self_idx] = np.inf
    if k >= len(d):
        raise ValueError("k must be smaller than number of stores")
    return np.argpartition(d, k)[:k]


def _history_geometry(perf_raw: np.ndarray) -> dict:
    level = np.nanmean(perf_raw, axis=1, keepdims=True)
    shape = center_rows(perf_raw)
    raw_z, raw_mu, raw_sd = _standardize_reference(perf_raw)
    level_z, level_mu, level_sd = _standardize_reference(level)
    shape_z, shape_mu, shape_sd = _standardize_reference(shape)

    raw_dist, raw_scale = _pairwise_scaled(raw_z, raw_z)
    level_dist, level_scale = _pairwise_scaled(level_z, level_z)
    shape_dist, shape_scale = _pairwise_scaled(shape_z, shape_z)
    return {
        "raw_dist": raw_dist,
        "level_dist": level_dist,
        "shape_dist": shape_dist,
        "raw_scale": raw_scale,
        "level_scale": level_scale,
        "shape_scale": shape_scale,
        "raw_stats": (raw_mu, raw_sd),
        "level_stats": (level_mu, level_sd),
        "shape_stats": (shape_mu, shape_sd),
    }


def _shocked_history_distances(
    perf_raw: np.ndarray,
    *,
    shock_log: float,
    contam_frac: float,
    geometry: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_weeks = perf_raw.shape[1]
    n_contam = int(round(n_weeks * float(contam_frac)))
    target = perf_raw.copy()
    if n_contam > 0:
        target[:, n_weeks - n_contam :] += shock_log

    donor_level = np.nanmean(perf_raw, axis=1, keepdims=True)
    donor_shape = center_rows(perf_raw)
    target_level = np.nanmean(target, axis=1, keepdims=True)
    target_shape = center_rows(target)

    raw_mu, raw_sd = geometry["raw_stats"]
    level_mu, level_sd = geometry["level_stats"]
    shape_mu, shape_sd = geometry["shape_stats"]

    donor_raw_z = _standardize_target(perf_raw, raw_mu, raw_sd)
    donor_level_z = _standardize_target(donor_level, level_mu, level_sd)
    donor_shape_z = _standardize_target(donor_shape, shape_mu, shape_sd)
    target_raw_z = _standardize_target(target, raw_mu, raw_sd)
    target_level_z = _standardize_target(target_level, level_mu, level_sd)
    target_shape_z = _standardize_target(target_shape, shape_mu, shape_sd)

    raw_dist, _ = _pairwise_scaled(target_raw_z, donor_raw_z, float(geometry["raw_scale"]))
    level_dist, _ = _pairwise_scaled(target_level_z, donor_level_z, float(geometry["level_scale"]))
    shape_dist, _ = _pairwise_scaled(target_shape_z, donor_shape_z, float(geometry["shape_scale"]))
    return raw_dist, level_dist, shape_dist


def _months_since(year: pd.Series, month: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce")
    m = pd.to_numeric(month, errors="coerce")
    value = (as_of.year - y) * 12 + (as_of.month - m)
    return value.where((value >= 0) & y.notna() & m.notna())


def _weeks_since_iso(year: pd.Series, week: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce")
    w = pd.to_numeric(week, errors="coerce")
    out = pd.Series(np.nan, index=year.index, dtype=float)
    mask = y.notna() & w.notna()
    for idx in year.index[mask]:
        try:
            d = pd.Timestamp.fromisocalendar(int(y.loc[idx]), int(w.loc[idx]), 1)
            out.loc[idx] = max(0.0, (as_of - d).days / 7.0)
        except ValueError:
            pass
    return out


def prepare_rossmann(
    train_path: str | Path,
    store_path: str | Path,
    *,
    history_weeks: int = 24,
    eval_weeks: int = 8,
) -> dict:
    train = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path, low_memory=False)
    train["Date"] = pd.to_datetime(train["Date"])

    max_date = train["Date"].max().normalize()
    eval_end = max_date
    eval_start = eval_end - pd.Timedelta(days=eval_weeks * 7 - 1)
    hist_end = eval_start - pd.Timedelta(days=1)
    hist_start = hist_end - pd.Timedelta(days=history_weeks * 7 - 1)

    hist_all = train[(train["Date"] >= hist_start) & (train["Date"] <= hist_end)].copy()
    hist_sales = hist_all[(hist_all["Open"] == 1) & (hist_all["Sales"] > 0)].copy()
    ev = train[
        (train["Date"] >= eval_start)
        & (train["Date"] <= eval_end)
        & (train["Open"] == 1)
        & (train["Sales"] > 0)
    ].copy()
    hist_sales["log_sales"] = np.log(hist_sales["Sales"].astype(float))
    ev["log_sales"] = np.log(ev["Sales"].astype(float))

    hist_sales["week"] = hist_sales["Date"].dt.to_period("W-FRI")
    expected_weeks = pd.period_range(hist_start, hist_end, freq="W-FRI")
    weekly = hist_sales.groupby(["Store", "week"], observed=True)["log_sales"].mean().unstack("week")
    weekly = weekly.reindex(columns=expected_weeks)

    eval_store = ev.groupby("Store")["log_sales"].agg(["mean", "count"]).rename(
        columns={"mean": "eval_log", "count": "eval_n"}
    )
    hist_coverage = weekly.notna().sum(axis=1)
    min_hist_weeks = max(1, len(expected_weeks) - 4)
    eligible = hist_coverage[hist_coverage >= min_hist_weeks].index.intersection(
        eval_store[eval_store["eval_n"] >= 30].index
    )
    weekly = weekly.loc[eligible].copy()
    eval_store = eval_store.loc[eligible].copy()
    weekly = weekly.T.fillna(weekly.mean(axis=1)).T

    ids = weekly.index.to_numpy()
    store = store[store["Store"].isin(ids)].copy().set_index("Store").reindex(ids)

    context = pd.DataFrame(index=store.index)
    comp = pd.to_numeric(store["CompetitionDistance"], errors="coerce")
    context["log_competition_distance"] = np.log1p(comp.fillna(comp.median()))
    context["competition_age_months"] = _months_since(
        store.get("CompetitionOpenSinceYear", pd.Series(index=store.index, dtype=float)),
        store.get("CompetitionOpenSinceMonth", pd.Series(index=store.index, dtype=float)),
        hist_end,
    )
    promo2 = store["Promo2"] if "Promo2" in store.columns else pd.Series(0.0, index=store.index)
    context["promo2"] = pd.to_numeric(promo2, errors="coerce").fillna(0).astype(float)
    context["promo2_age_weeks"] = _weeks_since_iso(
        store.get("Promo2SinceYear", pd.Series(index=store.index, dtype=float)),
        store.get("Promo2SinceWeek", pd.Series(index=store.index, dtype=float)),
        hist_end,
    )

    hist_ops = hist_all.groupby("Store").agg(
        promo_rate=("Promo", "mean"),
        open_rate=("Open", "mean"),
        school_holiday_rate=("SchoolHoliday", "mean"),
    )
    context = context.join(hist_ops, how="left")

    cat_cols = [c for c in ["StoreType", "Assortment", "PromoInterval"] if c in store.columns]
    if cat_cols:
        context = pd.concat(
            [context, pd.get_dummies(store[cat_cols].astype(str), drop_first=False, dtype=float)],
            axis=1,
        )
    for col in context.columns:
        if context[col].isna().any():
            med = context[col].median()
            context[col] = context[col].fillna(0.0 if pd.isna(med) else med)

    context_z, _, _ = _standardize_reference(context.to_numpy(dtype=float))
    context_dist, _ = _pairwise_scaled(context_z, context_z)

    return {
        "ids": ids,
        "perf_raw": weekly.to_numpy(dtype=float),
        "y_eval": eval_store.loc[ids, "eval_log"].to_numpy(dtype=float),
        "context_dist": context_dist,
        "context_feature_count": int(context.shape[1]),
        "date_range": {
            "history_start": str(hist_start.date()),
            "history_end": str(hist_end.date()),
            "eval_start": str(eval_start.date()),
            "eval_end": str(eval_end.date()),
        },
    }


def run_frontier_experiment(
    data: dict,
    *,
    k: int = 15,
    deltas: Iterable[float] = DELTAS,
    contam_fracs: Iterable[float] = CONTAM_FRACS,
    configs: pd.DataFrame | None = None,
) -> dict:
    ids = np.asarray(data["ids"])
    perf_raw = np.asarray(data["perf_raw"], dtype=float)
    y_eval = np.asarray(data["y_eval"], dtype=float)
    context_dist = np.asarray(data["context_dist"], dtype=float)
    n = len(ids)
    if n <= k:
        raise ValueError(f"Not enough stores ({n}) for k={k}")
    configs = make_config_grid() if configs is None else configs.copy()
    geom = _history_geometry(perf_raw)

    clean_rows: list[dict] = []
    clean_peers: dict[str, list[np.ndarray]] = {}
    clean_bench: dict[str, np.ndarray] = {}
    clean_score: dict[str, np.ndarray] = {}

    for cfg in configs.to_dict("records"):
        cid = cfg["config_id"]
        strategy = cfg["strategy"]
        alpha = float(cfg["alpha"])
        if strategy == "context_only":
            d = context_dist
        elif strategy == "raw_trajectory_reference":
            d = (1.0 - alpha) * context_dist + alpha * np.asarray(geom["raw_dist"])
        else:
            d = compose_distance(
                context_dist,
                np.asarray(geom["level_dist"]),
                np.asarray(geom["shape_dist"]),
                alpha=alpha,
                level_share=float(cfg["level_share"]),
            )
        peers_list: list[np.ndarray] = []
        bench = np.empty(n, dtype=float)
        for i in range(n):
            peers = _topk_peers(d[i], i, k)
            peers_list.append(peers)
            bench[i] = y_eval[peers].mean()
        clean_peers[cid] = peers_list
        clean_bench[cid] = bench
        clean_score[cid] = bench - y_eval
        ae = np.abs(bench - y_eval)
        clean_rows.append(
            {
                **cfg,
                "n_stores": n,
                "k": k,
                "clean_mae_log": float(ae.mean()),
                "clean_median_ae_log": float(np.median(ae)),
                "clean_rmse_log": float(np.sqrt(np.mean((bench - y_eval) ** 2))),
            }
        )

    clean = pd.DataFrame(clean_rows)
    base_mae = float(clean.loc[clean["config_id"] == "context_only", "clean_mae_log"].iloc[0])
    clean["comparability_gain_pct"] = 100.0 * (base_mae - clean["clean_mae_log"]) / base_mae

    detail_rows: list[dict] = []
    for delta in deltas:
        delta = float(delta)
        shock_log = float(np.log1p(-delta))
        true_effect = -shock_log
        for q in contam_fracs:
            q = float(q)
            raw_shock, level_shock, shape_shock = _shocked_history_distances(
                perf_raw,
                shock_log=shock_log,
                contam_frac=q,
                geometry=geom,
            )
            n_contam = int(round(perf_raw.shape[1] * q))
            for cfg in configs.to_dict("records"):
                cid = cfg["config_id"]
                strategy = cfg["strategy"]
                alpha = float(cfg["alpha"])
                if strategy == "context_only":
                    dshock = context_dist
                elif strategy == "raw_trajectory_reference":
                    dshock = (1.0 - alpha) * context_dist + alpha * raw_shock
                else:
                    dshock = compose_distance(
                        context_dist,
                        level_shock,
                        shape_shock,
                        alpha=alpha,
                        level_share=float(cfg["level_share"]),
                    )
                cbench = clean_bench[cid]
                cscore = clean_score[cid]
                cpeers = clean_peers[cid]
                for i in range(n):
                    peers = _topk_peers(dshock[i], i, k)
                    sbench = float(y_eval[peers].mean())
                    sy = y_eval[i] + shock_log
                    sscore = sbench - sy
                    detected = sscore - cscore[i]
                    recovery = detected / true_effect
                    overlap = len(set(cpeers[i].tolist()).intersection(peers.tolist())) / k
                    detail_rows.append(
                        {
                            "Store": int(ids[i]),
                            **cfg,
                            "delta": delta,
                            "contam_frac": q,
                            "n_contam_weeks": n_contam,
                            "true_log_effect": true_effect,
                            "clean_benchmark_log": float(cbench[i]),
                            "shock_benchmark_log": sbench,
                            "benchmark_shift_log": sbench - float(cbench[i]),
                            "detected_log_effect": detected,
                            "recovery": recovery,
                            "attenuation": 1.0 - recovery,
                            "peer_overlap": overlap,
                        }
                    )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(
            ["config_id", "strategy", "alpha", "level_share", "delta", "contam_frac"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_stores=("Store", "size"),
            mean_recovery=("recovery", "mean"),
            median_recovery=("recovery", "median"),
            p10_recovery=("recovery", lambda x: np.quantile(x, 0.10)),
            p90_recovery=("recovery", lambda x: np.quantile(x, 0.90)),
            mean_attenuation=("attenuation", "mean"),
            median_attenuation=("attenuation", "median"),
            mean_peer_overlap=("peer_overlap", "mean"),
            mean_benchmark_shift_log=("benchmark_shift_log", "mean"),
            share_recovery_below_80pct=("recovery", lambda x: np.mean(x < 0.80)),
            share_recovery_below_50pct=("recovery", lambda x: np.mean(x < 0.50)),
        )
    )

    primary = summary[np.isclose(summary["delta"], 0.10)].copy()
    metric_rows: list[dict] = []
    clean_idx = clean.set_index("config_id")
    for cid, grp in primary.groupby("config_id", dropna=False):
        grp = grp.sort_values("contam_frac")
        row = clean_idx.loc[cid].to_dict()
        q = grp["contam_frac"].to_numpy(dtype=float)
        r = grp["mean_recovery"].to_numpy(dtype=float)
        row.update(
            {
                "config_id": cid,
                "diagnostic_retention_auc": diagnostic_retention_auc(q, r),
                "diagnostic_error_auc": diagnostic_error_auc(q, r),
                "recovery_q0": float(grp.iloc[np.argmin(np.abs(q - 0.0))]["mean_recovery"]),
                "recovery_q50": float(grp.iloc[np.argmin(np.abs(q - 0.5))]["mean_recovery"]),
                "recovery_q100": float(grp.iloc[np.argmin(np.abs(q - 1.0))]["mean_recovery"]),
                "peer_overlap_q100": float(grp.iloc[np.argmin(np.abs(q - 1.0))]["mean_peer_overlap"]),
            }
        )
        if row["strategy"] == "raw_trajectory_reference" and np.isclose(float(row["alpha"]), 1.0):
            row["one_minus_q_mae"] = float(np.mean(np.abs(r - (1.0 - q))))
        else:
            row["one_minus_q_mae"] = np.nan
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    frontier = pareto_frontier(metrics)

    useful = metrics[metrics["comparability_gain_pct"] >= 5.0]
    comparability_has_value = bool(metrics["comparability_gain_pct"].max() >= 5.0)
    diagnostic_harm_present = bool(
        (not useful.empty)
        and (
            (np.abs(1.0 - useful["recovery_q50"]) >= 0.10).any()
            or (useful["diagnostic_error_auc"] >= 0.10).any()
        )
    )
    nontrivial_frontier = bool(
        len(frontier) >= 2
        and (frontier["comparability_gain_pct"].max() - frontier["comparability_gain_pct"].min() >= 2.0)
        and (frontier["diagnostic_error_auc"].max() - frontier["diagnostic_error_auc"].min() >= 0.02)
    )
    screening_verdict = (
        "FRONTIER_SURVIVE"
        if comparability_has_value and diagnostic_harm_present and nontrivial_frontier
        else "KILL_OR_REDESIGN"
    )
    verdict = {
        "n_eligible_stores": int(n),
        "k": int(k),
        "context_feature_count": int(data.get("context_feature_count", 0)),
        "primary_delta": 0.10,
        "max_comparability_gain_pct": float(metrics["comparability_gain_pct"].max()),
        "pareto_config_count": int(len(frontier)),
        "comparability_has_value_ge_5pct": comparability_has_value,
        "diagnostic_harm_present_ge_10pp": diagnostic_harm_present,
        "nontrivial_pareto_frontier": nontrivial_frontier,
        "screening_verdict": screening_verdict,
    }
    return {
        "clean": clean,
        "detail": detail,
        "summary": summary,
        "metrics": metrics,
        "frontier": frontier,
        "verdict": verdict,
    }


def write_results(result: dict, outdir: str | Path, metadata: dict | None = None) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result["clean"].to_csv(out / "clean_comparability.csv", index=False)
    result["detail"].to_csv(out / "shock_recovery_detail.csv", index=False)
    result["summary"].to_csv(out / "shock_recovery_summary.csv", index=False)
    result["metrics"].to_csv(out / "frontier_metrics.csv", index=False)
    result["frontier"].to_csv(out / "pareto_frontier.csv", index=False)
    payload = dict(result["verdict"])
    if metadata:
        payload["metadata"] = metadata
    (out / "verdict.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Peer-History Comparability–Diagnostic Frontier",
        "",
        "## Screening verdict",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        "## Pareto frontier (10% shock)",
        "",
        result["frontier"][
            [
                "config_id",
                "strategy",
                "alpha",
                "level_share",
                "comparability_gain_pct",
                "diagnostic_error_auc",
                "diagnostic_retention_auc",
                "recovery_q50",
                "recovery_q100",
            ]
        ].round(4).to_markdown(index=False),
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--outdir", default="peer_minexp/history_frontier_v2")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--history-weeks", type=int, default=24)
    ap.add_argument("--eval-weeks", type=int, default=8)
    args = ap.parse_args()

    data = prepare_rossmann(
        args.train,
        args.store,
        history_weeks=args.history_weeks,
        eval_weeks=args.eval_weeks,
    )
    result = run_frontier_experiment(data, k=args.k)
    metadata = {
        "history_weeks": args.history_weeks,
        "eval_weeks": args.eval_weeks,
        "date_range": data["date_range"],
    }
    write_results(result, args.outdir, metadata)

    print("=== PEER_HISTORY_FRONTIER_V2 ===")
    print(result["frontier"].round(5).to_string(index=False))
    print("VERDICT")
    print(json.dumps({**result["verdict"], **metadata}, indent=2))
    print("=== END_PEER_HISTORY_FRONTIER_V2 ===")


if __name__ == "__main__":
    main()
