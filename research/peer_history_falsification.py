from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DELTAS = [0.05, 0.10, 0.20]
CONTAM_FRACS = [0.0, 0.25, 0.50, 0.75, 1.0]


def robust_scale_dist(d: np.ndarray) -> np.ndarray:
    mask = np.isfinite(d) & (d > 0)
    med = float(np.median(d[mask])) if mask.any() else 1.0
    return d / med if med > 0 else d


def zscore_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd, mu, sd


def topk_peers(distance_row: np.ndarray, self_idx: int, k: int) -> np.ndarray:
    d = distance_row.copy()
    d[self_idx] = np.inf
    return np.argpartition(d, k)[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--outdir", default="results/peer_history")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--history-weeks", type=int, default=24)
    ap.add_argument("--eval-weeks", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train, low_memory=False)
    store = pd.read_csv(args.store, low_memory=False)
    train["Date"] = pd.to_datetime(train["Date"])
    train = train[(train["Open"] == 1) & (train["Sales"] > 0)].copy()
    train["log_sales"] = np.log(train["Sales"].astype(float))

    max_date = train["Date"].max().normalize()
    eval_end = max_date
    eval_start = eval_end - pd.Timedelta(days=args.eval_weeks * 7 - 1)
    hist_end = eval_start - pd.Timedelta(days=1)
    hist_start = hist_end - pd.Timedelta(days=args.history_weeks * 7 - 1)

    hist = train[(train["Date"] >= hist_start) & (train["Date"] <= hist_end)].copy()
    ev = train[(train["Date"] >= eval_start) & (train["Date"] <= eval_end)].copy()

    hist["week"] = hist["Date"].dt.to_period("W-FRI")
    expected_weeks = pd.period_range(hist_start, hist_end, freq="W-FRI")
    weekly = hist.groupby(["Store", "week"], observed=True)["log_sales"].mean().unstack("week")
    weekly = weekly.reindex(columns=expected_weeks)

    eval_store = ev.groupby("Store")["log_sales"].agg(["mean", "count"]).rename(columns={"mean": "eval_log", "count": "eval_n"})
    hist_coverage = weekly.notna().sum(axis=1)
    eligible_ids = hist_coverage[hist_coverage >= max(1, args.history_weeks - 4)].index.intersection(
        eval_store[eval_store["eval_n"] >= 30].index
    )

    weekly = weekly.loc[eligible_ids].copy()
    eval_store = eval_store.loc[eligible_ids].copy()
    store = store[store["Store"].isin(eligible_ids)].copy().set_index("Store").loc[eligible_ids]

    # Impute occasional missing weeks by each store's own history mean.
    weekly = weekly.T.fillna(weekly.mean(axis=1)).T
    ids = weekly.index.to_numpy()
    n = len(ids)
    if n <= args.k:
        raise RuntimeError(f"Not enough eligible stores: {n}")

    # Performance history = weekly log-sales trajectory. Standardization is per week across stores.
    perf_raw = weekly.to_numpy(dtype=float)
    perf_z, perf_mu, perf_sd = zscore_columns(perf_raw)
    perf_dist_clean = pairwise_distances(perf_z, metric="euclidean") / np.sqrt(perf_z.shape[1])
    perf_dist_clean = robust_scale_dist(perf_dist_clean)

    # Structural/context features only. Customers are intentionally excluded because they are a contemporaneous outcome-like variable.
    context = pd.DataFrame(index=store.index)
    comp = pd.to_numeric(store["CompetitionDistance"], errors="coerce")
    comp = comp.fillna(comp.median())
    context["log_competition_distance"] = np.log1p(comp)
    context["promo2"] = pd.to_numeric(store["Promo2"], errors="coerce").fillna(0).astype(float)
    cats = pd.get_dummies(store[["StoreType", "Assortment"]].astype(str), drop_first=False, dtype=float)
    context = pd.concat([context, cats], axis=1)
    context_z, _, _ = zscore_columns(context.to_numpy(dtype=float))
    context_dist = pairwise_distances(context_z, metric="euclidean") / np.sqrt(context_z.shape[1])
    context_dist = robust_scale_dist(context_dist)

    y_eval = eval_store.loc[ids, "eval_log"].to_numpy(dtype=float)

    # Clean peer sets and baseline benchmarking error for each history weight alpha.
    clean_peer_sets: dict[float, list[np.ndarray]] = {}
    clean_benchmarks: dict[float, np.ndarray] = {}
    clean_scores: dict[float, np.ndarray] = {}
    comparability_rows = []

    for alpha in ALPHAS:
        d = (1.0 - alpha) * context_dist + alpha * perf_dist_clean
        peers_for_alpha = []
        bench = np.empty(n, dtype=float)
        for i in range(n):
            peers = topk_peers(d[i], i, args.k)
            peers_for_alpha.append(peers)
            bench[i] = y_eval[peers].mean()
        clean_peer_sets[alpha] = peers_for_alpha
        clean_benchmarks[alpha] = bench
        clean_scores[alpha] = bench - y_eval
        err = np.abs(bench - y_eval)
        comparability_rows.append({
            "alpha": alpha,
            "n_stores": n,
            "k": args.k,
            "clean_mae_log": float(err.mean()),
            "clean_median_ae_log": float(np.median(err)),
            "clean_rmse_log": float(np.sqrt(np.mean((bench - y_eval) ** 2))),
        })

    comparability = pd.DataFrame(comparability_rows)
    base_mae = float(comparability.loc[comparability["alpha"] == 0.0, "clean_mae_log"].iloc[0])
    comparability["mae_improvement_vs_context_pct"] = 100 * (base_mae - comparability["clean_mae_log"]) / base_mae
    comparability.to_csv(outdir / "clean_comparability.csv", index=False)

    # Shock recovery. Only the focal store is shocked; candidate peers remain observed.
    detail_rows = []
    for delta in DELTAS:
        shock_log = float(np.log(1.0 - delta))  # negative
        true_effect = -shock_log
        for contam_frac in CONTAM_FRACS:
            n_contam = int(round(args.history_weeks * contam_frac))
            contam_idx = np.arange(args.history_weeks - n_contam, args.history_weeks) if n_contam > 0 else np.array([], dtype=int)

            for alpha in ALPHAS:
                clean_b = clean_benchmarks[alpha]
                clean_s = clean_scores[alpha]
                clean_peers = clean_peer_sets[alpha]

                for i in range(n):
                    if alpha == 0.0 or n_contam == 0:
                        shock_peers = clean_peers[i]
                    else:
                        v = perf_z[i].copy()
                        v[contam_idx] += shock_log / perf_sd[contam_idx]
                        dp = np.sqrt(np.mean((perf_z - v) ** 2, axis=1))
                        # Same normalization as the clean performance distance matrix.
                        # perf_dist_clean was normalized by its median; recover that scale from raw pairwise distances.
                        raw_clean = pairwise_distances(perf_z[[i]], perf_z, metric="euclidean")[0] / np.sqrt(perf_z.shape[1])
                        nz = raw_clean[raw_clean > 0]
                        # Use global-ish stable scaling: median of focal clean row. This only rescales the performance component.
                        scale = float(np.median(nz)) if len(nz) else 1.0
                        dp = dp / scale
                        drow = (1.0 - alpha) * context_dist[i] + alpha * dp
                        shock_peers = topk_peers(drow, i, args.k)

                    shock_b = float(y_eval[shock_peers].mean())
                    shock_target_eval = y_eval[i] + shock_log
                    shock_score = shock_b - shock_target_eval
                    detected = shock_score - clean_s[i]
                    recovery = detected / true_effect
                    attenuation = 1.0 - recovery
                    overlap = len(set(clean_peers[i].tolist()).intersection(shock_peers.tolist())) / args.k

                    detail_rows.append({
                        "Store": int(ids[i]),
                        "alpha": alpha,
                        "delta": delta,
                        "contam_frac": contam_frac,
                        "n_contam_weeks": n_contam,
                        "true_log_effect": true_effect,
                        "clean_benchmark_log": float(clean_b[i]),
                        "shock_benchmark_log": shock_b,
                        "benchmark_shift_log": shock_b - float(clean_b[i]),
                        "detected_log_effect": detected,
                        "recovery": recovery,
                        "attenuation": attenuation,
                        "peer_overlap": overlap,
                    })

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(outdir / "shock_recovery_detail.csv", index=False)

    summary = (
        detail.groupby(["alpha", "delta", "contam_frac"], as_index=False)
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
            share_recovery_below_80pct=("recovery", lambda x: np.mean(x < 0.8)),
            share_recovery_below_50pct=("recovery", lambda x: np.mean(x < 0.5)),
        )
    )
    summary.to_csv(outdir / "shock_recovery_summary.csv", index=False)

    best_alpha = float(comparability.sort_values("clean_mae_log").iloc[0]["alpha"])
    best_clean_improvement = float(comparability.loc[comparability["alpha"] == best_alpha, "mae_improvement_vs_context_pct"].iloc[0])

    # Falsification-oriented verdict. Require BOTH benefit under clean history and deterioration under contamination.
    best_slice = summary[(summary["alpha"] == best_alpha) & (summary["delta"] == 0.10)].sort_values("contam_frac")
    rec0 = float(best_slice.loc[best_slice["contam_frac"] == 0.0, "mean_recovery"].iloc[0])
    rec100 = float(best_slice.loc[best_slice["contam_frac"] == 1.0, "mean_recovery"].iloc[0])
    monotonic = bool(np.all(np.diff(best_slice["mean_recovery"].to_numpy()) <= 0.01))
    material_benefit = best_clean_improvement >= 5.0
    material_harm = (rec0 - rec100) >= 0.10
    survive = bool(material_benefit and material_harm and monotonic and best_alpha > 0)

    verdict = {
        "n_eligible_stores": int(n),
        "date_range": {
            "history_start": str(hist_start.date()),
            "history_end": str(hist_end.date()),
            "eval_start": str(eval_start.date()),
            "eval_end": str(eval_end.date()),
        },
        "k": args.k,
        "best_alpha_clean": best_alpha,
        "best_clean_mae_improvement_vs_context_pct": best_clean_improvement,
        "delta_used_for_primary_falsification": 0.10,
        "mean_recovery_clean_history": rec0,
        "mean_recovery_full_contamination": rec100,
        "recovery_drop": rec0 - rec100,
        "recovery_nonincreasing_with_contamination": monotonic,
        "material_comparability_benefit_ge_5pct": material_benefit,
        "material_recovery_harm_ge_10pp": material_harm,
        "verdict": "PROVISIONAL_SURVIVE" if survive else "KILL_OR_REDESIGN",
    }
    (outdir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Compact Markdown for inspection through GitHub text tools.
    lines = []
    lines.append("# Peer-History Falsification Experiment")
    lines.append("")
    lines.append(f"Eligible stores: **{n}**; K={args.k}")
    lines.append(f"History: {hist_start.date()} to {hist_end.date()} ({args.history_weeks} weeks)")
    lines.append(f"Evaluation: {eval_start.date()} to {eval_end.date()} ({args.eval_weeks} weeks)")
    lines.append("")
    lines.append("## Clean comparability")
    lines.append("")
    lines.append(comparability.round(4).to_markdown(index=False))
    lines.append("")
    lines.append("## Primary falsification (10% persistent shock)")
    lines.append("")
    primary = summary[summary["delta"] == 0.10][["alpha", "contam_frac", "mean_recovery", "median_recovery", "mean_peer_overlap", "mean_benchmark_shift_log", "share_recovery_below_80pct"]]
    lines.append(primary.round(4).to_markdown(index=False))
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(verdict, indent=2))
    lines.append("```")
    (outdir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
