import json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path("peer_minexp")
train_path = ROOT / "train.csv"
store_path = ROOT / "store.csv"

train = pd.read_csv(train_path, low_memory=False)
store = pd.read_csv(store_path, low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])
train["StateHoliday"] = train["StateHoliday"].astype(str)

def store_perf(df, start, end):
    x = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    x = x[(x["Open"] == 1) & (x["Promo"] == 0) & (x["StateHoliday"].isin(["0", "0.0"])) & (x["Sales"] > 0)]
    g = x.groupby(["Store", "DayOfWeek"], observed=True)["Sales"].median().reset_index()
    perf = g.groupby("Store", observed=True)["Sales"].mean()
    n_days = x.groupby("Store", observed=True).size()
    return pd.DataFrame({"perf": perf, "n_days": n_days})

p0 = store_perf(train, "2014-01-01", "2014-12-31").rename(columns={"perf":"perf0","n_days":"n0"})
p1 = store_perf(train, "2015-01-01", "2015-07-31").rename(columns={"perf":"perf1","n_days":"n1"})
meta = store.set_index("Store").join(p0, how="inner").join(p1, how="inner").reset_index()
meta = meta[(meta["n0"] >= 60) & (meta["n1"] >= 30)].copy().reset_index(drop=True)

# Static exogenous metadata only; Customers and Sales never enter peer construction.
meta["log_comp_dist"] = np.log1p(meta["CompetitionDistance"])
meta["log_comp_dist"] = meta["log_comp_dist"].fillna(meta["log_comp_dist"].median())
meta["comp_missing"] = meta["CompetitionDistance"].isna().astype(int)
meta["comp_age"] = 2014 - meta["CompetitionOpenSinceYear"]
meta.loc[(meta["comp_age"] < 0) | (meta["comp_age"] > 30), "comp_age"] = np.nan
meta["comp_age_missing"] = meta["comp_age"].isna().astype(int)
meta["comp_age"] = meta["comp_age"].fillna(meta["comp_age"].median())
meta["PromoInterval"] = meta["PromoInterval"].fillna("None").astype(str)
meta["StoreType"] = meta["StoreType"].astype(str)
meta["Assortment"] = meta["Assortment"].astype(str)
meta["Promo2"] = meta["Promo2"].fillna(0).astype(int).astype(str)

feature_sets = {
    "F1_format_compdist": {"cat": ["StoreType","Assortment"], "num": ["log_comp_dist","comp_missing"]},
    "F2_plus_compage": {"cat": ["StoreType","Assortment"], "num": ["log_comp_dist","comp_missing","comp_age","comp_age_missing"]},
    "F3_plus_promo": {"cat": ["StoreType","Assortment","Promo2","PromoInterval"], "num": ["log_comp_dist","comp_missing","comp_age","comp_age_missing"]},
}

def gower_matrix(df, cat_cols, num_cols):
    n = len(df)
    D = np.zeros((n,n), dtype=np.float32)
    denom = 0
    for c in cat_cols:
        a = df[c].astype(str).to_numpy()
        D += (a[:,None] != a[None,:]).astype(np.float32)
        denom += 1
    for c in num_cols:
        a = df[c].astype(float).to_numpy()
        lo, hi = np.nanmin(a), np.nanmax(a)
        rng = hi - lo
        if not np.isfinite(rng) or rng <= 0:
            continue
        z = (a - lo) / rng
        D += np.abs(z[:,None] - z[None,:]).astype(np.float32)
        denom += 1
    D /= max(denom,1)
    np.fill_diagonal(D, np.inf)
    return D

def euclid_matrix(df, cat_cols, num_cols):
    X_parts = []
    if cat_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_parts.append(enc.fit_transform(df[cat_cols].astype(str)))
    if num_cols:
        X_parts.append(StandardScaler().fit_transform(df[num_cols].astype(float)))
    X = np.hstack(X_parts)
    D = pairwise_distances(X, metric="euclidean").astype(np.float32)
    np.fill_diagonal(D, np.inf)
    return D

specs, peer_sets, gap0, gap1 = [], {}, {}, {}
ks = [10,20,40]
for fname, cols in feature_sets.items():
    for metric in ["gower","euclid"]:
        D = gower_matrix(meta, cols["cat"], cols["num"]) if metric=="gower" else euclid_matrix(meta, cols["cat"], cols["num"])
        order = np.argsort(D, axis=1)
        for k in ks:
            sid = f"{fname}__{metric}__k{k}"
            peers = order[:,:k]
            peer_sets[sid] = peers
            exp0 = np.median(meta["perf0"].to_numpy()[peers], axis=1)
            exp1 = np.median(meta["perf1"].to_numpy()[peers], axis=1)
            gap0[sid] = meta["perf0"].to_numpy()/exp0 - 1.0
            gap1[sid] = meta["perf1"].to_numpy()/exp1 - 1.0
            specs.append(sid)

G0 = pd.DataFrame(gap0)
G1 = pd.DataFrame(gap1)
n, M = len(meta), len(specs)

pair_js = []
for a,b in itertools.combinations(specs,2):
    pa, pb = peer_sets[a], peer_sets[b]
    for i in range(n):
        A, B = set(pa[i].tolist()), set(pb[i].tolist())
        pair_js.append(len(A & B) / len(A | B))

gap_iqr = (G0.quantile(.75,axis=1) - G0.quantile(.25,axis=1)) * 100
gap_range = (G0.max(axis=1) - G0.min(axis=1)) * 100

U_abs0 = (G0 < -0.15).mean(axis=1)
U_abs1 = (G1 < -0.15).mean(axis=1)
bottom0, bottom1 = pd.DataFrame(index=G0.index), pd.DataFrame(index=G1.index)
for s in specs:
    bottom0[s] = G0[s] <= G0[s].quantile(.20)
    bottom1[s] = G1[s] <= G1[s].quantile(.20)
U_q0, U_q1 = bottom0.mean(axis=1), bottom1.mean(axis=1)

def verdict_summary(U):
    return {
        "any_flip_pct": float(((U > 0) & (U < 1)).mean()*100),
        "ambiguous_20_80_pct": float(((U >= .2) & (U <= .8)).mean()*100),
        "robust_under_ge80_pct": float((U >= .8).mean()*100),
        "robust_nonunder_le20_pct": float((U <= .2).mean()*100),
    }

def safe_rate(mask, outcome):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum()==0: return None, 0
    return float(np.asarray(outcome)[mask].mean()), int(mask.sum())

def persistence_block(U0, future_major, default0):
    r_default, n_default = safe_rate(default0==1, future_major)
    r_robust, n_robust = safe_rate(U0>=.8, future_major)
    r_frag, n_frag = safe_rate((default0==1) & (U0>.2) & (U0<.8), future_major)
    return {"future_major_rate_default_under":r_default,"n_default_under":n_default,
            "future_major_rate_robust_under":r_robust,"n_robust_under":n_robust,
            "future_major_rate_fragile_default_under":r_frag,"n_fragile_default_under":n_frag}

def cv_auc_addition(U0, future_major, default_spec):
    y = np.asarray(future_major, dtype=int)
    xgap = (-G0[default_spec].to_numpy()).reshape(-1,1)
    xboth = np.column_stack([-G0[default_spec].to_numpy(), np.asarray(U0)])
    if len(np.unique(y)) < 2: return {"auc_gap_only":None,"auc_gap_plus_consensus":None,"delta_auc":None}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    a1 = cross_val_score(model, xgap, y, scoring="roc_auc", cv=cv).mean()
    a2 = cross_val_score(model, xboth, y, scoring="roc_auc", cv=cv).mean()
    return {"auc_gap_only":float(a1),"auc_gap_plus_consensus":float(a2),"delta_auc":float(a2-a1)}

default_spec = "F2_plus_compage__gower__k20"
default_abs0 = (G0[default_spec] < -0.15).astype(int)
default_q0 = bottom0[default_spec].astype(int)
future_abs_major = (U_abs1 >= .5).astype(int)
future_q_major = (U_q1 >= .5).astype(int)

summary = {
    "n_stores":int(n),"n_specs":int(M),"period0":"2014","period1":"2015-01-01..2015-07-31",
    "peer_jaccard":{"median":float(np.median(pair_js)),"p10":float(np.quantile(pair_js,.10)),"p90":float(np.quantile(pair_js,.90))},
    "gap_dispersion_pp":{"median_iqr":float(gap_iqr.median()),"p90_iqr":float(gap_iqr.quantile(.90)),"median_range":float(gap_range.median()),"p90_range":float(gap_range.quantile(.90))},
    "absolute_minus15":{**verdict_summary(U_abs0),"persistence":persistence_block(U_abs0,future_abs_major,default_abs0),
        "spearman_consensus_2014_2015":float(pd.Series(U_abs0).corr(pd.Series(U_abs1),method="spearman")),
        "cv_auc":cv_auc_addition(U_abs0,future_abs_major,default_spec)},
    "bottom_quintile":{**verdict_summary(U_q0),"persistence":persistence_block(U_q0,future_q_major,default_q0),
        "spearman_consensus_2014_2015":float(pd.Series(U_q0).corr(pd.Series(U_q1),method="spearman")),
        "cv_auc":cv_auc_addition(U_q0,future_q_major,default_spec)},
}
for thr in [-.10,-.20]:
    u0, u1 = (G0 < thr).mean(axis=1), (G1 < thr).mean(axis=1)
    summary[f"absolute_{int(abs(thr)*100)}pct"] = {**verdict_summary(u0),"spearman_consensus_2014_2015":float(pd.Series(u0).corr(pd.Series(u1),method="spearman"))}

diag = meta[["Store","StoreType","Assortment","perf0","perf1"]].copy()
diag["default_gap0"], diag["default_gap1"] = G0[default_spec], G1[default_spec]
diag["gap_iqr_pp"], diag["gap_range_pp"] = gap_iqr, gap_range
diag["U_abs0"], diag["U_abs1"], diag["U_q0"], diag["U_q1"] = U_abs0, U_abs1, U_q0, U_q1
diag.to_csv(ROOT/"store_diagnostics.csv", index=False)
G0.to_csv(ROOT/"gaps_2014.csv", index=False)
with open(ROOT/"summary.json","w") as f: json.dump(summary,f,indent=2)
print("=== PEER_UNCERTAINTY_MINEXP_SUMMARY ===")
print(json.dumps(summary,indent=2))
print("=== END_SUMMARY ===")
