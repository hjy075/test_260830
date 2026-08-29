from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research.topic_lock import gdex_operational_gfs as base
from research.topic_lock import gdex_sensitivity as sens
from research.topic_lock.gfc17_seed_null import (
    HORIZON,
    _load_fev_long,
    _supervised,
    exact_cut_indices,
    model_factory,
    safe_feature_columns,
)


def evaluate_analysis_only(windows, outdir: Path, seed: int = 20260829):
    df = _load_fev_long()
    rows = []
    cache = outdir / 'analysis_cache'

    first = next(iter(df.groupby('id', sort=True)))[1].sort_values('timestamp').reset_index(drop=True)
    cuts0 = exact_cut_indices(len(first))
    analysis_by_window = {}
    for window in windows:
        forecast_start = pd.Timestamp(first.iloc[cuts0[window - 1]]['timestamp'])
        analysis_by_window[window] = sens.acquire_analysis_weather(forecast_start, cache)

    for sid, group in df.groupby('id', sort=True):
        zone = base.normalize_zone(str(sid))
        group = group.sort_values('timestamp').reset_index(drop=True)
        cuts = exact_cut_indices(len(group))
        supervised = _supervised(group)
        columns = safe_feature_columns(True)
        for window in windows:
            cut = cuts[window - 1]
            train = supervised.iloc[:cut]
            good = train[columns + ['target']].dropna().index
            X_train = train.loc[good, columns]
            y_train = train.loc[good, 'target'].astype(float)
            oracle = supervised.iloc[cut:cut + HORIZON].copy()
            truth = oracle['target'].astype(float).to_numpy()
            tests = {
                mapping: sens._weather_test(
                    oracle, analysis_by_window[window], mapping, zone, columns
                )
                for mapping in base.MAPPING_VARIANTS
            }
            for model_name in ('HGB', 'RF', 'ET'):
                model = model_factory(model_name, seed)
                model.fit(X_train, y_train)
                oracle_mae = float(np.mean(np.abs(truth - model.predict(oracle[columns]))))
                for mapping in base.MAPPING_VARIANTS:
                    analysis_mae = float(np.mean(np.abs(truth - model.predict(tests[mapping]))))
                    rows.extend([
                        {
                            'zone': zone, 'window': window, 'mapping': mapping,
                            'condition': 'ORACLE_WEATHER', 'model': model_name, 'mae': oracle_mae,
                        },
                        {
                            'zone': zone, 'window': window, 'mapping': mapping,
                            'condition': 'GFS_ANALYSIS', 'model': model_name, 'mae': analysis_mae,
                        },
                    ])
            print('DONE_ANALYSIS_ONLY', zone, window, flush=True)

    result = pd.DataFrame(rows)
    expected = len(windows) * df['id'].nunique() * len(base.MAPPING_VARIANTS) * 3 * 2
    if len(result) != expected:
        raise RuntimeError(f'incomplete analysis-only panel: got={len(result)} expected={expected}')
    result.to_csv(outdir / 'analysis_only_mae_by_window.csv', index=False)
    return result


def parse_windows(value):
    return base.parse_windows(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--seed', type=int, default=20260829)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    evaluate_analysis_only(parse_windows(args.windows), outdir, args.seed)


if __name__ == '__main__':
    main()
