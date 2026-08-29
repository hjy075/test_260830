from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research.topic_lock import gdex_operational_gfs as base
from research.topic_lock.gfc17_seed_null import (
    HORIZON,
    _load_fev_long,
    _supervised,
    exact_cut_indices,
    model_factory,
    safe_feature_columns,
)


CONDITIONS = ('ORACLE_WEATHER', 'GFS_ANALYSIS', 'GFS_FORECAST_SAME00')


def cycle_and_leads(forecast_start, rule: str = 'previous18'):
    """Return a UTC-naive GFS cycle and 3-hour leads bracketing the local horizon."""
    if rule == 'previous18':
        return base.cycle_and_leads(forecast_start)
    if rule != 'same00':
        raise ValueError(f'unknown cycle rule: {rule}')

    target = base.local_horizon_to_utc(forecast_start)
    cycle = pd.Timestamp(forecast_start).normalize()
    lo = int(math.floor(float((target[0] - cycle) / pd.Timedelta(hours=1)) / 3) * 3)
    hi = int(math.ceil(float((target[-1] - cycle) / pd.Timedelta(hours=1)) / 3) * 3)
    return cycle, list(range(lo, hi + 1, 3))


def analysis_cycles_for_horizon(forecast_start) -> list[pd.Timestamp]:
    """Six-hourly GFS f000 cycles bracketing the full UTC target horizon."""
    target = base.local_horizon_to_utc(forecast_start)
    first = pd.Timestamp(target[0]).floor('6h')
    last = pd.Timestamp(target[-1]).ceil('6h')
    return list(pd.date_range(first, last, freq='6h'))


def _interpolate_station_rows(rows, forecast_start):
    target = base.local_horizon_to_utc(forecast_start)
    series = pd.DataFrame(rows).set_index('timestamp_utc').sort_index()
    if series.index.has_duplicates:
        raise RuntimeError('duplicate UTC weather timestamps')
    series = series.reindex(series.index.union(target)).interpolate(method='time').reindex(target)
    if series.isna().any().any():
        raise RuntimeError('weather interpolation left missing values')
    out = series.reset_index()
    out.insert(
        0,
        'timestamp_local',
        pd.date_range(pd.Timestamp(forecast_start), periods=HORIZON, freq='h'),
    )
    return out


def acquire_forecast_weather(forecast_start, cache_dir: Path, rule: str = 'same00'):
    cycle, leads = cycle_and_leads(forecast_start, rule=rule)
    session = requests.Session()
    rows = []
    for lead in leads:
        values, mode = base.fetch_station_values_osdf(cycle, lead, session, cache_dir)
        rows.append({'timestamp_utc': cycle + pd.Timedelta(hours=lead), **values})
        print('GFS_FORECAST', rule, cycle, 'lead', lead, 'mode', mode, 'ok', flush=True)
    return _interpolate_station_rows(rows, forecast_start)


def acquire_analysis_weather(forecast_start, cache_dir: Path):
    session = requests.Session()
    rows = []
    for cycle in analysis_cycles_for_horizon(forecast_start):
        values, mode = base.fetch_station_values_osdf(cycle, 0, session, cache_dir)
        rows.append({'timestamp_utc': cycle, **values})
        print('GFS_ANALYSIS', cycle, 'lead', 0, 'mode', mode, 'ok', flush=True)
    return _interpolate_station_rows(rows, forecast_start)


def _weather_test(oracle, station_df, mapping, zone, columns):
    weather = base.zone_weather(station_df, mapping)[zone].to_numpy(float)
    test = oracle.copy()
    test.loc[:, 'weather'] = weather
    test.loc[:, 'weather_sq'] = weather ** 2
    test.loc[:, 'weather_hour'] = weather * test['hour_sin'].to_numpy(float)
    return test[columns]


def evaluate_windows(windows, outdir: Path, seed: int = 20260829):
    df = _load_fev_long()
    rows = []
    same00_cache = outdir / 'same00_cache'
    analysis_cache = outdir / 'analysis_cache'

    first = next(iter(df.groupby('id', sort=True)))[1].sort_values('timestamp').reset_index(drop=True)
    cuts0 = exact_cut_indices(len(first))
    weather_by_window = {}
    for window in windows:
        forecast_start = pd.Timestamp(first.iloc[cuts0[window - 1]]['timestamp'])
        weather_by_window[window] = {
            'GFS_FORECAST_SAME00': acquire_forecast_weather(
                forecast_start,
                same00_cache,
                rule='same00',
            ),
            'GFS_ANALYSIS': acquire_analysis_weather(
                forecast_start,
                analysis_cache,
            ),
        }

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

            condition_tests = {}
            for mapping in base.MAPPING_VARIANTS:
                condition_tests[mapping] = {
                    'GFS_ANALYSIS': _weather_test(
                        oracle,
                        weather_by_window[window]['GFS_ANALYSIS'],
                        mapping,
                        zone,
                        columns,
                    ),
                    'GFS_FORECAST_SAME00': _weather_test(
                        oracle,
                        weather_by_window[window]['GFS_FORECAST_SAME00'],
                        mapping,
                        zone,
                        columns,
                    ),
                }

            for model_name in ('HGB', 'RF', 'ET'):
                model = model_factory(model_name, seed)
                model.fit(X_train, y_train)
                oracle_mae = float(np.mean(np.abs(truth - model.predict(oracle[columns]))))
                for mapping in base.MAPPING_VARIANTS:
                    rows.append({
                        'zone': zone,
                        'window': window,
                        'mapping': mapping,
                        'condition': 'ORACLE_WEATHER',
                        'model': model_name,
                        'mae': oracle_mae,
                    })
                    for condition in ('GFS_ANALYSIS', 'GFS_FORECAST_SAME00'):
                        pred = model.predict(condition_tests[mapping][condition])
                        rows.append({
                            'zone': zone,
                            'window': window,
                            'mapping': mapping,
                            'condition': condition,
                            'model': model_name,
                            'mae': float(np.mean(np.abs(truth - pred))),
                        })
            print('DONE_SENSITIVITY', zone, window, flush=True)

    result = pd.DataFrame(rows)
    expected = len(windows) * df['id'].nunique() * len(base.MAPPING_VARIANTS) * 3 * len(CONDITIONS)
    if len(result) != expected:
        raise RuntimeError(f'incomplete sensitivity panel: got={len(result)} expected={expected}')
    result.to_csv(outdir / 'sensitivity_mae_by_window.csv', index=False)
    return result


def _order_and_margin(matrix, condition, models):
    order = tuple(sorted(models, key=lambda x: (float(matrix.loc[condition, x]), x)))
    values = sorted(float(matrix.loc[condition, x]) for x in models)
    return order, values[1] - values[0]


def summarize(result, outdir: Path):
    pivot = result.pivot_table(
        index=['zone', 'window', 'mapping', 'model'],
        columns='condition',
        values='mae',
    ).reset_index()
    for condition in CONDITIONS:
        if condition not in pivot.columns:
            raise RuntimeError(f'missing condition in sensitivity panel: {condition}')
    pivot['D_analysis_minus_oracle'] = pivot['GFS_ANALYSIS'] - pivot['ORACLE_WEATHER']
    pivot['D_same00_minus_analysis'] = pivot['GFS_FORECAST_SAME00'] - pivot['GFS_ANALYSIS']
    pivot['D_same00_minus_oracle'] = pivot['GFS_FORECAST_SAME00'] - pivot['ORACLE_WEATHER']
    pivot.to_csv(outdir / 'sensitivity_contrasts.csv', index=False)

    comparisons = (
        ('oracle_to_analysis', 'ORACLE_WEATHER', 'GFS_ANALYSIS'),
        ('analysis_to_same00', 'GFS_ANALYSIS', 'GFS_FORECAST_SAME00'),
        ('oracle_to_same00', 'ORACLE_WEATHER', 'GFS_FORECAST_SAME00'),
    )
    rank_rows = []
    models = ['HGB', 'RF', 'ET']
    for (mapping, zone, window), group in result.groupby(['mapping', 'zone', 'window']):
        matrix = group.pivot_table(index='condition', columns='model', values='mae')
        for label, from_condition, to_condition in comparisons:
            from_order, from_margin = _order_and_margin(matrix, from_condition, models)
            to_order, _ = _order_and_margin(matrix, to_condition, models)
            rank_rows.append({
                'mapping': mapping,
                'zone': zone,
                'window': window,
                'comparison': label,
                'from_condition': from_condition,
                'to_condition': to_condition,
                'rank_changed': from_order != to_order,
                'winner_changed': from_order[0] != to_order[0],
                'from_margin': from_margin,
                'from_rank': '>'.join(from_order),
                'to_rank': '>'.join(to_order),
            })
    pd.DataFrame(rank_rows).to_csv(outdir / 'sensitivity_ranking.csv', index=False)


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
    result = evaluate_windows(parse_windows(args.windows), outdir, args.seed)
    summarize(result, outdir)


if __name__ == '__main__':
    main()
