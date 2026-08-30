# trigger loader contract through merge after installing parquet engine
import pandas as pd
import pytest

from research.topic_lock.gfc17_seed_null import (
    _load_fev_long,
    compare_seed_rankings,
    summarize_seed_null,
)


def _panel(seed, rows):
    return pd.DataFrame([
        {'zone': zone, 'window': window, 'model': model, 'mae': mae, 'seed': seed}
        for zone, window, model, mae in rows
    ])


def test_compare_seed_rankings_carries_reference_margin():
    reference = _panel(100, [
        ('Z', 1, 'HGB', 1.0), ('Z', 1, 'RF', 3.0), ('Z', 1, 'ET', 4.0),
        ('Z', 2, 'HGB', 1.0), ('Z', 2, 'RF', 1.4), ('Z', 2, 'ET', 5.0),
    ])
    alternate = _panel(200, [
        ('Z', 1, 'HGB', 1.2), ('Z', 1, 'RF', 0.9), ('Z', 1, 'ET', 4.0),
        ('Z', 2, 'HGB', 1.0), ('Z', 2, 'RF', 1.3), ('Z', 2, 'ET', 5.0),
    ])
    detail = compare_seed_rankings(reference, alternate, 100, 200)
    assert detail['reference_margin'].tolist() == pytest.approx([2.0, 0.4])
    assert detail['winner_changed'].tolist() == [True, False]


def test_margin_summary_conditions_on_reference_margin():
    detail = pd.DataFrame([
        {'alternate_seed': 200, 'reference_margin': 2.0, 'rank_changed': True, 'winner_changed': True},
        {'alternate_seed': 200, 'reference_margin': 0.4, 'rank_changed': False, 'winner_changed': False},
        {'alternate_seed': 300, 'reference_margin': 2.0, 'rank_changed': False, 'winner_changed': False},
        {'alternate_seed': 300, 'reference_margin': 0.4, 'rank_changed': True, 'winner_changed': True},
    ])
    summary = summarize_seed_null(detail, thresholds=(0.0, 2.0))
    all_rows = summary[summary['margin_threshold'] == 0.0].set_index('alternate_seed')
    hard_rows = summary[summary['margin_threshold'] == 2.0].set_index('alternate_seed')
    assert all_rows.loc[200, 'winner_change_rate'] == 0.5
    assert all_rows.loc[300, 'winner_change_rate'] == 0.5
    assert hard_rows.loc[200, 'winner_change_rate'] == 1.0
    assert hard_rows.loc[300, 'winner_change_rate'] == 0.0
    assert hard_rows.loc[200, 'n_cells'] == 1


def test_load_fev_long_honors_fixed_local_parquet(monkeypatch, tmp_path):
    p = tmp_path / 'gfc17.parquet'
    pd.DataFrame([
        {
            'id': 'Z1',
            'timestamp': [pd.Timestamp('2017-01-01 00:00'), pd.Timestamp('2017-01-01 01:00')],
            'target': [10.0, 11.0],
            'airtemperature': [30.0, 31.0],
        }
    ]).to_parquet(p)
    monkeypatch.setenv('FEV_GFC17_PARQUET', str(p))
    out = _load_fev_long()
    assert out[['id', 'target', 'airtemperature']].to_dict('records') == [
        {'id': 'Z1', 'target': 10.0, 'airtemperature': 30.0},
        {'id': 'Z1', 'target': 11.0, 'airtemperature': 31.0},
    ]
