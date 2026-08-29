import pandas as pd
import pytest

from research.topic_lock.gdex_sensitivity import (
    analysis_cycles_for_horizon,
    cycle_and_leads,
)


def test_same00_cycle_brackets_august_horizon():
    cycle, leads = cycle_and_leads(pd.Timestamp('2017-08-14 00:00:00'), rule='same00')
    assert cycle == pd.Timestamp('2017-08-14 00:00:00')
    assert leads[0] == 3
    assert leads[-1] == 171


def test_same00_cycle_brackets_december_horizon():
    cycle, leads = cycle_and_leads(pd.Timestamp('2017-12-25 00:00:00'), rule='same00')
    assert cycle == pd.Timestamp('2017-12-25 00:00:00')
    assert leads[0] == 3
    assert leads[-1] == 174


def test_previous18_remains_default_contract():
    default_cycle, default_leads = cycle_and_leads(pd.Timestamp('2017-08-14 00:00:00'))
    explicit_cycle, explicit_leads = cycle_and_leads(
        pd.Timestamp('2017-08-14 00:00:00'), rule='previous18'
    )
    assert default_cycle == explicit_cycle == pd.Timestamp('2017-08-13 18:00:00')
    assert default_leads == explicit_leads


def test_unknown_cycle_rule_fails_closed():
    with pytest.raises(ValueError, match='cycle rule'):
        cycle_and_leads(pd.Timestamp('2017-08-14 00:00:00'), rule='future99')


def test_analysis_cycles_bracket_entire_utc_horizon_at_six_hour_spacing():
    cycles = analysis_cycles_for_horizon(pd.Timestamp('2017-08-14 00:00:00'))
    assert cycles[0] == pd.Timestamp('2017-08-14 00:00:00')
    assert cycles[-1] == pd.Timestamp('2017-08-21 06:00:00')
    assert len(cycles) == 30
    assert all((b - a) == pd.Timedelta(hours=6) for a, b in zip(cycles[:-1], cycles[1:]))
