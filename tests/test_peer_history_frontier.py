import numpy as np
import pandas as pd

from research.peer_history_frontier import (
    center_rows,
    compose_distance,
    diagnostic_error_auc,
    diagnostic_retention_auc,
    make_config_grid,
    pareto_frontier,
    run_frontier_experiment,
)
from research.peer_history_frontier_fev import prepare_fev_rossmann_weekly_frame


def test_center_rows_removes_full_constant_level_shift():
    base = np.array([[1.0, 1.2, 0.9, 1.1], [2.0, 2.3, 1.8, 2.1]])
    shifted = base + np.log(0.90)
    np.testing.assert_allclose(center_rows(base), center_rows(shifted), atol=1e-12)


def test_center_rows_preserves_partial_contamination_shape_change():
    base = np.array([[1.0, 1.2, 0.9, 1.1]])
    shifted = base.copy()
    shifted[:, -2:] += np.log(0.90)
    assert not np.allclose(center_rows(base), center_rows(shifted))


def test_compose_distance_has_interpretable_endpoints():
    context = np.array([[0.0, 1.0], [1.0, 0.0]])
    level = np.array([[0.0, 2.0], [2.0, 0.0]])
    shape = np.array([[0.0, 3.0], [3.0, 0.0]])
    np.testing.assert_allclose(compose_distance(context, level, shape, alpha=0.0, level_share=0.5), context)
    np.testing.assert_allclose(compose_distance(context, level, shape, alpha=1.0, level_share=1.0), level)
    np.testing.assert_allclose(compose_distance(context, level, shape, alpha=1.0, level_share=0.0), shape)
    np.testing.assert_allclose(
        compose_distance(context, level, shape, alpha=1.0, level_share=0.25),
        0.25 * level + 0.75 * shape,
    )


def test_config_grid_contains_one_context_and_decomposed_history_arms():
    grid = make_config_grid()
    context = grid[grid["strategy"] == "context_only"]
    assert len(context) == 1
    assert (grid["strategy"] == "level_only").any()
    assert (grid["strategy"] == "shape_only").any()
    assert (grid["strategy"] == "level_shape_hybrid").any()
    assert (grid["strategy"] == "raw_trajectory_reference").any()


def test_diagnostic_retention_auc_matches_linear_one_minus_q():
    q = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    recovery = 1.0 - q
    assert abs(diagnostic_retention_auc(q, recovery) - 0.5) < 1e-12


def test_diagnostic_error_auc_penalizes_under_and_over_recovery_symmetrically():
    q = np.array([0.0, 0.5, 1.0])
    under = np.array([1.0, 0.8, 0.6])
    over = np.array([1.0, 1.2, 1.4])
    assert abs(diagnostic_error_auc(q, under) - diagnostic_error_auc(q, over)) < 1e-12
    assert diagnostic_error_auc(q, np.ones_like(q)) == 0.0


def test_pareto_frontier_maximizes_gain_and_minimizes_diagnostic_error():
    frame = pd.DataFrame(
        {
            "config_id": ["A", "B", "C", "D"],
            "comparability_gain_pct": [0.0, 10.0, 8.0, 12.0],
            "diagnostic_error_auc": [0.0, 0.30, 0.15, 0.40],
        }
    )
    front = pareto_frontier(frame)
    assert set(front["config_id"]) == {"A", "B", "C", "D"}
    dominated = pd.concat(
        [frame, pd.DataFrame({"config_id": ["E"], "comparability_gain_pct": [7.0], "diagnostic_error_auc": [0.30]})],
        ignore_index=True,
    )
    assert "E" not in set(pareto_frontier(dominated)["config_id"])


def test_prepare_fev_weekly_uses_pre_eval_history_and_non_outcome_context():
    timestamps = pd.date_range("2020-01-05", periods=8, freq="W-SUN")
    frame = pd.DataFrame(
        {
            "id": ["1", "2"],
            "timestamp": [list(timestamps), list(timestamps)],
            "Sales": [
                [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
                [200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0],
            ],
            "Promo": [[0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0, 0]],
            "Open": [[1] * 8, [1] * 8],
            "SchoolHoliday": [[0, 0, 1, 1, 0, 0, 1, 1], [0] * 8],
            "Store": [1.0, 2.0],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "c"],
            "CompetitionDistance": [100.0, 1000.0],
            "CompetitionOpenSinceMonth": [1.0, 6.0],
            "CompetitionOpenSinceYear": [2019.0, 2018.0],
            "Promo2": [0.0, 1.0],
            "Promo2SinceWeek": [np.nan, 1.0],
            "Promo2SinceYear": [np.nan, 2019.0],
            "PromoInterval": [np.nan, "Jan,Apr,Jul,Oct"],
        }
    )
    data = prepare_fev_rossmann_weekly_frame(frame, history_weeks=4, eval_weeks=2)
    assert data["perf_raw"].shape == (2, 4)
    np.testing.assert_allclose(data["perf_raw"][0], np.log([120.0, 130.0, 140.0, 150.0]))
    assert abs(data["y_eval"][0] - np.mean(np.log([160.0, 170.0]))) < 1e-12
    assert data["context_feature_count"] >= 7
    assert data["date_range"]["history_end"] == "2020-02-09"
    assert data["date_range"]["eval_start"] == "2020-02-16"


def test_screening_gate_kills_when_history_adds_no_comparability_information():
    n, weeks = 24, 12
    latent = np.linspace(0.0, 2.3, n)
    data = {
        "ids": np.arange(1, n + 1),
        "perf_raw": np.zeros((n, weeks), dtype=float),
        "y_eval": latent,
        "context_dist": np.abs(latent[:, None] - latent[None, :]),
        "context_feature_count": 1,
    }
    result = run_frontier_experiment(
        data,
        k=3,
        deltas=[0.10],
        contam_fracs=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    assert result["verdict"]["screening_verdict"] == "KILL_OR_REDESIGN"
    assert result["verdict"]["max_comparability_gain_pct"] < 5.0


def test_screening_gate_survives_when_history_buys_comparability_at_diagnostic_cost():
    n, weeks = 40, 12
    latent = np.linspace(8.0, 10.0, n)
    season = 0.04 * np.sin(np.linspace(0.0, 2.0 * np.pi, weeks, endpoint=False))
    data = {
        "ids": np.arange(1, n + 1),
        "perf_raw": latent[:, None] + season[None, :],
        "y_eval": latent + 0.02,
        "context_dist": np.zeros((n, n), dtype=float),
        "context_feature_count": 0,
    }
    result = run_frontier_experiment(
        data,
        k=5,
        deltas=[0.10],
        contam_fracs=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    assert result["verdict"]["screening_verdict"] == "FRONTIER_SURVIVE"
    assert result["verdict"]["max_comparability_gain_pct"] >= 5.0
    assert result["verdict"]["nontrivial_pareto_frontier"] is True
