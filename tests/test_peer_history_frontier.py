import numpy as np
import pandas as pd

from research.peer_history_frontier import (
    center_rows,
    compose_distance,
    diagnostic_error_auc,
    diagnostic_retention_auc,
    make_config_grid,
    pareto_frontier,
)


def test_center_rows_removes_full_constant_level_shift():
    base = np.array([
        [1.0, 1.2, 0.9, 1.1],
        [2.0, 2.3, 1.8, 2.1],
    ])
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

    np.testing.assert_allclose(
        compose_distance(context, level, shape, alpha=0.0, level_share=0.5), context
    )
    np.testing.assert_allclose(
        compose_distance(context, level, shape, alpha=1.0, level_share=1.0), level
    )
    np.testing.assert_allclose(
        compose_distance(context, level, shape, alpha=1.0, level_share=0.0), shape
    )
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
        [
            frame,
            pd.DataFrame(
                {
                    "config_id": ["E"],
                    "comparability_gain_pct": [7.0],
                    "diagnostic_error_auc": [0.30],
                }
            ),
        ],
        ignore_index=True,
    )
    front2 = pareto_frontier(dominated)
    assert "E" not in set(front2["config_id"])
