from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


# Gate-0 RED contract: this test file intentionally lands before the implementation.
MODULE_PATH = Path("research/walmart_peer_history_gate0.py")


def _load_module():
    assert MODULE_PATH.exists(), "Walmart Gate-0 module has not been implemented yet"
    spec = importlib.util.spec_from_file_location("walmart_gate0", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_walmart_gate0_module_exists():
    assert MODULE_PATH.exists(), "Walmart Gate-0 module has not been implemented yet"


def test_recovery_placebo_is_exactly_one_when_benchmark_does_not_move():
    m = _load_module()
    got = m.recovery_from_benchmark_shift(benchmark_shift_log=0.0, delta=0.10)
    assert np.isclose(got, 1.0, atol=1e-12)


def test_history_contamination_only_changes_most_recent_fraction():
    m = _load_module()
    x = np.zeros(8, dtype=float)
    got = m.contaminate_history(x, contam_frac=0.25, delta=0.10)
    expected = np.zeros(8, dtype=float)
    expected[-2:] = np.log(0.90)
    assert np.allclose(got, expected)


def test_topk_excludes_self_and_returns_requested_k():
    m = _load_module()
    d = np.array([0.0, 0.3, 0.1, 0.2])
    peers = m.topk_peers(d, self_idx=0, k=2)
    assert set(peers.tolist()) == {2, 3}
