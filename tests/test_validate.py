"""Tests for OOS validation and multiple-testing controls in ``validate.py``.

Covers the two headline correctness claims: purged K-fold really removes the
purge+embargo rows from the training side, the Benjamini-Hochberg reject mask
against a hand-worked example, and the cost-stress break-even interpolation.
"""
from __future__ import annotations

import numpy as np
import pytest

from overfit_gauntlet.metrics import sharpe
from overfit_gauntlet.validate import (
    bh_fdr,
    cost_stress,
    purged_kfold_cv,
    walk_forward_oos,
)


# --- walk-forward ---------------------------------------------------------
def test_walk_forward_shape_and_means(edge_returns):
    folds, mean_oos, std_oos = walk_forward_oos(edge_returns, n_folds=5)
    assert folds
    assert mean_oos == pytest.approx(np.mean([o for _, o in folds]))
    assert std_oos >= 0.0


# --- purged K-fold: the core exclusion guarantee --------------------------
def test_purged_kfold_excludes_purge_and_embargo_rows():
    """Reconstruct the train mask by hand and confirm the fold's train Sharpe
    is computed on exactly (all rows) minus (test block + purge band on both
    sides + trailing embargo)."""
    n, n_folds, purge, embargo = 200, 5, 3, 5
    rng = np.random.default_rng(11)
    r = rng.normal(0.0005, 0.01, n)
    # inject a distinctive regime inside the embargo band of fold 2 so that
    # forgetting to exclude it would visibly change the train Sharpe.
    r[123:128] = 0.5

    folds, _, _ = purged_kfold_cv(r, n_folds=n_folds, purge=purge, embargo=embargo)

    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    k = 2
    t0, t1 = int(bounds[k]), int(bounds[k + 1])  # 80, 120
    lo = max(0, t0 - purge)
    hi = min(n, t1 + purge + embargo)            # excludes trailing embargo band

    expected_mask = np.ones(n, dtype=bool)
    expected_mask[lo:hi] = False
    expected_train_sharpe = sharpe(r[expected_mask])
    expected_test_sharpe = sharpe(r[t0:t1])

    got_train, got_test = folds[k]
    assert got_train == pytest.approx(expected_train_sharpe)
    assert got_test == pytest.approx(expected_test_sharpe)

    # And the exclusion is non-vacuous: a mask WITHOUT the embargo band (the
    # bug we are guarding against) would give a different train Sharpe.
    naive_mask = np.ones(n, dtype=bool)
    naive_mask[max(0, t0 - purge):min(n, t1 + purge)] = False
    assert sharpe(r[naive_mask]) != pytest.approx(expected_train_sharpe)

    # The purged/embargoed rows are genuinely absent from the training set.
    excluded = set(range(lo, hi))
    assert excluded == set(range(t0 - purge, t1 + purge + embargo))
    assert len(r[expected_mask]) == n - (hi - lo)


def test_purged_kfold_too_few_rows_returns_empty():
    folds, mean_t, std_t = purged_kfold_cv(np.random.default_rng(0).normal(0, 0.01, 30))
    assert folds == []
    assert (mean_t, std_t) == (0.0, 0.0)


# --- Benjamini-Hochberg ---------------------------------------------------
def test_bh_fdr_worked_example_rejects_first_three():
    # m=5, alpha=0.05 -> thresholds k/m*alpha = [.01,.02,.03,.04,.05]
    # p = [.001,.01,.02,.5,.6]: first three clear their thresholds, last two don't
    mask = bh_fdr([0.001, 0.01, 0.02, 0.5, 0.6], alpha=0.05)
    assert list(mask) == [True, True, True, False, False]


def test_bh_fdr_is_step_up_not_pointwise():
    # p2=0.03 fails its own threshold 0.02, but a larger rank passes, so the
    # step-up procedure rejects everything up to that rank (incl. p2).
    mask = bh_fdr([0.001, 0.03, 0.03, 0.035, 0.9], alpha=0.05)
    assert list(mask) == [True, True, True, True, False]


def test_bh_fdr_mask_aligned_to_input_order():
    ordered = [0.001, 0.01, 0.02, 0.5, 0.6]
    perm = [3, 1, 4, 0, 2]  # arbitrary shuffle of indices
    shuffled = [ordered[i] for i in perm]
    mask = bh_fdr(shuffled, alpha=0.05)
    # rejected iff the original value was one of the first three (< threshold)
    rejected_vals = {ordered[0], ordered[1], ordered[2]}
    assert [bool(m) for m in mask] == [v in rejected_vals for v in shuffled]


def test_bh_fdr_none_pass_and_empty():
    assert not bh_fdr([0.2, 0.3, 0.4], alpha=0.05).any()
    assert bh_fdr([], alpha=0.05).tolist() == []


# --- cost stress ----------------------------------------------------------
def test_cost_stress_breakeven_interpolation():
    rng = np.random.default_rng(0)
    g = rng.normal(0.002, 0.01, 300)
    c = np.full(300, 0.001)
    results, breakeven = cost_stress(g, c)

    # net Sharpe decreases monotonically as we scale up costs
    sharpes = [s for _, s in results]
    assert sharpes == sorted(sharpes, reverse=True)

    # break-even is the first sign-change bracket, linearly interpolated
    bracket = next(
        i for i in range(len(results) - 1)
        if (results[i][1] > 0) != (results[i + 1][1] > 0)
    )
    m0, s0 = results[bracket]
    m1, s1 = results[bracket + 1]
    expected = m0 + (m1 - m0) * s0 / (s0 - s1)
    assert breakeven == pytest.approx(expected)
    assert m0 < breakeven < m1


def test_cost_stress_indestructible_edge_never_breaks_even():
    g = np.full(100, 0.01) + np.random.default_rng(1).normal(0, 1e-4, 100)
    c = np.zeros(100)
    _, breakeven = cost_stress(g, c)
    assert breakeven == float("inf")


def test_cost_stress_length_mismatch_raises():
    with pytest.raises(ValueError):
        cost_stress(np.zeros(10), np.zeros(9))
