"""Sanity tests for the basic performance primitives in ``metrics.py``."""
from __future__ import annotations

import math

import numpy as np
import pytest

from overfit_gauntlet.metrics import (
    EquitySummary,
    PerfReport,
    calmar,
    equity_summary,
    hit_rate,
    max_drawdown,
    perf_report,
    sharpe,
    sortino,
)


def test_sharpe_matches_hand_formula():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, 500)
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert sharpe(r) == pytest.approx(expected)


def test_sharpe_annualization_scales_with_ann():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, 500)
    assert sharpe(r, ann=365) == pytest.approx(sharpe(r, ann=252) * math.sqrt(365 / 252))


def test_sharpe_constant_series_is_zero():
    assert sharpe(np.full(100, 0.001)) == 0.0


def test_sharpe_too_short_is_zero():
    # fewer than 5 observations -> undefined, returns 0.0
    assert sharpe(np.array([0.01, 0.02, -0.01])) == 0.0


def test_sharpe_ignores_nonfinite():
    clean = np.array([0.01, -0.02, 0.03, 0.0, 0.015, -0.005])
    dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
    assert sharpe(dirty) == pytest.approx(sharpe(clean))


def test_sortino_penalizes_only_downside():
    r = np.array([0.02, 0.03, -0.01, 0.04, -0.02, 0.01, 0.05])
    down = r[r < 0]
    expected = r.mean() / down.std(ddof=1) * math.sqrt(252)
    assert sortino(r) == pytest.approx(expected)


def test_max_drawdown_is_nonpositive_and_known():
    # +10% then -50% -> equity 1.1 then 0.55; trough drawdown = 0.55/1.1 - 1 = -0.5
    r = np.array([0.10, -0.50])
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert max_drawdown(np.array([0.01, 0.02, 0.03])) <= 0.0


def test_calmar_zero_when_no_drawdown():
    # strictly increasing equity has no drawdown -> calmar guards against /0
    assert calmar(np.full(50, 0.01)) == 0.0


def test_hit_rate_fraction_positive():
    r = np.array([0.1, -0.1, 0.2, 0.0, 0.3])
    assert hit_rate(r) == pytest.approx(3 / 5)  # strictly > 0 only (0.0 excluded)


def test_perf_report_fields(edge_returns):
    rep = perf_report(edge_returns)
    assert isinstance(rep, PerfReport)
    assert rep.n == len(edge_returns)
    assert rep.sharpe == pytest.approx(sharpe(edge_returns))
    assert rep.ann_vol > 0
    assert "Sharpe=" in str(rep)


def test_equity_summary_consistency(edge_returns):
    eq = equity_summary(edge_returns)
    assert isinstance(eq, EquitySummary)
    assert eq.n_days == len(edge_returns)
    growth = float(np.prod(1 + edge_returns))
    assert eq.final_equity == pytest.approx(growth)
    assert eq.total_return == pytest.approx(growth - 1.0)
    assert 0.0 <= eq.time_under_water <= 1.0
    assert eq.worst_day <= eq.best_day


def test_equity_summary_empty_is_neutral():
    eq = equity_summary(np.array([]))
    assert eq.n_days == 0
    assert eq.final_equity == 1.0
