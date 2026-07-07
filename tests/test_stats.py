"""Tests for the anti-overfit statistical toolkit in ``stats.py``."""
from __future__ import annotations

import numpy as np
import pytest

from overfit_gauntlet.stats import (
    block_bootstrap_sharpe_ci,
    bootstrap_pvalue_zero,
    deflated_sharpe,
    permutation_pvalue,
    probabilistic_sharpe,
)


def test_psr_is_probability(edge_returns, noise_returns):
    for r in (edge_returns, noise_returns):
        p = probabilistic_sharpe(r)
        assert 0.0 <= p <= 1.0


def test_psr_higher_for_stronger_edge(edge_returns, noise_returns):
    assert probabilistic_sharpe(edge_returns) > probabilistic_sharpe(noise_returns)


def test_psr_short_series_is_zero():
    assert probabilistic_sharpe(np.array([0.01, 0.02, 0.03])) == 0.0


def test_deflated_sharpe_bounded_and_shrinks_with_trials(edge_returns):
    dsr1 = deflated_sharpe(edge_returns, n_trials=1)
    dsr100 = deflated_sharpe(edge_returns, n_trials=100)
    assert 0.0 <= dsr100 <= dsr1 <= 1.0


def test_deflated_sharpe_footgun_docstring_preserved():
    # The deflated-Sharpe variance footgun docstring is a key selling point.
    doc = deflated_sharpe.__doc__ or ""
    assert "1.0" in doc and "red flag" in doc.lower()


def test_deflated_sharpe_var_override(edge_returns):
    # explicit cross-trial variance is honored (differs from the 1/(n-1) fallback)
    fallback = deflated_sharpe(edge_returns, n_trials=50)
    override = deflated_sharpe(edge_returns, n_trials=50, var_sr_trials=0.5)
    assert override != fallback


def test_block_bootstrap_ci_ordered_and_seeded(edge_returns):
    lo, med, hi = block_bootstrap_sharpe_ci(edge_returns, seed=0)
    assert lo <= med <= hi
    # deterministic under a fixed seed
    assert (lo, med, hi) == block_bootstrap_sharpe_ci(edge_returns, seed=0)
    # the point Sharpe should sit inside a 95% CI for a strong, stable edge
    from overfit_gauntlet.metrics import sharpe

    assert lo <= sharpe(edge_returns) <= hi


def test_permutation_pvalue_edge_vs_noise(edge_returns, noise_returns):
    assert permutation_pvalue(edge_returns, seed=0) < 0.05
    assert permutation_pvalue(noise_returns, seed=0) > 0.05


def test_permutation_pvalue_nonpositive_sharpe_is_one():
    # a losing series has obs Sharpe <= 0 -> p-value defined as 1.0
    r = np.linspace(-0.02, -0.001, 100)
    assert permutation_pvalue(r) == 1.0


def test_bootstrap_pvalue_zero_edge_vs_noise(edge_returns, noise_returns):
    assert bootstrap_pvalue_zero(edge_returns, seed=7) < 0.05
    # noise has no edge; its p-value against the zero null should be large
    assert bootstrap_pvalue_zero(noise_returns, seed=7) > 0.05


def test_bootstrap_pvalue_zero_is_seeded(edge_returns):
    assert bootstrap_pvalue_zero(edge_returns, seed=7) == bootstrap_pvalue_zero(
        edge_returns, seed=7
    )
