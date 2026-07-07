"""End-to-end tests for the orchestrator: verdicts and report rendering."""
from __future__ import annotations

import numpy as np

from overfit_gauntlet.gauntlet import (
    GauntletReport,
    Verdict,
    run_gauntlet,
)


def test_positive_edge_is_validated(edge_returns):
    rep = run_gauntlet(edge_returns, seed=0)
    assert rep.verdict is Verdict.VALIDATED
    assert rep.passed is True
    # every documented bar cleared
    assert rep.psr > 0.95
    assert rep.dsr > 0.95
    assert rep.permutation_p < 0.05
    assert rep.purged_kfold[1] > 0.5


def test_pure_noise_is_not_validated(noise_returns):
    rep = run_gauntlet(noise_returns, seed=0)
    assert rep.verdict is Verdict.NOT_VALIDATED
    assert rep.passed is False
    # noise: permutation null not rejected, PSR well below the bar
    assert rep.permutation_p > 0.05
    assert rep.psr < 0.95


def test_gauntlet_is_deterministic(edge_returns):
    a = run_gauntlet(edge_returns, seed=0)
    b = run_gauntlet(edge_returns, seed=0)
    assert a.verdict is b.verdict
    assert a.psr == b.psr and a.permutation_p == b.permutation_p


def test_report_markdown_structure(edge_returns):
    rep = run_gauntlet(edge_returns, seed=0)
    md = rep.to_markdown()
    assert "# Overfit Gauntlet Report" in md
    assert "**Verdict: VALIDATED**" in md
    assert "## Checks" in md
    assert "## Verdict" in md
    # the checks table renders every documented bar
    for name in (
        "Probabilistic Sharpe (PSR)",
        "Deflated Sharpe (DSR)",
        "Permutation p-value",
        "Mean OOS Sharpe (purged K-fold)",
    ):
        assert name in md
    # verdict prose warns against cherry-picking
    assert "not a licence to keep searching" in md


def test_cost_stress_included_when_costs_given(edge_returns):
    costs = np.full(len(edge_returns), 0.0005)
    rep = run_gauntlet(edge_returns, costs=costs, seed=0)
    assert rep.cost is not None
    results, breakeven = rep.cost
    assert len(results) == 6  # default multiplier grid
    md = rep.to_markdown()
    assert "Cost stress" in md


def test_cost_absent_by_default(edge_returns):
    rep = run_gauntlet(edge_returns, seed=0)
    assert rep.cost is None


def test_weak_verdict_for_underpowered_edge():
    # A strong-looking but too-short series: it clears PSR/permutation but the
    # purged K-fold OOS bar can't run (n < min_test*n_folds), so not every bar
    # passes -> WEAK, and the prose refuses to bless it.
    rng = np.random.default_rng(0)
    vol = 0.01
    mu = 3.0 * vol / np.sqrt(252)
    r = rng.normal(mu, vol, 80)
    rep = run_gauntlet(r, seed=0)
    assert rep.verdict is Verdict.WEAK
    assert isinstance(rep, GauntletReport)
    md = rep.to_markdown()
    assert "**Verdict: WEAK**" in md
    assert "unproven candidate" in md
