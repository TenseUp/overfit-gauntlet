"""Out-of-sample validation & multiple-testing controls.

Walk-forward and purged K-fold cross-validation (Lopez de Prado, *Advances in
Financial ML* §7), Benjamini-Hochberg FDR control for a battery of candidate
strategies, and a fee-sensitivity stress test. All built on the Sharpe primitive
in :mod:`overfit_gauntlet.metrics`.

The default annualization is ``ann=252`` (equities); every function takes an
``ann`` parameter.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .metrics import ANN, _clean, sharpe


def walk_forward_oos(r: np.ndarray, n_folds: int = 5, min_oos: int = 20,
                     ann: int = ANN) -> Tuple[List[Tuple[float, float]], float, float]:
    """Expanding-window walk-forward OOS on a (already causal) return series.

    "Training" here only warms up the causal model; the value of this test is
    consistency: does the edge persist in sequential, never-seen tail windows, or is
    it one lucky regime?

    Returns (folds[(train_sharpe, oos_sharpe)], mean_oos_sharpe, oos_sharpe_std).
    """
    r = _clean(r)
    n = len(r)
    folds: List[Tuple[float, float]] = []
    for k in range(1, n_folds + 1):
        tr_end = int(n * k / (n_folds + 1))
        oos_end = int(n * (k + 1) / (n_folds + 1))
        if oos_end - tr_end < min_oos:
            continue
        folds.append((sharpe(r[:tr_end], ann), sharpe(r[tr_end:oos_end], ann)))
    oos = [o for _, o in folds]
    mean_oos = float(np.mean(oos)) if oos else 0.0
    std_oos = float(np.std(oos)) if len(oos) > 1 else 0.0
    return folds, mean_oos, std_oos


def purged_kfold_cv(r: np.ndarray, n_folds: int = 5, purge: int = 1, embargo: int = 0,
                    min_test: int = 20, ann: int = ANN
                    ) -> Tuple[List[Tuple[float, float]], float, float]:
    """Purged & embargoed K-fold cross-validation (Lopez de Prado, *Advances in
    Financial ML* §7) on a serially-overlapping return series.

    Plain walk-forward leaks: because positions are held for several days, the rows
    straddling a train/test boundary share overlapping holding periods, so the
    "train" side sees information about the adjacent test rows. Here each fold is a
    contiguous *test* block; the training set is every OTHER row EXCEPT

      * a ``purge`` band on BOTH sides of the test block — removes rows whose holding
        period overlaps the test window (the label-leakage fix), and
      * an ``embargo`` band immediately AFTER the test block — removes rows whose
        features are serially correlated with the just-ended test window (the
        forward-leakage fix; embargo only trails because information flows forward).

    Unlike ``walk_forward_oos`` (expanding, tail-only) this evaluates EVERY segment
    out-of-sample and cross-checks each against a genuinely disjoint train side, so a
    high mean-test Sharpe here is much harder to fake with one lucky regime.

    Returns (folds[(train_sharpe, test_sharpe)], mean_test_sharpe, test_sharpe_std).
    """
    r = _clean(r)
    n = len(r)
    folds: List[Tuple[float, float]] = []
    if n < min_test * n_folds:
        return folds, 0.0, 0.0
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    for k in range(n_folds):
        t0, t1 = int(bounds[k]), int(bounds[k + 1])
        if t1 - t0 < min_test:
            continue
        lo = max(0, t0 - purge)
        hi = min(n, t1 + purge + embargo)          # embargo trails the test block only
        mask = np.ones(n, dtype=bool)
        mask[lo:hi] = False                        # drop test block + purge/embargo
        train = r[mask]
        test = r[t0:t1]
        folds.append((sharpe(train, ann), sharpe(test, ann)))
    tests = [o for _, o in folds]
    mean_test = float(np.mean(tests)) if tests else 0.0
    std_test = float(np.std(tests)) if len(tests) > 1 else 0.0
    return folds, mean_test, std_test


def bh_fdr(pvalues: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR control. Returns a boolean reject mask.

    When you test many candidate strategies, some will clear ``p<alpha`` by luck.
    BH controls the *expected proportion of false discoveries* among the rejections
    at level ``alpha``: sort the p-values ascending, find the largest rank ``k`` with
    ``p_(k) <= (k/m) * alpha``, and reject every hypothesis with a p-value at or
    below that threshold. The returned mask is aligned to the input order.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    reject = np.zeros(m, dtype=bool)
    if m == 0:
        return reject
    order = np.argsort(p, kind="stable")            # ascending
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = ranked <= thresh
    if not passed.any():
        return reject
    k_max = np.max(np.nonzero(passed)[0])           # largest passing rank (0-indexed)
    reject[order[:k_max + 1]] = True                # reject all up to and including k_max
    return reject


def cost_stress(gross_returns: np.ndarray, costs: np.ndarray,
                multipliers: Sequence[float] = (0, 0.5, 1, 1.5, 2, 3),
                ann: int = ANN) -> Tuple[List[Tuple[float, float]], float]:
    """Fee-sensitivity stress: re-price the book at scaled cost drags.

    ``gross_returns`` is the per-period return before costs; ``costs`` is the
    per-period cost drag (same length, non-negative magnitudes). For each
    ``mult`` in ``multipliers`` we compute the net Sharpe of
    ``gross_returns - mult * costs``.

    Returns ``(results, breakeven_mult)`` where ``results`` is a list of
    ``(mult, sharpe)`` and ``breakeven_mult`` is the cost multiple at which the net
    Sharpe crosses zero, found by linear interpolation between the bracketing
    multipliers (``inf`` if it never crosses within the tested range).

    "Edges that die at realistic fees are not edges."
    """
    g = np.asarray(gross_returns, dtype=float)
    c = np.asarray(costs, dtype=float)
    if len(g) != len(c):
        raise ValueError("gross_returns and costs must be the same length")
    mults = list(multipliers)
    results: List[Tuple[float, float]] = []
    for m in mults:
        net = g - m * c
        results.append((float(m), sharpe(net, ann)))

    # linear interpolation for the first zero crossing of sharpe vs multiplier
    breakeven = float("inf")
    for i in range(len(results) - 1):
        m0, s0 = results[i]
        m1, s1 = results[i + 1]
        if s0 == 0.0:
            breakeven = m0
            break
        if (s0 > 0) != (s1 > 0):                    # sign change brackets a root
            breakeven = m0 + (m1 - m0) * s0 / (s0 - s1)
            break
    else:
        # loop completed without break; check the final point exactly on zero
        if results and results[-1][1] == 0.0:
            breakeven = results[-1][0]
    return results, breakeven
