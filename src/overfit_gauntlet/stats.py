"""Anti-overfit statistical toolkit.

Built on the primitives in :mod:`overfit_gauntlet.metrics`. Includes probabilistic
& deflated Sharpe (Bailey & Lopez de Prado 2014), stationary block-bootstrap CIs,
a sign-permutation null, and a bootstrapped zero-return p-value. These let us state
how likely a positive Sharpe is *real* vs. luck / multiple-testing.

The default annualization is ``ann=252`` (equities); every function takes an
``ann`` parameter.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy import stats as _sps

from .metrics import ANN, _clean, sharpe


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0, ann: int = ANN) -> float:
    """P(true SR > benchmark), correcting for skew/kurtosis and sample length.

    Returns a probability in [0,1]; >0.95 is the usual 'significant' bar.
    """
    r = _clean(r)
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)                # per-period (non-annualized)
    sr_b = sr_benchmark / math.sqrt(ann)
    sk = float(_sps.skew(r))
    ku = float(_sps.kurtosis(r, fisher=False))   # non-excess
    denom = math.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    z = (sr - sr_b) * math.sqrt(n - 1) / denom
    return float(_sps.norm.cdf(z))


def deflated_sharpe(r: np.ndarray, n_trials: int, ann: int = ANN,
                    var_sr_trials: Optional[float] = None) -> float:
    """Deflated Sharpe: PSR against the benchmark you'd expect as the MAX of
    `n_trials` independent strategy searches. Guards against selection bias.

    Bailey & Lopez de Prado (2014): the multiple-testing benchmark is the expected
    maximum of `n_trials` Sharpe estimates drawn from N(0, V), where ``V`` is the
    *cross-trial variance of the per-period (non-annualized) Sharpe estimates* that
    the search actually produced. Pass that variance as ``var_sr_trials``.

    The historical default of ``V = 1.0`` is WRONG for a per-period Sharpe: a unit
    per-period SR variance implies trial Sharpes swinging by ~1.0 *per bar*, which
    annualizes the benchmark to ``sqrt(ann)`` ~30 — an impossible bar that drives
    PSR(sr_star) -> 0 for every real strategy. Hence a deflated Sharpe of *exactly*
    0.000 is a red flag for a DEGENERATE benchmark (mis-scaled ``var_sr_trials``),
    not evidence that the edge is fake. When the cross-trial variance is unknown we
    instead fall back to the asymptotic sampling variance of a single per-period
    Sharpe under the null, ~``1/(n-1)`` (Lo 2002), which keeps the benchmark on the
    same per-bar scale as the strategy's own SR.
    """
    r = _clean(r)
    n = len(r)
    if n_trials < 1 or n < 10:
        return 0.0
    # Cross-trial variance of the per-period Sharpe estimates. Fall back to the
    # ~1/(n-1) sampling variance (NOT the degenerate unit variance) when unknown.
    if var_sr_trials is not None and var_sr_trials > 0:
        v = var_sr_trials
    else:
        v = 1.0 / max(n - 1, 1)
    emc = 0.5772156649
    e_max = math.sqrt(v) * ((1 - emc) * _sps.norm.ppf(1 - 1.0 / n_trials)
                            + emc * _sps.norm.ppf(1 - 1.0 / (n_trials * math.e)))
    sr_star = e_max * math.sqrt(ann)             # annualized benchmark
    return probabilistic_sharpe(r, sr_benchmark=sr_star, ann=ann)


def block_bootstrap_sharpe_ci(r: np.ndarray, block: int = 10, n_boot: int = 2000,
                              ci: float = 0.95, ann: int = ANN,
                              seed: int = 0) -> Tuple[float, float, float]:
    """Stationary-block bootstrap CI for the Sharpe ratio. Returns (lo, median, hi)."""
    r = _clean(r)
    n = len(r)
    if n < block + 5:
        s = sharpe(r, ann)
        return (s, s, s)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n_blocks = int(math.ceil(n / block))
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        boots[b] = sharpe(r[idx[:n]], ann)
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return (lo, float(np.median(boots)), hi)


def permutation_pvalue(r: np.ndarray, n_perm: int = 2000, ann: int = ANN, seed: int = 0) -> float:
    """One-sided p-value: P(shuffled-sign Sharpe >= observed) under the null that
    the strategy has no timing skill (random signs). Low p => real edge."""
    r = _clean(r)
    obs = sharpe(r, ann)
    if obs <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(r))
        if sharpe(r * signs, ann) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def bootstrap_pvalue_zero(r: np.ndarray, block: int = 10, n_boot: int = 5000,
                          ann: int = ANN, seed: int = 7) -> float:
    """One-sided bootstrapped p-value against the zero-return (no-edge) null.

    We mean-center the realized returns (imposing H0: E[r]=0) and draw stationary
    block-bootstrap resamples, preserving autocorrelation. p = P(bootstrap Sharpe
    >= observed Sharpe). A small p means the observed edge is unlikely under pure
    noise with the same volatility structure.
    """
    r = _clean(r)
    n = len(r)
    obs = sharpe(r, ann)
    if n < block + 5 or obs <= 0:
        return 1.0
    centered = r - r.mean()             # impose the null E[r]=0
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    cnt = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        if sharpe(centered[idx[:n]], ann) >= obs:
            cnt += 1
    return (cnt + 1) / (n_boot + 1)
