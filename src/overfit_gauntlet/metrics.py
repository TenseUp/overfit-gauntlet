"""Rigorous performance statistics.

Pure functions over a 1-D array of per-period returns. This module holds the
basic performance measures (Sharpe, Sortino, drawdown, Calmar, hit-rate) plus the
two summary containers (:class:`PerfReport`, :class:`EquitySummary`). The
anti-overfit toolkit (probabilistic / deflated Sharpe, bootstrap CIs, permutation
nulls) lives in :mod:`overfit_gauntlet.stats`, which builds on the primitives
here.

Ported from private, battle-tested research code. The default annualization is
``ann=252`` (equities); every function takes an ``ann`` parameter so crypto/daily
(365) or intraday scales can be supplied explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ANN = 252  # trading days per year (equities)


# --- basic ----------------------------------------------------------------
_VOL_FLOOR = 1e-12   # treat sub-floor dispersion as zero vol (constant series)


def _clean(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return r[np.isfinite(r)]


def sharpe(r: np.ndarray, ann: int = ANN) -> float:
    r = _clean(r)
    sd = r.std(ddof=1) if len(r) >= 5 else 0.0
    if len(r) < 5 or sd <= _VOL_FLOOR:
        return 0.0
    return float(r.mean() / sd * math.sqrt(ann))


def sortino(r: np.ndarray, ann: int = ANN) -> float:
    r = _clean(r)
    downside = r[r < 0]
    dd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if dd <= _VOL_FLOOR:
        return 0.0
    return float(r.mean() / dd * math.sqrt(ann))


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1 + _clean(r))
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min()) if len(eq) else 0.0


def calmar(r: np.ndarray, ann: int = ANN) -> float:
    mdd = abs(max_drawdown(r))
    if mdd < 1e-9:
        return 0.0
    cagr = float(np.prod(1 + _clean(r)) ** (ann / max(len(r), 1)) - 1)
    return cagr / mdd


def hit_rate(r: np.ndarray) -> float:
    r = _clean(r)
    return float((r > 0).mean()) if len(r) else 0.0


@dataclass
class PerfReport:
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    hit_rate: float
    ann_return: float
    ann_vol: float
    n: int

    def __str__(self) -> str:
        return (f"Sharpe={self.sharpe:.2f} Sortino={self.sortino:.2f} "
                f"Calmar={self.calmar:.2f} maxDD={self.max_dd:.1%} "
                f"hit={self.hit_rate:.1%} annRet={self.ann_return:.1%} "
                f"annVol={self.ann_vol:.1%} n={self.n}")


def perf_report(r: np.ndarray, ann: int = ANN) -> PerfReport:
    r = _clean(r)
    return PerfReport(
        sharpe=sharpe(r, ann), sortino=sortino(r, ann), calmar=calmar(r, ann),
        max_dd=max_drawdown(r), hit_rate=hit_rate(r),
        ann_return=float(r.mean() * ann) if len(r) else 0.0,
        ann_vol=float(r.std(ddof=1) * math.sqrt(ann)) if len(r) > 1 else 0.0,
        n=len(r),
    )


@dataclass
class EquitySummary:
    n_days: int
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_dd: float
    time_under_water: float        # fraction of days spent below a prior peak
    longest_dd_days: int
    best_day: float
    worst_day: float
    final_equity: float            # growth of $1


def equity_summary(r: np.ndarray, ann: int = ANN) -> EquitySummary:
    r = _clean(r)
    n = len(r)
    if n == 0:
        return EquitySummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0)
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    underwater = eq < peak * (1 - 1e-12)
    # longest consecutive run below a prior peak
    longest = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    total = float(eq[-1] - 1.0)
    cagr = float(eq[-1] ** (ann / max(n, 1)) - 1.0)
    return EquitySummary(
        n_days=n, total_return=total, cagr=cagr,
        ann_vol=float(r.std(ddof=1) * math.sqrt(ann)) if n > 1 else 0.0,
        sharpe=sharpe(r, ann), max_dd=max_drawdown(r),
        time_under_water=float(underwater.mean()),
        longest_dd_days=int(longest),
        best_day=float(r.max()), worst_day=float(r.min()),
        final_equity=float(eq[-1]),
    )
