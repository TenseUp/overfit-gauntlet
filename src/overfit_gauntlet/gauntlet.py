"""The gauntlet: run every anti-overfit check and render a verdict.

This is the prosecution. :func:`run_gauntlet` takes a single causal return
series and subjects it to the full battery — performance & equity summaries,
probabilistic and deflated Sharpe, block-bootstrap CIs, a sign-permutation null,
a bootstrapped zero-return p-value, walk-forward and purged-K-fold OOS, and an
optional cost-sensitivity stress — then reports a plain-English verdict.

The bars are documented, not tuned to any particular strategy:

* ``PSR  > 0.95``  — probabilistic Sharpe clears the usual significance bar.
* ``DSR  > 0.95``  — deflated Sharpe survives the multiple-testing benchmark.
* ``p    < 0.05``  — permutation p-value rejects the no-skill null.
* ``meanOOS > 0.5``— mean out-of-sample Sharpe (purged K-fold) stays economically real.

A strategy is ``VALIDATED`` only if it clears *every* bar. It is ``WEAK`` if its
Sharpe is positive and it clears at least one bar; otherwise ``NOT_VALIDATED``.
The report never phrases anything to encourage cherry-picking — passing the
gauntlet on one series is evidence, not a licence to keep searching.

The default annualization is ``ann=252`` (equities); pass ``ann`` explicitly for
other scales.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .metrics import ANN, EquitySummary, PerfReport, equity_summary, perf_report
from .stats import (
    block_bootstrap_sharpe_ci,
    bootstrap_pvalue_zero,
    deflated_sharpe,
    permutation_pvalue,
    probabilistic_sharpe,
)
from .validate import cost_stress, purged_kfold_cv, walk_forward_oos

__all__ = ["Verdict", "GauntletReport", "run_gauntlet"]

# Documented bars (NOT tuned to any strategy). See module docstring.
PSR_BAR = 0.95
DSR_BAR = 0.95
PVALUE_BAR = 0.05
MEAN_OOS_BAR = 0.5


class Verdict(Enum):
    """The gauntlet's three possible conclusions."""

    VALIDATED = "VALIDATED"
    WEAK = "WEAK"
    NOT_VALIDATED = "NOT_VALIDATED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value.replace("_", " ")


@dataclass
class GauntletReport:
    """The full result of running the gauntlet on one return series.

    ``checks`` is an ordered list of ``(name, value_str, passed_or_None)`` rows;
    ``passed`` is ``None`` for informational rows that are not pass/fail bars.
    """

    verdict: Verdict
    perf: PerfReport
    equity: EquitySummary
    psr: float
    dsr: float
    sharpe_ci: Tuple[float, float, float]
    permutation_p: float
    bootstrap_zero_p: float
    walk_forward: Tuple[List[Tuple[float, float]], float, float]
    purged_kfold: Tuple[List[Tuple[float, float]], float, float]
    cost: Optional[Tuple[List[Tuple[float, float]], float]] = None
    checks: List[Tuple[str, str, Optional[bool]]] = field(default_factory=list)
    n_trials: int = 1
    ann: int = ANN

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.VALIDATED

    # -- rendering ---------------------------------------------------------
    def to_markdown(self) -> str:
        """Render a full markdown report: header, checks table, verdict prose."""
        p = self.perf
        eq = self.equity
        lo, med, hi = self.sharpe_ci
        _, mean_wfo, std_wfo = self.walk_forward
        _, mean_pk, std_pk = self.purged_kfold

        lines: List[str] = []
        lines.append("# Overfit Gauntlet Report")
        lines.append("")
        lines.append(f"**Verdict: {self.verdict}**")
        lines.append("")

        # Performance snapshot.
        lines.append("## Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Observations | {p.n} |")
        lines.append(f"| Sharpe (annualized) | {p.sharpe:.2f} |")
        lines.append(f"| Sortino | {p.sortino:.2f} |")
        lines.append(f"| Calmar | {p.calmar:.2f} |")
        lines.append(f"| Max drawdown | {p.max_dd:.1%} |")
        lines.append(f"| Hit rate | {p.hit_rate:.1%} |")
        lines.append(f"| Ann. return | {p.ann_return:.1%} |")
        lines.append(f"| Ann. vol | {p.ann_vol:.1%} |")
        lines.append(f"| Total return (growth of $1) | {eq.final_equity:.2f}x |")
        lines.append(f"| Time under water | {eq.time_under_water:.1%} |")
        lines.append(f"| Longest drawdown (days) | {eq.longest_dd_days} |")
        lines.append("")

        # The bars.
        lines.append("## Checks")
        lines.append("")
        lines.append("| Check | Value | Bar | Result |")
        lines.append("| --- | ---: | --- | :---: |")
        for name, value, passed in self.checks:
            mark = "—" if passed is None else ("PASS" if passed else "FAIL")
            bar = _BAR_TEXT.get(name, "")
            lines.append(f"| {name} | {value} | {bar} | {mark} |")
        lines.append("")

        # Supporting detail.
        lines.append("## Robustness")
        lines.append("")
        lines.append(f"- Block-bootstrap Sharpe CI (95%): [{lo:.2f}, {hi:.2f}], median {med:.2f}")
        lines.append(f"- Walk-forward OOS Sharpe: mean {mean_wfo:.2f} (std {std_wfo:.2f})")
        lines.append(f"- Purged K-fold OOS Sharpe: mean {mean_pk:.2f} (std {std_pk:.2f})")
        if self.cost is not None:
            results, breakeven = self.cost
            grid = ", ".join(f"{m:g}x->{s:.2f}" for m, s in results)
            be = "never (survives the tested range)" if breakeven == float("inf") else f"{breakeven:.2f}x"
            lines.append(f"- Cost stress (net Sharpe by cost multiple): {grid}")
            lines.append(f"- Break-even cost multiple: {be}")
        lines.append("")

        # Plain-English verdict.
        lines.append("## Verdict")
        lines.append("")
        lines.append(self._verdict_paragraph())
        lines.append("")
        return "\n".join(lines)

    def _verdict_paragraph(self) -> str:
        failed = [name for name, _, passed in self.checks if passed is False]
        if self.verdict is Verdict.VALIDATED:
            return (
                "This series clears every bar in the gauntlet: the probabilistic and "
                "deflated Sharpe both exceed 0.95, the permutation null is rejected at "
                "p<0.05, and the mean out-of-sample Sharpe stays above 0.5. That is "
                "the strongest signal this toolkit can give — but it is evidence on "
                "*one* series, not a licence to keep searching until something passes. "
                "Re-run on fresh, untouched data before you trust it with capital."
            )
        if self.verdict is Verdict.WEAK:
            missed = ", ".join(failed) if failed else "one or more bars"
            return (
                "The Sharpe is positive and some checks pass, but the strategy does "
                f"not clear the full gauntlet (failed: {missed}). Treat this as an "
                "unproven candidate: the edge may be real but under-powered, regime-"
                "dependent, or partly luck. Gather more out-of-sample data rather than "
                "tuning parameters until the bars are met — that is exactly how "
                "backtests get overfit."
            )
        return (
            "This series does not survive the gauntlet. The evidence is consistent "
            "with no genuine edge — a Sharpe indistinguishable from luck once "
            "multiple testing, autocorrelation, and out-of-sample decay are accounted "
            "for. Do not deploy it, and resist the urge to re-parameterize until it "
            "passes: that path manufactures false discoveries."
        )


_BAR_TEXT = {
    "Probabilistic Sharpe (PSR)": "> 0.95",
    "Deflated Sharpe (DSR)": "> 0.95",
    "Permutation p-value": "< 0.05",
    "Mean OOS Sharpe (purged K-fold)": "> 0.50",
}


def run_gauntlet(
    returns: Sequence[float],
    *,
    ann: int = ANN,
    n_trials: int = 1,
    costs: Optional[Sequence[float]] = None,
    var_sr_trials: Optional[float] = None,
    n_folds: int = 5,
    purge: int = 1,
    embargo: int = 0,
    seed: int = 0,
) -> GauntletReport:
    """Run the full anti-overfit gauntlet on a causal return series.

    Args:
        returns: 1-D per-period returns (already causal — no look-ahead).
        ann: Annualization factor (252 equities, 365 daily-crypto, etc.).
        n_trials: How many strategy variants were searched to find this one.
            Feeds the deflated-Sharpe multiple-testing benchmark; ``1`` means no
            selection.
        costs: Optional per-period cost drag (same length as ``returns``). When
            given, a cost-sensitivity stress is added to the report.
        var_sr_trials: Cross-trial variance of the per-period Sharpe estimates,
            for the deflated Sharpe. If ``None``, falls back to the ~``1/(n-1)``
            sampling variance (see :func:`deflated_sharpe`).
        n_folds: Folds for walk-forward and purged K-fold CV.
        purge, embargo: Purge/embargo bands for purged K-fold CV.
        seed: RNG seed for the bootstrap/permutation resamples.

    Returns:
        A :class:`GauntletReport` with every computed statistic and a
        :class:`Verdict`.
    """
    r = np.asarray(returns, dtype=float)

    perf = perf_report(r, ann)
    equity = equity_summary(r, ann)
    psr = probabilistic_sharpe(r, ann=ann)
    dsr = deflated_sharpe(r, n_trials=n_trials, ann=ann, var_sr_trials=var_sr_trials)
    ci = block_bootstrap_sharpe_ci(r, ann=ann, seed=seed)
    perm_p = permutation_pvalue(r, ann=ann, seed=seed)
    boot_p = bootstrap_pvalue_zero(r, ann=ann, seed=seed + 7)
    wfo = walk_forward_oos(r, n_folds=n_folds, ann=ann)
    pk = purged_kfold_cv(r, n_folds=n_folds, purge=purge, embargo=embargo, ann=ann)

    cost = None
    if costs is not None:
        cost = cost_stress(r, costs, ann=ann)

    mean_oos = pk[1]

    # Evaluate the documented bars.
    psr_ok = psr > PSR_BAR
    dsr_ok = dsr > DSR_BAR
    p_ok = perm_p < PVALUE_BAR
    oos_ok = mean_oos > MEAN_OOS_BAR
    bars = [psr_ok, dsr_ok, p_ok, oos_ok]

    if all(bars):
        verdict = Verdict.VALIDATED
    elif perf.sharpe > 0 and any(bars):
        verdict = Verdict.WEAK
    else:
        verdict = Verdict.NOT_VALIDATED

    checks: List[Tuple[str, str, Optional[bool]]] = [
        ("Probabilistic Sharpe (PSR)", f"{psr:.3f}", psr_ok),
        ("Deflated Sharpe (DSR)", f"{dsr:.3f}", dsr_ok),
        ("Permutation p-value", f"{perm_p:.3f}", p_ok),
        ("Mean OOS Sharpe (purged K-fold)", f"{mean_oos:.2f}", oos_ok),
        ("Bootstrap zero-return p-value", f"{boot_p:.3f}", None),
        ("Walk-forward mean OOS Sharpe", f"{wfo[1]:.2f}", None),
    ]

    return GauntletReport(
        verdict=verdict,
        perf=perf,
        equity=equity,
        psr=psr,
        dsr=dsr,
        sharpe_ci=ci,
        permutation_p=perm_p,
        bootstrap_zero_p=boot_p,
        walk_forward=wfo,
        purged_kfold=pk,
        cost=cost,
        checks=checks,
        n_trials=n_trials,
        ann=ann,
    )
