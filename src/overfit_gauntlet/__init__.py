"""overfit-gauntlet — an opinionated anti-overfit validation gauntlet for
trading-strategy backtests, plus a static look-ahead linter for backtest code.

The package's brand is honesty: it tries to *disprove* your edge and reports
whatever it finds. Import the public API from here::

    from overfit_gauntlet import run_gauntlet, sharpe, lint_path

or drive it from the command line with the ``gauntlet`` console script.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .metrics import (
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
from .stats import (
    block_bootstrap_sharpe_ci,
    bootstrap_pvalue_zero,
    deflated_sharpe,
    permutation_pvalue,
    probabilistic_sharpe,
)
from .validate import (
    bh_fdr,
    cost_stress,
    purged_kfold_cv,
    walk_forward_oos,
)
from .lint import Finding, lint_file, lint_path, lint_source
from .gauntlet import GauntletReport, Verdict, run_gauntlet

__all__ = [
    "__version__",
    # metrics
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "perf_report",
    "equity_summary",
    "PerfReport",
    "EquitySummary",
    # stats
    "probabilistic_sharpe",
    "deflated_sharpe",
    "block_bootstrap_sharpe_ci",
    "permutation_pvalue",
    "bootstrap_pvalue_zero",
    # validate
    "walk_forward_oos",
    "purged_kfold_cv",
    "bh_fdr",
    "cost_stress",
    # lint
    "Finding",
    "lint_source",
    "lint_file",
    "lint_path",
    # gauntlet
    "run_gauntlet",
    "GauntletReport",
    "Verdict",
]
