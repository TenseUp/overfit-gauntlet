---
name: overfit-gauntlet
description: Validate a trading-strategy backtest honestly. Use when the user asks to validate a strategy/backtest, check a Sharpe ratio for significance, test for overfitting or lookahead bias, or review backtest code for data leakage. Runs the overfit-gauntlet statistical battery on a returns series and/or the lookahead linter on backtest code, then reports the verdict without spin.
---

# Overfit Gauntlet

You are running an adversarial validation of a trading strategy. Your job is to
try to *disprove* the edge and report whatever you find — you are the
prosecution, not the marketing department.

## Setup

Ensure the package is available (prefer the user's environment):

```bash
python -c "import overfit_gauntlet" 2>/dev/null || pip install overfit-gauntlet
```

If pip install fails and this plugin was installed from the repo, install from
the plugin root: `pip install <plugin-root>`.

## Step 1 — Lint the strategy code for lookahead bias

If there is backtest/strategy source code available, lint it first. A leaky
backtest makes every statistic downstream meaningless, so do this before
interpreting any Sharpe.

```bash
gauntlet lint path/to/strategy_code/
```

Rules LA01–LA08 catch: negative `.shift()`/`.pct_change()`, centered rolling
windows, backfill, shuffled `train_test_split`, plain/shuffled KFold on time
series, transformers fit before the split, and full-sample normalization.
`error` findings are near-certain leakage; `warn` findings are heuristics —
read the flagged lines yourself and judge. A clean lint is necessary, never
sufficient.

If you find `error`-severity leakage, STOP and report it. Do not run the
statistical gauntlet on returns produced by leaky code — fix the leak and
regenerate the returns first.

## Step 2 — Run the statistical gauntlet on the returns

You need a 1-D series of per-period strategy returns (CSV column, array, etc.)
and two honest inputs from the user — ask if not obvious:

- `ann`: periods per year (252 equities daily, 365 crypto daily, 52 weekly, 12 monthly).
- `n_trials`: how many strategy variants/parameter sets were tried before
  settling on this one. People under-report this; when in doubt, round UP.
  `n_trials=1` is only honest for a pre-registered, never-tuned strategy.

```bash
gauntlet run returns.csv --col ret --ann 252 --trials 20
```

or in Python:

```python
from overfit_gauntlet import run_gauntlet
report = run_gauntlet(returns, ann=252, n_trials=20, costs=cost_series)
print(report.to_markdown())
```

Pass `costs=` (per-period cost drag) whenever turnover data exists — an edge
that dies at realistic fees is not an edge.

## Step 3 — Report the verdict honestly

Relay the report's verdict (VALIDATED / WEAK / NOT VALIDATED) and the failed
bars verbatim. Hard rules:

- Never soften a NOT VALIDATED verdict or suggest parameter tweaks to make the
  bars pass — re-tuning until a significance test passes is how overfitting
  happens, and it defeats the entire point of this tool.
- Never cite the in-sample Sharpe without the deflated Sharpe and permutation
  p-value next to it.
- A WEAK verdict means "gather more out-of-sample data", not "almost there".
- If the deflated Sharpe is *exactly* 0.000, flag a possibly mis-scaled
  `var_sr_trials` (degenerate benchmark) rather than declaring the edge fake —
  see the `deflated_sharpe` docstring.
- If many candidate strategies were tested, collect all their p-values and
  apply `overfit_gauntlet.validate.bh_fdr` — individual p<0.05 among dozens of
  candidates means little.

Interpretation bars (documented, not tunable): PSR > 0.95, DSR > 0.95,
permutation p < 0.05, mean purged-K-fold OOS Sharpe > 0.5.
