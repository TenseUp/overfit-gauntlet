# overfit-gauntlet

[![CI](https://github.com/TenseUp/overfit-gauntlet/actions/workflows/ci.yml/badge.svg)](https://github.com/TenseUp/overfit-gauntlet/actions/workflows/ci.yml)

**An anti-overfit validation gauntlet for trading-strategy backtests — plus a
static lookahead linter for backtest code.**

Most backtests are overfit. A curve that goes up and to the right is the *easiest*
thing in quantitative finance to manufacture: run enough parameter sweeps, peek at
a little future data, normalize on the full sample, and almost any noise becomes a
"strategy." `overfit-gauntlet` is the prosecution. It assumes your edge is a
mirage and tries to disprove it — with published, peer-reviewed tests — and then
reports whatever it actually finds. No hype, no dashboards designed to make you
feel good. If the edge survives, you get a `VALIDATED` verdict you can defend to a
risk committee. If it doesn't, you find out here, before it costs money.

The package's brand is honesty. It will not phrase its output to encourage
cherry-picking, and it will tell you plainly what it cannot check.

---

## Install

```bash
pip install overfit-gauntlet
```

Requires Python ≥ 3.9. Dependencies: `numpy`, `scipy`.

---

## 60-second example

```python
import numpy as np
from overfit_gauntlet import run_gauntlet

# Your strategy's per-period returns (e.g. daily), as a 1-D array.
rng = np.random.default_rng(0)
returns = 0.0006 + 0.01 * rng.standard_normal(1500)   # a small, real-ish edge

report = run_gauntlet(returns, ann=252, n_trials=1)
print(report.to_markdown())
print(report.verdict)          # Verdict.VALIDATED / WEAK / NOT_VALIDATED
```

`run_gauntlet` runs the full battery — performance summary, Probabilistic and
Deflated Sharpe, block-bootstrap confidence interval, a sign-permutation null
test, a bootstrap test against zero, walk-forward out-of-sample, purged k-fold
cross-validation, and (optionally) cost stress — then renders a report table and
a plain-English verdict paragraph.

If you tested **many** candidate strategies to find this one, say so with
`n_trials=`. That switches the Deflated Sharpe benchmark to the level you'd expect
from the *best of N* pure-noise trials — which is exactly how multiple-testing
inflates apparent Sharpe.

---

## The checks

Every check has a citation. This is settled statistics, not house rules.

| Check | What it asks | Bar | Reference |
|-------|--------------|-----|-----------|
| **Probabilistic Sharpe (PSR)** | Given track-record length, skew and kurtosis, what's the probability the true Sharpe exceeds the benchmark? | PSR > 0.95 | Bailey & López de Prado (2014) |
| **Deflated Sharpe (DSR)** | PSR against the Sharpe you'd expect as the *maximum* of `n_trials` pure-noise trials — the multiple-testing correction. | DSR > 0.95 | Bailey & López de Prado (2014); Lo (2002) |
| **Sign-permutation p-value** | Shuffle the signs of returns; how often does random luck beat your Sharpe? | p < 0.05 | Permutation / Monte-Carlo null |
| **Bootstrap-vs-zero p-value** | Resample the mean return; is it distinguishable from zero? | p < 0.05 | Stationary block bootstrap |
| **Block-bootstrap Sharpe CI** | Confidence interval on Sharpe that respects autocorrelation. | CI lower bound > 0 | Politis & Romano (1994) |
| **Walk-forward OOS** | Fit on the past, measure on the *future* only, roll forward. | mean OOS Sharpe > 0.5 | Standard OOS discipline |
| **Purged k-fold CV** | k-fold with purge + embargo so train never touches test-adjacent bars. | folds agree | López de Prado, *AFML* §7 |
| **Cost stress** | Scale realistic fees up; where does the edge die? | breakeven multiple ≥ 1 | "Edges that die at realistic fees are not edges." |

**Verdict logic.** `VALIDATED` requires **all** bars to pass. `WEAK` means the
Sharpe is positive and some bars pass but not all — a maybe, not a yes.
`NOT_VALIDATED` is everything else. The bars are *documented, not tuned*: PSR > 0.95,
DSR > 0.95, p < 0.05, mean OOS Sharpe > 0.5.

> **The Deflated-Sharpe variance footgun.** Under the multiple-testing null, the
> benchmark Sharpe is the expected *maximum* of `n_trials` draws, whose scale
> depends on the variance of Sharpe estimates across trials. If you pass a
> variance of zero (or forget `n_trials`), the benchmark collapses and DSR
> silently reverts to plain PSR — flattering every strategy. The default keeps
> the per-trial variance at `~1/(n-1)` (Lo 2002) so the benchmark stays honest.
> Read the `deflated_sharpe` docstring before you trust a DSR.

---

## Lookahead linter

Statistics catch overfitting *after* you have returns. The linter catches the more
common sin *before*: code that quietly peeks at the future. It's an AST walk over
your `.py` files — no execution — that flags the usual leakage patterns.

Run it:

```bash
gauntlet lint src/ strategies/
```

| Code | Pattern | Why it leaks | Severity |
|------|---------|--------------|----------|
| **LA01** | `.shift(-n)` (negative literal) | Pulls future values into the present. | error |
| **LA02** | `.rolling(..., center=True)` | Centered window sees future bars. | error |
| **LA03** | `.fillna(method='bfill')` / `.bfill()` | Backfill copies future values backward. | error |
| **LA04** | `train_test_split(...)` without `shuffle=False` | Shuffles a time series into the past. | error |
| **LA05** | `KFold(..., shuffle=True)` / non-`TimeSeriesSplit` CV | Random folds mix future into train. | error |
| **LA06** | `.fit(` / `.fit_transform(` on a scaler/PCA *before* the split | Scaler learns from the test set. | error |
| **LA07** | Full-sample `(x - x.mean()) / x.std()` | Normalizes using stats it shouldn't know yet. | warn |
| **LA08** | `.pct_change(-n)` (negative periods) | Percent change against the future. | warn |

Each finding is `Finding(file, line, code, message)` with `severity` in
`{error, warn}`. `gauntlet lint` exits non-zero if any **error**-severity finding
is present, so it drops straight into CI.

**Honest about the linter:** it catches *common* sins, not all leakage. It is a
set of syntactic heuristics. Clean output means "none of these known patterns" —
not "provably leak-free."

---

## CLI usage

```bash
# Run the full gauntlet on a CSV of returns. Exit 0 only if VALIDATED (CI-friendly).
gauntlet run returns.csv --col ret --ann 252 --trials 40 --costs-col cost

# Lint a tree of backtest code. Exit 1 if any error-severity finding.
gauntlet lint src/ notebooks/
```

`gauntlet run` reads a CSV with one returns column (`--col`, default first column)
and an optional per-period cost column (`--costs-col`), prints the markdown report,
and sets its exit code from the verdict — `0` for `VALIDATED`, non-zero otherwise —
so a failing backtest fails your build.

---

## Claude Code skill

This repo doubles as a [Claude Code](https://claude.com/claude-code) plugin: it
ships an `overfit-gauntlet` skill that teaches Claude to run the gauntlet and the
lookahead linter on your strategies — and, crucially, to relay the verdicts
without spin (no softening `NOT VALIDATED`, no suggesting parameter tweaks until
a significance test passes).

```
/plugin marketplace add TenseUp/overfit-gauntlet
/plugin install overfit-gauntlet@overfit-gauntlet
```

Then ask Claude to "validate my backtest" or "check this strategy code for
lookahead bias" and it will run the full workflow: lint first (leaky code makes
every downstream statistic meaningless), then the statistical battery, then an
honest verdict.

---

## Honest limitations

This package tries to disprove your edge. Here is what it *cannot* do for you:

- **Garbage in, garbage out.** If your returns already contain lookahead bias, the
  statistics will faithfully validate a fantasy. Lint the code *and* audit the data
  pipeline; the two checks are complementary, not redundant.
- **It sees returns, not markets.** Capacity, market impact, borrow availability,
  liquidity, and regime change are invisible to a returns series. A `VALIDATED`
  verdict is a statement about a statistical sample, not a promise of future PnL.
- **`n_trials` is your responsibility.** The Deflated Sharpe can only correct for
  the number of trials you *disclose*. Forgotten sweeps, abandoned variants, and
  "I just eyeballed a few" all count — and none of them are observable from code.
- **Heuristic linter.** The linter flags known syntactic patterns. Novel or
  obfuscated leakage will pass clean. Absence of findings is not proof of safety.
- **Cross-sectional and path-dependent strategies** need care: these tests assume a
  single returns stream. Portfolio-level and multi-asset effects are out of scope.

If a check makes an assumption, its docstring says so. Read them.

---

## References

- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.*
  Journal of Portfolio Management.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, §7 (Purged
  and Embargoed Cross-Validation). Wiley.
- Lo, A. W. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts Journal.
- Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap.* JASA.

---

## License

MIT © 2026 Dominic Falso. See [LICENSE](LICENSE).
