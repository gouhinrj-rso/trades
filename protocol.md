# Protocol: frozen baseline B1 (ETF trend + 12-1 momentum)

Written and committed on 2026-11-14 **before any backtest code for this strategy existed
and before any result was seen**. The git history is the proof: this file's first commit
comes before `src/strategy/`, `src/research/backtest.py` and every row of
`experiments/trials.csv`.

After this commit the rules below do not change. If a rule turns out to be wrong, the fix
is written up as a defect with the corrected number next to the original; the original is
never deleted.

Paper only. No orders, no broker, no LLM anywhere in the signal or order path.

---

## 1. What is being tested

One strategy, specified by the owner. Nothing in it was fitted to data by this project.

| Item | Rule |
|---|---|
| Universe | SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, VNQ. Long only. |
| Cash vehicle | SHV. If SHV fails its data-health checks, BIL. Before the cash vehicle has data, idle cash earns 0%. |
| Signal date | The last trading day of each month (NYSE calendar), at the close. Call it t. |
| Momentum | "12-1": total return from the month-end 12 months before t to the month-end 1 month before t. `P(M-1) / P(M-12) - 1`, on total-return (split and dividend adjusted) closes. Must be > 0. |
| Trend | Split-adjusted close at t above the simple average of the last 200 split-adjusted closes (t included). |
| In or out | An asset is ON only if both conditions hold. |
| Eligible | An asset is eligible on t only if it has a month-end close 12 months back, 200 closes and 61 closes (for 60 daily returns). Not eligible means not in the weights at all. |
| Weights | Among eligible assets: `raw_i = 1 / (standard deviation of the last 60 daily total returns)`, `w_i = raw_i / sum(raw)`. The weights of eligible assets sum to 1. An ON asset gets its `w_i`. An OFF asset's `w_i` goes to the cash vehicle. No leverage, no shorts. |
| Rebalance | Every month. Orders are sized from the close on t and fill at the **open of the first trading day of the next month**. Every position is reset to its target each month. |
| Sizing | Target shares = weight x account equity at the close on t / close on t. Fractional shares allowed. Sells fill first; if cash is short after the opening gap, all buys are scaled down by one common factor. |
| Prices traded | Total-return adjusted open and close, so dividends are inside the return. The trend test uses the split-adjusted (price) close, as the owner wrote "close". |
| Costs | Commission 2 bps of notional per side (the owner's existing `fee_bps_per_side` in `config/paper.yaml`) plus slippage 5 bps per side against the open. 7 bps each way in total. Both are settings in `config/paper.yaml`. The cash vehicle pays the same costs. |
| Starting capital | 10,000 paper dollars (`config/paper.yaml`). |

Two readings were possible and are fixed here:

- "12-month total return minus 1-month" is read as the standard skip-month momentum (return
  from 12 months ago to 1 month ago), not as the arithmetic difference R12 - R1.
- "equal risk weight ... else hold cash" is read as: each asset keeps its inverse-volatility
  slot, and the slot of an OFF asset is held in cash. Weights are not re-spread over the ON assets.

## 2. Windows

| Window | Dates | Use |
|---|---|---|
| Warm-up | 2003-01-01 to 2004-12-31 | History for the first signals only. No trading, no metrics. |
| Train (in sample, IS) | 2005-01-01 to 2016-12-31 | Reported. Nothing is fitted on it. |
| Validation (out of sample, OOS) | 2017-01-01 to 2021-12-31 | Reported. This is the OOS leg of every gate. |
| Holdout | 2022-01-01 to the last fetched session | **LOCKED.** |

- The strategy runs once, continuously, from the first trading day of 2005 to the last of
  2021. Train and validation numbers are cut from that one run.
- Some ETFs did not exist in 2005 (for example DBC). An asset joins when it becomes
  eligible under the rule in section 1. First-eligible dates are reported.
- **Holdout.** The holdout bars are saved in separate files. The loader refuses them unless
  the run is started with `--unlock-holdout`, and that flag appends who, when, the git
  commit and the command line to `experiments/holdout_access.csv`. Claude never passes that
  flag. Only the owner unlocks, once, after the scorecard is accepted.
- The data-health checks do read holdout bars (to find gaps and bad prices) and report
  dates and counts only. No holdout price or return is printed, plotted or used in a metric.

## 3. Benchmarks

Both run through the same engine, same windows, same costs, same starting capital.

1. **SPY buy and hold.** Bought at the open of the first trading day of 2005, never rebalanced.
2. **60/40 SPY/IEF.** Rebalanced to 60/40 at the open of the first trading day of each month.

## 4. Metrics

Definitions are in `reports/metrics_spec.md` and code in `src/reporting/metrics.py`, each
tested against a hand-computed value. Fixed choices:

- Returns are **daily**, close to close, net of all costs.
- Sharpe and Sortino are annualized with **sqrt(252)**. The risk-free rate is the **daily RF
  from the Ken French data library** (1-month Treasury bill). If that file cannot be
  fetched, RF is the cash vehicle's daily return and the report says so.
- A **trade** is one holding episode in one universe asset: from the fill that takes the
  position above zero to the fill that takes it back to zero. Monthly resizing inside the
  episode belongs to the same trade. Cash-vehicle holdings are not trades. An episode still
  open at the end of the run is closed at the last close for reporting and marked open.
- The strategy has no stop, so **1R** is defined as the first entry's notional x the 60-day
  daily volatility used for sizing x sqrt(21): a one-month, one-standard-deviation loss on
  the initial position. Expectancy in R = mean of (trade P&L after costs / its 1R).
- **Cost drag** = CAGR of the zero-cost run minus CAGR of the 1x-cost run.
- **Cost break-even multiple** = the multiple of the stated costs at which CAGR reaches
  zero, from a straight line through the 0x, 1x, 2x and 3x runs.

## 5. Robustness tests (all on 2005-2021 only)

| Test | Fixed settings |
|---|---|
| Bootstrap CI, Sharpe | Circular block bootstrap of daily excess returns, block 21 days, 2,000 resamples, seed 20261114, 95% interval. |
| Bootstrap CI, expectancy | Trades grouped by exit month; months resampled in circular blocks of 3, 2,000 resamples, same seed, 95% interval. |
| Probabilistic Sharpe (PSR) | Against a Sharpe of 0, with skewness and kurtosis of the daily returns. |
| Deflated Sharpe (DSR) | **N = the number of rows in `experiments/trials.csv`** at the time the scorecard is built, whatever kind of run each row is. The spread of trial Sharpes is taken from the baseline and neighbourhood rows at 1x cost. |
| PBO | Combinatorially symmetric cross-validation, 16 blocks (12,870 splits), on the daily returns of the 9 neighbourhood configurations, ranked by Sharpe. |
| Walk-forward | Each year from 2010 to 2021: pick the neighbourhood configuration with the best Sharpe on all earlier data, record its return in that year, and compare with the frozen baseline's return in the same year. |
| Parameter neighbourhood | Momentum lookback {9, 12, 15} months x trend average {150, 200, 250} days. Volatility window fixed at 60. 9 cells; the centre is the baseline. |
| Cost stress | The baseline rerun at 0x, 2x and 3x the stated costs. |
| Factor regression | Daily strategy excess return on SPY excess return and the Ken French daily momentum factor. Newey-West standard errors, 21 lags. Report alpha, its t-statistic and both betas. If the factor file cannot be fetched, the regression is reported as not run. |
| Regimes | Return and drawdown for 2008 and 2020 separately. **2022 is in the holdout and is not examined.** |

## 6. Variant budget and trial counting

- **The baseline is frozen. Nothing is optimized.** No result of any test below changes a
  parameter, the universe, the costs or a window.
- The owner allows **at most 5 variants**. That budget is for alternative strategies that
  could replace the baseline. **0 of 5 are used by this protocol.**
- The 8 neighbourhood cells, the 3 cost-stress runs and the 2 benchmarks are diagnostics
  declared here in advance. None of them can be promoted to replace the baseline. If the
  owner counts neighbourhood cells against the budget, the budget is already exceeded by
  this protocol and the owner should say so.
- **Every run of the engine on real data appends one row to `experiments/trials.csv`**:
  baseline, each neighbour, each cost run, each benchmark, reruns and failed runs. Rows are
  never edited or deleted. A rerun after a bug fix adds rows and so raises N.
- Planned rows for the first full run: 1 baseline + 8 neighbours + 3 cost runs + 2 benchmarks = **14**.
- Tests use made-up fixture data, never the real cache, and do not write to `trials.csv`.

## 7. Scorecard gates

`reports/scorecard.md` gives PASS, FAIL or INSUFFICIENT EVIDENCE for each gate.

| Gate | PASS when | INSUFFICIENT EVIDENCE when |
|---|---|---|
| 1. Deflated Sharpe | DSR > 0.95 on 2005-2021 | Fewer than 2 trial Sharpes to measure their spread, or fewer than 252 daily returns. |
| 2. Overfitting | PBO < 0.30 | Fewer than 4 configurations, or fewer than 16 x 21 daily returns. |
| 3. Out of sample holds up | Validation Sharpe >= 50% of train Sharpe, and train Sharpe > 0 | Either window has fewer than 252 daily returns. If train Sharpe <= 0 the gate is FAIL. |
| 4. Survives costs | Total return after 2x costs is above zero in train AND in validation | A cost run failed. |
| 5. Positive expectancy | Lower end of the 95% bootstrap interval of expectancy in R is above 0 | Fewer than 30 closed trades. |
| 6. Beats the benchmarks | Validation Sharpe above both benchmarks' validation Sharpe | A benchmark run failed, or validation has fewer than 252 daily returns. |

Overall verdict: PASS only if all six pass. Any FAIL makes it FAIL. Otherwise
INSUFFICIENT EVIDENCE. A PASS here is a pass on 2005-2021 only; it says nothing about the
holdout, which stays locked.

## 8. Data rules

- Source: yfinance (research only). History is back-adjusted as of the fetch date. Returns
  and price-to-average ratios do not change when later dividends rescale the past, so the
  signals are not affected; absolute adjusted prices are.
- A symbol that fails a health check is blocked and not cached. Checks are never loosened
  to admit it. If a universe symbol is blocked, the baseline runs without it, the exclusion
  is reported at the top of the scorecard, and the scorecard is marked "universe incomplete".
- No bar is filled, interpolated or invented.

## 9. Known limits, stated in advance

- **Hindsight in the design.** The rules (12-month momentum, 200-day average) were published
  years ago and the ten ETFs are funds that survived and became large. The universe was
  chosen in 2026 with knowledge of the whole period. No test in this protocol can remove that.
- **Model contamination.** The code and the unstated details (the 1R definition, the cash
  slot reading, the neighbourhood grid) were chosen by an AI model whose training data runs
  to about mid-2026, so it has seen this market history. Only data after that date is clean
  of it, which is about four months of the holdout.
- 60/40 and SPY are simple benchmarks; neither is a like-for-like trend benchmark.
- Fractional shares and fills exactly at the open plus fixed slippage are simplifications.
  Taxes, borrow and market impact are not modelled.

## 10. Audit

After the run, the `quant-auditor` subagent reviews the code and results assuming the
backtest is wrong: lookahead (shift errors, same-bar fills, adjusted-price leakage),
survivorship, cost realism, trial count against git history, the holdout access log, model
contamination, regime dependence and concentration in a few trades. Defects are listed by
severity with file and line and, where it can be computed, the corrected metric. Strategy
parameters are not edited in response.
