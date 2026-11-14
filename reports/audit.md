# Audit of frozen baseline B1

Date: 2026-11-14. Run audited: group `fac22b41` (trials T0001 to T0014). Rules: `protocol.md`.

## Bottom line

**The scorecard's PASS is not evidence of an edge.** The arithmetic is right: no lookahead,
no same-bar fill and no metric bug was found, and an independent recomputation matches the
engine to 15 decimal places. But three of the six gates pass only because of how the
protocol defined them, and each fails under a fairer reading:

| Gate | As reported | Corrected reading | Why |
|---|---|---|---|
| 1. Deflated Sharpe > 0.95 | PASS, 0.990 | **FAIL, 0.78** | The hurdle was built from 9 near-identical neighbours, so it was almost zero. With the sampling error of one Sharpe as the spread and N = 19 rows, DSR is 0.78. |
| 2. PBO < 0.30 | PASS, 0.264 | **INSUFFICIENT EVIDENCE** | The baseline was not chosen from the 9 cells, so PBO does not test it. Pure noise scores below 0.30 on this test fairly often. |
| 3. Validation Sharpe >= 50% of train | PASS, 224% | **INSUFFICIENT EVIDENCE** | SPY's own ratio is 231% and 60/40's is 195%. The jump is the 2017-2021 market, not the strategy. |
| 4. Profitable at 2x costs | PASS | PASS | Holds, but any low-turnover long-only book passes it in this period. |
| 5. Expectancy interval above 0 | PASS, +0.62R to +2.23R | **FAIL, -0.13R to +0.30R** | 1R was a one-month risk; trades last about five months (median). With R matched to each trade's holding period the interval includes zero. |
| 6. Beats benchmarks (validation Sharpe) | PASS, 1.11 vs 1.09 | **FAIL** | Ahead of 60/40 by 0.02 in validation only; behind in train (0.50 vs 0.56) and overall (0.65 vs 0.72). At 2x costs it is behind in validation too (1.07 vs 1.09). |

Corrected overall reading: **FAIL** (gates 1, 5 and 6). The reported scorecard is left as it
was produced; this file sits next to it. No strategy parameter, threshold or window was changed.

The holdout was not opened. `experiments/holdout_access.csv` does not exist.

## How this was checked

- `quant-auditor` review (read-only, pytest only), standing in through a general agent
  because that agent type could not be loaded by name in this session. Verdict: NOT DISPROVEN
  on arithmetic, with the defects below.
- `experiments/audit_recheck.py`: weights and equity recomputed with plain pandas, without the
  engine, ledger or strategy code. Output: `reports/baseline/audit_recheck.txt`.
- `experiments/audit_addendum.py`: the remaining corrected numbers. Output:
  `reports/baseline/audit_addendum.txt`.
- Tests rerun by the auditor: 155 passed, real trial log untouched.

## Defects, most severe first

Severity: blocker = the result cannot be used; major = changes a gate or the meaning of a
number; minor = small or conservative effect.

### Protocol definitions (these change the verdict)

| # | Severity | Where | Defect | Reported | Corrected |
|---|---|---|---|---|---|
| 1 | Major | `protocol.md:83`, `src/research/backtest.py:173` | 1R is a one-month, one-sigma move, but the median trade lasts 105 trading days (mean 171, longest 881). Long winners are divided by a one-month risk. | Expectancy +1.396R, interval +0.616 to +2.232 | **+0.087R, interval -0.127 to +0.299** with R = entry notional x volatility x sqrt(holding days). Median R is +0.04; median trade profit is 3 dollars against a mean of 73. |
| 2 | Major | `protocol.md:132`, `src/reporting/scorecard.py:128` | Gate 6 looks at validation Sharpe only. | Strategy 1.11 vs 60/40 1.09 | Difference +0.02, 95% interval **-0.81 to +0.69**; strategy ahead in 48% of resamples. Train 0.50 vs 0.56; overall 0.65 vs 0.72. At 2x costs validation is **1.07 vs 1.09**. Beat 60/40 in 4 of 17 calendar years. CAGR 5.0% vs 8.5%. |
| 3 | Major | `protocol.md:97`, `src/research/run_baseline.py:309` | The spread of trial Sharpes comes from 9 near-identical cells (0.047 a year), so the Sharpe to beat is 0.08 and the gate reduces to "Sharpe above zero". | DSR 0.990 | **DSR 0.781** with the sampling error of one Sharpe as the spread (hurdle 0.46 a year), N = 19. Probability the true Sharpe exceeds 60/40's: **0.395**. |
| 4 | Major | `src/research/run_baseline.py:349` | The factor regression has no bond factor, although bond funds hold a large share of the weight. | Alpha +2.07% a year, t = 1.89 | **Alpha +0.78% a year, t = 0.73** with IEF added. No significant alpha either way. |
| 5 | Major | `protocol.md:129`, `src/reporting/scorecard.py:92` | A one-sided "at least 50%" gate passes whenever the later window is better. | 224% | SPY 231%, 60/40 195% over the same windows. The gate cannot tell skill from the market. |
| 6 | Major | `protocol.md:98` | PBO over the neighbourhood does not test the baseline, which was not selected from it. | PBO 0.264 | Not a test of B1. Walk-forward never picked the baseline (0 of 12 years). All 9 cells have now been seen on validation; promoting one later would be selection. |
| 7 | Major | results | Profit is concentrated and tied to two regimes. | Top 5 trades 36% of profit, best 10% of trades 80% | Without the best 13 trades expectancy is +0.37R, interval -0.07 to +0.84. 2007 and 2008 carry the comparison with 60/40; 2009 was 0.0% against +12.7%. 2022 is locked and untested. |

### Code and process

| # | Severity | Where | Defect | Effect | Status |
|---|---|---|---|---|---|
| 8 | Major | `experiments/audit_recheck.py:41` | The re-check ran the strategy on real data four times (two replications, two runs of a variant where idle cash earns the T-bill rate) without logging them. This broke the rule it was auditing. | N was 14, should have been 18 | **Fixed in the log**: four rows appended late and marked as such, plus one for the addendum run. N is now 19. DSR as defined moves from 0.990 to 0.989. |
| 9 | Major | `src/data/pipeline.py:211`, `experiments/audit_recheck.py:9` | The holdout lock is in the loader only. Any script can read `holdout_LOCKED.parquet` directly, and the factor cache holds post-2021 market returns. | No number affected; no access happened | Open. Needs the owner's decision (see below). |
| 10 | Minor | `tests/test_backtest.py:116`, `:148` | One lookahead test could not fail; another could not see a one-bar shift. Other tests did cover it. | Test strength only | **Fixed**: the first is replaced, the second now also checks the first fill after the tampering point. |
| 11 | Minor | `src/research/run_baseline.py:291` | The expectancy bootstrap resampled only the 77 months with exits, not all 204. | Interval +0.616 to +2.232 | +0.586 to +2.228 over all 204 months. No change to the gate as defined. |
| 12 | Minor | `src/research/trial_log.py:69` | `git_dirty` ignores untracked files, and reads "yes" as soon as the log itself changes. | The log cannot prove which code ran | Open. |
| 13 | Minor | `protocol.md:23` | Idle cash earns 0% before the cash fund exists (28.5% of the account on average until 2007-03). | Train Sharpe 0.50, CAGR 4.21% | Train Sharpe **0.53**, CAGR **4.44%** if idle cash earns the T-bill rate. Conservative; train still trails 60/40. |
| 14 | Minor | cost model | No minimum order size (some fills are about 1 dollar); break-even multiples extrapolate far beyond the 0x to 3x runs. | Break-even 16.7x to 32.8x | Treat as "well above 3x", not as a measured figure. 7 bps a side is plausible for these funds at this size. |
| 15 | Minor | `src/research/run_baseline.py:186` | Missing risk-free days would be counted as 0. | None | Checked: 0 of 4,280 days were missing. |
| 16 | Minor | `src/research/run_baseline.py:161` | The default trial-log path is the real file; tests avoid it by passing a temporary path. | None so far | Open. |

### Limits no test removes

- **Hindsight and survivorship.** The ten funds were picked in 2026 from funds that survived
  and grew. The rules were published years ago.
- **Model contamination.** The code and every unstated choice (the 1R definition, the cash-slot
  reading, the neighbourhood grid) came from an AI model that has seen this market history. By
  the owner's rule that makes every number here, validation included, contaminated. Only
  holdout data after about mid-2026 is clean.

## Checked and found clean

- Signal timing: momentum excludes the signal month; the 200-day average and 60-day volatility
  end on the signal day. First-eligible dates match fund listings.
- All 1,445 fills are on the session after the signal, on the first trading day of a month, at
  that day's open made worse by slippage.
- Order sizes use the signal-day close and equity.
- Signals are ratios, so later splits and dividends cannot change a past signal.
- Sharpe, CAGR, PSR, DSR, the CSCV rank direction and the Newey-West weights are correct as coded.
- `reports/data_health.md` holds dates and counts only. No 2022+ value appears in `reports/baseline/`.
- Commit order: protocol, then code with zero trial rows, then the first run.
- The unexplained `trial_log` cache file the auditor saw came from the one-off script that
  replaced the empty log's header before any run.

## Not checked

- Whether yfinance's prices are themselves right. Only gaps and impossible values are tested.
- Behaviour in 2022 to 2026: locked.

## Decisions for the owner

1. **Verdict.** The scorecard says PASS as the protocol defined it. The audit reads it as FAIL.
   Which stands?
2. **Holdout.** Recommended: keep it locked. Opening it for a strategy that trails 60/40 in
   two of three windows would spend it for little.
3. **Gate definitions.** If B1 or anything else is tested again, fix gates 1, 3, 5 and 6 in the
   protocol first (independent-trial spread for DSR, holding-matched R, benchmark comparison on
   every window with an interval). That needs a new protocol version, written before any run.
4. **Holdout lock (defect 9).** Options: move the holdout files outside the repo folder, or
   encrypt them with a key only the owner holds.
5. **Variant budget.** Whether the 8 neighbourhood cells count against the 5 variants.
