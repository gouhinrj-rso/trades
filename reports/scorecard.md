# Scorecard: frozen baseline B1

**Overall: PASS**

- Judged on train (2005-01-01 to 2016-12-31) and validation (2017-01-01 to 2021-12-31) only.
- The holdout (2022 onward) is **locked and was not used**. A pass here says nothing about it.
- Costs: 2 bps commission + 5 bps slippage per side.
- Sharpe: daily, annualized with sqrt(252); risk-free = Ken French data library (daily) RF (1-month Treasury bill).
- Trials counted (rows in experiments/trials.csv): 14.
- Run group: fac22b41. Rules: protocol.md. Definitions: reports/metrics_spec.md.

| Gate | Rule | Verdict | Evidence |
|---|---|---|---|
| 1. Deflated Sharpe | DSR > 0.95 | **PASS** | DSR = 0.990 with N = 14 trials (PSR against zero = 0.996; Sharpe to beat = 0.08 annualized) |
| 2. Overfitting (PBO) | PBO < 0.3 | **PASS** | PBO = 0.264 over 12870 splits of 9 configurations |
| 3. Out of sample holds up | validation Sharpe >= 50% of train Sharpe | **PASS** | validation 1.11 / train 0.50 = 224% |
| 4. Survives costs | total return above zero at 2x costs in train and in validation | **PASS** | at 2x costs: train +59.0%, validation +39.0% |
| 5. Positive expectancy | lower end of the 95% interval of expectancy (in R) above 0 | **PASS** | expectancy +1.40R, 95% interval +0.62R to +2.23R, 132 closed trades |
| 6. Beats the benchmarks | validation Sharpe above both benchmarks' validation Sharpe | **PASS** | strategy 1.11 vs SPY_buy_and_hold 0.93, 60_40_SPY_IEF 1.09 |

A gate answers INSUFFICIENT EVIDENCE when there is too little data to decide. Thresholds are in `src/reporting/scorecard.py` and are the owner's.
