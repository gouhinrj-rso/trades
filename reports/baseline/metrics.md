# Baseline B1: metrics

Definitions: `reports/metrics_spec.md`. Rules: `protocol.md`. All returns are net of costs.

Sharpe and Sortino: daily, annualized with sqrt(252); risk-free = Ken French data library (daily) RF (1-month Treasury bill).

## Strategy

| Metric | train | validation | train_plus_validation |
|---|---|---|---|
| Days | 3021 | 1259 | 4280 |
| Total return | 64.0% | 40.5% | 130.4% |
| CAGR | 4.21% | 7.03% | 5.03% |
| Max drawdown | -7.8% | -7.2% | -7.8% |
| Drawdown length (days) | 662 | 151 | 662 |
| Sharpe | 0.50 | 1.11 | 0.65 |
| Sortino | 0.69 | 1.51 | 0.90 |
| Calmar | 0.54 | 0.98 | 0.64 |
| Closed trades | 96 | 36 | 132 |
| Win rate | 53.1% | 47.2% | 51.5% |
| Profit factor | 3.15 | 4.06 | 3.43 |
| Payoff ratio | 2.78 | 4.54 | 3.22 |
| Expectancy (R) | 1.309 | 1.627 | 1.396 |
| Exposure | 59.9% | 53.9% | 58.1% |
| Turnover (x per year, two-way) | 3.70 | 2.96 | 3.43 |
| Worst day | -2.53% | -1.83% | -2.53% |
| Worst month | -4.42% | -2.85% | -4.42% |
| CVaR 95% (daily) | -0.96% | -0.85% | -0.93% |
| Drawdown peak | 2011-04-29 | 2020-02-20 | 2011-04-29 |
| Drawdown trough | 2011-08-08 | 2020-03-18 | 2011-08-08 |
| Worst day date | 2011-08-08 | 2020-02-27 | 2011-08-08 |
| Worst month | 2010-05 | 2018-10 | 2010-05 |
| Cost drag (CAGR lost to costs) | 0.27% | 0.22% | 0.25% |
| Cost break-even multiple | 16.7 | 32.8 | 20.8 |

## Benchmarks

| Run | Window | CAGR | Max drawdown | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|---|
| Strategy | train | 4.21% | -7.8% | 0.50 | 0.69 | 0.54 |
| Strategy | validation | 7.03% | -7.2% | 1.11 | 1.51 | 0.98 |
| Strategy | train_plus_validation | 5.03% | -7.8% | 0.65 | 0.90 | 0.64 |
| SPY_buy_and_hold | train | 7.36% | -55.2% | 0.40 | 0.57 | 0.13 |
| SPY_buy_and_hold | validation | 18.34% | -33.7% | 0.93 | 1.28 | 0.54 |
| SPY_buy_and_hold | train_plus_validation | 10.48% | -55.2% | 0.55 | 0.78 | 0.19 |
| 60_40_SPY_IEF | train | 6.76% | -32.4% | 0.56 | 0.80 | 0.21 |
| 60_40_SPY_IEF | validation | 12.67% | -18.9% | 1.09 | 1.50 | 0.67 |
| 60_40_SPY_IEF | train_plus_validation | 8.46% | -32.4% | 0.72 | 1.01 | 0.26 |

## Cost stress (baseline rerun at multiples of the stated costs)

| Window | 0x CAGR | 1x CAGR | 2x CAGR | 3x CAGR | 0x total | 1x total | 2x total | 3x total |
|---|---|---|---|---|---|---|---|---|
| train | 4.48% | 4.21% | 3.94% | 3.68% | 69.1% | 64.0% | 59.0% | 54.2% |
| validation | 7.26% | 7.03% | 6.81% | 6.59% | 42.0% | 40.5% | 39.0% | 37.6% |
| train_plus_validation | 5.29% | 5.03% | 4.78% | 4.53% | 140.1% | 130.4% | 121.1% | 112.2% |

## Robustness (train + validation only)

- Sharpe 0.65, 95% block-bootstrap interval 0.24 to 1.09 (blocks of 21 days, 2000 resamples).
- Expectancy +1.396R, 95% interval +0.616R to +2.232R (132 closed trades in 77 exit months, blocks of 3 months).
- Probabilistic Sharpe Ratio against zero: 0.996.
- Deflated Sharpe Ratio: 0.990 with N = 14 trials; spread of trial Sharpes taken from 9 baseline and neighbour rows; daily skew -0.39, kurtosis 7.6.
- PBO (CSCV, 16 blocks): 0.264 over 12870 splits of 9 configurations; 8 early rows dropped to make equal blocks. With 9 near-identical configurations this is weak evidence either way.
- Walk-forward 12 years: picking the best neighbour on past data each year compounds to +74.9%; the frozen baseline to +72.1%. The pick beat the baseline in 5 years and was the baseline in 0. Detail: `reports/baseline/walk_forward.csv`.
- Factor regression (daily, Newey-West 21 lags, 4280 days): alpha +2.07% a year, t = 1.89; betas: spy_excess 0.16 (t 5.5), momentum_factor 0.15 (t 10.3); R-squared 0.32. Factor: Ken French data library (daily) (US momentum factor).

### Parameter neighbourhood: Sharpe, train_plus_validation

| Momentum lookback | trend 150 days | trend 200 days | trend 250 days |
|---|---|---|---|
| 9 months | 0.73 | 0.66 | 0.63 |
| 12 months | 0.71 | **0.65** | 0.63 |
| 15 months | 0.63 | 0.60 | 0.59 |

### Parameter neighbourhood: Sharpe, train

| Momentum lookback | trend 150 days | trend 200 days | trend 250 days |
|---|---|---|---|
| 9 months | 0.53 | 0.47 | 0.46 |
| 12 months | 0.57 | **0.50** | 0.50 |
| 15 months | 0.47 | 0.44 | 0.45 |

### Parameter neighbourhood: Sharpe, validation

| Momentum lookback | trend 150 days | trend 200 days | trend 250 days |
|---|---|---|---|
| 9 months | 1.27 | 1.23 | 1.17 |
| 12 months | 1.11 | **1.11** | 1.03 |
| 15 months | 1.09 | 1.06 | 0.97 |

The centre cell (bold) is the frozen baseline. The other cells are diagnostics and cannot replace it.

## Regimes

| Run | Year | Return | Max drawdown in the year |
|---|---|---|---|
| strategy | 2008 | 4.1% | -4.4% |
| strategy | 2020 | 5.1% | -7.2% |
| SPY_buy_and_hold | 2008 | -36.8% | -47.6% |
| SPY_buy_and_hold | 2020 | 18.3% | -33.7% |
| 60_40_SPY_IEF | 2008 | -17.9% | -28.0% |
| 60_40_SPY_IEF | 2020 | 16.2% | -18.9% |

2022 is in the locked holdout and was not examined

## Concentration of profit (closed trades)

- Total profit of closed trades: 9,672 paper dollars.
- Share from the 5 best trades: 36%; from the best 10% of trades: 80%.
- Profit without the 5 best trades: 6,171.
- By asset: GLD 1,631, SPY 1,575, QQQ 1,293, IEF 1,120, VNQ 920, EFA 901, EEM 817, TLT 651, IWM 621, DBC 144.

## When each asset first became eligible

EEM 2004-12-31, EFA 2004-12-31, IEF 2004-12-31, IWM 2004-12-31, QQQ 2004-12-31, SPY 2004-12-31, TLT 2004-12-31, VNQ 2005-09-30, GLD 2005-11-30, DBC 2007-02-28, SHV 2007-02-28.
