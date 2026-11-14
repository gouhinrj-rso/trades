# Metrics specification

This file defines every performance metric the project reports. The code is in
`src/reporting/metrics.py` and each definition below is tested against a value worked
out by hand in `tests/test_metrics.py`. If this file and the code ever disagree, that is
a bug: fix one of them before any number is reported.

This file contains definitions only. It holds no results and no numbers from real data.

## 1. Fixed conventions (from `protocol.md` section 4)

- **Returns are daily, close to close, net of all costs.** A return of 0.01 means the
  account was worth 1% more at this close than at the previous close, after every trading
  cost has been taken out.
- **Annualization uses sqrt(252).** Sharpe and Sortino are computed from daily returns
  and multiplied by the square root of 252 (the usual count of trading days in a year) to
  express them per year.
- **Risk-free rate.** The "risk-free rate" is what money earns with no market risk. It is
  the **daily RF from the Ken French data library** (the 1-month Treasury bill rate,
  expressed per day). **Fallback:** if that file cannot be fetched, RF is the cash
  vehicle's daily return, and the report says so. The metric functions do not fetch
  anything; the caller passes RF in.
- **Trade** (copied from `protocol.md` section 4): A **trade** is one holding episode in
  one universe asset: from the fill that takes the position above zero to the fill that
  takes it back to zero. Monthly resizing inside the episode belongs to the same trade.
  Cash-vehicle holdings are not trades. An episode still open at the end of the run is
  closed at the last close for reporting and marked open.
- **1R** (copied from `protocol.md` section 4): The strategy has no stop, so **1R** is
  defined as the first entry's notional x the 60-day daily volatility used for sizing x
  sqrt(21): a one-month, one-standard-deviation loss on the initial position. Expectancy
  in R = mean of (trade P&L after costs / its 1R).
- **Cost drag** (same section) = CAGR of the zero-cost run minus CAGR of the 1x-cost run.
- **Cost break-even multiple** (same section) = the multiple of the stated costs at which
  CAGR reaches zero, from a straight line through the 0x, 1x, 2x and 3x runs.

## 2. Words used in this file

| Word | Meaning |
|---|---|
| `returns` | Series of daily simple returns, one per trading session, indexed by date. |
| `equity` | Series of account value in dollars at each close, indexed by date. |
| `rf` | Risk-free return per day: one number used for every day, or a dated series. With a series, values are matched to `returns` by date and **a day with no RF value counts as 0**. |
| excess return | `returns - rf`, day by day: what the strategy earned above the risk-free rate. |
| `pnl` | Profit and loss ("P&L") of each closed trade in dollars, after costs. Positive is a win, negative is a loss. |
| `risk` | Each trade's 1R in dollars (definition above). |
| notional | The dollar value of a position or of an order. |
| calendar days | Every day on the calendar, weekends included, counted from one date to another. |
| nan | "Not a number": the metric cannot be computed from this input. |
| inf | Infinity. |
| fraction | 0.25 means 25%. All rates and returns here are fractions, not percentages. |

Missing values (blank days) in any input series are dropped before a metric is computed.

## 3. Growth

### Total return — `total_return(equity)`
- **Meaning:** how much the account grew over the whole period.
- **Formula:** `last equity / first equity - 1`
- **Units:** fraction. **Frequency:** whole period.
- **Needs:** `equity` with at least 2 points.
- **Edge cases:** raises an error (ValueError) with fewer than 2 points or any value at or below zero.

### CAGR — `cagr(equity)`
- **Meaning:** compound annual growth rate: the steady yearly growth rate that would turn
  the first value into the last value over the same stretch of time.
- **Formula:** `(last / first) ** (365.25 / calendar days from first to last date) - 1`
- **Units:** fraction per year. **Frequency:** annualized.
- **Needs:** `equity` with at least 2 points.
- **Edge cases:** raises ValueError with fewer than 2 points, any value at or below zero,
  or a first and last date on the same day. Over a period much shorter than a year the
  number is an extrapolation and can be very large.

## 4. Drawdown

### Maximum drawdown — `max_drawdown(equity)`
- **Meaning:** the deepest fall of the account from a high point (the peak) to a later
  low point (the trough), with its dates.
- **Formula:** on each day, `drawdown = equity / (highest equity up to and including that day) - 1`.
  The **depth** is the most negative drawdown.
- **Returns** a dictionary:
  - `depth`: negative fraction, for example -0.25 for a 25% fall.
  - `peak_date`: the date of the high the fall started from.
  - `trough_date`: the date of the lowest point.
  - `recovery_date`: the first date after the trough on which equity is back **at or above**
    the peak value; `None` if that never happened.
  - `duration_days`: calendar days from peak to recovery; if not recovered, from peak to
    the last date in the series.
  - `recovered`: true or false.
- **Units:** depth is a fraction; duration is calendar days. **Frequency:** whole period, measured on daily closes.
- **Needs:** `equity`.
- **Edge cases:** only the deepest drawdown is reported. If two are equally deep, the
  earlier one is reported. If equity never fell: `depth` 0.0, the three dates `None`,
  `duration_days` 0, `recovered` true. An empty series raises ValueError. Falls inside a
  day are not seen, because only closing values are used.

### Calmar ratio — `calmar(equity)`
- **Meaning:** yearly growth earned per unit of worst fall.
- **Formula:** `CAGR / abs(maximum drawdown depth)`
- **Units:** a plain ratio. **Frequency:** annualized growth over a whole-period drawdown.
- **Needs:** `equity`.
- **Edge cases:** nan if there was no drawdown. Raises ValueError when CAGR does.

## 5. Return per unit of risk

### Sharpe ratio — `sharpe(returns, rf=0.0, periods_per_year=252)`
- **Meaning:** average excess return divided by how much the excess return wobbles from
  day to day. Higher means more return for the same bumpiness.
- **Formula:** `mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year)`.
  `std(..., ddof=1)` is the sample standard deviation: take each day's distance from the
  average, square it, add them up, divide by (number of days - 1), take the square root.
- **Units:** a plain ratio. **Frequency:** computed from daily returns, annualized with sqrt(252).
- **Needs:** `returns`, `rf`.
- **Edge cases:** nan with fewer than 2 returns. nan if the standard deviation is 0 (every
  day identical); anything below 0.000000000001 (1e-12) is treated as 0 so that rounding
  noise in the computer's arithmetic cannot produce a huge false ratio.

### Sharpe ratio per period — `sharpe_per_period(returns, rf=0.0)`
- **Meaning:** the same ratio per day, not annualized. This is the form the statistical
  tests in `protocol.md` section 5 work with.
- **Formula:** `mean(excess) / std(excess, ddof=1)`
- **Units:** a plain ratio. **Frequency:** daily.
- **Needs / edge cases:** same as Sharpe.

### Sortino ratio — `sortino(returns, rf=0.0, periods_per_year=252)`
- **Meaning:** like Sharpe, but only losing days count as risk; good days are not penalized.
- **Formula:** `mean(excess) / sqrt(mean(min(excess, 0) ** 2)) * sqrt(periods_per_year)`.
  The bottom part is the "downside deviation": replace every day at or above zero by 0,
  square each day, average over **all** days (not only the losing ones), take the square root.
- **Units:** a plain ratio. **Frequency:** computed from daily returns, annualized with sqrt(252).
- **Needs:** `returns`, `rf`.
- **Edge cases:** nan if no day has an excess return below zero, or if there are no returns.

## 6. Trade statistics

These use closed trades only (see the trade definition in section 1).

### Profit factor — `profit_factor(pnl)`
- **Meaning:** dollars won for every dollar lost.
- **Formula:** `sum of winning P&L / abs(sum of losing P&L)`. A win is P&L above 0, a loss
  is P&L below 0; a trade at exactly 0 is neither.
- **Units:** a plain ratio. **Frequency:** per set of trades.
- **Needs:** `pnl`.
- **Edge cases:** inf if there are winners and no losers; nan if there are no trades (or
  only trades at exactly 0); 0.0 if there are losers and no winners.

### Win rate — `win_rate(pnl)`
- **Meaning:** share of trades that made money.
- **Formula:** `number of trades with P&L > 0 / number of trades`
- **Units:** fraction. **Frequency:** per set of trades.
- **Needs:** `pnl`.
- **Edge cases:** a trade at exactly 0 counts as a trade but not as a win. nan if no trades.

### Payoff ratio — `payoff_ratio(pnl)`
- **Meaning:** how big the average win is compared with the average loss.
- **Formula:** `average winning P&L / abs(average losing P&L)`
- **Units:** a plain ratio. **Frequency:** per set of trades.
- **Needs:** `pnl`.
- **Edge cases:** nan if there are no winners or no losers.

### R multiples — `r_multiples(pnl, risk)`
- **Meaning:** each trade's result measured in units of the risk taken on it. +2R means
  the trade made twice its 1R; -1R means it lost exactly its 1R.
- **Formula:** `R_i = pnl_i / risk_i` for each trade i.
- **Units:** R (a plain ratio). **Frequency:** per trade.
- **Needs:** `pnl` and `risk`, the same length, matched by position (first with first).
- **Edge cases:** raises ValueError if any risk is 0 or negative, if the lengths differ, or
  if a value is missing.

### Expectancy in R — `expectancy_r(pnl, risk)`
- **Meaning:** what an average trade makes per unit of risk.
- **Formula:** `mean of (pnl_i / risk_i)`
- **Units:** R per trade. **Frequency:** per set of trades.
- **Needs:** `pnl`, `risk`.
- **Edge cases:** raises ValueError as R multiples does. nan if no trades.

## 7. Activity and costs

### Exposure — `exposure(risk_weight)`
- **Meaning:** on an average day, what share of the account was invested in risk assets
  (the universe assets) and not in cash.
- **Formula:** `mean of the daily risk weight`, where the risk weight is the fraction of
  equity invested in risk assets that day.
- **Units:** fraction. **Frequency:** daily average.
- **Needs:** a daily series of risk weights.
- **Edge cases:** nan if the series is empty.

### Turnover — `turnover(traded_notional, equity)`
- **Meaning:** how many times the value of the account is traded in a year.
- **Formula:** `sum of abs(dollars traded each day) / mean equity / years`, with
  `years = calendar days from the first to the last equity date / 365.25`.
- **Units:** times per year. It is **two-way**: buys and sells are both counted, so
  selling $100 of one asset to buy $100 of another is $200 traded. **Frequency:** annualized.
- **Needs:** a daily series of dollars traded, and `equity`.
- **Edge cases:** nan if equity has fewer than 2 points, covers zero days, or averages zero or less.

### Cost drag — `cost_drag(gross_equity, net_equity)`
- **Meaning:** yearly growth given up to trading costs.
- **Formula:** `CAGR(equity of the zero-cost run) - CAGR(equity of the 1x-cost run)`
- **Units:** fraction per year. **Frequency:** annualized.
- **Needs:** the equity series of both runs.
- **Edge cases:** raises ValueError when CAGR does.

### Cost break-even multiple — `cost_breakeven_multiple(cagr_by_multiple)`
- **Meaning:** how many times larger than stated the costs would have to be before the
  strategy stops growing.
- **Formula:** take the points (cost multiple, CAGR) for the 0x, 1x, 2x and 3x runs. Fit
  the least-squares straight line `CAGR = intercept + slope * multiple` ("least squares"
  means the line that makes the squared vertical misses to the points as small as possible):
  - `slope = sum((x - mean x) * (y - mean y)) / sum((x - mean x) ** 2)`
  - `intercept = mean y - slope * mean x`
  - `break-even multiple = -intercept / slope`
- **Units:** a multiple of the stated costs (3.0 means three times the stated costs). **Frequency:** whole period.
- **Needs:** a mapping from cost multiple to CAGR with at least 2 points.
- **Edge cases**, checked in this order: 0.0 if the intercept is at or below 0 (no growth
  even with zero costs); inf if the slope is 0 or positive (more cost does not reduce
  growth, so the line never reaches zero). Raises ValueError with fewer than 2 points or if
  all multiples are equal. A result above 3 lies beyond the runs that were made and is an
  extrapolation of the line.

## 8. Bad days and bad months

### Worst day — `worst_day(returns)`
- **Meaning:** the single worst daily return and when it happened.
- **Formula:** `value = smallest daily return`; `date` = its date (the first, if tied).
- **Units:** fraction. **Frequency:** daily.
- **Needs:** `returns`.
- **Edge cases:** `date` None and `value` nan if there are no returns.

### Monthly returns — `monthly_returns(returns)`
- **Meaning:** the return of each calendar month.
- **Formula:** for the daily returns dated in a month, `(1 + r_1) * (1 + r_2) * ... * (1 + r_n) - 1`.
  This is "compounding": each day's gain or loss applies to what the previous days left.
- **Units:** fraction per month. **Frequency:** monthly, indexed by month ("YYYY-MM").
- **Needs:** `returns`.
- **Edge cases:** a month with no daily return is left out. A first or last month that is
  only partly covered is reported from the days it has.

### Worst month — `worst_month(returns)`
- **Meaning:** the worst calendar month.
- **Formula:** the smallest of the monthly returns; `month` is given as "YYYY-MM".
- **Units:** fraction. **Frequency:** monthly.
- **Needs:** `returns`.
- **Edge cases:** `month` None and `value` nan if there are no returns. The first of tied months is reported.

### CVaR (expected shortfall) — `cvar(returns, level=0.95)`
- **Meaning:** the average of the worst days. "CVaR" stands for conditional value at risk;
  "expected shortfall" is another name for the same thing. At level 0.95 it answers: on
  the worst 5% of days, what was the average return?
- **Formula:** with n daily returns, `k = ceil((1 - level) * n)` (rounded up to a whole
  number) and at least 1; `CVaR = mean of the k smallest daily returns`. With 100 returns
  k is 5; with 20 returns k is 1.
- **Units:** fraction per day; negative when those days are losses. **Frequency:** daily.
- **Needs:** `returns`.
- **Edge cases:** nan if there are no returns. This is the plain historical version: it
  uses the returns that happened and assumes no shape for them.

## 9. Summary — `summarize(returns, equity, rf=0.0, trades=None, risk_weight=None, traded_notional=None)`

Returns one flat dictionary that can be written straight to a JSON file.

| Key | Source |
|---|---|
| `n_days` | number of daily returns |
| `start`, `end` | first and last date of `returns`, as "YYYY-MM-DD" |
| `sharpe_frequency` | always `"daily"` |
| `periods_per_year` | always `252` |
| `total_return`, `cagr`, `calmar` | the functions of the same name, on `equity` |
| `max_drawdown_depth`, `max_drawdown_peak_date`, `max_drawdown_trough_date`, `max_drawdown_recovery_date`, `max_drawdown_duration_days`, `max_drawdown_recovered` | the six fields of maximum drawdown |
| `sharpe`, `sharpe_per_period`, `sortino` | on `returns` and `rf`, annualized with 252 where it applies |
| `worst_day_date`, `worst_day_value` | worst day |
| `worst_month`, `worst_month_value` | worst month |
| `cvar_95` | CVaR at level 0.95 |
| `monthly_returns` | a mapping from "YYYY-MM" to that month's return |
| `n_trades`, `profit_factor`, `win_rate`, `payoff_ratio`, `expectancy_r` | only if `trades` is given (a table with columns `pnl` and `risk`, closed trades only) |
| `exposure` | only if `risk_weight` is given |
| `turnover` | only if `traded_notional` is given |

Rules:
- Dates are text in the form "YYYY-MM-DD"; months are "YYYY-MM".
- Numbers are plain decimal numbers.
- **A value that is nan or inf is written as `null`**, because the JSON format has no way
  to write either. For example a profit factor with no losing trades (inf) appears as
  `null`; `n_trades` and `win_rate` show why.
- An optional input that is not given simply leaves its keys out.
- Cost drag and the cost break-even multiple are not in the summary: they compare several
  runs, and the summary describes one run.
- Errors from CAGR (fewer than 2 equity points, a value at or below zero) are not hidden; they stop the summary.

## 10. What these metrics do not include

The statistical tests in `protocol.md` section 5 (bootstrap intervals, probabilistic and
deflated Sharpe, PBO, walk-forward, factor regression) and the scorecard gates in section 7
are not defined here. They are built on top of these metrics: the gates read the daily
Sharpe, total return and expectancy in R exactly as defined above.
