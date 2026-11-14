"""Performance metrics for the paper-trading research project.

Every function here is PURE: it takes pandas objects, returns numbers (or small
dicts), and never reads a file, writes a file or touches the network. That makes
each one easy to test against a value worked out by hand (see tests/test_metrics.py).

The written definition of every metric is in reports/metrics_spec.md and the fixed
conventions come from protocol.md section 4:

- Returns are DAILY, close to close, net of all costs.
- Sharpe and Sortino are annualized with sqrt(252).
- The risk-free rate is a daily rate (Ken French daily RF, or the cash vehicle's
  daily return as the fallback). The caller passes it in; nothing is fetched here.

Words used below:

- returns : pandas Series of daily simple returns (0.01 means +1%), indexed by date.
- equity  : pandas Series of account value at each close, indexed by date.
- rf      : the risk-free return per day. Either one float used for every day, or a
            Series of daily values matched to `returns` by date (a missing day counts as 0).
- excess  : returns minus rf, day by day.
"""

import math

import numpy as np
import pandas as pd

# Trading days in a year, used to annualize Sharpe and Sortino (protocol.md section 4).
PERIODS_PER_YEAR = 252

# Average length of a calendar year in days (three normal years and one leap year).
DAYS_PER_YEAR = 365.25

# A standard deviation smaller than this is treated as zero. Identical returns can
# produce a tiny non-zero number (like 1e-19) from floating-point rounding, and
# dividing by that would give a huge, meaningless Sharpe ratio.
ZERO_TOLERANCE = 1e-12


# ----------------------------------------------------------------------------
# Small helpers (names start with "_" because they are not metrics themselves)
# ----------------------------------------------------------------------------

def _clean(series):
    """Return the series as floats with missing values (NaN) dropped.

    Formula: none. This only tidies the input so every metric treats gaps the same way.
    """
    return pd.Series(series).astype(float).dropna()


def _check_equity(equity):
    """Clean an equity series and refuse one that cannot give a growth rate.

    Formula: none. Raises ValueError if there are fewer than 2 points or if any
    value is zero or negative (growth from a non-positive value has no meaning).
    """
    equity = _clean(equity)
    if len(equity) < 2:
        raise ValueError("equity needs at least 2 points")
    if (equity <= 0).any():
        raise ValueError("equity values must all be positive")
    return equity


def _years(index):
    """Length of a date index in years.

    Formula: years = (last date - first date in calendar days) / 365.25
    """
    calendar_days = (index[-1] - index[0]).days
    return calendar_days / DAYS_PER_YEAR


def _excess(returns, rf=0.0):
    """Daily returns minus the daily risk-free return.

    Formula: excess_t = return_t - rf_t

    If rf is a Series it is matched to the returns by date; a date with no rf
    value counts as 0. If rf is a single number it is subtracted from every day.
    """
    returns = _clean(returns)
    if isinstance(rf, pd.Series):
        rf_matched = rf.astype(float).reindex(returns.index).fillna(0.0)
        return returns - rf_matched
    return returns - float(rf)


def _date_text(date):
    """Turn a date into "YYYY-MM-DD" text, or keep None as None.

    Formula: none. Used so that summarize() returns plain strings, not Timestamps.
    """
    if date is None:
        return None
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def _plain(value):
    """Turn a number into a plain Python float; nan and inf become None.

    Formula: none. Strict JSON has no way to write "not a number" or "infinity",
    so summarize() reports those as null (None in Python).
    """
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


# ----------------------------------------------------------------------------
# Growth
# ----------------------------------------------------------------------------

def total_return(equity) -> float:
    """Growth of the account over the whole period, as a fraction.

    Formula: total return = last equity / first equity - 1

    Example: 100 growing to 121 is 0.21 (21%).
    Raises ValueError if fewer than 2 points or a non-positive value.
    """
    equity = _check_equity(equity)
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity) -> float:
    """Compound annual growth rate: the steady yearly growth that gives the same result.

    Formula: CAGR = (last / first) ** (365.25 / calendar days from first to last date) - 1

    Example: 100 growing to 146.41 over 1,461 calendar days (4 years) is 0.10 (10% a year).
    Raises ValueError if fewer than 2 points, a non-positive value, or if the first
    and last dates are the same day.
    """
    equity = _check_equity(equity)
    calendar_days = (equity.index[-1] - equity.index[0]).days
    if calendar_days <= 0:
        raise ValueError("equity must cover at least one calendar day")
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (DAYS_PER_YEAR / calendar_days) - 1.0)


# ----------------------------------------------------------------------------
# Drawdown
# ----------------------------------------------------------------------------

def max_drawdown(equity) -> dict:
    """The deepest fall from a high point (peak) to a later low point (trough).

    Formula: drawdown_t = equity_t / (highest equity up to and including day t) - 1
             depth      = the most negative drawdown_t

    Returns a dict:
      depth         negative fraction (for example -0.25), or 0.0 if equity never fell
      peak_date     date of the high the deepest fall started from
      trough_date   date of the lowest point of that fall
      recovery_date first date after the trough with equity back at or above the
                    peak value; None if that never happened
      duration_days calendar days from peak to recovery, or from peak to the last
                    date if not recovered
      recovered     True or False

    If two falls are equally deep, the earlier one is reported. If equity never
    fell, depth is 0.0, the three dates are None, duration_days is 0 and recovered is True.
    """
    equity = _clean(equity)
    if len(equity) == 0:
        raise ValueError("equity is empty")

    # Highest value seen so far on each day, and how far below it we are.
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    depth = float(drawdown.min())

    # No fall at all: nothing to date.
    if depth >= 0:
        return {"depth": 0.0, "peak_date": None, "trough_date": None,
                "recovery_date": None, "duration_days": 0, "recovered": True}

    # idxmin gives the FIRST date of the lowest drawdown, so ties go to the earlier fall.
    trough_date = drawdown.idxmin()

    # The peak is the first date, up to the trough, where equity hit the running high
    # that was in force at the trough.
    peak_value = running_peak.loc[trough_date]
    before_trough = equity.loc[:trough_date]
    peak_date = before_trough[before_trough == peak_value].index[0]

    # Recovery: first date after the trough with equity back at or above the peak value.
    after_trough = equity.loc[trough_date:].iloc[1:]
    back_above = after_trough[after_trough >= peak_value]
    if len(back_above) > 0:
        recovery_date = back_above.index[0]
        recovered = True
        end_date = recovery_date
    else:
        recovery_date = None
        recovered = False
        end_date = equity.index[-1]

    return {"depth": depth, "peak_date": peak_date, "trough_date": trough_date,
            "recovery_date": recovery_date,
            "duration_days": int((end_date - peak_date).days),
            "recovered": recovered}


def calmar(equity) -> float:
    """Yearly growth per unit of worst fall.

    Formula: Calmar = CAGR / abs(max drawdown depth)

    Example: CAGR 0.10 with a deepest drawdown of -0.25 gives 0.4.
    Returns nan if there was no drawdown (depth is 0).
    """
    depth = max_drawdown(equity)["depth"]
    if depth == 0:
        return float("nan")
    return float(cagr(equity) / abs(depth))


# ----------------------------------------------------------------------------
# Risk-adjusted return
# ----------------------------------------------------------------------------

def sharpe_per_period(returns, rf=0.0) -> float:
    """Sharpe ratio per day, NOT annualized: average excess return per unit of wobble.

    Formula: mean(excess) / std(excess, ddof=1)
             where excess = returns - rf, and std with ddof=1 is the sample standard
             deviation (divide the sum of squared deviations by n - 1, then square root).

    Returns nan if there are fewer than 2 returns or the standard deviation is 0
    (anything below 1e-12 counts as 0).
    """
    excess = _excess(returns, rf)
    if len(excess) < 2:
        return float("nan")
    spread = excess.std(ddof=1)
    if not spread > ZERO_TOLERANCE:
        return float("nan")
    return float(excess.mean() / spread)


def sharpe(returns, rf=0.0, periods_per_year=PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe ratio.

    Formula: mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year)
             with periods_per_year = 252 for daily returns.

    Returns nan if there are fewer than 2 returns or the standard deviation is 0.
    """
    return float(sharpe_per_period(returns, rf) * math.sqrt(periods_per_year))


def sortino(returns, rf=0.0, periods_per_year=PERIODS_PER_YEAR) -> float:
    """Annualized Sortino ratio: like Sharpe, but only losing days count as risk.

    Formula: mean(excess) / sqrt(mean(min(excess, 0) ** 2)) * sqrt(periods_per_year)

    The bottom part is the "downside deviation": replace every day at or above zero
    with 0, square, average over ALL days (not just the losing ones), square root.

    Returns nan if there are no returns or no day with excess below zero.
    """
    excess = _excess(returns, rf)
    if len(excess) == 0:
        return float("nan")
    # Keep the losses, turn every gain into 0.
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = math.sqrt(float(np.mean(downside ** 2)))
    if not downside_deviation > ZERO_TOLERANCE:
        return float("nan")
    return float(excess.mean() / downside_deviation * math.sqrt(periods_per_year))


# ----------------------------------------------------------------------------
# Trade statistics (pnl = profit and loss in dollars of each closed trade)
# ----------------------------------------------------------------------------

def profit_factor(pnl) -> float:
    """Dollars won for every dollar lost.

    Formula: sum of winning trade P&L / abs(sum of losing trade P&L)

    A win is P&L > 0, a loss is P&L < 0; a trade at exactly 0 is neither.
    Returns nan if there are no trades or no winners and no losers, inf if there
    are winners and no losers, and 0.0 if there are losers and no winners.
    """
    pnl = _clean(pnl)
    wins = float(pnl[pnl > 0].sum())
    losses = float(pnl[pnl < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / abs(losses)


def win_rate(pnl) -> float:
    """Share of trades that made money.

    Formula: number of trades with P&L > 0 / number of trades

    A trade at exactly 0 counts as a trade but not as a win. Returns nan if no trades.
    """
    pnl = _clean(pnl)
    if len(pnl) == 0:
        return float("nan")
    return float((pnl > 0).sum() / len(pnl))


def payoff_ratio(pnl) -> float:
    """Size of the average win compared with the average loss.

    Formula: average winning P&L / abs(average losing P&L)

    Returns nan if there are no winners or no losers.
    """
    pnl = _clean(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def r_multiples(pnl, risk) -> pd.Series:
    """Each trade's result measured in units of the risk taken on it ("R").

    Formula: R_i = pnl_i / risk_i

    `risk` is each trade's 1R in dollars (protocol.md section 4). The two inputs are
    matched by position: first with first, second with second. The result keeps the
    index of `pnl`. Raises ValueError if the lengths differ, a value is missing, or
    any risk is zero or negative.
    """
    pnl = pd.Series(pnl).astype(float)
    risk_values = np.asarray(pd.Series(risk).astype(float))
    if len(pnl) != len(risk_values):
        raise ValueError("pnl and risk must have the same length")
    if pnl.isna().any() or np.isnan(risk_values).any():
        raise ValueError("pnl and risk must not contain missing values")
    if (risk_values <= 0).any():
        raise ValueError("every risk must be greater than 0")
    return pd.Series(pnl.to_numpy() / risk_values, index=pnl.index)


def expectancy_r(pnl, risk) -> float:
    """Average trade result in R: what one trade is expected to make per unit of risk.

    Formula: mean of (pnl_i / risk_i)

    Raises ValueError if any risk is zero or negative. Returns nan if no trades.
    """
    multiples = r_multiples(pnl, risk)
    if len(multiples) == 0:
        return float("nan")
    return float(multiples.mean())


# ----------------------------------------------------------------------------
# Activity and costs
# ----------------------------------------------------------------------------

def exposure(risk_weight) -> float:
    """Average share of the account invested in risk assets (not in cash).

    Formula: mean of the daily risk weight (0.6 means 60% invested that day)

    Returns nan if the series is empty.
    """
    risk_weight = _clean(risk_weight)
    if len(risk_weight) == 0:
        return float("nan")
    return float(risk_weight.mean())


def turnover(traded_notional, equity) -> float:
    """How many times the account's value is traded in a year (buys plus sells).

    Formula: sum of abs(traded dollars per day) / mean equity / years
             where years = calendar days from first to last equity date / 365.25

    "Two-way" means buys and sells are both counted: selling $100 of one asset and
    buying $100 of another is $200 traded. Returns nan if equity has fewer than 2
    points, covers zero days, or averages to zero or less.
    """
    equity = _clean(equity)
    traded = _clean(traded_notional)
    if len(equity) < 2:
        return float("nan")
    years = _years(equity.index)
    mean_equity = float(equity.mean())
    if years <= 0 or mean_equity <= 0:
        return float("nan")
    return float(traded.abs().sum() / mean_equity / years)


def cost_drag(gross_equity, net_equity) -> float:
    """Yearly growth given up to trading costs.

    Formula: CAGR(equity of the zero-cost run) - CAGR(equity of the run with costs)

    Example: 10% a year before costs and 8% after is a drag of 0.02.
    """
    return float(cagr(gross_equity) - cagr(net_equity))


def cost_breakeven_multiple(cagr_by_multiple: dict) -> float:
    """How many times the stated costs it takes to wipe out the growth.

    Input: {cost multiple: CAGR}, for example {0: 0.08, 1: 0.07, 2: 0.06, 3: 0.05}.

    Formula: fit the least-squares straight line  CAGR = intercept + slope * multiple
               slope     = sum((x - mean x) * (y - mean y)) / sum((x - mean x) ** 2)
               intercept = mean y - slope * mean x
             break-even multiple = -intercept / slope   (where the line reaches CAGR = 0)

    Checked in this order:
      1. intercept <= 0  -> 0.0  (no growth even at zero cost)
      2. slope >= 0      -> inf  (more cost does not lower growth, so no break-even)
    Raises ValueError with fewer than 2 points or if every multiple is the same.
    """
    if len(cagr_by_multiple) < 2:
        raise ValueError("need at least 2 (multiple, CAGR) points")
    x = np.array([float(key) for key in cagr_by_multiple.keys()])
    y = np.array([float(value) for value in cagr_by_multiple.values()])

    x_spread = float(np.sum((x - x.mean()) ** 2))
    if x_spread == 0:
        raise ValueError("the cost multiples must not all be the same")
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / x_spread)
    intercept = float(y.mean() - slope * x.mean())

    if intercept <= 0:
        return 0.0
    if slope >= 0:
        return float("inf")
    return -intercept / slope


# ----------------------------------------------------------------------------
# Bad days and bad months
# ----------------------------------------------------------------------------

def worst_day(returns) -> dict:
    """The single worst daily return.

    Formula: value = min(returns); date = the day it happened (the first, if tied)

    Returns {"date": ..., "value": ...}; both None/nan if there are no returns.
    """
    returns = _clean(returns)
    if len(returns) == 0:
        return {"date": None, "value": float("nan")}
    date = returns.idxmin()
    return {"date": date, "value": float(returns.loc[date])}


def monthly_returns(returns) -> pd.Series:
    """Return of each calendar month, from the daily returns inside it.

    Formula: month return = (1 + r_1) * (1 + r_2) * ... * (1 + r_n) - 1
             over the n daily returns dated in that month (this is "compounding").

    The result is indexed by month period (prints as "2023-01"). Months with no
    daily return are left out.
    """
    returns = _clean(returns)
    months = pd.DatetimeIndex(returns.index).to_period("M")
    # Group the days by month, multiply the growth factors together, subtract 1.
    return (1.0 + returns).groupby(months).prod() - 1.0


def worst_month(returns) -> dict:
    """The worst calendar month.

    Formula: value = min(monthly_returns(returns)); month = that month as "YYYY-MM"

    Returns {"month": ..., "value": ...}; both None/nan if there are no returns.
    """
    by_month = monthly_returns(returns)
    if len(by_month) == 0:
        return {"month": None, "value": float("nan")}
    month = by_month.idxmin()
    return {"month": str(month), "value": float(by_month.loc[month])}


def cvar(returns, level=0.95) -> float:
    """Expected shortfall (CVaR): the average of the worst days.

    Formula: k    = ceil((1 - level) * n), but at least 1   (n = number of returns)
             CVaR = mean of the k smallest daily returns

    At level 0.95 this is the average of the worst 5% of days: with 100 returns
    k = 5, with 20 returns k = 1. It is a negative number when those days are losses.
    Returns nan if there are no returns.
    """
    returns = _clean(returns)
    n = len(returns)
    if n == 0:
        return float("nan")
    # Rounding first matters: in floating point (1 - 0.95) * 100 is 5.000000000000004,
    # and ceil of that would wrongly give 6.
    k = max(1, math.ceil(round((1.0 - level) * n, 9)))
    worst = returns.sort_values().iloc[:k]
    return float(worst.mean())


# ----------------------------------------------------------------------------
# Everything in one dict
# ----------------------------------------------------------------------------

def summarize(returns, equity, rf=0.0, trades=None, risk_weight=None,
              traded_notional=None) -> dict:
    """Every metric its inputs allow, as one flat dict that json.dumps can write.

    Formula: none of its own; each value comes from the function of the same name.

    Always present:
      n_days, start, end, sharpe_frequency ("daily"), periods_per_year (252),
      total_return, cagr, max_drawdown_depth, max_drawdown_peak_date,
      max_drawdown_trough_date, max_drawdown_recovery_date, max_drawdown_duration_days,
      max_drawdown_recovered, sharpe, sharpe_per_period, sortino, calmar,
      worst_day_date, worst_day_value, worst_month, worst_month_value, cvar_95,
      monthly_returns ({"YYYY-MM": value})
    Only if `trades` is given (DataFrame with columns pnl and risk, closed trades only):
      n_trades, profit_factor, win_rate, payoff_ratio, expectancy_r
    Only if `risk_weight` is given:      exposure
    Only if `traded_notional` is given:  turnover

    Dates are "YYYY-MM-DD" text. Numbers are plain Python floats. A value that is
    nan or inf is written as None (JSON null), because strict JSON has neither.
    n_days, start and end describe the `returns` series.
    """
    returns = _clean(returns)
    drawdown = max_drawdown(equity)
    bad_day = worst_day(returns)
    bad_month = worst_month(returns)

    summary = {
        "n_days": int(len(returns)),
        "start": _date_text(returns.index[0]) if len(returns) else None,
        "end": _date_text(returns.index[-1]) if len(returns) else None,
        "sharpe_frequency": "daily",
        "periods_per_year": PERIODS_PER_YEAR,
        "total_return": _plain(total_return(equity)),
        "cagr": _plain(cagr(equity)),
        "max_drawdown_depth": _plain(drawdown["depth"]),
        "max_drawdown_peak_date": _date_text(drawdown["peak_date"]),
        "max_drawdown_trough_date": _date_text(drawdown["trough_date"]),
        "max_drawdown_recovery_date": _date_text(drawdown["recovery_date"]),
        "max_drawdown_duration_days": int(drawdown["duration_days"]),
        "max_drawdown_recovered": bool(drawdown["recovered"]),
        "sharpe": _plain(sharpe(returns, rf, PERIODS_PER_YEAR)),
        "sharpe_per_period": _plain(sharpe_per_period(returns, rf)),
        "sortino": _plain(sortino(returns, rf, PERIODS_PER_YEAR)),
        "calmar": _plain(calmar(equity)),
        "worst_day_date": _date_text(bad_day["date"]),
        "worst_day_value": _plain(bad_day["value"]),
        "worst_month": bad_month["month"],
        "worst_month_value": _plain(bad_month["value"]),
        "cvar_95": _plain(cvar(returns, 0.95)),
        "monthly_returns": {str(month): _plain(value)
                            for month, value in monthly_returns(returns).items()},
    }

    # Trade statistics need the list of closed trades.
    if trades is not None:
        summary["n_trades"] = int(len(trades))
        summary["profit_factor"] = _plain(profit_factor(trades["pnl"]))
        summary["win_rate"] = _plain(win_rate(trades["pnl"]))
        summary["payoff_ratio"] = _plain(payoff_ratio(trades["pnl"]))
        summary["expectancy_r"] = _plain(expectancy_r(trades["pnl"], trades["risk"]))

    if risk_weight is not None:
        summary["exposure"] = _plain(exposure(risk_weight))

    if traded_notional is not None:
        summary["turnover"] = _plain(turnover(traded_notional, equity))

    return summary
