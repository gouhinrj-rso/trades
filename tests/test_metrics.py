"""Tests for src/reporting/metrics.py.

Every test builds a tiny series by hand and states the arithmetic in a comment,
so the expected value can be checked with a calculator. No real data is used.
"""

import json
import math

import pandas as pd
import pytest

from src.reporting import metrics


def daily(values, start="2023-01-02"):
    """A Series on consecutive calendar days starting at `start`."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


# 2020-01-01 to 2024-01-01 is 366 + 365 + 365 + 365 = 1461 days = exactly 4 x 365.25.
FOUR_YEAR_DATES = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2024-01-01"])


# ---------------------------------------------------------------- growth

def test_cagr_ten_percent_over_exactly_four_years():
    # 100 -> 146.41 over 1461 days. 365.25 / 1461 = 0.25, and 1.4641 ** 0.25 = 1.10.
    equity = pd.Series([100.0, 146.41], index=pd.to_datetime(["2020-01-01", "2024-01-01"]))
    assert metrics.cagr(equity) == pytest.approx(0.10)


def test_cagr_100_to_121_over_two_years():
    # 2021-01-01 to 2023-01-01 is 365 + 365 = 730 days (half a day short of 2 x 365.25).
    # CAGR = 1.21 ** (365.25 / 730) - 1 = 1.21 ** 0.500342 - 1 = 0.10007, i.e. 10%.
    equity = pd.Series([100.0, 121.0], index=pd.to_datetime(["2021-01-01", "2023-01-01"]))
    assert metrics.cagr(equity) == pytest.approx(1.21 ** (365.25 / 730) - 1)
    assert metrics.cagr(equity) == pytest.approx(0.10, abs=1e-3)


def test_cagr_rejects_bad_input():
    with pytest.raises(ValueError):
        metrics.cagr(daily([100.0]))                 # fewer than 2 points
    with pytest.raises(ValueError):
        metrics.cagr(daily([100.0, 0.0, 50.0]))      # a non-positive value
    with pytest.raises(ValueError):
        metrics.cagr(daily([100.0, -5.0]))


def test_total_return():
    # 121 / 100 - 1 = 0.21
    assert metrics.total_return(daily([100.0, 90.0, 121.0])) == pytest.approx(0.21)


# ---------------------------------------------------------------- drawdown

def test_max_drawdown_recovered():
    # Dates: Jan 2 .. Jan 7.  Equity: 100, 120, 90, 96, 130, 125.
    # Peak 120 on Jan 3, trough 90 on Jan 4: 90 / 120 - 1 = -0.25.
    # First close back at or above 120 is 130 on Jan 6. Jan 3 -> Jan 6 = 3 days.
    # The later dip 130 -> 125 is only -3.8%, so it is not the deepest.
    result = metrics.max_drawdown(daily([100.0, 120.0, 90.0, 96.0, 130.0, 125.0]))
    assert result["depth"] == pytest.approx(-0.25)
    assert result["peak_date"] == pd.Timestamp("2023-01-03")
    assert result["trough_date"] == pd.Timestamp("2023-01-04")
    assert result["recovery_date"] == pd.Timestamp("2023-01-06")
    assert result["duration_days"] == 3
    assert result["recovered"] is True


def test_max_drawdown_not_recovered():
    # Equity: 100, 120, 90, 100 on Jan 2 .. Jan 5. Depth 90 / 120 - 1 = -0.25.
    # Never back to 120, so duration runs from the peak (Jan 3) to the last date (Jan 5) = 2 days.
    result = metrics.max_drawdown(daily([100.0, 120.0, 90.0, 100.0]))
    assert result["depth"] == pytest.approx(-0.25)
    assert result["peak_date"] == pd.Timestamp("2023-01-03")
    assert result["trough_date"] == pd.Timestamp("2023-01-04")
    assert result["recovery_date"] is None
    assert result["duration_days"] == 2
    assert result["recovered"] is False


def test_max_drawdown_picks_the_deepest_of_two():
    # First fall: 100 -> 90 = -10%. Second fall: 110 -> 77 = -30%. The second is reported.
    result = metrics.max_drawdown(daily([100.0, 90.0, 110.0, 77.0, 120.0]))
    assert result["depth"] == pytest.approx(-0.30)
    assert result["peak_date"] == pd.Timestamp("2023-01-04")
    assert result["trough_date"] == pd.Timestamp("2023-01-05")
    assert result["recovery_date"] == pd.Timestamp("2023-01-06")


def test_max_drawdown_when_equity_never_falls():
    result = metrics.max_drawdown(daily([100.0, 101.0, 101.0, 105.0]))
    assert result == {"depth": 0.0, "peak_date": None, "trough_date": None,
                      "recovery_date": None, "duration_days": 0, "recovered": True}


def test_calmar():
    # Equity 100, 200, 150, 146.41 over exactly four years.
    # CAGR = 1.4641 ** 0.25 - 1 = 0.10. The peak is 200 and the lowest later close is
    # the last one: 146.41 / 200 - 1 = -0.26795. Calmar = 0.10 / 0.26795 = 0.37321.
    equity = pd.Series([100.0, 200.0, 150.0, 146.41], index=FOUR_YEAR_DATES)
    assert metrics.calmar(equity) == pytest.approx(0.10 / (1 - 146.41 / 200))
    assert metrics.calmar(equity) == pytest.approx(0.37321, abs=1e-5)


def test_calmar_nan_without_drawdown():
    equity = pd.Series([100.0, 146.41], index=pd.to_datetime(["2020-01-01", "2024-01-01"]))
    assert math.isnan(metrics.calmar(equity))


# ---------------------------------------------------------------- Sharpe and Sortino

# Returns used for the Sharpe tests: 0.01, 0.03, -0.01, 0.01.
#   mean = 0.04 / 4 = 0.01
#   deviations from the mean: 0, 0.02, -0.02, 0 -> squares sum to 0.0008
#   sample variance = 0.0008 / (4 - 1) = 0.000266667, std = 0.0163299
SHARPE_RETURNS = [0.01, 0.03, -0.01, 0.01]
SHARPE_STD = math.sqrt(0.0008 / 3)


def test_sharpe_per_period_by_hand():
    # 0.01 / 0.0163299 = 0.612372
    assert metrics.sharpe_per_period(daily(SHARPE_RETURNS)) == pytest.approx(0.01 / SHARPE_STD)
    assert metrics.sharpe_per_period(daily(SHARPE_RETURNS)) == pytest.approx(0.612372, abs=1e-6)


def test_sharpe_annualized_by_hand():
    # 0.612372 * sqrt(252) = 0.612372 * 15.8745 = 9.7211
    assert metrics.sharpe(daily(SHARPE_RETURNS)) == pytest.approx(9.7211, abs=1e-4)
    # A different periods_per_year only changes the multiplier: sqrt(4) = 2.
    assert metrics.sharpe(daily(SHARPE_RETURNS), periods_per_year=4) == pytest.approx(
        2 * 0.01 / SHARPE_STD)


def test_sharpe_with_float_rf():
    # rf = 0.002 every day: excess mean = 0.01 - 0.002 = 0.008; the std does not change.
    # 0.008 / 0.0163299 = 0.489898, times sqrt(252) = 7.7769
    result = metrics.sharpe(daily(SHARPE_RETURNS), rf=0.002)
    assert result == pytest.approx(0.008 / SHARPE_STD * math.sqrt(252))
    assert result == pytest.approx(7.7769, abs=1e-4)


def test_sharpe_with_series_rf_and_a_missing_day():
    # rf is 0.001 on the first three days and missing on the fourth (counts as 0).
    # excess = 0.009, 0.029, -0.011, 0.010 -> mean = 0.037 / 4 = 0.00925
    # deviations: -0.00025, 0.01975, -0.02025, 0.00075
    # squares: 0.0000000625 + 0.0003900625 + 0.0004100625 + 0.0000005625 = 0.00080075
    # std = sqrt(0.00080075 / 3) = 0.0163376; per period = 0.00925 / 0.0163376 = 0.566179
    returns = daily(SHARPE_RETURNS)
    rf = pd.Series([0.001, 0.001, 0.001], index=returns.index[:3])
    expected = 0.00925 / math.sqrt(0.00080075 / 3)
    assert metrics.sharpe_per_period(returns, rf=rf) == pytest.approx(expected)
    assert metrics.sharpe_per_period(returns, rf=rf) == pytest.approx(0.566179, abs=1e-6)
    assert metrics.sharpe(returns, rf=rf) == pytest.approx(expected * math.sqrt(252))


def test_sharpe_nan_on_constant_returns_or_too_few_points():
    # Every return the same -> standard deviation 0 -> no ratio.
    assert math.isnan(metrics.sharpe(daily([0.01] * 7)))
    assert math.isnan(metrics.sharpe_per_period(daily([0.01] * 7)))
    # One point has no sample standard deviation.
    assert math.isnan(metrics.sharpe(daily([0.01])))


def test_sortino_by_hand():
    # Returns 0.02, -0.01, 0.03, -0.02, rf = 0.
    #   mean = 0.02 / 4 = 0.005
    #   losses only: 0, -0.01, 0, -0.02 -> squares 0.0001 + 0.0004 = 0.0005
    #   mean over all 4 days = 0.000125, square root = 0.0111803
    #   0.005 / 0.0111803 = 0.447214, times sqrt(252) = 7.0993
    returns = daily([0.02, -0.01, 0.03, -0.02])
    assert metrics.sortino(returns) == pytest.approx(0.005 / math.sqrt(0.000125) * math.sqrt(252))
    assert metrics.sortino(returns) == pytest.approx(7.0993, abs=1e-4)


def test_sortino_with_rf_and_nan_without_downside():
    # rf = 0.01: excess = 0.01, -0.02, 0.02, -0.03 -> mean = -0.005
    # losses squared: 0.0004 + 0.0009 = 0.0013, / 4 = 0.000325, sqrt = 0.0180278
    returns = daily([0.02, -0.01, 0.03, -0.02])
    assert metrics.sortino(returns, rf=0.01) == pytest.approx(
        -0.005 / math.sqrt(0.000325) * math.sqrt(252))
    # No losing day -> nothing to divide by.
    assert math.isnan(metrics.sortino(daily([0.01, 0.02, 0.0])))


# ---------------------------------------------------------------- trade statistics

# Five closed trades.  P&L: +100, -50, +200, -50, +50.  1R: 100, 100, 100, 50, 50.
TRADE_PNL = pd.Series([100.0, -50.0, 200.0, -50.0, 50.0])
TRADE_RISK = pd.Series([100.0, 100.0, 100.0, 50.0, 50.0])


def test_profit_factor():
    # winners 100 + 200 + 50 = 350; losers -50 - 50 = -100; 350 / 100 = 3.5
    assert metrics.profit_factor(TRADE_PNL) == pytest.approx(3.5)


def test_profit_factor_edge_cases():
    assert metrics.profit_factor(pd.Series([10.0, 20.0])) == float("inf")   # no losers
    assert math.isnan(metrics.profit_factor(pd.Series([], dtype=float)))    # no trades
    assert metrics.profit_factor(pd.Series([-10.0, -20.0])) == 0.0          # no winners


def test_win_rate():
    # 3 winners out of 5 trades = 0.6
    assert metrics.win_rate(TRADE_PNL) == pytest.approx(0.6)
    # A trade at exactly 0 is a trade but not a win: 1 of 3.
    assert metrics.win_rate(pd.Series([5.0, 0.0, -5.0])) == pytest.approx(1 / 3)
    assert math.isnan(metrics.win_rate(pd.Series([], dtype=float)))


def test_payoff_ratio():
    # average win 350 / 3 = 116.667; average loss -100 / 2 = -50; 116.667 / 50 = 2.33333
    assert metrics.payoff_ratio(TRADE_PNL) == pytest.approx(7 / 3)
    assert math.isnan(metrics.payoff_ratio(pd.Series([10.0, 20.0])))


def test_r_multiples_and_expectancy():
    # 100/100 = 1, -50/100 = -0.5, 200/100 = 2, -50/50 = -1, 50/50 = 1
    assert list(metrics.r_multiples(TRADE_PNL, TRADE_RISK)) == pytest.approx(
        [1.0, -0.5, 2.0, -1.0, 1.0])
    # mean = (1 - 0.5 + 2 - 1 + 1) / 5 = 2.5 / 5 = 0.5
    assert metrics.expectancy_r(TRADE_PNL, TRADE_RISK) == pytest.approx(0.5)


def test_expectancy_rejects_non_positive_risk():
    with pytest.raises(ValueError):
        metrics.expectancy_r(pd.Series([10.0, -5.0]), pd.Series([100.0, 0.0]))
    with pytest.raises(ValueError):
        metrics.r_multiples(pd.Series([10.0, -5.0]), pd.Series([100.0, -1.0]))


# ---------------------------------------------------------------- activity and costs

def test_exposure():
    # (1.0 + 0.5 + 0.0 + 0.5) / 4 = 0.5
    assert metrics.exposure(daily([1.0, 0.5, 0.0, 0.5])) == pytest.approx(0.5)


def test_turnover():
    # Equity is 100 on four dates spanning 1461 days = 4 years; mean equity = 100.
    # Traded 50 + 0 + 150 + 200 = 400 dollars -> 400 / 100 = 4 times the account
    # in 4 years = 1.0 a year.
    equity = pd.Series([100.0, 100.0, 100.0, 100.0], index=FOUR_YEAR_DATES)
    traded = pd.Series([50.0, 0.0, 150.0, 200.0], index=FOUR_YEAR_DATES)
    assert metrics.turnover(traded, equity) == pytest.approx(1.0)


def test_cost_drag():
    # Both over exactly 4 years. Gross: 100 -> 146.41 = 10% a year.
    # Net: 100 -> 121.550625 = 1.05 ** 4, so 5% a year. Drag = 0.10 - 0.05 = 0.05.
    dates = pd.to_datetime(["2020-01-01", "2024-01-01"])
    gross = pd.Series([100.0, 146.41], index=dates)
    net = pd.Series([100.0, 121.550625], index=dates)
    assert metrics.cost_drag(gross, net) == pytest.approx(0.05)


def test_cost_breakeven_on_an_exact_line():
    # CAGR falls 0.03 per cost multiple from 0.09: the line is 0.09 - 0.03 x,
    # which reaches 0 at x = 0.09 / 0.03 = 3.
    assert metrics.cost_breakeven_multiple({0: .09, 1: .06, 2: .03, 3: 0}) == pytest.approx(3.0)
    # The brief's example: 0.08 - 0.01 x reaches 0 at x = 8 (beyond the tested range).
    assert metrics.cost_breakeven_multiple({0: .08, 1: .07, 2: .06, 3: .05}) == pytest.approx(8.0)


def test_cost_breakeven_least_squares_on_points_off_the_line():
    # Points (0, 0.10), (1, 0.04), (2, 0.04). mean x = 1, mean y = 0.06.
    # slope = ((-1)(0.04) + 0 + (1)(-0.02)) / (1 + 0 + 1) = -0.06 / 2 = -0.03
    # intercept = 0.06 + 0.03 * 1 = 0.09 -> crosses zero at 0.09 / 0.03 = 3.
    assert metrics.cost_breakeven_multiple({0: .10, 1: .04, 2: .04}) == pytest.approx(3.0)


def test_cost_breakeven_edge_cases():
    # Flat or rising line: costs never bring CAGR to zero.
    assert metrics.cost_breakeven_multiple({0: .05, 1: .05, 2: .05}) == float("inf")
    assert metrics.cost_breakeven_multiple({0: .05, 1: .06, 2: .07}) == float("inf")
    # Already at or below zero with no costs.
    assert metrics.cost_breakeven_multiple({0: -.01, 1: -.02, 2: -.03}) == 0.0
    assert metrics.cost_breakeven_multiple({0: 0.0, 1: -.01}) == 0.0
    with pytest.raises(ValueError):
        metrics.cost_breakeven_multiple({1: .05})


# ---------------------------------------------------------------- bad days and months

# Four days across a month boundary: Jan 30, Jan 31, Feb 1, Feb 2 (2023).
BOUNDARY_RETURNS = pd.Series(
    [-0.10, 0.10, -0.02, -0.03],
    index=pd.to_datetime(["2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02"]))


def test_worst_day():
    result = metrics.worst_day(BOUNDARY_RETURNS)
    assert result["date"] == pd.Timestamp("2023-01-30")
    assert result["value"] == pytest.approx(-0.10)


def test_monthly_returns_and_worst_month_across_a_boundary():
    # January: 0.90 * 1.10 - 1 = -0.01.   February: 0.98 * 0.97 - 1 = -0.0494.
    # The worst single DAY is in January, but the worst MONTH is February.
    by_month = metrics.monthly_returns(BOUNDARY_RETURNS)
    assert [str(month) for month in by_month.index] == ["2023-01", "2023-02"]
    assert list(by_month) == pytest.approx([-0.01, -0.0494])
    result = metrics.worst_month(BOUNDARY_RETURNS)
    assert result["month"] == "2023-02"
    assert result["value"] == pytest.approx(-0.0494)


def test_cvar_on_20_returns_uses_the_single_worst():
    # n = 20: k = ceil(0.05 * 20) = ceil(1.0) = 1 -> just the worst day, -0.10.
    returns = daily([0.01] * 10 + [-0.10] + [-0.02] * 9)
    assert metrics.cvar(returns) == pytest.approx(-0.10)


def test_cvar_on_100_returns_uses_the_worst_five():
    # Returns are -0.001, -0.002, ..., -0.100. n = 100: k = ceil(0.05 * 100) = 5.
    # Worst five: -0.100, -0.099, -0.098, -0.097, -0.096 -> mean = -0.490 / 5 = -0.098.
    returns = daily([-i / 1000 for i in range(1, 101)])
    assert metrics.cvar(returns) == pytest.approx(-0.098)
    # level 0.99: k = ceil(0.01 * 100) = 1 -> -0.100
    assert metrics.cvar(returns, level=0.99) == pytest.approx(-0.100)


# ---------------------------------------------------------------- summarize

ALWAYS_KEYS = {
    "n_days", "start", "end", "sharpe_frequency", "periods_per_year",
    "total_return", "cagr",
    "max_drawdown_depth", "max_drawdown_peak_date", "max_drawdown_trough_date",
    "max_drawdown_recovery_date", "max_drawdown_duration_days", "max_drawdown_recovered",
    "sharpe", "sharpe_per_period", "sortino", "calmar",
    "worst_day_date", "worst_day_value", "worst_month", "worst_month_value",
    "cvar_95", "monthly_returns",
}
TRADE_KEYS = {"n_trades", "profit_factor", "win_rate", "payoff_ratio", "expectancy_r"}


def summary_inputs():
    """Equity 100, 110, 99, 104.5 from Jan 30 to Feb 2 and the returns between closes."""
    dates = pd.to_datetime(["2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02"])
    equity = pd.Series([100.0, 110.0, 99.0, 104.5], index=dates)
    # 110/100 - 1 = 0.10;  99/110 - 1 = -0.10;  104.5/99 - 1 = 0.0555...
    returns = equity.pct_change().dropna()
    return returns, equity


def test_summarize_minimal_inputs():
    returns, equity = summary_inputs()
    summary = metrics.summarize(returns, equity)
    assert set(summary) == ALWAYS_KEYS
    assert summary["n_days"] == 3
    assert summary["start"] == "2023-01-31"
    assert summary["end"] == "2023-02-02"
    assert summary["sharpe_frequency"] == "daily"
    assert summary["periods_per_year"] == 252
    assert summary["total_return"] == pytest.approx(0.045)          # 104.5 / 100 - 1
    assert summary["max_drawdown_depth"] == pytest.approx(-0.10)    # 99 / 110 - 1
    assert summary["max_drawdown_peak_date"] == "2023-01-31"
    assert summary["max_drawdown_trough_date"] == "2023-02-01"
    assert summary["max_drawdown_recovery_date"] is None
    assert summary["max_drawdown_recovered"] is False
    assert summary["worst_day_date"] == "2023-02-01"
    assert summary["worst_month"] == "2023-02"
    assert summary["cvar_95"] == pytest.approx(-0.10)               # 3 returns -> k = 1
    assert summary["sharpe"] == pytest.approx(metrics.sharpe(returns))
    # Strict JSON (no NaN or Infinity allowed) must accept it.
    json.dumps(summary, allow_nan=False)


def test_summarize_with_every_optional_input():
    returns, equity = summary_inputs()
    trades = pd.DataFrame({"pnl": TRADE_PNL, "risk": TRADE_RISK})
    risk_weight = pd.Series([1.0, 0.5, 0.0, 0.5], index=equity.index)
    traded = pd.Series([100.0, 0.0, 50.0, 0.0], index=equity.index)
    summary = metrics.summarize(returns, equity, rf=0.0001, trades=trades,
                                risk_weight=risk_weight, traded_notional=traded)
    assert set(summary) == ALWAYS_KEYS | TRADE_KEYS | {"exposure", "turnover"}
    assert summary["n_trades"] == 5
    assert summary["profit_factor"] == pytest.approx(3.5)
    assert summary["win_rate"] == pytest.approx(0.6)
    assert summary["expectancy_r"] == pytest.approx(0.5)
    assert summary["exposure"] == pytest.approx(0.5)
    assert summary["sharpe"] == pytest.approx(metrics.sharpe(returns, rf=0.0001))
    # Round trip through strict JSON: every value is a plain type.
    text = json.dumps(summary, allow_nan=False)
    assert json.loads(text) == summary


def test_summarize_writes_nan_and_inf_as_none():
    # Equity only rises: no drawdown -> calmar is nan; no losing day -> sortino is nan.
    # All trades win -> profit factor is inf and payoff ratio is nan.
    equity = daily([100.0, 101.0, 103.0])
    returns = equity.pct_change().dropna()
    trades = pd.DataFrame({"pnl": [10.0, 20.0], "risk": [10.0, 10.0]})
    summary = metrics.summarize(returns, equity, trades=trades)
    assert summary["calmar"] is None
    assert summary["sortino"] is None
    assert summary["profit_factor"] is None
    assert summary["payoff_ratio"] is None
    assert summary["max_drawdown_peak_date"] is None
    json.dumps(summary, allow_nan=False)
