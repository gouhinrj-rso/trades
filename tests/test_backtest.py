"""Tests for the frozen baseline's signal (src/strategy/momentum.py) and the engine (src/research/backtest.py).

Everything here runs on made-up TEST FIXTURE prices built on real NYSE trading
days. No test reads the real data cache and no test writes to experiments/trials.csv.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import sessions as cal
from src.research.backtest import Costs, fixed_weights, prepare_bars, run_backtest
from src.strategy.momentum import Params, asset_state, target_weights

CALENDAR = "NYSE"
SESSIONS = cal.expected_sessions(CALENDAR, "2018-01-02", "2021-12-31")
NO_COSTS = Costs(commission_bps=0.0, slippage_bps=0.0)
COSTS = Costs(commission_bps=2.0, slippage_bps=5.0)


def fixture_bars(seed: int, drift: float = 0.0004, vol: float = 0.01, start="2018-01-02",
                 dividend_yield: float = 0.0) -> pd.DataFrame:
    """Made-up bars: a seeded random walk. `dividend_yield` makes adj_close differ from close."""
    days = SESSIONS[SESSIONS >= pd.Timestamp(start)]
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + drift + vol * rng.standard_normal(len(days)))
    open_ = close * (1.0 + 0.002 * rng.standard_normal(len(days)))
    # A total-return series that grows a little faster than price (dividends reinvested).
    adj_close = close * np.cumprod(np.full(len(days), 1.0 + dividend_yield / 252.0))
    adj_close = adj_close / adj_close[-1] * close[-1]          # back-adjusted: equal on the last day
    return prepare_bars(pd.DataFrame({
        "session": days, "open": open_, "close": close, "adj_close": adj_close,
        "available_at": pd.DatetimeIndex(days).tz_localize("UTC") + pd.Timedelta(hours=21)}))


@pytest.fixture(scope="module")
def market() -> dict:
    return {"AAA": fixture_bars(1, drift=0.0008), "BBB": fixture_bars(2, drift=-0.0006, vol=0.02),
            "CCC": fixture_bars(3, drift=0.0003, vol=0.005, dividend_yield=0.03),
            "LATE": fixture_bars(4, drift=0.0008, start="2019-06-03"),
            "CASHV": fixture_bars(5, drift=0.00005, vol=0.0001)}


def baseline_fn(histories):
    return target_weights(histories, Params())


def run(market, weights_fn=baseline_fn, costs=COSTS, start="2019-07-01", end="2021-12-31", **kwargs):
    kwargs.setdefault("cash_vehicle", "CASHV")
    return run_backtest(market, SESSIONS, weights_fn, costs, 10_000.0, start, end, **kwargs)


# ============================================================================
# The signal
# ============================================================================
def hand_history() -> pd.DataFrame:
    """14 month-ends of total-return closes we choose by hand, one bar per trading day."""
    days = SESSIONS[(SESSIONS >= "2018-01-02") & (SESSIONS <= "2019-02-28")]
    month_end_value = {"2018-01": 100, "2018-02": 110, "2018-03": 105, "2018-04": 108, "2018-05": 112,
                       "2018-06": 115, "2018-07": 118, "2018-08": 120, "2018-09": 119, "2018-10": 121,
                       "2018-11": 125, "2018-12": 130, "2019-01": 132, "2019-02": 90}
    values = [float(month_end_value[f"{day:%Y-%m}"]) for day in days]
    return pd.DataFrame({"session": days, "close": values, "tr_close": values})


def test_momentum_skips_the_most_recent_month():
    state = asset_state(hand_history(), Params())
    # Signal month is 2019-02. One month back = 2019-01 (132). Twelve months back = 2018-02 (110).
    # 12-1 momentum = 132 / 110 - 1 = 0.20. The crash to 90 in the last month is NOT in it.
    assert state["eligible"]
    assert state["momentum"] == pytest.approx(0.20)


def test_trend_uses_the_last_200_closes():
    history = hand_history()
    state = asset_state(history, Params())
    average = history["close"].iloc[-200:].mean()
    assert state["trend_ok"] is bool(90.0 > average) is False        # price 90 is below its average
    assert state["on"] is False                                      # momentum > 0 but trend fails


def test_volatility_is_the_std_of_the_last_60_daily_returns():
    history = fixture_bars(7)
    state = asset_state(history, Params())
    expected = history["tr_close"].pct_change().iloc[-60:].std(ddof=1)
    assert state["volatility"] == pytest.approx(expected)


@pytest.mark.parametrize("rows, eligible", [(150, False), (199, False), (290, True)])
def test_eligibility_needs_enough_history(rows, eligible):
    # 290 trading days cover 13 month-ends and 200 closes; 150 and 199 do not have 200 closes.
    assert asset_state(fixture_bars(7).iloc[:rows], Params())["eligible"] is eligible


def test_asset_without_a_month_end_twelve_months_back_is_not_eligible():
    history = hand_history()
    history = history[history["session"] >= "2018-03-01"]            # first month-end is now 2018-03
    assert len(history) >= 200
    assert asset_state(history, Params())["eligible"] is False


def test_weights_are_inverse_volatility_slots_and_off_slots_go_to_cash():
    histories = {"UP": fixture_bars(11, drift=0.002, vol=0.01), "DOWN": fixture_bars(12, drift=-0.002, vol=0.02)}
    weights, details = target_weights(histories, Params())
    up, down = details["UP"], details["DOWN"]
    assert up["on"] and not down["on"]
    # Slots: (1/vol) / sum(1/vol) over BOTH eligible assets.
    total = 1 / up["volatility"] + 1 / down["volatility"]
    assert up["slot"] == pytest.approx((1 / up["volatility"]) / total)
    assert up["slot"] + down["slot"] == pytest.approx(1.0)
    # Only the ON asset is held; DOWN's slot is left for cash, not given to UP.
    assert weights == {"UP": pytest.approx(up["slot"])}
    assert sum(weights.values()) < 1.0


def test_signal_does_not_change_when_later_bars_change():
    """Lookahead guard at the signal: bars after the signal day cannot matter."""
    history = fixture_bars(13)
    cut = 600
    before = asset_state(history.iloc[:cut], Params())
    tampered = history.copy()
    tampered.loc[tampered.index[cut:], ["close", "tr_close"]] *= 5.0
    assert asset_state(tampered.iloc[:cut], Params()) == before


# ============================================================================
# The engine: timing
# ============================================================================
def test_every_fill_is_at_the_open_of_the_first_trading_day_of_the_month(market):
    result = run(market)
    fills = result.fills
    first_days = {SESSIONS[(SESSIONS.year == y) & (SESSIONS.month == m)][0]
                  for y, m in {(d.year, d.month) for d in SESSIONS}}
    assert set(fills["fill_session"]) <= first_days
    for _, fill in fills.iterrows():
        # The signal is the trading day before; the fill is strictly later.
        position = SESSIONS.get_loc(fill["fill_session"])
        assert fill["signal_session"] == SESSIONS[position - 1]
        assert fill["signal_session"] < fill["fill_session"]
        # Filled at that day's open, made worse by 5 bps slippage - never at a close.
        bars = market[fill["symbol"]].set_index("session")
        open_price = bars.loc[fill["fill_session"], "tr_open"]
        worse = open_price * (1 + 0.0005) if fill["side"] == "buy" else open_price * (1 - 0.0005)
        assert fill["fill_price"] == pytest.approx(worse)
        assert fill["commission"] == pytest.approx(fill["notional"] * 0.0002)


def test_changing_the_future_does_not_change_the_past(market):
    """Lookahead guard at the engine: rewrite every bar from 2021 on; 2019-2020 must be identical."""
    honest = run(market)
    tampered = {symbol: table.copy() for symbol, table in market.items()}
    for table in tampered.values():
        late = table["session"] >= "2021-01-01"
        table.loc[late, ["open", "close", "tr_open", "tr_close"]] *= np.linspace(1.5, 3.0, late.sum())[:, None]
    changed = run(tampered)
    cutoff = pd.Timestamp("2020-12-31")
    pd.testing.assert_series_equal(honest.daily.loc[:cutoff, "equity"], changed.daily.loc[:cutoff, "equity"])
    past = lambda fills: fills[fills["fill_session"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(past(honest.fills), past(changed.fills))
    # And the tampering really did change something later, so the test is not vacuous.
    assert not np.isclose(honest.daily["equity"].iloc[-1], changed.daily["equity"].iloc[-1])


def test_weight_function_is_never_handed_a_bar_after_the_signal_day(market):
    seen = []

    def spying_fn(histories):
        seen.append({symbol: history["session"].iloc[-1] for symbol, history in histories.items()})
        return baseline_fn(histories)

    result = run(market, weights_fn=spying_fn)
    signal_days = sorted(set(result.rebalances["signal_session"]))
    assert len(seen) == len(signal_days)
    for last_days, signal_day in zip(seen, signal_days):
        assert set(last_days.values()) == {signal_day}


def test_no_signal_is_taken_when_its_fill_would_fall_after_the_run(market):
    result = run(market, end="2021-06-30")
    assert result.rebalances["signal_session"].max() < pd.Timestamp("2021-06-30")
    assert result.fills["fill_session"].max() <= pd.Timestamp("2021-06-30")


# ============================================================================
# The engine: arithmetic
# ============================================================================
def test_buy_and_hold_without_costs_matches_hand_arithmetic(market):
    result = run({"AAA": market["AAA"]}, weights_fn=fixed_weights({"AAA": 1.0}), costs=NO_COSTS,
                 cash_vehicle=None, rebalance="once")
    bars = market["AAA"].set_index("session")
    signal_day, first_day = pd.Timestamp("2019-06-28"), pd.Timestamp("2019-07-01")
    # Shares wanted = 10,000 / close on the signal day. They cost shares x next open.
    wanted = 10_000.0 / bars.loc[signal_day, "tr_close"]
    cost = wanted * bars.loc[first_day, "tr_open"]
    shares = wanted if cost <= 10_000.0 else wanted * 10_000.0 / cost     # scaled if the open gapped up
    cash = 10_000.0 - shares * bars.loc[first_day, "tr_open"]
    last_day = result.daily.index[-1]
    assert result.daily["equity"].iloc[-1] == pytest.approx(cash + shares * bars.loc[last_day, "tr_close"], rel=1e-9)
    assert len(result.fills) == 1                                          # bought once, never rebalanced


def test_engine_matches_an_independent_reimplementation(market):
    """Recompute a 60/40 monthly rebalance with plain arithmetic and no ledger; equity must agree."""
    symbols = ["AAA", "CCC"]
    weights = {"AAA": 0.6, "CCC": 0.4}
    result = run({s: market[s] for s in symbols}, weights_fn=fixed_weights(weights), cash_vehicle=None)

    tables = {s: market[s].set_index("session") for s in symbols}
    days = SESSIONS[(SESSIONS >= "2019-07-01") & (SESSIONS <= "2021-12-31")]
    cash, shares, equity = 10_000.0, {s: 0.0 for s in symbols}, []
    slip, fee = 0.0005, 0.0002
    for day in days:
        previous = SESSIONS[SESSIONS.get_loc(day) - 1]
        if previous.month != day.month:                                    # first trading day of a month
            value = cash + sum(shares[s] * tables[s].loc[previous, "tr_close"] for s in symbols)
            target = {s: weights[s] * value / tables[s].loc[previous, "tr_close"] for s in symbols}
            delta = {s: target[s] - shares[s] for s in symbols}
            for s in symbols:                                              # sells first
                if delta[s] < 0:
                    proceeds = -delta[s] * tables[s].loc[day, "tr_open"] * (1 - slip)
                    cash += proceeds * (1 - fee)
                    shares[s] += delta[s]
            need = sum(delta[s] * tables[s].loc[day, "tr_open"] * (1 + slip) * (1 + fee) for s in symbols if delta[s] > 0)
            scale = 1.0 if need <= cash else cash / need * (1 - 1e-12)
            for s in symbols:
                if delta[s] > 0:
                    cash -= delta[s] * scale * tables[s].loc[day, "tr_open"] * (1 + slip) * (1 + fee)
                    shares[s] += delta[s] * scale
        equity.append(cash + sum(shares[s] * tables[s].loc[day, "tr_close"] for s in symbols))
    np.testing.assert_allclose(result.daily["equity"].to_numpy(), np.array(equity), rtol=1e-9)


def test_higher_costs_never_help(market):
    final = [run(market, costs=COSTS.scaled(multiple)).daily["equity"].iloc[-1] for multiple in (0, 1, 2, 3)]
    assert final[0] > final[1] > final[2] > final[3]


def test_costs_scaled_by_zero_are_zero():
    assert COSTS.scaled(0) == Costs(0.0, 0.0)
    assert COSTS.scaled(3) == Costs(6.0, 15.0)


def test_returns_include_the_cost_of_the_first_purchases(market):
    result = run(market)
    compounded = (1 + result.returns).prod()
    assert compounded == pytest.approx(result.daily["equity"].iloc[-1] / 10_000.0)


def test_never_more_than_fully_invested_and_cash_never_negative(market):
    result = run(market)
    assert (result.daily["cash"] >= -1e-6).all()
    assert (result.daily["risk_weight"] <= 1.0 + 1e-9).all() and (result.daily["risk_weight"] >= -1e-9).all()


# ============================================================================
# The engine: universe, cash and trades
# ============================================================================
def test_an_asset_listed_late_joins_only_when_it_becomes_eligible(market):
    result = run(market)
    late = result.rebalances[result.rebalances["symbol"] == "LATE"]
    # LATE lists 2019-06-03. It needs a month-end 12 months back: first possible signal is June 2020's month-end.
    first_eligible = late[late["eligible"] == True]["signal_session"].min()       # noqa: E712
    assert first_eligible == pd.Timestamp("2020-06-30")
    assert result.fills[result.fills["symbol"] == "LATE"]["fill_session"].min() >= pd.Timestamp("2020-07-01")


def test_leftover_weight_goes_to_the_cash_vehicle_when_it_exists(market):
    result = run(market)
    cash_rows = result.rebalances[result.rebalances["symbol"] == "CASHV"]
    risk = result.rebalances[result.rebalances["symbol"] != "CASHV"].groupby("signal_session")["weight"].sum()
    together = cash_rows.set_index("signal_session")["weight"] + risk
    assert np.allclose(together.to_numpy(), 1.0)
    assert not result.fills[result.fills["is_cash_vehicle"]].empty


def test_without_a_cash_vehicle_leftover_stays_as_idle_cash(market):
    without = {symbol: table for symbol, table in market.items() if symbol != "CASHV"}
    result = run(without, cash_vehicle=None)
    assert (result.daily["cash_vehicle_value"] == 0).all()
    assert result.daily["cash"].max() > 100.0


def test_round_trips_add_up_to_the_accounts_profit(market):
    result = run(market)
    trips = result.round_trips
    assert {"symbol", "entry_session", "exit_session", "pnl", "risk", "r_multiple", "open"} <= set(trips.columns)
    assert "CASHV" not in set(trips["symbol"])
    assert (trips["exit_session"] >= trips["entry_session"]).all()
    assert (trips["risk"] > 0).all()
    # 1R = first notional x daily volatility x sqrt(21)
    first = trips.iloc[0]
    assert first["risk"] == pytest.approx(first["entry_notional"] * first["entry_volatility"] * 21 ** 0.5)
    # Profit of every episode (closed, and open ones marked at the last close) plus the cash
    # vehicle's profit equals the change in the account.
    cash_fills = result.fills[result.fills["is_cash_vehicle"]]
    cash_flow = (-cash_fills["quantity"] * cash_fills["fill_price"] - cash_fills["commission"]).sum()
    cash_profit = cash_flow + result.daily["cash_vehicle_value"].iloc[-1]
    assert trips["pnl"].sum() + cash_profit == pytest.approx(result.daily["equity"].iloc[-1] - 10_000.0, abs=1e-6)


def test_weights_above_one_are_refused(market):
    with pytest.raises(ValueError, match="more than 1"):
        run(market, weights_fn=fixed_weights({"AAA": 0.8, "BBB": 0.8}), cash_vehicle=None)
