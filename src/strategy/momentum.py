"""Frozen baseline B1: trend + 12-1 momentum on ETFs (rules in protocol.md section 1).

This file only answers one question: "given the bars known at the close of the
signal day, what share of the account should each asset have?" It never sees a
broker, the ledger, the kill switch or a model, and it is only ever handed
bars up to the signal day (the backtest engine slices them).

Each asset's history is a table, oldest row first, with at least:
    session    the trading day
    close      split-adjusted price close      (used for the 200-day trend test)
    tr_close   total-return close (dividends reinvested)  (momentum and volatility)

The rule, for the signal day t (the last trading day of a month):
    momentum = tr_close at the month-end 1 month before t
               / tr_close at the month-end 12 months before t  - 1        must be > 0
    trend    = close at t > average of the last 200 closes                 must hold
    weight   = (1 / 60-day volatility) / (sum of that over all eligible assets)
An asset that fails either test keeps its slot, but the slot is held in cash.
"""

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Params:
    """The strategy's numbers. The defaults ARE the frozen baseline; do not change them."""
    lookback_months: int = 12     # momentum looks back this many month-ends
    skip_months: int = 1          # and ignores the most recent this-many months
    sma_days: int = 200           # trend: close above the average of this many closes
    vol_days: int = 60            # weights: volatility of this many daily returns

    def to_dict(self) -> dict:
        return asdict(self)


def month_end_closes(history: pd.DataFrame) -> pd.Series:
    """Last total-return close of every calendar month in `history`, indexed by month."""
    months = pd.DatetimeIndex(history["session"]).to_period("M")
    return history["tr_close"].groupby(months).last()


def asset_state(history: pd.DataFrame, params: Params) -> dict:
    """Work out one asset's numbers on the last day of `history`.

    Returns a dictionary with `eligible` (enough history for every test), and if
    eligible: `momentum`, `trend_ok`, `volatility` (daily) and `on`.
    """
    state = {"eligible": False, "bars": len(history)}
    if history.empty:
        return state

    # --- momentum from month-end closes -------------------------------------------
    by_month = month_end_closes(history)
    this_month = by_month.index[-1]
    recent, old = this_month - params.skip_months, this_month - params.lookback_months
    has_momentum = recent in by_month.index and old in by_month.index

    # --- enough daily bars for the trend average and the volatility ------------------
    has_trend = len(history) >= params.sma_days
    has_vol = len(history) >= params.vol_days + 1        # N returns need N+1 closes
    if not (has_momentum and has_trend and has_vol):
        return state

    momentum = by_month[recent] / by_month[old] - 1.0
    average = history["close"].iloc[-params.sma_days:].mean()
    trend_ok = bool(history["close"].iloc[-1] > average)
    daily_returns = history["tr_close"].pct_change().iloc[-params.vol_days:]
    volatility = float(daily_returns.std(ddof=1))
    if not volatility > 0:
        # No measurable risk means the weight formula would divide by zero. Leave the asset out.
        return state

    state.update(eligible=True, momentum=float(momentum), trend_ok=trend_ok, volatility=volatility,
                 on=bool(momentum > 0 and trend_ok))
    return state


def target_weights(histories: dict, params: Params = Params()) -> tuple:
    """Turn every asset's history (up to the signal day) into target weights.

    `histories` maps symbol -> table. Returns (weights, details):
      weights  symbol -> share of the account, only for assets that are ON
      details  symbol -> the numbers behind the decision (for the rebalance log)
    Whatever the weights do not add up to (1 - sum) is meant for cash.
    """
    details = {symbol: asset_state(history, params) for symbol, history in histories.items()}
    eligible = {symbol: state for symbol, state in details.items() if state["eligible"]}

    # Every eligible asset gets a slot sized by 1 / volatility, ON or OFF.
    total_inverse_vol = sum(1.0 / state["volatility"] for state in eligible.values())
    weights = {}
    for symbol, state in eligible.items():
        slot = (1.0 / state["volatility"]) / total_inverse_vol
        state["slot"] = slot
        if state["on"]:
            weights[symbol] = slot        # an OFF asset's slot stays in cash
    return weights, details
