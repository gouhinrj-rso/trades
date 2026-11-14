"""The lookahead guard.

Lookahead means a backtest quietly using information from the future. It is
the most common way a backtest looks good and then fails. Two rules stop it:

    Rule 1. A signal made at bar t may use only bars up to and including t.
    Rule 2. An order from that signal fills at the OPEN of bar t+1, never at
            bar t (whose close the signal has already seen).

This file enforces both:

    visible_bars(bars, at)        the bars that were known at a moment in time
    run_signals(bars, signal_fn)  calls the strategy once per bar and hands it
                                  ONLY bars up to that bar - the future rows
                                  are not in the table it receives
    next_open_fills(bars, sigs)   pairs each signal with the next bar's open
    assert_no_lookahead(...)      a test helper: cut the data short and check
                                  that earlier signals do not change

A strategy is a plain function: it gets a table of past bars and returns one
number (the position it wants, for example 1.0 or 0.0). It never gets a broker.
"""

import pandas as pd


class LookaheadError(AssertionError):
    """A signal changed when future bars were removed, so it was using them."""


def visible_bars(bars: pd.DataFrame, at) -> pd.DataFrame:
    """Return only the bars that had already closed at the moment `at` (UTC)."""
    at = pd.Timestamp(at)
    if at.tzinfo is None:
        raise ValueError("`at` needs a time zone (use UTC); a bare time is ambiguous")
    known = pd.to_datetime(bars["available_at"], utc=True) <= at.tz_convert("UTC")
    return bars[known]


def run_signals(bars: pd.DataFrame, signal_fn) -> pd.Series:
    """Call `signal_fn` once per bar, giving it only the bars up to that bar.

    Returns a Series of signals indexed by session. The bars must be one symbol,
    in time order, with no repeated session.
    """
    sessions = pd.DatetimeIndex(bars["session"])
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise ValueError("bars must be in time order with no repeated session")
    ordered = bars.reset_index(drop=True)
    signals = {}
    for position in range(len(ordered)):
        # .iloc[: position + 1] is rows 0..position. Row position+1 and later are
        # not passed in at all, so the strategy cannot read them. .copy() stops it
        # from changing our table.
        history = ordered.iloc[: position + 1].copy()
        signals[ordered["session"].iloc[position]] = signal_fn(history)
    return pd.Series(signals, name="signal", dtype="float64").rename_axis("session")


def next_open_fills(bars: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """Pair every signal with the bar it would fill on: the NEXT session's open.

    Returns one row per signal that has a next bar, with columns:
    signal_session, signal, fill_session, fill_price. The last signal has no
    next bar yet, so it produces no fill.
    """
    ordered = bars.sort_values("session").reset_index(drop=True)
    table = pd.DataFrame({
        "signal_session": ordered["session"],
        "signal": ordered["session"].map(signals),
        "fill_session": ordered["session"].shift(-1),
        "fill_price": ordered["open"].shift(-1),
    })
    table = table.dropna(subset=["signal", "fill_session"]).reset_index(drop=True)
    # A fill must come strictly after the signal that caused it.
    if not (table["fill_session"] > table["signal_session"]).all():
        raise LookaheadError("a fill is not later than its signal")
    return table


def assert_no_lookahead(bars: pd.DataFrame, signal_fn, signals: pd.Series, checks: int = 5) -> None:
    """Prove that `signals` did not use future bars.

    For a few cut points, drop every bar after the cut and compute the signals
    again from scratch. The signals up to the cut must be exactly the same. If
    one differs, the first run was reading bars it should not have seen.

    `signals` is whatever the code under test produced (it may have been
    computed in one pass over the whole table, which is where lookahead hides).
    """
    ordered = bars.sort_values("session").reset_index(drop=True)
    count = len(ordered)
    if count < 2:
        return
    # Evenly spaced cut points, always including "all but the last bar".
    cuts = sorted({max(1, (count * step) // (checks + 1)) for step in range(1, checks + 1)} | {count - 1})
    for cut in cuts:
        shortened = ordered.iloc[:cut]
        again = run_signals(shortened, signal_fn)
        before = signals.reindex(again.index)
        different = ~((before == again) | (before.isna() & again.isna()))
        if different.any():
            first = different[different].index[0]
            raise LookaheadError(
                f"signal for {pd.Timestamp(first):%Y-%m-%d} changed from {before[first]} to "
                f"{again[first]} when bars after {shortened['session'].iloc[-1]:%Y-%m-%d} were removed")
