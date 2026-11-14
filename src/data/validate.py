"""Data-health checks. A data set that fails ANY check is blocked.

Nothing here repairs data. There is no fill, no interpolation and no dropping
of bad rows: a problem is listed and the caller is stopped.

Checks, each against the exchange calendar where that makes sense:

    empty            no rows at all
    missing_value    a price or volume is blank (NaN)
    non_positive     a price is zero or negative
    negative_volume  volume below zero
    bad_range        high below low, or open/close outside the high-low range
    duplicate        the same session appears twice
    out_of_order     sessions are not in time order
    off_calendar     a row dated on a day the exchange was closed
    gap              a trading session with no row
    stale_repeat     a bar identical to the one before it (prices and volume)
    zero_volume      no volume on a trading day
    stale_end        the newest bar is older than the last session it should cover
    unadjusted_split a split happened but prices still show the full jump,
                     although the series says it is split adjusted
    future_bar       a bar that was "available" after it was fetched

Run `check_bars(...)` to get a HealthReport, then `report.raise_if_failed()`.
"""

from dataclasses import dataclass, field

import pandas as pd

from src.data import sessions as cal
from src.data.schema import PRICE_COLUMNS, check_schema

# How many example dates to keep per problem (the count is always exact).
MAX_EXAMPLES = 10

# Prices are decimals stored as floats, so allow this much rounding error when
# comparing open/close with the high-low range.
RANGE_TOLERANCE = 1e-6


class DataHealthError(RuntimeError):
    """Raised when a data set fails a health check. The data must not be used."""


@dataclass
class Issue:
    """One failed check: which check, how many rows, and a few example dates."""
    check: str
    count: int
    examples: list
    detail: str = ""


@dataclass
class HealthReport:
    """The result of checking one symbol's bars."""
    symbol: str
    rows: int
    first_session: str | None
    last_session: str | None
    expected_sessions: int
    issues: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def raise_if_failed(self) -> None:
        """Stop here if any check failed."""
        if self.issues:
            lines = [f"{issue.check}: {issue.count} ({issue.detail}) e.g. {issue.examples[:3]}"
                     for issue in self.issues]
            raise DataHealthError(f"{self.symbol}: data blocked, {len(self.issues)} check(s) failed - "
                                  + "; ".join(lines))

    def to_dict(self) -> dict:
        """Plain dictionary for the JSON report. Contains dates and counts, never prices."""
        return {
            "symbol": self.symbol, "passed": self.passed, "rows": self.rows,
            "first_session": self.first_session, "last_session": self.last_session,
            "expected_sessions": self.expected_sessions,
            "issues": [vars(issue) for issue in self.issues],
        }


def _dates(index_or_series) -> list:
    """Turn dates into short text, keeping only the first few as examples."""
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in list(index_or_series)[:MAX_EXAMPLES]]


def check_bars(bars: pd.DataFrame, symbol: str, calendar_name: str, start, end) -> HealthReport:
    """Run every check on one symbol's bars for the period start..end (both included).

    `start` and `end` are the first and last trading days the data is supposed
    to cover. Returns a HealthReport; it does not raise. Call
    `report.raise_if_failed()` to block.
    """
    expected = cal.expected_sessions(calendar_name, start, end)
    report = HealthReport(symbol=symbol, rows=len(bars), first_session=None, last_session=None,
                          expected_sessions=len(expected))

    def add(check: str, bad, detail: str = "") -> None:
        """Record a failed check if `bad` (a list of dates) is not empty."""
        bad = list(bad)
        if bad:
            report.issues.append(Issue(check, len(bad), _dates(bad), detail))

    if bars.empty:
        report.issues.append(Issue("empty", 0, [], "no rows were returned"))
        return report

    check_schema(bars)
    sessions = pd.DatetimeIndex(bars["session"])
    report.first_session = sessions.min().strftime("%Y-%m-%d")
    report.last_session = sessions.max().strftime("%Y-%m-%d")

    # --- values -------------------------------------------------------------
    blank = bars[PRICE_COLUMNS + ["volume"]].isna().any(axis=1)
    add("missing_value", sessions[blank], "a price or volume is blank")

    non_positive = (bars[PRICE_COLUMNS] <= 0).any(axis=1)
    add("non_positive", sessions[non_positive], "a price is zero or negative")

    add("negative_volume", sessions[bars["volume"] < 0], "volume below zero")

    tolerance = bars["high"].abs() * RANGE_TOLERANCE
    bad_range = ((bars["high"] < bars["low"])
                 | (bars[["open", "close"]].max(axis=1) > bars["high"] + tolerance)
                 | (bars[["open", "close"]].min(axis=1) < bars["low"] - tolerance))
    add("bad_range", sessions[bad_range], "high below low, or open/close outside high-low")

    # --- order and duplicates -------------------------------------------------
    add("duplicate", sessions[sessions.duplicated(keep=False)], "session appears more than once")
    if not sessions.is_monotonic_increasing:
        earlier = sessions[1:][sessions[1:] < sessions[:-1]]
        add("out_of_order", earlier, "session is earlier than the row before it")

    # --- against the exchange calendar ------------------------------------------
    unique_sessions = sessions.unique()
    add("off_calendar", unique_sessions.difference(expected).intersection(
        pd.date_range(pd.Timestamp(start), pd.Timestamp(end))), "exchange was closed on this day")
    add("gap", expected.difference(unique_sessions), "trading session with no bar")
    if len(expected) and sessions.max() < expected[-1]:
        report.issues.append(Issue("stale_end", 1, _dates([sessions.max()]),
                                   f"newest bar is older than {expected[-1]:%Y-%m-%d}"))

    # --- stale bars ---------------------------------------------------------------
    ordered = bars.sort_values("session")
    same_as_before = (ordered[PRICE_COLUMNS + ["volume"]] == ordered[PRICE_COLUMNS + ["volume"]].shift(1)).all(axis=1)
    add("stale_repeat", ordered.loc[same_as_before, "session"], "identical to the previous bar")
    add("zero_volume", ordered.loc[ordered["volume"] == 0, "session"], "no volume on a trading day")

    # --- splits -----------------------------------------------------------------------
    # If the file says "split adjusted" then a 2-for-1 split must NOT show up as a
    # 50% overnight drop. If it does, the flag is wrong and every signal would be too.
    if bars["adjustment"].iloc[0] != "raw":
        has_split = ordered["split_ratio"].notna() & (ordered["split_ratio"] > 0) & (ordered["split_ratio"] != 1)
        day_change = ordered["close"] / ordered["close"].shift(1)
        raw_split_change = 1.0 / ordered["split_ratio"]
        looks_unadjusted = has_split & ((day_change - raw_split_change).abs() < 0.1 * raw_split_change)
        add("unadjusted_split", ordered.loc[looks_unadjusted, "session"],
            "price jump equals the split ratio, so the series is not split adjusted")

    # --- point in time ------------------------------------------------------------------
    from_the_future = pd.to_datetime(bars["available_at"], utc=True) > pd.to_datetime(bars["fetched_at"], utc=True)
    add("future_bar", sessions[from_the_future], "bar was fetched before its session closed")

    return report
