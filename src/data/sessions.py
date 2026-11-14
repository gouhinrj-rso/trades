"""Exchange calendar: which days the market was open, and when it closed.

Uses the pandas_market_calendars library, so holidays and early closes come
from a maintained calendar instead of a guess like "every weekday".
"""

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal


@lru_cache(maxsize=8)
def _calendar(name: str):
    """Load one exchange calendar by name (for example "NYSE"). Cached."""
    return mcal.get_calendar(name)


def session_closes(calendar_name: str, start, end) -> pd.Series:
    """Return the closing time (UTC) of every trading session from start to end.

    The result is a Series: the index is the session date (no time zone), the
    value is the moment the market closed that day, in UTC. Early closes (for
    example the day after Thanksgiving) are handled by the calendar.
    """
    schedule = _calendar(calendar_name).schedule(start_date=pd.Timestamp(start).date(),
                                                 end_date=pd.Timestamp(end).date())
    closes = pd.to_datetime(schedule["market_close"], utc=True)
    closes.index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    closes.index.name = "session"
    return closes


def expected_sessions(calendar_name: str, start, end) -> pd.DatetimeIndex:
    """Return every trading day from start to end (both included)."""
    return session_closes(calendar_name, start, end).index


def last_completed_session(calendar_name: str, now=None) -> pd.Timestamp:
    """Return the most recent session whose close is already in the past."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    # Look back two weeks; that always contains at least one finished session.
    closes = session_closes(calendar_name, now - pd.Timedelta(days=14), now)
    finished = closes[closes <= now]
    if finished.empty:
        raise ValueError(f"no finished {calendar_name} session in the two weeks before {now}")
    return finished.index[-1]
