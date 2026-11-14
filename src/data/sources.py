"""Where daily bars come from.

A "source" is any object with a `name` and a `fetch_daily(symbol, start, end)`
method that returns bars in the project's table shape (see schema.py), WITHOUT
the calendar columns; `normalize()` in this file adds those.

Only market-data hosts are used. Nothing here can place an order.

Sources:
    YFinanceSource   Yahoo Finance through the yfinance library. No key needed.
                     RESEARCH ONLY: unofficial, can change or disappear, and its
                     history is rewritten after every split and dividend.

An Alpaca market-data source is not written yet (it needs a paper key from the owner).
"""

import pandas as pd

from src.data import sessions as cal
from src.data.schema import COLUMNS


class SourceError(RuntimeError):
    """The source could not return data. The caller reports the gap; nothing is made up."""


class YFinanceSource:
    """Daily bars from Yahoo Finance (research only)."""

    name = "yfinance"
    # Yahoo's open/high/low/close are already adjusted for splits but not for
    # dividends. "Adj Close" is adjusted for both.
    adjustment = "split_adjusted"

    def fetch_daily(self, symbol: str, start, end) -> pd.DataFrame:
        """Download one symbol from `start` to `end` (both included)."""
        import yfinance as yf   # imported here so tests never need the library or the network

        try:
            # Yahoo treats `end` as "up to but not including", so ask for one more day.
            raw = yf.Ticker(symbol).history(
                start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=False, actions=True, raise_errors=True)
        except Exception as error:   # any failure becomes a reported gap
            raise SourceError(f"yfinance could not fetch {symbol}: {type(error).__name__}: {error}") from error

        if raw is None or raw.empty:
            raise SourceError(f"yfinance returned no rows for {symbol}")

        bars = pd.DataFrame({
            # The index is midnight New York time on the trading day; keep only the date.
            "session": pd.DatetimeIndex(raw.index).tz_localize(None).normalize(),
            "open": raw["Open"].to_numpy(), "high": raw["High"].to_numpy(),
            "low": raw["Low"].to_numpy(), "close": raw["Close"].to_numpy(),
            "adj_close": raw["Adj Close"].to_numpy(), "volume": raw["Volume"].to_numpy(),
            "dividend": raw["Dividends"].to_numpy(),
            # Yahoo writes 0 on days without a split; 2.0 means a 2-for-1 split that day.
            "split_ratio": raw["Stock Splits"].to_numpy(),
        })
        return bars


def normalize(bars: pd.DataFrame, symbol: str, source, calendar_name: str, fetched_at=None) -> pd.DataFrame:
    """Add symbol, source, fetched_at, available_at and the adjustment flag.

    `available_at` is the exchange's closing time on the bar's session, taken
    from the calendar. A row dated on a day the exchange was closed gets no
    `available_at` (it stays blank) and the health check reports it.
    """
    fetched_at = pd.Timestamp.now(tz="UTC") if fetched_at is None else pd.Timestamp(fetched_at)
    out = bars.copy()
    out["session"] = pd.to_datetime(out["session"]).dt.normalize()
    out["symbol"] = symbol
    out["source"] = source.name
    out["fetched_at"] = fetched_at
    out["adjustment"] = source.adjustment
    if out.empty:
        out["available_at"] = pd.Series(dtype="datetime64[ns, UTC]")
    else:
        closes = cal.session_closes(calendar_name, out["session"].min(), out["session"].max())
        out["available_at"] = out["session"].map(closes)
    return out[COLUMNS]
