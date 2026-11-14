"""The one table shape every daily bar in this project uses.

A "bar" is one trading day for one symbol. Every row carries, besides prices:

    source       where the row came from (for example "yfinance")
    fetched_at   when we downloaded it (UTC)
    available_at when the bar could first have been known (UTC). For a daily
                 bar that is the exchange's closing time on that session.
                 Nothing downstream may use a row before this moment.
    adjustment   what has already been done to the prices (see ADJUSTMENTS)

Prices are never invented here. If a value is missing it stays missing and the
health checks in validate.py block the data set.
"""

import pandas as pd

# Column order for a cached file. "session" is the trading day (no time, no zone).
COLUMNS = [
    "symbol", "session", "open", "high", "low", "close", "adj_close", "volume",
    "dividend", "split_ratio", "source", "fetched_at", "available_at", "adjustment",
]

PRICE_COLUMNS = ["open", "high", "low", "close"]

# What "adjustment" may say.
#   raw                  prices exactly as traded on the day
#   split_adjusted       past prices divided by later split ratios; dividends NOT removed
#   split_div_adjusted   split adjusted AND scaled down for later dividends (total return)
ADJUSTMENTS = ("raw", "split_adjusted", "split_div_adjusted")


class SchemaError(ValueError):
    """The table does not have the shape described in this file."""


def check_schema(bars: pd.DataFrame) -> None:
    """Raise SchemaError unless `bars` has every column and one known adjustment flag."""
    missing = [column for column in COLUMNS if column not in bars.columns]
    if missing:
        raise SchemaError(f"bars are missing columns: {missing}")
    flags = set(bars["adjustment"].unique())
    if len(flags) > 1:
        raise SchemaError(f"one series mixes adjustment flags {sorted(flags)}; never mix them")
    if flags and not flags <= set(ADJUSTMENTS):
        raise SchemaError(f"unknown adjustment flag {sorted(flags)}; allowed: {ADJUSTMENTS}")
