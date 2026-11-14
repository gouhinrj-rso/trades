"""Split and dividend adjustments, done in the open.

Why this matters: on the day of a 2-for-1 split a share's price halves, but
nobody lost money. A backtest on unadjusted prices would see a 50% crash.
Dividends are the same idea in small: the price drops by the dividend on the
ex-dividend day, and the holder gets the cash.

Three kinds of series (the `adjustment` column says which one a file is):

    raw                  prices as traded
    split_adjusted       history rescaled so splits leave no jump
    split_div_adjusted   also rescaled for dividends (a total-return series)

Rules:
- Every function returns a NEW table with the flag updated. Nothing is changed in place.
- A function refuses a table whose flag says the work is already done, so an
  adjustment can never be applied twice.
- Adjusted history is only correct as of the day it was built: a later split or
  dividend rescales the whole past. `fetched_at` records that day.
"""

import pandas as pd

from src.data.schema import PRICE_COLUMNS, check_schema


class AdjustmentError(ValueError):
    """The requested adjustment does not fit the table's current flag."""


def _flag(bars: pd.DataFrame) -> str:
    check_schema(bars)
    return bars["adjustment"].iloc[0]


def apply_splits(bars: pd.DataFrame) -> pd.DataFrame:
    """Turn a RAW series into a split-adjusted one.

    For each split, every bar BEFORE the split day has its prices divided by
    the ratio and its volume multiplied by it. A split_ratio of 2 means 2 new
    shares for 1 old share. Days with no split have split_ratio 0, 1 or blank.
    """
    if _flag(bars) != "raw":
        raise AdjustmentError(f"apply_splits needs raw prices, got '{_flag(bars)}'")
    out = bars.sort_values("session").reset_index(drop=True).copy()

    ratio = out["split_ratio"].fillna(1.0).where(lambda value: value > 0, 1.0)
    # Factor for a bar = product of the ratios of every split AFTER that bar.
    # Reverse, multiply up, shift by one so the split day itself is not divided.
    later_splits = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)

    for column in PRICE_COLUMNS:
        out[column] = out[column] / later_splits
    out["dividend"] = out["dividend"] / later_splits     # dividends are per share too
    out["volume"] = out["volume"] * later_splits
    out["adjustment"] = "split_adjusted"
    return out


def apply_dividends(bars: pd.DataFrame) -> pd.DataFrame:
    """Turn a split-adjusted series into a total-return (split and dividend adjusted) one.

    On each ex-dividend day the factor is (previous close - dividend) / previous
    close; every bar BEFORE that day is multiplied by it. `adj_close` is set to
    the adjusted close.
    """
    if _flag(bars) != "split_adjusted":
        raise AdjustmentError(f"apply_dividends needs split_adjusted prices, got '{_flag(bars)}'")
    out = bars.sort_values("session").reset_index(drop=True).copy()

    dividend = out["dividend"].fillna(0.0)
    previous_close = out["close"].shift(1)
    # No dividend, or no previous close on the first row -> factor 1 (no change).
    day_factor = ((previous_close - dividend) / previous_close).where(dividend > 0, 1.0).fillna(1.0)
    later_dividends = day_factor[::-1].cumprod()[::-1].shift(-1).fillna(1.0)

    for column in PRICE_COLUMNS:
        out[column] = out[column] * later_dividends
    out["adj_close"] = out["close"]
    out["adjustment"] = "split_div_adjusted"
    return out


def total_return_from_adj_close(bars: pd.DataFrame) -> pd.DataFrame:
    """Build a total-return series using the source's own adjusted close.

    Some sources (Yahoo) hand over split-adjusted prices plus an `adj_close`
    that also removes dividends. Scaling open/high/low/close by
    adj_close / close gives the same thing as apply_dividends, using the
    source's numbers instead of ours.
    """
    if _flag(bars) != "split_adjusted":
        raise AdjustmentError(f"needs split_adjusted prices, got '{_flag(bars)}'")
    out = bars.copy()
    factor = out["adj_close"] / out["close"]
    for column in PRICE_COLUMNS:
        out[column] = out[column] * factor
    out["adjustment"] = "split_div_adjusted"
    return out
