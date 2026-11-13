"""Tests for the cleaning, gap and split logic in fetch_data.py.

These tests use made-up rows. They never call Kraken and never open data/.
"""

import fetch_data

BAR = fetch_data.BAR_SECONDS
START = 1_700_000_100  # an arbitrary time that sits on a 5-minute boundary


def make_row(start):
    """Build one raw row in Kraken's layout: time, o, h, l, c, vwap, volume, count."""
    return [start, "10.0", "11.0", "9.0", "10.5", "10.2", "3.0", 7]


def test_clean_sorts_drops_duplicates_and_unfinished_bar():
    # Out of order, with one duplicate, and a last bar that is still forming.
    rows = [make_row(START + BAR), make_row(START), make_row(START), make_row(START + 2 * BAR)]
    now = START + 2 * BAR + 10  # 10 seconds into the third bar

    bars, duplicates, unfinished = fetch_data.clean_rows(rows, now)

    assert [b["timestamp"] for b in bars] == [fetch_data.to_iso(START), fetch_data.to_iso(START + BAR)]
    assert duplicates == 1
    assert unfinished == 1
    assert set(bars[0]) == {"timestamp", "open", "high", "low", "close", "volume"}
    assert bars[0]["volume"] == 3.0  # volume, not vwap


def test_find_gaps_reports_missing_bars():
    # Bars at 0, 1, then a jump to 4: bars 2 and 3 are missing.
    rows = [make_row(START), make_row(START + BAR), make_row(START + 4 * BAR)]
    bars, _, _ = fetch_data.clean_rows(rows, START + 100 * BAR)

    gaps = fetch_data.find_gaps(bars)

    assert len(gaps) == 1
    assert gaps[0]["missing_bars"] == 2
    assert gaps[0]["after"] == fetch_data.to_iso(START + BAR)
    assert gaps[0]["before"] == fetch_data.to_iso(START + 4 * BAR)


def test_split_is_by_time_and_60_20_20():
    rows = [make_row(START + i * BAR) for i in range(100)]
    bars, _, _ = fetch_data.clean_rows(rows, START + 1000 * BAR)

    train, validation, test = fetch_data.split_by_time(bars)

    assert (len(train), len(validation), len(test)) == (60, 20, 20)
    # Every train bar is earlier than every validation bar, and so on.
    assert train[-1]["timestamp"] < validation[0]["timestamp"]
    assert validation[-1]["timestamp"] < test[0]["timestamp"]
    # Nothing lost, nothing repeated.
    assert train + validation + test == bars


def test_iso_round_trip():
    assert fetch_data.from_iso(fetch_data.to_iso(START)) == START
