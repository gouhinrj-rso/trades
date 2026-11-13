"""Fetch 5-minute OHLC bars from Kraken's public API and split them by time.

Run from the trading-terminal folder:

    python fetch_data.py            # first run: fetch, clean, split, save, report
    python fetch_data.py            # later runs: print the saved report only
    python fetch_data.py --refresh  # fetch again and REPLACE all saved windows

No API key is used. This script only reads public market data. It places no
orders and connects to no account.

Why a rerun does not fetch again by default:
    Kraken only serves the most recent bars. Fetching again moves the whole
    data range forward, which would move the train/validation/test boundaries
    and could turn old test bars into training bars (a look-ahead leak).
    So a rerun prints the saved report, and you must pass --refresh on purpose.

Files written to data/:
    bars_5m.json         all cleaned bars (NOTE: this includes the test bars)
    train_5m.json        first 60% of bars
    validation_5m.json   next 20% of bars
    test_5m_LOCKED.json  last 20% of bars - do not open until "unlock test"
    split_report.json    range, counts, gaps and window boundaries (no prices)
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Settings (change the pair here if you want a different asset)
# ----------------------------------------------------------------------------
PAIR = "ETHUSD"                # Kraken pair name, e.g. "XBTUSD" for BTC/USD
INTERVAL_MINUTES = 5           # bar size in minutes
BAR_SECONDS = INTERVAL_MINUTES * 60
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

TRAIN_FRACTION = 0.60          # first 60% of bars
VALIDATION_FRACTION = 0.20     # next 20%; the test window is whatever is left

# All output goes in a "data" folder next to this script.
DATA_DIR = Path(__file__).resolve().parent / "data"
ALL_BARS_FILE = DATA_DIR / "bars_5m.json"
TRAIN_FILE = DATA_DIR / "train_5m.json"
VALIDATION_FILE = DATA_DIR / "validation_5m.json"
TEST_FILE = DATA_DIR / "test_5m_LOCKED.json"
REPORT_FILE = DATA_DIR / "split_report.json"


def to_iso(unix_seconds: int) -> str:
    """Turn a Unix time (seconds) into readable UTC text like 2026-11-12T23:05:00Z."""
    moment = datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def from_iso(text: str) -> int:
    """Turn UTC text like 2026-11-12T23:05:00Z back into Unix seconds."""
    moment = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    return int(moment.replace(tzinfo=timezone.utc).timestamp())


def fetch_raw_rows() -> list:
    """Ask Kraken for the most recent bars and return its raw rows.

    Each raw row looks like:
        [time, open, high, low, close, vwap, volume, trade_count]
    Kraken returns the most recent bars only (about 720 per call).
    """
    query = urllib.parse.urlencode({"pair": PAIR, "interval": INTERVAL_MINUTES})
    request = urllib.request.Request(
        f"{KRAKEN_OHLC_URL}?{query}",
        headers={"User-Agent": "trading-terminal-fetch"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    # Kraken reports problems in an "error" list instead of an HTTP error code.
    if payload.get("error"):
        raise RuntimeError(f"Kraken returned an error: {payload['error']}")

    # "result" holds two keys: the pair's rows and "last". Kraken may rename
    # the pair (ETHUSD comes back as XETHZUSD), so take the key that is not "last".
    result = payload["result"]
    pair_keys = [key for key in result if key != "last"]
    if len(pair_keys) != 1:
        raise RuntimeError(f"Expected one pair in the response, got: {pair_keys}")
    return result[pair_keys[0]]


def clean_rows(raw_rows: list, now_seconds: int) -> tuple:
    """Convert raw rows to bar dictionaries, sorted, de-duplicated, complete only.

    Returns (bars, duplicates_dropped, unfinished_dropped).
    """
    # Sort by the bar's start time (first item in each row).
    sorted_rows = sorted(raw_rows, key=lambda row: row[0])

    bars = []
    seen_times = set()
    duplicates_dropped = 0
    unfinished_dropped = 0

    for row in sorted_rows:
        start = int(row[0])

        # Drop a repeated timestamp (keep the first one we saw).
        if start in seen_times:
            duplicates_dropped += 1
            continue
        seen_times.add(start)

        # Drop the bar that is still forming. Its close and volume will change,
        # so using it would mean using numbers that are not final yet.
        if start + BAR_SECONDS > now_seconds:
            unfinished_dropped += 1
            continue

        bars.append(
            {
                "timestamp": to_iso(start),   # start of the bar, UTC
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[6]),      # row[5] is vwap, which we skip
            }
        )

    return bars, duplicates_dropped, unfinished_dropped


def find_gaps(bars: list) -> list:
    """Find every place where two neighbouring bars are more than one bar apart."""
    gaps = []
    for previous, current in zip(bars, bars[1:]):
        step = from_iso(current["timestamp"]) - from_iso(previous["timestamp"])
        if step > BAR_SECONDS:
            gaps.append(
                {
                    "after": previous["timestamp"],
                    "before": current["timestamp"],
                    "missing_bars": step // BAR_SECONDS - 1,
                }
            )
    return gaps


def split_by_time(bars: list) -> tuple:
    """Cut the time-ordered bars into train / validation / test. Never random."""
    total = len(bars)
    train_end = int(total * TRAIN_FRACTION)
    validation_end = train_end + int(total * VALIDATION_FRACTION)
    return bars[:train_end], bars[train_end:validation_end], bars[validation_end:]


def describe_window(name: str, file_path: Path, window: list) -> dict:
    """Summarise one window for the report (times and size only, no prices)."""
    return {
        "name": name,
        "file": file_path.name,
        "bars": len(window),
        "first_timestamp": window[0]["timestamp"] if window else None,
        "last_timestamp": window[-1]["timestamp"] if window else None,
    }


def save_json(file_path: Path, content) -> None:
    """Write content to a JSON file."""
    with open(file_path, "w") as handle:
        json.dump(content, handle, indent=1)


def fetch_and_save() -> dict:
    """Do the full job: fetch, clean, check gaps, split, save. Returns the report."""
    now_seconds = int(time.time())
    raw_rows = fetch_raw_rows()
    bars, duplicates_dropped, unfinished_dropped = clean_rows(raw_rows, now_seconds)
    if not bars:
        raise RuntimeError("No complete bars came back from Kraken.")

    gaps = find_gaps(bars)
    train, validation, test = split_by_time(bars)

    DATA_DIR.mkdir(exist_ok=True)
    save_json(ALL_BARS_FILE, bars)
    save_json(TRAIN_FILE, train)
    save_json(VALIDATION_FILE, validation)
    save_json(TEST_FILE, test)

    report = {
        "pair": PAIR,
        "interval_minutes": INTERVAL_MINUTES,
        "source": KRAKEN_OHLC_URL,
        "fetched_at": to_iso(now_seconds),
        "rows_returned_by_kraken": len(raw_rows),
        "duplicates_dropped": duplicates_dropped,
        "unfinished_bars_dropped": unfinished_dropped,
        "bar_count": len(bars),
        "first_timestamp": bars[0]["timestamp"],
        "last_timestamp": bars[-1]["timestamp"],
        "gaps": gaps,
        "windows": [
            describe_window("train", TRAIN_FILE, train),
            describe_window("validation", VALIDATION_FILE, validation),
            describe_window("test (LOCKED)", TEST_FILE, test),
        ],
    }
    save_json(REPORT_FILE, report)
    return report


def print_report(report: dict, freshly_fetched: bool) -> None:
    """Print the range, bar count, gaps and window sizes."""
    if freshly_fetched:
        print("Fetched new bars from Kraken and saved them.")
    else:
        print("Using saved data (no fetch). Pass --refresh to fetch again;")
        print("that REPLACES every window, including the locked test window.")
    print()
    print(f"Pair / interval    : {report['pair']} / {report['interval_minutes']} min")
    print(f"Fetched at         : {report['fetched_at']}")
    print(f"Rows from Kraken   : {report['rows_returned_by_kraken']}")
    print(f"Duplicates dropped : {report['duplicates_dropped']}")
    print(f"Unfinished dropped : {report['unfinished_bars_dropped']}")
    print()
    print(f"First timestamp    : {report['first_timestamp']}")
    print(f"Last timestamp     : {report['last_timestamp']}")
    print(f"Bar count          : {report['bar_count']}")
    print()

    gaps = report["gaps"]
    if not gaps:
        print("Gaps larger than one bar: none")
    else:
        print(f"Gaps larger than one bar: {len(gaps)}")
        for gap in gaps:
            print(
                f"  between {gap['after']} and {gap['before']}"
                f"  ({gap['missing_bars']} missing bar(s))"
            )
    print()

    print("Windows (split by time):")
    for window in report["windows"]:
        print(
            f"  {window['name']:<14} {window['bars']:>4} bars  "
            f"{window['first_timestamp']} -> {window['last_timestamp']}  "
            f"[{window['file']}]"
        )
    print()
    print('The test file stays closed until the owner says "unlock test".')
    print("bars_5m.json also contains the test bars; later steps must read only")
    print("train_5m.json and validation_5m.json.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and split Kraken 5-minute bars.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="fetch again and replace all saved windows, including the locked test window",
    )
    args = parser.parse_args()

    # A rerun reads only the small report file. It never opens the test file.
    if REPORT_FILE.exists() and not args.refresh:
        with open(REPORT_FILE) as handle:
            report = json.load(handle)
        print_report(report, freshly_fetched=False)
        return 0

    report = fetch_and_save()
    print_report(report, freshly_fetched=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
