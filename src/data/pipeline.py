"""Download, check, split and cache daily bars for the universe in config/paper.yaml.

Run from the repo root:

    python -m src.data.pipeline             # first run fetches; later runs print the saved report
    python -m src.data.pipeline --refresh   # refetch everything (OWNER ONLY - it moves the holdout)

What one run does, per symbol:
  1. Fetch daily bars from the configured source.
  2. Add source, fetched_at, available_at and the adjustment flag.
  3. Run every health check against the exchange calendar.
  4. If a check fails, or the source fails: cache NOTHING for that symbol and
     record the problem. No bar is ever filled in or made up.
  5. If it passes: split by DATE (set in config/paper.yaml and protocol.md) into
     train (warm-up included), validation and holdout, and save three parquet files.

An ETF that was listed after the requested start simply begins later; the
report says so and the gap check starts at its first bar.

The holdout file is locked. `load_bars` gives "train" and "validation" to
anyone, and "holdout" only to a caller holding a HoldoutAccess, which can only
be made by `src.data.holdout.unlock(...)` - and that writes who, when and why
to experiments/holdout_access.csv first.

A report with dates and counts (never prices) goes to reports/data_health.json
and reports/data_health.md. The run exits with code 1 if any symbol was blocked.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.data import sessions as cal
from src.data import validate
from src.data.holdout import HoldoutAccess
from src.data.sources import SourceError, YFinanceSource, normalize

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_DIR / "config" / "paper.yaml"
REPORT_DIR = PROJECT_DIR / "reports"

# The windows anyone may load. "holdout" needs a HoldoutAccess (see load_bars).
OPEN_WINDOWS = ("train", "validation")
HOLDOUT_FILE = "holdout_LOCKED.parquet"

SOURCES = {"yfinance": YFinanceSource}


class DataBlocked(RuntimeError):
    """Asked for data that is locked, missing or failed its health checks."""


def load_settings(config_file: Path = CONFIG_FILE) -> dict:
    """Read the data settings from config/paper.yaml and check they are filled in."""
    with open(config_file) as handle:
        config = yaml.safe_load(handle)
    data = dict(config.get("data") or {})
    # Fetch the tradable universe and the cash vehicles together.
    data["universe"] = list(config.get("universe") or []) + list(config.get("cash_vehicles") or [])
    for key in ("source", "calendar", "start", "train_end", "validation_end", "cache_dir"):
        if data.get(key) in (None, "TBD"):
            raise DataBlocked(f"config/paper.yaml: data.{key} is not set (TBD). Nothing was fetched.")
    if data["source"] not in SOURCES:
        raise DataBlocked(f"config/paper.yaml: unknown data.source '{data['source']}'. Known: {sorted(SOURCES)}")
    return data


def split_sessions(sessions: pd.DatetimeIndex, train_end, validation_end) -> dict:
    """Cut a list of trading days into train / validation / holdout by date.

    train      = every session up to and including `train_end` (warm-up included)
    validation = after `train_end`, up to and including `validation_end`
    holdout    = everything after `validation_end`
    """
    train_end, validation_end = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    return {"train": sessions[sessions <= train_end],
            "validation": sessions[(sessions > train_end) & (sessions <= validation_end)],
            "holdout": sessions[sessions > validation_end]}


def _window_summary(windows: dict) -> dict:
    """First day, last day and number of sessions for each window (for the report)."""
    return {name: {"first_session": f"{days[0]:%Y-%m-%d}" if len(days) else None,
                   "last_session": f"{days[-1]:%Y-%m-%d}" if len(days) else None,
                   "sessions": len(days)} for name, days in windows.items()}


def run(settings: dict, source=None, cache_dir: Path | None = None, report_dir: Path | None = None,
        now=None) -> dict:
    """Fetch, check, split and cache every symbol. Returns the report as a dictionary.

    `source`, `cache_dir`, `report_dir` and `now` can be passed in by tests; by
    default they come from the settings and the clock.
    """
    source = source or SOURCES[settings["source"]]()
    cache_dir = Path(cache_dir or PROJECT_DIR / settings["cache_dir"]) / "daily"
    report_dir = Path(report_dir or REPORT_DIR)
    calendar_name = settings["calendar"]
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)

    # The period: from the configured start up to the last finished session.
    end = cal.last_completed_session(calendar_name, now)
    expected = cal.expected_sessions(calendar_name, settings["start"], end)
    windows = split_sessions(expected, settings["train_end"], settings["validation_end"])

    report = {
        "generated_at": now.isoformat(), "source": source.name, "calendar": calendar_name,
        "requested": {"first_session": f"{expected[0]:%Y-%m-%d}", "last_session": f"{expected[-1]:%Y-%m-%d}",
                      "sessions": len(expected)},
        "windows": _window_summary(windows),
        "symbols": [], "blocked": [],
    }

    for symbol in settings["universe"]:
        entry = {"symbol": symbol, "status": None, "cached": False}
        report["symbols"].append(entry)
        symbol_dir = cache_dir / symbol

        # 1-2. Fetch and label. A source failure is a gap we report, never a reason to invent rows.
        try:
            bars = normalize(source.fetch_daily(symbol, expected[0], expected[-1]), symbol, source,
                             calendar_name, fetched_at=now)
        except SourceError as error:
            entry.update(status="SOURCE_FAILED", reason=str(error))
            report["blocked"].append(symbol)
            continue

        # 3. Health checks. A fund listed after the requested start begins at its
        # own first bar; that is reported, and gaps are counted from there.
        first_bar = bars["session"].min() if len(bars) else expected[0]
        check_from = max(first_bar, expected[0]) if first_bar in expected else expected[0]
        health = validate.check_bars(bars, symbol, calendar_name, check_from, expected[-1])
        entry.update(health.to_dict())
        entry["starts_late"] = bool(check_from > expected[0])
        entry["adjustment"] = source.adjustment
        entry["dividend_days"] = int((bars["dividend"].fillna(0) > 0).sum())
        entry["split_days"] = int(((bars["split_ratio"].fillna(0) > 0) & (bars["split_ratio"] != 1)).sum())

        # 4. Block on failure: nothing is cached for this symbol.
        if not health.passed:
            entry["status"] = "BLOCKED"
            report["blocked"].append(symbol)
            continue

        # 5. Split by time and save.
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for name, days in windows.items():
            part = bars[bars["session"].isin(days)]
            file_name = HOLDOUT_FILE if name == "holdout" else f"{name}.parquet"
            part.to_parquet(symbol_dir / file_name, index=False)
        entry.update(status="OK", cached=True)

    report["passed"] = not report["blocked"]
    write_report(report, report_dir)
    return report


def write_report(report: dict, report_dir: Path) -> None:
    """Save the report as JSON (for code) and Markdown (for people)."""
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "data_health.json", "w") as handle:
        json.dump(report, handle, indent=2)
    with open(report_dir / "data_health.md", "w") as handle:
        handle.write(render_markdown(report))


def render_markdown(report: dict) -> str:
    """Turn the report dictionary into a short readable page. Dates and counts only."""
    lines = [
        "# Data health report", "",
        f"- Generated: {report['generated_at']}",
        f"- Source: {report['source']}" + (" (research only)" if report["source"] == "yfinance" else ""),
        f"- Calendar: {report['calendar']}",
        f"- Requested: {report['requested']['first_session']} to {report['requested']['last_session']} "
        f"({report['requested']['sessions']} sessions)",
        f"- Result: {'PASS - every symbol cached' if report['passed'] else 'BLOCKED: ' + ', '.join(report['blocked'])}",
        "", "## Windows", "", "| Window | First session | Last session | Sessions |", "|---|---|---|---|",
    ]
    for name, window in report["windows"].items():
        label = "holdout (LOCKED)" if name == "holdout" else name
        lines.append(f"| {label} | {window['first_session']} | {window['last_session']} | {window['sessions']} |")
    lines += ["", "## Symbols", "",
              "| Symbol | Status | Rows | Expected | First | Last | Starts late | Dividend days | Split days | Adjustment |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for entry in report["symbols"]:
        lines.append("| {symbol} | {status} | {rows} | {expected} | {first} | {last} | {late} | {div} | {split} | {adj} |".format(
            symbol=entry["symbol"], status=entry["status"], rows=entry.get("rows", "-"),
            expected=entry.get("expected_sessions", "-"), first=entry.get("first_session", "-"),
            last=entry.get("last_session", "-"),
            late="yes" if entry.get("starts_late") else "no", div=entry.get("dividend_days", "-"),
            split=entry.get("split_days", "-"), adj=entry.get("adjustment", "-")))
    lines += ["", "## Problems", ""]
    problems = []
    for entry in report["symbols"]:
        if entry["status"] == "SOURCE_FAILED":
            problems.append(f"- **{entry['symbol']}**: source failed, no data. {entry['reason']}")
        for issue in entry.get("issues", []):
            problems.append(f"- **{entry['symbol']}** `{issue['check']}`: {issue['count']} "
                            f"({issue['detail']}); examples: {', '.join(issue['examples']) or 'n/a'}")
    lines += problems or ["None. Every check passed for every symbol."]
    lines += ["", "Checks run: " + ", ".join(
        ["empty", "missing_value", "non_positive", "negative_volume", "bad_range", "duplicate", "out_of_order",
         "off_calendar", "gap", "stale_end", "stale_repeat", "zero_volume", "unadjusted_split", "future_bar"]) + ".", ""]
    return "\n".join(lines)


def load_bars(symbol: str, window: str, cache_dir: Path | None = None,
              holdout_access: HoldoutAccess | None = None) -> pd.DataFrame:
    """Load cached bars for one symbol.

    "train" and "validation" load for anyone. "holdout" loads only when the
    caller passes a HoldoutAccess, which `src.data.holdout.unlock` hands out
    after writing the access to the log. Anything else is refused.
    """
    if window == "holdout":
        if not isinstance(holdout_access, HoldoutAccess):
            raise DataBlocked("the holdout is LOCKED. It opens only through src.data.holdout.unlock(...), "
                              "which logs who and when. Only the owner does that.")
        file_name = HOLDOUT_FILE
    elif window in OPEN_WINDOWS:
        file_name = f"{window}.parquet"
    else:
        raise DataBlocked(f"window '{window}' cannot be loaded. Allowed: {OPEN_WINDOWS} (and the locked holdout).")
    if cache_dir is None:
        cache_dir = PROJECT_DIR / load_settings()["cache_dir"]
    path = Path(cache_dir) / "daily" / symbol / file_name
    if not path.exists():
        raise DataBlocked(f"no cached {window} bars for {symbol}: it was never fetched or it failed its health checks")
    return pd.read_parquet(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch, check and cache daily bars.")
    parser.add_argument("--refresh", action="store_true",
                        help="refetch everything. OWNER ONLY: this moves the train/validation/holdout windows.")
    args = parser.parse_args(argv)

    saved = REPORT_DIR / "data_health.md"
    if saved.exists() and not args.refresh:
        # Same habit as fetch_data.py: a rerun shows the saved report instead of refetching,
        # so the holdout window cannot drift by accident.
        print(saved.read_text())
        print("(saved report; use --refresh to refetch - owner only)")
        if not (PROJECT_DIR / load_settings()["cache_dir"] / "daily").exists():
            # The cache is git-ignored, so a fresh checkout has the report but no bars.
            print("NOTE: the bar cache is not on this machine. The report above describes an "
                  "earlier fetch; nothing can be loaded until the owner runs --refresh.")
        return 0

    try:
        report = run(load_settings())
    except DataBlocked as error:
        print(f"BLOCKED: {error}")
        return 1
    print(render_markdown(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
