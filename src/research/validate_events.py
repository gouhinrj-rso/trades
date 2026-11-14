"""Check a Market Scout event file before anything trusts it.

The Grok Bot writes one file per day to research/events/YYYY-MM-DD.jsonl, one
JSON object per line (format: research/grok/market_scout_prompt.md). That file
comes from outside the project, from a model reading the web, so it is treated
as untrusted DATA:

  - nothing in it is executed or followed as an instruction
  - every line must match the fixed format below exactly
  - a line that fails is rejected and reported; it is never repaired or guessed at
  - an empty file is valid and means "no material events"

Run from the repo root:

    python -m src.research.validate_events research/events/2026-11-16.jsonl

Exit code 0 = every line is valid; 1 = at least one line was rejected.

This is a checker only. It makes no trading decision and nothing reads its
result yet (that is PRD Prompt 7, in shadow mode).
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_DIR / "config" / "paper.yaml"

# ---- the format: every key a line must have, and what each may contain ----------
EVENT_TYPES = ("scheduled_release", "guidance_change", "supply_disruption", "regulatory_action", "halt", "other")
CHOICES = {
    "event_type": EVENT_TYPES,
    "is_surprise_vs_expectations": ("yes", "no", "unknown"),
    "direction_hint": ("up", "down", "unclear"),
    "magnitude_hint": ("low", "med", "high"),
    "confidence_evidence_quality": ("low", "med", "high"),
}
TRUE_OR_FALSE = ("scheduled", "veto_new_entry", "abstain")
TEXT = ("event_id", "ticker", "headline", "already_priced_evidence", "veto_reason")
REQUIRED_KEYS = (list(TEXT) + list(CHOICES) + list(TRUE_OR_FALSE)
                 + ["event_time_utc", "first_seen_utc", "primary_source_url", "corroborating_urls"])

FILE_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")
EVENT_ID = re.compile(r"^\d{4}-\d{2}-\d{2}-[A-Z.]{1,6}-[a-z0-9-]{1,60}$")
MAX_HEADLINE = 300
# A clock on another machine can be a little ahead of ours; allow this much.
CLOCK_SLACK = timedelta(minutes=10)


def load_watchlist(config_file: Path = CONFIG_FILE) -> list:
    """The tickers an event may name: the universe in config/paper.yaml."""
    with open(config_file) as handle:
        return list(yaml.safe_load(handle).get("universe") or [])


def parse_utc(value) -> datetime | None:
    """Turn '2026-11-16T13:30:00Z' into a time. Returns None if it is not in that form."""
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def is_https_url(value) -> bool:
    """True for a plain https web address with no spaces."""
    return isinstance(value, str) and value.startswith("https://") and " " not in value and len(value) > 12


def check_event(event, watchlist: list, now: datetime, file_date: str | None = None) -> list:
    """Return every problem with one event (an empty list means it is valid)."""
    if not isinstance(event, dict):
        return ["the line is not a JSON object"]
    problems = []

    missing = [key for key in REQUIRED_KEYS if key not in event]
    extra = [key for key in event if key not in REQUIRED_KEYS]
    if missing:
        problems.append(f"missing keys: {missing}")
    if extra:
        problems.append(f"unexpected keys: {extra}")
    if missing:
        return problems                      # the checks below need every key

    for key in TEXT:
        if not isinstance(event[key], str):
            problems.append(f"{key} must be text")
    for key, allowed in CHOICES.items():
        if event[key] not in allowed:
            problems.append(f"{key} must be one of {list(allowed)}, got {event[key]!r}")
    for key in TRUE_OR_FALSE:
        if not isinstance(event[key], bool):
            problems.append(f"{key} must be true or false, got {event[key]!r}")

    if isinstance(event["event_id"], str) and not EVENT_ID.match(event["event_id"]):
        problems.append("event_id must look like 2026-11-16-SPY-short-slug")
    if event["ticker"] not in watchlist:
        problems.append(f"ticker {event['ticker']!r} is not on the watchlist")
    if isinstance(event["headline"], str) and not 0 < len(event["headline"].strip()) <= MAX_HEADLINE:
        problems.append(f"headline must be 1 to {MAX_HEADLINE} characters")

    # ---- times --------------------------------------------------------------------
    first_seen = parse_utc(event["first_seen_utc"])
    if first_seen is None:
        problems.append("first_seen_utc must be an ISO time ending in Z")
    elif first_seen > now + CLOCK_SLACK:
        problems.append("first_seen_utc is in the future (an event cannot be seen before it is seen)")
    elif file_date and first_seen.strftime("%Y-%m-%d") not in _dates_near(file_date):
        problems.append(f"first_seen_utc {event['first_seen_utc']} does not belong in the file for {file_date}")
    if event["event_time_utc"] is not None and parse_utc(event["event_time_utc"]) is None:
        problems.append("event_time_utc must be an ISO time ending in Z, or null")
    if event["scheduled"] is True and event["event_time_utc"] is None:
        problems.append("a scheduled event needs an event_time_utc")

    # ---- sources -------------------------------------------------------------------
    primary = event["primary_source_url"]
    if primary is not None and not is_https_url(primary):
        problems.append("primary_source_url must be an https address or null")
    urls = event["corroborating_urls"]
    if not isinstance(urls, list) or not all(is_https_url(url) for url in urls):
        problems.append("corroborating_urls must be a list of https addresses")
    elif len(set(urls)) != len(urls) or primary in urls:
        problems.append("the same address is listed twice as a source")
    if primary is None and isinstance(urls, list) and not urls and event["abstain"] is False:
        problems.append("an event with no source at all must set abstain to true")

    # ---- vetoes ---------------------------------------------------------------------
    if event["veto_new_entry"] is True:
        if not (isinstance(event["veto_reason"], str) and event["veto_reason"].strip()):
            problems.append("a veto needs a veto_reason")
        if primary is None:
            problems.append("a veto needs a primary_source_url")
        if event["abstain"] is True:
            problems.append("an event cannot both veto and abstain")
    return problems


def _dates_near(file_date: str) -> set:
    """The file's own date and the days either side (UTC and New York dates differ by hours)."""
    day = datetime.strptime(file_date, "%Y-%m-%d")
    return {(day + timedelta(days=offset)).strftime("%Y-%m-%d") for offset in (-1, 0, 1)}


def check_file(path: Path, watchlist: list | None = None, now: datetime | None = None) -> dict:
    """Check one event file. Returns a report; raises nothing for bad content.

    The report has: `file`, `valid` (True/False), `events` (number of valid lines),
    `rejected` (list of {line, problems}) and `file_problems`.
    """
    path = Path(path)
    watchlist = load_watchlist() if watchlist is None else watchlist
    now = now or datetime.now(timezone.utc)
    report = {"file": path.name, "valid": False, "events": 0, "vetoes": 0, "rejected": [], "file_problems": []}

    name = FILE_NAME.match(path.name)
    if not name:
        report["file_problems"].append("the file must be named YYYY-MM-DD.jsonl")
    file_date = name.group(1) if name else None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        report["file_problems"].append(f"could not read the file: {type(error).__name__}")
        return report

    seen_ids = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue                                    # blank lines are ignored
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            report["rejected"].append({"line": number, "problems": ["not valid JSON"]})
            continue
        problems = check_event(event, watchlist, now, file_date)
        if not problems and event["event_id"] in seen_ids:
            problems = [f"event_id {event['event_id']} appears twice in this file"]
        if problems:
            report["rejected"].append({"line": number, "problems": problems})
            continue
        seen_ids.add(event["event_id"])
        report["events"] += 1
        report["vetoes"] += int(event["veto_new_entry"])

    report["valid"] = not report["rejected"] and not report["file_problems"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check Market Scout event files.")
    parser.add_argument("files", nargs="+", help="one or more research/events/YYYY-MM-DD.jsonl files")
    args = parser.parse_args(argv)
    all_valid = True
    for name in args.files:
        report = check_file(Path(name))
        all_valid = all_valid and report["valid"]
        verdict = "VALID" if report["valid"] else "REJECTED"
        note = " (empty file: no material events)" if report["valid"] and report["events"] == 0 else ""
        print(f"{report['file']}: {verdict} - {report['events']} valid event(s), {report['vetoes']} veto(es), "
              f"{len(report['rejected'])} rejected line(s){note}")
        for problem in report["file_problems"]:
            print(f"  file: {problem}")
        for item in report["rejected"]:
            print(f"  line {item['line']}: " + "; ".join(item["problems"]))
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
