"""Market Scout events in SHADOW mode (PRD Prompt 7).

The Grok Bot writes event files to research/events/. Some events carry a
"veto": a suggestion not to open a new position in one instrument for a while.
This module measures whether listening to those vetoes would have helped,
WITHOUT letting them touch the real paper strategy.

    1. ingest      read research/events/*.jsonl, check every line, keep the good
                   ones with the time we first saw them, log the bad ones
    2. vetoes      answer "was a veto in force for this instrument at this moment?"
                   using only what had been ingested by that moment
    3. overlay     run the SAME strategy twice: the baseline ignores vetoes;
                   the overlay skips new entries a veto covers
    4. compare     baseline against overlay: profit, drawdown, skipped trades,
                   what the skipped trades would have made, veto precision
    5. report      reports/grok_value.md, with keep / drop / not enough evidence

Safety rules built in (PRD: "missing or stale Grok data must never create a trade"):
  - A veto can only REMOVE a new entry. It can never add a position, enlarge
    one, or force a sale. With no events at all the overlay equals the baseline.
  - An event counts from the moment it was INGESTED, not from the time written
    inside it. A file added later cannot veto an earlier decision.
  - Event files are untrusted outside data. Nothing in them is executed, and
    no model is called anywhere in this module.

Run from the repo root:

    python -m src.research.grok_events ingest     # check and store new event lines
    python -m src.research.grok_events report     # write reports/grok_value.md
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, ValidationError

from src.data import sessions as cal
from src.research import robustness
from src.research import validate_events as ve

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_DIR / "config" / "paper.yaml"
EVENTS_DIR = PROJECT_DIR / "research" / "events"
STORE_DIR = PROJECT_DIR / "research" / "ingested"
STORE_FILE = STORE_DIR / "events.jsonl"          # every accepted event, with ingested_at
REJECTS_FILE = STORE_DIR / "rejected.jsonl"      # every rejected line, with the reasons
COSTS_FILE = PROJECT_DIR / "research" / "grok" / "costs.csv"
REPORT_FILE = PROJECT_DIR / "reports" / "grok_value.md"

# ============================================================================
# SHADOW SETTINGS - fixed here before any forward data exists.
# ============================================================================
MODE = "shadow"                    # the only mode this module has; the baseline never sees a veto
UNSCHEDULED_VETO_SESSIONS = 2      # a veto with no event time lapses at the close of the 2nd trading day
MIN_FORWARD_DAYS = 63              # about 3 months of trading days (PRD: ">= 3 months")
MIN_VETOES_APPLIED = 20            # PRD: ">= 20 vetoes"
BOOTSTRAP_BLOCK_DAYS = 21
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261114
# ============================================================================


class ScoutEvent(BaseModel):
    """The exact shape of one event line. Unknown keys and loose types are refused."""
    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: StrictStr
    ticker: StrictStr
    event_type: StrictStr
    scheduled: StrictBool
    event_time_utc: Optional[StrictStr]
    first_seen_utc: StrictStr
    headline: StrictStr
    primary_source_url: Optional[StrictStr]
    corroborating_urls: list[StrictStr]
    is_surprise_vs_expectations: Literal["yes", "no", "unknown"]
    direction_hint: Literal["up", "down", "unclear"]
    magnitude_hint: Literal["low", "med", "high"]
    already_priced_evidence: StrictStr
    veto_new_entry: StrictBool
    veto_reason: Optional[StrictStr]
    confidence_evidence_quality: Literal["low", "med", "high"]
    abstain: StrictBool


def validate_line(line: str, watchlist: list, now: datetime, file_date: str | None) -> tuple:
    """Check one line. Returns (event dict, []) if it is good, or (None, [reasons]) if not."""
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None, ["not valid JSON"]
    if not isinstance(raw, dict):
        return None, ["the line is not a JSON object"]
    try:
        event = ScoutEvent(**raw).model_dump()
    except ValidationError as error:
        return None, [f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()]
    # The cross-field rules (times, sources, veto rules) live in one place: validate_events.
    problems = ve.check_event(event, watchlist, now, file_date)
    # PRD: unsourced records are rejected outright at ingest, even if the Bot abstained.
    if event["primary_source_url"] is None and not event["corroborating_urls"]:
        problems.append("unsourced: no primary source and no corroborating address")
    return (None, problems) if problems else (event, [])


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def ingest(events_dir: Path = EVENTS_DIR, store_file: Path = STORE_FILE, rejects_file: Path = REJECTS_FILE,
           watchlist: list | None = None, now: datetime | None = None) -> dict:
    """Read every event file, store new good lines, log new bad ones. Safe to run again.

    Each stored event gets `ingested_at` (the moment of this run), `source_file`
    and `source_line`. An event_id already in the store is not stored twice, and
    a rejected line already in the log is not logged twice. Existing rows are
    never changed. A missing folder or an empty file simply adds nothing.
    """
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    watchlist = ve.load_watchlist() if watchlist is None else watchlist
    known_ids = {row["event_id"] for row in _read_jsonl(store_file)}
    known_rejects = {(row["source_file"], row["source_line"]) for row in _read_jsonl(rejects_file)}

    stored, rejected, lines_seen = [], [], 0
    files = sorted(Path(events_dir).glob("*.jsonl")) if Path(events_dir).exists() else []
    for path in files:
        name = ve.FILE_NAME.match(path.name)
        file_date = name.group(1) if name else None
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if not line.strip():
                continue
            lines_seen += 1
            event, problems = validate_line(line, watchlist, now, file_date)
            if event is None and not name:
                problems.append("the file must be named YYYY-MM-DD.jsonl")
            if event is not None and name:
                if event["event_id"] not in known_ids:
                    known_ids.add(event["event_id"])
                    stored.append({**event, "ingested_at": stamp, "source_file": path.name, "source_line": number})
            elif (path.name, number) not in known_rejects:
                known_rejects.add((path.name, number))
                rejected.append({"source_file": path.name, "source_line": number, "rejected_at": stamp,
                                 "problems": problems or ["the file must be named YYYY-MM-DD.jsonl"]})
    _append_jsonl(store_file, stored)
    _append_jsonl(rejects_file, rejected)
    return {"files": len(files), "lines_seen": lines_seen, "stored_now": len(stored), "rejected_now": len(rejected),
            "stored_total": len(known_ids), "rejected_total": len(known_rejects)}


def load_store(store_file: Path = STORE_FILE) -> list:
    """Every accepted event, oldest first."""
    return _read_jsonl(store_file)


def load_ticker_map(config_file: Path = CONFIG_FILE) -> dict:
    """Which traded instrument each watched ticker stands for (`scout_ticker_map` in config).

    An empty map means a veto only applies to an instrument with exactly the same ticker.
    """
    with open(config_file) as handle:
        return dict(yaml.safe_load(handle).get("scout_ticker_map") or {})


def veto_expiry(event: dict, calendar_name: str = "NYSE") -> pd.Timestamp:
    """The moment a veto stops applying.

    Scheduled event: when the event happens (its event_time_utc).
    No event time: the close of the 2nd trading day after it was first seen.
    """
    if event["event_time_utc"]:
        return pd.Timestamp(event["event_time_utc"])
    seen = pd.Timestamp(event["first_seen_utc"])
    closes = cal.session_closes(calendar_name, seen.tz_localize(None).normalize(),
                                seen.tz_localize(None).normalize() + pd.Timedelta(days=14))
    later = closes[closes > seen]
    return later.iloc[min(UNSCHEDULED_VETO_SESSIONS, len(later)) - 1]


def active_vetoes(store: list, symbol: str, at, ticker_map: dict | None = None, calendar_name: str = "NYSE") -> list:
    """The vetoes in force for `symbol` at the moment `at` (UTC).

    A veto counts only if, at that moment, it had ALREADY BEEN INGESTED and has
    not lapsed. Events that abstain never veto. An empty or missing store gives
    an empty list, which means "no veto" - never anything else.
    """
    at = pd.Timestamp(at)
    if at.tzinfo is None:
        raise ValueError("`at` needs a time zone (UTC)")
    ticker_map = ticker_map or {}
    found = []
    for event in store or []:
        if not event.get("veto_new_entry") or event.get("abstain"):
            continue
        if ticker_map.get(event["ticker"], event["ticker"]) != symbol:
            continue
        if pd.Timestamp(event["ingested_at"]) > at:          # not known yet at that moment
            continue
        if at > veto_expiry(event, calendar_name):           # already lapsed
            continue
        found.append(event)
    return found


class VetoOverlay:
    """Wrap a strategy's weight function so that vetoed NEW entries are skipped.

    It is called exactly like the weight function it wraps. For each asset the
    strategy wants:
      - already held (it had a weight at the previous rebalance): kept as is
      - a new entry with a veto in force at the signal day's close: dropped, and
        its weight is left in cash
      - anything else: kept as is
    It never adds an asset and never raises a weight, so the overlay can only
    hold less than the baseline, never more.
    """

    def __init__(self, weights_fn, store: list, ticker_map: dict | None = None, calendar_name: str = "NYSE"):
        self.weights_fn, self.store = weights_fn, store
        self.ticker_map, self.calendar_name = ticker_map or {}, calendar_name
        self.held = set()        # what the OVERLAY held after its last rebalance
        self.skipped = []        # one record per skipped entry

    def __call__(self, histories: dict) -> tuple:
        weights, details = self.weights_fn(histories)
        kept = {}
        for symbol, weight in weights.items():
            signal_close = pd.Timestamp(histories[symbol]["available_at"].iloc[-1])
            vetoes = [] if symbol in self.held else active_vetoes(self.store, symbol, signal_close,
                                                                  self.ticker_map, self.calendar_name)
            if vetoes:
                self.skipped.append({"signal_session": histories[symbol]["session"].iloc[-1], "symbol": symbol,
                                     "weight": weight, "event_ids": [event["event_id"] for event in vetoes]})
            else:
                kept[symbol] = weight
        self.held = set(kept)
        return kept, details


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def compare(baseline, overlay, skipped: list, bars: dict) -> dict:
    """Baseline against overlay over the same days.

    `baseline` and `overlay` are BacktestResult objects from the same engine,
    data and costs. `skipped` is VetoOverlay.skipped. `bars` are the prepared
    bars, used to work out what each skipped entry would have made.
    """
    base_eq, over_eq = baseline.daily["equity"], overlay.daily["equity"]
    start = baseline.daily.attrs["starting_cash"]
    result = {
        "days": int(len(base_eq)),
        "baseline_pnl": float(base_eq.iloc[-1] - start), "overlay_pnl": float(over_eq.iloc[-1] - start),
        "incremental_pnl": float(over_eq.iloc[-1] - base_eq.iloc[-1]),
        "baseline_max_drawdown": _max_drawdown(base_eq), "overlay_max_drawdown": _max_drawdown(over_eq),
        "trades_skipped": len(skipped),
    }

    # What every NEW entry in the baseline made over its first holding month (before costs):
    # from the open it filled at to the open of the next rebalance (or the last close).
    signal_days = sorted(set(baseline.rebalances["signal_session"]))
    equity_at = baseline.daily["equity"]

    def first_month_return(symbol, signal_day):
        table = bars[symbol].set_index("session")
        after = table.index[table.index > signal_day]
        if len(after) == 0:
            return None
        entry = table.at[after[0], "tr_open"]
        later_signals = [day for day in signal_days if day > signal_day]
        exit_days = table.index[table.index > later_signals[0]] if later_signals else []
        exit_price = table.at[exit_days[0], "tr_open"] if len(exit_days) else table["tr_close"].iloc[-1]
        return float(exit_price / entry - 1.0)

    weights = baseline.rebalances[(baseline.rebalances["weight"] > 0)
                                  & (baseline.rebalances.get("is_cash_vehicle") != True)]       # noqa: E712
    held_before, entries = set(), []
    for day in signal_days:
        today = set(weights[weights["signal_session"] == day]["symbol"])
        for symbol in sorted(today - held_before):
            outcome = first_month_return(symbol, day)
            if outcome is not None:
                entries.append({"signal_session": day, "symbol": symbol, "return": outcome})
        held_before = today
    result["baseline_new_entries"] = len(entries)
    result["baseline_entry_loss_rate"] = float(np.mean([e["return"] < 0 for e in entries])) if entries else None

    would_have = []
    for item in skipped:
        outcome = first_month_return(item["symbol"], item["signal_session"])
        if outcome is None:
            continue
        before = equity_at[equity_at.index <= item["signal_session"]]
        account = float(before.iloc[-1]) if len(before) else float(start)
        would_have.append({**item, "would_have_return": outcome, "would_have_pnl": item["weight"] * account * outcome})
    result["skipped_detail"] = would_have
    result["would_have_pnl_of_skipped"] = float(sum(item["would_have_pnl"] for item in would_have))
    # Veto precision: of the entries a veto skipped, the share that would have lost money.
    result["veto_precision"] = float(np.mean([item["would_have_return"] < 0 for item in would_have])) \
        if would_have else None

    # Paired bootstrap of the daily return difference (overlay minus baseline), annualized.
    difference = (overlay.returns - baseline.returns).to_numpy()
    result["incremental_return_annualized"] = float(difference.mean() * 252)
    if len(difference) >= 2 * BOOTSTRAP_BLOCK_DAYS:
        result["incremental_return_ci"] = robustness.circular_block_bootstrap_ci(
            difference, lambda values: float(np.mean(values) * 252), BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_RESAMPLES,
            0.95, BOOTSTRAP_SEED)
    return result


def read_costs(costs_file: Path = COSTS_FILE) -> dict:
    """Total Grok spend from the manual costs file (columns: date, usd, note)."""
    if not Path(costs_file).exists():
        return {"available": False, "total_usd": None, "rows": 0}
    with open(costs_file, newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("usd") or "").strip()]
    return {"available": bool(rows), "total_usd": float(sum(float(row["usd"]) for row in rows)) if rows else None,
            "rows": len(rows)}


def recommend(comparison: dict | None, costs: dict, starting_cash: float | None = None) -> tuple:
    """Keep, drop, or not enough evidence. Returns (verdict, reason)."""
    if comparison is None:
        return "INSUFFICIENT EVIDENCE", "no forward baseline-and-overlay run exists yet"
    if comparison["days"] < MIN_FORWARD_DAYS or comparison["trades_skipped"] < MIN_VETOES_APPLIED:
        return "INSUFFICIENT EVIDENCE", (
            f"needs at least {MIN_FORWARD_DAYS} trading days and {MIN_VETOES_APPLIED} applied vetoes; "
            f"have {comparison['days']} days and {comparison['trades_skipped']} applied vetoes")
    interval = comparison.get("incremental_return_ci")
    if interval is None:
        return "INSUFFICIENT EVIDENCE", "the bootstrap interval could not be computed"
    if not costs["available"] or not starting_cash:
        return "INSUFFICIENT EVIDENCE", "Grok costs are not recorded in research/grok/costs.csv, so net benefit is unknown"
    # The overlay must pay for itself: yearly Grok cost as a share of the account.
    cost_rate = costs["total_usd"] / starting_cash * 252 / comparison["days"]
    if interval["lower"] > cost_rate:
        return "KEEP", (f"the whole 95% interval of the added return ({interval['lower']:+.2%} to "
                        f"{interval['upper']:+.2%} a year) is above the Grok cost ({cost_rate:.2%} a year)")
    if interval["upper"] < cost_rate:
        return "DROP", (f"the whole 95% interval of the added return ({interval['lower']:+.2%} to "
                        f"{interval['upper']:+.2%} a year) is below the Grok cost ({cost_rate:.2%} a year)")
    return "INCONCLUSIVE - stay in shadow", (
        f"the 95% interval of the added return ({interval['lower']:+.2%} to {interval['upper']:+.2%} a year) "
        f"straddles the Grok cost ({cost_rate:.2%} a year)")


def render_report(store: list, rejects: list, comparison: dict | None, costs: dict, tradable: list,
                  ticker_map: dict, starting_cash: float | None = None, now: datetime | None = None) -> str:
    """The text of reports/grok_value.md."""
    now = now or datetime.now(timezone.utc)
    verdict, reason = recommend(comparison, costs, starting_cash)
    vetoes = [event for event in store if event["veto_new_entry"] and not event["abstain"]]
    applicable = [event for event in vetoes if ticker_map.get(event["ticker"], event["ticker"]) in tradable]
    total_lines = len(store) + len(rejects)
    days = sorted({event["source_file"][:10] for event in store} | {row["source_file"][:10] for row in rejects})
    lines = [
        "# Grok value report: does Market Scout earn its keep?", "",
        f"**Recommendation: {verdict}.** {reason[0].upper() + reason[1:]}.", "",
        f"- Generated: {now:%Y-%m-%d %H:%M} UTC. Mode: **{MODE}** (the paper strategy ignores every veto).",
        f"- Event files ingested: {len(days)} day(s)" + (f", {days[0]} to {days[-1]}" if days else "") + ".",
        f"- Lines: {total_lines} seen, {len(store)} accepted, {len(rejects)} rejected. "
        f"Invalid-output rate: {len(rejects) / total_lines:.1%}." if total_lines else
        "- Lines: none ingested yet.",
        f"- Vetoes received: {len(vetoes)}. Vetoes on an instrument the strategy trades: **{len(applicable)}**.",
    ]
    if vetoes and not applicable:
        watched = sorted({event["ticker"] for event in vetoes})
        lines.append(f"- **No veto can apply.** The Bot's vetoes are on {', '.join(watched)}; the strategy trades "
                     f"{', '.join(tradable)}. Either the Bot watches the traded instruments, or the owner fills "
                     "`scout_ticker_map` in `config/paper.yaml`.")
    if costs["available"]:
        per_veto = f"{costs['total_usd'] / len(vetoes):.2f} dollars per veto" if vetoes else "no vetoes to divide by"
        per_event = f"{costs['total_usd'] / len(store):.2f} per accepted event" if store else "no events"
        lines.append(f"- Grok cost recorded: {costs['total_usd']:.2f} dollars over {costs['rows']} entries "
                     f"({per_veto}; {per_event}).")
    else:
        lines.append("- Grok cost: **not recorded**. Add rows to `research/grok/costs.csv` (date, usd, note).")

    lines += ["", "## Baseline against overlay", ""]
    if comparison is None:
        lines += ["Not run. A forward comparison needs daily bars dated after the first event file, and a paper",
                  "strategy running on them. Neither exists yet (see \"What is still needed\")."]
    else:
        interval = comparison.get("incremental_return_ci")
        lines += ["| | Baseline | Overlay |", "|---|---|---|",
                  f"| Profit (paper dollars) | {comparison['baseline_pnl']:,.2f} | {comparison['overlay_pnl']:,.2f} |",
                  f"| Max drawdown | {comparison['baseline_max_drawdown']:.2%} | {comparison['overlay_max_drawdown']:.2%} |",
                  "", f"- Trading days compared: {comparison['days']}.",
                  f"- Entries skipped by a veto: {comparison['trades_skipped']}. What they would have made "
                  f"(before costs): {comparison['would_have_pnl_of_skipped']:,.2f} dollars.",
                  "- Veto precision (share of skipped entries that would have lost): "
                  + (f"{comparison['veto_precision']:.0%}" if comparison["veto_precision"] is not None else "n/a")
                  + "; loss rate of all new entries in the baseline: "
                  + (f"{comparison['baseline_entry_loss_rate']:.0%}" if comparison["baseline_entry_loss_rate"] is not None
                     else "n/a") + f" ({comparison['baseline_new_entries']} entries).",
                  f"- Added return from the overlay: {comparison['incremental_return_annualized']:+.2%} a year"
                  + (f", 95% paired block-bootstrap interval {interval['lower']:+.2%} to {interval['upper']:+.2%}."
                     if interval else " (too few days for an interval).")]
    lines += ["", "## Rules this report follows", "",
              f"- A recommendation needs at least {MIN_FORWARD_DAYS} trading days and {MIN_VETOES_APPLIED} applied vetoes.",
              "- A veto only removes a new entry. It never adds, enlarges or sells a position.",
              "- An event counts from when it was ingested, so a file added later cannot veto an earlier decision.",
              "- A missing, empty or stale event file means no veto. It can never cause a trade.",
              f"- A veto with no event time lapses at the close of trading day {UNSCHEDULED_VETO_SESSIONS} after it was seen.",
              "", "## What is still needed", "",
              "1. A paper strategy running forward on new daily bars (PRD Prompt 8).",
              "2. Vetoes on instruments that strategy trades (see above).",
              "3. Grok spend entered in `research/grok/costs.csv`.",
              f"4. Time: {MIN_FORWARD_DAYS} trading days and {MIN_VETOES_APPLIED} applied vetoes.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Market Scout events in shadow mode.")
    parser.add_argument("command", choices=["ingest", "report"])
    args = parser.parse_args(argv)
    with open(CONFIG_FILE) as handle:
        config = yaml.safe_load(handle)

    if args.command == "ingest":
        summary = ingest()
        print(f"Ingested {summary['files']} file(s): {summary['stored_now']} new event(s) stored, "
              f"{summary['rejected_now']} new line(s) rejected. Totals: {summary['stored_total']} stored, "
              f"{summary['rejected_total']} rejected.")
        return 0

    text = render_report(load_store(), _read_jsonl(REJECTS_FILE), None, read_costs(), list(config["universe"]),
                         load_ticker_map(), float(config["account"]["starting_cash"]))
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
