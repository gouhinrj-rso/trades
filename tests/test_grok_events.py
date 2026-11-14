"""Tests for Market Scout shadow mode (src/research/grok_events.py).

Events, web addresses and prices here are made-up TEST FIXTURES. No test reads
research/events/, the real ingested store, the real data cache or the trial log.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data import sessions as cal
from src.research import grok_events as ge
from src.research.backtest import Costs, prepare_bars, run_backtest

WATCHLIST = ["AAA", "BBB", "GC"]
SESSIONS = cal.expected_sessions("NYSE", "2020-12-01", "2021-08-31")
COSTS = Costs(commission_bps=2.0, slippage_bps=5.0)


def event(**changes) -> dict:
    """A valid fixture veto on BBB: seen 30 March 2021, event on 2 April 2021."""
    base = {
        "event_id": "2021-03-30-BBB-fixture-release", "ticker": "BBB", "event_type": "scheduled_release",
        "scheduled": True, "event_time_utc": "2021-04-02T12:30:00Z", "first_seen_utc": "2021-03-30T12:00:00Z",
        "headline": "Fixture: a scheduled release is due.", "primary_source_url": "https://example.gov/calendar",
        "corroborating_urls": [], "is_surprise_vs_expectations": "unknown", "direction_hint": "unclear",
        "magnitude_hint": "med", "already_priced_evidence": "", "veto_new_entry": True,
        "veto_reason": "Binary scheduled event within 2 trading days.", "confidence_evidence_quality": "high",
        "abstain": False,
    }
    base.update(changes)
    return base


def stored(**changes) -> dict:
    """The same event as it sits in the store, ingested an hour after it was seen."""
    extra = {"ingested_at": "2021-03-30T13:00:00Z", "source_file": "2021-03-30.jsonl", "source_line": 1}
    extra.update({key: changes.pop(key) for key in list(changes) if key in extra})
    return {**event(**changes), **extra}


def write_events(folder, name, lines):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("\n".join(json.dumps(line) if isinstance(line, dict) else line for line in lines))


def run_ingest(tmp_path, now="2021-03-30T13:00:00"):
    return ge.ingest(tmp_path / "events", tmp_path / "store.jsonl", tmp_path / "rejects.jsonl", WATCHLIST,
                     datetime.fromisoformat(now).replace(tzinfo=timezone.utc))


# ============================================================================
# Ingest
# ============================================================================
def test_good_lines_are_stored_with_ingested_at_and_bad_ones_logged(tmp_path):
    write_events(tmp_path / "events", "2021-03-30.jsonl",
                 [event(), "not json", event(event_id="2021-03-30-AAA-x", ticker="ZZZ")])
    summary = run_ingest(tmp_path)
    assert (summary["stored_now"], summary["rejected_now"], summary["lines_seen"]) == (1, 2, 3)
    row = ge.load_store(tmp_path / "store.jsonl")[0]
    assert row["event_id"] == "2021-03-30-BBB-fixture-release"
    assert row["ingested_at"] == "2021-03-30T13:00:00Z"
    assert (row["source_file"], row["source_line"]) == ("2021-03-30.jsonl", 1)
    rejects = [json.loads(line) for line in (tmp_path / "rejects.jsonl").read_text().splitlines()]
    assert [r["source_line"] for r in rejects] == [2, 3]
    assert rejects[0]["problems"] == ["not valid JSON"] and "watchlist" in rejects[1]["problems"][0]


def test_ingest_is_safe_to_repeat_and_never_rewrites_a_row(tmp_path):
    write_events(tmp_path / "events", "2021-03-30.jsonl", [event(), "not json"])
    run_ingest(tmp_path)
    before = (tmp_path / "store.jsonl").read_text(), (tmp_path / "rejects.jsonl").read_text()
    again = run_ingest(tmp_path, now="2021-03-31T09:00:00")
    assert (again["stored_now"], again["rejected_now"]) == (0, 0)
    assert ((tmp_path / "store.jsonl").read_text(), (tmp_path / "rejects.jsonl").read_text()) == before
    # A new file the next day is added; the old event keeps its original ingested_at.
    write_events(tmp_path / "events", "2021-03-31.jsonl",
                 [event(event_id="2021-03-31-AAA-y", ticker="AAA", first_seen_utc="2021-03-31T08:00:00Z")])
    third = run_ingest(tmp_path, now="2021-03-31T09:00:00")
    assert (third["stored_now"], third["stored_total"]) == (1, 2)
    assert [row["ingested_at"] for row in ge.load_store(tmp_path / "store.jsonl")] == \
        ["2021-03-30T13:00:00Z", "2021-03-31T09:00:00Z"]


@pytest.mark.parametrize("bad, expected", [
    ({"scheduled": "true"}, "scheduled"),                                              # text where true/false is needed
    ({"position_size": "10%"}, "position_size"),                                       # a key that must not exist
    ({"direction_hint": "sideways"}, "direction_hint"),
    ({"first_seen_utc": "2021-03-30T14:00:00Z"}, "in the future"),                     # dated after we read it
    ({"primary_source_url": None, "veto_new_entry": False, "veto_reason": None, "abstain": True}, "unsourced"),
])
def test_malformed_unsourced_and_future_dated_lines_are_rejected(tmp_path, bad, expected):
    write_events(tmp_path / "events", "2021-03-30.jsonl", [event(**bad)])
    summary = run_ingest(tmp_path)
    assert (summary["stored_now"], summary["rejected_now"]) == (0, 1)
    problems = json.loads((tmp_path / "rejects.jsonl").read_text())["problems"]
    assert any(expected in problem for problem in problems), problems


def test_missing_folder_and_empty_file_store_nothing(tmp_path):
    assert run_ingest(tmp_path)["stored_total"] == 0                                   # no folder at all
    write_events(tmp_path / "events", "2021-03-30.jsonl", [])
    assert run_ingest(tmp_path) == {"files": 1, "lines_seen": 0, "stored_now": 0, "rejected_now": 0,
                                    "stored_total": 0, "rejected_total": 0}


def test_wrongly_named_file_is_rejected(tmp_path):
    write_events(tmp_path / "events", "events.jsonl", [event()])
    assert run_ingest(tmp_path)["stored_now"] == 0


# ============================================================================
# Which vetoes are in force, point in time
# ============================================================================
def at(text):
    return pd.Timestamp(text, tz="UTC")


def test_veto_is_in_force_only_between_ingestion_and_the_event():
    store = [stored()]
    assert ge.active_vetoes(store, "BBB", at("2021-03-30 12:59")) == []          # seen by the Bot, not yet ingested by us
    assert len(ge.active_vetoes(store, "BBB", at("2021-03-30 13:00"))) == 1
    assert len(ge.active_vetoes(store, "BBB", at("2021-04-02 12:30"))) == 1      # up to the event itself
    assert ge.active_vetoes(store, "BBB", at("2021-04-02 12:31")) == []          # lapsed


def test_a_file_added_later_cannot_veto_an_earlier_decision():
    late = [stored(ingested_at="2021-04-01T09:00:00Z")]
    assert ge.active_vetoes(late, "BBB", at("2021-03-31 21:00")) == []


def test_veto_without_an_event_time_lapses_at_the_close_of_the_second_trading_day():
    store = [stored(scheduled=False, event_time_utc=None, event_type="trading_halt")]
    # Seen Tuesday 30 March 12:00 UTC. Closes after that: Tue 30 (20:00), Wed 31 (20:00) -> lapses Wed 20:00.
    assert ge.veto_expiry(store[0]) == at("2021-03-31 20:00")
    assert len(ge.active_vetoes(store, "BBB", at("2021-03-31 20:00"))) == 1
    assert ge.active_vetoes(store, "BBB", at("2021-03-31 20:01")) == []


def test_only_real_vetoes_on_the_same_instrument_count():
    moment = at("2021-03-31 20:00")
    assert ge.active_vetoes([stored(veto_new_entry=False, veto_reason=None)], "BBB", moment) == []
    assert ge.active_vetoes([stored()], "AAA", moment) == []                      # a different instrument
    assert ge.active_vetoes([], "BBB", moment) == [] and ge.active_vetoes(None, "BBB", moment) == []
    with pytest.raises(ValueError):
        ge.active_vetoes([stored()], "BBB", pd.Timestamp("2021-03-31 20:00"))     # no time zone


def test_ticker_map_points_a_watched_ticker_at_a_traded_one():
    store = [stored(ticker="GC")]
    moment = at("2021-03-31 20:00")
    assert ge.active_vetoes(store, "GLD", moment) == []                           # no map: exact match only
    assert len(ge.active_vetoes(store, "GLD", moment, ticker_map={"GC": "GLD"})) == 1
    assert ge.load_ticker_map() == {}                                             # the owner has not filled it


# ============================================================================
# Overlay: a veto can only remove a new entry
# ============================================================================
def fixture_bars(seed, drift):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + drift + 0.01 * rng.standard_normal(len(SESSIONS)))
    return prepare_bars(pd.DataFrame({
        "session": SESSIONS, "open": close * (1 + 0.002 * rng.standard_normal(len(SESSIONS))), "close": close,
        "adj_close": close, "available_at": pd.DatetimeIndex(SESSIONS).tz_localize("UTC") + pd.Timedelta(hours=20)}))


BARS = {"AAA": fixture_bars(1, 0.0005), "BBB": fixture_bars(2, -0.0010)}


def strategy(histories):
    """Fixture strategy: always 50% AAA; from the March 2021 month-end also 40% BBB."""
    weights = {"AAA": 0.5}
    if histories["AAA"]["session"].iloc[-1] >= pd.Timestamp("2021-03-31"):
        weights["BBB"] = 0.4
    return weights, {symbol: {"eligible": True} for symbol in weights}


def run(weights_fn):
    return run_backtest(BARS, SESSIONS, weights_fn, COSTS, 10_000.0, "2021-01-01", "2021-08-31")


def first_buy(result, symbol):
    fills = result.fills[(result.fills["symbol"] == symbol) & (result.fills["side"] == "buy")]
    return fills["fill_session"].min()


def test_with_no_events_the_overlay_is_exactly_the_baseline():
    """Missing Grok data must change nothing."""
    baseline = run(strategy)
    for empty in ([], None):
        overlay_fn = ge.VetoOverlay(strategy, empty)
        overlay = run(overlay_fn)
        pd.testing.assert_series_equal(baseline.daily["equity"], overlay.daily["equity"])
        pd.testing.assert_frame_equal(baseline.fills, overlay.fills)
        assert overlay_fn.skipped == []


def test_a_veto_skips_the_new_entry_and_nothing_else():
    baseline = run(strategy)
    overlay_fn = ge.VetoOverlay(strategy, [stored()])
    overlay = run(overlay_fn)
    assert first_buy(baseline, "BBB") == pd.Timestamp("2021-04-01")               # baseline enters in April
    assert first_buy(overlay, "BBB") == pd.Timestamp("2021-05-03")                # overlay waits a month
    assert [(item["symbol"], item["signal_session"]) for item in overlay_fn.skipped] == \
        [("BBB", pd.Timestamp("2021-03-31"))]
    assert overlay_fn.skipped[0]["event_ids"] == ["2021-03-30-BBB-fixture-release"]
    # AAA is untouched: same fills on the same days up to the skipped entry.
    early = lambda result: result.fills[result.fills["fill_session"] < "2021-04-01"].reset_index(drop=True)
    pd.testing.assert_frame_equal(early(baseline), early(overlay))


def test_overlay_never_holds_more_than_the_baseline_wanted():
    """A veto can only take weight away. It never adds an asset or raises a weight."""
    overlay_fn = ge.VetoOverlay(strategy, [stored()])
    overlay, baseline = run(overlay_fn), run(strategy)
    wanted = baseline.rebalances.set_index(["signal_session", "symbol"])["weight"]
    got = overlay.rebalances.set_index(["signal_session", "symbol"])["weight"]
    assert set(got.index) <= set(wanted.index)
    assert (got <= wanted.reindex(got.index) + 1e-12).all()


def test_a_veto_does_not_touch_a_position_already_held():
    """AAA is held from January. A veto on AAA in March must not sell or shrink it."""
    veto_on_held = [stored(event_id="2021-03-30-AAA-x", ticker="AAA")]
    overlay_fn = ge.VetoOverlay(strategy, veto_on_held)
    overlay, baseline = run(overlay_fn), run(strategy)
    assert overlay_fn.skipped == []
    pd.testing.assert_series_equal(baseline.daily["equity"], overlay.daily["equity"])


def test_a_veto_ingested_after_the_signal_does_not_apply():
    """Lookahead guard: the event was written on 30 March but only ingested on 1 April."""
    overlay_fn = ge.VetoOverlay(strategy, [stored(ingested_at="2021-04-01T09:00:00Z")])
    overlay = run(overlay_fn)
    assert overlay_fn.skipped == [] and first_buy(overlay, "BBB") == pd.Timestamp("2021-04-01")


def test_a_lapsed_veto_does_not_apply():
    stale = [stored(event_time_utc="2021-03-31T12:00:00Z")]                       # event passed before the signal close
    overlay_fn = ge.VetoOverlay(strategy, stale)
    run(overlay_fn)
    assert overlay_fn.skipped == []


# ============================================================================
# Comparison and recommendation
# ============================================================================
@pytest.fixture(scope="module")
def comparison():
    baseline = run(strategy)
    overlay_fn = ge.VetoOverlay(strategy, [stored()])
    overlay = run(overlay_fn)
    return ge.compare(baseline, overlay, overlay_fn.skipped, BARS), baseline, overlay


def test_comparison_numbers_match_hand_arithmetic(comparison):
    result, baseline, overlay = comparison
    assert result["days"] == len(baseline.daily)
    assert result["baseline_pnl"] == pytest.approx(baseline.daily["equity"].iloc[-1] - 10_000.0)
    assert result["incremental_pnl"] == pytest.approx(overlay.daily["equity"].iloc[-1] - baseline.daily["equity"].iloc[-1])
    assert result["trades_skipped"] == 1
    # The skipped BBB entry: from the 1 April open to the 3 May open (the next rebalance), before costs.
    table = BARS["BBB"].set_index("session")
    by_hand = table.at[pd.Timestamp("2021-05-03"), "tr_open"] / table.at[pd.Timestamp("2021-04-01"), "tr_open"] - 1
    detail = result["skipped_detail"][0]
    assert detail["would_have_return"] == pytest.approx(by_hand)
    account = baseline.daily["equity"].loc[pd.Timestamp("2021-03-31")]
    assert detail["would_have_pnl"] == pytest.approx(0.4 * account * by_hand)
    assert result["would_have_pnl_of_skipped"] == pytest.approx(detail["would_have_pnl"])
    assert result["veto_precision"] == (1.0 if by_hand < 0 else 0.0)
    # Baseline new entries: AAA in January and BBB in April.
    assert result["baseline_new_entries"] == 2
    assert "incremental_return_ci" in result
    difference = (overlay.returns - baseline.returns).mean() * 252
    assert result["incremental_return_annualized"] == pytest.approx(difference)


def enough(**changes):
    base = {"days": 70, "trades_skipped": 25, "incremental_return_ci": {"lower": 0.02, "upper": 0.05}}
    base.update(changes)
    return base


COSTS_KNOWN = {"available": True, "total_usd": 27.78, "rows": 3}      # 27.78 / 10,000 * 252 / 70 = 1.00% a year


@pytest.mark.parametrize("comparison_numbers, costs, expected", [
    (None, COSTS_KNOWN, "INSUFFICIENT EVIDENCE"),
    (enough(days=62), COSTS_KNOWN, "INSUFFICIENT EVIDENCE"),                       # under 3 months
    (enough(trades_skipped=19), COSTS_KNOWN, "INSUFFICIENT EVIDENCE"),             # under 20 vetoes
    (enough(), {"available": False, "total_usd": None, "rows": 0}, "INSUFFICIENT EVIDENCE"),
    (enough(), COSTS_KNOWN, "KEEP"),                                               # +2% to +5% clears 1%
    (enough(incremental_return_ci={"lower": -0.04, "upper": 0.005}), COSTS_KNOWN, "DROP"),
    (enough(incremental_return_ci={"lower": -0.01, "upper": 0.03}), COSTS_KNOWN, "INCONCLUSIVE - stay in shadow"),
])
def test_recommendation_rules(comparison_numbers, costs, expected):
    assert ge.recommend(comparison_numbers, costs, 10_000.0)[0] == expected


def test_thresholds_are_the_prds():
    assert (ge.MIN_FORWARD_DAYS, ge.MIN_VETOES_APPLIED, ge.MODE) == (63, 20, "shadow")


def test_costs_file_is_read_and_a_missing_one_is_not_guessed(tmp_path):
    assert ge.read_costs(tmp_path / "none.csv") == {"available": False, "total_usd": None, "rows": 0}
    path = tmp_path / "costs.csv"
    path.write_text("date,usd,note\n2021-03-30,0.50,fixture\n2021-03-31,0.75,fixture\n")
    assert ge.read_costs(path) == {"available": True, "total_usd": 1.25, "rows": 2}


def test_report_says_when_no_veto_can_apply():
    text = ge.render_report([stored(ticker="GC")], [{"source_file": "2021-03-30.jsonl", "source_line": 3}], None,
                            {"available": False, "total_usd": None, "rows": 0}, ["SPY", "GLD"], {}, 10_000.0)
    assert "**Recommendation: INSUFFICIENT EVIDENCE.**" in text
    assert "**No veto can apply.**" in text and "GC" in text
    assert "Invalid-output rate: 50.0%" in text
    assert "not recorded" in text and "Mode: **shadow**" in text


def test_report_with_a_comparison_shows_both_ledgers(comparison):
    result, _, _ = comparison
    text = ge.render_report([stored()], [], result, COSTS_KNOWN, ["AAA", "BBB"], {}, 10_000.0)
    assert "| Profit (paper dollars) |" in text and "Entries skipped by a veto: 1" in text
    assert "INSUFFICIENT EVIDENCE" in text                    # one veto and five months is not enough
