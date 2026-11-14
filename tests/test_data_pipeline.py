"""Tests for src/data: health checks, blocking, adjustments and the lookahead guard.

No test touches the network. The bars used here are small made-up TEST
FIXTURES built on real NYSE trading days; they exist only to exercise the
checks and are never written to the project's data cache.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import adjust, lookahead, pipeline, validate
from src.data import sessions as cal
from src.data.schema import SchemaError
from src.data.sources import SourceError, normalize

CALENDAR = "NYSE"
START, END = "2024-01-02", "2024-03-28"       # a quarter of real trading days
FETCHED = pd.Timestamp("2024-04-01 12:00", tz="UTC")


class FixtureSource:
    """Stands in for a real source in tests."""
    name = "test_fixture"
    adjustment = "split_adjusted"


def make_bars(start=START, end=END, adjustment="split_adjusted") -> pd.DataFrame:
    """A clean fixture series: one bar per NYSE session, prices rising 0.5 a day from 100."""
    days = cal.expected_sessions(CALENDAR, start, end)
    close = 100.0 + 0.5 * np.arange(len(days))
    raw = pd.DataFrame({
        "session": days, "open": close - 0.2, "high": close + 1.0, "low": close - 1.0, "close": close,
        "adj_close": close, "volume": 1_000_000.0 + np.arange(len(days)), "dividend": 0.0, "split_ratio": 0.0,
    })
    source = FixtureSource()
    source.adjustment = adjustment
    return normalize(raw, "TEST", source, CALENDAR, fetched_at=FETCHED)


def check(bars) -> validate.HealthReport:
    return validate.check_bars(bars, "TEST", CALENDAR, START, END)


def failed_checks(bars) -> set:
    return {issue.check for issue in check(bars).issues}


# ============================================================================
# Health checks
# ============================================================================
def test_clean_bars_pass_every_check():
    report = check(make_bars())
    assert report.passed, report.issues
    assert report.rows == report.expected_sessions == 61     # NYSE sessions 2024-01-02..2024-03-28
    report.raise_if_failed()                                 # does not raise


def test_calendar_knows_holidays_and_closing_times():
    days = cal.expected_sessions(CALENDAR, "2024-01-01", "2024-01-31")
    assert pd.Timestamp("2024-01-01") not in days            # New Year's Day
    assert pd.Timestamp("2024-01-15") not in days            # Martin Luther King Jr. Day
    closes = cal.session_closes(CALENDAR, "2024-07-03", "2024-07-05")
    assert closes[pd.Timestamp("2024-07-03")] == pd.Timestamp("2024-07-03 17:00", tz="UTC")   # early close 1pm ET
    assert closes[pd.Timestamp("2024-07-05")] == pd.Timestamp("2024-07-05 20:00", tz="UTC")   # normal 4pm ET


def test_missing_session_is_a_gap_and_blocks():
    bars = make_bars()
    bars = bars[bars["session"] != pd.Timestamp("2024-02-14")]
    report = check(bars)
    gap = [issue for issue in report.issues if issue.check == "gap"][0]
    assert gap.count == 1 and gap.examples == ["2024-02-14"]
    with pytest.raises(validate.DataHealthError, match="gap"):
        report.raise_if_failed()


def test_duplicate_session_is_caught():
    bars = make_bars()
    assert "duplicate" in failed_checks(pd.concat([bars, bars.iloc[[5]]], ignore_index=True).sort_values("session"))


def test_out_of_order_rows_are_caught():
    bars = make_bars()
    swapped = pd.concat([bars.iloc[[1, 0]], bars.iloc[2:]], ignore_index=True)
    assert "out_of_order" in failed_checks(swapped)


def test_row_on_a_closed_day_is_caught():
    bars = make_bars()
    weekend = bars.iloc[[3]].copy()
    weekend["session"] = pd.Timestamp("2024-01-06")          # a Saturday
    assert "off_calendar" in failed_checks(pd.concat([bars, weekend], ignore_index=True).sort_values("session"))


@pytest.mark.parametrize("bad_price", [0.0, -5.0])
def test_zero_or_negative_price_is_caught(bad_price):
    bars = make_bars()
    bars.loc[bars.index[10], "low"] = bad_price
    assert "non_positive" in failed_checks(bars)


def test_blank_price_is_caught_not_filled():
    bars = make_bars()
    bars.loc[bars.index[10], "close"] = np.nan
    report = check(bars)
    assert "missing_value" in {issue.check for issue in report.issues}
    assert np.isnan(bars["close"].iloc[10])                  # the check did not fill it in


def test_high_below_low_and_close_outside_range_are_caught():
    bars = make_bars()
    bars.loc[bars.index[4], "high"] = bars["low"].iloc[4] - 1
    assert "bad_range" in failed_checks(bars)
    bars = make_bars()
    bars.loc[bars.index[4], "close"] = bars["high"].iloc[4] + 5
    assert "bad_range" in failed_checks(bars)


def test_stale_repeated_bar_is_caught():
    bars = make_bars()
    for column in ["open", "high", "low", "close", "volume"]:
        bars.loc[bars.index[20], column] = bars[column].iloc[19]     # bar 20 is a copy of bar 19
    assert "stale_repeat" in failed_checks(bars)


def test_zero_volume_is_caught():
    bars = make_bars()
    bars.loc[bars.index[7], "volume"] = 0
    assert "zero_volume" in failed_checks(bars)


def test_series_that_stops_early_is_stale():
    assert "stale_end" in failed_checks(make_bars(end="2024-03-20"))


def test_empty_result_is_caught():
    assert failed_checks(make_bars().iloc[0:0]) == {"empty"}


def test_bar_fetched_before_its_close_is_caught():
    bars = make_bars()
    bars["fetched_at"] = pd.Timestamp("2024-03-28 15:00", tz="UTC")     # last session closes 20:00 UTC
    assert "future_bar" in failed_checks(bars)


def test_available_at_is_the_session_close():
    bars = make_bars()
    first = bars.iloc[0]
    assert first["session"] == pd.Timestamp("2024-01-02")
    assert first["available_at"] == pd.Timestamp("2024-01-02 21:00", tz="UTC")   # 4pm ET in winter
    assert set(bars["source"]) == {"test_fixture"} and set(bars["adjustment"]) == {"split_adjusted"}


def test_mixed_adjustment_flags_are_refused():
    bars = make_bars()
    bars.loc[bars.index[:5], "adjustment"] = "raw"
    with pytest.raises(SchemaError, match="mixes adjustment"):
        check(bars)


# ============================================================================
# Splits and dividends
# ============================================================================
def raw_series_with_split() -> pd.DataFrame:
    """Fixture: flat at 100, then a 2-for-1 split on the 4th day so it trades at 50."""
    bars = make_bars(end="2024-01-10", adjustment="raw")              # 7 sessions
    price = np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0])
    for column, offset in [("open", 0.0), ("high", 1.0), ("low", -1.0), ("close", 0.0), ("adj_close", 0.0)]:
        bars[column] = price + offset
    bars["volume"] = 1000.0
    bars.loc[bars.index[3], "split_ratio"] = 2.0
    return bars


def test_split_adjustment_removes_the_jump():
    adjusted = adjust.apply_splits(raw_series_with_split())
    # Before the split: 100 / 2 = 50. From the split day on: unchanged 50.
    assert adjusted["close"].tolist() == [50.0] * 7
    # Volume before the split doubles (1000 * 2); after it is unchanged.
    assert adjusted["volume"].tolist() == [2000.0] * 3 + [1000.0] * 4
    assert set(adjusted["adjustment"]) == {"split_adjusted"}
    # The largest one-day move is now 0%, not -50%.
    assert adjusted["close"].pct_change().abs().max() == 0.0


def test_two_splits_compound():
    bars = raw_series_with_split()
    bars.loc[bars.index[5], "split_ratio"] = 5.0                       # a later 5-for-1
    adjusted = adjust.apply_splits(bars)
    # First three bars are before both splits: 100 / (2 * 5) = 10.
    assert adjusted["close"].iloc[0] == pytest.approx(10.0)
    # Bars 3 and 4 are after the 2-for-1 but before the 5-for-1: 50 / 5 = 10.
    assert adjusted["close"].iloc[4] == pytest.approx(10.0)


def test_split_adjustment_cannot_be_applied_twice():
    once = adjust.apply_splits(raw_series_with_split())
    with pytest.raises(adjust.AdjustmentError):
        adjust.apply_splits(once)


def test_unadjusted_split_in_a_series_marked_adjusted_is_blocked():
    """A file that claims to be split adjusted but still shows the 50% drop must fail."""
    bars = raw_series_with_split()
    bars["adjustment"] = "split_adjusted"                              # the flag lies
    report = validate.check_bars(bars, "TEST", CALENDAR, "2024-01-02", "2024-01-10")
    assert "unadjusted_split" in {issue.check for issue in report.issues}


def test_properly_adjusted_split_is_not_flagged():
    adjusted = adjust.apply_splits(raw_series_with_split())
    report = validate.check_bars(adjusted, "TEST", CALENDAR, "2024-01-02", "2024-01-10")
    assert "unadjusted_split" not in {issue.check for issue in report.issues}


def test_dividend_adjustment_makes_a_total_return_series():
    """Fixture: closes 100, 100, 99, 99 with a 1.00 dividend going ex on day 3."""
    bars = make_bars(end="2024-01-05")                                 # 4 sessions
    price = np.array([100.0, 100.0, 99.0, 99.0])
    for column, offset in [("open", 0.0), ("high", 1.0), ("low", -1.0), ("close", 0.0)]:
        bars[column] = price + offset
    bars.loc[bars.index[2], "dividend"] = 1.0
    adjusted = adjust.apply_dividends(bars)
    # Factor on the ex-day = (100 - 1) / 100 = 0.99; earlier closes become 100 * 0.99 = 99.
    assert adjusted["close"].tolist() == pytest.approx([99.0, 99.0, 99.0, 99.0])
    # The holder lost nothing: price fell 1, cash received 1. Total return is 0%.
    assert adjusted["close"].iloc[-1] / adjusted["close"].iloc[0] - 1 == pytest.approx(0.0)
    assert set(adjusted["adjustment"]) == {"split_div_adjusted"}
    with pytest.raises(adjust.AdjustmentError):
        adjust.apply_dividends(adjusted)                               # not twice


def test_total_return_from_the_sources_adjusted_close():
    bars = make_bars(end="2024-01-05")
    bars["adj_close"] = bars["close"] * 0.9
    scaled = adjust.total_return_from_adj_close(bars)
    assert scaled["close"].tolist() == pytest.approx(bars["adj_close"].tolist())
    assert scaled["open"].tolist() == pytest.approx((bars["open"] * 0.9).tolist())


# ============================================================================
# Lookahead guard
# ============================================================================
def momentum_signal(history: pd.DataFrame) -> float:
    """An honest strategy: long (1.0) if the latest close is above the average of the last 5."""
    closes = history["close"]
    if len(closes) < 5:
        return 0.0
    return 1.0 if closes.iloc[-1] > closes.iloc[-5:].mean() else 0.0


def wavy_bars() -> pd.DataFrame:
    """Fixture with ups and downs so signals actually change."""
    bars = make_bars()
    wave = 5.0 * np.sin(np.arange(len(bars)) / 3.0)
    for column in ["open", "high", "low", "close"]:
        bars[column] = bars[column] + wave
    return bars


def test_strategy_is_handed_only_past_bars():
    bars = wavy_bars()
    seen = []

    def spy(history):
        seen.append((len(history), history["session"].iloc[-1]))
        return 0.0

    lookahead.run_signals(bars, spy)
    # Call number i gets exactly i+1 rows, and the newest row is bar i itself.
    assert [length for length, _ in seen] == list(range(1, len(bars) + 1))
    assert [day for _, day in seen] == list(bars["session"])


def test_signal_at_t_does_not_change_when_the_future_changes():
    bars = wavy_bars()
    signals = lookahead.run_signals(bars, momentum_signal)
    changed = bars.copy()
    changed.loc[changed.index[40:], ["open", "high", "low", "close"]] *= 3.0     # rewrite the future
    signals_after = lookahead.run_signals(changed, momentum_signal)
    assert signals.iloc[:40].equals(signals_after.iloc[:40])
    lookahead.assert_no_lookahead(bars, momentum_signal, signals)                # passes


def test_guard_catches_a_signal_that_peeks_at_tomorrow():
    bars = wavy_bars()
    # A cheating signal computed in one pass over the whole table: "be long today
    # if TOMORROW's close is higher". shift(-1) reaches into the future.
    cheating = (bars["close"].shift(-1) > bars["close"]).astype(float)
    cheating.index = pd.DatetimeIndex(bars["session"], name="session")

    def honest_version(history):
        # What the same rule can know at bar t: there is no tomorrow yet.
        return 0.0

    with pytest.raises(lookahead.LookaheadError):
        lookahead.assert_no_lookahead(bars, honest_version, cheating)


def test_guard_catches_a_statistic_fitted_on_the_whole_series():
    """Comparing with the mean of ALL closes uses future bars; the guard must notice."""
    bars = wavy_bars()
    whole_mean = bars["close"].mean()
    leaky = (bars["close"] > whole_mean).astype(float)
    leaky.index = pd.DatetimeIndex(bars["session"], name="session")

    def same_rule_point_in_time(history):
        return 1.0 if history["close"].iloc[-1] > history["close"].mean() else 0.0

    with pytest.raises(lookahead.LookaheadError):
        lookahead.assert_no_lookahead(bars, same_rule_point_in_time, leaky)


def test_fill_is_at_the_next_bars_open():
    bars = wavy_bars()
    signals = lookahead.run_signals(bars, momentum_signal)
    fills = lookahead.next_open_fills(bars, signals)
    # One fill per signal except the last, which has no next bar.
    assert len(fills) == len(bars) - 1
    assert (fills["fill_session"] > fills["signal_session"]).all()
    # Row k: signal from bar k, filled on bar k+1 at bar k+1's open.
    assert fills["fill_session"].tolist() == bars["session"].iloc[1:].tolist()
    assert fills["fill_price"].tolist() == pytest.approx(bars["open"].iloc[1:].tolist())
    # Never the signal bar's own close.
    assert not np.allclose(fills["fill_price"].to_numpy(), bars["close"].iloc[:-1].to_numpy())


def test_visible_bars_respects_available_at():
    bars = make_bars()
    # 2024-01-03 at 20:59 UTC: one minute before that day's close, so only Jan 2 is known.
    assert len(lookahead.visible_bars(bars, pd.Timestamp("2024-01-03 20:59", tz="UTC"))) == 1
    assert len(lookahead.visible_bars(bars, pd.Timestamp("2024-01-03 21:00", tz="UTC"))) == 2
    with pytest.raises(ValueError):
        lookahead.visible_bars(bars, pd.Timestamp("2024-01-03 21:00"))            # no time zone


# ============================================================================
# Pipeline: blocking, no fabricated data, locked holdout
# ============================================================================
SETTINGS = {"universe": ["GOOD", "GAPPY", "DOWN"], "source": "yfinance", "calendar": CALENDAR,
            "history_years": 1, "cache_dir": "unused"}
NOW = pd.Timestamp("2024-03-28 22:00", tz="UTC")                                  # after that day's close


class FakeSource:
    """GOOD returns clean bars, GAPPY is missing a session, DOWN raises like a dead source."""
    name = "test_fixture"
    adjustment = "split_adjusted"

    def fetch_daily(self, symbol, start, end):
        if symbol == "DOWN":
            raise SourceError("connection refused")
        raw = make_bars(start, end)[["session", "open", "high", "low", "close", "adj_close",
                                     "volume", "dividend", "split_ratio"]]
        if symbol == "GAPPY":
            raw = raw.drop(raw.index[30])
        return raw


@pytest.fixture
def pipeline_run(tmp_path):
    report = pipeline.run(SETTINGS, source=FakeSource(), cache_dir=tmp_path / "cache",
                          report_dir=tmp_path / "reports", now=NOW)
    return report, tmp_path


def test_good_symbol_is_cached_in_three_windows(pipeline_run):
    report, tmp_path = pipeline_run
    entry = {e["symbol"]: e for e in report["symbols"]}["GOOD"]
    assert entry["status"] == "OK" and entry["cached"]
    train = pipeline.load_bars("GOOD", "train", cache_dir=tmp_path / "cache")
    validation = pipeline.load_bars("GOOD", "validation", cache_dir=tmp_path / "cache")
    total = report["requested"]["sessions"]
    assert len(train) == int(total * 0.6)
    assert len(train) + len(validation) == int(total * 0.8)
    # Windows do not overlap and are in time order.
    assert train["session"].max() < validation["session"].min()
    assert str(validation["session"].max().date()) < report["windows"]["holdout"]["first_session"]
    # The cached file carries the point-in-time columns.
    for column in ["source", "fetched_at", "available_at", "adjustment"]:
        assert column in train.columns and train[column].notna().all()


def test_symbol_with_missing_data_is_blocked_and_not_cached(pipeline_run):
    report, tmp_path = pipeline_run
    entry = {e["symbol"]: e for e in report["symbols"]}["GAPPY"]
    assert entry["status"] == "BLOCKED" and not entry["cached"]
    assert [issue["check"] for issue in entry["issues"]] == ["gap"]
    assert not (tmp_path / "cache" / "daily" / "GAPPY").exists()
    with pytest.raises(pipeline.DataBlocked):
        pipeline.load_bars("GAPPY", "train", cache_dir=tmp_path / "cache")


def test_failed_source_is_reported_as_a_gap_with_no_rows_invented(pipeline_run):
    report, tmp_path = pipeline_run
    entry = {e["symbol"]: e for e in report["symbols"]}["DOWN"]
    assert entry["status"] == "SOURCE_FAILED" and not entry["cached"]
    assert "connection refused" in entry["reason"]
    assert "rows" not in entry                                # nothing was counted because nothing exists
    assert not (tmp_path / "cache" / "daily" / "DOWN").exists()


def test_run_is_marked_failed_and_report_is_written(pipeline_run):
    report, tmp_path = pipeline_run
    assert report["passed"] is False
    assert report["blocked"] == ["GAPPY", "DOWN"]
    text = (tmp_path / "reports" / "data_health.md").read_text()
    assert "BLOCKED: GAPPY, DOWN" in text and "`gap`: 1" in text and "connection refused" in text
    assert (tmp_path / "reports" / "data_health.json").exists()


def test_holdout_cannot_be_loaded(pipeline_run):
    _, tmp_path = pipeline_run
    assert (tmp_path / "cache" / "daily" / "GOOD" / "holdout_LOCKED.parquet").exists()
    for window in ["holdout", "holdout_LOCKED", "test", "all"]:
        with pytest.raises(pipeline.DataBlocked):
            pipeline.load_bars("GOOD", window, cache_dir=tmp_path / "cache")


def test_report_contains_no_prices(pipeline_run):
    _, tmp_path = pipeline_run
    text = (tmp_path / "reports" / "data_health.json").read_text()
    for column in ["open", "high", "low", "close", "adj_close"]:
        assert f'"{column}"' not in text


def test_unset_config_blocks_before_any_fetch(tmp_path):
    config = tmp_path / "paper.yaml"
    config.write_text("universe: [SPY]\ndata:\n  source: TBD\n  calendar: NYSE\n  history_years: 10\n  cache_dir: x\n")
    with pytest.raises(pipeline.DataBlocked, match="data.source"):
        pipeline.load_settings(config)
