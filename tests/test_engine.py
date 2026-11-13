"""Tests for engine.py.

All tests use made-up bars. None of them reads anything from data/.
"""

import pytest

import engine


def make_bars(prices):
    """Build bars from a list of (open, close) pairs."""
    bars = []
    for index, (open_price, close_price) in enumerate(prices):
        bars.append({
            "timestamp": f"bar-{index:03d}",
            "open": open_price,
            "high": max(open_price, close_price),
            "low": min(open_price, close_price),
            "close": close_price,
            "volume": 1.0,
        })
    return bars


# A price path that moves every bar, so the fill bar matters.
MOVING_BARS = make_bars([
    (100, 101), (101, 103), (103, 106), (106, 110), (110, 115),
    (115, 121), (121, 128), (128, 136), (136, 145), (145, 155),
])


# --- Required test 1: shifting every signal one bar later changes the results ---
def test_shifting_signals_one_bar_later_changes_results():
    # Go long after bar 1, go flat after bar 5.
    signals = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # The same decisions, each made one bar later.
    shifted = [0] + signals[:-1]

    on_time = engine.run_backtest(MOVING_BARS, signals)
    late = engine.run_backtest(MOVING_BARS, shifted)

    # If the engine ignored signal timing, these would be identical.
    assert on_time["total_return"] != late["total_return"]
    assert on_time["trades"][0]["entry_price"] != late["trades"][0]["entry_price"]
    assert on_time["trades"][0]["exit_price"] != late["trades"][0]["exit_price"]


def test_fill_is_at_next_bar_open_not_signal_bar():
    # Signal goes long at the close of bar 1 -> must fill at the OPEN of bar 2.
    signals = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    result = engine.run_backtest(MOVING_BARS, signals)

    trade = result["trades"][0]
    assert trade["entry_time"] == "bar-002"
    assert trade["entry_price"] == MOVING_BARS[2]["open"]      # 103, not bar 1's 101 or 103 close
    assert trade["exit_time"] == "bar-006"                     # flat decided at bar 5 -> fill bar 6
    assert trade["exit_price"] == MOVING_BARS[6]["open"]


# --- Required test 2: fees are really charged ---
def test_fees_are_really_charged():
    # Flat prices: with no fees a round trip would return exactly zero.
    flat_bars = make_bars([(100, 100)] * 6)
    signals = [1, 1, 1, 0, 0, 0]   # one round trip: buy at bar 1, sell at bar 4

    with_fees = engine.run_backtest(flat_bars, signals)             # default fee
    no_fees = engine.run_backtest(flat_bars, signals, fee_bps_per_side=0.0)

    assert no_fees["total_return"] == 0.0
    assert no_fees["fees_paid"] == 0.0

    # Two sides at 2 bps each: equity is multiplied by (1 - 0.0002) twice.
    expected = (1 - 0.0002) ** 2 - 1
    assert with_fees["trade_count"] == 1
    assert with_fees["total_return"] == pytest.approx(expected)
    assert with_fees["total_return"] < 0
    assert with_fees["fees_paid"] > 0


def test_default_fee_is_two_bps_per_side():
    assert engine.FEE_BPS_PER_SIDE == 2.0


# --- Extra look-ahead checks on the two strategy families ---
@pytest.mark.parametrize("candidate", engine.build_candidates(), ids=lambda c: c["name"])
def test_signals_do_not_depend_on_future_bars(candidate):
    # A wavy price path long enough for the slowest settings.
    closes = [100 + 5 * ((i * 7) % 13 - 6) / 6 + (i % 29) * 0.3 for i in range(300)]

    full = engine.make_signals(candidate, closes)

    # Cutting off the future must not change any earlier signal.
    for cut in (130, 200, 299):
        assert engine.make_signals(candidate, closes[:cut]) == full[:cut]

    # Changing only the last price must not change any earlier signal either.
    changed = closes[:-1] + [closes[-1] * 2]
    assert engine.make_signals(candidate, changed)[:-1] == full[:-1]


def test_validation_run_starts_flat_and_only_trades_after_trade_start():
    # Always-long signals, but trading only allowed from bar 5 onward.
    result = engine.run_backtest(MOVING_BARS, [1] * 10, trade_start=5)

    assert len(result["equity_curve"]) == 5                 # bars 5..9 only
    assert result["trades"][0]["entry_time"] == "bar-005"   # decided at bar 4, filled at bar 5 open


def test_engine_refuses_locked_files(tmp_path):
    with pytest.raises(PermissionError):
        engine.load_bars(tmp_path / "test_1h_LOCKED.json")
    with pytest.raises(PermissionError):
        engine.load_bars(tmp_path / "bars_1h.json")
    with pytest.raises(PermissionError):
        engine.load_bars(tmp_path / "test_5m_LOCKED.json")
    with pytest.raises(PermissionError):
        engine.load_bars(tmp_path / "bars_5m.json")


def test_grid_size():
    candidates = engine.build_candidates()
    assert len(candidates) == 18
    assert len({c["name"] for c in candidates}) == 18
