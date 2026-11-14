"""Tests for the accounting ledger. Every expected number is worked out by hand in a comment.

The instruments below are TEST EXAMPLES ONLY. Their multipliers and tick sizes
are made-up round numbers chosen to make the arithmetic easy to follow; they
are not the specification of any real contract.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.accounting import (
    Fill,
    InsufficientCash,
    Instrument,
    Ledger,
    LedgerError,
    MissingPrice,
)

# ---- test example instruments (not real contract specs) ---------------------
STOCK = Instrument("STOCK", "equity", multiplier=1.0, tick_size=0.01)
COIN = Instrument("COIN", "crypto", multiplier=1.0, tick_size=0.01)
# An example future: 50 dollars per point, quarter-point ticks -> 12.50 per tick.
FUT_A = Instrument("FUT_A", "future", multiplier=50.0, tick_size=0.25)   # the expiring contract
FUT_B = Instrument("FUT_B", "future", multiplier=50.0, tick_size=0.25)   # the next contract
FUT_SMALL = Instrument("FUT_SMALL", "future", multiplier=5.0, tick_size=0.25)   # different multiplier
ALL = [STOCK, COIN, FUT_A, FUT_B, FUT_SMALL]

START = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)


def at(hours: int) -> datetime:
    """A timezone-aware timestamp `hours` after the start of the test clock."""
    return START + timedelta(hours=hours)


def make_ledger(cash: float = 10_000.0) -> Ledger:
    """A fresh ledger that knows every test instrument."""
    return Ledger(cash, ALL)


# =============================================================================
# Equity: buy, sell, cash, fees, realized P&L, slippage
# =============================================================================
def test_buy_then_sell_equity():
    ledger = make_ledger(10_000.0)

    # Buy 10 at 100.50 (decided against 100.00), fee 1.00.
    #   cash     = 10,000 - 10 * 100.50 - 1.00 = 10,000 - 1,005 - 1 = 8,994.00
    #   slippage = (100.50 - 100.00) * 10 = 5.00  (paid 0.50 too much on 10 shares)
    ledger.record_fill(Fill(at(0), "STOCK", 10, 100.50, fee=1.00, reference_price=100.00))
    assert ledger.cash == pytest.approx(8_994.00)
    assert ledger.position("STOCK").quantity == pytest.approx(10)
    assert ledger.position("STOCK").avg_price == pytest.approx(100.50)
    assert ledger.gross_realized_pnl == pytest.approx(0.0)
    assert ledger.total_slippage_cost == pytest.approx(5.00)

    # Sell 10 at 110.00 (decided against 110.20), fee 1.10.
    #   cash     = 8,994 + 10 * 110.00 - 1.10 = 8,994 + 1,100 - 1.10 = 10,092.90
    #   realized = (110.00 - 100.50) * 10 = 95.00   (gross: fees are not in it)
    #   fees     = 1.00 + 1.10 = 2.10
    #   slippage = (110.00 - 110.20) * -10 = 2.00 ; running total 5.00 + 2.00 = 7.00
    ledger.record_fill(Fill(at(1), "STOCK", -10, 110.00, fee=1.10, reference_price=110.20))
    assert ledger.cash == pytest.approx(10_092.90)
    assert ledger.gross_realized_pnl == pytest.approx(95.00)
    assert ledger.total_fees == pytest.approx(2.10)
    assert ledger.total_slippage_cost == pytest.approx(7.00)
    assert ledger.position("STOCK").is_flat
    assert ledger.positions() == {}

    # Cross-check: 10,000 + 95.00 gross - 2.10 fees = 10,092.90. Fees counted once.
    assert ledger.equity({}) == pytest.approx(10_092.90)
    assert ledger.check_identity({})
    assert len(ledger.fills) == 2


# =============================================================================
# Average cost: add, partial sell, full close, flip
# =============================================================================
def test_average_cost_over_two_buys_then_partial_and_full_close():
    ledger = make_ledger(10_000.0)

    # Buy 10 at 100, then 30 at 120.
    #   average = (10 * 100 + 30 * 120) / 40 = (1,000 + 3,600) / 40 = 115.00
    #   cash    = 10,000 - 1,000 - 3,600 = 5,400
    ledger.record_fill(Fill(at(0), "STOCK", 10, 100.0))
    ledger.record_fill(Fill(at(1), "STOCK", 30, 120.0))
    assert ledger.position("STOCK").quantity == pytest.approx(40)
    assert ledger.position("STOCK").avg_price == pytest.approx(115.0)
    assert ledger.cash == pytest.approx(5_400.0)
    assert ledger.gross_realized_pnl == pytest.approx(0.0)

    # Partial sell: 20 at 125.
    #   realized = (125 - 115) * 20 = 200 ; average stays 115 ; 20 left
    #   cash     = 5,400 + 20 * 125 = 7,900
    ledger.record_fill(Fill(at(2), "STOCK", -20, 125.0))
    assert ledger.position("STOCK").quantity == pytest.approx(20)
    assert ledger.position("STOCK").avg_price == pytest.approx(115.0)
    assert ledger.gross_realized_pnl == pytest.approx(200.0)
    assert ledger.cash == pytest.approx(7_900.0)

    # Full close: the last 20 at 110.
    #   realized = (110 - 115) * 20 = -100 ; total 200 - 100 = 100
    #   cash     = 7,900 + 20 * 110 = 10,100  (= 10,000 + 100, no fees)
    ledger.record_fill(Fill(at(3), "STOCK", -20, 110.0))
    assert ledger.position("STOCK").is_flat
    assert ledger.position("STOCK").avg_price == 0.0
    assert ledger.gross_realized_pnl == pytest.approx(100.0)
    assert ledger.realized_pnl_by_symbol["STOCK"] == pytest.approx(100.0)
    assert ledger.cash == pytest.approx(10_100.0)


def test_flip_long_to_short_opens_remainder_at_fill_price():
    ledger = make_ledger(10_000.0)

    # Buy 10 at 100 -> cash 9,000. Then sell 25 at 104.
    #   the first 10 close the long: realized = (104 - 100) * 10 = 40
    #   the other 15 open a short at the fill price: -15 @ 104
    #   cash = 9,000 + 25 * 104 = 9,000 + 2,600 = 11,600
    ledger.record_fill(Fill(at(0), "STOCK", 10, 100.0))
    ledger.record_fill(Fill(at(1), "STOCK", -25, 104.0))
    assert ledger.position("STOCK").quantity == pytest.approx(-15)
    assert ledger.position("STOCK").avg_price == pytest.approx(104.0)
    assert ledger.gross_realized_pnl == pytest.approx(40.0)
    assert ledger.cash == pytest.approx(11_600.0)

    # At 104 the new short has made nothing yet:
    #   equity = 11,600 + (-15 * 104) = 11,600 - 1,560 = 10,040 = 10,000 + 40
    assert ledger.mark_to_market({"STOCK": 104.0})["total"] == pytest.approx(0.0)
    assert ledger.equity({"STOCK": 104.0}) == pytest.approx(10_040.0)
    assert ledger.check_identity({"STOCK": 104.0})


def test_fractional_crypto_quantities_close_cleanly():
    ledger = make_ledger(10_000.0)

    # Buy 0.3 at 2,000, sell 0.1 and then 0.2 at 2,100. In floats 0.3 - 0.1 - 0.2
    # is not exactly 0; the ledger must still call the position flat.
    #   realized = (2,100 - 2,000) * 0.3 = 30
    ledger.record_fill(Fill(at(0), "COIN", 0.3, 2_000.0))
    ledger.record_fill(Fill(at(1), "COIN", -0.1, 2_100.0))
    ledger.record_fill(Fill(at(2), "COIN", -0.2, 2_100.0))
    assert ledger.position("COIN").is_flat
    assert ledger.positions() == {}
    assert ledger.gross_realized_pnl == pytest.approx(30.0)
    assert ledger.cash == pytest.approx(10_030.0)


# =============================================================================
# Short sales
# =============================================================================
def test_short_sale_profit_when_price_falls():
    ledger = make_ledger(10_000.0)
    # Sell short 10 at 50, buy back at 45: realized = (45 - 50) * 10 * -1 = +50
    ledger.record_fill(Fill(at(0), "STOCK", -10, 50.0))
    assert ledger.position("STOCK").quantity == pytest.approx(-10)
    # While short, a price of 48 is an unrealized gain: (48 - 50) * -10 = +20
    assert ledger.mark_to_market({"STOCK": 48.0})["by_symbol"]["STOCK"] == pytest.approx(20.0)
    ledger.record_fill(Fill(at(1), "STOCK", 10, 45.0))
    assert ledger.gross_realized_pnl == pytest.approx(50.0)
    assert ledger.cash == pytest.approx(10_050.0)      # 10,000 + 500 - 450


def test_short_sale_loss_when_price_rises():
    ledger = make_ledger(10_000.0)
    # Sell short 10 at 50, buy back at 55: realized = (55 - 50) * 10 * -1 = -50
    ledger.record_fill(Fill(at(0), "STOCK", -10, 50.0))
    ledger.record_fill(Fill(at(1), "STOCK", 10, 55.0))
    assert ledger.gross_realized_pnl == pytest.approx(-50.0)
    assert ledger.cash == pytest.approx(9_950.0)       # 10,000 + 500 - 550


# =============================================================================
# Refusals: insufficient cash, bad timestamps, unknown symbol, zero quantity
# =============================================================================
def test_buy_with_insufficient_cash_raises_and_changes_nothing():
    ledger = make_ledger(1_000.0)
    # 11 * 100 = 1,100 is more than the 1,000 held.
    with pytest.raises(InsufficientCash):
        ledger.record_fill(Fill(at(0), "STOCK", 11, 100.0))
    # 10 * 100 = 1,000 fits on its own, but the 1.00 fee tips it to 1,001.
    with pytest.raises(InsufficientCash):
        ledger.record_fill(Fill(at(0), "STOCK", 10, 100.0, fee=1.0, reference_price=99.0))

    # Nothing moved.
    assert ledger.cash == pytest.approx(1_000.0)
    assert ledger.positions() == {}
    assert ledger.fills == []
    assert ledger.total_fees == 0.0
    assert ledger.total_slippage_cost == 0.0
    assert ledger.gross_realized_pnl == 0.0

    # The refused fills did not advance the clock either: an exact fit still works.
    ledger.record_fill(Fill(at(0), "STOCK", 10, 100.0))
    assert ledger.cash == pytest.approx(0.0)


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        Fill(datetime(2026, 1, 5, 14, 0), "STOCK", 1, 100.0)      # no timezone
    ledger = make_ledger()
    with pytest.raises(ValueError):
        ledger.deposit(datetime(2026, 1, 5, 14, 0), 100.0)
    with pytest.raises(ValueError):
        ledger.record_equity(datetime(2026, 1, 5, 14, 0), {})


def test_out_of_order_fill_rejected():
    ledger = make_ledger()
    ledger.record_fill(Fill(at(5), "STOCK", 1, 100.0))
    with pytest.raises(LedgerError):
        ledger.record_fill(Fill(at(4), "STOCK", 1, 100.0))        # one hour earlier
    assert len(ledger.fills) == 1
    assert ledger.position("STOCK").quantity == pytest.approx(1)
    # The same timestamp is fine (two fills in the same bar).
    ledger.record_fill(Fill(at(5), "STOCK", 1, 100.0))
    assert len(ledger.fills) == 2


def test_unknown_symbol_rejected():
    ledger = make_ledger()
    with pytest.raises(LedgerError):
        ledger.record_fill(Fill(at(0), "NOT_REGISTERED", 1, 100.0))
    with pytest.raises(LedgerError):
        ledger.position("NOT_REGISTERED")
    assert ledger.fills == []
    assert ledger.cash == pytest.approx(10_000.0)


def test_zero_quantity_and_other_bad_fills_rejected():
    with pytest.raises(ValueError):
        Fill(at(0), "STOCK", 0, 100.0)             # zero quantity
    with pytest.raises(ValueError):
        Fill(at(0), "STOCK", 1, 0.0)               # price must be above 0
    with pytest.raises(ValueError):
        Fill(at(0), "STOCK", 1, 100.0, fee=-0.01)  # fee cannot be negative
    with pytest.raises(ValueError):
        Fill(at(0), "STOCK", 1, 100.0, reference_price=0.0)


# =============================================================================
# Slippage sign
# =============================================================================
def test_slippage_signs():
    # Buy above the reference costs money: (100.10 - 100.00) * 20 = +2.00
    assert Fill(at(0), "STOCK", 20, 100.10, reference_price=100.00).slippage_cost() == pytest.approx(2.00)
    # Sell below the reference costs money: (99.90 - 100.00) * -20 = +2.00
    assert Fill(at(0), "STOCK", -20, 99.90, reference_price=100.00).slippage_cost() == pytest.approx(2.00)
    # Sell above the reference is a gain, so a negative cost: (100.10 - 100.00) * -20 = -2.00
    assert Fill(at(0), "STOCK", -20, 100.10, reference_price=100.00).slippage_cost() == pytest.approx(-2.00)
    # Buy below the reference is also a gain: (99.90 - 100.00) * 20 = -2.00
    assert Fill(at(0), "STOCK", 20, 99.90, reference_price=100.00).slippage_cost() == pytest.approx(-2.00)
    # No reference price -> nothing to compare against -> 0.
    assert Fill(at(0), "STOCK", 20, 100.10).slippage_cost() == 0.0


def test_slippage_uses_multiplier_and_accumulates_in_ledger():
    ledger = make_ledger(100_000.0)
    # Future, multiplier 50: buy 2 one tick (0.25) above the reference.
    #   (4000.25 - 4000.00) * 2 * 50 = 25.00  (= 2 contracts * 12.50 per tick)
    ledger.record_fill(Fill(at(0), "FUT_A", 2, 4000.25, reference_price=4000.00))
    assert ledger.total_slippage_cost == pytest.approx(25.00)
    # Sell 2 one tick above the reference: (4010.25 - 4010.00) * -2 * 50 = -25.00 ; total 0
    ledger.record_fill(Fill(at(1), "FUT_A", -2, 4010.25, reference_price=4010.00))
    assert ledger.total_slippage_cost == pytest.approx(0.0)


# =============================================================================
# Mark to market
# =============================================================================
def test_mark_to_market_values_and_missing_price():
    ledger = make_ledger(100_000.0)
    ledger.record_fill(Fill(at(0), "STOCK", 100, 20.0))     # long 100 @ 20
    ledger.record_fill(Fill(at(1), "FUT_A", -1, 4000.0))    # short 1 @ 4000

    # STOCK at 21: (21 - 20) * 100 = +100
    # FUT_A at 4002: (4002 - 4000) * -1 * 50 = -100
    marks = ledger.mark_to_market({"STOCK": 21.0, "FUT_A": 4002.0})
    assert marks["by_symbol"]["STOCK"] == pytest.approx(100.0)
    assert marks["by_symbol"]["FUT_A"] == pytest.approx(-100.0)
    assert marks["total"] == pytest.approx(0.0)

    # A held symbol without a price is an error, never a guess.
    with pytest.raises(MissingPrice):
        ledger.mark_to_market({"STOCK": 21.0})
    with pytest.raises(MissingPrice):
        ledger.equity({"FUT_A": 4002.0})
    with pytest.raises(MissingPrice):
        ledger.summary({})
    with pytest.raises(MissingPrice):
        ledger.record_equity(at(2), {"STOCK": 21.0})
    with pytest.raises(MissingPrice):
        ledger.mark_to_market({"STOCK": 21.0, "FUT_A": float("nan")})
    assert ledger.equity_curve() == []

    # Prices for symbols that are not held are simply ignored.
    assert ledger.mark_to_market({"STOCK": 21.0, "FUT_A": 4002.0, "COIN": 1.0})["total"] == pytest.approx(0.0)


# =============================================================================
# Cash flows and the unit-value (time-weighted) series
# =============================================================================
def test_deposit_does_not_change_unit_value_and_matches_hand_computed_twr():
    ledger = make_ledger(10_000.0)

    # Start: 10,000 units at 1.00.
    first = ledger.record_equity(at(0), {})
    assert first["unit_value"] == pytest.approx(1.0)
    assert first["adjusted_return"] == pytest.approx(0.0)

    # Buy 100 at 50 (no fee): cash 5,000, holding worth 5,000.
    ledger.record_fill(Fill(at(1), "STOCK", 100, 50.0))

    # Price 55: equity = 5,000 + 100 * 55 = 10,500 ; unit value = 10,500 / 10,000 = 1.05
    before = ledger.record_equity(at(2), {"STOCK": 55.0})
    assert before["equity"] == pytest.approx(10_500.0)
    assert before["unit_value"] == pytest.approx(1.05)

    # Deposit 2,100 at unit value 1.05 -> 2,100 / 1.05 = 2,000 new units -> 12,000 units.
    flow = ledger.deposit(at(3), 2_100.0, {"STOCK": 55.0})
    assert flow["unit_value"] == pytest.approx(1.05)
    assert flow["units_after"] == pytest.approx(12_000.0)

    # Same price right after: equity = 7,100 + 5,500 = 12,600 ; 12,600 / 12,000 = 1.05 (unchanged)
    after = ledger.record_equity(at(3), {"STOCK": 55.0})
    assert after["equity"] == pytest.approx(12_600.0)
    assert after["net_cash_flow_to_date"] == pytest.approx(2_100.0)
    assert after["unit_value"] == pytest.approx(before["unit_value"])
    assert after["adjusted_return"] == pytest.approx(before["adjusted_return"])

    # Price 60.50: equity = 7,100 + 100 * 60.50 = 13,150 ; unit value = 13,150 / 12,000 = 1.0958333...
    last = ledger.record_equity(at(4), {"STOCK": 60.50})
    assert last["equity"] == pytest.approx(13_150.0)

    # Hand-computed time-weighted return, chaining the two sub-periods around the deposit:
    #   before the deposit: 10,500 / 10,000 = 1.05
    #   after the deposit:  13,150 / 12,600 = 1.0436507936...
    #   1.05 * 1.0436507936... = 1.0958333...  -> +9.58333...%
    hand_twr = (10_500.0 / 10_000.0) * (13_150.0 / 12_600.0) - 1.0
    assert hand_twr == pytest.approx(0.0958333333)
    assert last["unit_value"] == pytest.approx(1.0958333333)
    assert last["adjusted_return"] == pytest.approx(hand_twr)

    # The naive number (13,150 / 10,000 - 1 = 31.5%) would count the deposit as profit.
    assert last["adjusted_return"] < 0.10

    curve = ledger.equity_curve()
    assert [point["timestamp"] for point in curve] == [at(0), at(2), at(3), at(4)]
    assert set(curve[0]) == {"timestamp", "equity", "net_cash_flow_to_date", "unit_value", "adjusted_return"}
    assert ledger.check_identity({"STOCK": 60.50})


def test_withdrawal_does_not_change_unit_value():
    ledger = make_ledger(10_000.0)
    ledger.record_fill(Fill(at(0), "STOCK", 100, 50.0))       # cash 5,000

    # Price 45: equity = 5,000 + 4,500 = 9,500 ; unit value 0.95
    before = ledger.record_equity(at(1), {"STOCK": 45.0})
    assert before["unit_value"] == pytest.approx(0.95)

    # Withdraw 1,900 at 0.95 -> 1,900 / 0.95 = 2,000 units sold -> 8,000 units left.
    flow = ledger.withdraw(at(2), 1_900.0, {"STOCK": 45.0})
    assert flow["amount"] == pytest.approx(-1_900.0)
    assert flow["units_after"] == pytest.approx(8_000.0)
    assert ledger.cash == pytest.approx(3_100.0)

    # Equity = 3,100 + 4,500 = 7,600 ; 7,600 / 8,000 = 0.95 (unchanged)
    after = ledger.record_equity(at(2), {"STOCK": 45.0})
    assert after["equity"] == pytest.approx(7_600.0)
    assert after["net_cash_flow_to_date"] == pytest.approx(-1_900.0)
    assert after["unit_value"] == pytest.approx(0.95)
    assert after["adjusted_return"] == pytest.approx(-0.05)
    assert ledger.check_identity({"STOCK": 45.0})


def test_cash_flow_refusals():
    ledger = make_ledger(1_000.0)
    with pytest.raises(InsufficientCash):
        ledger.withdraw(at(0), 1_000.01)            # more than the cash held
    with pytest.raises(ValueError):
        ledger.deposit(at(0), 0.0)                  # must be above 0
    with pytest.raises(ValueError):
        ledger.withdraw(at(0), -5.0)
    assert ledger.cash == pytest.approx(1_000.0)
    assert ledger.cash_flows == []
    assert ledger.net_cash_flow == 0.0

    # With a position open, a cash flow needs prices to value the units.
    ledger.record_fill(Fill(at(1), "STOCK", 5, 100.0))
    with pytest.raises(MissingPrice):
        ledger.deposit(at(2), 100.0)
    assert ledger.cash == pytest.approx(500.0)
    assert ledger.cash_flows == []

    # A cash flow dated before the last fill is refused like an out-of-order fill.
    with pytest.raises(LedgerError):
        ledger.deposit(at(0), 100.0, {"STOCK": 100.0})


# =============================================================================
# The accounting identity
# =============================================================================
def test_accounting_identity_after_mixed_sequence():
    ledger = make_ledger(50_000.0)

    # 1. Buy 100 STOCK at 20, fee 1:        cash = 50,000 - 2,000 - 1         = 47,999
    ledger.record_fill(Fill(at(0), "STOCK", 100, 20.0, fee=1.0))
    assert ledger.cash == pytest.approx(47_999.0)
    # 2. Buy 2 FUT_A at 4000, fee 5:        futures move no notional -> cash  = 47,994
    ledger.record_fill(Fill(at(1), "FUT_A", 2, 4000.0, fee=5.0))
    assert ledger.cash == pytest.approx(47_994.0)
    # 3. Deposit 5,000:                     cash = 47,994 + 5,000             = 52,994
    ledger.deposit(at(2), 5_000.0, {"STOCK": 21.0, "FUT_A": 4004.0})
    assert ledger.cash == pytest.approx(52_994.0)
    # 4. Sell 40 STOCK at 22, fee 1:        realized (22 - 20) * 40 = 80
    #                                       cash = 52,994 + 880 - 1           = 53,873
    ledger.record_fill(Fill(at(3), "STOCK", -40, 22.0, fee=1.0))
    assert ledger.cash == pytest.approx(53_873.0)
    # 5. Sell 1 FUT_A at 4010, fee 5:       realized (4010 - 4000) * 1 * 50 = 500
    #                                       cash = 53,873 + 500 - 5           = 54,368
    ledger.record_fill(Fill(at(4), "FUT_A", -1, 4010.0, fee=5.0))
    assert ledger.cash == pytest.approx(54_368.0)

    # Mark at STOCK 23, FUT_A 3990:
    #   STOCK: 60 left @ 20 -> unrealized (23 - 20) * 60 = 180 ; market value 60 * 23 = 1,380
    #   FUT_A: 1 left @ 4000 -> unrealized (3990 - 4000) * 1 * 50 = -500
    #   equity = cash 54,368 + stock value 1,380 + futures unrealized -500 = 55,248
    prices = {"STOCK": 23.0, "FUT_A": 3990.0}
    assert ledger.equity(prices) == pytest.approx(55_248.0)

    # Identity, by hand:
    #   50,000 start + 5,000 deposit + 580 realized (80 + 500) + (-320) unrealized (180 - 500)
    #   - 12 fees (1 + 5 + 1 + 5) = 55,248
    assert 50_000.0 + 5_000.0 + 580.0 - 320.0 - 12.0 == pytest.approx(55_248.0)
    assert ledger.check_identity(prices)

    summary = ledger.summary(prices)
    assert summary["cash"] == pytest.approx(54_368.0)
    assert summary["equity"] == pytest.approx(55_248.0)
    assert summary["realized_pnl"] == pytest.approx(580.0)
    assert summary["unrealized_pnl"] == pytest.approx(-320.0)
    assert summary["total_fees"] == pytest.approx(12.0)
    assert summary["total_slippage_cost"] == pytest.approx(0.0)
    assert summary["net_pnl"] == pytest.approx(248.0)          # 580 - 320 - 12
    assert summary["net_external_cash_flow"] == pytest.approx(5_000.0)
    assert summary["number_of_fills"] == 4
    assert summary["positions"] == {
        "STOCK": {"quantity": pytest.approx(60), "avg_price": pytest.approx(20.0)},
        "FUT_A": {"quantity": pytest.approx(1), "avg_price": pytest.approx(4000.0)},
    }


def test_check_identity_raises_when_books_are_tampered_with():
    ledger = make_ledger(10_000.0)
    ledger.record_fill(Fill(at(0), "STOCK", 10, 100.0, fee=1.0))
    assert ledger.check_identity({"STOCK": 100.0})
    ledger.cash += 0.01                       # one cent appears from nowhere
    with pytest.raises(LedgerError):
        ledger.check_identity({"STOCK": 100.0})


# =============================================================================
# Instruments and futures arithmetic
# =============================================================================
def test_tick_value_and_notional_use_multiplier():
    # Test example future: tick_value = 0.25 * 50 = 12.50
    assert FUT_A.tick_value == pytest.approx(12.50)
    # notional = 4000 * 3 * 50 = 600,000 ; signed like the quantity
    assert FUT_A.notional(4000.0, 3) == pytest.approx(600_000.0)
    assert FUT_A.notional(4000.0, -3) == pytest.approx(-600_000.0)
    # Equity: multiplier 1 -> tick_value 0.01, notional 20 * 100 = 2,000
    assert STOCK.tick_value == pytest.approx(0.01)
    assert STOCK.notional(20.0, 100) == pytest.approx(2_000.0)
    assert FUT_A.is_future and not STOCK.is_future


def test_one_tick_move_changes_pnl_by_contracts_times_tick_value():
    contracts = 3
    # Long 3 @ 4000.00. One tick up (4000.25): +3 * 12.50 = +37.50. One tick down: -37.50.
    long_ledger = make_ledger(100_000.0)
    long_ledger.record_fill(Fill(at(0), "FUT_A", contracts, 4000.00))
    assert long_ledger.cash == pytest.approx(100_000.0)       # no notional leaves cash
    assert long_ledger.mark_to_market({"FUT_A": 4000.25})["total"] == pytest.approx(37.50)
    assert long_ledger.mark_to_market({"FUT_A": 3999.75})["total"] == pytest.approx(-37.50)
    assert long_ledger.mark_to_market({"FUT_A": 4000.25})["total"] == pytest.approx(contracts * FUT_A.tick_value)
    assert long_ledger.equity({"FUT_A": 4000.25}) == pytest.approx(100_037.50)

    # Short 3 @ 4000.00. One tick up: -37.50. One tick down: +37.50.
    short_ledger = make_ledger(100_000.0)
    short_ledger.record_fill(Fill(at(0), "FUT_A", -contracts, 4000.00))
    assert short_ledger.mark_to_market({"FUT_A": 4000.25})["total"] == pytest.approx(-37.50)
    assert short_ledger.mark_to_market({"FUT_A": 3999.75})["total"] == pytest.approx(37.50)

    # Closing realizes exactly that into cash: short 3 covered one tick lower, fee 1.50.
    #   realized = (3999.75 - 4000.00) * 3 * -1 * 50 = +37.50 ; cash = 100,000 + 37.50 - 1.50 = 100,036.00
    short_ledger.record_fill(Fill(at(1), "FUT_A", contracts, 3999.75, fee=1.50))
    assert short_ledger.gross_realized_pnl == pytest.approx(37.50)
    assert short_ledger.cash == pytest.approx(100_036.00)
    assert short_ledger.check_identity({})


def test_futures_buy_is_not_refused_for_cash():
    # Notional 4000 * 1 * 50 = 200,000 is far above the 1,000 cash, but futures do not
    # pay notional (margin is not modelled yet), so the fill is accepted.
    ledger = make_ledger(1_000.0)
    ledger.record_fill(Fill(at(0), "FUT_A", 1, 4000.0, fee=2.0))
    assert ledger.cash == pytest.approx(998.0)


def test_is_on_tick_and_round_to_tick():
    assert FUT_A.is_on_tick(4000.25)
    assert FUT_A.is_on_tick(4000.00)
    assert not FUT_A.is_on_tick(4000.10)
    # 0.1 + 0.2 is 0.30000000000000004 in floats; it must still count as on a 0.01 tick.
    assert STOCK.is_on_tick(0.1 + 0.2)
    assert not STOCK.is_on_tick(0.305)

    # 4000.10 / 0.25 = 16000.4 ticks -> 16000 ticks -> 4000.00
    assert FUT_A.round_to_tick(4000.10) == pytest.approx(4000.00)
    # 4000.13 / 0.25 = 16000.52 ticks -> 16001 ticks -> 4000.25
    assert FUT_A.round_to_tick(4000.13) == pytest.approx(4000.25)
    # Exactly half way (4000.125 = 16000.5 ticks) rounds up -> 4000.25
    assert FUT_A.round_to_tick(4000.125) == pytest.approx(4000.25)
    assert FUT_A.is_on_tick(FUT_A.round_to_tick(4000.13))
    assert STOCK.round_to_tick(10.004) == pytest.approx(10.00)
    assert STOCK.round_to_tick(10.006) == pytest.approx(10.01)


def test_bad_instrument_values_raise():
    with pytest.raises(ValueError):
        Instrument("X", "future", multiplier=0.0, tick_size=0.25)      # multiplier must be > 0
    with pytest.raises(ValueError):
        Instrument("X", "future", multiplier=-50.0, tick_size=0.25)
    with pytest.raises(ValueError):
        Instrument("X", "future", multiplier=50.0, tick_size=0.0)      # tick size must be > 0
    with pytest.raises(ValueError):
        Instrument("X", "future", multiplier=50.0, tick_size=-0.25)
    with pytest.raises(ValueError):
        Instrument("X", "bond", multiplier=1.0, tick_size=0.01)        # unknown asset class
    with pytest.raises(ValueError):
        Instrument("X", "equity", multiplier=float("nan"), tick_size=0.01)


def test_ledger_rejects_bad_instrument_sets():
    with pytest.raises(ValueError):
        Ledger(1_000.0, [STOCK, STOCK])                                # same symbol twice
    with pytest.raises(ValueError):
        Ledger(1_000.0, {"WRONG_KEY": STOCK})                          # key does not match symbol
    with pytest.raises(ValueError):
        Ledger(1_000.0, [STOCK, Instrument("EUROSTOCK", "equity", currency="EUR")])   # two currencies
    with pytest.raises(ValueError):
        Ledger(-1.0, [STOCK])
    assert Ledger(1_000.0, {"STOCK": STOCK}).instrument("STOCK") is STOCK


# =============================================================================
# Futures rolls
# =============================================================================
def test_roll_carries_quantity_charges_fees_and_gap_is_not_pnl():
    ledger = make_ledger(100_000.0)
    ledger.record_fill(Fill(at(0), "FUT_A", 2, 4000.0))          # long 2 @ 4000, no fee

    # Roll: close FUT_A at 4010, open FUT_B at 4030 (the next contract trades 20 points higher).
    #   realized on FUT_A = (4010 - 4000) * 2 * 50 = 1,000   <- only the old contract's own move
    #   fees = 2.50 + 2.50 = 5.00
    #   cash = 100,000 + 1,000 - 5 = 100,995
    #   The 20-point gap would be 20 * 2 * 50 = 2,000 if it were (wrongly) booked. It must not be.
    closing, opening = ledger.roll(at(1), "FUT_A", "FUT_B", close_price=4010.0, open_price=4030.0,
                                   fee_close=2.50, fee_open=2.50)

    # Two ordinary fills, both tagged "roll", in order.
    assert closing.note == "roll" and opening.note == "roll"
    assert ledger.fills[-2:] == [closing, opening]
    assert len(ledger.fills) == 3
    assert (closing.symbol, closing.quantity, closing.price) == ("FUT_A", -2, 4010.0)
    assert (opening.symbol, opening.quantity, opening.price) == ("FUT_B", 2, 4030.0)

    # Quantity carried over; the new contract starts at its own open price.
    assert ledger.position("FUT_A").is_flat
    assert ledger.position("FUT_B").quantity == pytest.approx(2)
    assert ledger.position("FUT_B").avg_price == pytest.approx(4030.0)

    assert ledger.gross_realized_pnl == pytest.approx(1_000.0)
    assert ledger.realized_pnl_by_symbol["FUT_A"] == pytest.approx(1_000.0)
    assert ledger.realized_pnl_by_symbol["FUT_B"] == pytest.approx(0.0)
    assert ledger.total_fees == pytest.approx(5.00)
    assert ledger.cash == pytest.approx(100_995.0)

    # With FUT_B still at 4030 nothing has been made on it: equity = cash.
    assert ledger.mark_to_market({"FUT_B": 4030.0})["total"] == pytest.approx(0.0)
    assert ledger.equity({"FUT_B": 4030.0}) == pytest.approx(100_995.0)
    assert ledger.check_identity({"FUT_B": 4030.0})


def test_roll_of_a_short_position_and_slippage_references():
    ledger = make_ledger(100_000.0)
    ledger.record_fill(Fill(at(0), "FUT_A", -3, 4000.0))         # short 3 @ 4000

    # Close (buy 3) at 3990 against a reference of 3989.75; open (sell 3) at 4005 against 4005.25.
    #   realized = (3990 - 4000) * 3 * -1 * 50 = +1,500
    #   slippage close = (3990.00 - 3989.75) * 3 * 50 = 37.50   (bought a tick too high)
    #   slippage open  = (4005.00 - 4005.25) * -3 * 50 = 37.50  (sold a tick too low)
    closing, opening = ledger.roll(at(1), "FUT_A", "FUT_B", 3990.0, 4005.0,
                                   reference_close=3989.75, reference_open=4005.25)
    assert closing.quantity == pytest.approx(3)
    assert opening.quantity == pytest.approx(-3)
    assert ledger.position("FUT_B").quantity == pytest.approx(-3)
    assert ledger.position("FUT_B").avg_price == pytest.approx(4005.0)
    assert ledger.gross_realized_pnl == pytest.approx(1_500.0)
    assert ledger.total_slippage_cost == pytest.approx(75.0)
    assert ledger.cash == pytest.approx(101_500.0)


def test_roll_refusals_leave_ledger_unchanged():
    ledger = make_ledger(100_000.0)

    # No position in the expiring contract.
    with pytest.raises(LedgerError):
        ledger.roll(at(0), "FUT_A", "FUT_B", 4010.0, 4030.0)

    ledger.record_fill(Fill(at(0), "FUT_A", 2, 4000.0))
    ledger.record_fill(Fill(at(0), "STOCK", 10, 100.0))
    cash_before = ledger.cash

    # Different multipliers (50 vs 5): the same quantity would be a different exposure.
    with pytest.raises(LedgerError):
        ledger.roll(at(1), "FUT_A", "FUT_SMALL", 4010.0, 4030.0)
    # Rolling out of, or into, something that is not a future.
    with pytest.raises(LedgerError):
        ledger.roll(at(1), "STOCK", "FUT_B", 100.0, 4030.0)
    with pytest.raises(LedgerError):
        ledger.roll(at(1), "FUT_A", "STOCK", 4010.0, 100.0)
    # Unknown symbol, same symbol, a time before the last fill.
    with pytest.raises(LedgerError):
        ledger.roll(at(1), "FUT_A", "NOT_REGISTERED", 4010.0, 4030.0)
    with pytest.raises(LedgerError):
        ledger.roll(at(1), "FUT_A", "FUT_A", 4010.0, 4010.0)
    with pytest.raises(LedgerError):
        ledger.roll(at(0) - timedelta(hours=1), "FUT_A", "FUT_B", 4010.0, 4030.0)
    # A bad opening price must not leave the position half rolled (closed but not reopened).
    with pytest.raises(ValueError):
        ledger.roll(at(1), "FUT_A", "FUT_B", 4010.0, -1.0)

    assert len(ledger.fills) == 2
    assert ledger.cash == pytest.approx(cash_before)
    assert ledger.position("FUT_A").quantity == pytest.approx(2)
    assert ledger.position("FUT_A").avg_price == pytest.approx(4000.0)
    assert ledger.position("FUT_B").is_flat
    assert ledger.gross_realized_pnl == 0.0


def test_only_roll_changes_which_symbol_a_position_is_in():
    ledger = make_ledger(100_000.0)
    ledger.record_fill(Fill(at(0), "FUT_A", 2, 4000.0))
    prices = {"FUT_A": 4020.0, "FUT_B": 4040.0, "STOCK": 101.0}

    def futures_symbols_held():
        """Which futures symbols the ledger holds, and how many of each."""
        return {symbol: position.quantity for symbol, position in ledger.positions().items()
                if ledger.instrument(symbol).is_future}

    # Call every public method except roll(), including fills in a different symbol,
    # and check after each one that the position is still 2 contracts of FUT_A.
    other_actions = [
        lambda: ledger.record_fill(Fill(at(1), "STOCK", 10, 100.0, fee=1.0)),
        lambda: ledger.mark_to_market(prices),
        lambda: ledger.equity(prices),
        lambda: ledger.deposit(at(2), 1_000.0, prices),
        lambda: ledger.withdraw(at(3), 500.0, prices),
        lambda: ledger.record_equity(at(4), prices),
        lambda: ledger.equity_curve(),
        lambda: ledger.summary(prices),
        lambda: ledger.check_identity(prices),
        lambda: ledger.position("FUT_B"),
        lambda: ledger.positions(),
        lambda: ledger.instrument("FUT_B"),
    ]
    for action in other_actions:
        action()
        assert futures_symbols_held() == {"FUT_A": 2}

    # Time passing and FUT_B having a price does not move the position either.
    ledger.record_equity(at(500), prices)
    assert futures_symbols_held() == {"FUT_A": 2}
    assert [fill.symbol for fill in ledger.fills] == ["FUT_A", "STOCK"]
    assert not any(fill.note == "roll" for fill in ledger.fills)

    # Only the explicit roll moves it.
    ledger.roll(at(501), "FUT_A", "FUT_B", 4020.0, 4040.0)
    assert futures_symbols_held() == {"FUT_B": 2}
    assert [fill.note for fill in ledger.fills[-2:]] == ["roll", "roll"]
