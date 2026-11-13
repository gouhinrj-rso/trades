"""Tests for paper.py. Made-up bars; throwaway state folder; no network."""

import ast
from pathlib import Path

import pytest

import kill_switch
import paper

REPO = Path(paper.__file__).resolve().parent


def bar(day_time, open_price, high, low, close):
    return {"timestamp": f"2026-01-{day_time}:00Z", "open": open_price, "high": high,
            "low": low, "close": close, "volume": 1.0}


def test_settings_are_the_owners_numbers():
    assert paper.STARTING_CASH == 10_000.0
    assert paper.STOP_PCT == 1.0
    assert paper.DAILY_LOSS_CAP_PCT == 2.0
    assert paper.ALLOWED_ORDER_TYPES == ("limit",)


# --- limit orders only -------------------------------------------------------
@pytest.mark.parametrize("order_type", ["market", "stop", "stop_limit", "MARKET", ""])
def test_only_limit_orders_are_accepted(owner_released, order_type):
    broker = paper.PaperBroker()
    order = broker.submit_order("buy", order_type, 100.0, "t0", stop_price=99.0)
    assert order["status"] == "refused" and "limit orders only" in order["reason"]
    assert broker.open_order is None


# --- a stop on every trade ---------------------------------------------------
@pytest.mark.parametrize("stop", [None, 0, -5, 100.0, 101.0])
def test_entry_without_a_valid_stop_is_refused(owner_released, stop):
    broker = paper.PaperBroker()
    order = broker.submit_order("buy", "limit", 100.0, "t0", stop_price=stop)
    assert order["status"] == "refused" and "stop price" in order["reason"]


def test_entry_with_a_stop_is_accepted_and_the_stop_protects_the_position(owner_released):
    broker = paper.PaperBroker()
    broker.on_bar(bar("01T00:00", 100, 100, 100, 100))
    order = broker.submit_order("buy", "limit", 100.0, "t0", stop_price=99.0)
    assert order["status"] == "open"

    broker.on_bar(bar("01T00:05", 100, 100.5, 99.8, 100.2))     # trades at 100 -> fills
    assert order["status"] == "filled" and order["fill_price"] == 100.0
    assert broker.units > 0 and broker.stop_price == 99.0

    broker.on_bar(bar("01T00:10", 100, 100, 98.5, 98.8))        # trades down through 99
    assert broker.units == 0
    assert broker.trades[-1]["exit_reason"] == "stop" and broker.trades[-1]["exit_price"] == 99.0


def test_stop_fills_at_the_open_when_the_bar_gaps_below_it(owner_released):
    broker = paper.PaperBroker()
    broker.on_bar(bar("01T00:00", 100, 100, 100, 100))
    broker.submit_order("buy", "limit", 100.0, "t0", stop_price=99.0)
    broker.on_bar(bar("01T00:05", 100, 100, 99.9, 100))
    broker.on_bar(bar("01T00:10", 97.0, 97.5, 96.5, 97.0))      # opens below the stop
    assert broker.trades[-1]["exit_price"] == 97.0             # the worse price, not 99


def test_an_order_cannot_fill_on_the_bar_it_was_placed_after(owner_released):
    broker = paper.PaperBroker()
    broker.on_bar(bar("01T00:00", 100, 101, 99, 100))           # this bar traded at 99.5...
    order = broker.submit_order("buy", "limit", 99.5, "t0", stop_price=98.5)
    assert order["status"] == "open"                            # ...but the order came after it
    assert broker.units == 0


def test_unfilled_limit_order_expires(owner_released):
    broker = paper.PaperBroker()
    broker.on_bar(bar("01T00:00", 100, 100, 100, 100))
    order = broker.submit_order("buy", "limit", 95.0, "t0", stop_price=94.0)
    broker.on_bar(bar("01T00:05", 100, 101, 99, 100))           # never trades at 95
    assert order["status"] == "cancelled" and broker.open_order is None


# --- kill switch -------------------------------------------------------------
def test_engaged_kill_switch_refuses_every_entry():
    assert kill_switch.is_engaged()                             # default: engaged
    broker = paper.PaperBroker()
    order = broker.submit_order("buy", "limit", 100.0, "t0", stop_price=99.0)
    assert order["status"] == "refused" and order["reason"] == "kill switch is engaged"


def test_engaging_the_switch_cancels_a_resting_entry_but_exits_still_work(owner_released):
    broker = paper.PaperBroker()
    broker.on_bar(bar("01T00:00", 100, 100, 100, 100))
    resting = broker.submit_order("buy", "limit", 99.0, "t0", stop_price=98.0)
    kill_switch.engage("owner pressed stop", by="test")
    broker.on_bar(bar("01T00:05", 100, 100, 98.9, 99.5))        # would have filled at 99
    assert resting["status"] == "cancelled" and resting["reason"] == "kill switch is engaged"
    assert broker.units == 0

    # A position opened earlier can still be closed while the switch is engaged.
    broker.units, broker.entry_price, broker.entry_time = 10.0, 99.0, "t"
    broker.entry_cost, broker.stop_price, broker.cash = 990.0, 98.0, 0.0
    exit_order = broker.submit_order("sell", "limit", 99.5, "t1")
    assert exit_order["status"] == "open"


def test_broker_has_no_way_to_be_told_the_switch_is_off():
    # The broker asks kill_switch itself; no argument or attribute can override that.
    import inspect
    for function in (paper.PaperBroker.__init__, paper.PaperBroker.submit_order, paper.run_paper):
        parameters = inspect.signature(function).parameters
        assert not [name for name in parameters if "kill" in name or "switch" in name or "force" in name]


# --- daily loss cap ----------------------------------------------------------
def test_daily_loss_cap_blocks_entries_until_the_next_day(owner_released):
    broker = paper.PaperBroker()
    broker.on_bar(bar("01T00:00", 100, 100, 100, 100))
    broker.submit_order("buy", "limit", 100.0, "t0", stop_price=99.0)
    broker.on_bar(bar("01T00:05", 100, 100, 99.9, 100))         # filled at 100
    broker.on_bar(bar("01T00:10", 96.0, 96.5, 95.5, 96.0))      # gaps down: stopped at 96, about -4%
    assert broker.day_halted is True
    assert any(event["kind"] == "daily loss cap hit" for event in broker.events)

    refused = broker.submit_order("buy", "limit", 96.0, "t2", stop_price=95.0)
    assert refused["status"] == "refused" and "daily loss cap" in refused["reason"]

    broker.on_bar(bar("02T00:00", 96, 96, 96, 96))              # a new UTC day
    assert broker.day_halted is False
    allowed = broker.submit_order("buy", "limit", 96.0, "t3", stop_price=95.0)
    assert allowed["status"] == "open"


# --- end to end --------------------------------------------------------------
CANDIDATE = {"name": "zscore(window=5, entry=1.0)", "family": "zscore_reversion",
             "params": {"window": 5, "entry": 1.0}}


def wavy_bars(count=120):
    bars = []
    for i in range(count):
        close = 100 + 3 * ((i % 12) - 6) / 6 + (i % 5) * 0.2
        day = 1 + i // 100
        bars.append({"timestamp": f"2026-01-{day:02d}T{(i % 100) // 12:02d}:{(i % 12) * 5:02d}:00Z",
                     "open": close - 0.1, "high": close + 0.4, "low": close - 0.4, "close": close, "volume": 1.0})
    return bars


def test_run_with_the_switch_engaged_trades_nothing():
    broker = paper.run_paper(CANDIDATE, wavy_bars())
    summary = broker.summary()
    assert summary["orders_filled"] == 0 and summary["round_trips"] == 0
    assert summary["orders_refused_by_kill_switch"] == summary["orders_submitted"] > 0
    assert summary["equity"] == paper.STARTING_CASH


def test_run_with_the_switch_released_uses_limit_orders_with_stops(owner_released):
    broker = paper.run_paper(CANDIDATE, wavy_bars())
    assert broker.summary()["orders_filled"] > 0
    assert all(order["type"] == "limit" for order in broker.orders)
    buys = [order for order in broker.orders if order["side"] == "buy"]
    assert buys and all(order["stop_price"] == pytest.approx(order["limit_price"] * 0.99, abs=0.01) for order in buys)


def test_paper_refuses_a_candidate_the_gate_blocks(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["paper.py", "--candidate", "ema_cross(fast=5, slow=30)"])
    assert paper.main() == 4
    assert "risk gate blocks" in capsys.readouterr().out
    assert not paper.PAPER_STATE_FILE.exists()


def test_paper_refuses_to_run_when_strategy_code_reaches_the_switch(monkeypatch, capsys):
    monkeypatch.setattr(kill_switch, "check_strategy_isolation", lambda files=None: ["engine.py:1 imports kill_switch"])
    monkeypatch.setattr("sys.argv", ["paper.py", "--candidate", "zscore(window=40, entry=1.5)"])
    assert paper.main() == 3
    assert "Refusing to run" in capsys.readouterr().out


# --- no broker, no keys, no network ------------------------------------------
NETWORK_MODULES = {"urllib", "requests", "http", "httpx", "socket", "ssl", "websocket", "websockets",
                   "aiohttp", "ccxt", "krakenex", "alpaca", "ib_insync", "smtplib", "ftplib"}


@pytest.mark.parametrize("file_name", ["paper.py", "kill_switch.py", "build_view.py"])
def test_execution_code_imports_no_network_or_broker_library(file_name):
    tree = ast.parse((REPO / file_name).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & NETWORK_MODULES


@pytest.mark.parametrize("file_name", ["paper.py", "kill_switch.py"])
def test_execution_code_reads_no_keys(file_name):
    source = (REPO / file_name).read_text().lower()
    for word in ("api_key", "apikey", "api_secret", "private_key", "password", "token"):
        assert word not in source
