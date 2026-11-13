"""Paper execution: simulated orders only. No broker, no exchange, no keys, no network.

Run from the repo root:

    python paper.py --candidate "zscore(window=40, entry=1.5)"

That replays one gate-passed candidate over the VALIDATION window through the
paper broker and saves the result to state/paper_state.json. It is a replay on
bars the backtest has already seen, not forward trading on new bars.

Rules the paper broker enforces (none can be switched off by a caller):

1. Limit orders only. Any other order type is refused.
2. A stop on every trade. A buy without a valid stop price is refused.
3. A daily loss cap. Once the account is down DAILY_LOSS_CAP_PCT from the start
   of the UTC day, new entries are refused until the next day.
4. The kill switch. The broker asks kill_switch.py directly on every order and
   every bar. While it is engaged, entries are refused and resting entry orders
   are cancelled. Exits and protective stops still work, because they reduce risk.

The protective stop is the one non-limit fill: when price trades down to the
stop, the position is sold at the stop price (or at the bar's open if the bar
opened below it).

Strategy code never receives the broker. A strategy is a function that turns
closing prices into wanted positions (see engine.py); this file decides what
orders, if any, follow from that.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import engine
import gate
import kill_switch

PROJECT_DIR = Path(__file__).resolve().parent

# ============================================================================
# PAPER TRADING SETTINGS - set by the owner on 2026-11-13.
# ============================================================================
STARTING_CASH = 10_000.0         # paper dollars
STOP_PCT = 1.0                   # every entry carries a stop this far below its limit price
DAILY_LOSS_CAP_PCT = 2.0         # no new entries once down this much on the UTC day
FEE_BPS_PER_SIDE = engine.FEE_BPS_PER_SIDE   # same 2 bps per side as the backtest
ORDER_LIFETIME_BARS = 1          # an unfilled limit order is cancelled after this many bars
ALLOWED_ORDER_TYPES = ("limit",) # the only order type that exists here
# ============================================================================

STATE_DIR = Path(os.environ.get("TT_STATE_DIR", PROJECT_DIR / "state"))
PAPER_STATE_FILE = STATE_DIR / "paper_state.json"


class PaperBroker:
    """A simulated account: cash, at most one long position, at most one resting order."""

    def __init__(self, starting_cash: float = STARTING_CASH):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.units = 0.0              # how much of the asset is held
        self.entry_price = None
        self.entry_time = None
        self.entry_cost = None        # cash spent on the open position, fees included
        self.stop_price = None        # protective stop for the open position
        self.open_order = None        # the single resting limit order, if any
        self.orders = []              # every order ever submitted, with its outcome
        self.trades = []              # completed round trips
        self.events = []              # refusals, cancellations, halts - with reasons
        self.equity_curve = []        # [timestamp, account value] at each bar close
        self.current_day = None       # UTC date of the bar being processed
        self.day_start_equity = starting_cash
        self.day_halted = False       # True once the daily loss cap is hit
        self.last_close = None
        self.fee_rate = FEE_BPS_PER_SIDE / 10_000.0

    # ------------------------------------------------------------------ helpers
    def equity(self) -> float:
        """Account value: cash plus the position valued at the last close."""
        price = self.last_close if self.last_close is not None else 0.0
        return self.cash + self.units * price

    def _note(self, timestamp: str, kind: str, detail: str) -> None:
        self.events.append({"time": timestamp, "kind": kind, "detail": detail})

    def _refuse(self, order: dict, reason: str) -> dict:
        order["status"] = "refused"
        order["reason"] = reason
        self.orders.append(order)
        self._note(order["submitted_at"], "order refused", f"{order['side']} {order['type']}: {reason}")
        return order

    # ------------------------------------------------------------------ orders
    def submit_order(self, side: str, order_type: str, limit_price: float,
                     timestamp: str, stop_price: float = None) -> dict:
        """Submit one order. Returns the order record, whose status says what happened."""
        order = {"id": len(self.orders) + 1, "side": side, "type": order_type,
                 "limit_price": limit_price, "stop_price": stop_price,
                 "submitted_at": timestamp, "status": "open", "age_bars": 0}

        # Rule 1: limit orders only.
        if order_type not in ALLOWED_ORDER_TYPES:
            return self._refuse(order, f"order type '{order_type}' is not allowed; limit orders only")
        if side not in ("buy", "sell"):
            return self._refuse(order, f"unknown side '{side}'")
        if not isinstance(limit_price, (int, float)) or limit_price <= 0:
            return self._refuse(order, "a limit price above zero is required")
        if self.open_order is not None:
            return self._refuse(order, "another order is already resting")

        if side == "buy":
            # Rule 4: the kill switch. Asked directly, every time.
            if kill_switch.is_engaged():
                return self._refuse(order, "kill switch is engaged")
            # Rule 3: the daily loss cap.
            if self.day_halted:
                return self._refuse(order, f"daily loss cap of {DAILY_LOSS_CAP_PCT:g}% reached")
            # Rule 2: a stop on every trade.
            if not isinstance(stop_price, (int, float)) or not 0 < stop_price < limit_price:
                return self._refuse(order, "a stop price below the limit price is required on every entry")
            if self.units > 0:
                return self._refuse(order, "already holding a position")
            if self.cash <= 0:
                return self._refuse(order, "no cash available")
        else:
            if self.units <= 0:
                return self._refuse(order, "nothing to sell")

        self.orders.append(order)
        self.open_order = order
        return order

    def _cancel_open_order(self, timestamp: str, reason: str) -> None:
        order = self.open_order
        order["status"] = "cancelled"
        order["reason"] = reason
        self.open_order = None
        self._note(timestamp, "order cancelled", f"{order['side']} limit {order['limit_price']:.2f}: {reason}")

    # ------------------------------------------------------------------ fills
    def _sell_all(self, price: float, timestamp: str, why: str) -> None:
        gross = self.units * price
        fee = gross * self.fee_rate
        self.cash = gross - fee
        self.trades.append({
            "entry_time": self.entry_time, "entry_price": self.entry_price,
            "exit_time": timestamp, "exit_price": price, "exit_reason": why,
            "stop_price": self.stop_price,
            "return_pct": (self.cash / self.entry_cost - 1.0) * 100,
        })
        self.units, self.entry_price, self.stop_price = 0.0, None, None

    def on_bar(self, bar: dict) -> None:
        """Advance the simulation by one finished bar.

        Orders submitted before this bar may fill during it. Orders submitted
        after this call can only fill on later bars, so there is no look-ahead.
        """
        timestamp = bar["timestamp"]

        # A new UTC day resets the daily loss cap.
        day = timestamp[:10]
        if day != self.current_day:
            self.current_day = day
            self.day_start_equity = self.equity() if self.last_close is not None else self.cash
            if self.day_halted:
                self._note(timestamp, "daily cap reset", "new UTC day: entries allowed again")
            self.day_halted = False

        # Kill switch: cancel any resting entry order.
        if self.open_order and self.open_order["side"] == "buy" and kill_switch.is_engaged():
            self._cancel_open_order(timestamp, "kill switch is engaged")

        # Protective stop on an existing position (checked before anything else).
        if self.units > 0 and bar["low"] <= self.stop_price:
            self._sell_all(min(self.stop_price, bar["open"]), timestamp, "stop")
            if self.open_order:   # a resting sell is now pointless
                self._cancel_open_order(timestamp, "position closed by its stop")

        # The resting limit order.
        order = self.open_order
        if order and order["side"] == "buy" and bar["low"] <= order["limit_price"]:
            price = min(order["limit_price"], bar["open"])     # never worse than the limit
            self.entry_cost = self.cash
            fee = self.cash * self.fee_rate
            self.units = (self.cash - fee) / price
            self.cash = 0.0
            self.entry_price, self.entry_time = price, timestamp
            self.stop_price = order["stop_price"]
            order.update({"status": "filled", "fill_price": price, "filled_at": timestamp})
            self.open_order = None
            # If the same bar also traded down to the stop, assume the worst: stopped out.
            if bar["low"] <= self.stop_price:
                self._sell_all(self.stop_price, timestamp, "stop (same bar as entry)")
        elif order and order["side"] == "sell" and bar["high"] >= order["limit_price"]:
            price = max(order["limit_price"], bar["open"])     # never worse than the limit
            order.update({"status": "filled", "fill_price": price, "filled_at": timestamp})
            self.open_order = None
            self._sell_all(price, timestamp, "limit exit")
        elif order:
            order["age_bars"] += 1
            if order["age_bars"] >= ORDER_LIFETIME_BARS:
                self._cancel_open_order(timestamp, "not filled in time")

        # Mark the account at this bar's close and check the daily loss cap.
        self.last_close = bar["close"]
        value = self.equity()
        self.equity_curve.append([timestamp, round(value, 2)])
        floor = self.day_start_equity * (1 - DAILY_LOSS_CAP_PCT / 100.0)
        if not self.day_halted and value <= floor:
            self.day_halted = True
            self._note(timestamp, "daily loss cap hit",
                       f"equity {value:.2f} is at or below {floor:.2f} "
                       f"({DAILY_LOSS_CAP_PCT:g}% under the day's start {self.day_start_equity:.2f}); "
                       f"no new entries today")
            if self.open_order and self.open_order["side"] == "buy":
                self._cancel_open_order(timestamp, "daily loss cap reached")

    # ------------------------------------------------------------------ report
    def summary(self) -> dict:
        count = lambda status, side=None: sum(
            1 for o in self.orders if o["status"] == status and (side is None or o["side"] == side))
        refused_by_switch = sum(1 for o in self.orders
                                if o["status"] == "refused" and "kill switch" in o.get("reason", ""))
        value = self.equity() if self.last_close is not None else self.cash
        return {
            "starting_cash": self.starting_cash,
            "equity": round(value, 2),
            "return_pct": round((value / self.starting_cash - 1.0) * 100, 4),
            "orders_submitted": len(self.orders),
            "orders_filled": count("filled"),
            "orders_refused": count("refused"),
            "orders_refused_by_kill_switch": refused_by_switch,
            "orders_cancelled": count("cancelled"),
            "round_trips": len(self.trades),
            "stopped_out": sum(1 for t in self.trades if t["exit_reason"].startswith("stop")),
            "holding_position": self.units > 0,
            "day_halted": self.day_halted,
        }


def run_paper(candidate: dict, bars: list, trade_start: int = 0, broker: PaperBroker = None) -> PaperBroker:
    """Feed bars to the paper broker and turn the strategy's wanted position into limit orders.

    The strategy only supplies `signals` (0 = flat, 1 = long). It never sees the broker.
    """
    broker = broker or PaperBroker()
    closes = [bar["close"] for bar in bars]
    signals = engine.make_signals(candidate, closes)   # signals[t] uses closes[0..t] only

    for t in range(trade_start, len(bars)):
        bar = bars[t]
        broker.on_bar(bar)                              # fills orders placed on earlier bars

        wanted = signals[t]                             # decided at this bar's close
        if broker.open_order is not None:
            continue
        if wanted == 1 and broker.units == 0:
            limit = bar["close"]
            broker.submit_order("buy", "limit", limit, bar["timestamp"],
                                stop_price=round(limit * (1 - STOP_PCT / 100.0), 2))
        elif wanted == 0 and broker.units > 0:
            broker.submit_order("sell", "limit", bar["close"], bar["timestamp"])
    return broker


def settings() -> dict:
    """The rules in force, for the saved state and the page."""
    return {"starting_cash": STARTING_CASH, "stop_pct": STOP_PCT,
            "daily_loss_cap_pct": DAILY_LOSS_CAP_PCT, "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "order_types": list(ALLOWED_ORDER_TYPES), "order_lifetime_bars": ORDER_LIFETIME_BARS}


def save_state(broker: PaperBroker, candidate: dict, mode: str, window: dict) -> dict:
    """Write the paper account to state/paper_state.json."""
    state = {
        "paper_only": True,
        "mode": mode,
        "candidate": candidate["name"],
        "window": window,
        "saved_at": kill_switch._now(),
        "kill_switch_engaged_at_end": kill_switch.is_engaged(),
        "settings": settings(),
        "summary": broker.summary(),
        "trades": broker.trades,
        "orders": broker.orders[-200:],
        "events": broker.events[-200:],
        "equity_curve": broker.equity_curve,
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAPER_STATE_FILE, "w") as handle:
        json.dump(state, handle, indent=1)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one gate-passed candidate through the paper broker.")
    parser.add_argument("--candidate", required=True, help='name, e.g. "zscore(window=40, entry=1.5)"')
    args = parser.parse_args()

    print("PAPER TRADING ONLY - simulated orders, no broker, no exchange, no keys.")
    print()

    # The kill switch module checks that strategy code keeps away from it.
    problems = kill_switch.check_strategy_isolation()
    if problems:
        print("Refusing to run: strategy code must not touch the kill switch or the broker.")
        for problem in problems:
            print(f"  {problem}")
        return 3

    matches = [c for c in engine.build_candidates() if c["name"] == args.candidate]
    if not matches:
        print(f"Unknown candidate: {args.candidate}")
        return 2
    candidate = matches[0]

    # Only a candidate the risk gate passes may be paper traded.
    train = engine.load_bars(engine.TRAIN_FILE)
    validation = engine.load_bars(engine.VALIDATION_FILE)
    verdict = gate.check_backtest_result(engine.evaluate_candidate(candidate, train, validation))
    if verdict["verdict"] != "PASS":
        print(f"Refusing to run: the risk gate blocks {candidate['name']}:")
        for failure in verdict["failures"]:
            print(f"  {failure}")
        return 4

    kill_switch.print_status()
    print()

    broker = run_paper(candidate, train + validation, trade_start=len(train))
    window = {"first_timestamp": validation[0]["timestamp"], "last_timestamp": validation[-1]["timestamp"],
              "bars": len(validation)}
    state = save_state(broker, candidate, "replay on the validation window (not forward trading)", window)

    summary = state["summary"]
    print(f"Candidate          : {candidate['name']}")
    print(f"Mode               : {state['mode']}")
    print(f"Bars replayed      : {window['bars']}  ({window['first_timestamp']} -> {window['last_timestamp']})")
    print(f"Rules              : limit orders only | stop {STOP_PCT:g}% below entry | "
          f"daily loss cap {DAILY_LOSS_CAP_PCT:g}% | fee {FEE_BPS_PER_SIDE:g} bps per side")
    print(f"Orders submitted   : {summary['orders_submitted']}")
    print(f"  filled           : {summary['orders_filled']}")
    print(f"  refused          : {summary['orders_refused']}  "
          f"(by kill switch: {summary['orders_refused_by_kill_switch']})")
    print(f"  cancelled        : {summary['orders_cancelled']}")
    print(f"Round trips        : {summary['round_trips']}  (stopped out: {summary['stopped_out']})")
    print(f"Paper equity       : {summary['equity']:.2f}  ({summary['return_pct']:+.2f}%)")
    print(f"Saved              : {PAPER_STATE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
