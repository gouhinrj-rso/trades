"""The backtest engine: monthly target weights in, a ledger of fills and a daily equity curve out.

Timing, which is where backtests usually go wrong:

    close of day t (last trading day of a month)
        The weight function is handed every asset's bars UP TO AND INCLUDING t,
        nothing later. It returns target weights. Order sizes are fixed here,
        from the account's value and prices at this close.
    open of day t+1 (first trading day of the next month)
        The orders fill at this open, made worse by slippage, plus commission.
        A fill is never on the day that produced the signal.

Prices traded are total-return adjusted (dividends are inside the price), so a
position's value grows with dividends without separate cash payments.

The engine knows nothing about any particular strategy. The baseline and the
benchmarks are all just weight functions.
"""

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from src.accounting import Fill, Instrument, Ledger

# A position smaller than this many shares is treated as nothing.
DUST_SHARES = 1e-9


@dataclass(frozen=True)
class Costs:
    """Trading costs, in basis points (1 bp = 0.01%) of the traded amount, per side."""
    commission_bps: float
    slippage_bps: float

    def scaled(self, multiple: float) -> "Costs":
        """The same costs times `multiple` (used for the 0x, 2x, 3x stress runs)."""
        return Costs(self.commission_bps * multiple, self.slippage_bps * multiple)


@dataclass
class BacktestResult:
    """Everything one run produced."""
    daily: pd.DataFrame          # one row per trading day: equity, cash, risk_weight, traded_notional, fees
    fills: pd.DataFrame          # every fill
    round_trips: pd.DataFrame    # one row per holding episode (a "trade")
    rebalances: pd.DataFrame     # one row per asset per signal day: the numbers behind the weights
    ledger: Ledger = field(repr=False, default=None)

    @property
    def returns(self) -> pd.Series:
        """Daily close-to-close returns of the account, net of costs.

        The first day's return is measured from the starting cash, so the cost of
        the very first purchases is included.
        """
        equity = self.daily["equity"]
        previous = equity.shift(1)
        previous.iloc[0] = self.daily.attrs["starting_cash"]
        return (equity / previous - 1.0).rename("return")


def prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Add total-return open and close to one symbol's cached bars.

    The source gives split-adjusted prices and an `adj_close` that also removes
    dividends. Scaling by adj_close / close turns the open into a total-return open.
    """
    out = bars.sort_values("session").reset_index(drop=True)
    factor = out["adj_close"] / out["close"]
    out["tr_open"] = out["open"] * factor
    out["tr_close"] = out["adj_close"]
    return out[["session", "open", "close", "tr_open", "tr_close", "available_at"]]


def _open_time(session: pd.Timestamp):
    """A timestamp for the opening fill: 14:30 UTC on the session day (before any close)."""
    return (pd.Timestamp(session).tz_localize("UTC") + timedelta(hours=14, minutes=30)).to_pydatetime()


def run_backtest(bars: dict, sessions: pd.DatetimeIndex, weights_fn, costs: Costs, starting_cash: float,
                 trade_start, trade_end, cash_vehicle: str | None = None, rebalance: str = "monthly") -> BacktestResult:
    """Run one backtest.

    bars          symbol -> prepared bars (see prepare_bars), cash vehicle included
    sessions      every trading day of the exchange calendar covering the data
    weights_fn    function(histories) -> (weights, details); histories maps symbol ->
                  that symbol's bars up to the signal day only
    costs         commission and slippage per side
    trade_start   first day the account may trade; the first signal is taken at the
                  close of the trading day before it
    trade_end     last day of the run
    cash_vehicle  symbol that receives whatever the weights leave over (or None for idle cash)
    rebalance     "monthly" = new targets every month end; "once" = only the first time
    """
    trade_start, trade_end = pd.Timestamp(trade_start), pd.Timestamp(trade_end)
    run_days = sessions[(sessions >= trade_start) & (sessions <= trade_end)]
    if len(run_days) == 0:
        raise ValueError("no trading days between trade_start and trade_end")
    earlier = sessions[sessions < run_days[0]]
    if len(earlier) == 0:
        raise ValueError("need at least one trading day before trade_start for the first signal")
    first_signal_day = earlier[-1]

    # Fast lookups: symbol -> {session: row position}, and plain arrays of prices.
    position_of = {symbol: {day: i for i, day in enumerate(table["session"])} for symbol, table in bars.items()}
    ledger = Ledger(starting_cash, [Instrument(symbol, "equity", 1.0, 0.01, "USD") for symbol in bars])

    daily_rows, fill_rows, rebalance_rows, round_trips = [], [], [], []
    open_trips = {}       # symbol -> the holding episode currently open
    pending = None        # targets decided at the last signal close, waiting for the next open
    rebalanced_once = False

    def price(symbol: str, day, column: str):
        """The symbol's price on `day`, or None if it has no bar that day (not listed yet)."""
        index = position_of[symbol].get(day)
        return None if index is None else float(bars[symbol][column].iat[index])

    def closes_on(day) -> dict:
        """Closing prices for every symbol currently held."""
        return {symbol: price(symbol, day, "tr_close") for symbol in ledger.positions()}

    def decide(day, next_day) -> dict | None:
        """At the close of `day`: ask the weight function and fix the order sizes."""
        # THE LOOKAHEAD LINE: each history ends at `day`. Later rows are not passed on.
        histories = {symbol: table.iloc[: position_of[symbol][day] + 1]
                     for symbol, table in bars.items() if symbol != cash_vehicle and day in position_of[symbol]}
        weights, details = weights_fn(histories)
        if sum(weights.values()) > 1.0 + 1e-9 or any(weight < 0 for weight in weights.values()):
            raise ValueError(f"weights on {day:%Y-%m-%d} are negative or add up to more than 1: {weights}")
        weights = dict(weights)
        # Whatever is left over goes to the cash vehicle, if it trades today and tomorrow.
        leftover = 1.0 - sum(weights.values())
        if cash_vehicle and leftover > 1e-12 and price(cash_vehicle, day, "tr_close") and next_day is not None \
                and price(cash_vehicle, next_day, "tr_open"):
            weights[cash_vehicle] = leftover

        equity = ledger.equity(closes_on(day))
        target_shares = {symbol: weight * equity / price(symbol, day, "tr_close") for symbol, weight in weights.items()}
        for symbol, state in details.items():
            rebalance_rows.append({"signal_session": day, "symbol": symbol, "weight": weights.get(symbol, 0.0), **state})
        if cash_vehicle:
            rebalance_rows.append({"signal_session": day, "symbol": cash_vehicle, "weight": weights.get(cash_vehicle, 0.0),
                                   "eligible": cash_vehicle in weights, "is_cash_vehicle": True})
        return {"signal_session": day, "weights": weights, "target_shares": target_shares, "details": details}

    def record(day, symbol, quantity, open_price, order):
        """Send one fill to the ledger and remember it for the trade list."""
        slip = costs.slippage_bps / 10_000.0
        fill_price = open_price * (1 + slip) if quantity > 0 else open_price * (1 - slip)
        notional = abs(quantity) * fill_price
        fee = notional * costs.commission_bps / 10_000.0
        fill = Fill(timestamp=_open_time(day), symbol=symbol, quantity=quantity, price=fill_price, fee=fee,
                    reference_price=open_price, note="rebalance")
        ledger.record_fill(fill)
        fill_rows.append({
            "fill_session": day, "signal_session": order["signal_session"], "symbol": symbol,
            "side": "buy" if quantity > 0 else "sell", "quantity": quantity, "reference_open": open_price,
            "fill_price": fill_price, "notional": notional, "commission": fee,
            "slippage_cost": fill.slippage_cost(1.0), "target_weight": order["weights"].get(symbol, 0.0),
            "is_cash_vehicle": symbol == cash_vehicle})
        if symbol == cash_vehicle:
            return
        # --- holding episodes ("trades") -------------------------------------------------
        cash_flow = -(notional + fee) if quantity > 0 else (notional - fee)
        trip = open_trips.get(symbol)
        if trip is None:
            volatility = order["details"].get(symbol, {}).get("volatility")
            trip = open_trips[symbol] = {
                "symbol": symbol, "entry_session": day, "entry_notional": notional, "entry_volatility": volatility,
                # 1R = a one-month, one-standard-deviation loss on the first position (protocol.md section 4).
                "risk": notional * volatility * 21 ** 0.5 if volatility else float("nan"),
                "cash_flow": 0.0, "costs": 0.0, "fills": 0}
        trip["cash_flow"] += cash_flow
        trip["costs"] += fee + fill.slippage_cost(1.0)
        trip["fills"] += 1
        if ledger.position(symbol).is_flat:
            trip.update(exit_session=day, pnl=trip["cash_flow"], open=False)
            round_trips.append(open_trips.pop(symbol))

    def execute(day, order) -> float:
        """At the open of `day`: move every position to its target. Returns the amount traded."""
        held = {symbol: position.quantity for symbol, position in ledger.positions().items()}
        deltas = {}
        for symbol in set(held) | set(order["target_shares"]):
            open_price = price(symbol, day, "tr_open")
            if open_price is None:
                continue                                     # no bar today: cannot trade it
            delta = order["target_shares"].get(symbol, 0.0) - held.get(symbol, 0.0)
            if abs(delta) > DUST_SHARES:
                deltas[symbol] = (delta, open_price)
        traded = 0.0
        # Sells first, so their cash is there for the buys.
        for symbol, (delta, open_price) in sorted(deltas.items()):
            if delta < 0:
                record(day, symbol, delta, open_price, order)
                traded += abs(delta) * open_price
        buys = {symbol: pair for symbol, pair in deltas.items() if pair[0] > 0}
        slip, fee_rate = costs.slippage_bps / 10_000.0, costs.commission_bps / 10_000.0
        needed = sum(delta * open_price * (1 + slip) * (1 + fee_rate) for delta, open_price in buys.values())
        # If the opening gap left us short of cash, shrink every buy by the same factor.
        scale = 1.0 if needed <= ledger.cash else (ledger.cash / needed) * (1 - 1e-12)
        for symbol, (delta, open_price) in sorted(buys.items()):
            if delta * scale > DUST_SHARES:
                record(day, symbol, delta * scale, open_price, order)
                traded += delta * scale * open_price
        return traded

    all_days = sessions[(sessions >= first_signal_day) & (sessions <= trade_end)]
    for day in all_days:
        in_run = day >= run_days[0]
        # After the run's last day comes whatever the calendar says, even if we hold no bars for it.
        later = sessions[sessions > day]
        next_day = later[0] if len(later) else None

        traded = 0.0
        if in_run and pending is not None:                   # ---- the open ----
            traded = execute(day, pending)
            pending = None

        if in_run:                                           # ---- the close ----
            closes = closes_on(day)
            # 21:00 UTC is at or after every NYSE close and after the 14:30 opening fills.
            stamp = (pd.Timestamp(day).tz_localize("UTC") + timedelta(hours=21)).to_pydatetime()
            snapshot = ledger.record_equity(stamp, closes)
            cash_value = 0.0
            if cash_vehicle and cash_vehicle in closes:
                cash_value = ledger.position(cash_vehicle).quantity * closes[cash_vehicle]
            equity = snapshot["equity"]
            daily_rows.append({"session": day, "equity": equity, "cash": ledger.cash,
                               "cash_vehicle_value": cash_value,
                               "risk_weight": (equity - ledger.cash - cash_value) / equity if equity else 0.0,
                               "traded_notional": traded})

        # A signal is taken at the close of the last trading day of a month, if the run continues after it.
        month_end = next_day is not None and (next_day.month != day.month or next_day.year != day.year)
        if month_end and next_day <= trade_end and not (rebalance == "once" and rebalanced_once):
            pending = decide(day, next_day)
            rebalanced_once = True

    # Episodes still open at the end are valued at the last close and marked open.
    last_day = run_days[-1]
    for symbol, trip in open_trips.items():
        value = ledger.position(symbol).quantity * price(symbol, last_day, "tr_close")
        trip.update(exit_session=last_day, pnl=trip["cash_flow"] + value, open=True)
        round_trips.append(trip)

    daily = pd.DataFrame(daily_rows).set_index("session")
    daily.attrs["starting_cash"] = starting_cash
    trips = pd.DataFrame(round_trips)
    if not trips.empty:
        trips["r_multiple"] = trips["pnl"] / trips["risk"]
        trips = trips.sort_values(["exit_session", "symbol"]).reset_index(drop=True)
    ledger.check_identity(closes_on(last_day))               # the books must balance
    return BacktestResult(daily=daily, fills=pd.DataFrame(fill_rows), round_trips=trips,
                          rebalances=pd.DataFrame(rebalance_rows), ledger=ledger)


def fixed_weights(weights: dict):
    """A weight function that always answers with the same weights (for the benchmarks)."""
    def weights_fn(histories: dict) -> tuple:
        available = {symbol: weight for symbol, weight in weights.items() if symbol in histories}
        return available, {symbol: {"eligible": True, "fixed": True} for symbol in available}
    return weights_fn
