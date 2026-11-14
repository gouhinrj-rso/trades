"""The ledger: cash, positions, fills, fees, slippage and equity. Simulated only.

This file is pure bookkeeping. It sends no orders and makes no network calls;
it only records fills that some other simulated component says happened.

CONVENTIONS (read these before trusting a number)
-------------------------------------------------
1. Quantity is signed. Positive = buy, negative = sell. A position is signed
   the same way: positive = long, negative = short.

2. Average-cost method. A position has ONE average entry price.
   * Adding to a position re-averages the entry price.
   * Reducing or closing a position realizes P&L against the average and
     leaves the average unchanged for what remains.
   * Flipping through zero (long 10, sell 25) closes the old position against
     its average and opens the remainder (short 15) at the fill price.
   This is not FIFO: individual lots are not tracked.

3. Realized P&L is GROSS: it is price movement only,
       (exit price - average entry) * quantity closed * multiplier
   (sign-correct for shorts). Fees are NEVER inside it. Fees are kept in their
   own running total. So:
       total P&L = gross realized P&L + unrealized P&L - total fees
   and nothing is counted twice.

4. Slippage is information, not an extra charge. For one fill:
       slippage cost = (price - reference_price) * quantity * multiplier
   Positive means the fill was worse than the price the order was decided
   against (buying higher, or selling lower); negative means it was better.
   The fill price already contains the slippage, so slippage is already inside
   cash and P&L. It is reported separately but must not be subtracted again,
   and it is not part of the accounting identity.

5. Equities and crypto move cash by the full notional. Buying takes
   price * quantity * multiplier + fee out of cash; selling puts
   price * quantity * multiplier - fee in. A buy that would leave cash below
   zero is refused with `InsufficientCash` (no borrowing). Short selling is
   allowed and is not margined: the sale proceeds simply land in cash and the
   short shows up as a negative market value in equity.

6. Futures do NOT move cash by notional. Cash changes only when
   * a fee is paid (at every fill), and
   * a position is reduced or closed: the gross realized P&L goes into cash.
   While a futures position is open its gain or loss is unrealized and is
   added to equity, not to cash.
   NOT MODELLED YET: daily mark-to-market settlement (variation margin moving
   cash every day) and initial/maintenance margin. A futures fill is never
   refused for lack of cash, so nothing here stops an oversized futures
   position. That check belongs to the risk layer until margin is modelled.

7. One currency. Every instrument in a ledger must have the same currency,
   because there is a single cash balance and no FX conversion.

8. One clock. Fills, deposits, withdrawals and equity snapshots are all
   append-only and time-ordered: anything timestamped earlier than the last
   recorded event is refused. Equal timestamps are allowed. Timestamps must be
   timezone-aware.

9. Never assume a price. Valuing a held position needs a price supplied by the
   caller; a missing price raises `MissingPrice`. The ledger never falls back
   to the entry price or the last fill.

10. Rolls are explicit. Only `roll()` moves a position from one futures
    contract to another, and it does so as two ordinary fills. Nothing else in
    this file ever changes which symbol a position is in.

THE ACCOUNTING IDENTITY
-----------------------
At any moment, for any set of prices:

    starting cash + net external cash flow + gross realized P&L
        + unrealized P&L - total fees  ==  equity

`check_identity(prices)` tests this and raises `LedgerError` if it is broken.

CASH-FLOW-ADJUSTED EQUITY (unit value)
--------------------------------------
Raw equity jumps when money is deposited or withdrawn, which would look like a
profit or a loss. To measure performance only, the ledger works like a fund
that issues units:

* At the start the account holds `starting_cash` units worth 1.0 each.
* unit value = equity / units outstanding.
* A deposit BUYS new units at the current unit value; a withdrawal SELLS
  units at the current unit value. So a cash flow changes the number of
  units, never the unit value.
* adjusted return = unit value - 1. This is the time-weighted return.

Tiny worked example:

    Start with 10,000            -> 10,000 units, unit value 1.00
    Trading makes 5%             -> equity 10,500, unit value 1.05
    Deposit 2,100 at 1.05        -> 2,000 new units, 12,000 units,
                                    equity 12,600, unit value still 1.05
    Trading makes 10% more       -> equity 13,860, unit value 1.155

    Adjusted return = 15.5% (1.05 * 1.10 - 1), which is what the trading did.
    Raw equity rose 38.6%, most of which was the deposit.

Because a deposit needs the current unit value, `deposit()` and `withdraw()`
need prices whenever a position is open (convention 9).

Standard library only.
"""

import math
from dataclasses import dataclass
from datetime import datetime

from src.accounting.instruments import Instrument

# Quantities smaller than this are treated as zero, so that float dust left by
# fractional (crypto) quantities, like 0.3 - 0.1 - 0.2, closes a position cleanly.
QUANTITY_EPSILON = 1e-12

# How far below zero cash may dip from float rounding before a buy or a
# withdrawal is refused.
CASH_EPSILON = 1e-9

# How closely the two sides of the accounting identity must agree. It is scaled
# by the size of the account in check_identity(), so it is a relative tolerance.
IDENTITY_TOLERANCE = 1e-9

# The note that roll() puts on both of its fills.
ROLL_NOTE = "roll"


class LedgerError(Exception):
    """The ledger was asked to do something that would make the books wrong."""


class InsufficientCash(LedgerError):
    """A buy or a withdrawal needs more cash than the ledger holds."""


class MissingPrice(LedgerError):
    """A held position cannot be valued because no price was given for it."""


@dataclass(frozen=True)
class Fill:
    """One executed (simulated) trade. Frozen: a recorded fill never changes."""

    timestamp: datetime            # when it happened; must be timezone-aware
    symbol: str                    # must match an Instrument the ledger knows
    quantity: float                # signed: + buy, - sell; never 0
    price: float                   # the price actually paid or received, above 0
    fee: float = 0.0               # cash cost of the trade, 0 or more
    reference_price: float | None = None   # the price the order was decided against
    note: str = ""                 # free text, e.g. "roll"

    def __post_init__(self):
        """Refuse a fill that could not have happened or cannot be accounted for."""
        _require_aware(self.timestamp)
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("fill symbol must be a non-empty string")
        if not _is_number(self.quantity) or self.quantity == 0:
            raise ValueError(f"fill quantity must be a non-zero number, got {self.quantity!r}")
        if not _is_number(self.price) or self.price <= 0:
            raise ValueError(f"fill price must be above 0, got {self.price!r}")
        if not _is_number(self.fee) or self.fee < 0:
            raise ValueError(f"fill fee must be 0 or more, got {self.fee!r}")
        if self.reference_price is not None:
            if not _is_number(self.reference_price) or self.reference_price <= 0:
                raise ValueError(
                    f"reference_price must be above 0 or None, got {self.reference_price!r}")

    def slippage_cost(self, multiplier: float = 1.0) -> float:
        """Money lost to slippage on this fill (negative = the fill beat the reference).

        (price - reference_price) * quantity * multiplier. For a buy, paying
        more than the reference is a cost. For a sell the quantity is negative,
        so receiving less than the reference also comes out positive.
        Returns 0.0 when there is no reference price to compare against.
        """
        if self.reference_price is None:
            return 0.0
        return (self.price - self.reference_price) * self.quantity * multiplier


@dataclass(frozen=True)
class Position:
    """What the ledger holds in one symbol.

    `quantity` is signed (+ long, - short). `avg_price` is the average entry
    price; it is 0.0 and meaningless when `quantity` is 0 (flat).
    """

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        """True when nothing is held."""
        return abs(self.quantity) < QUANTITY_EPSILON


class Ledger:
    """Cash, positions and the full history of what changed them.

    See the module docstring for every convention used here.
    """

    def __init__(self, starting_cash: float, instruments):
        """Open the books.

        `instruments` is either a dict of symbol -> Instrument or a list of
        Instruments. A fill for a symbol that is not in it is refused.
        """
        if not _is_number(starting_cash) or starting_cash < 0:
            raise ValueError(f"starting_cash must be 0 or more, got {starting_cash!r}")
        self.instruments = _index_instruments(instruments)
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)

        self._positions = {}          # symbol -> Position, open positions only
        self.fills = []               # every Fill, in the order it was recorded
        self.cash_flows = []          # every deposit and withdrawal, in order
        self.gross_realized_pnl = 0.0   # price movement only; fees are NOT in here
        self.realized_pnl_by_symbol = {}   # the same number, split by symbol
        self.total_fees = 0.0         # every fee paid, kept apart from P&L
        self.total_slippage_cost = 0.0   # informational; already inside fill prices
        self.net_cash_flow = 0.0      # deposits minus withdrawals

        # Unit-value bookkeeping (see "CASH-FLOW-ADJUSTED EQUITY" above).
        self._units = float(starting_cash)   # units outstanding; 1.0 each at the start
        self._unit_value = 1.0               # last known value of one unit
        self._equity_curve = []              # snapshots appended by record_equity()

        self._last_timestamp = None   # the one clock every event must respect

    # ------------------------------------------------------------------ lookups
    def instrument(self, symbol: str) -> Instrument:
        """The Instrument for `symbol`, or LedgerError if the ledger does not know it."""
        if symbol not in self.instruments:
            raise LedgerError(f"unknown symbol {symbol!r}: no instrument was registered for it")
        return self.instruments[symbol]

    def position(self, symbol: str) -> Position:
        """The position in `symbol` (a flat Position if nothing is held)."""
        self.instrument(symbol)   # an unknown symbol is an error, not "flat"
        return self._positions.get(symbol, Position(symbol))

    def positions(self) -> dict:
        """A copy of every open position: symbol -> Position."""
        return dict(self._positions)

    # -------------------------------------------------------------------- fills
    def record_fill(self, fill: Fill) -> Fill:
        """Apply one fill to cash, the position, P&L, fees and slippage.

        Everything is checked first, so a refused fill leaves the ledger
        exactly as it was. Returns the fill for convenience.
        """
        instrument = self.instrument(fill.symbol)
        self._check_time_order(fill.timestamp, "fill")

        old = self._positions.get(fill.symbol, Position(fill.symbol))
        new_position, realized = _apply_to_position(old, fill, instrument.multiplier)

        if instrument.is_future:
            # Futures: no notional changes hands. Cash gets the P&L that was
            # just realized (zero when opening or adding) and pays the fee.
            cash_change = realized - fill.fee
        else:
            # Equity / crypto: the full notional moves. quantity is signed, so
            # a buy (positive) takes cash out and a sell (negative) puts it in.
            cash_change = -instrument.notional(fill.price, fill.quantity) - fill.fee
            if fill.quantity > 0 and self.cash + cash_change < -CASH_EPSILON:
                raise InsufficientCash(
                    f"buying {fill.quantity} {fill.symbol} at {fill.price} needs "
                    f"{-cash_change:.2f} but only {self.cash:.2f} cash is available")

        # All checks passed: from here on nothing can fail, so the books stay whole.
        self.cash += cash_change
        if new_position.is_flat:
            self._positions.pop(fill.symbol, None)
        else:
            self._positions[fill.symbol] = new_position
        self.gross_realized_pnl += realized
        self.realized_pnl_by_symbol[fill.symbol] = (
            self.realized_pnl_by_symbol.get(fill.symbol, 0.0) + realized)
        self.total_fees += fill.fee
        self.total_slippage_cost += fill.slippage_cost(instrument.multiplier)
        self.fills.append(fill)
        self._last_timestamp = fill.timestamp
        return fill

    def roll(self, timestamp: datetime, from_symbol: str, to_symbol: str,
             close_price: float, open_price: float,
             fee_close: float = 0.0, fee_open: float = 0.0,
             reference_close: float | None = None,
             reference_open: float | None = None) -> tuple:
        """Move a whole futures position from an expiring contract to the next one.

        This is two ordinary fills, both tagged with the note "roll":
          1. close the entire position in `from_symbol` at `close_price`
          2. open the same signed quantity in `to_symbol` at `open_price`

        The old contract's P&L is realized at its own close price. The new
        contract starts at its own open price. The price gap between the two
        contracts (the calendar spread) is therefore never booked as P&L.

        Refused (and nothing changes) if: there is no position in
        `from_symbol`, either symbol is not a future, the two symbols are the
        same, or their multipliers differ. Returns (closing fill, opening fill).

        A roll only happens when this method is called. The ledger never rolls,
        renames or re-symbols a position by itself.
        """
        old_instrument = self.instrument(from_symbol)
        new_instrument = self.instrument(to_symbol)
        if not old_instrument.is_future or not new_instrument.is_future:
            raise LedgerError(
                f"roll needs two futures; {from_symbol!r} is {old_instrument.asset_class} "
                f"and {to_symbol!r} is {new_instrument.asset_class}")
        if from_symbol == to_symbol:
            raise LedgerError(f"cannot roll {from_symbol!r} into itself")
        if not math.isclose(old_instrument.multiplier, new_instrument.multiplier):
            raise LedgerError(
                f"cannot roll {from_symbol!r} (multiplier {old_instrument.multiplier}) into "
                f"{to_symbol!r} (multiplier {new_instrument.multiplier}): the multipliers "
                "differ, so the same quantity would not be the same exposure")
        held = self.position(from_symbol)
        if held.is_flat:
            raise LedgerError(f"cannot roll {from_symbol!r}: there is no position in it")
        self._check_time_order(timestamp, "roll")

        # Build both fills before applying either. If a price or fee is invalid
        # the error is raised here, before the first leg has touched the books.
        closing = Fill(timestamp, from_symbol, -held.quantity, close_price,
                       fee=fee_close, reference_price=reference_close, note=ROLL_NOTE)
        opening = Fill(timestamp, to_symbol, held.quantity, open_price,
                       fee=fee_open, reference_price=reference_open, note=ROLL_NOTE)
        self.record_fill(closing)
        self.record_fill(opening)
        return closing, opening

    # --------------------------------------------------------------- cash flows
    def deposit(self, timestamp: datetime, amount: float, prices: dict | None = None) -> dict:
        """Add outside money. Changes the number of units, not the unit value.

        `prices` is required whenever a position is open, because the new units
        are bought at the current unit value and that needs current equity.
        Returns the cash-flow record that was stored.
        """
        return self._external_cash_flow(timestamp, amount, prices, "deposit")

    def withdraw(self, timestamp: datetime, amount: float, prices: dict | None = None) -> dict:
        """Take money out. Changes the number of units, not the unit value.

        Raises InsufficientCash if `amount` is more than the cash held, and
        LedgerError if it is more than the whole account is worth. `prices` is
        required whenever a position is open (see deposit()).
        """
        return self._external_cash_flow(timestamp, amount, prices, "withdrawal")

    def _external_cash_flow(self, timestamp, amount, prices, kind) -> dict:
        """Shared work for deposit() and withdraw(): validate, then buy or sell units."""
        _require_aware(timestamp)
        if not _is_number(amount) or amount <= 0:
            raise ValueError(f"{kind} amount must be above 0, got {amount!r}")
        self._check_time_order(timestamp, kind)

        signed_amount = amount if kind == "deposit" else -amount
        if kind == "withdrawal" and amount > self.cash + CASH_EPSILON:
            raise InsufficientCash(
                f"cannot withdraw {amount:.2f}: only {self.cash:.2f} cash is available")

        # Price the units BEFORE the money moves, so the flow cannot change the price.
        equity_before = self.equity(prices or {})
        unit_value = self._current_unit_value(equity_before)
        if unit_value <= 0:
            raise LedgerError(
                f"cannot process a {kind}: the account is worth {equity_before:.2f}, "
                "so units cannot be priced")
        if kind == "withdrawal" and amount > equity_before + CASH_EPSILON:
            raise LedgerError(
                f"cannot withdraw {amount:.2f}: the whole account is worth {equity_before:.2f}")

        self.cash += signed_amount
        self.net_cash_flow += signed_amount
        self._units += signed_amount / unit_value
        if abs(self._units) < QUANTITY_EPSILON:
            self._units = 0.0          # everything was withdrawn; clear float dust
        self._unit_value = unit_value
        self._last_timestamp = timestamp
        record = {"timestamp": timestamp, "kind": kind, "amount": signed_amount,
                  "unit_value": unit_value, "units_after": self._units}
        self.cash_flows.append(record)
        return record

    # ---------------------------------------------------------------- valuation
    def mark_to_market(self, prices: dict) -> dict:
        """Unrealized P&L of every open position at the given prices.

        Returns {"by_symbol": {symbol: unrealized P&L}, "total": the sum}.
        Raises MissingPrice if any held symbol has no price in `prices`.
        Unrealized P&L = (price - average entry) * quantity * multiplier, which
        is sign-correct for shorts because quantity is negative.
        """
        by_symbol = {}
        for symbol, position in self._positions.items():
            price = self._price_for(symbol, prices)
            multiplier = self.instruments[symbol].multiplier
            by_symbol[symbol] = (price - position.avg_price) * position.quantity * multiplier
        return {"by_symbol": by_symbol, "total": sum(by_symbol.values())}

    def equity(self, prices: dict) -> float:
        """What the account is worth at the given prices.

        cash + market value of equity/crypto positions (negative for shorts)
             + unrealized P&L of futures positions.
        Futures add only their unrealized P&L because their notional never left cash.
        Raises MissingPrice if any held symbol has no price.
        """
        total = self.cash
        for symbol, position in self._positions.items():
            price = self._price_for(symbol, prices)
            instrument = self.instruments[symbol]
            if instrument.is_future:
                total += (price - position.avg_price) * position.quantity * instrument.multiplier
            else:
                total += instrument.notional(price, position.quantity)
        return total

    def record_equity(self, timestamp: datetime, prices: dict) -> dict:
        """Append one equity snapshot to the curve and return it.

        Call it whenever a point on the equity curve is wanted (for example at
        every bar close). It reads prices but changes no cash or position.
        """
        _require_aware(timestamp)
        self._check_time_order(timestamp, "equity snapshot")
        equity = self.equity(prices)
        self._unit_value = self._current_unit_value(equity)
        self._last_timestamp = timestamp
        snapshot = {
            "timestamp": timestamp,
            "equity": equity,
            "net_cash_flow_to_date": self.net_cash_flow,
            "unit_value": self._unit_value,
            "adjusted_return": self._unit_value - 1.0,
        }
        self._equity_curve.append(snapshot)
        return dict(snapshot)

    def equity_curve(self) -> list:
        """Every snapshot so far, oldest first, as a list of dicts.

        Keys: timestamp, equity, net_cash_flow_to_date, unit_value (1.0 at the
        start) and adjusted_return (unit_value - 1, the time-weighted return).
        """
        return [dict(snapshot) for snapshot in self._equity_curve]

    # ------------------------------------------------------------------ reports
    def summary(self, prices: dict) -> dict:
        """Everything about the account in one dict, valued at `prices`."""
        marks = self.mark_to_market(prices)
        return {
            "cash": self.cash,
            "positions": {symbol: {"quantity": position.quantity, "avg_price": position.avg_price}
                          for symbol, position in self._positions.items()},
            "equity": self.equity(prices),
            "realized_pnl": self.gross_realized_pnl,     # GROSS: fees are not inside it
            "unrealized_pnl": marks["total"],
            "unrealized_pnl_by_symbol": marks["by_symbol"],
            "total_fees": self.total_fees,
            "total_slippage_cost": self.total_slippage_cost,   # informational only
            "net_pnl": self.gross_realized_pnl + marks["total"] - self.total_fees,
            "net_external_cash_flow": self.net_cash_flow,
            "number_of_fills": len(self.fills),
        }

    def check_identity(self, prices: dict) -> bool:
        """Prove the books add up; raise LedgerError if they do not.

        starting cash + net external cash flow + gross realized P&L
            + unrealized P&L - total fees  must equal  equity.
        The left side is built from the running totals, the right side from
        cash and positions, so a bug in either shows up as a difference.
        """
        unrealized = self.mark_to_market(prices)["total"]
        expected = (self.starting_cash + self.net_cash_flow + self.gross_realized_pnl
                    + unrealized - self.total_fees)
        actual = self.equity(prices)
        # Scale the tolerance by the size of the numbers involved.
        allowed = IDENTITY_TOLERANCE * max(1.0, abs(expected), abs(actual))
        if abs(expected - actual) > allowed:
            raise LedgerError(
                f"accounting identity broken: expected equity {expected!r} from the running "
                f"totals but cash and positions give {actual!r} (difference {expected - actual!r})")
        return True

    # ------------------------------------------------------------------ helpers
    def _price_for(self, symbol: str, prices: dict) -> float:
        """The caller's price for a held symbol. Never guessed, never defaulted."""
        if prices is None or symbol not in prices:
            raise MissingPrice(f"no price given for held symbol {symbol!r}")
        price = prices[symbol]
        if not _is_number(price):
            raise MissingPrice(f"price for held symbol {symbol!r} is not a usable number: {price!r}")
        return price

    def _check_time_order(self, timestamp: datetime, what: str) -> None:
        """Refuse anything timestamped before the last recorded event."""
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise LedgerError(
                f"{what} at {timestamp.isoformat()} is earlier than the last recorded event "
                f"at {self._last_timestamp.isoformat()}; the ledger is append-only and time-ordered")

    def _current_unit_value(self, equity: float) -> float:
        """equity / units, or the last known unit value when no units are outstanding."""
        if self._units > QUANTITY_EPSILON:
            return equity / self._units
        return self._unit_value


def _apply_to_position(old: Position, fill: Fill, multiplier: float) -> tuple:
    """Work out the position after a fill and the gross P&L it realizes.

    Returns (new Position, realized P&L). Changes nothing itself, which is what
    lets record_fill() check everything before touching the books.
    """
    same_direction = old.is_flat or (old.quantity > 0) == (fill.quantity > 0)
    if same_direction:
        # Opening or adding: re-average the entry price, realize nothing.
        new_quantity = old.quantity + fill.quantity
        cost = old.avg_price * abs(old.quantity) + fill.price * abs(fill.quantity)
        return Position(old.symbol, new_quantity, cost / abs(new_quantity)), 0.0

    # Reducing, closing or flipping. Only as much as is held can be closed.
    closed = min(abs(fill.quantity), abs(old.quantity))
    direction = 1.0 if old.quantity > 0 else -1.0      # +1 long, -1 short
    realized = (fill.price - old.avg_price) * closed * direction * multiplier

    new_quantity = old.quantity + fill.quantity
    if abs(new_quantity) < QUANTITY_EPSILON:
        return Position(old.symbol), realized                       # fully closed
    if (new_quantity > 0) == (old.quantity > 0):
        return Position(old.symbol, new_quantity, old.avg_price), realized   # partly closed
    return Position(old.symbol, new_quantity, fill.price), realized  # flipped: opens at fill price


def _index_instruments(instruments) -> dict:
    """Turn a dict or list of Instruments into a checked symbol -> Instrument dict."""
    if isinstance(instruments, dict):
        items = list(instruments.items())
    else:
        items = [(getattr(instrument, "symbol", None), instrument) for instrument in instruments]
    indexed = {}
    for symbol, instrument in items:
        if not isinstance(instrument, Instrument):
            raise ValueError(f"expected an Instrument, got {instrument!r}")
        if symbol != instrument.symbol:
            raise ValueError(f"instrument {instrument.symbol!r} was registered under {symbol!r}")
        if symbol in indexed:
            raise ValueError(f"instrument {symbol!r} was given twice")
        indexed[symbol] = instrument
    currencies = {instrument.currency for instrument in indexed.values()}
    if len(currencies) > 1:
        raise ValueError(
            f"all instruments must share one currency, got {sorted(currencies)}: "
            "the ledger has a single cash balance and does no FX conversion")
    return indexed


def _require_aware(timestamp) -> None:
    """Refuse anything that is not a timezone-aware datetime."""
    if not isinstance(timestamp, datetime):
        raise ValueError(f"timestamp must be a datetime, got {timestamp!r}")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(
            f"timestamp {timestamp!r} has no timezone; use a timezone-aware datetime (e.g. UTC)")


def _is_number(value) -> bool:
    """True for a real, finite int or float (bool, NaN and infinity are refused)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)
