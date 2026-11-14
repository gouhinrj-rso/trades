"""Instruments: what one unit of a tradable thing is worth.

The ledger needs three facts about anything it holds:

* `multiplier` - how many units of the underlying one contract (or share, or
  coin) stands for. A share of stock is 1.0. A futures contract is whatever
  its exchange says (for example 50 index points' worth of dollars per point).
* `tick_size`  - the smallest step the price is allowed to move by.
* `asset_class` - "equity", "future" or "crypto". The ledger treats futures
  cash differently from the other two (see ledger.py).

From those two numbers everything else follows:

    tick_value = tick_size * multiplier          (money per one-tick move, per contract)
    notional   = price * quantity * multiplier   (money value of a position)

There is deliberately NO table of real contract specifications in this file.
Multipliers and tick sizes change and differ between exchanges, so the caller
must state them when building an `Instrument`. Nothing here should be read as
an authoritative spec for any real contract.

Standard library only. No network, no broker, no orders.
"""

import math
from dataclasses import dataclass

# The only asset classes the ledger knows how to account for.
ASSET_CLASSES = ("equity", "future", "crypto")

# How far (measured in ticks) a price may sit from an exact tick and still count
# as "on tick". Floats cannot store most decimals exactly (0.1 + 0.2 is not 0.3),
# so an exact comparison would wrongly reject good prices.
TICK_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Instrument:
    """One tradable thing and the numbers needed to turn its price into money.

    Frozen: once built it cannot be changed, so a ledger can rely on it.
    """

    symbol: str            # the name fills refer to, e.g. "ETHUSD"
    asset_class: str       # "equity", "future" or "crypto"
    multiplier: float = 1.0   # contract multiplier; 1.0 for equities
    tick_size: float = 0.01   # smallest allowed price step
    currency: str = "USD"     # the currency prices and cash are in

    def __post_init__(self):
        """Refuse an instrument whose numbers make no sense."""
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(
                f"unknown asset class {self.asset_class!r}; expected one of {ASSET_CLASSES}")
        if not _is_positive_number(self.multiplier):
            raise ValueError(f"multiplier must be a number above 0, got {self.multiplier!r}")
        if not _is_positive_number(self.tick_size):
            raise ValueError(f"tick_size must be a number above 0, got {self.tick_size!r}")
        if not isinstance(self.currency, str) or not self.currency:
            raise ValueError("currency must be a non-empty string")

    @property
    def is_future(self) -> bool:
        """True for futures, which the ledger settles in cash differently."""
        return self.asset_class == "future"

    @property
    def tick_value(self) -> float:
        """Money gained or lost per contract when the price moves one tick."""
        return self.tick_size * self.multiplier

    def notional(self, price: float, quantity: float) -> float:
        """Money value of `quantity` units at `price`. Signed like `quantity`."""
        return price * quantity * self.multiplier

    def is_on_tick(self, price: float) -> bool:
        """True if `price` is a whole number of ticks, allowing for float error."""
        ticks = price / self.tick_size
        return abs(ticks - round(ticks)) <= TICK_TOLERANCE

    def round_to_tick(self, price: float) -> float:
        """The nearest valid price to `price`. Exactly half way rounds up.

        (Python's own round() sends halves to the nearest even number, which
        would surprise a beginner, so this uses floor(x + 0.5) instead.)
        """
        ticks = math.floor(price / self.tick_size + 0.5)
        # Multiplying back can leave float dust (3 * 0.1 = 0.30000000000000004);
        # rounding to 10 decimals removes it without moving any real price.
        return round(ticks * self.tick_size, 10)


def _is_positive_number(value) -> bool:
    """True for a real, finite number above zero (bool, NaN and infinity are refused)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0
