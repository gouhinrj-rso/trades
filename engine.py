"""Backtest engine and two strategy families (EMA crossover, z-score mean reversion).

Run from the trading-terminal folder:

    python engine.py

It backtests every candidate on the train and validation windows and prints a
table. It never opens the locked test window.

The three rules this engine is built around:

1. No look-ahead. The signal for bar t is computed from bars 0..t only.
2. Next-bar fill. A signal made at the close of bar t is filled at the OPEN of
   bar t+1, never at bar t's own prices.
3. Fees always on. Every buy and every sell pays FEE_BPS_PER_SIDE.

Paper only: this file reads saved bars and does arithmetic. It places no orders.
"""

import json
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
FEE_BPS_PER_SIDE = 2.0          # 2 basis points = 0.02% charged on each buy and each sell
STARTING_CASH = 10_000.0        # paper money; only the percentage results matter

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_FILE = DATA_DIR / "train_1h.json"          # 1-hour bars (see fetch_data.py)
VALIDATION_FILE = DATA_DIR / "validation_1h.json"

# The only files this engine is allowed to read. The locked test file and the
# all-bars file (which contains the test bars) are deliberately not on the list.
ALLOWED_FILES = {TRAIN_FILE.name, VALIDATION_FILE.name}

# Parameter grids. Each combination is one candidate.
EMA_FAST_PERIODS = [5, 10, 20]          # bars (1 hour each): 5 h, 10 h, 20 h
EMA_SLOW_PERIODS = [30, 60, 120]        # bars: 30 h, 60 h (2.5 days), 120 h (5 days)
ZSCORE_WINDOWS = [20, 40, 80]           # bars: 20 h, 40 h, 80 h
ZSCORE_ENTRY_LEVELS = [1.0, 1.5, 2.0]   # enter when z-score drops below minus this


# ----------------------------------------------------------------------------
# Loading bars
# ----------------------------------------------------------------------------
def load_bars(file_path: Path) -> list:
    """Load one window of bars. Refuses any file that is not train or validation."""
    file_path = Path(file_path)
    if file_path.name not in ALLOWED_FILES:
        raise PermissionError(
            f"{file_path.name} is not allowed. The engine may only read "
            f"{sorted(ALLOWED_FILES)}; the test window is locked."
        )
    with open(file_path) as handle:
        return json.load(handle)


# ----------------------------------------------------------------------------
# Strategy family 1: EMA crossover
# ----------------------------------------------------------------------------
def ema_crossover_signals(closes: list, fast: int, slow: int) -> list:
    """Return the wanted position (0 = flat, 1 = long) after each bar's close.

    Go long when the fast EMA crosses above the slow EMA.
    Go flat when the fast EMA crosses back below the slow EMA.

    Causal by construction: the loop at index t reads closes[t] and values
    carried from earlier bars. It never reads closes[t+1] or later.
    """
    # EMA smoothing weights (the standard 2 / (period + 1) formula).
    fast_weight = 2.0 / (fast + 1)
    slow_weight = 2.0 / (slow + 1)

    signals = []
    position = 0
    fast_ema = closes[0]
    slow_ema = closes[0]
    was_above = False  # was the fast EMA above the slow EMA on the previous bar?

    for t, close in enumerate(closes):
        if t > 0:
            fast_ema = fast_weight * close + (1 - fast_weight) * fast_ema
            slow_ema = slow_weight * close + (1 - slow_weight) * slow_ema
        is_above = fast_ema > slow_ema

        # Warm-up: the slow EMA needs about `slow` bars before it means much,
        # so we do not act on crosses before then.
        warmed_up = t >= slow

        if warmed_up and is_above and not was_above:
            position = 1      # crossed above -> go long
        elif position == 1 and not is_above:
            position = 0      # crossed back below -> exit

        was_above = is_above
        signals.append(position)

    return signals


# ----------------------------------------------------------------------------
# Strategy family 2: z-score mean reversion
# ----------------------------------------------------------------------------
def zscore_reversion_signals(closes: list, window: int, entry: float) -> list:
    """Return the wanted position (0 = flat, 1 = long) after each bar's close.

    z-score = (close - rolling mean) / rolling standard deviation,
    where the rolling window is the last `window` closes INCLUDING the current bar.

    Go long when the z-score drops below -entry (price is unusually low).
    Go flat when the z-score returns to zero or above (price is back at its mean).

    Causal by construction: the window for index t is closes[t-window+1 .. t].
    """
    signals = []
    position = 0

    for t in range(len(closes)):
        # Not enough history for a full window yet -> stay flat.
        if t + 1 < window:
            signals.append(0)
            continue

        recent = closes[t + 1 - window : t + 1]   # ends at t, never beyond
        mean = sum(recent) / window
        variance = sum((value - mean) ** 2 for value in recent) / window
        deviation = variance ** 0.5

        # If prices did not move at all, the z-score is undefined; treat as 0.
        z_score = (closes[t] - mean) / deviation if deviation > 0 else 0.0

        if position == 0 and z_score < -entry:
            position = 1
        elif position == 1 and z_score >= 0:
            position = 0

        signals.append(position)

    return signals


# ----------------------------------------------------------------------------
# The backtest engine
# ----------------------------------------------------------------------------
def run_backtest(bars: list, signals: list, trade_start: int = 0,
                 fee_bps_per_side: float = FEE_BPS_PER_SIDE,
                 starting_cash: float = STARTING_CASH) -> dict:
    """Simulate trading `signals` on `bars` and return the results.

    bars        : list of bar dictionaries, oldest first
    signals     : signals[t] is the wanted position (0 or 1) decided at the
                  close of bar t, using bars 0..t only
    trade_start : index of the first bar we are allowed to trade on. Bars before
                  it are warm-up history only (used for validation, where the
                  train bars warm up the indicators but are not traded).

    Timing: the decision from bar t is filled at the OPEN of bar t+1.
    Fees: each buy and each sell loses fee_bps_per_side of its value.
    """
    if len(bars) != len(signals):
        raise ValueError("bars and signals must be the same length")

    fee_rate = fee_bps_per_side / 10_000.0   # 2 bps -> 0.0002
    cash = starting_cash
    units = 0.0            # how much of the asset we hold
    fees_paid = 0.0
    trades = []            # one entry per round trip
    equity_curve = []      # account value at each traded bar's close

    for t in range(trade_start, len(bars)):
        bar = bars[t]

        # What did we decide at the close of the PREVIOUS bar?
        # On the first tradable bar we only act on a decision made at or after
        # trade_start - 1, so the window always begins flat.
        wanted = signals[t - 1] if t - 1 >= max(trade_start - 1, 0) else 0

        # Fill at THIS bar's open (this bar is "t+1" relative to the decision).
        if wanted == 1 and units == 0.0:
            fee = cash * fee_rate
            units = (cash - fee) / bar["open"]
            fees_paid += fee
            cash = 0.0
            trades.append({"entry_time": bar["timestamp"], "entry_price": bar["open"]})
        elif wanted == 0 and units > 0.0:
            gross = units * bar["open"]
            fee = gross * fee_rate
            cash = gross - fee
            fees_paid += fee
            units = 0.0
            trades[-1].update({"exit_time": bar["timestamp"], "exit_price": bar["open"]})

        # Mark the account to this bar's close.
        equity_curve.append(cash + units * bar["close"])

    # If still long at the end, value the position at the last close and charge
    # the exit fee, so an open trade is not flattered by a missing fee.
    if units > 0.0 and equity_curve:
        last = bars[-1]
        gross = units * last["close"]
        fee = gross * fee_rate
        fees_paid += fee
        equity_curve[-1] = gross - fee
        trades[-1].update({"exit_time": last["timestamp"], "exit_price": last["close"],
                           "closed_at_end": True})

    final_equity = equity_curve[-1] if equity_curve else starting_cash
    return {
        "total_return": final_equity / starting_cash - 1.0,
        "max_drawdown": max_drawdown(equity_curve, starting_cash),
        "trade_count": len(trades),
        "fees_paid": fees_paid,
        "final_equity": final_equity,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def max_drawdown(equity_curve: list, starting_value: float) -> float:
    """Largest peak-to-trough fall in account value, as a fraction (0.05 = 5%)."""
    peak = starting_value
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    return worst


# ----------------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------------
def build_candidates() -> list:
    """Build every candidate from the two parameter grids."""
    candidates = []
    for fast in EMA_FAST_PERIODS:
        for slow in EMA_SLOW_PERIODS:
            candidates.append({
                "name": f"ema_cross(fast={fast}, slow={slow})",
                "family": "ema_crossover",
                "params": {"fast": fast, "slow": slow},
            })
    for window in ZSCORE_WINDOWS:
        for entry in ZSCORE_ENTRY_LEVELS:
            candidates.append({
                "name": f"zscore(window={window}, entry={entry})",
                "family": "zscore_reversion",
                "params": {"window": window, "entry": entry},
            })
    return candidates


def make_signals(candidate: dict, closes: list) -> list:
    """Compute the signals for one candidate on a list of closing prices."""
    params = candidate["params"]
    if candidate["family"] == "ema_crossover":
        return ema_crossover_signals(closes, params["fast"], params["slow"])
    if candidate["family"] == "zscore_reversion":
        return zscore_reversion_signals(closes, params["window"], params["entry"])
    raise ValueError(f"Unknown strategy family: {candidate['family']}")


def evaluate_candidate(candidate: dict, train: list, validation: list) -> dict:
    """Backtest one candidate on train, then on validation.

    Validation uses the train bars as warm-up history for the indicators (they
    are earlier in time, so this is not look-ahead) but only trades on, and is
    only scored on, the validation bars. It starts flat.
    """
    train_closes = [bar["close"] for bar in train]
    train_result = run_backtest(train, make_signals(candidate, train_closes))

    combined = train + validation
    combined_closes = [bar["close"] for bar in combined]
    validation_result = run_backtest(
        combined, make_signals(candidate, combined_closes), trade_start=len(train)
    )
    return {"candidate": candidate, "train": train_result, "validation": validation_result}


def buy_and_hold(bars: list) -> dict:
    """Reference only: buy at the second bar's open and hold to the end, with fees."""
    return run_backtest(bars, [1] * len(bars))


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def percent(value: float) -> str:
    """Format 0.0123 as '+1.23%'."""
    return f"{value * 100:+.2f}%"


def print_table(results: list, train: list, validation: list) -> None:
    """Print one row per candidate."""
    header = (f"{'candidate':<32} {'train ret':>10} {'val ret':>9} "
              f"{'train DD':>9} {'val DD':>8} {'train tr':>9} {'val tr':>7}")
    print(header)
    print("-" * len(header))
    for row in results:
        tr, va = row["train"], row["validation"]
        print(f"{row['candidate']['name']:<32} {percent(tr['total_return']):>10} "
              f"{percent(va['total_return']):>9} {percent(-tr['max_drawdown']):>9} "
              f"{percent(-va['max_drawdown']):>8} {tr['trade_count']:>9} {va['trade_count']:>7}")
    print("-" * len(header))

    # Reference row: what simply holding the asset did over each window.
    hold_train, hold_val = buy_and_hold(train), buy_and_hold(validation)
    print(f"{'buy and hold (reference)':<32} {percent(hold_train['total_return']):>10} "
          f"{percent(hold_val['total_return']):>9} {percent(-hold_train['max_drawdown']):>9} "
          f"{percent(-hold_val['max_drawdown']):>8} {hold_train['trade_count']:>9} "
          f"{hold_val['trade_count']:>7}")


def main() -> int:
    train = load_bars(TRAIN_FILE)
    validation = load_bars(VALIDATION_FILE)
    candidates = build_candidates()

    print(f"Train bars      : {len(train)}  ({train[0]['timestamp']} -> {train[-1]['timestamp']})")
    print(f"Validation bars : {len(validation)}  "
          f"({validation[0]['timestamp']} -> {validation[-1]['timestamp']})")
    print(f"Fee             : {FEE_BPS_PER_SIDE} bps per side")
    print(f"Fill rule       : signal at close of bar t, fill at open of bar t+1")
    print(f"Candidates      : {len(candidates)}")
    print("Test window     : not opened")
    print()

    results = [evaluate_candidate(candidate, train, validation) for candidate in candidates]
    print_table(results, train, validation)
    print()
    print("ret = total return after fees. DD = max drawdown. tr = trade count (round trips;")
    print("a position still open at the end is closed at the last close, fee charged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
