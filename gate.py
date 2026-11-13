"""Risk gate: every candidate must pass this before anything else happens.

Run from the trading-terminal folder:

    python gate.py

It backtests every candidate with engine.py (train and validation windows only)
and prints PASS or BLOCK for each one. For a BLOCK it prints every rule that
failed, with the numbers behind it.

The gate uses fixed rules. It never reads the test window, and it has no
setting that loosens itself when few candidates pass. If every candidate is
blocked, that is the result.
"""

import sys

import engine

# ============================================================================
# THRESHOLDS - the only place the gate's numbers live.
# Set by the owner. Do not change these to get more passes.
# ============================================================================
MIN_TRAIN_RETURN_PCT = 0.0          # rule 1: train return must be ABOVE this (percent)
MIN_VALIDATION_RETURN_PCT = 0.0     # rule 1: validation return must be ABOVE this (percent)
MAX_RETURN_GAP_PCT_POINTS = 2.0     # rule 2: |train return - validation return| at most this
MAX_VALIDATION_DRAWDOWN_PCT = 1.5   # rule 3: validation max drawdown at most this (percent)
MIN_VALIDATION_TRADES = 2           # rule 4: validation trade count at least this
# ============================================================================


def check_candidate(name: str, train_return: float, validation_return: float,
                    validation_drawdown: float, validation_trades: int) -> dict:
    """Apply all four rules to one candidate and return its verdict.

    Returns and drawdown come in as fractions (0.0123 means 1.23%).
    Every rule is checked; `failures` lists every one that failed, not just the first.
    """
    # Work in percent so the messages match the thresholds above.
    train_pct = train_return * 100
    validation_pct = validation_return * 100
    drawdown_pct = validation_drawdown * 100
    gap_points = abs(train_pct - validation_pct)

    failures = []        # plain-English lines, one per failed rule
    failure_codes = []   # short labels for the same rules, used by memory.py

    # Rule 1: both returns must be positive.
    if not train_pct > MIN_TRAIN_RETURN_PCT:
        failures.append(f"train return {train_pct:+.2f}% <= {MIN_TRAIN_RETURN_PCT:g}")
        failure_codes.append("train_return")
    if not validation_pct > MIN_VALIDATION_RETURN_PCT:
        failures.append(f"validation return {validation_pct:+.2f}% <= {MIN_VALIDATION_RETURN_PCT:g}")
        failure_codes.append("validation_return")

    # Rule 2: train and validation must not be far apart.
    if not gap_points <= MAX_RETURN_GAP_PCT_POINTS:
        failures.append(
            f"train/validation gap {gap_points:.2f} pts > {MAX_RETURN_GAP_PCT_POINTS:g} pts "
            f"(train {train_pct:+.2f}%, validation {validation_pct:+.2f}%)"
        )
        failure_codes.append("return_gap")

    # Rule 3: validation drawdown must be small enough.
    if not drawdown_pct <= MAX_VALIDATION_DRAWDOWN_PCT:
        failures.append(
            f"validation max drawdown {drawdown_pct:.2f}% > {MAX_VALIDATION_DRAWDOWN_PCT:g}%"
        )
        failure_codes.append("validation_drawdown")

    # Rule 4: validation must have enough trades to mean anything.
    if not validation_trades >= MIN_VALIDATION_TRADES:
        failures.append(f"validation trade count {validation_trades} < {MIN_VALIDATION_TRADES}")
        failure_codes.append("validation_trades")

    return {
        "name": name,
        "verdict": "PASS" if not failures else "BLOCK",
        "failures": failures,
        "failure_codes": failure_codes,
        "numbers": {
            "train_return_pct": train_pct,
            "validation_return_pct": validation_pct,
            "gap_pct_points": gap_points,
            "validation_drawdown_pct": drawdown_pct,
            "validation_trades": validation_trades,
        },
    }


def check_backtest_result(result: dict) -> dict:
    """Apply the gate to one row produced by engine.evaluate_candidate()."""
    verdict = check_candidate(
        name=result["candidate"]["name"],
        train_return=result["train"]["total_return"],
        validation_return=result["validation"]["total_return"],
        validation_drawdown=result["validation"]["max_drawdown"],
        validation_trades=result["validation"]["trade_count"],
    )
    # Carry the family and parameters along so memory.py can store them.
    verdict["family"] = result["candidate"]["family"]
    verdict["params"] = result["candidate"]["params"]
    return verdict


def run_gate(candidates: list = None) -> list:
    """Backtest candidates on train and validation, then gate each one.

    With no argument it uses the full grid from engine.build_candidates().
    """
    # engine.load_bars refuses anything except the train and validation files.
    train = engine.load_bars(engine.TRAIN_FILE)
    validation = engine.load_bars(engine.VALIDATION_FILE)
    verdicts = []
    if candidates is None:
        candidates = engine.build_candidates()
    for candidate in candidates:
        result = engine.evaluate_candidate(candidate, train, validation)
        verdicts.append(check_backtest_result(result))
    return verdicts


def print_verdicts(verdicts: list) -> None:
    """Print one line per PASS, one line per failed rule for each BLOCK, then a summary."""
    print("Risk gate thresholds:")
    print(f"  1) train return > {MIN_TRAIN_RETURN_PCT:g}% and validation return > "
          f"{MIN_VALIDATION_RETURN_PCT:g}%")
    print(f"  2) train/validation gap <= {MAX_RETURN_GAP_PCT_POINTS:g} percentage points")
    print(f"  3) validation max drawdown <= {MAX_VALIDATION_DRAWDOWN_PCT:g}%")
    print(f"  4) validation trade count >= {MIN_VALIDATION_TRADES}")
    print("Test window: not opened")
    print()

    for verdict in verdicts:
        numbers = verdict["numbers"]
        if verdict["verdict"] == "PASS":
            print(
                f"PASS  {verdict['name']}: train {numbers['train_return_pct']:+.2f}%, "
                f"validation {numbers['validation_return_pct']:+.2f}%, "
                f"gap {numbers['gap_pct_points']:.2f} pts, "
                f"validation drawdown {numbers['validation_drawdown_pct']:.2f}%, "
                f"validation trades {numbers['validation_trades']}"
            )
        else:
            for failure in verdict["failures"]:
                print(f"BLOCK {verdict['name']}: {failure}")

    passed = sum(1 for verdict in verdicts if verdict["verdict"] == "PASS")
    blocked = len(verdicts) - passed
    print()
    print(f"Summary: candidates {len(verdicts)}, passed {passed}, blocked {blocked}")


def main() -> int:
    print_verdicts(run_gate())
    return 0


if __name__ == "__main__":
    sys.exit(main())
