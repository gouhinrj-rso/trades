"""Run one round: read memory, propose, backtest, gate, save lessons, print memory.

Run from the trading-terminal folder:

    python run_round.py

Order of work in every round:
  1. Read memory and print it. Nothing is proposed before this.
  2. Propose candidates, skipping any that memory says already failed the same way.
  3. Backtest the rest on train and validation (never the test window).
  4. Run the risk gate.
  5. Save one lesson per blocked candidate to memory.
  6. Print the memory contents.

Each round is also logged to data/rounds.json (round number and counts).
"""

import json
import sys
from pathlib import Path

import engine
import gate
import memory as lesson_memory

ROUNDS_FILE = Path(__file__).resolve().parent / "data" / "rounds.json"


def load_round_log() -> list:
    """Read the log of past rounds (empty list if there is none yet)."""
    if not ROUNDS_FILE.exists():
        return []
    with open(ROUNDS_FILE) as handle:
        return json.load(handle)


def main() -> int:
    round_log = load_round_log()
    round_number = len(round_log) + 1
    print(f"===== ROUND {round_number} =====")
    print()

    # Step 1: read memory BEFORE proposing anything.
    store = lesson_memory.open_memory()
    lessons_before = lesson_memory.read_lessons(store)
    lesson_memory.print_memory(lessons_before, title="Step 1 - memory read before proposing")
    print()

    # Step 2: propose, minus anything memory rules out.
    proposals = engine.build_candidates()
    allowed, skipped = lesson_memory.filter_proposals(proposals, lessons_before)
    print(f"Step 2 - proposals: {len(proposals)} in the grid, "
          f"{len(skipped)} skipped because of memory, {len(allowed)} to test")
    for entry in skipped:
        print(f"  SKIP {entry['candidate']['name']}: {entry['reason']}")
    print()

    # Steps 3 and 4: backtest and gate what is left.
    print("Steps 3-4 - backtest and risk gate")
    verdicts = gate.run_gate(allowed)
    gate.print_verdicts(verdicts)
    print()

    # Step 5: save one lesson per blocked candidate.
    saved = lesson_memory.save_lessons(store, verdicts, round_number)
    print(f"Step 5 - lessons saved this round: {len(saved)}")
    print()

    # Step 6: print everything now in memory.
    lessons_after = lesson_memory.read_lessons(store)
    lesson_memory.print_memory(lessons_after, title="Step 6 - memory contents after this round")

    # Log the round.
    passed = [verdict["name"] for verdict in verdicts if verdict["verdict"] == "PASS"]
    round_log.append({
        "round": round_number,
        "proposed": len(proposals),
        "skipped_by_memory": len(skipped),
        "tested": len(verdicts),
        "passed": len(passed),
        "blocked": len(verdicts) - len(passed),
        "passed_candidates": passed,
        "lessons_saved": len(saved),
        "lessons_in_memory": len(lessons_after),
    })
    with open(ROUNDS_FILE, "w") as handle:
        json.dump(round_log, handle, indent=1)

    print()
    print(f"Round {round_number} summary: proposed {len(proposals)}, skipped {len(skipped)}, "
          f"tested {len(verdicts)}, passed {len(passed)}, blocked {len(verdicts) - len(passed)}, "
          f"lessons in memory {len(lessons_after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
