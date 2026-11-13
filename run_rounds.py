"""Run three full rounds: memory -> propose -> backtest -> gate -> save lessons -> optimize.

Run from the repo root (needs the model key in the environment):

    python run_rounds.py

Each round:
  1. reads memory and prints it
  2. asks the model for candidates, using the current best proposal prompt
  3. backtests them on train and validation (2 bps fee per side)
  4. runs the risk gate
  5. saves one lesson per blocked candidate to memory
  6. optimizes the proposal prompt with DSPy, under the 20-trial ceiling

It always stops after ROUNDS rounds. It never reads the test window (bars come
through engine.load_bars) and never changes the gate thresholds.
"""

import json
import sys
from collections import Counter

import engine
import gate
import memory as lesson_memory
import optimizer
from run_round import ROUNDS_FILE, load_round_log

ROUNDS = 3                      # hard stop
CANDIDATES_PER_ROUND = 8        # the most candidates the proposer may return
TRIALS_PER_ROUND = 20           # optimizer ceiling per round

REASON_NAMES = {
    "train_return": "train return not above 0",
    "validation_return": "validation return not above 0",
    "return_gap": "train/validation gap over 2 points",
    "validation_drawdown": "validation drawdown over 1.5%",
    "validation_trades": "fewer than 2 validation trades",
}


def current_prompt() -> tuple:
    """Return (version, instructions) of the best saved proposal prompt."""
    index = optimizer.read_prompt_index(optimizer.PROMPTS_DIR)
    if index["best_version"] is None:
        return None, optimizer.BASELINE_INSTRUCTIONS
    record = optimizer.read_prompt(optimizer.PROMPTS_DIR, index["best_version"])
    return record["version"], record["instructions"]


def propose(instructions: str, lessons: list, lm) -> tuple:
    """Ask the model for candidates. Returns (valid, rejected, error_text)."""
    import dspy
    program = optimizer.build_proposer(instructions)
    try:
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            answer = program(lessons=optimizer.describe_lessons(lessons),
                             allowed_families=optimizer.describe_families(),
                             max_candidates=CANDIDATES_PER_ROUND)
    except Exception as problem:
        return [], [], f"{type(problem).__name__}: {str(problem)[:200]}"
    valid, rejected = optimizer.clean_proposals(answer.candidates, CANDIDATES_PER_ROUND)
    return valid, rejected, None


def top_reason(verdicts: list) -> str:
    """The failed check that blocked the most candidates this round."""
    counts = Counter(code for v in verdicts if v["verdict"] == "BLOCK" for code in v["failure_codes"])
    if not counts:
        return "none (nothing blocked)"
    code, count = counts.most_common(1)[0]
    return f"{REASON_NAMES.get(code, code)} ({count} candidate(s))"


def run_one_round(position: int, store, fresh_lm, proposer_lm, reflection_lm, windows: dict) -> dict:
    round_log = load_round_log()
    round_number = len(round_log) + 1
    print()
    print(f"################ ROUND {position} of {ROUNDS}  (round {round_number} in memory) ################")

    # 1. Memory first.
    lessons = lesson_memory.read_lessons(store)
    lesson_memory.print_memory(lessons, title="Step 1 - memory read before proposing")

    # 2. Propose with the current prompt.
    version, instructions = current_prompt()
    print()
    print(f"Step 2 - proposing with prompt version {version or 'baseline (unsaved)'}")
    valid, rejected, error = propose(instructions, lessons, fresh_lm)
    if error:
        print(f"  The proposal call failed: {error}")
    for candidate in valid:
        print(f"  proposed  {candidate['name']}")
    for entry in rejected:
        print(f"  invalid   {entry['label']}: {entry['reason']}")
    allowed, skipped = lesson_memory.filter_proposals(valid, lessons)
    for entry in skipped:
        print(f"  SKIP      {entry['candidate']['name']}: {entry['reason']}")

    # 3 and 4. Backtest (train + validation, 2 bps per side) and gate.
    print()
    print(f"Steps 3-4 - backtest ({engine.FEE_BPS_PER_SIDE:g} bps per side) and risk gate")
    verdicts = gate.run_gate(allowed) if allowed else []
    if verdicts:
        gate.print_verdicts(verdicts)
    else:
        print("  Nothing to test.")

    # 5. Lessons.
    saved = lesson_memory.save_lessons(store, verdicts, round_number)
    lessons_after = lesson_memory.read_lessons(store)
    print()
    print(f"Step 5 - lessons saved this round: {len(saved)} (memory now holds {len(lessons_after)})")
    for text in saved:
        print(f"  {text}")

    passed = [v["name"] for v in verdicts if v["verdict"] == "PASS"]
    blocked = [v["name"] for v in verdicts if v["verdict"] == "BLOCK"]
    reason = top_reason(verdicts)

    # 6. Optimize the prompt.
    print()
    print(f"Step 6 - optimizing the proposal prompt (ceiling {TRIALS_PER_ROUND} trials)")
    try:
        outcome = optimizer.run_optimization(proposer_lm, reflection_lm, lessons_after, windows,
                                             max_trials=TRIALS_PER_ROUND)
        optimizer_summary = {
            "trials": len(outcome["trials"]),
            "proposer_requests": outcome["proposer_requests"],
            "prompts_tried": [{"prompt": r["prompt"], "train": r["train"], "validation": r["validation"],
                               "judgement": r["judgement"]} for r in outcome["results"]],
            "promoted_version": outcome["promoted_version"],
            "flagged_versions": outcome["flagged_versions"],
        }
    except Exception as problem:
        print(f"  The optimizer stopped with an error: {type(problem).__name__}: {str(problem)[:200]}")
        optimizer_summary = {"error": f"{type(problem).__name__}: {str(problem)[:200]}"}

    record = {
        "round": round_number,
        "run_position": position,
        "proposer": "model",
        "prompt_version": version,
        "lessons_at_start": len(lessons),
        "proposed": len(valid) + len(rejected),
        "proposed_valid": [c["name"] for c in valid],
        "invalid": len(rejected),
        "skipped_by_memory": len(skipped),
        "skipped_names": [e["candidate"]["name"] for e in skipped],
        "tested": len(verdicts),
        "passed": len(passed),
        "blocked": len(blocked),
        "passed_candidates": passed,
        "blocked_candidates": blocked,
        "top_reason": reason,
        "lessons_saved": len(saved),
        "lessons_in_memory": len(lessons_after),
        "proposal_error": error,
        "optimizer": optimizer_summary,
    }
    round_log.append(record)
    with open(ROUNDS_FILE, "w") as handle:
        json.dump(round_log, handle, indent=1)

    print()
    print(f"ROUND {position} RESULT: candidates {len(verdicts)}, passed {len(passed)}, "
          f"blocked {len(blocked)}, top reason: {reason}")
    return record


def print_side_by_side(records: list) -> None:
    def optimizer_cell(r):
        o = r["optimizer"]
        if "error" in o:
            return "error"
        if o["promoted_version"]:
            return f"new best v{o['promoted_version']}"
        if o["flagged_versions"]:
            return f"overfit v{o['flagged_versions']}"
        return "no better prompt"

    rows = [
        ("prompt version used", lambda r: f"v{r['prompt_version']}"),
        ("lessons in memory at start", lambda r: r["lessons_at_start"]),
        ("proposed by the model", lambda r: r["proposed"]),
        ("invalid", lambda r: r["invalid"]),
        ("skipped by memory", lambda r: r["skipped_by_memory"]),
        ("candidates tested", lambda r: r["tested"]),
        ("passed", lambda r: r["passed"]),
        ("blocked", lambda r: r["blocked"]),
        ("lessons saved", lambda r: r["lessons_saved"]),
        ("optimizer trials", lambda r: r["optimizer"].get("trials", "-")),
        ("optimizer result", optimizer_cell),
    ]
    print()
    print("================ THREE ROUNDS SIDE BY SIDE ================")
    print(f"{'':<28}" + "".join(f"{'Round ' + str(r['run_position']):>20}" for r in records))
    for label, pick in rows:
        print(f"{label:<28}" + "".join(f"{str(pick(r)):>20}" for r in records))
    print()
    for r in records:
        print(f"Round {r['run_position']} top reason: {r['top_reason']}")

    print()
    print("Candidates proposed in each round (PASS / BLOCK / SKIP = skipped by memory):")
    for r in records:
        print(f"  Round {r['run_position']}:")
        for name in r["proposed_valid"]:
            status = ("PASS" if name in r["passed_candidates"] else
                      "BLOCK" if name in r["blocked_candidates"] else "SKIP")
            print(f"    {status:<5} {name}")

    first = set(records[0]["proposed_valid"])
    print()
    print("Compared with round 1:")
    for r in records[1:]:
        now = set(r["proposed_valid"])
        new, dropped = sorted(now - first), sorted(first - now)
        print(f"  Round {r['run_position']}: {len(new)} new, {len(dropped)} dropped, {len(now & first)} repeated")
        print(f"    new    : {', '.join(new) or 'none'}")
        print(f"    dropped: {', '.join(dropped) or 'none'}")


def main() -> int:
    missing = sorted({name for name in (optimizer.missing_key_variable(optimizer.PROPOSER_MODEL),
                                        optimizer.missing_key_variable(optimizer.REFLECTION_MODEL)) if name})
    if missing:
        print(f"Cannot start: {', '.join(missing)} is not set in the environment.")
        return 2

    print(f"Running {ROUNDS} rounds, then stopping. Test window: not opened. Gate thresholds: unchanged.")
    print(f"Model calls per round, at most: 1 proposal + {TRIALS_PER_ROUND} optimizer trials + "
          f"{optimizer.plan_run([], TRIALS_PER_ROUND)['reflection_calls']} prompt rewrites.")

    store = lesson_memory.open_memory()
    windows = optimizer.load_windows()
    fresh_lm = optimizer.make_lm(optimizer.PROPOSER_MODEL, cache=False)        # each round's proposal is a new call
    proposer_lm = optimizer.make_lm(optimizer.PROPOSER_MODEL)                  # optimizer trials may reuse replies
    reflection_lm = optimizer.make_lm(optimizer.REFLECTION_MODEL, cache=False) # each rewrite is a new call

    records = [run_one_round(position, store, fresh_lm, proposer_lm, reflection_lm, windows)
               for position in range(1, ROUNDS + 1)]
    print_side_by_side(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
