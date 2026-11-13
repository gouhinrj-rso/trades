"""Optimize the prompt that proposes strategies, using DSPy.

Run from the trading-terminal folder:

    python optimizer.py            # run one optimization (needs an LLM key, see below)
    python optimizer.py --show     # print the best and previous prompts side by side

What it does:
  1. The proposal step is a DSPy module. It reads the memory lessons and the
     allowed strategy families and returns a list of candidates with parameters.
  2. Each trial scores one proposal twice, and logs both numbers:
       train metric      = candidates that pass the gate INSIDE the train window
                           (first two thirds of train vs last third of train)
       validation metric = candidates that pass the real gate (train vs validation)
  3. DSPy's GEPA optimizer rewrites the prompt between trials. It sees detailed
     feedback from the train metric only. From validation it sees one number.
  4. The best prompt and the previous prompt are saved to prompts/ with version
     numbers. A prompt that improves on train but not on validation is flagged
     as overfitting and is NOT promoted.

The test window is never read: all bars come through engine.load_bars(), which
refuses the test file.

Model settings come from environment variables (see SETTINGS). No key is ever
written in this file. DSPy reads the provider key from the environment itself.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import engine
import gate
import memory as lesson_memory

PROJECT_DIR = Path(__file__).resolve().parent

# Hide DSPy's progress bars so the trial lines stay readable.
os.environ.setdefault("TQDM_DISABLE", "1")

# ============================================================================
# SETTINGS - read from environment variables. No API keys in code.
# ============================================================================
# Model that proposes candidates (called on every trial).
PROPOSER_MODEL = os.environ.get("TT_PROPOSER_MODEL", "anthropic/claude-sonnet-5-5")
# Model that rewrites the prompt between trials (called about once per three trials).
REFLECTION_MODEL = os.environ.get("TT_REFLECTION_MODEL", "anthropic/claude-opus-5-5")
# Hard ceiling on trials (metric calls) in one optimization run.
MAX_TRIALS = int(os.environ.get("TT_MAX_TRIALS", "20"))
# Most tokens one model reply may use.
MAX_TOKENS = int(os.environ.get("TT_LLM_MAX_TOKENS", "16000"))

# The environment variable each provider's key must be in. DSPy reads it; we
# only check that it exists so we can stop with a clear message if it does not.
PROVIDER_KEY_VARIABLES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

PROMPTS_DIR = PROJECT_DIR / "prompts"
TRIAL_LOG_FILE = PROJECT_DIR / "data" / "optimizer_trials.jsonl"

# Share of the train window used as the "earlier" part for the train metric.
INNER_TRAIN_FRACTION = 2 / 3

# How many candidates the proposer may return in each of the practice tasks.
TASK_SIZES = [6, 8]

# List prices in US dollars per million tokens (input, output), used ONLY for the
# rough cost estimate. Source: Anthropic first-party API rates as cached on
# 2026-11-03. Check current prices before relying on the estimate.
PRICES_PER_MILLION_TOKENS = {
    "anthropic/claude-sonnet-5-5": (2.00, 10.00),
    "anthropic/claude-opus-5-5": (4.00, 20.00),
    "anthropic/claude-haiku-5-5": (0.10, 0.50),
}
# ============================================================================

# The strategy families the engine can run, and the allowed range of each
# parameter. The proposer may not go outside these.
FAMILY_SPECS = {
    "ema_crossover": {
        "description": ("Go long when the fast EMA crosses above the slow EMA; exit when it "
                        "crosses back. fast must be smaller than slow."),
        "params": {"fast": {"min": 2, "max": 50, "whole_number": True},
                   "slow": {"min": 10, "max": 200, "whole_number": True}},
    },
    "zscore_reversion": {
        "description": ("Go long when the z-score of price against its rolling mean drops below "
                        "minus the entry level; exit when it returns to zero."),
        "params": {"window": {"min": 10, "max": 150, "whole_number": True},
                   "entry": {"min": 0.5, "max": 3.0, "whole_number": False}},
    },
}

# Version 1 of the prompt: where every optimization starts from.
BASELINE_INSTRUCTIONS = (
    "You propose trading strategy candidates for a backtest on 5-minute price bars. "
    "Read the lessons from memory first. Each lesson records a candidate that a risk gate "
    "blocked, the exact check it failed and the numbers behind it. Do not propose a family or "
    "a parameter set that already failed the same way. Use only the allowed families and keep "
    "every parameter inside its allowed range. Return at most max_candidates candidates, with "
    "no duplicates."
)


# ----------------------------------------------------------------------------
# Text the proposer reads
# ----------------------------------------------------------------------------
def describe_families() -> str:
    """Describe the allowed families and parameter ranges in plain text."""
    lines = []
    for family, spec in FAMILY_SPECS.items():
        ranges = []
        for name, limits in spec["params"].items():
            kind = "whole number" if limits["whole_number"] else "number"
            ranges.append(f"{name}: {kind} from {limits['min']} to {limits['max']}")
        lines.append(f"- {family}: {spec['description']} Parameters: {'; '.join(ranges)}.")
    return "\n".join(lines)


def describe_lessons(lessons: list) -> str:
    """Turn the memory lessons into the text block the proposer reads."""
    if not lessons:
        return "(memory is empty: nothing has been blocked yet)"
    lines = [lesson["text"] for lesson in lessons]
    for family, info in lesson_memory.failed_families(lessons).items():
        lines.append(
            f"FAMILY RULED OUT: {family} failed as a whole in round {info['round']} "
            f"(all {info['blocked']} candidates blocked, every one on "
            f"{', '.join(info['shared_codes'])})."
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Turning the model's answer into candidates the engine can run
# ----------------------------------------------------------------------------
def candidate_name(family: str, params: dict) -> str:
    """Build the same name engine.build_candidates() uses, so memory can match it."""
    if family == "ema_crossover":
        return f"ema_cross(fast={params['fast']}, slow={params['slow']})"
    return f"zscore(window={params['window']}, entry={params['entry']})"


def clean_proposals(raw_candidates, max_candidates: int) -> tuple:
    """Check every proposed candidate. Returns (valid, rejected).

    The model's answer is treated as data. Anything outside the allowed
    families or ranges is rejected with a reason; nothing it wrote is executed.
    """
    valid, rejected, seen = [], [], set()
    for position, raw in enumerate(raw_candidates or []):
        # Accept either a dictionary or an object with .family and .params.
        family = raw.get("family") if isinstance(raw, dict) else getattr(raw, "family", None)
        params = raw.get("params") if isinstance(raw, dict) else getattr(raw, "params", None)
        label = f"#{position + 1} {family} {params}"

        if position >= max_candidates:
            rejected.append({"label": label, "reason": f"over the limit of {max_candidates} candidates"})
            continue
        if family not in FAMILY_SPECS:
            rejected.append({"label": label, "reason": f"family '{family}' is not allowed"})
            continue
        spec = FAMILY_SPECS[family]["params"]
        if not isinstance(params, dict) or set(params) != set(spec):
            rejected.append({"label": label, "reason": f"params must be exactly {sorted(spec)}"})
            continue

        cleaned, problem = {}, None
        for name, limits in spec.items():
            try:
                value = float(params[name])
            except (TypeError, ValueError):
                problem = f"{name} is not a number"
                break
            if limits["whole_number"]:
                if value != int(value):
                    problem = f"{name}={value} must be a whole number"
                    break
                value = int(value)
            else:
                value = round(value, 2)
            if not limits["min"] <= value <= limits["max"]:
                problem = f"{name}={value} is outside {limits['min']} to {limits['max']}"
                break
            cleaned[name] = value
        if problem is None and family == "ema_crossover" and cleaned["fast"] >= cleaned["slow"]:
            problem = f"fast={cleaned['fast']} must be smaller than slow={cleaned['slow']}"
        if problem:
            rejected.append({"label": label, "reason": problem})
            continue

        name = candidate_name(family, cleaned)
        if name in seen:
            rejected.append({"label": label, "reason": f"duplicate of {name}"})
            continue
        seen.add(name)
        valid.append({"name": name, "family": family, "params": cleaned})
    return valid, rejected


# ----------------------------------------------------------------------------
# Scoring one proposal on train and on validation
# ----------------------------------------------------------------------------
def load_windows() -> dict:
    """Load train and validation bars (never test) and cut the inner train split."""
    train = engine.load_bars(engine.TRAIN_FILE)
    validation = engine.load_bars(engine.VALIDATION_FILE)
    cut = int(len(train) * INNER_TRAIN_FRACTION)
    return {
        "train": train,
        "validation": validation,
        "train_early": train[:cut],    # first two thirds of train
        "train_late": train[cut:],     # last third of train
    }


def score_proposal(raw_candidates, max_candidates: int, lessons: list, windows: dict) -> dict:
    """Score one proposal. Returns both metrics and the detail behind them.

    train metric      : gate applied to (train_early -> train_late). Train bars only.
    validation metric : gate applied to (train -> validation). The owner's metric.
    """
    valid, rejected = clean_proposals(raw_candidates, max_candidates)
    allowed, skipped = lesson_memory.filter_proposals(valid, lessons)

    rows = []
    for candidate in allowed:
        on_train = gate.check_backtest_result(
            engine.evaluate_candidate(candidate, windows["train_early"], windows["train_late"]))
        on_validation = gate.check_backtest_result(
            engine.evaluate_candidate(candidate, windows["train"], windows["validation"]))
        rows.append({
            "name": candidate["name"],
            "train_verdict": on_train["verdict"],
            "train_failures": on_train["failures"],
            "validation_verdict": on_validation["verdict"],
        })

    return {
        "proposed": len(raw_candidates or []),
        "valid": len(valid),
        "rejected": rejected,
        "skipped_by_memory": [{"name": entry["candidate"]["name"], "reason": entry["reason"]}
                              for entry in skipped],
        "candidates": rows,
        "train_passes": sum(1 for row in rows if row["train_verdict"] == "PASS"),
        "validation_passes": sum(1 for row in rows if row["validation_verdict"] == "PASS"),
    }


def train_feedback(outcome: dict, max_candidates: int) -> str:
    """Detailed feedback for the reflection model. Built from TRAIN results only."""
    lines = [f"{outcome['train_passes']} of {max_candidates} possible candidates passed the gate "
             f"on the train window."]
    for row in outcome["candidates"]:
        if row["train_verdict"] == "PASS":
            lines.append(f"PASS {row['name']}")
        else:
            lines.append(f"BLOCK {row['name']}: {'; '.join(row['train_failures'])}")
    for entry in outcome["skipped_by_memory"]:
        lines.append(f"WASTED {entry['name']}: memory already rules this out ({entry['reason']})")
    for entry in outcome["rejected"]:
        lines.append(f"INVALID {entry['label']}: {entry['reason']}")
    unused = max_candidates - outcome["proposed"]
    if unused > 0:
        lines.append(f"{unused} of the {max_candidates} allowed slots were left unused.")
    return "\n".join(lines)


def validation_feedback(outcome: dict, max_candidates: int) -> str:
    """Feedback from validation is one number only, to limit leakage into the prompt."""
    return (f"{outcome['validation_passes']} of {max_candidates} possible candidates passed "
            f"the gate on validation.")


# ----------------------------------------------------------------------------
# Overfitting rule
# ----------------------------------------------------------------------------
def judge_change(base_train: int, base_validation: int,
                 new_train: int, new_validation: int) -> str:
    """Compare a new prompt with the one it started from.

    Returns one of:
      "improved"    - validation metric went up
      "overfitting" - train metric went up but validation did not
      "no_gain"     - neither went up
    """
    if new_validation > base_validation:
        return "improved"
    if new_train > base_train:
        return "overfitting"
    return "no_gain"


# ----------------------------------------------------------------------------
# Trial log
# ----------------------------------------------------------------------------
class TrialLog:
    """Writes one line per trial, with the train and validation metric separately."""

    def __init__(self, run_id: str, log_file: Path):
        self.run_id = run_id
        self.log_file = Path(log_file)
        self.rows = []
        self.prompt_labels = {}   # instruction text -> short label like "p0"

    def label_for(self, instructions: str) -> str:
        """Give every distinct prompt seen in this run a short label (p0 = starting prompt)."""
        if instructions not in self.prompt_labels:
            self.prompt_labels[instructions] = f"p{len(self.prompt_labels)}"
        return self.prompt_labels[instructions]

    def record(self, phase: str, instructions: str, scored_on: str,
               max_candidates: int, outcome: dict) -> dict:
        row = {
            "run_id": self.run_id,
            "trial": len(self.rows) + 1,
            "phase": phase,                      # "search" (GEPA) or "final" (our own check)
            "prompt": self.label_for(instructions),
            "prompt_hash": hashlib.sha256(instructions.encode()).hexdigest()[:12],
            "scored_on": scored_on,              # which metric the optimizer was given
            "max_candidates": max_candidates,
            "proposed": outcome["proposed"],
            "valid": outcome["valid"],
            "rejected": len(outcome["rejected"]),
            "skipped_by_memory": len(outcome["skipped_by_memory"]),
            "train_metric": outcome["train_passes"],
            "validation_metric": outcome["validation_passes"],
            "candidates": outcome["candidates"],
        }
        self.rows.append(row)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"  trial {row['trial']:>2} [{phase:<6}] prompt {row['prompt']:<3} "
              f"scored on {scored_on:<10} | proposed {row['proposed']}, valid {row['valid']}, "
              f"skipped {row['skipped_by_memory']} | train metric {row['train_metric']} | "
              f"validation metric {row['validation_metric']}")
        return row


# ----------------------------------------------------------------------------
# Prompt versions on disk
# ----------------------------------------------------------------------------
def read_prompt_index(prompts_dir: Path) -> dict:
    """Read prompts/index.json (which version is best, which is previous)."""
    index_file = Path(prompts_dir) / "index.json"
    if not index_file.exists():
        return {"best_version": None, "previous_version": None, "versions": []}
    with open(index_file) as handle:
        return json.load(handle)


def prompt_file(prompts_dir: Path, version: int) -> Path:
    return Path(prompts_dir) / f"proposer_v{version:03d}.json"


def read_prompt(prompts_dir: Path, version: int) -> dict:
    with open(prompt_file(prompts_dir, version)) as handle:
        return json.load(handle)


def save_prompt(prompts_dir: Path, instructions: str, status: str, metrics: dict,
                parent_version, run_id: str) -> int:
    """Save one prompt as the next version number. Returns that number."""
    prompts_dir = Path(prompts_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    index = read_prompt_index(prompts_dir)
    version = (max(index["versions"]) + 1) if index["versions"] else 1
    record = {
        "version": version,
        "status": status,                  # baseline | best | flagged_overfitting
        "parent_version": parent_version,
        "run_id": run_id,
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "proposer_model": PROPOSER_MODEL,
        "reflection_model": REFLECTION_MODEL,
        "train_metric": metrics.get("train"),
        "validation_metric": metrics.get("validation"),
        "instructions": instructions,
    }
    with open(prompt_file(prompts_dir, version), "w") as handle:
        json.dump(record, handle, indent=1)

    index["versions"].append(version)
    if status in ("baseline", "best"):
        # The old best becomes "previous" so the two can be compared.
        index["previous_version"] = index["best_version"]
        index["best_version"] = version
    with open(prompts_dir / "index.json", "w") as handle:
        json.dump(index, handle, indent=1)
    return version


def show_prompts(prompts_dir: Path = PROMPTS_DIR) -> None:
    """Print the best and the previous prompt with their metrics."""
    index = read_prompt_index(prompts_dir)
    if index["best_version"] is None:
        print("No prompts saved yet. Run `python optimizer.py` first.")
        return
    for title, version in (("PREVIOUS", index["previous_version"]), ("BEST", index["best_version"])):
        if version is None:
            print(f"{title}: (none yet)")
            continue
        record = read_prompt(prompts_dir, version)
        print(f"{title}: version {record['version']} ({record['status']}), "
              f"train metric {record['train_metric']}, validation metric {record['validation_metric']}")
        print(f"  file: {prompt_file(prompts_dir, version).name}")
        print(f"  {record['instructions']}")
        print()
    flagged = [v for v in index["versions"]
               if read_prompt(prompts_dir, v)["status"] == "flagged_overfitting"]
    if flagged:
        print(f"Flagged as overfitting (not promoted): versions {flagged}")


# ----------------------------------------------------------------------------
# The DSPy program
# ----------------------------------------------------------------------------
def build_proposer(instructions: str, call_budget: dict = None):
    """Build the DSPy module that proposes candidates, using the given prompt.

    call_budget is an optional {"used": 0, "limit": N} counter shared by every
    copy of the module; once used reaches limit, no further model call is made.
    """
    import dspy
    from pydantic import BaseModel

    class CandidateSpec(BaseModel):
        """One proposed strategy candidate."""
        family: str
        params: dict[str, float]

    class ProposeStrategies(dspy.Signature):
        lessons: str = dspy.InputField(desc="Lessons from memory: candidates the risk gate blocked, and why")
        allowed_families: str = dspy.InputField(desc="The strategy families you may use, with parameter ranges")
        max_candidates: int = dspy.InputField(desc="The most candidates you may return")
        candidates: list[CandidateSpec] = dspy.OutputField(desc="Proposed candidates: family and params for each")

    class StrategyProposer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.propose = dspy.Predict(ProposeStrategies.with_instructions(instructions))

        def forward(self, lessons, allowed_families, max_candidates):
            # Hard stop: never ask the model for more proposals than the ceiling.
            if call_budget is not None:
                if call_budget["used"] >= call_budget["limit"]:
                    raise RuntimeError("proposer call ceiling reached")
                call_budget["used"] += 1
            answer = self.propose(lessons=lessons, allowed_families=allowed_families,
                                  max_candidates=max_candidates)
            # Carry the prompt text along so the metric can log which prompt was used.
            return dspy.Prediction(candidates=answer.candidates,
                                   instructions_used=self.propose.signature.instructions)

    return StrategyProposer()


def current_instructions(program) -> str:
    """Read the prompt text out of a proposer program."""
    return program.propose.signature.instructions


def run_optimization(proposer_lm, reflection_lm, lessons: list, windows: dict,
                     prompts_dir: Path = PROMPTS_DIR, log_file: Path = TRIAL_LOG_FILE,
                     max_trials: int = MAX_TRIALS) -> dict:
    """Run one optimization and save the prompts. Returns a summary dictionary."""
    import dspy

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    log = TrialLog(run_id, log_file)
    lessons_text = describe_lessons(lessons)
    families_text = describe_families()

    # Start from the best saved prompt, or from the baseline on the first run.
    index = read_prompt_index(prompts_dir)
    first_run = index["best_version"] is None
    start_instructions = (BASELINE_INSTRUCTIONS if first_run
                          else read_prompt(prompts_dir, index["best_version"])["instructions"])
    start_version = index["best_version"]

    def make_examples(split: str) -> list:
        """The practice tasks. Same inputs for both splits; only the scoring differs."""
        return [
            dspy.Example(lessons=lessons_text, allowed_families=families_text,
                         max_candidates=size, split=split)
            .with_inputs("lessons", "allowed_families", "max_candidates")
            for size in TASK_SIZES
        ]

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """Score one proposal. Called by GEPA for every trial."""
        outcome = score_proposal(pred.candidates, gold.max_candidates, lessons, windows)
        # GEPA asks again with pred_name set when it only wants feedback text;
        # log each proposal once.
        if pred_name is None:
            log.record("search", pred.instructions_used, gold.split, gold.max_candidates, outcome)
        if gold.split == "train":
            score = outcome["train_passes"] / gold.max_candidates
            feedback = train_feedback(outcome, gold.max_candidates)
        else:
            score = outcome["validation_passes"] / gold.max_candidates
            feedback = validation_feedback(outcome, gold.max_candidates)
        return dspy.Prediction(score=score, feedback=feedback)

    print(f"Run {run_id}")
    print(f"Proposer model  : {PROPOSER_MODEL}")
    print(f"Reflection model: {REFLECTION_MODEL}")
    print(f"Trial ceiling   : {max_trials}")
    print(f"Starting prompt : {'baseline (first run)' if first_run else f'version {start_version}'}")
    print("Test window     : not opened")
    print()
    print("Trials:")

    call_budget = {"used": 0, "limit": max_trials}
    student = build_proposer(start_instructions, call_budget)
    # One request per trial: if a reply cannot be read, the trial scores zero
    # instead of DSPy quietly asking again in a second format.
    one_shot_adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
    with dspy.context(lm=proposer_lm, adapter=one_shot_adapter):
        optimizer = dspy.GEPA(
            metric=metric,
            max_metric_calls=max_trials,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=len(TASK_SIZES),
            use_merge=False,
            track_stats=True,
            num_threads=1,
            seed=0,
        )
        optimized = optimizer.compile(student, trainset=make_examples("train"),
                                      valset=make_examples("validation"))

    # Add up each prompt's two metrics from the trials already logged. No extra
    # model calls are made for this. A prompt counts as fully measured only if
    # it was scored on every practice task.
    first_rows = {}
    for row in log.rows:
        first_rows.setdefault((row["prompt"], row["max_candidates"]), row)
    totals = {}
    for instructions, label in log.prompt_labels.items():
        rows = [first_rows.get((label, size)) for size in TASK_SIZES]
        if all(rows):
            totals[instructions] = {"train": sum(r["train_metric"] for r in rows),
                                    "validation": sum(r["validation_metric"] for r in rows)}
    if start_instructions not in totals:
        raise RuntimeError("The starting prompt was not fully measured; nothing can be compared.")
    not_measured = [label for text, label in log.prompt_labels.items() if text not in totals]

    base = totals[start_instructions]
    results = []
    for instructions, metrics in totals.items():
        judgement = ("starting prompt" if instructions == start_instructions else
                     judge_change(base["train"], base["validation"],
                                  metrics["train"], metrics["validation"]))
        results.append({"prompt": log.label_for(instructions), "instructions": instructions,
                        "train": metrics["train"], "validation": metrics["validation"],
                        "judgement": judgement})

    # Save the starting prompt on the first run so there is always a "previous".
    if first_run:
        start_version = save_prompt(prompts_dir, start_instructions, "baseline", base, None, run_id)

    # Promote the prompt with the highest validation metric, if it beats the start.
    improved = [row for row in results if row["judgement"] == "improved"]
    promoted_version = None
    if improved:
        winner = max(improved, key=lambda row: (row["validation"], row["train"]))
        promoted_version = save_prompt(prompts_dir, winner["instructions"], "best",
                                       {"train": winner["train"], "validation": winner["validation"]},
                                       start_version, run_id)

    # Save (but do not promote) prompts that only got better on train.
    flagged_versions = []
    for row in results:
        if row["judgement"] == "overfitting":
            flagged_versions.append(save_prompt(
                prompts_dir, row["instructions"], "flagged_overfitting",
                {"train": row["train"], "validation": row["validation"]}, start_version, run_id))

    print()
    print(f"{'prompt':<8} {'train metric':>13} {'validation metric':>18}  judgement")
    for row in results:
        flag = "  <-- OVERFITTING: better on train, not on validation" if row["judgement"] == "overfitting" else ""
        print(f"{row['prompt']:<8} {row['train']:>13} {row['validation']:>18}  {row['judgement']}{flag}")
    print()
    if promoted_version:
        print(f"New best prompt saved as version {promoted_version}; previous is version {start_version}.")
    else:
        print(f"No prompt beat version {start_version} on validation. Best prompt is unchanged.")
    if flagged_versions:
        print(f"OVERFITTING FLAG: version(s) {flagged_versions} improved on train but not on "
              f"validation. Saved for inspection, not promoted.")
    if not_measured:
        print(f"Not fully measured (trial ceiling reached first), so not judged: {not_measured}")
    print(f"Trials logged: {len(log.rows)} -> {log.log_file}")
    print(f"Proposer requests made: {call_budget['used']} of at most {call_budget['limit']}")

    return {"run_id": run_id, "results": results, "proposer_requests": call_budget["used"], "start_version": start_version,
            "promoted_version": promoted_version, "flagged_versions": flagged_versions,
            "trials": log.rows}


def plan_run(lessons: list, max_trials: int = MAX_TRIALS) -> dict:
    """Work out the most model calls one run can make, and a rough cost.

    No model is called here. Token counts are estimated as characters / 4.
    """
    tasks = len(TASK_SIZES)
    proposer_calls = max_trials                       # one request per trial, no retries
    # GEPA first scores the starting prompt on validation (one trial per task).
    # After that every rewrite step first re-scores a prompt on train (one trial
    # per task), so the number of rewrite steps is limited by the trials left.
    reflection_calls = max(0, (max_trials - tasks) // tasks)

    prompt_tokens = (len(describe_lessons(lessons)) + len(describe_families())
                     + len(BASELINE_INSTRUCTIONS)) // 4 + 400      # + DSPy's own framing
    # A rewrite request holds the prompt plus every task's inputs and feedback.
    reflection_input_tokens = prompt_tokens * tasks + 1500

    def cost(model, calls, input_tokens, output_tokens):
        prices = PRICES_PER_MILLION_TOKENS.get(model)
        if prices is None:
            return None
        return calls * (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000

    # "Likely": moderate replies. "Ceiling": every reply uses the full MAX_TOKENS.
    likely = [cost(PROPOSER_MODEL, proposer_calls, prompt_tokens, 3000),
              cost(REFLECTION_MODEL, reflection_calls, reflection_input_tokens, 4000)]
    ceiling = [cost(PROPOSER_MODEL, proposer_calls, prompt_tokens, MAX_TOKENS),
               cost(REFLECTION_MODEL, reflection_calls, reflection_input_tokens, MAX_TOKENS)]
    return {
        "max_trials": max_trials,
        "proposer_calls": proposer_calls,
        "reflection_calls": reflection_calls,
        "total_calls": proposer_calls + reflection_calls,
        "proposer_input_tokens": prompt_tokens,
        "reflection_input_tokens": reflection_input_tokens,
        "likely_cost": None if None in likely else sum(likely),
        "ceiling_cost": None if None in ceiling else sum(ceiling),
    }


def print_plan(plan: dict) -> None:
    """Print the call ceiling and the rough cost."""
    print("Plan for one optimization run (nothing has been called yet):")
    print(f"  Trial ceiling                : {plan['max_trials']}")
    print(f"  Proposer calls, at most      : {plan['proposer_calls']}  ({PROPOSER_MODEL})")
    print(f"  Reflection calls, at most    : {plan['reflection_calls']}  ({REFLECTION_MODEL})")
    print(f"  TOTAL model calls, at most   : {plan['total_calls']}")
    print("  Retries and second-format fallbacks are switched off, so a failed or")
    print("  unreadable reply costs one call and scores zero; it is not asked again.")
    print(f"  Estimated input per call     : about {plan['proposer_input_tokens']:,} tokens (proposer), "
          f"{plan['reflection_input_tokens']:,} (reflection)")
    if plan["likely_cost"] is None:
        print("  Cost estimate                : not available (no price on file for these models)")
    else:
        print(f"  Rough cost, likely           : about ${plan['likely_cost']:.2f}")
        print(f"  Rough cost, absolute ceiling : about ${plan['ceiling_cost']:.2f} "
              f"(every reply at the {MAX_TOKENS:,}-token limit)")
        print("  Prices used are list prices cached on 2026-11-03; treat the figures as rough.")


def missing_key_variable(model_name: str):
    """Return the name of the key variable a model needs if it is not set, else None."""
    provider = model_name.split("/", 1)[0]
    variable = PROVIDER_KEY_VARIABLES.get(provider)
    if variable and not os.environ.get(variable):
        return variable
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize the strategy-proposal prompt with DSPy.")
    parser.add_argument("--show", action="store_true", help="print the best and previous prompts")
    parser.add_argument("--plan", action="store_true",
                        help="print the maximum model calls and a rough cost, then stop")
    args = parser.parse_args()

    if args.plan:
        lessons = lesson_memory.read_lessons(lesson_memory.open_memory())
        print_plan(plan_run(lessons))
        return 0

    if args.show:
        show_prompts()
        return 0

    # Stop early, with a clear message, if a key is missing. Nothing is spent.
    missing = sorted({name for name in (missing_key_variable(PROPOSER_MODEL),
                                        missing_key_variable(REFLECTION_MODEL)) if name})
    if missing:
        print("Cannot start: the model key is not set in the environment.")
        print(f"  Missing variable(s): {', '.join(missing)}")
        print(f"  Proposer model  (TT_PROPOSER_MODEL)  : {PROPOSER_MODEL}")
        print(f"  Reflection model (TT_REFLECTION_MODEL): {REFLECTION_MODEL}")
        print("Set the variable in your shell or environment settings. Never put a key in code.")
        return 2

    import dspy
    # num_retries=0: a failed request is not sent again, so the call ceiling holds.
    proposer_lm = dspy.LM(PROPOSER_MODEL, max_tokens=MAX_TOKENS, num_retries=0)
    reflection_lm = dspy.LM(REFLECTION_MODEL, max_tokens=MAX_TOKENS, num_retries=0)

    # Read memory before anything is proposed.
    store = lesson_memory.open_memory()
    lessons = lesson_memory.read_lessons(store)
    lesson_memory.print_memory(lessons, title="Memory read before proposing")
    print()
    print_plan(plan_run(lessons))
    print()

    run_optimization(proposer_lm, reflection_lm, lessons, load_windows())
    print()
    show_prompts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
