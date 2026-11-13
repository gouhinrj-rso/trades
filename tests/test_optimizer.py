"""Tests for optimizer.py.

No real model is called. The full-pipeline test uses a scripted stand-in model
(it returns fixed answers), so it proves the plumbing works - logging, prompt
versions, the overfitting flag - and says nothing about real model quality.
"""

import json

import dspy
import pytest
from dspy.clients.engines.dummy_engine import AsyncDummyEngine, DummyEngine
from dspy.lm15 import Message, Response, TextPart, Usage
from dspy.utils import DummyLM

import engine
import optimizer


# ----------------------------------------------------------------------------
# Small pieces
# ----------------------------------------------------------------------------
def test_clean_proposals_accepts_valid_and_names_them_like_the_engine():
    raw = [{"family": "zscore_reversion", "params": {"window": 40.0, "entry": 1.5}},
           {"family": "ema_crossover", "params": {"fast": 5, "slow": 30}}]
    valid, rejected = optimizer.clean_proposals(raw, max_candidates=8)

    assert rejected == []
    assert [c["name"] for c in valid] == ["zscore(window=40, entry=1.5)", "ema_cross(fast=5, slow=30)"]
    # Same names as the engine's own grid, so memory lessons match.
    grid_names = {c["name"] for c in engine.build_candidates()}
    assert {c["name"] for c in valid} <= grid_names


def test_clean_proposals_rejects_bad_input_with_reasons():
    raw = [
        {"family": "martingale", "params": {"x": 1}},                         # unknown family
        {"family": "ema_crossover", "params": {"fast": 30, "slow": 10}},      # fast >= slow
        {"family": "zscore_reversion", "params": {"window": 5, "entry": 1}},  # window too small
        {"family": "zscore_reversion", "params": {"window": 40}},             # missing param
        {"family": "zscore_reversion", "params": {"window": 40, "entry": 1.5}},
        {"family": "zscore_reversion", "params": {"window": 40, "entry": 1.5}},  # duplicate
        {"family": "zscore_reversion", "params": {"window": 41, "entry": 1.5}},  # over the limit
    ]
    valid, rejected = optimizer.clean_proposals(raw, max_candidates=6)

    assert [c["name"] for c in valid] == ["zscore(window=40, entry=1.5)"]
    reasons = " | ".join(entry["reason"] for entry in rejected)
    assert "not allowed" in reasons
    assert "must be smaller than slow" in reasons
    assert "outside 10 to 150" in reasons
    assert "params must be exactly" in reasons
    assert "duplicate" in reasons
    assert "over the limit" in reasons


def test_judge_change_flags_overfitting():
    # Train went up, validation did not: overfitting.
    assert optimizer.judge_change(2, 3, new_train=5, new_validation=3) == "overfitting"
    assert optimizer.judge_change(2, 3, new_train=5, new_validation=1) == "overfitting"
    # Validation went up: a real improvement, whatever train did.
    assert optimizer.judge_change(2, 3, new_train=2, new_validation=4) == "improved"
    assert optimizer.judge_change(2, 3, new_train=5, new_validation=4) == "improved"
    # Nothing went up.
    assert optimizer.judge_change(2, 3, new_train=2, new_validation=3) == "no_gain"


def test_windows_use_train_and_validation_only():
    windows = optimizer.load_windows()
    report = json.load(open(engine.DATA_DIR / "split_report.json"))
    test_start = report["windows"][2]["first_timestamp"]

    # The inner split is cut from train and nothing else.
    assert windows["train_early"] + windows["train_late"] == windows["train"]
    # Every bar the optimizer can see is earlier than the first test bar.
    for name in ("train", "validation"):
        assert windows[name][-1]["timestamp"] < test_start


def test_validation_feedback_is_a_count_only():
    outcome = {"validation_passes": 3, "train_passes": 1, "candidates": [
        {"name": "zscore(window=40, entry=1.5)", "train_verdict": "PASS", "train_failures": [],
         "validation_verdict": "PASS"}], "skipped_by_memory": [], "rejected": [], "proposed": 1}
    text = optimizer.validation_feedback(outcome, 8)
    assert text == "3 of 8 possible candidates passed the gate on validation."
    assert "zscore" not in text   # no per-candidate validation detail leaks into the prompt


def test_missing_key_is_reported_by_name(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert optimizer.missing_key_variable("anthropic/claude-sonnet-5-5") == "ANTHROPIC_API_KEY"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-for-test")
    assert optimizer.missing_key_variable("anthropic/claude-sonnet-5-5") is None


def test_no_key_appears_in_the_source():
    source = open(optimizer.__file__).read()
    assert "sk-ant-" not in source and "sk-proj-" not in source


# ----------------------------------------------------------------------------
# Full pipeline with a scripted stand-in model
# ----------------------------------------------------------------------------
class ScriptedEngine(DummyEngine):
    """Answers each call by asking a plain Python function what to say."""

    def _complete_messages(self, messages):
        reply = self.owner.respond(messages)
        text = self.owner._format_answer_fields(reply) if isinstance(reply, dict) else reply
        return Response(id=None, model="scripted", message=Message.assistant([TextPart(text)]),
                        finish_reason="stop", usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))


class ScriptedLM(DummyLM):
    """A stand-in model whose replies come from `respond(messages)`."""

    def __init__(self, respond):
        super().__init__([])
        self.respond = respond
        scripted = ScriptedEngine(self)
        self._engine_spec = scripted
        self._async_engine_spec = AsyncDummyEngine(scripted)


NEW_PROMPT_MARKER = "REVISED-PROMPT"

# Candidates the scripted proposer returns. The z-score ones are in the engine's grid.
WEAK_ANSWER = [{"family": "ema_crossover", "params": {"fast": 5, "slow": 30}},
               {"family": "zscore_reversion", "params": {"window": 40, "entry": 1.5}}]
STRONG_ANSWER = [{"family": "zscore_reversion", "params": {"window": 40, "entry": 1.5}},
                 {"family": "zscore_reversion", "params": {"window": 40, "entry": 2.0}},
                 {"family": "zscore_reversion", "params": {"window": 20, "entry": 1.5}}]


def scripted_proposer(messages):
    """Weak answer under the starting prompt, strong answer under the revised one."""
    whole_prompt = " ".join(message["content"] for message in messages)
    return {"candidates": STRONG_ANSWER if NEW_PROMPT_MARKER in whole_prompt else WEAK_ANSWER}


def scripted_reflection(messages):
    """Pretend to rewrite the prompt."""
    return {"new_instruction": f"{NEW_PROMPT_MARKER}: propose only z-score candidates that "
                               f"memory has not ruled out."}


def test_full_pipeline_logs_both_metrics_and_versions_prompts(tmp_path):
    prompts_dir = tmp_path / "prompts"
    log_file = tmp_path / "trials.jsonl"

    summary = optimizer.run_optimization(
        ScriptedLM(scripted_proposer), ScriptedLM(scripted_reflection),
        lessons=[], windows=optimizer.load_windows(),
        prompts_dir=prompts_dir, log_file=log_file, max_trials=12,
    )

    # Every trial logs the train metric and the validation metric separately.
    rows = [json.loads(line) for line in open(log_file)]
    assert rows and len(rows) == len(summary["trials"])
    for row in rows:
        assert isinstance(row["train_metric"], int)
        assert isinstance(row["validation_metric"], int)
        assert row["phase"] == "search"
    # The trial ceiling is a hard limit on logged trials and on proposer requests.
    assert len(rows) <= 12
    assert summary["proposer_requests"] <= 12
    assert {row["scored_on"] for row in rows if row["phase"] == "search"} == {"train", "validation"}

    # The revised prompt was tried and beat the starting prompt on validation.
    by_label = {row["prompt"]: row for row in summary["results"]}
    assert by_label["p0"]["judgement"] == "starting prompt"
    revised = [row for row in summary["results"] if NEW_PROMPT_MARKER in row["instructions"]]
    assert revised and revised[0]["validation"] > by_label["p0"]["validation"]
    assert revised[0]["judgement"] == "improved"

    # prompts/ holds the previous prompt (v1) and the best prompt (v2).
    index = optimizer.read_prompt_index(prompts_dir)
    assert index["previous_version"] == 1 and index["best_version"] == 2
    previous = optimizer.read_prompt(prompts_dir, 1)
    best = optimizer.read_prompt(prompts_dir, 2)
    assert previous["status"] == "baseline"
    assert previous["instructions"] == optimizer.BASELINE_INSTRUCTIONS
    assert best["status"] == "best" and NEW_PROMPT_MARKER in best["instructions"]
    assert best["validation_metric"] > previous["validation_metric"]
    assert best["parent_version"] == 1


def test_prompt_that_only_improves_on_train_is_flagged_not_promoted(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"

    # Make the scoring say: revised prompt is better on train, no better on validation.
    def fake_score(raw_candidates, max_candidates, lessons, windows):
        strong = len(raw_candidates) == len(STRONG_ANSWER)
        return {"proposed": len(raw_candidates), "valid": len(raw_candidates), "rejected": [],
                "skipped_by_memory": [], "candidates": [],
                "train_passes": 3 if strong else 1, "validation_passes": 1}

    monkeypatch.setattr(optimizer, "score_proposal", fake_score)
    summary = optimizer.run_optimization(
        ScriptedLM(scripted_proposer), ScriptedLM(scripted_reflection),
        lessons=[], windows=optimizer.load_windows(),
        prompts_dir=prompts_dir, log_file=tmp_path / "trials.jsonl", max_trials=12,
    )

    flagged = [row for row in summary["results"] if row["judgement"] == "overfitting"]
    assert flagged, "the train-only improvement must be flagged"
    assert summary["promoted_version"] is None
    index = optimizer.read_prompt_index(prompts_dir)
    assert index["best_version"] == 1                       # baseline stays the best
    assert summary["flagged_versions"]
    saved = optimizer.read_prompt(prompts_dir, summary["flagged_versions"][0])
    assert saved["status"] == "flagged_overfitting"


def test_plan_stays_within_twenty_trials():
    plan = optimizer.plan_run(lessons=[], max_trials=20)
    assert plan["proposer_calls"] == 20
    assert plan["reflection_calls"] == 9
    assert plan["total_calls"] == 29
    assert plan["likely_cost"] < plan["ceiling_cost"]


def test_default_trial_ceiling_is_twenty():
    assert optimizer.MAX_TRIALS == 20
