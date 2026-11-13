"""Tests for memory.py.

Most tests check the lesson-building and skip rules with made-up verdicts and
need no mem0 at all. The last test saves and reads real lessons through mem0 in
a temporary folder; it is skipped when the embedding model is not downloaded.
"""

import pytest

import gate
import memory


def make_verdict(name, family, params, train=0.010, validation=0.008, drawdown=0.005, trades=5):
    """Build a gate verdict the same way run_gate does."""
    verdict = gate.check_candidate(name, train, validation, drawdown, trades)
    verdict["family"] = family
    verdict["params"] = params
    return verdict


def lesson_from(verdict, round_number, family_proposed, family_blocked):
    """Build the dictionary read_lessons() would return for a saved verdict."""
    metadata = memory.build_lesson_metadata(verdict, round_number, family_proposed, family_blocked)
    return {
        "text": memory.build_lesson_text(verdict, round_number),
        "round": metadata["round"],
        "candidate": metadata["candidate"],
        "family": metadata["family"],
        "failed_checks": metadata["failed_checks"],
        "failure_codes": metadata["failure_codes"].split(","),
        "family_proposed_in_round": family_proposed,
        "family_blocked_in_round": family_blocked,
    }


BAD_EMA = make_verdict("ema_cross(fast=5, slow=30)", "ema_crossover", {"fast": 5, "slow": 30},
                       train=-0.0041, validation=-0.0204, drawdown=0.0208, trades=6)


def test_lesson_text_has_name_family_checks_numbers_and_round():
    text = memory.build_lesson_text(BAD_EMA, round_number=3)

    assert "Round 3" in text
    assert "ema_cross(fast=5, slow=30)" in text
    assert "family ema_crossover" in text
    # The exact failed checks, as the gate printed them.
    assert "train return -0.41% <= 0" in text
    assert "validation return -2.04% <= 0" in text
    assert "validation max drawdown 2.08% > 1.5%" in text
    # The numbers behind them.
    assert "validation trades 6" in text
    # Never a bare "failed".
    assert text.strip() != "strategy failed"


def test_lesson_metadata_keeps_the_numbers_as_fields():
    metadata = memory.build_lesson_metadata(BAD_EMA, 3, family_proposed=9, family_blocked=9)

    assert metadata["round"] == 3
    assert metadata["candidate"] == "ema_cross(fast=5, slow=30)"
    assert metadata["family"] == "ema_crossover"
    assert metadata["failure_codes"] == "train_return,validation_return,validation_drawdown"
    assert metadata["train_return_pct"] == pytest.approx(-0.41)
    assert metadata["validation_return_pct"] == pytest.approx(-2.04)
    assert metadata["validation_drawdown_pct"] == pytest.approx(2.08)
    assert metadata["validation_trades"] == 6


def test_exact_parameter_set_that_failed_is_not_proposed_again():
    blocked = make_verdict("zscore(window=80, entry=2.0)", "zscore_reversion",
                           {"window": 80, "entry": 2.0}, trades=1)
    # 9 proposed, only 1 blocked: the family as a whole did not fail.
    lessons = [lesson_from(blocked, 1, family_proposed=9, family_blocked=1)]
    candidates = [
        {"name": "zscore(window=80, entry=2.0)", "family": "zscore_reversion", "params": {}},
        {"name": "zscore(window=40, entry=1.5)", "family": "zscore_reversion", "params": {}},
    ]

    allowed, skipped = memory.filter_proposals(candidates, lessons)

    assert [c["name"] for c in allowed] == ["zscore(window=40, entry=1.5)"]
    assert len(skipped) == 1
    assert "validation trade count 1 < 2" in skipped[0]["reason"]


def test_family_that_failed_the_same_way_is_not_proposed_again():
    first = make_verdict("ema_a", "ema_crossover", {"fast": 5, "slow": 30}, train=-0.01)
    second = make_verdict("ema_b", "ema_crossover", {"fast": 10, "slow": 60}, train=-0.02, trades=0)
    lessons = [lesson_from(first, 1, 2, 2), lesson_from(second, 1, 2, 2)]

    # Both failed on train_return, so the whole family is ruled out,
    # including parameters that were never tried.
    new_idea = {"name": "ema_c", "family": "ema_crossover", "params": {"fast": 3, "slow": 15}}
    allowed, skipped = memory.filter_proposals([new_idea], lessons)

    assert allowed == []
    assert "family ema_crossover failed as a whole in round 1" in skipped[0]["reason"]
    assert "train_return" in skipped[0]["reason"]


def test_family_is_not_ruled_out_when_failures_have_nothing_in_common():
    first = make_verdict("ema_a", "ema_crossover", {"fast": 5, "slow": 30}, train=-0.01)
    second = make_verdict("ema_b", "ema_crossover", {"fast": 10, "slow": 60}, trades=0)
    lessons = [lesson_from(first, 1, 2, 2), lesson_from(second, 1, 2, 2)]

    assert memory.failed_families(lessons) == {}


def test_family_is_not_ruled_out_when_one_candidate_passed():
    first = make_verdict("ema_a", "ema_crossover", {"fast": 5, "slow": 30}, train=-0.01)
    # 3 proposed, 1 blocked: two passed.
    lessons = [lesson_from(first, 1, family_proposed=3, family_blocked=1)]

    assert memory.failed_families(lessons) == {}


def test_empty_memory_skips_nothing():
    candidates = [{"name": "anything", "family": "ema_crossover", "params": {}}]
    allowed, skipped = memory.filter_proposals(candidates, [])
    assert allowed == candidates
    assert skipped == []


def test_config_uses_no_api_key():
    config = memory.build_mem0_config()
    flat = str(config).lower()
    assert "api_key" not in flat
    assert config["embedder"]["provider"] == "fastembed"


@pytest.mark.skipif(
    not any(memory.EMBED_CACHE_DIR.glob("**/*.onnx")),
    reason="embedding model not downloaded; run `python run_round.py` once first",
)
def test_round_trip_through_real_mem0(tmp_path, monkeypatch):
    # Use a throwaway store so the real memory_store/ is untouched.
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path / "store")
    store = memory.open_memory()

    passing = make_verdict("zscore_ok", "zscore_reversion", {"window": 40, "entry": 1.5})
    saved = memory.save_lessons(store, [BAD_EMA, passing], round_number=1)
    assert len(saved) == 1                     # only the blocked one is saved

    # Saving the same round again must not create a duplicate.
    assert memory.save_lessons(store, [BAD_EMA, passing], round_number=1) == []

    lessons = memory.read_lessons(store)
    assert len(lessons) == 1
    assert lessons[0]["candidate"] == "ema_cross(fast=5, slow=30)"
    assert lessons[0]["round"] == 1
    assert "validation max drawdown 2.08% > 1.5%" in lessons[0]["text"]
    assert lessons[0]["failure_codes"] == ["train_return", "validation_return", "validation_drawdown"]

    similar = memory.find_similar_lessons(store, "EMA crossover lost money on the train window")
    assert similar and "ema_cross(fast=5, slow=30)" in similar[0]["text"]
