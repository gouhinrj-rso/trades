"""Tests for gate.py. All inputs are made-up numbers; nothing reads data/."""

import gate


def check(train=0.010, validation=0.008, drawdown=0.005, trades=5):
    """Run the gate with good default numbers; each test overrides one or more."""
    return gate.check_candidate("demo", train, validation, drawdown, trades)


def test_thresholds_are_the_owners_numbers():
    assert gate.MIN_TRAIN_RETURN_PCT == 0.0
    assert gate.MIN_VALIDATION_RETURN_PCT == 0.0
    assert gate.MAX_RETURN_GAP_PCT_POINTS == 2.0
    assert gate.MAX_VALIDATION_DRAWDOWN_PCT == 1.5
    assert gate.MIN_VALIDATION_TRADES == 2


def test_good_candidate_passes():
    verdict = check()
    assert verdict["verdict"] == "PASS"
    assert verdict["failures"] == []


def test_rule_1_train_return_must_be_positive():
    verdict = check(train=-0.003)
    assert verdict["verdict"] == "BLOCK"
    assert verdict["failures"] == ["train return -0.30% <= 0"]


def test_rule_1_validation_return_must_be_positive():
    verdict = check(validation=-0.003)
    assert verdict["verdict"] == "BLOCK"
    assert verdict["failures"] == ["validation return -0.30% <= 0"]


def test_rule_1_zero_return_is_blocked():
    # "Greater than zero" means exactly zero does not pass.
    verdict = check(validation=0.0)
    assert "validation return +0.00% <= 0" in verdict["failures"]


def test_rule_2_gap_too_large():
    verdict = check(train=0.035, validation=0.010)   # 2.5 points apart
    assert verdict["verdict"] == "BLOCK"
    assert len(verdict["failures"]) == 1
    assert verdict["failures"][0].startswith("train/validation gap 2.50 pts > 2 pts")


def test_rule_2_gap_counts_in_both_directions():
    verdict = check(train=0.010, validation=0.035)   # validation better by 2.5 points
    assert verdict["failures"][0].startswith("train/validation gap 2.50 pts > 2 pts")


def test_rule_3_drawdown_too_large():
    verdict = check(drawdown=0.016)
    assert verdict["failures"] == ["validation max drawdown 1.60% > 1.5%"]


def test_rule_3_drawdown_at_the_limit_passes():
    assert check(drawdown=0.015)["verdict"] == "PASS"


def test_rule_4_too_few_trades():
    verdict = check(trades=1)
    assert verdict["failures"] == ["validation trade count 1 < 2"]


def test_rule_4_two_trades_is_enough():
    assert check(trades=2)["verdict"] == "PASS"


def test_every_failed_rule_is_listed_not_just_the_first():
    # Breaks all five checks at once.
    verdict = check(train=-0.020, validation=-0.050, drawdown=0.030, trades=0)
    assert verdict["verdict"] == "BLOCK"
    assert len(verdict["failures"]) == 5
    joined = " | ".join(verdict["failures"])
    assert "train return" in joined
    assert "validation return" in joined
    assert "gap" in joined
    assert "drawdown" in joined
    assert "trade count" in joined


def test_printed_output_has_block_lines_and_summary(capsys):
    verdicts = [check(), check(validation=-0.003, trades=1)]
    gate.print_verdicts(verdicts)
    printed = capsys.readouterr().out

    assert "PASS  demo:" in printed
    assert "BLOCK demo: validation return -0.30% <= 0" in printed
    assert "BLOCK demo: validation trade count 1 < 2" in printed
    assert "Summary: candidates 2, passed 1, blocked 1" in printed
