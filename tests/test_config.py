"""Tests for the files in config/.

config/risk_limits.yaml is a copy of numbers the owner set in gate.py and
paper.py. These tests fail if the copy and the code ever disagree, so the two
cannot drift apart without someone noticing.
"""

from pathlib import Path

import yaml

import engine
import gate
import paper

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load(name: str) -> dict:
    """Read one YAML file from config/ into a dictionary."""
    with open(CONFIG_DIR / name) as handle:
        return yaml.safe_load(handle)


def test_gate_limits_match_gate_py():
    limits = load("risk_limits.yaml")["gate"]
    assert limits["min_train_return_pct"] == gate.MIN_TRAIN_RETURN_PCT
    assert limits["min_validation_return_pct"] == gate.MIN_VALIDATION_RETURN_PCT
    assert limits["max_return_gap_pct_points"] == gate.MAX_RETURN_GAP_PCT_POINTS
    assert limits["max_validation_drawdown_pct"] == gate.MAX_VALIDATION_DRAWDOWN_PCT
    assert limits["min_validation_trades"] == gate.MIN_VALIDATION_TRADES


def test_broker_limits_match_paper_py():
    limits = load("risk_limits.yaml")["paper_broker"]
    assert limits["stop_pct"] == paper.STOP_PCT
    assert limits["daily_loss_cap_pct"] == paper.DAILY_LOSS_CAP_PCT
    assert tuple(limits["allowed_order_types"]) == paper.ALLOWED_ORDER_TYPES


def test_unset_limits_are_marked_tbd_not_guessed():
    """Limits the owner has not chosen must say TBD, never a made-up number."""
    for name, value in load("risk_limits.yaml")["position_sizing"].items():
        assert value == "TBD", f"{name} has a value; only the owner sets limits"


def test_paper_yaml_is_paper_mode_and_matches_code():
    settings = load("paper.yaml")
    assert settings["mode"] == "paper"
    assert settings["account"]["starting_cash"] == paper.STARTING_CASH
    assert settings["costs"]["fee_bps_per_side"] == engine.FEE_BPS_PER_SIDE
    assert settings["orders"]["order_lifetime_bars"] == paper.ORDER_LIFETIME_BARS
