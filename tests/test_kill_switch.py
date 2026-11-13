"""Tests for kill_switch.py. Every test runs against a throwaway state folder."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import kill_switch

REPO = Path(kill_switch.__file__).resolve().parent


def test_missing_state_file_means_engaged():
    assert not kill_switch.STATE_FILE.exists()
    assert kill_switch.is_engaged() is True


def test_broken_state_file_means_engaged():
    kill_switch.STATE_DIR.mkdir(parents=True)
    kill_switch.STATE_FILE.write_text("{ this is not valid json")
    assert kill_switch.is_engaged() is True
    kill_switch.STATE_FILE.write_text(json.dumps({"engaged": "no"}))   # not a true/false value
    assert kill_switch.is_engaged() is True


def test_engage_saves_state_and_updates_the_page_file(owner_released):
    assert kill_switch.is_engaged() is False
    kill_switch.engage("daily review", by="test")

    assert kill_switch.is_engaged() is True
    state = kill_switch.read_state()
    assert state["reason"] == "daily review" and state["changed_by"] == "test"
    page_file = kill_switch.VIEW_STATE_FILE.read_text()
    assert "window.KILL_SWITCH" in page_file and '"engaged": true' in page_file


def test_engage_needs_a_reason(owner_released):
    with pytest.raises(ValueError):
        kill_switch.engage("   ")


def test_code_cannot_release_the_switch():
    kill_switch.engage("stop", by="test")
    with pytest.raises(PermissionError):
        kill_switch._release_from_command_line("trying to release from code")
    assert kill_switch.is_engaged() is True


def test_there_is_no_public_release_function():
    public = [name for name in dir(kill_switch) if not name.startswith("_")]
    assert not [name for name in public if "release" in name.lower() or "disengage" in name.lower()]


def run_cli(tmp_path, *arguments):
    """Run kill_switch.py as a real program against a throwaway folder."""
    environment = dict(os.environ, TT_STATE_DIR=str(tmp_path / "cli_state"), TT_VIEW_DIR=str(tmp_path / "cli_view"))
    result = subprocess.run([sys.executable, str(REPO / "kill_switch.py"), *arguments],
                            capture_output=True, text=True, env=environment)
    state_file = tmp_path / "cli_state" / "kill_switch.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else None
    return result, state


def test_command_line_release_needs_confirm_and_then_works(tmp_path):
    result, state = run_cli(tmp_path, "release")
    assert result.returncode == 2 and state is None            # nothing changed without --confirm

    result, state = run_cli(tmp_path, "release", "--confirm")
    assert result.returncode == 0 and state["engaged"] is False
    assert "RELEASED" in result.stdout

    result, state = run_cli(tmp_path, "engage", "end of day")
    assert state["engaged"] is True and state["reason"] == "end of day"
    assert len(state["history"]) == 2                          # release, then engage


def test_real_strategy_code_passes_the_isolation_check():
    assert kill_switch.check_strategy_isolation() == []


@pytest.mark.parametrize("source, expected", [
    ("import kill_switch\n", "imports kill_switch"),
    ("from kill_switch import engage\n", "imports from kill_switch"),
    ("import paper\n", "imports paper"),
    ("import importlib\n", "imports importlib"),
    ("import sys\nx = sys.modules\n", "loaded modules"),
    ("x = __import__('os')\n", "uses __import__"),
    ("exec('pass')\n", "uses exec"),
    ("path = 'state/kill_switch.json'\n", "mentions 'kill_switch'"),
])
def test_isolation_check_catches_strategy_code_reaching_the_switch(tmp_path, source, expected):
    strategy_file = tmp_path / "bad_strategy.py"
    strategy_file.write_text(source)
    problems = kill_switch.check_strategy_isolation([strategy_file])
    assert problems and any(expected in problem for problem in problems)


def test_engine_does_not_import_the_switch_or_the_broker():
    source = (REPO / "engine.py").read_text()
    assert "kill_switch" not in source
    assert "import paper" not in source
