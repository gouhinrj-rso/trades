"""Shared test setup: keep every test away from the real state/ and view/ folders."""

import json

import pytest

import kill_switch
import paper


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point the kill switch, paper state and view files at a throwaway folder."""
    state_dir = tmp_path / "state"
    view_dir = tmp_path / "view"
    monkeypatch.setattr(kill_switch, "STATE_DIR", state_dir)
    monkeypatch.setattr(kill_switch, "STATE_FILE", state_dir / "kill_switch.json")
    monkeypatch.setattr(kill_switch, "VIEW_DIR", view_dir)
    monkeypatch.setattr(kill_switch, "VIEW_STATE_FILE", view_dir / "kill_switch_state.js")
    monkeypatch.setattr(paper, "STATE_DIR", state_dir)
    monkeypatch.setattr(paper, "PAPER_STATE_FILE", state_dir / "paper_state.json")
    return tmp_path


@pytest.fixture
def owner_released(isolated_state):
    """Stand in for the owner releasing the switch (tests cannot use the command line path)."""
    kill_switch.STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(kill_switch.STATE_FILE, "w") as handle:
        json.dump({"engaged": False, "reason": "released for a test", "changed_at": None,
                   "changed_by": "test", "history": []}, handle)
