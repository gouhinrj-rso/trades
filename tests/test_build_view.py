"""Tests for build_view.py: what reaches the page, and that the test window never does."""

import builtins
import json

import build_view
import engine


def read_data_file(path):
    text = path.read_text()
    start = text.index("window.TERMINAL_DATA = ") + len("window.TERMINAL_DATA = ")
    return json.loads(text[start:text.rindex(";")])


def build_in(tmp_path, monkeypatch):
    monkeypatch.setattr(build_view, "VIEW_DIR", tmp_path / "view")
    monkeypatch.setattr(build_view, "DATA_FILE", tmp_path / "view" / "data.js")
    # Keep the test fast and offline: do not open the real mem0 store here.
    monkeypatch.setattr(build_view, "collect_lessons", lambda: {
        "available": True, "total": 1,
        "lessons": [{"round": 1, "candidate": "x", "family": "f", "failed_checks": "c", "text": "t"}]})
    build_view.build()
    return read_data_file(tmp_path / "view" / "data.js")


def test_data_file_has_every_panel(tmp_path, monkeypatch):
    data = build_in(tmp_path, monkeypatch)

    assert len(data["bars"]) == 576                       # train 432 + validation 144
    assert len(data["candidates"]) == 18
    row = data["candidates"][0]
    assert set(row) >= {"name", "train_return_pct", "validation_return_pct",
                        "validation_drawdown_pct", "validation_trades", "verdict", "failures"}
    blocked = [r for r in data["candidates"] if r["verdict"] == "BLOCK"]
    assert len(blocked) == 13 and all(r["failures"] for r in blocked)
    assert data["memory"]["lessons"]
    assert data["paper"]["settings"]["order_types"] == ["limit"]
    assert data["paper"]["settings"]["stop_pct"] == 1.0
    assert data["paper"]["settings"]["daily_loss_cap_pct"] == 2.0


def test_test_window_is_marked_locked_and_carries_no_prices(tmp_path, monkeypatch):
    data = build_in(tmp_path, monkeypatch)

    windows = {w["name"]: w for w in data["windows"]}
    assert set(windows) == {"train", "validation", "test"}
    assert windows["test"]["locked"] is True
    assert windows["train"]["locked"] is False and windows["validation"]["locked"] is False
    # No bar on the page is as late as the first test bar.
    assert max(bar[0] for bar in data["bars"]) < windows["test"]["first_timestamp"]


def test_building_the_view_never_opens_the_locked_files(tmp_path, monkeypatch):
    opened = []
    real_open = builtins.open

    def recording_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    build_in(tmp_path, monkeypatch)

    assert any(path.endswith(engine.TRAIN_FILE.name) for path in opened)
    assert not [path for path in opened if "LOCKED" in path or path.endswith("bars_5m.json")]
