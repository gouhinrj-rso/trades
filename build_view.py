"""Build the data file the static terminal page reads.

Run from the repo root after any pipeline step:

    python build_view.py

It writes view/data.js. Then open view/index.html in a browser - no server is
needed. The page is plain HTML and JavaScript and loads data.js with a normal
<script> tag, which works straight from disk.

What goes into data.js:
- price bars for the TRAIN and VALIDATION windows only
- the dates of all three windows (the test window's dates, never its prices)
- every candidate's backtest numbers and gate verdict
- the lessons stored in mem0
- the paper account state and paper trading rules

The locked test file is never opened here: bars come through engine.load_bars(),
which refuses it. The test window appears on the chart as a locked, empty band.
"""

import json
import sys
from pathlib import Path

import engine
import gate
import kill_switch
import paper

PROJECT_DIR = Path(__file__).resolve().parent
VIEW_DIR = kill_switch.VIEW_DIR
DATA_FILE = VIEW_DIR / "data.js"
SPLIT_REPORT_FILE = engine.DATA_DIR / "split_report.json"
ROUNDS_FILE = engine.DATA_DIR / "rounds.json"

MEMORY_FEED_LIMIT = 50


def read_json(file_path: Path, default):
    """Read a JSON file, or return `default` if it does not exist."""
    if not Path(file_path).exists():
        return default
    with open(file_path) as handle:
        return json.load(handle)


def collect_bars() -> list:
    """Train and validation bars as compact rows: [time, open, high, low, close]."""
    rows = []
    for file_path in (engine.TRAIN_FILE, engine.VALIDATION_FILE):
        for bar in engine.load_bars(file_path):
            rows.append([bar["timestamp"], bar["open"], bar["high"], bar["low"], bar["close"]])
    return rows


def collect_windows(report: dict) -> list:
    """Window names, sizes and dates from the split report (no prices in it)."""
    windows = []
    for window in report.get("windows", []):
        locked = "LOCKED" in window["name"]
        windows.append({
            "name": "test" if locked else window["name"],
            "locked": locked,
            "bars": window["bars"],
            "first_timestamp": window["first_timestamp"],
            "last_timestamp": window["last_timestamp"],
        })
    return windows


def collect_candidates() -> list:
    """Backtest every candidate on train and validation and gate it."""
    rows = []
    for verdict in gate.run_gate():
        numbers = verdict["numbers"]
        rows.append({
            "name": verdict["name"],
            "family": verdict["family"],
            "train_return_pct": round(numbers["train_return_pct"], 2),
            "validation_return_pct": round(numbers["validation_return_pct"], 2),
            "validation_drawdown_pct": round(numbers["validation_drawdown_pct"], 2),
            "validation_trades": numbers["validation_trades"],
            "verdict": verdict["verdict"],
            "failures": verdict["failures"],
        })
    return rows


def collect_lessons() -> dict:
    """Read the lessons from mem0, newest round first."""
    try:
        import memory
        lessons = memory.read_lessons(memory.open_memory())
    except Exception as problem:   # mem0 or its embedding model is not available
        return {"available": False, "note": f"Memory could not be opened: {problem}", "lessons": []}
    lessons.sort(key=lambda lesson: (-lesson["round"], lesson["candidate"]))
    feed = [{"round": l["round"], "candidate": l["candidate"], "family": l["family"],
             "failed_checks": l["failed_checks"], "text": l["text"]}
            for l in lessons[:MEMORY_FEED_LIMIT]]
    return {"available": True, "total": len(lessons), "lessons": feed}


def build() -> dict:
    report = read_json(SPLIT_REPORT_FILE, {})
    data = {
        "generated_at": kill_switch._now(),
        "pair": report.get("pair"),
        "interval_minutes": report.get("interval_minutes"),
        "bar_count_all_windows": report.get("bar_count"),
        "windows": collect_windows(report),
        "bars": collect_bars(),
        "gate_thresholds": {
            "min_train_return_pct": gate.MIN_TRAIN_RETURN_PCT,
            "min_validation_return_pct": gate.MIN_VALIDATION_RETURN_PCT,
            "max_return_gap_pct_points": gate.MAX_RETURN_GAP_PCT_POINTS,
            "max_validation_drawdown_pct": gate.MAX_VALIDATION_DRAWDOWN_PCT,
            "min_validation_trades": gate.MIN_VALIDATION_TRADES,
        },
        "candidates": collect_candidates(),
        "memory": collect_lessons(),
        "rounds": read_json(ROUNDS_FILE, []),
        "paper": {
            "settings": paper.settings(),
            "state": read_json(paper.PAPER_STATE_FILE, None),
        },
        "strategy_isolation_problems": kill_switch.check_strategy_isolation(),
    }
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as handle:
        handle.write("// Written by build_view.py. Do not edit by hand.\n")
        handle.write(f"window.TERMINAL_DATA = {json.dumps(data)};\n")
    kill_switch.publish_state()   # make sure the kill switch file the page reads exists
    return data


def main() -> int:
    data = build()
    passed = sum(1 for row in data["candidates"] if row["verdict"] == "PASS")
    print(f"Wrote {DATA_FILE}")
    print(f"  bars on chart     : {len(data['bars'])} (train + validation; test window not opened)")
    print(f"  candidates        : {len(data['candidates'])} ({passed} pass, "
          f"{len(data['candidates']) - passed} blocked)")
    print(f"  memory lessons    : {len(data['memory']['lessons'])}")
    print(f"  paper state       : {'present' if data['paper']['state'] else 'none yet'}")
    print(f"  kill switch       : {'ENGAGED' if kill_switch.is_engaged() else 'RELEASED'}")
    print(f"Open {VIEW_DIR / 'index.html'} in a browser. No server needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
