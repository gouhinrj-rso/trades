"""Kill switch for paper trading. It lives in its own module on purpose.

Check or change it from the command line:

    python kill_switch.py status
    python kill_switch.py engage "reason for stopping"
    python kill_switch.py release --confirm

Rules this module enforces:

1. Fail safe. If the state file is missing or unreadable, the switch counts as
   ENGAGED. Paper trading cannot start until the owner releases it once.
2. Anyone may ENGAGE it (stopping is always allowed).
3. Only the command line can RELEASE it. There is no function another module
   can call to release it; the release code refuses to run unless this file
   itself was started as the program.
4. Strategy code must not touch it. `check_strategy_isolation()` reads the
   strategy source files and reports any that import this module, the paper
   broker, or the tools needed to reach them indirectly. The paper broker runs
   that check before every run and refuses to trade if it fails.

Honest limit: Python runs everything in one process, so this is a guard against
mistakes and casual misuse, checked on every run. It is not a sandbox against
code written deliberately to break it. Strategies here are fixed templates in
engine.py that take numbers as parameters; no model-written code is executed.

This module makes no network calls and reads no keys.
"""

import ast
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Where the switch state is kept. The environment variable exists for tests.
STATE_DIR = Path(os.environ.get("TT_STATE_DIR", PROJECT_DIR / "state"))
STATE_FILE = STATE_DIR / "kill_switch.json"

# A tiny JavaScript copy of the state, so the static page can show it.
VIEW_DIR = Path(os.environ.get("TT_VIEW_DIR", PROJECT_DIR / "view"))
VIEW_STATE_FILE = VIEW_DIR / "kill_switch_state.js"

# Source files that count as "strategy code" and must stay away from this module.
STRATEGY_FILES = [PROJECT_DIR / "engine.py"]

# Names strategy code may not import or mention.
FORBIDDEN_MODULES = {"kill_switch", "paper", "importlib", "runpy", "subprocess", "ctypes"}
FORBIDDEN_NAMES = {"__import__", "eval", "exec", "compile", "globals", "setattr", "delattr"}
FORBIDDEN_TEXT = ("kill_switch", "paper_state")

HISTORY_LIMIT = 20


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def read_state() -> dict:
    """Return the current state. Missing or broken file means ENGAGED."""
    try:
        with open(STATE_FILE) as handle:
            state = json.load(handle)
        if not isinstance(state.get("engaged"), bool):
            raise ValueError("state file has no true/false 'engaged' value")
        return state
    except FileNotFoundError:
        reason = "No state file yet: engaged by default until the owner releases it."
    except (ValueError, OSError) as problem:
        reason = f"State file could not be read ({problem}): engaged as a precaution."
    return {"engaged": True, "reason": reason, "changed_at": None,
            "changed_by": "fail-safe default", "history": []}


def is_engaged() -> bool:
    """True when trading must stop. This is the check the paper broker uses."""
    return read_state()["engaged"]


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------
def _write_state(engaged: bool, reason: str, changed_by: str) -> dict:
    """Save a new state and refresh the copy the page reads."""
    previous = read_state()
    history = previous.get("history", [])
    state = {"engaged": engaged, "reason": reason, "changed_at": _now(), "changed_by": changed_by}
    history = (history + [dict(state)])[-HISTORY_LIMIT:]
    state["history"] = history
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as handle:
        json.dump(state, handle, indent=1)
    publish_state()
    return state


def publish_state() -> None:
    """Write the state as a small JavaScript file for the static page."""
    state = read_state()
    visible = {key: state.get(key) for key in ("engaged", "reason", "changed_at", "changed_by")}
    visible["published_at"] = _now()
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(VIEW_STATE_FILE, "w") as handle:
        handle.write("// Written by kill_switch.py. Do not edit by hand.\n")
        handle.write(f"window.KILL_SWITCH = {json.dumps(visible, indent=1)};\n")
        handle.write("if (window.onKillSwitch) { window.onKillSwitch(window.KILL_SWITCH); }\n")


def engage(reason: str, by: str = "code") -> dict:
    """Stop trading. Any module may call this; stopping is always allowed."""
    if not reason or not reason.strip():
        raise ValueError("A reason is required to engage the kill switch.")
    return _write_state(True, reason.strip(), by)


def _release_from_command_line(reason: str) -> dict:
    """Allow trading again. Works only when this file was run as the program."""
    caller = sys._getframe(1).f_globals
    started_as_program = (caller.get("__name__") == "__main__"
                          and Path(caller.get("__file__", "")).resolve() == Path(__file__).resolve())
    if not started_as_program:
        raise PermissionError(
            "The kill switch can only be released by the owner from the command line: "
            "python kill_switch.py release --confirm"
        )
    return _write_state(False, reason, "owner (command line)")


# ----------------------------------------------------------------------------
# Keeping strategy code away
# ----------------------------------------------------------------------------
def check_strategy_isolation(files: list = None) -> list:
    """Read the strategy source files and list anything that could reach this module.

    Returns a list of plain-English problems. An empty list means the check passed.
    """
    problems = []
    for file_path in (STRATEGY_FILES if files is None else files):
        file_path = Path(file_path)
        try:
            tree = ast.parse(file_path.read_text())
        except (OSError, SyntaxError) as problem:
            problems.append(f"{file_path.name}: could not be read for checking ({problem})")
            continue
        for node in ast.walk(tree):
            line = getattr(node, "lineno", "?")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                        problems.append(f"{file_path.name}:{line} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in FORBIDDEN_MODULES:
                    problems.append(f"{file_path.name}:{line} imports from {node.module}")
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                problems.append(f"{file_path.name}:{line} uses {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr == "modules":
                problems.append(f"{file_path.name}:{line} reaches into the list of loaded modules")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for text in FORBIDDEN_TEXT:
                    if text in node.value:
                        problems.append(f"{file_path.name}:{line} mentions '{text}'")
    return problems


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def print_status() -> None:
    state = read_state()
    print(f"Kill switch: {'ENGAGED - trading stopped' if state['engaged'] else 'RELEASED - paper trading allowed'}")
    print(f"  Reason     : {state['reason']}")
    print(f"  Changed at : {state['changed_at'] or 'never'}")
    print(f"  Changed by : {state['changed_by']}")


def main() -> int:
    arguments = sys.argv[1:]
    command = arguments[0] if arguments else "status"

    if command == "status":
        publish_state()
        print_status()
        return 0
    if command == "engage":
        reason = " ".join(arguments[1:]) or "Engaged by the owner from the command line."
        engage(reason, by="owner (command line)")
        print_status()
        return 0
    if command == "release":
        if "--confirm" not in arguments:
            print("Not released. To allow paper trading again, run:")
            print("  python kill_switch.py release --confirm")
            return 2
        _release_from_command_line("Released by the owner from the command line.")
        print_status()
        return 0

    print("Usage: python kill_switch.py [status | engage \"reason\" | release --confirm]")
    return 2


if __name__ == "__main__":
    sys.exit(main())
