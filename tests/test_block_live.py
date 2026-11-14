"""Tests for the two safety hooks in .claude/hooks/.

Each test runs the hook the same way Claude Code does: as a separate process,
with the tool call as JSON on standard input, and reads the exit code.
Exit code 2 means "blocked"; exit code 0 means "no objection".

The forbidden strings are glued together from pieces (for example
"api." + "alpaca.markets") so this file does not contain them whole. Otherwise
the hook would refuse any later edit to its own tests.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK_DIR = Path(__file__).resolve().parent.parent / ".claude" / "hooks"
BLOCK_LIVE = HOOK_DIR / "block_live.py"
TESTS_ONLY = HOOK_DIR / "tests_only_bash.py"

# --- forbidden strings, built from pieces -----------------------------------
ALPACA_LIVE = "https://api." + "alpaca.markets/v2/orders"
ALPACA_PAPER = "https://paper-api." + "alpaca.markets/v2/orders"
IBKR_WEB = "https://api." + "ibkr.com/v1/api/iserver/account/orders"
KRAKEN_PRIVATE = "https://api.kraken.com/0/" + "private/AddOrder"
KRAKEN_PUBLIC = "https://api.kraken.com/0/public/OHLC?pair=ETHUSD"
TWS_LIVE_PORT = "74" + "96"
GATEWAY_LIVE_PORT = "40" + "01"
REAL_NAME_ONE = "ALPACA_LI" + "VE_API_KEY"
REAL_NAME_TWO = "API_SECRET_" + "LIVE"
ALPACA_REAL_ID = "AK" + "ABCDEFGHIJ12345678"      # made-up, right shape (AK + 18)
ALPACA_PAPER_ID = "PK" + "ABCDEFGHIJ12345678"     # made-up paper-shaped ID
LIVE_TRUE_YAML = "live" + ": true"
LIVE_TRUE_JSON = '"live"' + ": true"
LIVE_TRUE_PY = "LIVE" + " = True"
LIVE_TRADING_ENV = "LIVE_TRADING" + "=1"
PAPER_FALSE = "paper" + ": false"
DOT_ENV = "." + "env"


def run_hook(script: Path, tool_name: str, tool_input: dict):
    """Run one hook script with a fake tool call; return (exit code, stderr)."""
    event = {"hook_event_name": "PreToolUse", "tool_name": tool_name, "tool_input": tool_input}
    result = subprocess.run([sys.executable, str(script)], input=json.dumps(event),
                            capture_output=True, text=True, timeout=30)
    return result.returncode, result.stderr


# ============================================================================
# block_live.py - calls that MUST be blocked
# ============================================================================
BLOCKED_CALLS = [
    # --- live endpoints ---
    ("bash curl to Alpaca live host", "Bash", {"command": f"curl -X POST {ALPACA_LIVE}"}),
    ("write Alpaca live host into code", "Write",
     {"file_path": "src/execution/broker.py", "content": f'BASE_URL = "{ALPACA_LIVE}"\n'}),
    ("edit Alpaca live host into code", "Edit",
     {"file_path": "src/execution/broker.py", "old_string": "x", "new_string": f'url = "{ALPACA_LIVE}"'}),
    ("Alpaca live host in upper case", "Bash", {"command": f"curl {ALPACA_LIVE.upper()}"}),
    ("IBKR web API host", "Bash", {"command": f"curl {IBKR_WEB}"}),
    ("Kraken private endpoint", "Write",
     {"file_path": "src/execution/kraken.py", "content": f'URL = "{KRAKEN_PRIVATE}"\n'}),
    ("TWS live port after host", "Write",
     {"file_path": "src/execution/ib.py", "content": f'ib.connect("127.0.0.1", {TWS_LIVE_PORT}, clientId=1)\n'}),
    ("TWS live port as host:port", "Bash", {"command": f"nc -z 127.0.0.1:{TWS_LIVE_PORT}"}),
    ("Gateway live port as port=", "Edit",
     {"file_path": "config/paper.yaml", "old_string": "x", "new_string": f"port: {GATEWAY_LIVE_PORT}"}),
    ("Gateway live port keyword argument", "Write",
     {"file_path": "src/execution/ib.py", "content": f"PORT = 1\nconnect(host, port={GATEWAY_LIVE_PORT})\n"}),
    # --- live key names and key shapes ---
    ("live key variable in code", "Write",
     {"file_path": "src/execution/keys.py", "content": f'key = os.environ["{REAL_NAME_ONE}"]\n'}),
    ("live secret variable, LIVE last", "Edit",
     {"file_path": "a.py", "old_string": "x", "new_string": f"{REAL_NAME_TWO} = None"}),
    ("live key exported in shell", "Bash", {"command": f"export {REAL_NAME_ONE}=abc && python run.py"}),
    ("Alpaca live key ID shape", "Write",
     {"file_path": "notes.txt", "content": f"key id {ALPACA_REAL_ID}\n"}),
    # --- the live switch ---
    ("live true in yaml", "Write", {"file_path": "config/paper.yaml", "content": f"mode: x\n{LIVE_TRUE_YAML}\n"}),
    ("live true in json", "Edit",
     {"file_path": "config/x.json", "old_string": "x", "new_string": "{" + LIVE_TRUE_JSON + "}"}),
    ("switch set in python", "Write", {"file_path": "src/execution/mode.py", "content": f"{LIVE_TRUE_PY}\n"}),
    ("switch set as a shell variable", "Bash", {"command": f"{LIVE_TRADING_ENV} python paper.py"}),
    ("paper false in yaml", "Write", {"file_path": "config/paper.yaml", "content": f"{PAPER_FALSE}\n"}),
    ("live true inside a MultiEdit", "MultiEdit",
     {"file_path": "config/paper.yaml",
      "edits": [{"old_string": "a", "new_string": "b"}, {"old_string": "c", "new_string": LIVE_TRUE_YAML}]}),
    ("live true inside a notebook cell", "NotebookEdit",
     {"notebook_path": "nb.ipynb", "new_source": LIVE_TRUE_PY}),
    # --- .env through the shell ---
    ("cat the env file", "Bash", {"command": f"cat {DOT_ENV}"}),
    ("cat a local env file by path", "Bash", {"command": f"cat ./config/{DOT_ENV}.local"}),
    ("redirect from the env file", "Bash", {"command": f"python load.py < {DOT_ENV}"}),
    ("source the env file", "Bash", {"command": f"set -a; source {DOT_ENV}; set +a"}),
]


@pytest.mark.parametrize("label, tool_name, tool_input", BLOCKED_CALLS, ids=[c[0] for c in BLOCKED_CALLS])
def test_block_live_blocks(label, tool_name, tool_input):
    code, stderr = run_hook(BLOCK_LIVE, tool_name, tool_input)
    assert code == 2, f"{label}: expected exit 2 (blocked), got {code}"
    # The message must say who blocked it and why, so Claude can explain it.
    assert "BLOCKED by block_live.py" in stderr
    assert "PAPER ONLY" in stderr


# ============================================================================
# block_live.py - calls that must NOT be blocked (no false alarms)
# ============================================================================
ALLOWED_CALLS = [
    ("run the tests", "Bash", {"command": "python -m pytest -q"}),
    ("Alpaca PAPER host", "Write",
     {"file_path": "src/execution/broker.py", "content": f'BASE_URL = "{ALPACA_PAPER}"\n'}),
    ("Alpaca paper key ID shape", "Write", {"file_path": "notes.txt", "content": ALPACA_PAPER_ID}),
    ("Kraken public data endpoint", "Bash", {"command": f"curl '{KRAKEN_PUBLIC}'"}),
    ("IBKR paper ports", "Write",
     {"file_path": "src/execution/ib.py", "content": 'ib.connect("127.0.0.1", 7497)\nport: 4002\n'}),
    ("live port digits as a plain number", "Write",
     {"file_path": "src/accounting/x.py", "content": f"cash = {GATEWAY_LIVE_PORT}.50\nrows = {TWS_LIVE_PORT}\n"}),
    ("live false", "Write", {"file_path": "config/paper.yaml", "content": "live: false\npaper: true\n"}),
    ("the word live in prose", "Write",
     {"file_path": "README.md", "content": "This never trades live. Deliver results to the owner.\n"}),
    ("paper key name", "Write", {"file_path": "a.py", "content": 'os.environ["ALPACA_PAPER_API_KEY"]\n'}),
    ("edit text of a file named like env", "Edit",
     {"file_path": "/repo/.gitignore", "old_string": "x", "new_string": f"{DOT_ENV}\n{DOT_ENV}.*\n"}),
    ("shell word that only contains env", "Bash", {"command": "python -m venv .venv && echo environment"}),
    ("a tool the hook does not watch", "Read", {"file_path": "README.md"}),
]


@pytest.mark.parametrize("label, tool_name, tool_input", ALLOWED_CALLS, ids=[c[0] for c in ALLOWED_CALLS])
def test_block_live_allows(label, tool_name, tool_input):
    code, stderr = run_hook(BLOCK_LIVE, tool_name, tool_input)
    assert code == 0, f"{label}: expected exit 0 (allowed), got {code}. stderr: {stderr}"


def test_block_live_names_the_rule_it_hit():
    """The message lists what was found, so a false alarm can be diagnosed."""
    code, stderr = run_hook(BLOCK_LIVE, "Bash", {"command": f"curl {ALPACA_LIVE}"})
    assert code == 2
    assert "Alpaca live trading host" in stderr


def test_message_hides_the_matched_id():
    """A key-shaped string is cut to four characters in the message."""
    code, stderr = run_hook(BLOCK_LIVE, "Write", {"file_path": "n.txt", "content": ALPACA_REAL_ID})
    assert code == 2
    assert ALPACA_REAL_ID not in stderr


def test_block_live_fails_closed_on_bad_input():
    """If the hook cannot read its input it blocks, rather than waving the call through."""
    result = subprocess.run([sys.executable, str(BLOCK_LIVE)], input="not json",
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2


def test_hook_source_does_not_trip_itself():
    """The hook files and this test file can still be edited with the hook switched on."""
    for path in (BLOCK_LIVE, TESTS_ONLY, Path(__file__)):
        code, stderr = run_hook(BLOCK_LIVE, "Write", {"file_path": str(path), "content": path.read_text()})
        assert code == 0, f"{path.name} would be blocked by the hook: {stderr}"


def test_settings_file_wires_the_hook():
    """settings.json runs block_live.py before Bash, Edit and Write, and denies .env reads."""
    settings = json.loads((HOOK_DIR.parent / "settings.json").read_text())
    entries = settings["hooks"]["PreToolUse"]
    wired = [e for e in entries if any("block_live.py" in h["command"] for h in e["hooks"])]
    assert wired, "block_live.py is not registered as a PreToolUse hook"
    for tool in ("Bash", "Edit", "Write"):
        assert tool in wired[0]["matcher"].split("|")
    deny = settings["permissions"]["deny"]
    assert "Read(.env)" in deny and "Read(.env.*)" in deny


# ============================================================================
# tests_only_bash.py - the quant-auditor's Bash limit
# ============================================================================
@pytest.mark.parametrize("command", [
    "python -m pytest",
    "python3 -m pytest tests/test_gate.py -k drawdown -q",
    "python3.12 -m pytest -x",
    "pytest -q tests/test_engine.py",
])
def test_auditor_may_run_pytest(command):
    code, stderr = run_hook(TESTS_ONLY, "Bash", {"command": command})
    assert code == 0, stderr


@pytest.mark.parametrize("command", [
    "python engine.py",
    "python kill_switch.py release --confirm",
    "cat data/train_1h.json",
    "git commit -am x",
    "python -m pytest; rm -rf data",
    "python -m pytest && python paper.py",
    "pytest -q > out.txt",
    "pytest $(cat x)",
    "python -m pytestx",
])
def test_auditor_may_not_run_anything_else(command):
    code, stderr = run_hook(TESTS_ONLY, "Bash", {"command": command})
    assert code == 2
    assert "BLOCKED by tests_only_bash.py" in stderr
