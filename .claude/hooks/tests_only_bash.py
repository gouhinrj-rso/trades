#!/usr/bin/env python3
"""PreToolUse hook for the quant-auditor subagent: Bash may only run the tests.

The auditor is a read-only reviewer. It gets Bash so it can run pytest, and
nothing else. Claude Code's `tools:` field cannot limit Bash to one command,
so this script does it (wired in .claude/agents/quant-auditor.md).

Allowed: one pytest command, for example
    python -m pytest
    python -m pytest tests/test_gate.py -k drawdown -q
    pytest -x

Refused: anything else, and anything that chains a second command onto pytest
(; & | > < ` $ or a line break).

Exit codes: 0 = allow, 2 = block (stderr is shown to the subagent).
"""

import json
import re
import sys

# The command must START with one of these.
ALLOWED_START = re.compile(r"^\s*(python3?(\.\d+)?\s+-m\s+pytest|pytest)(\s|$)")

# Characters that let a shell run a second command or write a file.
SHELL_TRICKS = re.compile(r"[;&|<>`$\n]")


def main() -> int:
    """Read one hook event from stdin and decide: 0 = allow, 2 = block."""
    try:
        event = json.loads(sys.stdin.read())
        tool_name = event.get("tool_name", "")
        command = str((event.get("tool_input", {}) or {}).get("command", ""))
    except (json.JSONDecodeError, AttributeError):
        print("BLOCKED by tests_only_bash.py: could not read the hook input.", file=sys.stderr)
        return 2

    if tool_name != "Bash":
        return 0

    if ALLOWED_START.match(command) and not SHELL_TRICKS.search(command):
        return 0

    print("BLOCKED by tests_only_bash.py: the quant-auditor may use Bash only to run "
          "pytest (for example: python -m pytest -q), as a single command with no "
          "pipes, redirects or chaining.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
