#!/usr/bin/env python3
"""PreToolUse hook: stop anything that points this project at live trading.

Claude Code runs this script before every Bash, Edit, Write, MultiEdit and
NotebookEdit call (see .claude/settings.json). The tool call arrives as JSON
on standard input. This script reads the text Claude is about to run or
write and looks for three things:

1. Live broker endpoints (hosts and ports that place real orders).
2. Names or shapes of live API keys.
3. A config switch that turns live trading on.

For Bash it also refuses commands that touch a .env file, because the Read
deny rules in settings.json only cover Claude's file tools, not the shell.

Exit codes (from the Claude Code hooks documentation):
    0 = no objection, the normal permission flow continues
    2 = block the tool call; the text on stderr is shown to Claude

The script uses only the standard library, so it runs before anything is
installed. It never opens a network connection.

Note for editors: the patterns below are regular expressions, so this file
does not contain the literal strings it blocks. Keep it that way, or the hook
will refuse edits to itself.
"""

import json
import re
import sys

# ----------------------------------------------------------------------------
# RULES - each entry is (label shown in the message, regular expression).
# All matching ignores upper/lower case.
# ----------------------------------------------------------------------------

LIVE_ENDPOINT_RULES = [
    # Alpaca live trading host. The paper host has "paper-" in front of the
    # same name, so the pattern refuses to match when a "-" or a letter comes
    # directly before "api".
    ("Alpaca live trading host", r"(?<![\w-])api\.alpaca\.markets"),
    # Interactive Brokers - 7496 is the TWS live port and 4001 is the IB
    # Gateway live port. (Paper uses 7497 and 4002, which are not blocked.)
    # A bare number is too common to block, so it must look like a port:
    # after a colon, after the word "port", or inside a connect(...) call.
    ("IBKR live port", r":\s*(7496|4001)\b"),
    ("IBKR live port", r"\bport\b\W{0,4}(7496|4001)\b"),
    ("IBKR live port", r"connect\w*\([^)]*\b(7496|4001)\b"),
    # Interactive Brokers web API host (live accounts).
    ("IBKR live web API host", r"(?<![\w-])api\.ibkr\.com"),
    # Kraken private (account and order) endpoints. This project's data comes
    # from Kraken's public endpoints, which stay allowed.
    ("Kraken private endpoint", r"api\.kraken\.com/0/private"),
]

KEY_NAME_RULES = [
    # Any variable whose name joins the word "live" with "key", "secret" or
    # "token", in either order and with anything around them.
    ("live key name", r"\b\w*live\w*(key|secret|token)\w*\b"),
    ("live key name", r"\b\w*(key|secret|token)\w*live\w*\b"),
    # Alpaca live key IDs start with AK and are 20 characters long.
    # (Paper key IDs start with PK.) Case matters here, so it is compiled
    # separately below.
]

# Checked with upper/lower case respected.
CASE_SENSITIVE_KEY_RULES = [
    ("Alpaca live key ID", r"\bAK[A-Z0-9]{18}\b"),
]

LIVE_SWITCH_RULES = [
    # The word "live" (or a name starting with it) set to true, yes, on or 1,
    # written with a colon or an equals sign, quoted or not.
    ("live trading switch", r"\blive\w*[\"']?\s*[:=]\s*[\"']?(true|yes|on|1)\b"),
    # The word "paper" set to false, no, off or 0 says the same thing.
    ("paper mode switched off", r"\bpaper\w*[\"']?\s*[:=]\s*[\"']?(false|no|off|0)\b"),
]

# Bash only: any mention of a .env file (.env, .env.local, path/.env ...).
ENV_FILE_RULE = ("shell access to a .env file", r"(?<![\w.])\.env(\.[\w-]+)*(?![\w.])")

# Tools this hook inspects.
WATCHED_TOOLS = ("Bash", "Edit", "Write", "MultiEdit", "NotebookEdit")


def text_to_check(tool_name: str, tool_input: dict) -> str:
    """Return the text Claude is about to run or write for this tool call."""
    if tool_name == "Bash":
        return str(tool_input.get("command", ""))
    if tool_name == "Write":
        return str(tool_input.get("content", ""))
    if tool_name == "Edit":
        return str(tool_input.get("new_string", ""))
    if tool_name == "MultiEdit":
        # MultiEdit carries a list of edits; check every replacement text.
        edits = tool_input.get("edits", []) or []
        return "\n".join(str(edit.get("new_string", "")) for edit in edits if isinstance(edit, dict))
    if tool_name == "NotebookEdit":
        return str(tool_input.get("new_source", ""))
    return ""


def find_violations(tool_name: str, text: str) -> list:
    """Return a list of (label, matched text) for every rule the text breaks."""
    violations = []

    insensitive_rules = LIVE_ENDPOINT_RULES + KEY_NAME_RULES + LIVE_SWITCH_RULES
    if tool_name == "Bash":
        insensitive_rules = insensitive_rules + [ENV_FILE_RULE]

    for label, pattern in insensitive_rules:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            violations.append((label, match.group(0)))

    for label, pattern in CASE_SENSITIVE_KEY_RULES:
        match = re.search(pattern, text)
        if match:
            # Never echo a possible key back; show only its first four characters.
            violations.append((label, match.group(0)[:4] + "..."))

    return violations


def main() -> int:
    """Read one hook event from stdin and decide: 0 = allow, 2 = block."""
    raw = sys.stdin.read()
    try:
        event = json.loads(raw)
        tool_name = event.get("tool_name", "")
        tool_input = event.get("tool_input", {}) or {}
    except (json.JSONDecodeError, AttributeError):
        # A safety hook that cannot read its input must not wave the call through.
        print("BLOCKED by block_live.py: could not read the hook input, so the "
              "call was refused to be safe.", file=sys.stderr)
        return 2

    if tool_name not in WATCHED_TOOLS:
        return 0

    violations = find_violations(tool_name, text_to_check(tool_name, tool_input))
    if not violations:
        return 0

    lines = [f"BLOCKED by block_live.py: this {tool_name} call is refused. "
             "This project is PAPER ONLY."]
    for label, matched in violations:
        lines.append(f"  - {label}: found '{matched}'")
    lines.append("Use paper endpoints and paper keys only. If this is a false alarm, "
                 "stop and ask the owner; do not reword the call to get around the hook.")
    print("\n".join(lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
