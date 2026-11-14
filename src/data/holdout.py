"""The lock on the holdout window.

The holdout is the last stretch of data, kept unseen so that one honest final
test is still possible. Looking at it even once to "check something" spends it.

There is exactly one way to open it:

    access = unlock(who="...", reason="...", command="...")

`unlock` first appends a line to experiments/holdout_access.csv (who, when,
why, git commit, the command line) and only then returns a HoldoutAccess. The
data loader refuses holdout bars without one. If the log cannot be written,
nothing is unlocked.

Only the owner unlocks. Claude never calls `unlock` and never passes
`--unlock-holdout`.
"""

import csv
import getpass
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
ACCESS_LOG = PROJECT_DIR / "experiments" / "holdout_access.csv"
LOG_COLUMNS = ["unlocked_at_utc", "who", "os_user", "git_user", "git_commit", "reason", "command"]

# A private marker so a HoldoutAccess cannot be built by hand from outside this file.
_MADE_BY_UNLOCK = object()


class HoldoutLocked(RuntimeError):
    """The holdout was asked for without a logged unlock."""


@dataclass(frozen=True)
class HoldoutAccess:
    """Proof that an unlock was logged. Made only by `unlock`."""
    who: str
    unlocked_at_utc: str
    _marker: object = None

    def __post_init__(self):
        if self._marker is not _MADE_BY_UNLOCK:
            raise HoldoutLocked("a HoldoutAccess can only come from src.data.holdout.unlock(...)")


def _git(*args: str) -> str:
    """Run a git command and return its output, or "unknown" if git is not available."""
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def unlock(who: str, reason: str, command: str = "", log_file: Path | None = None) -> HoldoutAccess:
    """Log an unlock of the holdout, then return the access object.

    `who` is the person's name as they typed it; the operating-system user and
    the git user are recorded next to it. `reason` must not be empty.
    """
    if not who or not who.strip():
        raise HoldoutLocked("unlocking the holdout needs a name (who)")
    if not reason or not reason.strip():
        raise HoldoutLocked("unlocking the holdout needs a reason")
    log_file = Path(log_file or ACCESS_LOG)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = {"unlocked_at_utc": now, "who": who.strip(), "os_user": getpass.getuser(),
           "git_user": _git("config", "user.name"), "git_commit": _git("rev-parse", "--short", "HEAD"),
           "reason": reason.strip(), "command": command}

    # Write the log line FIRST. If this fails, the exception stops us before any access exists.
    log_file.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_file.exists()
    with open(log_file, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    return HoldoutAccess(who=row["who"], unlocked_at_utc=now, _marker=_MADE_BY_UNLOCK)


def read_log(log_file: Path | None = None) -> list:
    """Return every logged unlock as a list of dictionaries (empty if there were none)."""
    log_file = Path(log_file or ACCESS_LOG)
    if not log_file.exists():
        return []
    with open(log_file, newline="") as handle:
        return list(csv.DictReader(handle))
