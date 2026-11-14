"""The trial log: one row in experiments/trials.csv for every backtest run on real data.

Why: if twenty versions of a strategy are tried and the best one is reported,
its result is partly luck. The only defence is an honest count of everything
that was tried. So every run is written down here - the baseline, every
neighbour, every cost run, every benchmark, reruns and failures.

Rules:
  - Rows are only ever appended. Nothing here edits or deletes a row.
  - The number of rows is the trial count N used for the Deflated Sharpe Ratio.
  - Tests write to a temporary file, never to the real log.
"""

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
TRIALS_FILE = PROJECT_DIR / "experiments" / "trials.csv"

COLUMNS = [
    "trial_id", "logged_at_utc", "git_commit", "git_dirty", "run_group", "run_kind", "strategy",
    "params_json", "cost_multiple", "commission_bps", "slippage_bps", "window_start", "window_end",
    "holdout_unlocked", "n_days", "total_return", "cagr", "sharpe_annual", "sharpe_per_day",
    "max_drawdown", "n_round_trips", "status", "failure_reason", "notes",
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def read_trials(trials_file: Path | None = None) -> list:
    """Every logged trial as a list of dictionaries, oldest first."""
    trials_file = Path(trials_file or TRIALS_FILE)
    if not trials_file.exists():
        return []
    with open(trials_file, newline="") as handle:
        return list(csv.DictReader(handle))


def count_trials(trials_file: Path | None = None) -> int:
    """N: the number of rows in the log, whatever kind of run each row is."""
    return len(read_trials(trials_file))


def append_trial(row: dict, trials_file: Path | None = None) -> dict:
    """Append one run to the log and return the full row that was written.

    `row` supplies the run's own fields; the id, the time and the git state are
    filled in here. Unknown keys are refused so a typo cannot silently drop a value.
    """
    trials_file = Path(trials_file or TRIALS_FILE)
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown trial fields: {sorted(unknown)}")
    existing = read_trials(trials_file)
    full = {column: "" for column in COLUMNS}
    full.update(row)
    full["trial_id"] = f"T{len(existing) + 1:04d}"
    full["logged_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    full["git_commit"] = _git("rev-parse", "--short", "HEAD") or "unknown"
    full["git_dirty"] = "yes" if _git("status", "--porcelain", "--untracked-files=no") else "no"
    if isinstance(full["params_json"], dict):
        full["params_json"] = json.dumps(full["params_json"], sort_keys=True)

    trials_file.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not trials_file.exists() or trials_file.stat().st_size == 0
    with open(trials_file, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(full)
    return full
