# research/ingested

Written by `python -m src.research.grok_events ingest`. Do not edit by hand.

- `events.jsonl`: every event line that passed the checks, with `ingested_at` (when this
  project first read it), `source_file` and `source_line`.
- `rejected.jsonl`: every line that failed, with the reasons.

Rows are only ever appended. `ingested_at` is what decides whether a veto was known in
time, so changing it, or adding events after the fact, would be lookahead.
