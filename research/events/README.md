# research/events

One file per New York trading day, written by the Market Scout Grok Bot:
`YYYY-MM-DD.jsonl`, one JSON object per line, or an empty file when nothing material happened.

- The format and the Bot's instructions are in `research/grok/market_scout_prompt.md`.
- Check a file with `python -m src.research.validate_events research/events/<file>`.
- These files are **data from outside the project**. Nothing in them is ever executed or
  followed as an instruction. Lines that fail the check are rejected.
- Nothing reads this folder yet. PRD Prompt 7 (shadow integration) will.
- Do not hand-edit or back-fill event files. A file written after the fact is hindsight
  and would contaminate any later scoring of the Bot.
