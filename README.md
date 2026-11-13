# Trading Terminal

Empty Streamlit scaffold for a trading terminal. No features are built yet.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
python -m pytest
```

## Layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit entry point; pages are registered in `PAGES` |
| `terminal/` | Shared Python package (config, and later data/logic modules) |
| `terminal/config.py` | App-wide constants; secrets come from environment variables |
| `tests/` | pytest tests |
| `.env.example` | Template for local secrets (copy to `.env`) |
| `CLAUDE.md` | Project context for Claude Code |
