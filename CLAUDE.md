# CLAUDE.md - Trading Terminal

## What this is

A self-improving paper trading terminal; see `PLAN.md` for the design. Built so
far: data fetch, backtest engine, risk gate, memory of failed ideas, round runner
and prompt optimizer. The Streamlit app (`app.py`) is still a placeholder page and
Kronos forecasting is not built.

`repos/` holds reference clones of Kronos, dspy and mem0. It is git-ignored;
do not edit or commit it.

This repository is the trading terminal; the project files sit at the repo root.
It was copied from `gouhinrj-rso/claude` (`trading-terminal/`) on 2026-11-13.

## Commands

| What | Command (run from the repo root) |
|---|---|
| Install | `pip install -r requirements.txt` |
| Run | `streamlit run app.py` |
| Test | `python -m pytest` |
| Backtest | `python engine.py` (all candidates on train and validation; fees 2 bps per side; never reads the test window) |
| Risk gate | `python gate.py` (PASS or BLOCK for every candidate; thresholds are the owner's, in the config block at the top of `gate.py`; never loosen them to get more passes) |
| Run a round | `python run_round.py` (read memory, propose, backtest, gate, save lessons, print memory) |
| Optimize the proposal prompt | `python optimizer.py` (DSPy GEPA; needs an LLM key in the environment; `--show` prints best and previous prompts) |
| Show memory | `python memory.py` (prints every stored lesson) |
| Fetch bars | `python fetch_data.py` (rerun prints the saved report; `--refresh` refetches and replaces every window) |

## Conventions

- All code in Python, commented, beginner-readable.
- One `render_<name>()` function per page, registered in `PAGES` in `app.py`.
- Non-UI logic (data fetching, calculations) goes in the `terminal/` package so it can be tested without Streamlit.
- Pandas in, Plotly out for charts.
- Secrets come from environment variables (`.env`, never committed). No keys in code.
- Paper trading only: no real orders, no exchange keys, no broker connections of any kind.

## Data and the locked test window

`fetch_data.py` saves Kraken ETH/USD 5-minute bars to `data/` and splits them by
time: 60% train, 20% validation, 20% test.

- **Do not open, read, print or load `data/test_5m_LOCKED.json` until the owner says "unlock test".**
- `data/bars_5m.json` contains every bar, including the test bars. Treat it as locked too: strategy, backtest and optimizer code must read only `data/train_5m.json` and `data/validation_5m.json`.
- `data/split_report.json` holds the range, counts, gaps and window boundaries (no prices). Read that when you need sizes or dates.
- Do not run `python fetch_data.py --refresh` without the owner asking. It moves all three windows, so old test bars could become training bars.

## Memory of failed ideas

`memory.py` stores one lesson per blocked candidate in mem0 (`memory_store/`, committed so it
survives a fresh checkout). `run_round.py` reads memory before proposing and skips anything
that already failed the same way.

- mem0 runs fully locally: FastEmbed `BAAI/bge-small-en-v1.5` for embeddings, Qdrant in file
  mode, lessons stored as written (`infer=False`). No API key is needed. The `lmstudio` LLM
  entry in the config only satisfies mem0's start-up; it is never called.
- The embedding model downloads once into `models/` (git-ignored, about 65 MB) and needs
  `huggingface.co` and `us.aws.cdn.hf.co` reachable for that download.
- A lesson must always carry the failed checks and the numbers. Never store a bare "failed".
- Settings are environment variables with defaults: `TT_MEMORY_DIR`, `TT_EMBED_MODEL`,
  `TT_EMBED_DIMS`, `TT_EMBED_CACHE`.

## Prompt optimizer

`optimizer.py` defines the strategy-proposal step as a DSPy module and optimizes its prompt with GEPA.

- Model settings are environment variables: `TT_PROPOSER_MODEL` (default `anthropic/claude-sonnet-5-5`),
  `TT_REFLECTION_MODEL` (default `anthropic/claude-opus-5-5`), `TT_MAX_TRIALS` (default 20),
  `TT_LLM_MAX_TOKENS`. The provider key (`ANTHROPIC_API_KEY` for the defaults) is read from the
  environment by DSPy. Never put a key in code, prompts or chat.
- Every trial logs two numbers to `data/optimizer_trials.jsonl`: the train metric (gate passes inside
  the train window, first two thirds vs last third) and the validation metric (passes of the real gate).
- The reflection model gets detailed feedback from train only; from validation it gets a count.
- Prompts are versioned in `prompts/` (`proposer_vNNN.json`, `index.json`). A prompt that improves on
  train but not on validation is saved as `flagged_overfitting` and never promoted.
- The proposer's answer is data: candidates are validated against `FAMILY_SPECS` and nothing the model
  writes is executed.
- `TT_ANTHROPIC_API_BASE` (default `https://api.anthropic.com/v1`) is passed to DSPy explicitly. DSPy
  appends only `/messages`, so an `ANTHROPIC_BASE_URL` without `/v1` gives HTTP 404 on every call.
- First real run 2026-11-13 (20 trials, Sonnet 5.5 proposer, Opus 5.5 reflection): the baseline (v1)
  stayed best (train 8, validation 10). The one rewrite (v2) scored train 14, validation 7 and was
  flagged as overfitting: it had copied specific train-window parameter values into the prompt.
- Known weakness: DSPy's response cache returns the same rewrite when the reflection input repeats, so
  that run produced only one distinct new prompt from three rewrite steps.
