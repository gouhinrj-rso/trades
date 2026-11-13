# CLAUDE.md - Trading Terminal

## What this is

A self-improving paper trading terminal; see `PLAN.md` for the design. Built so
far: data fetch, backtest engine, risk gate, memory of failed ideas, round runner
prompt optimizer, paper broker with kill switch, and a static terminal page (`view/index.html`).
The Streamlit app (`app.py`) is still a placeholder page and Kronos forecasting is not built.

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
| Build the page data | `python build_view.py` (writes `view/data.js`; then open `view/index.html`, no server) |
| Paper replay | `python paper.py --candidate "<name>"` (gate-passed candidates only; simulated orders) |
| Kill switch | `python kill_switch.py status` / `engage "reason"` / `release --confirm` |
| Run three full rounds | `python run_rounds.py` (model proposes; each round ends with a 20-trial prompt optimization; needs an LLM key; up to 90 model calls) |
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

`fetch_data.py` saves Kraken ETH/USD **1-hour** bars to `data/` (720 bars, about 30 days: 2026-10-14 to
2026-11-13) and splits them by time: 60% train, 20% validation, 20% test. File names carry the bar size
(`train_1h.json`, `validation_1h.json`, `test_1h_LOCKED.json`, `bars_1h.json`).

- **Do not open, read, print or load `data/test_1h_LOCKED.json` until the owner says "unlock test".**
- `data/bars_1h.json` contains every bar, including the test bars. Treat it as locked too: strategy, backtest and optimizer code must read only `data/train_1h.json` and `data/validation_1h.json`.
- `data/split_report.json` holds the range, counts, gaps and window boundaries (no prices). Read that when you need sizes or dates.
- Do not run `python fetch_data.py --refresh` without the owner asking. It moves all three windows, so old test bars could become training bars.
- `data/archive_5m/` holds the first data set (5-minute bars, 2026-11-10 to 2026-11-12) and its gate results.
  Its test file was never opened and is still locked. Nothing reads this folder.
- Strategy parameters are counted in bars, so the same candidate name means a different strategy on
  1-hour bars than it did on 5-minute bars (`window=40` is 40 hours now, 200 minutes before).
- **Stale since the switch to 1-hour bars (2026-11-13), pending an owner decision:** the 17 lessons in
  `memory_store/`, prompts v1-v9 in `prompts/`, `data/rounds.json` and `data/optimizer_trials.jsonl` all
  come from the 5-minute data. Do not run `run_round.py`, `run_rounds.py` or `optimizer.py` until the owner
  says whether to reset or tag them: memory would wrongly skip candidates by name. The old 5-minute period
  (Nov 10-12) also sits inside the new locked test window.
- Result on 1-hour bars with unchanged thresholds: 1 of 18 grid candidates passes the gate
  (`zscore(window=80, entry=1.5)`); on the 5-minute data 5 passed, and none of those 5 passes now.

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
- The rewrite (reflection) model now runs with caching off, so each rewrite step is a fresh call.
- Three-round run 2026-11-13 (`run_rounds.py`, rounds 3-5 in `data/rounds.json`): passed 6, 6, 8 of 8;
  blocked 2, 2, 0; memory grew from 13 to 17 lessons. All seven rewritten prompts (v3-v9) were flagged as
  overfitting and v1 stayed best. In rounds 4 and 5 v1 scored 14 of a possible 14 on validation, so the
  metric is at its ceiling and no rewrite can be promoted until the metric or the data changes.

## Paper execution and the kill switch

`paper.py` is a simulated broker. There is no real broker, exchange, key or network call anywhere in it.

- Owner's settings (2026-11-13), in the block at the top of `paper.py`: limit orders only, a stop 1% below
  every entry, a 2% daily loss cap, $10,000 paper capital. Do not change them without the owner asking.
- `kill_switch.py` is its own module. A missing or unreadable state file means ENGAGED. Code may engage it.
  **Only the owner releases it, from the command line. Never release it yourself, and never add a function
  that releases it.**
- Strategy code (`engine.py`) must never import or mention `kill_switch` or `paper`. `paper.py` runs
  `kill_switch.check_strategy_isolation()` before every run and refuses to trade if it fails. Strategies
  return wanted positions only; they never receive the broker.
- `python paper.py` is a replay over the validation window, not forward trading. There are no new bars yet.
- `view/index.html` is static: it loads `view/data.js` and `view/kill_switch_state.js` with script tags and
  re-reads the kill switch file every 5 seconds. It shows train and validation prices only; the test window
  is drawn as a locked band from its dates. `build_view.py` must never open the test file.
- Page colors pass the RSO brand palette check (no gold/amber other than `#FFC107` for the status banner).
