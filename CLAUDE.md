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

## Non-negotiable rules

These come from the owner (2026-11-14). They override anything else in this file.

1. **Paper only.** No real orders, no real-money broker or exchange endpoints, no real-money keys,
   and no config switch that turns real trading on. `.claude/hooks/block_live.py` blocks these
   in every Bash, Edit and Write call. If the hook blocks something, stop and tell the owner;
   never reword a call to get past it.
2. **No LLM in the order path.** Nothing between a signal and an order may call a model, build a
   prompt or act on model output. Models may propose candidates for research; their answers are
   validated data and are never executed.
3. **Never edit `config/risk_limits.yaml` without the owner's explicit approval** in the
   conversation. The same goes for the owner blocks at the top of `gate.py` and `paper.py`, which
   that file mirrors. Never loosen a limit to get more passes.
4. **Every backtest variant is logged to `experiments/trials.csv`, failures included.** One row
   per variant, appended, never edited or deleted. A result that was not logged does not count.
5. **Never tune on the holdout window.** The holdout is `data/test_1h_LOCKED.json` (see "Data and
   the locked test window"). Do not open it, and do not choose, adjust or compare anything using it.

## Commands

Python 3.12 (`.python-version`); every dependency in `requirements.txt` is pinned.

| What | Command (run from the repo root) |
|---|---|
| Set up | `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt` |
| Hook tests only | `python -m pytest tests/test_block_live.py -v` |
| Install | `pip install -r requirements.txt` |
| Baseline B1 | `python -m src.research.run_baseline` then `python -m src.reporting.scorecard` (adds 14 trial rows per run; never add `--unlock-holdout`) |
| Daily bars | `python -m src.data.pipeline` (fetch, health-check, split and cache the universe in `config/paper.yaml`; a rerun prints the saved report; `--refresh` refetches and moves the holdout, owner only) |
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

## Layout and guardrails (added 2026-11-14)

- `src/{data,strategy,risk,execution,accounting,reporting,research}/` is where new code goes. The
  older modules (`engine.py`, `gate.py`, `paper.py`, `kill_switch.py`, `memory.py`, `optimizer.py`,
  `terminal/`) still sit at the repo root and have not been moved.
- `config/paper.yaml` holds paper account settings. `config/risk_limits.yaml` is the owner's limits
  file; today it is a checked copy of the numbers in `gate.py` and `paper.py` (`tests/test_config.py`
  fails if they differ), and limits the owner has not set are written as `TBD`.
- `experiments/trials.csv` is the trial log (header only so far). `reports/` holds generated reports.
- `.claude/settings.json` denies reading `.env*`, denies fetching real-money broker hosts, asks before
  any edit to `config/risk_limits.yaml`, and runs `.claude/hooks/block_live.py` before every Bash,
  Edit, Write, MultiEdit and NotebookEdit call. The hook's patterns are regular expressions so the
  file never contains the strings it blocks; keep tests and docs free of those literal strings too.
- Subagents in `.claude/agents/`: `quant-auditor` (read-only; Bash limited to pytest by
  `.claude/hooks/tests_only_bash.py`), `data-engineer`, `risk-officer` (read-only).

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
- **Reset on 2026-11-13 at the owner's request.** Memory, prompts and round logs built on the 5-minute data
  were moved to `data/archive_5m/` (`memory_store/`, `lessons.json`, `prompts/`, `rounds.json`,
  `optimizer_trials.jsonl`). Memory now starts empty, there are no saved prompts, and the next round is
  round 1. Nothing reads the archive. The old 5-minute period (Nov 10-12) sits inside the new locked test
  window, so never copy lessons or prompts back from the archive.
- Round 1 on the 1-hour data ran on 2026-11-13 (fixed grid, no model calls): 1 passed, 17 blocked, 17 lessons
  saved. Memory now rules out the whole EMA crossover family (all 9 blocked, every one on train return and
  validation drawdown).
- Three model-proposer rounds on the 1-hour data ran on 2026-11-13 (rounds 2-4 in `data/rounds.json`):
  passed 1, 2, 3 of 8 and blocked 7, 6, 5; memory grew from 17 to 35 lessons. Three distinct candidates
  have passed the gate so far: `zscore(window=80, entry=1.5)`, `zscore(window=70, entry=1.2)` and
  `zscore(window=90, entry=1.5)`, all neighbours judged on the same six-day validation window.
- Result on 1-hour bars with unchanged thresholds: 1 of 18 grid candidates passes the gate
  (`zscore(window=80, entry=1.5)`); on the 5-minute data 5 passed, and none of those 5 passes now.

## Daily bars (`src/data`) and the ledger (`src/accounting`), added 2026-11-14

- `python -m src.data.pipeline` downloads daily bars for the universe and cash vehicles in `config/paper.yaml`
  (the owner's ten ETFs plus SHV and BIL) from yfinance, which is research only. From 2003, NYSE calendar.
- Every row carries `source`, `fetched_at`, `available_at` (the session's closing time, UTC) and `adjustment`.
  yfinance prices are split adjusted, not dividend adjusted; `adj_close` is both. Never mix flags in one series.
- A symbol that fails any health check, or whose source fails, is not cached at all. Nothing is filled,
  interpolated or made up. The report is `reports/data_health.md` and `.json` (dates and counts, no prices).
- Current fetch (2026-11-14, at the owner's new windows): 2003-01-02 to 2026-11-13. All ten ETFs and SHV passed;
  BIL is blocked (one repeated bar on 2014-07-28) and unused. Train to 2016-12-30, validation 2017-01-03 to
  2021-12-31, **holdout 2022-01-03 onward, locked**.
- The cache is `data/cache/daily/<SYMBOL>/{train,validation,holdout_LOCKED}.parquet`, git-ignored.
  **Never open `holdout_LOCKED.parquet`.** `pipeline.load_bars` gives the holdout only to a logged unlock. Do not run `--refresh` without the owner asking: it moves all three windows.
- Lookahead rules, enforced by `src/data/lookahead.py`: a signal at bar t uses bars up to t only, and its
  order fills at the open of bar t+1. New strategy code must go through `run_signals` and `next_open_fills`
  and have an `assert_no_lookahead` test.
- The ledger (`src/accounting`) records cash, positions, fills, fees, slippage and cash-flow-adjusted
  equity. Futures rolls happen only through an explicit `roll(...)` call.

## Frozen baseline B1 (ETF trend + 12-1 momentum), added 2026-11-14

- Rules are in `protocol.md`, committed before any code or result. **B1 is frozen: never change its
  parameters, universe, costs, windows or gate thresholds.** A change needs a new protocol version from the owner.
- Run: `python -m src.research.run_baseline` (14 engine runs, each appended to `experiments/trials.csv`), then
  `python -m src.reporting.scorecard`. **Every run on real data adds trial rows, including reruns and any
  ad-hoc script.** Do not rerun casually; develop and test on fixture data only.
- Windows: train 2005-2016, validation 2017-2021, **holdout 2022 onward LOCKED**. The holdout opens only through
  `--unlock-holdout --who ... --reason ...`, which logs to `experiments/holdout_access.csv`. Never pass that flag,
  never call `src.data.holdout.unlock`, and never read `holdout_LOCKED.parquet` or post-2021 rows of the factor cache.
- First run (group `fac22b41`): scorecard PASS on all six gates as defined. **`reports/audit.md` reads it as FAIL**:
  gates 1, 5 and 6 pass only because of their definitions; the strategy trails 60/40 on Sharpe in train (0.50 vs
  0.56) and overall (0.65 vs 0.72). Do not describe B1 as passing without the audit.
- Trial log: 19 rows (14 from the run, 4 logged late for unlogged re-check runs, 1 audit follow-up).
- Metric definitions: `reports/metrics_spec.md`. Gate thresholds are the owner's, at the top of
  `src/reporting/scorecard.py`.
- The `quant-auditor` agent type is only available in sessions started after `.claude/agents/` existed.

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
- History on the 5-minute data (archived): one 20-trial run and a three-round run produced eight rewritten
  prompts, all flagged as overfitting; the baseline stayed best and hit the metric's ceiling (14 of 14).
  On the 1-hour data (three 20-trial runs, 2026-11-13) no rewrite beat the baseline, which is saved as v1.
  The train metric was 0 for every prompt: no proposed candidate passes the gate inside the train window
  (first two thirds vs last third), so the rewrite model gets only failures as feedback.
- The rewrite (reflection) model runs with caching off, so each rewrite step is a fresh call.

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
