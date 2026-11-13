# PLAN.md - Self-Improving Paper Trading Terminal

Status: plan only. No code has been written for this plan.
Date: 2026-11-12

## Ground rules

- Paper trading only. No real orders, no exchange keys, no broker connections of any kind.
- Market data comes from public, keyless sources or from CSV files the owner supplies.
- Where a README or doc does not say something, this plan writes "not stated".
- Conventions from `CLAUDE.md` apply: Python only, commented and beginner-readable, non-UI logic in `terminal/`, one `render_<name>()` per page, pandas in and Plotly out, secrets in `.env`.

Sources read (cloned into `./repos`, which is git-ignored and not committed):

| Repo | Commit | Commit date | What was read |
|---|---|---|---|
| shiyu-coder/Kronos | `67b630e` | 2026-04-13 | `README.md`, `webui/README.md`, `requirements.txt`, signatures in `model/kronos.py` |
| stanfordnlp/dspy | `2c07ebb` | 2026-11-12 | `README.md`, `docs/docs/getting-started/installation.md`, `docs/docs/diving-deeper/` (choosing-an-optimizer, gepa-in-depth, metrics-and-evaluation, saving-and-loading) |
| mem0ai/mem0 | `f1791da` | 2026-11-06 | `README.md`, `docs/open-source/` (python-quickstart, configuration), `docs/core-concepts/memory-operations/add.mdx`, embedder and Qdrant component pages |

---

## 1. The three repos

### 1.1 Kronos - forecasting

**What we use it for.** Produce a forecast of the next N bars (open, high, low, close, volume, amount) from the most recent bars of the chosen asset. The forecast is one input to a strategy (for example "forecast close is above the last close by more than X"). It is never treated as ground truth.

**Entry points we will call** (README, "Making Forecasts"):

- `from model import Kronos, KronosTokenizer, KronosPredictor`
- `KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")`
- `Kronos.from_pretrained("NeoQuasar/Kronos-small")` (open-source options: `Kronos-mini`, `Kronos-small`, `Kronos-base`; `Kronos-large` is not open-source)
- `KronosPredictor(model, tokenizer, max_context=512)`
- `predictor.predict(df, x_timestamp, y_timestamp, pred_len, T, top_p, sample_count)` - returns a DataFrame indexed by `y_timestamp`
- `predictor.predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len, ...)` - all series must share the same lookback length and `pred_len`

**Facts from the README that shape the design.**

- Input DataFrame must have `open, high, low, close`; `volume` and `amount` are optional.
- Maximum context is 512 bars for `Kronos-small` and `Kronos-base`, 2048 for `Kronos-mini`. Longer input is truncated automatically.
- Python 3.10+. Dependencies are in `requirements.txt` (includes `torch>=2.0.0` and pinned `pandas==2.2.2`, `huggingface_hub==0.33.1`).
- Weights download from the Hugging Face Hub.
- License: MIT (code).
- The README's own disclaimer: the fine-tuning/backtest pipeline is "a simplified example and not a production-ready quantitative trading system".

**Not stated in the README.**

- A pip-installable package. The README imports `from model import ...`, which only works with the repo folder on the Python path.
- The end date of the pre-training data (matters for look-ahead, see section 4.1).
- Which assets, exchanges or bar intervals the model performs well on, beyond "trained on data from over 45 global exchanges".
- Forecast accuracy figures.
- Inference speed and hardware needs. (`webui/README.md` lists CPU, CUDA and MPS as supported devices; the code auto-selects one.)
- How to make sampling reproducible (no seed parameter is documented).
- License of the model weights on Hugging Face.

**Not used.** The `finetune/` pipeline (needs Qlib and multi-GPU `torchrun`), the Flask `webui/`, and `examples/run_backtest_kronos.py` (present in the repo but not described in the README).

### 1.2 DSPy - prompt optimization

**What we use it for.** The strategy proposer is an LLM step: given a summary of recent market behaviour and a list of ideas that already failed, it proposes new strategy candidates. DSPy defines that step as a program and then improves its instructions between rounds, using backtest results as the score.

**Entry points we will call.**

- `dspy.LM("provider/model")` and `dspy.configure(lm=...)` - connect to a language model. Anthropic is a native provider, configured with `ANTHROPIC_API_KEY` (installation doc).
- `dspy.Signature` and a `dspy.Module` - define the proposer's inputs and outputs.
- A metric function with signature `(gold, pred, trace=None, pred_name=None, pred_trace=None)` returning `dspy.Prediction(score, feedback)`.
- `dspy.GEPA(metric, max_metric_calls=..., reflection_lm=...)` then `.compile(student, trainset=..., valset=...)` - the optimizer.
- `dspy.Evaluate(devset=..., metric=...)` - score a program without optimizing it.
- `program.save("path.json")` and `program.load("path.json")` - keep the optimized prompt between rounds.

**Why GEPA.** The docs say GEPA is the only optimizer that reads text feedback from the metric. Our risk gate produces exactly that ("rejected: 4 trades, below the minimum" or "rejected: profit disappears when fees double"). The other instruction optimizers only see a number.

**Facts from the docs that shape the design.**

- `.compile()` returns a new program; the original is not changed.
- Only one of `auto`, `max_full_evals`, `max_metric_calls` may be set. We will always set `max_metric_calls` so every round has a hard budget.
- Cost warning from the docs: "GEPA and MIPROv2 can spend hundreds of dollars in LM calls on a single run." Their example: `auto="light"` on a 2-predictor, 100-example task is about 1,330 metric calls.
- `valset` defaults to `trainset` if not given. We must always pass a separate `valset`.
- The docs note that tuning demos "tends to overfit" while tuning instructions "tends to generalize". GEPA tunes instructions only.
- JSON state files load freely. Pickle files need `allow_pickle=True` and can run code on load. We will use JSON only.
- Install: `pip install dspy`. Python `>=3.10, <3.16` (`pyproject.toml`).

**Not stated in the README.** The README itself only gives the install command, the description and paper links; everything above comes from the `docs/` folder. Not stated anywhere I read: a dollar cost per run for a specific model, and any guidance on financial or time-series use.

### 1.3 mem0 - memory of failed ideas

**What we use it for.** Every strategy the risk gate rejects is written to memory with the reason. Before the proposer generates new candidates, we search memory for similar past failures and pass them in, so the system stops re-proposing ideas that already failed.

**Entry points we will call** (open-source library mode, runs locally):

- `from mem0 import Memory`
- `Memory.from_config(config)` - with our own LLM, embedder and vector-store settings
- `memory.add(text_or_messages, user_id=..., metadata={...}, infer=False)` - store a failure record exactly as written
- `memory.search(query, filters={"user_id": ...}, top_k=...)` - returns `{"results": [{"memory": ..., "score": ...}, ...]}`

**Facts from the README and docs that shape the design.**

- Three deployment modes: library (`pip install mem0ai`), self-hosted server (Docker), cloud platform. We use the library only.
- Defaults require an OpenAI key: `gpt-5-mini` for fact extraction and `text-embedding-3-small` for embeddings.
- Other providers are supported. LLM providers include `anthropic` and `ollama`; embedders include `huggingface` (local `sentence-transformers`, Python only) and `ollama`.
- Default storage is Qdrant on disk at `/tmp/qdrant` plus SQLite history at `~/.mem0/history.db`. We must set our own path so memory survives restarts.
- `infer=False` stores the payload as given with no LLM call. The docs warn that duplicates can land and that mixing `infer=True` and `infer=False` for the same fact can save it twice. We will use `infer=False` consistently, because our failure records are already structured.
- The April 2026 algorithm is "ADD-only": memories accumulate and nothing is overwritten. That suits a log of failures.
- Python 3.10+. License: Apache 2.0.

**Not stated in the README.**

- Whether a fully offline setup (local embedder, `infer=False`, no LLM key at all) works without any API key. The Hugging Face embedder example in the docs still sets `OPENAI_API_KEY` "for LLM". This must be tested before we rely on it.
- Storage size or performance limits for the local Qdrant mode.
- Published benchmark scores are for the managed platform; the README says open-source users "should expect directionally similar gains but not identical numbers".

**Not used.** The cloud platform, the `mem0` CLI (its `mem0 init --agent` command creates a cloud account and API key), the self-hosted server, and the agent skills.

---

## 2. How data moves

```
                          +--------------------------------------+
                          |  Owner picks asset + bar interval    |
                          +------------------+-------------------+
                                             |
                                             v
                               [1] DATA FETCH (public, keyless)
                               OHLCV bars -> local cache on disk
                                             |
             +-------------------------------+--------------------------------+
             |                               |                                |
             v                               v                                v
       TRAIN window                  VALIDATION window                 HOLDOUT window
   (proposer + optimizer             (risk gate judges                (locked; used once
    may look at this)                 candidates here)                 per promoted strategy)
             |
             v
   [2] KRONOS FORECAST
   at bar t, uses only bars <= t
   -> forecast cache (per asset, interval, bar, model, settings)
             |
             v
   [3] STRATEGY PROPOSER (DSPy program, LLM call)
   inputs : market summary of TRAIN window,
            failed ideas recalled from memory  <-----------------------+
   output : strategy candidates as a filled-in                         |
            form (rule template + numbers), not code                   |
             |                                                         |
             v                                                         |
   [4] BACKTEST ENGINE                                                 |
   bar by bar, decision at close of bar t,                             |
   fill at open of bar t+1, fees + slippage                            |
             |                                                         |
             v                                                         |
   [5] RISK GATE (fixed rules, no LLM)                                 |
   checks on VALIDATION window: enough trades,                         |
   max drawdown, beats buy-and-hold after costs,                       |
   still passes with doubled costs                                     |
             |                                                         |
       +-----+----------------------+                                  |
       |                            |                                  |
     FAIL                         PASS                                 |
       |                            |                                  |
       v                            v                                  |
   [6] MEMORY (mem0)          one-time HOLDOUT check                   |
   store idea + reason              |                                  |
   for failure  --------------------+---- fail -> also stored ---------+
       |                            |
       |                          pass
       |                            v
       |                    [8] PAPER TRADING LEDGER
       |                    simulated orders on new bars
       |                    as they arrive; no real orders
       |                            |
       v                            v
   [7] PROMPT OPTIMIZATION (DSPy GEPA)            [9] DASHBOARD (Streamlit)
   score + gate feedback text for each            bars, forecasts, candidates,
   candidate -> improved proposer                 gate results, memory, paper P&L,
   instructions, saved as JSON                    model-call spend per round
       |
       v
   NEXT ROUND: go back to [3] with the improved
   proposer and the larger memory of failures
```

Two points the diagram depends on:

- **The "self-improving" part is steps 6 and 7.** Memory stops repeats. Prompt optimization changes how the proposer thinks. Nothing else learns.
- **Poor paper-trading results also feed back.** A promoted strategy that performs badly on new bars is written to memory as a failure, the same as a gate rejection.

---

## 3. Files we will create

All paths are relative to the repo root. The six files named in the request are marked **(requested)**. The others are additions I think the design needs; each is flagged so the owner can cut it.

| File | What it does |
|---|---|
| `terminal/data_fetch.py` **(requested: data fetch)** | Downloads OHLCV bars for one asset and interval from a public keyless source or reads a CSV, validates them (no gaps, no NaN, sorted), and caches them on disk. |
| `terminal/engine.py` **(requested: engine)** | Backtest engine: walks bars in time order, applies a strategy, fills at the next bar's open, charges fees and slippage, returns trades and an equity curve. |
| `terminal/gate.py` **(requested: gate)** | Risk gate: applies fixed pass/fail rules to a backtest result and returns a verdict plus a plain-English reason. No LLM involved. |
| `terminal/memory.py` **(requested: memory)** | Thin wrapper around mem0: `remember_failure(...)` and `recall_similar_failures(...)`, with storage in a project folder. |
| `terminal/optimizer.py` **(requested: optimizer)** | Runs DSPy GEPA on the proposer with a hard call budget, using gate verdicts as score and feedback; saves and loads the optimized proposer as JSON. |
| `app.py` plus `terminal/dashboard.py` **(requested: dashboard)** | Streamlit pages (`render_<name>()` functions registered in `PAGES`) showing bars, forecasts, candidates, gate results, memory contents, paper P&L and spend. Chart-building helpers live in `dashboard.py`. |
| `terminal/forecast.py` (addition) | Wraps Kronos: builds inputs from bars up to time t only, calls `KronosPredictor`, caches forecasts so the same bar is never forecast twice. |
| `terminal/strategies.py` (addition) | Defines the fixed menu of strategy templates and the form a candidate must fill in; turns a filled-in form into a function the engine can run. |
| `terminal/proposer.py` (addition) | The DSPy signature and module that asks the LLM for strategy candidates. Kept separate from `optimizer.py` so it can run without optimizing. |
| `terminal/paper.py` (addition) | Paper trading ledger: simulated cash, positions and orders for promoted strategies on newly arrived bars, saved to a local file. |
| `terminal/loop.py` (addition) | Runs one full round (steps 3 to 7 in the diagram) and enforces the per-round budget. |
| `terminal/splits.py` (addition) | The single place that cuts bars into train, validation and holdout windows, and records how many times the holdout has been used. |
| `terminal/config.py` (exists) | Gains settings for asset, interval, costs, gate thresholds, model names and budgets. |
| `tests/test_*.py` (addition) | One test file per module. The most important ones: a look-ahead test for `engine.py` and `forecast.py`, and gate threshold tests. |
| `requirements.txt` (exists) | Gains `dspy`, `mem0ai`, `torch` and the Kronos dependencies, with versions pinned after a compatibility check. |
| `.env.example` (exists) | Gains the LLM key name(s). No exchange or broker keys, ever. |

Kronos is not a pip package (not stated in its README), so `forecast.py` needs the Kronos `model/` folder on the Python path. How we do that is open question 7.

---

## 4. Risks

### 4.1 Look-ahead leaks

| Leak | How it happens | Planned control |
|---|---|---|
| Same-bar fill | Strategy decides using bar t's close and also trades at bar t's close. | Engine fills at bar t+1's open. A test feeds the engine a strategy that "knows" the next close and asserts it cannot profit. |
| Forecast built from future bars | Kronos is given a window that extends past the decision bar. | `forecast.py` accepts a decision timestamp and slices bars `<= t` itself. Callers cannot pass a window directly. |
| **Kronos pre-training overlap** | The model was pre-trained on historical data "from over 45 global exchanges". The end date of that data is not stated. If our backtest period falls inside it, the model may have already seen those bars. | Treat Kronos-based backtest results as optimistic. Only forward paper trading on bars newer than the model release counts as clean evidence. Ask the owner whether to check the paper for a cutoff date (open question 8). |
| Normalization across the split | Scaling or summary statistics computed over the whole series. | All statistics are computed inside the window being used. `splits.py` is the only place windows are cut. |
| Proposer sees validation data | The market summary given to the LLM includes validation or holdout bars. | The summary is built from the train window only. |
| LLM background knowledge | The language model may know what happened to a well-known asset in a past period. | Cannot be fully removed. Mitigation: the proposer is not told the asset name or dates, only anonymized statistics. Forward paper trading is the real test. |
| Revised or adjusted data | Free data sources revise bars or back-adjust for splits and dividends. | Cache bars with the fetch date; never silently re-download history that has already been used in a backtest. |

### 4.2 Overfitting

- **Many tries, one winner.** Each round proposes several candidates and there are many rounds. With enough tries, something passes the gate by luck. Controls: a minimum trade count, a doubled-cost stress check, a requirement to beat buy-and-hold, and a count of total candidates tested shown on the dashboard beside every "pass".
- **The optimizer overfits the validation window.** GEPA is rewarded for gate passes on validation data, so over many rounds the proposer's instructions drift toward what worked in that one period. Controls: the holdout window is never shown to the optimizer; validation windows roll forward between rounds.
- **The holdout wears out.** Every strategy checked against the holdout uses it up a little. Control: `splits.py` logs each use; the dashboard shows the count; after an agreed number of uses the holdout is retired and replaced by newer bars.
- **Memory bias.** A good idea rejected once for a bad reason (for example a thin-data period) is then avoided forever. Control: store the market conditions with each failure and let the proposer see them, so "failed in a flat market" is not read as "always fails".
- **One asset, one period.** A strategy tuned to a single asset is a description of that asset's past. Control: optional check on a second asset before promotion (open question 5).
- **Gate thresholds are program decisions.** The numbers (minimum trades, maximum drawdown, cost assumptions) are TBD and must come from the owner, not from me (open question 4).

### 4.3 API keys and secrets

- No exchange or broker keys exist anywhere in this design. The data source must work without a key; if the chosen source needs one, that goes back to the owner as a decision.
- The only secrets are LLM keys (and an embedding key if a hosted embedder is chosen). They live in `.env`, which is already git-ignored.
- mem0's defaults silently need an OpenAI key. We will set the LLM and embedder explicitly so nothing falls back to a provider the owner did not choose.
- mem0's cloud platform and CLI would send strategy ideas to a third-party service. We do not use them.
- DSPy saved state can include LM settings. We save JSON state only, and check that no key is written into it before the file is ever committed.
- Kronos weights and any local embedding model are downloaded from the Hugging Face Hub. No key is needed for public models, but it is an outbound download of code-adjacent files; versions will be pinned.
- The proposer returns a filled-in form, not Python code. We never execute text written by the LLM.

### 4.4 Cost of model calls

- **GEPA is the main cost.** Each metric call means one proposer LLM call plus a backtest. The DSPy docs warn of "hundreds of dollars" for a single unbounded run. Control: always set `max_metric_calls`, cap rounds per day, and show spend per round on the dashboard. Actual price per round depends on the model chosen; pricing will be looked up at build time, not guessed here.
- **Two models, two budgets.** The docs recommend a strong model for reflection (rare) and a cheap one for the task (frequent). We follow that.
- **Kronos compute.** A walk-forward backtest needs one forecast per decision bar. Inference speed is not stated. Controls: cache every forecast, start with `Kronos-mini` or `Kronos-small`, use `predict_batch`, and measure before choosing the bar interval.
- **mem0 calls.** With `infer=False` there is no LLM call on write. Search needs one embedding call per query; a local embedder makes that free.
- **Runaway loop.** A self-improving loop with no stop condition keeps spending. Control: the loop runs a fixed number of rounds, started by the owner from the dashboard. It does not run unattended.

### 4.5 Other risks

- **Dependency conflicts.** Kronos pins `pandas==2.2.2` and `huggingface_hub==0.33.1`; Streamlit, DSPy and mem0 have their own requirements. This must be resolved in one environment before any feature work.
- **Reproducibility.** Kronos samples its forecasts (`T`, `top_p`, `sample_count`) and LLM outputs vary. Results will differ run to run unless forecasts are cached and seeds are fixed where possible.
- **"Any asset" is broad.** Stocks trade in sessions with gaps; crypto trades around the clock; futures roll. Kronos uses time-of-day and weekday features. Each asset class needs its own data checks.
- **Paper is not live.** A paper ledger has no real liquidity, queueing or outages. Paper results are an upper bound.
- **Not financial advice.** The system produces research output for the owner's own judgement.

---

## 5. Open questions for the owner

1. **First asset and bar interval.** Which asset do you want to start with, and at what interval (for example daily, hourly, 5-minute)? This decides the data source and how heavy Kronos inference is.
2. **Data source.** Are you content with a public keyless source chosen after I research current options, or do you want to supply CSV files? Recommendation: support CSV first (always works, no terms-of-service question), add one keyless source second.
3. **LLM provider and spend cap.** Which provider do you want for the proposer and for GEPA's reflection step, and what is the most you are willing to spend per round and per month? Recommendation: Anthropic for both, a cheaper model for the proposer and a stronger one for reflection.
4. **Risk gate thresholds.** What are your numbers for minimum trades, maximum drawdown, fee and slippage assumptions, and starting paper capital? I will not invent these.
5. **Long only, or long and short?** And is leverage ever allowed in the paper ledger? Recommendation: long-only, no leverage for the first version.
6. **Memory setup.** Options: (a) fully local - Hugging Face embedder, `infer=False`, no extra key; (b) hosted embeddings with a key. Recommendation: (a), provided a test confirms mem0 runs with no LLM key (not stated in its docs).
7. **How to include Kronos.** Options: (a) keep the clone in `./repos` and point Python at it; (b) copy its `model/` folder into this project under its MIT license; (c) add it as a git submodule. Recommendation: (b), pinned to commit `67b630e`, because it survives a fresh checkout and needs no extra setup step.
8. **Kronos training cutoff.** Do you want me to read the Kronos paper (arXiv 2508.02739) to find the pre-training data end date, so we know which backtest periods are contaminated?
9. **Where does this run?** Kronos needs PyTorch and benefits from a GPU. Will this run on your own machine (which one?), or only in cloud sessions like this one?
10. **Loop control.** Should a round only ever start when you press a button, or do you want a scheduled mode later? Recommendation: button only for the first version.
11. **Build order.** Recommendation: (1) data fetch, engine and look-ahead tests; (2) gate and dashboard with hand-written strategies; (3) Kronos forecast; (4) proposer; (5) memory; (6) optimizer; (7) paper ledger. Each stage is usable on its own. Say so if you want a different order.

If you are unsure on questions 3, 4 or 6, say "deep dive" on that number and I will walk through the options with you.
