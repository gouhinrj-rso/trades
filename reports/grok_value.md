# Grok value report: does Market Scout earn its keep?

**Recommendation: INSUFFICIENT EVIDENCE.** No forward baseline-and-overlay run exists yet.

- Generated: 2026-11-14 16:35 UTC. Mode: **shadow** (the paper strategy ignores every veto).
- Event files ingested: 1 day(s), 2026-11-14 to 2026-11-14.
- Lines: 11 seen, 10 accepted, 1 rejected. Invalid-output rate: 9.1%.
- Vetoes received: 5. Vetoes on an instrument the strategy trades: **0**.
- **No veto can apply.** The Bot's vetoes are on CL, GC, HG, ZC; the strategy trades SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, VNQ. Either the Bot watches the traded instruments, or the owner fills `scout_ticker_map` in `config/paper.yaml`.
- Grok cost: **not recorded**. Add rows to `research/grok/costs.csv` (date, usd, note).

## Baseline against overlay

Not run. A forward comparison needs daily bars dated after the first event file, and a paper
strategy running on them. Neither exists yet (see "What is still needed").

## Rules this report follows

- A recommendation needs at least 63 trading days and 20 applied vetoes.
- A veto only removes a new entry. It never adds, enlarges or sells a position.
- An event counts from when it was ingested, so a file added later cannot veto an earlier decision.
- A missing, empty or stale event file means no veto. It can never cause a trade.
- A veto with no event time lapses at the close of trading day 2 after it was seen.

## What is still needed

1. A paper strategy running forward on new daily bars (PRD Prompt 8).
2. Vetoes on instruments that strategy trades (see above).
3. Grok spend entered in `research/grok/costs.csv`.
4. Time: 63 trading days and 20 applied vetoes.
