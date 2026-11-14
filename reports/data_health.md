# Data health report

- Generated: 2026-11-14T14:05:41.164313+00:00
- Source: yfinance (research only)
- Calendar: NYSE
- Requested: 2016-11-14 to 2026-11-13 (2514 sessions)
- Result: PASS - every symbol cached

## Windows

| Window | First session | Last session | Sessions |
|---|---|---|---|
| train | 2016-11-14 | 2022-11-09 | 1508 |
| validation | 2022-11-10 | 2024-11-11 | 503 |
| holdout (LOCKED) | 2024-11-12 | 2026-11-13 | 503 |

## Symbols

| Symbol | Status | Rows | Expected | First | Last | Dividend days | Split days | Adjustment |
|---|---|---|---|---|---|---|---|---|
| SPY | OK | 2514 | 2514 | 2016-11-14 | 2026-11-13 | 40 | 0 | split_adjusted |
| QQQ | OK | 2514 | 2514 | 2016-11-14 | 2026-11-13 | 40 | 0 | split_adjusted |
| IWM | OK | 2514 | 2514 | 2016-11-14 | 2026-11-13 | 40 | 0 | split_adjusted |
| TLT | OK | 2514 | 2514 | 2016-11-14 | 2026-11-13 | 120 | 0 | split_adjusted |
| GLD | OK | 2514 | 2514 | 2016-11-14 | 2026-11-13 | 0 | 0 | split_adjusted |

## Problems

None. Every check passed for every symbol.

Checks run: empty, missing_value, non_positive, negative_volume, bad_range, duplicate, out_of_order, off_calendar, gap, stale_end, stale_repeat, zero_volume, unadjusted_split, future_bar.
