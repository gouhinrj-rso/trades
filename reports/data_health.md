# Data health report

- Generated: 2026-11-14T14:13:34.228128+00:00
- Source: yfinance (research only)
- Calendar: NYSE
- Requested: 2003-01-02 to 2026-11-13 (6006 sessions)
- Result: BLOCKED: BIL

## Windows

| Window | First session | Last session | Sessions |
|---|---|---|---|
| train | 2003-01-02 | 2016-12-30 | 3525 |
| validation | 2017-01-03 | 2021-12-31 | 1259 |
| holdout (LOCKED) | 2022-01-03 | 2026-11-13 | 1222 |

## Symbols

| Symbol | Status | Rows | Expected | First | Last | Starts late | Dividend days | Split days | Adjustment |
|---|---|---|---|---|---|---|---|---|---|
| SPY | OK | 6006 | 6006 | 2003-01-02 | 2026-11-13 | no | 96 | 0 | split_adjusted |
| QQQ | OK | 6006 | 6006 | 2003-01-02 | 2026-11-13 | no | 90 | 0 | split_adjusted |
| IWM | OK | 6006 | 6006 | 2003-01-02 | 2026-11-13 | no | 96 | 1 | split_adjusted |
| EFA | OK | 6006 | 6006 | 2003-01-02 | 2026-11-13 | no | 44 | 1 | split_adjusted |
| EEM | OK | 5936 | 5936 | 2003-04-14 | 2026-11-13 | yes | 47 | 2 | split_adjusted |
| TLT | OK | 6006 | 6006 | 2003-01-02 | 2026-11-13 | no | 285 | 0 | split_adjusted |
| IEF | OK | 6006 | 6006 | 2003-01-02 | 2026-11-13 | no | 287 | 0 | split_adjusted |
| GLD | OK | 5532 | 5532 | 2004-11-18 | 2026-11-13 | yes | 0 | 0 | split_adjusted |
| DBC | OK | 5227 | 5227 | 2006-02-06 | 2026-11-13 | yes | 9 | 0 | split_adjusted |
| VNQ | OK | 5568 | 5568 | 2004-09-29 | 2026-11-13 | yes | 88 | 0 | split_adjusted |
| SHV | OK | 4993 | 4993 | 2007-01-11 | 2026-11-13 | yes | 185 | 0 | split_adjusted |
| BIL | BLOCKED | 4898 | 4898 | 2007-05-30 | 2026-11-13 | yes | 129 | 1 | split_adjusted |

## Problems

- **BIL** `stale_repeat`: 1 (identical to the previous bar); examples: 2014-07-28

Checks run: empty, missing_value, non_positive, negative_volume, bad_range, duplicate, out_of_order, off_calendar, gap, stale_end, stale_repeat, zero_volume, unadjusted_split, future_bar.
