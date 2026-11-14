"""Follow-up numbers for the audit of baseline B1 (reports/audit.md).

Run from the repo root:  python experiments/audit_addendum.py

It does one new engine run on real data (the frozen baseline at 2x costs, to
compare with 60/40 in validation) and LOGS it in experiments/trials.csv.
Everything else is arithmetic on files already saved. Train and validation
only; the holdout is not loaded. No strategy parameter is changed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import sessions as cal                                    # noqa: E402
from src.data.factors import load_factors                               # noqa: E402
from src.reporting import metrics                                       # noqa: E402
from src.research import robustness, run_baseline, trial_log            # noqa: E402
from src.research.backtest import Costs, run_backtest                   # noqa: E402
from src.strategy.momentum import Params                                # noqa: E402

config = yaml.safe_load(open(ROOT / "config" / "paper.yaml"))
bars, cash_vehicle, _ = run_baseline.load_market(config)                # train + validation only
END = pd.Timestamp(config["data"]["validation_end"])
sessions = cal.expected_sessions("NYSE", config["data"]["start"], END + pd.Timedelta(days=10))
factors = load_factors(END, cache_dir=ROOT / "data" / "cache" / "factors")
windows = run_baseline.window_slices(config, END, unlocked=False)
VAL, BOTH = windows["validation"], windows["train_plus_validation"]

# ---- A6: days where the risk-free rate was missing and was counted as 0 -------------------
run_days = sessions[(sessions >= BOTH[0]) & (sessions <= BOTH[1])]
missing_rf = int(factors["rf"].reindex(run_days).isna().sum())
print(f"A6 risk-free rate missing on {missing_rf} of {len(run_days)} run days (those days were counted as 0)")
rf = factors["rf"].reindex(sessions).fillna(0.0)

# ---- B2: the frozen baseline at 2x costs, validation Sharpe against 60/40 (ONE LOGGED RUN) ---
curve = pd.read_csv(ROOT / "reports/baseline/equity_curve.csv", parse_dates=["session"]).set_index("session")
def returns_of(equity):
    r = equity.pct_change(); r.iloc[0] = equity.iloc[0] / 10000.0 - 1.0; return r
r_strategy, r_6040 = returns_of(curve["strategy_equity"]), returns_of(curve["60_40_SPY_IEF_equity"])
base = Costs(float(config["costs"]["fee_bps_per_side"]), float(config["costs"]["slippage_bps"]))
params = Params()
universe = [s for s in config["universe"] if s in bars]
result = run_backtest({s: bars[s] for s in universe + [cash_vehicle]}, sessions,
                      run_baseline.strategy_fn(params), base.scaled(2), 10000.0, BOTH[0], BOTH[1], cash_vehicle=cash_vehicle)
r2 = result.returns
excess2 = r2 - rf.reindex(r2.index)
trial_log.append_trial({
    "run_group": "audit", "run_kind": "audit", "strategy": "B1_mom12_sma200", "params_json": params.to_dict(),
    "cost_multiple": 2, "commission_bps": base.commission_bps * 2, "slippage_bps": base.slippage_bps * 2,
    "window_start": f"{BOTH[0]:%Y-%m-%d}", "window_end": f"{BOTH[1]:%Y-%m-%d}", "holdout_unlocked": "no",
    "n_days": len(r2), "total_return": float((1 + r2).prod() - 1),
    "sharpe_annual": float(metrics.sharpe(r2, rf.reindex(r2.index))),
    "sharpe_per_day": float(excess2.mean() / excess2.std(ddof=1)), "status": "ok",
    "notes": "audit follow-up B2: repeat of T0003 to get validation Sharpe at 2x costs"})
cut = run_baseline.cut
s2 = metrics.sharpe(cut(r2, VAL), rf.reindex(cut(r2, VAL).index))
s6 = metrics.sharpe(cut(r_6040, VAL), rf.reindex(cut(r_6040, VAL).index))
print(f"B2 validation Sharpe: strategy at 2x costs {s2:.3f} vs 60/40 at 1x costs {s6:.3f} -> gate 6 would {'PASS' if s2 > s6 else 'FAIL'}")

# ---- A5: expectancy bootstrap over all 204 months, empty months included ---------------------
trips = pd.read_csv(ROOT / "reports/baseline/round_trips.csv", parse_dates=["entry_session", "exit_session"])
closed = trips[~trips["open"]]
months = pd.period_range(BOTH[0], BOTH[1], freq="M")
by_month = closed.groupby(closed["exit_session"].dt.to_period("M"))["r_multiple"].apply(lambda v: v.to_numpy())
groups = [by_month.get(m, np.array([])) for m in months]
ci = robustness.circular_block_bootstrap_ci(groups, lambda v: float(np.mean(v)) if len(v) else float("nan"),
                                            3, 2000, 0.95, 20261114)
print(f"A5 expectancy, all {len(months)} months resampled in blocks of 3: {ci['point']:+.3f}R, "
      f"95% interval {ci['lower']:+.3f} to {ci['upper']:+.3f} (reported: +0.616 to +2.232 over 77 months with exits)")

# ---- B6: factor regression with a bond factor added ----------------------------------------------
r = cut(r_strategy, BOTH)
x_rf = rf.reindex(r.index)
design = pd.DataFrame({
    "spy_excess": bars["SPY"].set_index("session")["tr_close"].pct_change().reindex(r.index) - x_rf,
    "momentum_factor": factors["mom"].reindex(r.index),
    "ief_excess": bars["IEF"].set_index("session")["tr_close"].pct_change().reindex(r.index) - x_rf})
for label, columns in [("as reported (SPY + momentum)", ["spy_excess", "momentum_factor"]),
                       ("with IEF added", ["spy_excess", "momentum_factor", "ief_excess"])]:
    fit = robustness.ols_newey_west(r - x_rf, design[columns], 21)
    betas = ", ".join(f"{k} {v:.2f}" for k, v in fit["betas"].items())
    print(f"B6 {label}: alpha {fit['alpha_annualized']:+.2%} a year, t = {fit['alpha_t']:.2f}; betas {betas}; R2 {fit['r_squared']:.2f}")

# ---- B3: PSR of the strategy against 60/40's Sharpe, and DSR at the corrected trial count ----------
excess = (r - x_rf).to_numpy()
e6 = (cut(r_6040, BOTH) - x_rf).to_numpy()
print(f"B3 PSR against 60/40's per-day Sharpe ({e6.mean() / e6.std(ddof=1):.4f}): "
      f"{robustness.probabilistic_sharpe_ratio(excess, e6.mean() / e6.std(ddof=1)):.3f}")
n_rows = trial_log.count_trials()
sr = excess.mean() / excess.std(ddof=1)
skew = ((excess - excess.mean()) ** 3).mean() / excess.std() ** 3
kurt = ((excess - excess.mean()) ** 4).mean() / excess.std() ** 4
sampling_sd = float(np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (len(excess) - 1)))
for label, spread in [("protocol spread (9 neighbour cells)", 0.0029555), ("sampling error of one Sharpe", sampling_sd)]:
    d = robustness.deflated_sharpe_ratio(excess, n_rows, spread)
    print(f"B3 DSR with N = {n_rows} rows, {label}: {d['dsr']:.3f} (Sharpe to beat {d['expected_max_sharpe'] * 252 ** 0.5:.2f} annual)")
