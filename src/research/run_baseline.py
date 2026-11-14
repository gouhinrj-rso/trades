"""Run the frozen baseline B1 and everything protocol.md says goes with it.

Run from the repo root:

    python -m src.research.run_baseline

One invocation does 14 backtests on train + validation data (2005-2021) and
appends 14 rows to experiments/trials.csv:

    1  baseline at 1x costs
    3  cost stress: the baseline at 0x, 2x and 3x costs
    8  parameter neighbours (lookback 9/12/15 months x trend average 150/200/250 days, minus the centre)
    2  benchmarks: SPY buy and hold, 60/40 SPY/IEF

and writes reports/baseline/ (equity curve, fills, round trips, rebalances,
metrics, robustness numbers). Then src/reporting/scorecard.py turns the
numbers into reports/scorecard.md.

NOTHING HERE OPTIMIZES. The neighbours are diagnostics; no result changes a
parameter. Every run is logged, including ones that crash.

The holdout (2022 onward) is not loaded. The owner - and only the owner - can add

    --unlock-holdout --who "Name" --reason "why"

which first writes a line to experiments/holdout_access.csv and then extends
the run into the holdout.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import holdout, pipeline
from src.data import sessions as cal
from src.data.factors import SOURCE_NAME, load_factors
from src.data.sources import SourceError
from src.reporting import metrics
from src.research import robustness, trial_log
from src.research.backtest import Costs, fixed_weights, prepare_bars, run_backtest
from src.strategy.momentum import Params, target_weights

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_DIR / "config" / "paper.yaml"
OUT_DIR = PROJECT_DIR / "reports" / "baseline"

# Fixed by protocol.md section 5. Not tunable.
NEIGHBOUR_LOOKBACKS = (9, 12, 15)
NEIGHBOUR_SMAS = (150, 200, 250)
COST_MULTIPLES = (0, 2, 3)
BOOTSTRAP_SEED = 20261114
BOOTSTRAP_RESAMPLES = 2000
SHARPE_BLOCK_DAYS = 21
EXPECTANCY_BLOCK_MONTHS = 3
PBO_BLOCKS = 16
WALK_FORWARD_FIRST_YEAR = 2010
NEWEY_WEST_LAGS = 21
REGIME_YEARS = (2008, 2020)        # 2022 is in the holdout and is not examined


def config_name(params: Params) -> str:
    """Short label for one parameter set, e.g. 'mom12_sma200'."""
    return f"mom{params.lookback_months}_sma{params.sma_days}"


def strategy_fn(params: Params):
    """The baseline's weight function for one parameter set."""
    return lambda histories: target_weights(histories, params)


def load_market(config: dict, access=None) -> tuple:
    """Load train + validation bars (and the holdout only with a logged access).

    Returns (bars, cash_vehicle, excluded). A symbol whose data was blocked is
    left out and listed in `excluded`; nothing is substituted for it.
    """
    windows = ["train", "validation"] + (["holdout"] if access is not None else [])

    def load(symbol):
        parts = [pipeline.load_bars(symbol, window, holdout_access=access if window == "holdout" else None)
                 for window in windows]
        return prepare_bars(pd.concat(parts, ignore_index=True))

    bars, excluded = {}, []
    for symbol in config["universe"]:
        try:
            bars[symbol] = load(symbol)
        except pipeline.DataBlocked as error:
            excluded.append({"symbol": symbol, "reason": str(error)})
    cash_vehicle = None
    for symbol in config.get("cash_vehicles") or []:
        try:
            bars[symbol] = load(symbol)
            cash_vehicle = symbol
            break                       # the first cash vehicle with healthy data is used
        except pipeline.DataBlocked as error:
            excluded.append({"symbol": symbol, "reason": str(error), "cash_vehicle": True})
    return bars, cash_vehicle, excluded


def window_slices(config: dict, last_session, unlocked: bool) -> dict:
    """The reporting windows as (first day, last day)."""
    data = config["data"]
    windows = {"train": (pd.Timestamp(config["baseline"]["trade_start"]), pd.Timestamp(data["train_end"])),
               "validation": (pd.Timestamp(data["train_end"]) + pd.Timedelta(days=1), pd.Timestamp(data["validation_end"]))}
    windows["train_plus_validation"] = (windows["train"][0], windows["validation"][1])
    if unlocked:
        windows["holdout"] = (pd.Timestamp(data["validation_end"]) + pd.Timedelta(days=1), pd.Timestamp(last_session))
    return windows


def cut(series, window):
    """The part of a date-indexed Series or DataFrame inside a window (both ends included)."""
    return series[(series.index >= window[0]) & (series.index <= window[1])]


def equity_from_returns(returns: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Growth of 1 unit through `returns`, with a starting point of 1.0 on the trading day before."""
    before = sessions[sessions < returns.index[0]][-1]
    curve = (1.0 + returns).cumprod()
    return pd.concat([pd.Series([1.0], index=[before]), curve])


def window_metrics(result, window, rf: pd.Series, sessions) -> dict:
    """Every metric for one run inside one window."""
    returns = cut(result.returns, window)
    if returns.empty:
        return {"n_days": 0}
    daily = cut(result.daily, window)
    trips = result.round_trips
    closed = trips[(~trips["open"]) & (trips["exit_session"] >= window[0]) & (trips["exit_session"] <= window[1])] \
        if not trips.empty else trips
    summary = metrics.summarize(
        returns, equity_from_returns(returns, sessions), rf=rf.reindex(returns.index).fillna(0.0),
        trades=closed[["pnl", "risk"]] if not closed.empty else None,
        risk_weight=daily["risk_weight"], traded_notional=daily["traded_notional"])
    # Turnover needs dollars, so recompute it from the account's own equity.
    summary["turnover"] = float(metrics.turnover(daily["traded_notional"], daily["equity"]))
    summary["n_closed_trades"] = int(len(closed))
    summary.pop("monthly_returns", None)
    return summary


def year_stats(returns: pd.Series, year: int, sessions) -> dict:
    """Return and worst drawdown inside one calendar year."""
    part = returns[returns.index.year == year]
    if part.empty:
        return {"year": year, "available": False}
    curve = equity_from_returns(part, sessions)
    return {"year": year, "available": True, "return": float(curve.iloc[-1] - 1.0),
            "max_drawdown": float(metrics.max_drawdown(curve)["depth"])}


def run_everything(bars: dict, sessions: pd.DatetimeIndex, config: dict, factors: pd.DataFrame | None,
                   windows: dict, cash_vehicle: str | None, out_dir: Path, trials_file: Path | None = None,
                   holdout_unlocked: bool = False, excluded: list | None = None, rf_source: str = "") -> dict:
    """Do the 14 runs, log each, compute every number, write the output files. Returns the numbers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_group = uuid.uuid4().hex[:8]
    base_costs = Costs(float(config["costs"]["fee_bps_per_side"]), float(config["costs"]["slippage_bps"]))
    starting_cash = float(config["account"]["starting_cash"])
    full = windows["holdout"] if holdout_unlocked else windows["train_plus_validation"]
    run_window = (windows["train"][0], full[1])
    report_window = windows["train_plus_validation"]            # every robustness test uses this
    frozen = config["baseline"]
    baseline_params = Params(frozen["momentum_lookback_months"], frozen["momentum_skip_months"],
                             frozen["trend_sma_days"], frozen["volatility_days"])
    universe = [symbol for symbol in config["universe"] if symbol in bars]

    # Risk-free rate: Ken French daily RF, or the cash vehicle's own return if that file is missing.
    if factors is not None:
        rf = factors["rf"]
    elif cash_vehicle:
        rf = bars[cash_vehicle].set_index("session")["tr_close"].pct_change()
        rf_source = f"daily return of {cash_vehicle} (Ken French RF could not be fetched); 0 before it has data"
    else:
        rf = pd.Series(dtype=float)
        rf_source = "0 (no risk-free series available)"
    rf = rf.reindex(sessions).fillna(0.0)

    def one_run(kind: str, name: str, weights_fn, costs: Costs, cost_multiple: float, params: dict,
                symbols: list, vehicle, rebalance: str = "monthly", notes: str = ""):
        """Run one backtest and append its row to the trial log - also when it fails."""
        row = {"run_group": run_group, "run_kind": kind, "strategy": name, "params_json": params,
               "cost_multiple": cost_multiple, "commission_bps": costs.commission_bps,
               "slippage_bps": costs.slippage_bps, "window_start": f"{run_window[0]:%Y-%m-%d}",
               "window_end": f"{run_window[1]:%Y-%m-%d}", "holdout_unlocked": "yes" if holdout_unlocked else "no",
               "notes": notes}
        try:
            subset = {symbol: bars[symbol] for symbol in symbols + ([vehicle] if vehicle else [])}
            result = run_backtest(subset, sessions, weights_fn, costs, starting_cash, run_window[0], run_window[1],
                                  cash_vehicle=vehicle, rebalance=rebalance)
            returns = cut(result.returns, report_window)
            curve = equity_from_returns(returns, sessions)
            excess = returns - rf.reindex(returns.index)
            row.update(n_days=len(returns), total_return=float(curve.iloc[-1] - 1.0), cagr=float(metrics.cagr(curve)),
                       sharpe_annual=float(metrics.sharpe(returns, rf.reindex(returns.index))),
                       sharpe_per_day=float(excess.mean() / excess.std(ddof=1)),
                       max_drawdown=float(metrics.max_drawdown(curve)["depth"]),
                       n_round_trips=int((~result.round_trips["open"]).sum()) if not result.round_trips.empty else 0,
                       status="ok")
            trial_log.append_trial(row, trials_file)
            return result
        except Exception as error:                       # a failed run is still a trial
            row.update(status="failed", failure_reason=f"{type(error).__name__}: {error}"[:300])
            trial_log.append_trial(row, trials_file)
            return None

    # ---- the 14 runs ---------------------------------------------------------------------
    runs = {"baseline": one_run("baseline", "B1_" + config_name(baseline_params), strategy_fn(baseline_params),
                                base_costs, 1, baseline_params.to_dict(), universe, cash_vehicle)}
    cost_runs = {1: runs["baseline"]}
    for multiple in COST_MULTIPLES:
        cost_runs[multiple] = one_run("cost_stress", "B1_" + config_name(baseline_params), strategy_fn(baseline_params),
                                      base_costs.scaled(multiple), multiple, baseline_params.to_dict(), universe,
                                      cash_vehicle, notes=f"baseline at {multiple}x costs")
    neighbours = {config_name(baseline_params): runs["baseline"]}
    for lookback in NEIGHBOUR_LOOKBACKS:
        for sma in NEIGHBOUR_SMAS:
            params = Params(lookback, baseline_params.skip_months, sma, baseline_params.vol_days)
            if params == baseline_params:
                continue
            neighbours[config_name(params)] = one_run(
                "neighbour", "B1_" + config_name(params), strategy_fn(params), base_costs, 1, params.to_dict(),
                universe, cash_vehicle, notes="diagnostic only; cannot replace the baseline")
    benchmarks = {
        "SPY_buy_and_hold": one_run("benchmark", "SPY_buy_and_hold", fixed_weights({"SPY": 1.0}), base_costs, 1,
                                    {"SPY": 1.0}, ["SPY"], None, rebalance="once"),
        "60_40_SPY_IEF": one_run("benchmark", "60_40_SPY_IEF", fixed_weights({"SPY": 0.6, "IEF": 0.4}), base_costs, 1,
                                 {"SPY": 0.6, "IEF": 0.4}, ["SPY", "IEF"], None),
    }

    numbers = {
        "run_group": run_group, "holdout_unlocked": holdout_unlocked, "universe_used": universe,
        "excluded": excluded or [], "cash_vehicle": cash_vehicle,
        "costs": {"commission_bps": base_costs.commission_bps, "slippage_bps": base_costs.slippage_bps},
        "risk_free": rf_source, "sharpe_frequency": "daily, annualized with sqrt(252)",
        "windows": {name: [f"{w[0]:%Y-%m-%d}", f"{w[1]:%Y-%m-%d}"] for name, w in windows.items()},
        "failed_runs": [name for name, result in {**{f"cost_{m}x": r for m, r in cost_runs.items()}, **neighbours,
                                                  **benchmarks}.items() if result is None],
    }
    baseline = runs["baseline"]
    if baseline is None:
        numbers["baseline_failed"] = True
        _write_json(out_dir / "metrics.json", numbers)
        return numbers

    # ---- metrics per window ----------------------------------------------------------------
    numbers["metrics"] = {name: window_metrics(baseline, window, rf, sessions) for name, window in windows.items()}
    numbers["benchmarks"] = {bench: {name: window_metrics(result, window, rf, sessions)
                                     for name, window in windows.items()}
                             for bench, result in benchmarks.items() if result is not None}
    numbers["cost_stress"] = {}
    for name, window in windows.items():
        cagrs, totals = {}, {}
        for multiple, result in cost_runs.items():
            if result is None:
                continue
            curve = equity_from_returns(cut(result.returns, window), sessions)
            cagrs[multiple], totals[multiple] = float(metrics.cagr(curve)), float(curve.iloc[-1] - 1.0)
            if multiple in (0, 1):
                numbers.setdefault("_curves", {}).setdefault(name, {})[multiple] = curve
        entry = {"cagr_by_multiple": cagrs, "total_return_by_multiple": totals}
        if 0 in cagrs and 1 in cagrs:
            entry["cost_drag"] = cagrs[0] - cagrs[1]
        if len(cagrs) >= 2:
            entry["breakeven_multiple"] = float(metrics.cost_breakeven_multiple(cagrs))
        numbers["cost_stress"][name] = entry
    numbers.pop("_curves", None)

    # ---- robustness, on train + validation only ----------------------------------------------
    returns = cut(baseline.returns, report_window)
    excess = (returns - rf.reindex(returns.index)).to_numpy()
    annual_sharpe = lambda values: float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(252)) \
        if np.std(values, ddof=1) > 0 else float("nan")
    robust = {"window": numbers["windows"]["train_plus_validation"]}
    robust["sharpe_ci"] = robustness.circular_block_bootstrap_ci(
        excess, annual_sharpe, SHARPE_BLOCK_DAYS, BOOTSTRAP_RESAMPLES, 0.95, BOOTSTRAP_SEED)

    trips = baseline.round_trips
    closed = trips[(~trips["open"]) & (trips["exit_session"] <= report_window[1])] if not trips.empty else trips
    robust["closed_trades"] = int(len(closed))
    if len(closed) >= 2:
        by_month = [group["r_multiple"].to_numpy() for _, group in
                    closed.groupby(pd.DatetimeIndex(closed["exit_session"]).to_period("M"))]
        robust["expectancy_ci"] = robustness.circular_block_bootstrap_ci(
            by_month, lambda values: float(np.mean(values)), EXPECTANCY_BLOCK_MONTHS, BOOTSTRAP_RESAMPLES, 0.95,
            BOOTSTRAP_SEED)
        robust["expectancy_ci"]["exit_months"] = len(by_month)
        # Concentration: how much of the total profit comes from the best few trades.
        ordered = closed["pnl"].sort_values(ascending=False)
        total = float(ordered.sum())
        robust["concentration"] = {
            "total_pnl_closed": total,
            "top_5_share": float(ordered.head(5).sum() / total) if total else None,
            "top_10pct_share": float(ordered.head(max(1, len(ordered) // 10)).sum() / total) if total else None,
            "pnl_without_top_5": float(ordered.iloc[5:].sum()),
            "by_symbol": {symbol: float(value) for symbol, value in closed.groupby("symbol")["pnl"].sum().items()}}

    # Deflated Sharpe: N = every row in the trial log; spread from baseline + neighbour rows at 1x.
    trials = trial_log.read_trials(trials_file)
    spread_rows = [float(row["sharpe_per_day"]) for row in trials
                   if row["run_kind"] in ("baseline", "neighbour") and row["status"] == "ok"
                   and float(row["cost_multiple"]) == 1.0]
    robust["n_trials"] = len(trials)
    robust["trial_sharpes_used_for_spread"] = len(spread_rows)
    robust["psr_vs_zero"] = float(robustness.probabilistic_sharpe_ratio(excess, 0.0))
    if len(spread_rows) >= 2:
        robust["trial_sharpe_std_per_day"] = float(np.std(spread_rows, ddof=1))
        robust["dsr"] = robustness.deflated_sharpe_ratio(excess, len(trials), robust["trial_sharpe_std_per_day"])

    # PBO and walk-forward over the neighbourhood.
    matrix = pd.DataFrame({name: cut(result.returns, report_window) for name, result in neighbours.items()
                           if result is not None})
    robust["neighbourhood"] = {}
    for name, result in neighbours.items():
        if result is None:
            robust["neighbourhood"][name] = None
            continue
        cell = {}
        for window_name in ("train", "validation", "train_plus_validation"):
            part = cut(result.returns, windows[window_name])
            cell[window_name] = {"sharpe": float(metrics.sharpe(part, rf.reindex(part.index))),
                                 "cagr": float(metrics.cagr(equity_from_returns(part, sessions)))}
        robust["neighbourhood"][name] = cell
    if matrix.shape[1] >= 2:
        pbo = robustness.pbo_cscv(matrix, PBO_BLOCKS)
        robust["pbo"] = {key: (float(value) if isinstance(value, (float, np.floating)) else value)
                         for key, value in pbo.items() if key != "logits"}
        forward = robustness.walk_forward_selection(matrix, config_name(baseline_params), WALK_FORWARD_FIRST_YEAR)
        forward.to_csv(out_dir / "walk_forward.csv", index=False)
        robust["walk_forward"] = {
            "years": int(len(forward)),
            "selected_compounded": float((1 + forward["selected_return"]).prod() - 1),
            "baseline_compounded": float((1 + forward["baseline_return"]).prod() - 1),
            "years_selected_beat_baseline": int((forward["selected_return"] > forward["baseline_return"]).sum()),
            "years_selected_was_baseline": int(forward["selected_is_baseline"].sum())}

    # Factor regression: strategy excess return on SPY excess return and the momentum factor.
    if factors is not None and "SPY" in bars:
        spy = bars["SPY"].set_index("session")["tr_close"].pct_change()
        design = pd.DataFrame({"spy_excess": spy.reindex(returns.index) - rf.reindex(returns.index),
                               "momentum_factor": factors["mom"].reindex(returns.index)})
        fit = robustness.ols_newey_west(returns - rf.reindex(returns.index), design, NEWEY_WEST_LAGS)
        robust["factor_regression"] = {key: (float(value) if isinstance(value, (float, np.floating)) else value)
                                       for key, value in fit.items()}
        robust["factor_regression"]["factor_source"] = SOURCE_NAME + " (US momentum factor)"
    else:
        robust["factor_regression"] = {"run": False, "reason": "factor file could not be fetched"}

    robust["regimes"] = {"strategy": [year_stats(baseline.returns, year, sessions) for year in REGIME_YEARS],
                         **{bench: [year_stats(result.returns, year, sessions) for year in REGIME_YEARS]
                            for bench, result in benchmarks.items() if result is not None},
                         "note": "2022 is in the locked holdout and was not examined"}
    # When each asset first became eligible (so the reader can see who was missing early on).
    eligible = baseline.rebalances[baseline.rebalances["eligible"] == True]            # noqa: E712
    robust["first_eligible"] = {symbol: f"{day:%Y-%m-%d}" for symbol, day in
                                eligible.groupby("symbol")["signal_session"].min().items()}
    numbers["robustness"] = robust

    # ---- files ---------------------------------------------------------------------------------
    curve = pd.DataFrame({"strategy_equity": baseline.daily["equity"]})
    for bench, result in benchmarks.items():
        if result is not None:
            curve[bench + "_equity"] = result.daily["equity"]
    curve["window"] = ["train" if day <= windows["train"][1] else
                       "validation" if day <= windows["validation"][1] else "holdout" for day in curve.index]
    curve.to_csv(out_dir / "equity_curve.csv", index_label="session")
    baseline.fills.to_csv(out_dir / "trades.csv", index=False)
    baseline.round_trips.to_csv(out_dir / "round_trips.csv", index=False)
    baseline.rebalances.to_csv(out_dir / "rebalances.csv", index=False)
    _write_json(out_dir / "metrics.json", numbers)
    return numbers


def _write_json(path: Path, numbers: dict) -> None:
    """Save numbers as JSON, turning numpy and pandas values into plain ones."""
    def plain(value):
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return [plain(item) for item in value.tolist()]
        if isinstance(value, (pd.Timestamp,)):
            return f"{value:%Y-%m-%d}"
        raise TypeError(f"cannot write {type(value)} to JSON")
    with open(path, "w") as handle:
        json.dump(numbers, handle, indent=2, default=plain)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen baseline B1 (protocol.md).")
    parser.add_argument("--unlock-holdout", action="store_true",
                        help="OWNER ONLY. Extends the run into the holdout and logs the access.")
    parser.add_argument("--who", default="", help="your name (required with --unlock-holdout)")
    parser.add_argument("--reason", default="", help="why the holdout is being opened (required with --unlock-holdout)")
    args = parser.parse_args(argv)

    with open(CONFIG_FILE) as handle:
        config = yaml.safe_load(handle)

    access = None
    if args.unlock_holdout:
        # Logs who / when / why FIRST; raises if the name or reason is missing.
        access = holdout.unlock(args.who, args.reason, command=" ".join(sys.argv))
        print(f"HOLDOUT UNLOCKED by {access.who} at {access.unlocked_at_utc}. Logged in experiments/holdout_access.csv.")

    bars, cash_vehicle, excluded = load_market(config, access)
    last_bar = max(table["session"].max() for table in bars.values())
    sessions = cal.expected_sessions(config["data"]["calendar"], config["data"]["start"],
                                     last_bar + pd.Timedelta(days=10))
    windows = window_slices(config, last_bar, access is not None)
    try:
        factors = load_factors(last_bar, cache_dir=PROJECT_DIR / config["data"]["cache_dir"] / "factors")
        rf_source = SOURCE_NAME + " RF (1-month Treasury bill)"
    except SourceError as error:
        print(f"WARNING: factor data unavailable ({error}). Falling back; the regression will not run.")
        factors, rf_source = None, ""

    numbers = run_everything(bars, sessions, config, factors, windows, cash_vehicle, OUT_DIR,
                             holdout_unlocked=access is not None, excluded=excluded, rf_source=rf_source)
    print(f"Run group {numbers['run_group']}: wrote {OUT_DIR.relative_to(PROJECT_DIR)}/ and appended to "
          f"experiments/trials.csv (now {trial_log.count_trials()} rows).")
    if numbers.get("failed_runs") or numbers.get("baseline_failed"):
        print(f"FAILED RUNS: {numbers.get('failed_runs')} baseline_failed={numbers.get('baseline_failed', False)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
