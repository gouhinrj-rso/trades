"""End-to-end test of the baseline runner and the scorecard, on made-up fixture data.

Nothing here reads the real data cache, and the trial log is a temporary file,
so the real experiments/trials.csv is never touched by a test.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.data import sessions as cal
from src.reporting import scorecard
from src.research import run_baseline, trial_log
from src.research.backtest import prepare_bars

SESSIONS = cal.expected_sessions("NYSE", "2010-01-04", "2018-01-10")
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]

CONFIG = {
    "account": {"starting_cash": 10000.0},
    "costs": {"fee_bps_per_side": 2.0, "slippage_bps": 5.0},
    "universe": UNIVERSE, "cash_vehicles": ["SHV", "BIL"],
    "data": {"calendar": "NYSE", "start": "2010-01-04", "train_end": "2015-12-31", "validation_end": "2017-12-31",
             "cache_dir": "unused"},
    "baseline": {"momentum_lookback_months": 12, "momentum_skip_months": 1, "trend_sma_days": 200,
                 "volatility_days": 60, "trade_start": "2012-01-01"},
}


def fixture_bars(seed: int, drift: float, vol: float) -> pd.DataFrame:
    """Made-up bars up to the end of the fixture's validation window."""
    days = SESSIONS[SESSIONS <= "2017-12-31"]
    rng = np.random.default_rng(seed)
    # A slow wave on top of the drift gives the trend rule something to switch on and off.
    wave = 0.0006 * np.sin(np.arange(len(days)) / 90.0 + seed)
    close = 100.0 * np.cumprod(1.0 + drift + wave + vol * rng.standard_normal(len(days)))
    return prepare_bars(pd.DataFrame({
        "session": days, "open": close * (1.0 + 0.001 * rng.standard_normal(len(days))), "close": close,
        "adj_close": close, "available_at": pd.DatetimeIndex(days).tz_localize("UTC") + pd.Timedelta(hours=21)}))


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("baseline")
    trials_file = out_dir / "trials.csv"
    bars = {symbol: fixture_bars(index, 0.0003, 0.008 + 0.001 * index) for index, symbol in enumerate(UNIVERSE)}
    bars["SHV"] = fixture_bars(99, 0.00004, 0.0001)
    days = SESSIONS[SESSIONS <= "2017-12-31"]
    rng = np.random.default_rng(5)
    factors = pd.DataFrame({"rf": 0.00004, "mkt_rf": 0.01 * rng.standard_normal(len(days)),
                            "mom": 0.006 * rng.standard_normal(len(days))}, index=days)
    windows = run_baseline.window_slices(CONFIG, days[-1], unlocked=False)
    numbers = run_baseline.run_everything(bars, SESSIONS, CONFIG, factors, windows, "SHV", out_dir,
                                          trials_file=trials_file, rf_source="fixture RF")
    return numbers, out_dir, trials_file


def test_fourteen_runs_are_logged_one_row_each(outcome):
    _, _, trials_file = outcome
    rows = trial_log.read_trials(trials_file)
    assert len(rows) == 14
    kinds = [row["run_kind"] for row in rows]
    assert kinds.count("baseline") == 1 and kinds.count("cost_stress") == 3
    assert kinds.count("neighbour") == 8 and kinds.count("benchmark") == 2
    assert [row["trial_id"] for row in rows] == [f"T{n:04d}" for n in range(1, 15)]
    assert all(row["status"] == "ok" and row["holdout_unlocked"] == "no" for row in rows)
    assert {row["cost_multiple"] for row in rows if row["run_kind"] == "cost_stress"} == {"0", "2", "3"}
    assert len({row["params_json"] for row in rows if row["run_kind"] in ("baseline", "neighbour")}) == 9


def test_a_second_invocation_appends_and_raises_the_trial_count(tmp_path):
    trials_file = tmp_path / "trials.csv"
    for expected in (1, 2, 3):
        trial_log.append_trial({"run_kind": "baseline", "strategy": "x", "status": "ok"}, trials_file)
        assert trial_log.count_trials(trials_file) == expected
    with pytest.raises(ValueError, match="unknown trial fields"):
        trial_log.append_trial({"sharp_annual": 1.0}, trials_file)


def test_a_crashing_run_is_still_logged_as_a_failed_trial(tmp_path):
    bars = {"SPY": fixture_bars(1, 0.0003, 0.01)}                 # IEF and the rest are missing on purpose
    config = dict(CONFIG, universe=["SPY"])
    days = SESSIONS[SESSIONS <= "2017-12-31"]
    windows = run_baseline.window_slices(config, days[-1], unlocked=False)
    numbers = run_baseline.run_everything(bars, SESSIONS, config, None, windows, None, tmp_path,
                                          trials_file=tmp_path / "trials.csv")
    rows = trial_log.read_trials(tmp_path / "trials.csv")
    assert len(rows) == 14                                        # every run has a row, failed or not
    failed = [row for row in rows if row["status"] == "failed"]
    assert [row["strategy"] for row in failed] == ["60_40_SPY_IEF"]
    assert failed[0]["failure_reason"]
    assert "60_40_SPY_IEF" in numbers["failed_runs"]
    # A missing benchmark makes gate 6 unanswerable rather than a pass.
    gates = {gate["gate"]: gate["verdict"] for gate in scorecard.evaluate_gates(numbers)}
    assert gates["6. Beats the benchmarks"] == scorecard.INSUFFICIENT


def test_output_files_exist_and_do_not_reach_past_validation(outcome):
    numbers, out_dir, _ = outcome
    for name in ["equity_curve.csv", "trades.csv", "round_trips.csv", "rebalances.csv", "metrics.json", "walk_forward.csv"]:
        assert (out_dir / name).exists(), name
    curve = pd.read_csv(out_dir / "equity_curve.csv", parse_dates=["session"])
    assert curve["session"].min() == pd.Timestamp("2012-01-03") and curve["session"].max() == pd.Timestamp("2017-12-29")
    assert set(curve["window"]) == {"train", "validation"}
    assert {"strategy_equity", "SPY_buy_and_hold_equity", "60_40_SPY_IEF_equity"} <= set(curve.columns)
    assert "holdout" not in numbers["windows"] and numbers["holdout_unlocked"] is False
    json.loads((out_dir / "metrics.json").read_text())            # valid JSON


def test_every_requested_metric_is_reported(outcome):
    numbers, _, _ = outcome
    window = numbers["metrics"]["train_plus_validation"]
    for key in ["cagr", "max_drawdown_depth", "max_drawdown_duration_days", "sharpe", "sortino", "calmar",
                "profit_factor", "expectancy_r", "win_rate", "payoff_ratio", "exposure", "turnover",
                "worst_day_value", "worst_month_value", "cvar_95"]:
        assert key in window, key
    stress = numbers["cost_stress"]["train_plus_validation"]
    assert set(stress["cagr_by_multiple"]) == {0, 1, 2, 3}
    assert stress["cost_drag"] > 0                                 # costs can only hurt
    assert "breakeven_multiple" in stress
    robust = numbers["robustness"]
    for key in ["sharpe_ci", "expectancy_ci", "psr_vs_zero", "dsr", "pbo", "walk_forward", "factor_regression",
                "neighbourhood", "regimes", "concentration", "first_eligible"]:
        assert key in robust, key
    assert robust["n_trials"] == 14 and robust["dsr"]["n_trials"] == 14
    assert robust["trial_sharpes_used_for_spread"] == 9
    assert robust["pbo"]["n_configs"] == 9 and robust["pbo"]["n_splits"] == 12870
    assert len(robust["neighbourhood"]) == 9
    assert "alpha_t" in robust["factor_regression"]


def test_window_numbers_come_from_one_continuous_run(outcome):
    numbers, out_dir, _ = outcome
    m = numbers["metrics"]
    assert m["train"]["n_days"] + m["validation"]["n_days"] == m["train_plus_validation"]["n_days"]
    growth = (1 + m["train"]["total_return"]) * (1 + m["validation"]["total_return"])
    assert growth == pytest.approx(1 + m["train_plus_validation"]["total_return"])
    curve = pd.read_csv(out_dir / "equity_curve.csv")
    assert curve["strategy_equity"].iloc[-1] / 10000.0 == pytest.approx(growth)


def test_scorecard_gives_one_verdict_per_gate(outcome):
    numbers, _, _ = outcome
    gates = scorecard.evaluate_gates(json.loads(json.dumps(numbers, default=float)))
    assert [gate["gate"][0] for gate in gates] == list("123456")
    assert all(gate["verdict"] in (scorecard.PASS, scorecard.FAIL, scorecard.INSUFFICIENT) for gate in gates)
    page = scorecard.render_scorecard(numbers)
    assert "**Overall:" in page and "locked and was not used" in page
    assert "Trials counted (rows in experiments/trials.csv): 14" in page
    assert "Parameter neighbourhood" in scorecard.render_metrics(numbers)


def test_overall_verdict_rules():
    gate = lambda verdict: {"verdict": verdict}
    assert scorecard.overall_verdict([gate("PASS")] * 6) == "PASS"
    assert scorecard.overall_verdict([gate("PASS")] * 5 + [gate("FAIL")]) == "FAIL"
    assert scorecard.overall_verdict([gate("PASS")] * 5 + [gate("INSUFFICIENT EVIDENCE")]) == "INSUFFICIENT EVIDENCE"
    assert scorecard.overall_verdict([gate("FAIL"), gate("INSUFFICIENT EVIDENCE")]) == "FAIL"


def test_gates_use_the_owners_thresholds():
    assert (scorecard.MIN_DSR, scorecard.MAX_PBO, scorecard.MIN_OOS_SHARE_OF_IS) == (0.95, 0.30, 0.50)
    assert scorecard.COST_STRESS_MULTIPLE == 2 and scorecard.MIN_EXPECTANCY_CI_LOWER == 0.0


def test_gate_logic_on_hand_made_numbers():
    numbers = {
        "metrics": {"train": {"n_days": 3000, "sharpe": 1.0}, "validation": {"n_days": 1200, "sharpe": 0.4},
                    "train_plus_validation": {"n_days": 4200}},
        "benchmarks": {"A": {"validation": {"sharpe": 0.3}}, "B": {"validation": {"sharpe": 0.5}}},
        "cost_stress": {"train": {"total_return_by_multiple": {"2": 0.5}},
                        "validation": {"total_return_by_multiple": {"2": -0.01}}},
        "robustness": {"dsr": {"dsr": 0.96, "n_trials": 14, "psr_vs_zero": 0.99, "expected_max_sharpe": 0.01},
                       "trial_sharpes_used_for_spread": 9, "pbo": {"pbo": 0.30, "n_configs": 9, "n_splits": 12870,
                                                                   "rows_used": 4192},
                       "closed_trades": 29, "expectancy_ci": {"point": 0.2, "lower": 0.1, "upper": 0.3}},
    }
    verdicts = {gate["gate"][0]: gate["verdict"] for gate in scorecard.evaluate_gates(numbers)}
    assert verdicts["1"] == "PASS"                      # 0.96 > 0.95
    assert verdicts["2"] == "FAIL"                      # 0.30 is not below 0.30
    assert verdicts["3"] == "FAIL"                      # 0.4 / 1.0 = 40% < 50%
    assert verdicts["4"] == "FAIL"                      # validation loses money at 2x costs
    assert verdicts["5"] == "INSUFFICIENT EVIDENCE"     # 29 closed trades < 30
    assert verdicts["6"] == "FAIL"                      # 0.4 beats A (0.3) but not B (0.5)
