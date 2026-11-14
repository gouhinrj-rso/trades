"""Turn the baseline's numbers into reports/scorecard.md and reports/baseline/metrics.md.

Each gate in protocol.md section 7 gets exactly one of three answers:

    PASS                   the rule is met
    FAIL                   the rule is not met
    INSUFFICIENT EVIDENCE  there is not enough data to say either way

The thresholds below are the owner's. They are written here once and nothing
in this file moves them to turn a FAIL into a PASS.

Run from the repo root (after `python -m src.research.run_baseline`):

    python -m src.reporting.scorecard
"""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
METRICS_FILE = PROJECT_DIR / "reports" / "baseline" / "metrics.json"
SCORECARD_FILE = PROJECT_DIR / "reports" / "scorecard.md"
METRICS_PAGE = PROJECT_DIR / "reports" / "baseline" / "metrics.md"

# ============================================================================
# GATE THRESHOLDS - the owner's (2026-11-14). Do not change them to get a pass.
# ============================================================================
MIN_DSR = 0.95                    # gate 1: Deflated Sharpe Ratio must be ABOVE this
MAX_PBO = 0.30                    # gate 2: Probability of Backtest Overfitting must be BELOW this
MIN_OOS_SHARE_OF_IS = 0.50        # gate 3: validation Sharpe at least this share of train Sharpe
COST_STRESS_MULTIPLE = 2          # gate 4: still profitable at this multiple of costs
MIN_EXPECTANCY_CI_LOWER = 0.0     # gate 5: lower end of the expectancy interval must be ABOVE this
# gate 6: validation Sharpe above both benchmarks' validation Sharpe
# ---- how much data a gate needs before it may answer at all ------------------
MIN_DAYS = 252
MIN_CLOSED_TRADES = 30
MIN_PBO_CONFIGS = 4
MIN_PBO_ROWS = 16 * 21
# ============================================================================

PASS, FAIL, INSUFFICIENT = "PASS", "FAIL", "INSUFFICIENT EVIDENCE"


def _number(value, digits=2, percent=False) -> str:
    """Format a number for the page; a missing one shows as 'n/a'."""
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%" if percent else f"{value:.{digits}f}"


def evaluate_gates(numbers: dict) -> list:
    """Apply the six gates. Returns a list of dicts: gate, verdict, evidence."""
    metrics = numbers.get("metrics", {})
    robust = numbers.get("robustness", {})
    train, validation = metrics.get("train", {}), metrics.get("validation", {})
    both = metrics.get("train_plus_validation", {})
    gates = []

    def add(name, rule, verdict, evidence):
        gates.append({"gate": name, "rule": rule, "verdict": verdict, "evidence": evidence})

    # ---- 1. Deflated Sharpe ------------------------------------------------------------
    dsr = robust.get("dsr")
    if dsr is None or both.get("n_days", 0) < MIN_DAYS or dsr.get("dsr") is None:
        add("1. Deflated Sharpe", f"DSR > {MIN_DSR}", INSUFFICIENT,
            f"needs at least 2 trial Sharpes and {MIN_DAYS} daily returns; "
            f"have {robust.get('trial_sharpes_used_for_spread', 0)} and {both.get('n_days', 0)}")
    else:
        add("1. Deflated Sharpe", f"DSR > {MIN_DSR}", PASS if dsr["dsr"] > MIN_DSR else FAIL,
            f"DSR = {dsr['dsr']:.3f} with N = {dsr['n_trials']} trials "
            f"(PSR against zero = {dsr['psr_vs_zero']:.3f}; Sharpe to beat = "
            f"{dsr['expected_max_sharpe'] * 252 ** 0.5:.2f} annualized)")

    # ---- 2. Overfitting -------------------------------------------------------------------
    pbo = robust.get("pbo")
    if pbo is None or pbo.get("n_configs", 0) < MIN_PBO_CONFIGS or pbo.get("rows_used", 0) < MIN_PBO_ROWS:
        add("2. Overfitting (PBO)", f"PBO < {MAX_PBO}", INSUFFICIENT,
            f"needs at least {MIN_PBO_CONFIGS} configurations and {MIN_PBO_ROWS} daily returns")
    else:
        add("2. Overfitting (PBO)", f"PBO < {MAX_PBO}", PASS if pbo["pbo"] < MAX_PBO else FAIL,
            f"PBO = {pbo['pbo']:.3f} over {pbo['n_splits']} splits of {pbo['n_configs']} configurations")

    # ---- 3. Out of sample holds up -----------------------------------------------------------
    rule = f"validation Sharpe >= {MIN_OOS_SHARE_OF_IS:.0%} of train Sharpe"
    if train.get("n_days", 0) < MIN_DAYS or validation.get("n_days", 0) < MIN_DAYS \
            or train.get("sharpe") is None or validation.get("sharpe") is None:
        add("3. Out of sample holds up", rule, INSUFFICIENT, f"a window has fewer than {MIN_DAYS} daily returns")
    elif train["sharpe"] <= 0:
        add("3. Out of sample holds up", rule, FAIL, f"train Sharpe is {train['sharpe']:.2f}, not above zero")
    else:
        share = validation["sharpe"] / train["sharpe"]
        add("3. Out of sample holds up", rule, PASS if share >= MIN_OOS_SHARE_OF_IS else FAIL,
            f"validation {validation['sharpe']:.2f} / train {train['sharpe']:.2f} = {share:.0%}")

    # ---- 4. Survives costs -----------------------------------------------------------------------
    rule = f"total return above zero at {COST_STRESS_MULTIPLE}x costs in train and in validation"
    stress = numbers.get("cost_stress", {})
    key = str(COST_STRESS_MULTIPLE)
    totals = {window: {str(k): v for k, v in stress.get(window, {}).get("total_return_by_multiple", {}).items()}.get(key)
              for window in ("train", "validation")}
    if None in totals.values():
        add("4. Survives costs", rule, INSUFFICIENT, "a cost-stress run failed or is missing")
    else:
        ok = all(value > 0 for value in totals.values())
        add("4. Survives costs", rule, PASS if ok else FAIL,
            f"at {COST_STRESS_MULTIPLE}x costs: train {totals['train']:+.1%}, validation {totals['validation']:+.1%}")

    # ---- 5. Positive expectancy ---------------------------------------------------------------------
    rule = "lower end of the 95% interval of expectancy (in R) above 0"
    interval = robust.get("expectancy_ci")
    if interval is None or robust.get("closed_trades", 0) < MIN_CLOSED_TRADES or interval.get("lower") is None:
        add("5. Positive expectancy", rule, INSUFFICIENT,
            f"needs at least {MIN_CLOSED_TRADES} closed trades; have {robust.get('closed_trades', 0)}")
    else:
        add("5. Positive expectancy", rule, PASS if interval["lower"] > MIN_EXPECTANCY_CI_LOWER else FAIL,
            f"expectancy {interval['point']:+.2f}R, 95% interval {interval['lower']:+.2f}R to "
            f"{interval['upper']:+.2f}R, {robust['closed_trades']} closed trades")

    # ---- 6. Beats the benchmarks ------------------------------------------------------------------------
    rule = "validation Sharpe above both benchmarks' validation Sharpe"
    benchmarks = numbers.get("benchmarks", {})
    bench = {name: values.get("validation", {}).get("sharpe") for name, values in benchmarks.items()}
    if len(bench) < 2 or None in bench.values() or validation.get("sharpe") is None \
            or validation.get("n_days", 0) < MIN_DAYS:
        add("6. Beats the benchmarks", rule, INSUFFICIENT, "a benchmark run failed or validation is too short")
    else:
        ok = all(validation["sharpe"] > value for value in bench.values())
        add("6. Beats the benchmarks", rule, PASS if ok else FAIL,
            f"strategy {validation['sharpe']:.2f} vs " + ", ".join(f"{name} {value:.2f}" for name, value in bench.items()))
    return gates


def overall_verdict(gates: list) -> str:
    """PASS only if every gate passes; any FAIL makes it FAIL; otherwise INSUFFICIENT EVIDENCE."""
    verdicts = [gate["verdict"] for gate in gates]
    if any(verdict == FAIL for verdict in verdicts):
        return FAIL
    return PASS if all(verdict == PASS for verdict in verdicts) else INSUFFICIENT


def render_scorecard(numbers: dict) -> str:
    """The scorecard page."""
    if numbers.get("baseline_failed"):
        return ("# Scorecard: frozen baseline B1\n\n**Overall: INSUFFICIENT EVIDENCE.** The baseline run failed; "
                "see experiments/trials.csv for the reason.\n")
    gates = evaluate_gates(numbers)
    windows = numbers["windows"]
    lines = ["# Scorecard: frozen baseline B1", "",
             f"**Overall: {overall_verdict(gates)}**", "",
             f"- Judged on train ({windows['train'][0]} to {windows['train'][1]}) and validation "
             f"({windows['validation'][0]} to {windows['validation'][1]}) only.",
             "- The holdout (2022 onward) is " + ("**UNLOCKED** for this run; see experiments/holdout_access.csv."
                                                  if numbers.get("holdout_unlocked") else
                                                  "**locked and was not used**. A pass here says nothing about it."),
             f"- Costs: {numbers['costs']['commission_bps']:g} bps commission + {numbers['costs']['slippage_bps']:g} bps "
             "slippage per side.",
             f"- Sharpe: {numbers['sharpe_frequency']}; risk-free = {numbers['risk_free']}.",
             f"- Trials counted (rows in experiments/trials.csv): {numbers.get('robustness', {}).get('n_trials', 'n/a')}.",
             f"- Run group: {numbers['run_group']}. Rules: protocol.md. Definitions: reports/metrics_spec.md."]
    if numbers.get("excluded"):
        risk = [item["symbol"] for item in numbers["excluded"] if not item.get("cash_vehicle")]
        other = [item["symbol"] for item in numbers["excluded"] if item.get("cash_vehicle")]
        if risk:
            lines.append(f"- **UNIVERSE INCOMPLETE**: {', '.join(risk)} failed data-health checks and "
                         "were left out. The result is for the remaining assets.")
        if other:
            lines.append(f"- Cash vehicle not used (data blocked): {', '.join(other)}. Using {numbers['cash_vehicle']}.")
    if numbers.get("failed_runs"):
        lines.append(f"- **Failed runs**: {', '.join(numbers['failed_runs'])}.")
    lines += ["", "| Gate | Rule | Verdict | Evidence |", "|---|---|---|---|"]
    for gate in gates:
        lines.append(f"| {gate['gate']} | {gate['rule']} | **{gate['verdict']}** | {gate['evidence']} |")
    lines += ["", "A gate answers INSUFFICIENT EVIDENCE when there is too little data to decide. "
              "Thresholds are in `src/reporting/scorecard.py` and are the owner's.", ""]
    return "\n".join(lines)


def render_metrics(numbers: dict) -> str:
    """The metrics page: every number, per window, with the benchmarks next to it."""
    if numbers.get("baseline_failed"):
        return "# Baseline B1 metrics\n\nThe baseline run failed. See experiments/trials.csv.\n"
    names = [name for name in ("train", "validation", "train_plus_validation", "holdout") if name in numbers["metrics"]]
    metrics, robust, stress = numbers["metrics"], numbers.get("robustness", {}), numbers.get("cost_stress", {})
    lines = ["# Baseline B1: metrics", "",
             "Definitions: `reports/metrics_spec.md`. Rules: `protocol.md`. All returns are net of costs.", "",
             f"Sharpe and Sortino: {numbers['sharpe_frequency']}; risk-free = {numbers['risk_free']}.", "",
             "## Strategy", "", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = [("Days", "n_days", 0, False), ("Total return", "total_return", 1, True), ("CAGR", "cagr", 2, True),
            ("Max drawdown", "max_drawdown_depth", 1, True), ("Drawdown length (days)", "max_drawdown_duration_days", 0, False),
            ("Sharpe", "sharpe", 2, False), ("Sortino", "sortino", 2, False), ("Calmar", "calmar", 2, False),
            ("Closed trades", "n_closed_trades", 0, False), ("Win rate", "win_rate", 1, True),
            ("Profit factor", "profit_factor", 2, False), ("Payoff ratio", "payoff_ratio", 2, False),
            ("Expectancy (R)", "expectancy_r", 3, False), ("Exposure", "exposure", 1, True),
            ("Turnover (x per year, two-way)", "turnover", 2, False), ("Worst day", "worst_day_value", 2, True),
            ("Worst month", "worst_month_value", 2, True), ("CVaR 95% (daily)", "cvar_95", 2, True)]
    for label, key, digits, percent in rows:
        lines.append(f"| {label} | " + " | ".join(_number(metrics[name].get(key), digits, percent) for name in names) + " |")
    for label, key in (("Drawdown peak", "max_drawdown_peak_date"), ("Drawdown trough", "max_drawdown_trough_date"),
                       ("Worst day date", "worst_day_date"), ("Worst month", "worst_month")):
        lines.append(f"| {label} | " + " | ".join(str(metrics[name].get(key) or "n/a") for name in names) + " |")
    lines.append("| Cost drag (CAGR lost to costs) | " + " | ".join(
        _number(stress.get(name, {}).get("cost_drag"), 2, True) for name in names) + " |")
    lines.append("| Cost break-even multiple | " + " | ".join(
        _number(stress.get(name, {}).get("breakeven_multiple"), 1) for name in names) + " |")

    lines += ["", "## Benchmarks", "", "| Run | Window | CAGR | Max drawdown | Sharpe | Sortino | Calmar |", "|---|---|---|---|---|---|---|"]
    for label, table in [("Strategy", metrics)] + list(numbers.get("benchmarks", {}).items()):
        for name in names:
            row = table.get(name, {})
            lines.append(f"| {label} | {name} | {_number(row.get('cagr'), 2, True)} | "
                         f"{_number(row.get('max_drawdown_depth'), 1, True)} | {_number(row.get('sharpe'))} | "
                         f"{_number(row.get('sortino'))} | {_number(row.get('calmar'))} |")

    lines += ["", "## Cost stress (baseline rerun at multiples of the stated costs)", "",
              "| Window | " + " | ".join(f"{m}x CAGR" for m in (0, 1, 2, 3)) + " | " +
              " | ".join(f"{m}x total" for m in (0, 1, 2, 3)) + " |", "|---|" + "---|" * 8]
    for name in names:
        entry = stress.get(name, {})
        cagrs = {str(k): v for k, v in entry.get("cagr_by_multiple", {}).items()}
        totals = {str(k): v for k, v in entry.get("total_return_by_multiple", {}).items()}
        lines.append(f"| {name} | " + " | ".join(_number(cagrs.get(str(m)), 2, True) for m in (0, 1, 2, 3)) + " | " +
                     " | ".join(_number(totals.get(str(m)), 1, True) for m in (0, 1, 2, 3)) + " |")

    lines += ["", "## Robustness (train + validation only)", ""]
    interval = robust.get("sharpe_ci")
    if interval:
        lines.append(f"- Sharpe {interval['point']:.2f}, 95% block-bootstrap interval {interval['lower']:.2f} to "
                     f"{interval['upper']:.2f} (blocks of {interval['block_length']} days, {interval['n_resamples']} resamples).")
    interval = robust.get("expectancy_ci")
    if interval:
        lines.append(f"- Expectancy {interval['point']:+.3f}R, 95% interval {interval['lower']:+.3f}R to "
                     f"{interval['upper']:+.3f}R ({robust.get('closed_trades')} closed trades in "
                     f"{interval.get('exit_months')} exit months, blocks of {interval['block_length']} months).")
    lines.append(f"- Probabilistic Sharpe Ratio against zero: {_number(robust.get('psr_vs_zero'), 3)}.")
    dsr = robust.get("dsr")
    if dsr:
        lines.append(f"- Deflated Sharpe Ratio: {dsr['dsr']:.3f} with N = {dsr['n_trials']} trials; spread of trial "
                     f"Sharpes taken from {robust.get('trial_sharpes_used_for_spread')} baseline and neighbour rows; "
                     f"daily skew {dsr['skew']:.2f}, kurtosis {dsr['kurtosis']:.1f}.")
    pbo = robust.get("pbo")
    if pbo:
        lines.append(f"- PBO (CSCV, 16 blocks): {pbo['pbo']:.3f} over {pbo['n_splits']} splits of {pbo['n_configs']} "
                     f"configurations; {pbo['rows_dropped']} early rows dropped to make equal blocks. "
                     "With 9 near-identical configurations this is weak evidence either way.")
    forward = robust.get("walk_forward")
    if forward:
        lines.append(f"- Walk-forward {forward['years']} years: picking the best neighbour on past data each year "
                     f"compounds to {forward['selected_compounded']:+.1%}; the frozen baseline to "
                     f"{forward['baseline_compounded']:+.1%}. The pick beat the baseline in "
                     f"{forward['years_selected_beat_baseline']} years and was the baseline in "
                     f"{forward['years_selected_was_baseline']}. Detail: `reports/baseline/walk_forward.csv`.")
    fit = robust.get("factor_regression", {})
    if fit.get("run") is False:
        lines.append(f"- Factor regression: not run ({fit.get('reason')}).")
    elif fit:
        betas = ", ".join(f"{name} {value:.2f} (t {fit['beta_t'][name]:.1f})" for name, value in fit["betas"].items())
        lines.append(f"- Factor regression (daily, Newey-West {fit['lags']} lags, {fit['n_obs']} days): alpha "
                     f"{fit['alpha_annualized']:+.2%} a year, t = {fit['alpha_t']:.2f}; betas: {betas}; "
                     f"R-squared {fit['r_squared']:.2f}. Factor: {fit.get('factor_source')}.")

    cells = robust.get("neighbourhood", {})
    if cells:
        for window in ("train_plus_validation", "train", "validation"):
            lines += ["", f"### Parameter neighbourhood: Sharpe, {window}", "",
                      "| Momentum lookback | trend 150 days | trend 200 days | trend 250 days |", "|---|---|---|---|"]
            for lookback in (9, 12, 15):
                values = []
                for sma in (150, 200, 250):
                    cell = cells.get(f"mom{lookback}_sma{sma}")
                    text = _number(cell[window]["sharpe"]) if cell else "failed"
                    values.append(f"**{text}**" if (lookback, sma) == (12, 200) else text)
                lines.append(f"| {lookback} months | " + " | ".join(values) + " |")
        lines += ["", "The centre cell (bold) is the frozen baseline. The other cells are diagnostics and cannot replace it."]

    regimes = robust.get("regimes", {})
    if regimes:
        lines += ["", "## Regimes", "", "| Run | Year | Return | Max drawdown in the year |", "|---|---|---|---|"]
        for label, years in regimes.items():
            if label == "note":
                continue
            for item in years:
                lines.append(f"| {label} | {item['year']} | {_number(item.get('return'), 1, True)} | "
                             f"{_number(item.get('max_drawdown'), 1, True)} |")
        lines += ["", regimes.get("note", "")]

    focus = robust.get("concentration")
    if focus:
        lines += ["", "## Concentration of profit (closed trades)", "",
                  f"- Total profit of closed trades: {focus['total_pnl_closed']:,.0f} paper dollars.",
                  f"- Share from the 5 best trades: {_number(focus['top_5_share'], 0, True)}; from the best 10% of trades: "
                  f"{_number(focus['top_10pct_share'], 0, True)}.",
                  f"- Profit without the 5 best trades: {focus['pnl_without_top_5']:,.0f}.",
                  "- By asset: " + ", ".join(f"{symbol} {value:,.0f}" for symbol, value in
                                             sorted(focus["by_symbol"].items(), key=lambda pair: -pair[1])) + "."]
    first = robust.get("first_eligible")
    if first:
        lines += ["", "## When each asset first became eligible", "",
                  ", ".join(f"{symbol} {day}" for symbol, day in sorted(first.items(), key=lambda pair: pair[1])) + ".", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    if not METRICS_FILE.exists():
        print("No reports/baseline/metrics.json yet. Run: python -m src.research.run_baseline")
        return 1
    numbers = json.loads(METRICS_FILE.read_text())
    SCORECARD_FILE.write_text(render_scorecard(numbers))
    METRICS_PAGE.write_text(render_metrics(numbers))
    print(SCORECARD_FILE.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
