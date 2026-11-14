"""Independent re-check of baseline B1. Uses pandas only: no engine, no ledger, no strategy module.
Reads train and validation parquet files only (never the holdout file)."""
import json
import numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]
UNI = ["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","GLD","DBC","VNQ"]; CASHV = "SHV"
def load(sym):
    df = pd.concat([pd.read_parquet(R/f"data/cache/daily/{sym}/{w}.parquet") for w in ("train","validation")])
    df = df.sort_values("session").set_index("session")
    f = df["adj_close"]/df["close"]
    return pd.DataFrame({"close": df["close"], "tro": df["open"]*f, "trc": df["adj_close"]})
B = {s: load(s) for s in UNI+[CASHV]}
assert max(b.index.max() for b in B.values()) <= pd.Timestamp("2021-12-31")
days = B["SPY"].index
# ---- A1. signals, computed a different way (resample / rolling) ----
sig = {}
for s in UNI:
    b = B[s]
    me = b["trc"].groupby(b.index.to_period("M")).last()
    mom = (me.shift(1)/me.shift(12) - 1)
    sma = b["close"].rolling(200).mean()
    vol = b["trc"].pct_change().rolling(60).std()
    sig[s] = pd.DataFrame({"mom": mom.reindex(b.index.to_period("M")).to_numpy(), "trend": (b["close"] > sma).where(sma.notna()),
                           "vol": vol}, index=b.index)
month_ends = [d for i, d in enumerate(days[:-1]) if days[i+1].month != d.month and d >= pd.Timestamp("2004-12-01")]
W = {}
for t in month_ends:
    el = {s: sig[s].loc[t] for s in UNI if t in sig[s].index and sig[s].loc[t].notna().all()}
    tot = sum(1/x["vol"] for x in el.values())
    W[t] = {s: (1/x["vol"])/tot for s, x in el.items() if x["mom"] > 0 and bool(x["trend"])}
reb = pd.read_csv(R/"reports/baseline/rebalances.csv", parse_dates=["signal_session"])
reb = reb[reb["symbol"] != CASHV]
maxdiff = 0.0
for t, g in reb.groupby("signal_session"):
    for _, r in g.iterrows():
        maxdiff = max(maxdiff, abs(r["weight"] - W[t].get(r["symbol"], 0.0)))
print(f"A1 weights: {len(W)} signal days, max |engine - independent| = {maxdiff:.2e}")
# ---- A2. simulate with plain arithmetic ----
slip, fee = 5e-4, 2e-4
def simulate(mult=1.0, cash_rate=None):
    cash, sh = 10000.0, {}
    eq, cashfrac, pend = [], [], None
    for i, d in enumerate(days):
        if d < pd.Timestamp("2005-01-01"):
            if d == month_ends[0]: pend = (d, dict(W[d]))
            continue
        if cash_rate is not None and CASHV not in sh:   # idle cash earns the T-bill rate
            cash *= (1 + cash_rate.get(d, 0.0))
        if pend is not None:
            t, w = pend; pend = None
            val = cash + sum(q*B[s].at[t, "trc"] for s, q in sh.items())
            left = 1 - sum(w.values())
            if left > 1e-12 and t in B[CASHV].index and d in B[CASHV].index: w[CASHV] = left
            tgt = {s: x*val/B[s].at[t, "trc"] for s, x in w.items()}
            dl = {s: tgt.get(s, 0.0) - sh.get(s, 0.0) for s in set(tgt) | set(sh)}
            for s, q in dl.items():
                if q < -1e-9:
                    cash += -q*B[s].at[d, "tro"]*(1 - slip*mult)*(1 - fee*mult); sh[s] = sh.get(s, 0.0) + q
            need = sum(q*B[s].at[d, "tro"]*(1 + slip*mult)*(1 + fee*mult) for s, q in dl.items() if q > 1e-9)
            sc = 1.0 if need <= cash else cash/need*(1 - 1e-12)
            for s, q in dl.items():
                if q*sc > 1e-9:
                    cash -= q*sc*B[s].at[d, "tro"]*(1 + slip*mult)*(1 + fee*mult); sh[s] = sh.get(s, 0.0) + q*sc
            sh = {s: q for s, q in sh.items() if abs(q) > 1e-9}
        e = cash + sum(q*B[s].at[d, "trc"] for s, q in sh.items())
        eq.append(e); cashfrac.append(cash/e)
        if d in W and i + 1 < len(days): pend = (d, dict(W[d]))
    idx = days[days >= pd.Timestamp("2005-01-01")]
    return pd.Series(eq, index=idx), pd.Series(cashfrac, index=idx)
mine, cashfrac = simulate()
curve = pd.read_csv(R/"reports/baseline/equity_curve.csv", parse_dates=["session"]).set_index("session")
rel = (mine/curve["strategy_equity"] - 1).abs()
print(f"A2 equity: independent final {mine.iloc[-1]:.2f} vs engine {curve['strategy_equity'].iloc[-1]:.2f}; max rel diff over {len(rel)} days = {rel.max():.2e}")
# ---- B. timing in trades.csv ----
tr = pd.read_csv(R/"reports/baseline/trades.csv", parse_dates=["fill_session","signal_session"])
pos = {d: i for i, d in enumerate(days)}
ok_next = all(pos[f] == pos[s] + 1 for f, s in zip(tr["fill_session"], tr["signal_session"]))
ok_first = all(days[pos[f]-1].month != f.month for f in tr["fill_session"])
ok_open = np.allclose(tr["reference_open"], [B[s].at[f, "tro"] for s, f in zip(tr["symbol"], tr["fill_session"])])
worse = np.where(tr["side"] == "buy", tr["fill_price"] >= tr["reference_open"], tr["fill_price"] <= tr["reference_open"]).all()
print(f"B fills: {len(tr)} fills; all on the session after the signal: {ok_next}; all on the first session of a month: {ok_first}; "
      f"reference price equals that day's open: {ok_open}; every fill worse than the open: {bool(worse)}")
# ---- C. Sharpe from the saved curve ----
fac = pd.read_parquet(R/"data/cache/factors/french_daily.parquet"); fac = fac[fac.index <= "2021-12-31"]
rf = fac["rf"].reindex(curve.index).fillna(0.0)
def rets(e): 
    r = e.pct_change(); r.iloc[0] = e.iloc[0]/10000.0 - 1; return r
def sharpe(r, rfx=None):
    x = r - (rf.reindex(r.index) if rfx is None else rfx); return x.mean()/x.std(ddof=1)*np.sqrt(252)
WIN = {"train": ("2005-01-01","2016-12-31"), "validation": ("2017-01-01","2021-12-31"), "both": ("2005-01-01","2021-12-31")}
r_s, r_b, r_6 = rets(curve["strategy_equity"]), rets(curve["SPY_buy_and_hold_equity"]), rets(curve["60_40_SPY_IEF_equity"])
print("C Sharpe recomputed from equity_curve.csv:", {k: round(sharpe(r_s[a:b]), 3) for k, (a, b) in WIN.items()})
# ---- D1. idle cash earning the T-bill rate before the cash vehicle exists ----
first_shv = B[CASHV].index.min()
print(f"D1 cash vehicle first bar {first_shv.date()}; mean idle-cash share of equity before 2007-03: {cashfrac[:'2007-02-28'].mean():.1%}, after: {cashfrac['2007-03-01':].mean():.2%}")
adj, _ = simulate(cash_rate=rf.to_dict())
r_adj = rets(adj)
cg = lambda e, a, b: (e[a:b].iloc[-1]/ (e[:a].iloc[-1] if len(e[:a]) and e[:a].index[-1] < pd.Timestamp(a) else 10000.0))
def cagr(r): 
    yrs = (r.index[-1] - r.index[0]).days/365.25; return (1 + r).prod()**(1/yrs) - 1
print("D1 with idle cash earning RF until the cash vehicle exists: Sharpe", {k: round(sharpe(r_adj[a:b]), 3) for k, (a, b) in WIN.items()},
      "CAGR", {k: round(cagr(r_adj[a:b]), 4) for k, (a, b) in WIN.items()})
# ---- D2. strategy vs 60/40: is the validation gap real? paired block bootstrap ----
rng = np.random.default_rng(20261114)
def boot_diff(a, b, block=21, n=2000):
    xa, xb = (r_s[a:b] - rf[a:b]).to_numpy(), (r_6[a:b] - rf[a:b]).to_numpy(); L = len(xa); out = []
    for _ in range(n):
        st = rng.integers(0, L, size=int(np.ceil(L/block)))
        ix = np.concatenate([(s + np.arange(block)) % L for s in st])[:L]
        sa = xa[ix].mean()/xa[ix].std(ddof=1); sb = xb[ix].mean()/xb[ix].std(ddof=1); out.append((sa - sb)*np.sqrt(252))
    return np.quantile(out, [0.025, 0.975]), float(np.mean(np.array(out) > 0))
for k, (a, b) in WIN.items():
    ci, p = boot_diff(a, b)
    print(f"D2 {k}: Sharpe strategy {sharpe(r_s[a:b]):.2f}, 60/40 {sharpe(r_6[a:b]):.2f}, SPY {sharpe(r_b[a:b]):.2f}; "
          f"difference vs 60/40 = {sharpe(r_s[a:b]) - sharpe(r_6[a:b]):+.2f}, 95% interval {ci[0]:+.2f} to {ci[1]:+.2f}, share of resamples where strategy is ahead {p:.0%}")
# ---- D3. DSR sensitivity ----
from statistics import NormalDist
N01 = NormalDist()
x = (r_s - rf).to_numpy(); n = len(x); sr = x.mean()/x.std(ddof=1)
g3 = ((x - x.mean())**3).mean()/x.std(ddof=0)**3; g4 = ((x - x.mean())**4).mean()/x.std(ddof=0)**4
def psr(bench): return N01.cdf((sr - bench)*np.sqrt(n - 1)/np.sqrt(1 - g3*sr + (g4 - 1)/4*sr**2))
def emax(N, sd): 
    g = 0.5772156649; return sd*((1 - g)*N01.inv_cdf(1 - 1/N) + g*N01.inv_cdf(1 - 1/(N*np.e)))
tl = pd.read_csv(R/"experiments/trials.csv")
sd9 = tl[(tl.run_kind.isin(["baseline","neighbour"])) & (tl.cost_multiple == 1)]["sharpe_per_day"].std(ddof=1)
sd14 = tl["sharpe_per_day"].std(ddof=1)
se_sr = np.sqrt((1 - g3*sr + (g4 - 1)/4*sr**2)/(n - 1))     # sampling error of one Sharpe estimate
print(f"D3 per-day Sharpe {sr:.4f} (annual {sr*np.sqrt(252):.2f}), skew {g3:.2f}, kurt {g4:.1f}, rows in trials.csv {len(tl)}")
for label, sd in [("spread of 9 neighbour cells (as reported)", sd9), ("spread of all 14 rows", sd14), ("sampling error of one Sharpe (independent-trials assumption)", se_sr)]:
    for N in (14, 50, 100):
        e = emax(N, sd); print(f"   {label}: sd {sd*np.sqrt(252):.3f} annual, N={N}: Sharpe to beat {e*np.sqrt(252):.2f}, DSR {psr(e):.3f}")
# ---- D4. expectancy in R under a holding-period-matched R, and without the best trades ----
rt = pd.read_csv(R/"reports/baseline/round_trips.csv", parse_dates=["entry_session","exit_session"])
cl = rt[~rt["open"]].copy()
cl["hold_days"] = [pos[e] - pos[s] for s, e in zip(cl["entry_session"], cl["exit_session"])]
print(f"D4 closed trades {len(cl)}; holding period in trading days: median {cl.hold_days.median():.0f}, mean {cl.hold_days.mean():.0f}, max {cl.hold_days.max()}; open at end: {int(rt['open'].sum())}")
cl["r_hold"] = cl["pnl"]/(cl["entry_notional"]*cl["entry_volatility"]*np.sqrt(cl["hold_days"].clip(lower=1)))
def boot_mean(col, frame):
    groups = [g[col].to_numpy() for _, g in frame.groupby(frame["exit_session"].dt.to_period("M"))]; L = len(groups); out = []
    r2 = np.random.default_rng(20261114)
    for _ in range(2000):
        st = r2.integers(0, L, size=int(np.ceil(L/3))); ix = np.concatenate([(s + np.arange(3)) % L for s in st])[:L]
        out.append(np.concatenate([groups[i] for i in ix]).mean())
    return np.quantile(out, [0.025, 0.975])
for label, col, frame in [("protocol R (1-month sigma)", "r_multiple", cl), ("R matched to each trade's holding period", "r_hold", cl),
                          ("protocol R, 5 best trades removed", "r_multiple", cl.drop(cl["pnl"].nlargest(5).index)),
                          ("protocol R, best 10% of trades removed", "r_multiple", cl.drop(cl["pnl"].nlargest(13).index))]:
    ci = boot_mean(col, frame); print(f"   {label}: expectancy {frame[col].mean():+.3f}R, 95% interval {ci[0]:+.3f} to {ci[1]:+.3f} (n={len(frame)})")
print(f"   median R {cl.r_multiple.median():+.2f}; share of trades with pnl>0 {np.mean(cl.pnl>0):.1%}; dollar pnl: mean {cl.pnl.mean():.0f}, median {cl.pnl.median():.0f}")
# ---- E. year-by-year vs 60/40 ----
yr = pd.DataFrame({"strategy": (1+r_s).groupby(r_s.index.year).prod()-1, "60/40": (1+r_6).groupby(r_6.index.year).prod()-1, "SPY": (1+r_b).groupby(r_b.index.year).prod()-1})
print("E calendar years:\n", (yr*100).round(1).to_string())
print("   years strategy beat 60/40:", int((yr.strategy > yr["60/40"]).sum()), "of", len(yr), "; years with a loss:", int((yr.strategy < 0).sum()))
