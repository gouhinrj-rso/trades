"""Robustness statistics for the paper-only research protocol (protocol.md, section 5).

Everything in this file is a PURE function: numbers in, numbers out. Nothing here reads a
file, writes a file, touches the network or places an order. The callers decide which
returns to pass in; the settings fixed by the protocol (block 21 days, 2,000 resamples,
seed 20261114, 16 PBO blocks, 21 Newey-West lags) are the defaults of the functions below.

Only numpy, pandas and the standard library are used. The normal distribution comes from
`statistics.NormalDist` (no scipy, no statsmodels).

Contents
--------
1. circular_block_bootstrap_ci   confidence interval that respects autocorrelation
2. probabilistic_sharpe_ratio    chance the true Sharpe is above a benchmark
3. expected_max_sharpe           the Sharpe luck alone produces among N trials
4. deflated_sharpe_ratio         PSR measured against that lucky Sharpe
5. pbo_cscv                      Probability of Backtest Overfitting
6. ols_newey_west                factor regression with HAC standard errors
7. walk_forward_selection        pick the best configuration on past data, year by year

All Sharpe ratios in this file are PER PERIOD (mean / standard deviation of the series as
given, no sqrt(252)). Annualizing is the job of the reporting code.
"""

import itertools
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

# The standard normal distribution: .cdf(x) is Phi(x), .inv_cdf(p) is Phi^-1(p).
_STANDARD_NORMAL = NormalDist()

# Euler-Mascheroni constant, used by the expected-maximum formula.
EULER_MASCHERONI = 0.5772156649

# Trading days per year, used only to annualize the regression intercept.
TRADING_DAYS_PER_YEAR = 252

# A variance computed from sums (sum of squares minus squared sum) is never exactly zero
# for a constant series because of floating point rounding. A variance smaller than this
# fraction of the mean square is treated as zero. It means "standard deviation below one
# millionth of the typical size of the numbers", far below anything a real return has.
_ZERO_VARIANCE_RELATIVE_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# 1. Circular block bootstrap
# ---------------------------------------------------------------------------

def _is_grouped(values):
    """True when `values` is a list (or tuple) of arrays rather than a flat list of numbers."""
    if not isinstance(values, (list, tuple)):
        return False
    # A flat list of numbers has only 0-dimensional items; a grouped list has arrays in it.
    return any(np.ndim(item) > 0 for item in values)


def circular_block_bootstrap_ci(values, statistic, block_length, n_resamples=2000,
                                level=0.95, seed=20261114):
    """Percentile confidence interval of `statistic` by the circular block bootstrap.

    Source: Politis and Romano (1992), "A circular block-resampling procedure for
    stationary data". Interval: the percentile method of Efron (1979).

    Why blocks: daily returns are not independent (volatility clusters). Resampling single
    days would destroy that structure and make the interval too narrow. Resampling blocks
    of consecutive items keeps it.

    Method, with n items and block length L:
      1. Draw k = ceil(n / L) block starts, each uniform on 0 .. n-1.
      2. A block is the items start, start+1, ..., start+L-1, with the position taken
         modulo n. That is the "circular" part: a block starting near the end wraps round
         to the start, so every item is equally likely to be picked.
      3. Join the k blocks and keep the first n items, so each resample has exactly as
         many items as the original.
      4. Compute the statistic on the resample. Repeat `n_resamples` times.
      5. lower = the (1 - level) / 2 quantile of the resampled statistics,
         upper = the 1 - (1 - level) / 2 quantile (2.5% and 97.5% for level 0.95).

    Two input shapes:
      * a 1-D array of numbers: the items are the numbers (e.g. daily returns);
      * a list of numpy arrays ("grouped"): the items are the arrays (e.g. one array of
        trade results per exit month). Blocks of consecutive list items are resampled and
        the chosen arrays are concatenated into one 1-D array before `statistic` is
        applied. An array in the list may be empty (a month without trades).

    Arguments:
      values        1-D array, or list of arrays for the grouped case.
      statistic     function taking a 1-D numpy array and returning one number.
      block_length  L, in items (days, or months in the grouped case). At least 1.
      n_resamples   number of resamples.
      level         confidence level, strictly between 0 and 1.
      seed          seed for numpy.random.default_rng, so the result is reproducible.

    Returns a dict:
      point        the statistic on the original data
      lower, upper the ends of the interval (nan if every resample failed)
      n            number of items (days, or groups)
      block_length, n_resamples   as given
      n_failed     resamples whose statistic was nan; they are dropped before the
                   quantiles are taken
    """
    grouped = _is_grouped(values)
    if grouped:
        # Each group becomes its own 1-D float array.
        items = [np.asarray(group, dtype=float).ravel() for group in values]
        n_items = len(items)
    else:
        flat = np.asarray(values, dtype=float)
        if flat.ndim != 1:
            raise ValueError("values must be a 1-D array or a list of arrays")
        n_items = flat.size

    block_length = int(block_length)
    n_resamples = int(n_resamples)
    if n_items < 1:
        raise ValueError("values must hold at least one item")
    if block_length < 1:
        raise ValueError("block_length must be at least 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")

    def apply_statistic(positions):
        """Build the data for the given item positions and run the statistic on it."""
        if grouped:
            chosen = [items[position] for position in positions]
            data = np.concatenate(chosen) if chosen else np.array([], dtype=float)
        else:
            data = flat[positions]
        return float(statistic(data))

    # The statistic on the data as it is (positions 0, 1, ..., n-1).
    point = apply_statistic(np.arange(n_items))

    # Step 1: all block starts at once. One row per resample, one column per block.
    blocks_per_resample = math.ceil(n_items / block_length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_items, size=(n_resamples, blocks_per_resample))

    # Offsets inside a block: 0, 1, ..., L-1.
    offsets = np.arange(block_length)

    resampled = np.empty(n_resamples, dtype=float)
    for row in range(n_resamples):
        # Step 2: start + offset, modulo n (the wrap-around). Shape (blocks, L).
        positions = (starts[row][:, None] + offsets[None, :]) % n_items
        # Step 3: join the blocks in order and keep the first n positions.
        positions = positions.ravel()[:n_items]
        # Step 4.
        resampled[row] = apply_statistic(positions)

    # Drop failed resamples (statistic was nan, e.g. a Sharpe with zero spread).
    usable = resampled[~np.isnan(resampled)]
    n_failed = int(n_resamples - usable.size)

    # Step 5: percentile interval.
    if usable.size == 0:
        lower, upper = float("nan"), float("nan")
    else:
        tail = (1.0 - level) / 2.0
        lower, upper = (float(q) for q in np.quantile(usable, [tail, 1.0 - tail]))

    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "n": int(n_items),
        "block_length": block_length,
        "n_resamples": n_resamples,
        "n_failed": n_failed,
    }


# ---------------------------------------------------------------------------
# 2. Probabilistic Sharpe ratio
# ---------------------------------------------------------------------------

def _sample_moments(returns):
    """Per-period Sharpe, count, skewness and kurtosis of a return series.

    Formulas (x = the returns with nan removed, n = their count, m = their mean):
      Sharpe   SR = m / s, with s the standard deviation using n - 1 (ddof=1)
      skewness g3 = m3 / m2**1.5
      kurtosis g4 = m4 / m2**2        (NOT excess: a normal distribution gives 3)
    where mk = mean((x - m)**k) are the plain central moments (divided by n).

    Returns (sr, n, g3, g4). Sharpe, skewness and kurtosis are nan when there are fewer
    than 2 observations or the returns have no spread.
    """
    x = np.asarray(returns, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 2:
        return float("nan"), n, float("nan"), float("nan")

    mean = x.mean()
    deviations = x - mean
    m2 = np.mean(deviations ** 2)
    if m2 <= 0.0:
        # Every return is the same number: the Sharpe ratio is undefined.
        return float("nan"), n, float("nan"), float("nan")

    std = x.std(ddof=1)
    sharpe = float(mean / std)
    skew = float(np.mean(deviations ** 3) / m2 ** 1.5)
    kurt = float(np.mean(deviations ** 4) / m2 ** 2)
    return sharpe, n, skew, kurt


def _psr_from_moments(sr, n, skew, kurt, benchmark):
    """The PSR formula itself, taking the moments directly.

    Source: Bailey and Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier",
    Journal of Risk 15(2), equation for PSR(SR*).

        PSR = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR**2) )

      SR   observed per-period Sharpe          SR*  benchmark per-period Sharpe
      n    number of observations              g3   skewness
      g4   kurtosis (normal = 3)               Phi  standard normal CDF

    The square root in the denominator is the standard error of the Sharpe estimate
    (times sqrt(n - 1)) when returns are not normal: negative skew and fat tails make the
    estimate less certain, which pulls PSR toward 0.5.

    Returns nan if an input is nan, n < 2, or the term under the square root is not
    positive.
    """
    if any(math.isnan(v) for v in (sr, skew, kurt, benchmark)) or n < 2:
        return float("nan")
    under_root = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if under_root <= 0.0:
        return float("nan")
    z = (sr - benchmark) * math.sqrt(n - 1) / math.sqrt(under_root)
    return float(_STANDARD_NORMAL.cdf(z))


def probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0):
    """Probability that the true Sharpe ratio is above `benchmark_sharpe`.

    Source: Bailey and Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".

        PSR = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR**2) )

    `returns` are per-period EXCESS returns (risk-free already subtracted). SR is their
    per-period Sharpe (mean / std with ddof=1), g3 their skewness and g4 their kurtosis
    (not excess; normal = 3). `benchmark_sharpe` (SR*) is per period too, so compare a
    daily Sharpe with a daily benchmark. nan values in `returns` are dropped.

    Returns a float between 0 and 1, or nan if the Sharpe cannot be computed.
    """
    sharpe, n, skew, kurt = _sample_moments(returns)
    return _psr_from_moments(sharpe, n, skew, kurt, float(benchmark_sharpe))


# ---------------------------------------------------------------------------
# 3. and 4. Expected maximum Sharpe and the deflated Sharpe ratio
# ---------------------------------------------------------------------------

def expected_max_sharpe(n_trials, trial_sharpe_std):
    """Expected best per-period Sharpe among N independent trials that all have true Sharpe 0.

    Source: Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio", Journal of
    Portfolio Management 40(5). It is an approximation to the expected maximum of N
    independent normal draws.

        E[max SR] = std * ( (1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - 1/(N*e)) )

      std    standard deviation of the per-period Sharpe ratios across the trials
      N      number of trials
      gamma  Euler-Mascheroni constant, 0.5772156649
      Z^-1   inverse of the standard normal CDF
      e      2.71828...

    Meaning: try N useless strategies and the best of them still shows about this Sharpe
    by luck. With one trial (or fewer) there is no selection, so the answer is 0.0.
    """
    if n_trials <= 1:
        return 0.0
    spread = float(trial_sharpe_std)
    if spread < 0.0:
        raise ValueError("trial_sharpe_std cannot be negative")
    first = _STANDARD_NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    second = _STANDARD_NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(spread * ((1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second))


def deflated_sharpe_ratio(returns, n_trials, trial_sharpe_std):
    """Deflated Sharpe ratio: PSR against the Sharpe that luck alone would produce.

    Source: Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio".

        SR0 = expected_max_sharpe(n_trials, trial_sharpe_std)
        DSR = PSR(returns, benchmark = SR0)

    `returns` are per-period excess returns. `n_trials` is how many configurations were
    tried in total. `trial_sharpe_std` is the standard deviation of the PER-PERIOD Sharpe
    ratios across those trials (same units as the Sharpe of `returns`).

    Returns a dict:
      dsr                  the deflated Sharpe ratio (a probability)
      psr_vs_zero          PSR against a Sharpe of 0, for comparison
      sharpe_per_period    SR of `returns`
      expected_max_sharpe  SR0
      n_trials, n_obs      as used
      skew, kurtosis       of `returns` (kurtosis not excess; normal = 3)
    """
    sharpe, n, skew, kurt = _sample_moments(returns)
    hurdle = expected_max_sharpe(n_trials, trial_sharpe_std)
    return {
        "dsr": _psr_from_moments(sharpe, n, skew, kurt, hurdle),
        "psr_vs_zero": _psr_from_moments(sharpe, n, skew, kurt, 0.0),
        "sharpe_per_period": sharpe,
        "expected_max_sharpe": hurdle,
        "n_trials": int(n_trials),
        "n_obs": n,
        "skew": skew,
        "kurtosis": kurt,
    }


# ---------------------------------------------------------------------------
# 5. Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------

def _sharpe_from_sums(total, total_of_squares, count):
    """Per-period Sharpe from a sum, a sum of squares and a count (works on arrays).

        mean     = total / count
        variance = (total_of_squares - total**2 / count) / (count - 1)      (ddof=1)
        Sharpe   = mean / sqrt(variance)

    This is the same number as mean / std(ddof=1) on the raw rows, but it needs only three
    totals, which is what makes the PBO fast. A column with zero variance gets -inf, so it
    ranks below every column that has a real Sharpe.
    """
    mean = total / count
    variance = (total_of_squares - total ** 2 / count) / (count - 1)
    # Rounding can leave a tiny positive or negative "variance" for a constant column.
    is_zero = variance <= _ZERO_VARIANCE_RELATIVE_TOLERANCE * (total_of_squares / count)
    safe_variance = np.where(is_zero, 1.0, variance)
    return np.where(is_zero, -np.inf, mean / np.sqrt(safe_variance))


def _cscv_split_sharpes(block_sums, block_sums_of_squares, rows_per_block, in_sample_mask):
    """In-sample and out-of-sample Sharpe of every column for every split.

    Arguments:
      block_sums             array (n_blocks, n_configs): sum of returns per block
      block_sums_of_squares  array (n_blocks, n_configs): sum of squared returns per block
      rows_per_block         rows in each block (all blocks have the same size)
      in_sample_mask         array (n_splits, n_blocks) of 0.0 / 1.0; 1.0 marks a block
                             that is in-sample for that split

    Adding the block totals of the chosen blocks gives the totals of the joined rows, so a
    matrix product does all splits at once: each split costs O(n_configs) arithmetic and
    no rows are sliced again.

    Returns (in_sample_sharpe, out_of_sample_sharpe), each of shape (n_splits, n_configs).
    """
    out_of_sample_mask = 1.0 - in_sample_mask
    blocks_in = in_sample_mask.sum(axis=1, keepdims=True)
    blocks_out = out_of_sample_mask.sum(axis=1, keepdims=True)

    in_sample = _sharpe_from_sums(in_sample_mask @ block_sums,
                                  in_sample_mask @ block_sums_of_squares,
                                  blocks_in * rows_per_block)
    out_of_sample = _sharpe_from_sums(out_of_sample_mask @ block_sums,
                                      out_of_sample_mask @ block_sums_of_squares,
                                      blocks_out * rows_per_block)
    return in_sample, out_of_sample


def pbo_cscv(returns_matrix, n_blocks=16):
    """Probability of Backtest Overfitting by combinatorially symmetric cross-validation.

    Source: Bailey, Borwein, Lopez de Prado and Zhu (2017), "The Probability of Backtest
    Overfitting", Journal of Computational Finance 20(4).

    `returns_matrix` is a DataFrame: one row per day, one column per configuration.

    Method:
      1. Drop rows from the START until the row count divides by `n_blocks`, then cut the
         rows into `n_blocks` consecutive blocks of equal size.
      2. For every way of choosing n_blocks / 2 blocks as in-sample (the other half is
         out-of-sample; rows keep their order inside each half):
           a. find the column with the best in-sample Sharpe (mean / std, ddof=1);
           b. rank all N columns by out-of-sample Sharpe, ascending: 1 = worst, N = best,
              tied columns share the average of their ranks;
           c. relative rank w = rank of the in-sample winner / (N + 1);
           d. logit = ln(w / (1 - w)). It is negative when the in-sample winner lands in
              the bottom half out-of-sample.
      3. PBO = share of splits with logit <= 0.

    With 16 blocks there are C(16, 8) = 12,870 splits. A PBO near 0 means the in-sample
    winner keeps winning out-of-sample; near 0.5 means picking the winner is no better
    than chance; near 1 means the in-sample winner tends to do worst afterwards.

    Details: a column with zero variance inside a split gets Sharpe -inf for ranking. If
    several columns tie for the best in-sample Sharpe, the leftmost one is taken.

    Raises ValueError for fewer than 2 columns, fewer than 2 rows per block, an odd or too
    small `n_blocks`, or nan / infinite values in the matrix.

    Returns a dict:
      pbo                        share of splits with logit <= 0
      n_splits                   C(n_blocks, n_blocks / 2)
      n_configs                  number of columns N
      rows_used, rows_dropped    after and by the trimming in step 1
      logits                     numpy array, one logit per split, in the order of
                                 itertools.combinations over the block numbers
      median_logit               median of `logits`
      oos_rank_of_is_best_mean   mean out-of-sample rank (1 .. N) of the in-sample winner
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        raise ValueError("returns_matrix must be a pandas DataFrame")
    n_blocks = int(n_blocks)
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be an even number, at least 2")

    data = returns_matrix.to_numpy(dtype=float)
    n_rows, n_configs = data.shape
    if n_configs < 2:
        raise ValueError("PBO needs at least 2 configurations (columns)")
    rows_per_block = n_rows // n_blocks
    if rows_per_block < 2:
        raise ValueError("PBO needs at least 2 rows per block")
    if not np.isfinite(data).all():
        raise ValueError("returns_matrix contains nan or infinite values")

    # Step 1: trim from the start, then cut into blocks.
    rows_used = rows_per_block * n_blocks
    rows_dropped = n_rows - rows_used
    blocks = data[rows_dropped:].reshape(n_blocks, rows_per_block, n_configs)

    # Totals per block, computed once. Shape (n_blocks, n_configs).
    block_sums = blocks.sum(axis=1)
    block_sums_of_squares = (blocks ** 2).sum(axis=1)

    # Step 2: every choice of half the blocks, as rows of 0/1 flags.
    half = n_blocks // 2
    combinations = np.array(list(itertools.combinations(range(n_blocks), half)))
    n_splits = combinations.shape[0]
    in_sample_mask = np.zeros((n_splits, n_blocks), dtype=float)
    in_sample_mask[np.arange(n_splits)[:, None], combinations] = 1.0

    in_sample, out_of_sample = _cscv_split_sharpes(
        block_sums, block_sums_of_squares, rows_per_block, in_sample_mask)

    # 2a. The in-sample winner of each split (argmax takes the leftmost on a tie).
    winner = np.argmax(in_sample, axis=1)

    # 2b. Its out-of-sample rank. Ascending rank with average ties =
    #     (columns strictly worse) + (columns equal to it, itself included, + 1) / 2.
    winner_oos = out_of_sample[np.arange(n_splits), winner][:, None]
    strictly_worse = (out_of_sample < winner_oos).sum(axis=1)
    equal = (out_of_sample == winner_oos).sum(axis=1)
    rank = strictly_worse + (equal + 1) / 2.0

    # 2c and 2d. w is strictly between 0 and 1 because 1 <= rank <= N.
    relative_rank = rank / (n_configs + 1)
    logits = np.log(relative_rank / (1.0 - relative_rank))

    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "n_splits": int(n_splits),
        "n_configs": int(n_configs),
        "rows_used": int(rows_used),
        "rows_dropped": int(rows_dropped),
        "logits": logits,
        "median_logit": float(np.median(logits)),
        "oos_rank_of_is_best_mean": float(rank.mean()),
    }


# ---------------------------------------------------------------------------
# 6. OLS with Newey-West standard errors
# ---------------------------------------------------------------------------

def ols_newey_west(y, X, lags=21):
    """OLS of `y` on the columns of `X` plus an intercept, with Newey-West t-statistics.

    Sources: Newey and West (1987), "A simple, positive semi-definite, heteroskedasticity
    and autocorrelation consistent covariance matrix", Econometrica 55(3). With lags=0 it
    reduces to White (1980) heteroskedasticity-robust errors (the "HC1" version, because
    of the n / (n - k) factor).

    Model, per period t:   y_t = alpha + beta_1 * x1_t + ... + e_t

    Steps (x_t is the row of regressors including the leading 1, n rows, k columns):
      coefficients  b = (X'X)^-1 X'y
      residuals     e = y - X b
      HAC "meat"    S = sum_t e_t^2 x_t x_t'
                        + sum_{l=1..L} w_l * sum_t e_t e_{t-l} (x_t x_{t-l}' + x_{t-l} x_t')
                    with Bartlett weights w_l = 1 - l / (L + 1), L = `lags`
      covariance    cov = n / (n - k) * (X'X)^-1 S (X'X)^-1
      t-statistic   coefficient / sqrt(its diagonal entry of cov)

    Why: daily strategy returns are autocorrelated and their volatility changes over time.
    Plain OLS standard errors ignore both and usually overstate the t-statistic.

    Data handling: `y` (Series) and `X` (DataFrame, or a Series for one regressor) are
    aligned on their common index, sorted by that index, and any row with a nan in `y` or
    `X` is dropped. The lags count rows of what is left.

    Raises ValueError if there are not more rows than coefficients.

    Returns a dict:
      alpha             intercept, per period
      alpha_annualized  alpha * 252
      alpha_t           Newey-West t-statistic of alpha
      alpha_t_ols       plain OLS t-statistic of alpha (cov = s^2 (X'X)^-1,
                        s^2 = sum(e^2) / (n - k)), for comparison
      betas             dict column name -> coefficient
      beta_t            dict column name -> Newey-West t-statistic
      r_squared         1 - sum(e^2) / sum((y - mean(y))^2); nan if y is constant
      n_obs             rows used
      lags              L as given
    """
    lags = int(lags)
    if lags < 0:
        raise ValueError("lags cannot be negative")
    if isinstance(X, pd.Series):
        X = X.to_frame()

    # Align on the dates both have, in date order, and drop incomplete rows.
    common = y.index.intersection(X.index).sort_values()
    y_aligned = y.loc[common]
    x_aligned = X.loc[common]
    complete = (y_aligned.notna() & x_aligned.notna().all(axis=1)).to_numpy()

    y_values = y_aligned.to_numpy(dtype=float)[complete]
    x_values = x_aligned.to_numpy(dtype=float)[complete]
    names = list(X.columns)

    # Design matrix: a column of ones (the intercept) followed by the regressors.
    n = y_values.size
    design = np.column_stack([np.ones(n), x_values])
    k = design.shape[1]
    if n <= k:
        raise ValueError(f"need more rows than coefficients: {n} rows, {k} coefficients")

    # Coefficients and residuals.
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ (design.T @ y_values)
    residuals = y_values - design @ coefficients

    # Each row of `scores` is e_t * x_t. Then sum_t e_t^2 x_t x_t' = scores' scores.
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    # Autocovariance terms. Lags beyond the sample length have no pairs and add nothing.
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        # sum_t (e_t x_t)(e_{t-lag} x_{t-lag})'
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)

    small_sample = n / (n - k)
    hac_covariance = small_sample * (xtx_inverse @ meat @ xtx_inverse)
    hac_errors = np.sqrt(np.diag(hac_covariance))

    # Plain OLS errors, for the comparison t-statistic.
    sum_squared_residuals = float(residuals @ residuals)
    ols_errors = np.sqrt(np.diag(xtx_inverse) * sum_squared_residuals / (n - k))

    # Division by a zero standard error (a perfect fit) gives inf or nan, not a crash.
    with np.errstate(divide="ignore", invalid="ignore"):
        hac_t = coefficients / hac_errors
        ols_t = coefficients / ols_errors

    total_sum_of_squares = float(np.sum((y_values - y_values.mean()) ** 2))
    if total_sum_of_squares > 0.0:
        r_squared = 1.0 - sum_squared_residuals / total_sum_of_squares
    else:
        r_squared = float("nan")

    alpha = float(coefficients[0])
    return {
        "alpha": alpha,
        "alpha_annualized": alpha * TRADING_DAYS_PER_YEAR,
        "alpha_t": float(hac_t[0]),
        "alpha_t_ols": float(ols_t[0]),
        "betas": {name: float(value) for name, value in zip(names, coefficients[1:])},
        "beta_t": {name: float(value) for name, value in zip(names, hac_t[1:])},
        "r_squared": float(r_squared),
        "n_obs": int(n),
        "lags": lags,
    }


# ---------------------------------------------------------------------------
# 7. Walk-forward selection
# ---------------------------------------------------------------------------

def _column_sharpes(frame):
    """Per-period Sharpe (mean / std, ddof=1) of every column; -inf where it is undefined."""
    mean = frame.mean()
    std = frame.std(ddof=1)
    sharpe = mean / std.where(std > 0.0)        # zero or missing std becomes nan ...
    return sharpe.fillna(-np.inf)               # ... and nan becomes "worst possible"


def _compounded_return(series):
    """Compounded return of a series of simple returns: (1 + r1)(1 + r2)... - 1."""
    return float((1.0 + series.dropna()).prod() - 1.0)


def walk_forward_selection(returns_matrix, baseline, first_test_year, min_history_days=252):
    """Year by year: pick the best configuration on past data, then see what it earned.

    Source: the walk-forward test of protocol.md section 5 (the usual anchored
    walk-forward: the history window starts at the first row and grows each year).

    `returns_matrix` is a DataFrame of daily returns: rows = days (a DatetimeIndex), one
    column per configuration. `baseline` is the name of the frozen baseline's column.

    For each calendar year Y from `first_test_year` to the last year in the index:
      1. history = ALL rows dated strictly before 1 January of Y. Nothing from Y or later
         is looked at when choosing, so there is no lookahead.
      2. If history has fewer than `min_history_days` rows, or Y itself has no rows, the
         year is skipped (no row in the result).
      3. selected = the column with the best Sharpe on history (daily mean / std, ddof=1;
         the leftmost column on a tie; a column with no spread counts as worst).
      4. Record what `selected` and `baseline` returned in Y, compounded:
         (1 + r1)(1 + r2)...(1 + rn) - 1.

    Returns a DataFrame, one row per tested year, with columns:
      year, selected, selected_return, baseline_return, selected_is_baseline, n_days
    """
    if baseline not in returns_matrix.columns:
        raise ValueError(f"baseline column {baseline!r} is not in returns_matrix")

    # The calendar year of every row. Only this is used to split past from present.
    row_years = np.asarray(pd.DatetimeIndex(returns_matrix.index).year)

    columns = ["year", "selected", "selected_return", "baseline_return",
               "selected_is_baseline", "n_days"]
    rows = []
    if row_years.size == 0:
        return pd.DataFrame(rows, columns=columns)

    for year in range(int(first_test_year), int(row_years.max()) + 1):
        history = returns_matrix[row_years < year]      # strictly earlier years only
        test = returns_matrix[row_years == year]        # the year being tested
        if len(history) < min_history_days or len(test) == 0:
            continue

        # idxmax returns the first (leftmost) column on a tie.
        selected = _column_sharpes(history).idxmax()

        rows.append({
            "year": year,
            "selected": selected,
            "selected_return": _compounded_return(test[selected]),
            "baseline_return": _compounded_return(test[baseline]),
            "selected_is_baseline": bool(selected == baseline),
            "n_days": int(len(test)),
        })

    return pd.DataFrame(rows, columns=columns)
