"""Tests for src/research/robustness.py against values worked out by hand.

Every test uses made-up numbers built inside the test. Nothing here reads the data folder,
the trial log or the network.
"""

import itertools
import math
import time
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.research import robustness


# ---------------------------------------------------------------------------
# 1. Circular block bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_constant_series_has_zero_width_interval():
    # Every resample of a constant series is the same constant, so its mean is 2.0
    # every time: lower == upper == point == 2.0. (2.0 is exact in floating point.)
    result = robustness.circular_block_bootstrap_ci(np.full(50, 2.0), np.mean,
                                                    block_length=5, n_resamples=200)
    assert result["point"] == 2.0
    assert result["lower"] == 2.0
    assert result["upper"] == 2.0
    assert result["n"] == 50
    assert result["block_length"] == 5
    assert result["n_resamples"] == 200
    assert result["n_failed"] == 0


def test_bootstrap_same_seed_same_interval_different_seed_different_interval():
    values = np.random.default_rng(1).normal(0.0, 1.0, 300)
    first = robustness.circular_block_bootstrap_ci(values, np.mean, 10, n_resamples=300, seed=7)
    again = robustness.circular_block_bootstrap_ci(values, np.mean, 10, n_resamples=300, seed=7)
    other = robustness.circular_block_bootstrap_ci(values, np.mean, 10, n_resamples=300, seed=8)
    assert (first["lower"], first["upper"]) == (again["lower"], again["upper"])
    assert (first["lower"], first["upper"]) != (other["lower"], other["upper"])
    # The point estimate does not depend on the seed.
    assert first["point"] == other["point"] == pytest.approx(values.mean())


def test_bootstrap_interval_of_the_mean_contains_the_true_mean():
    # 500 draws from a normal with true mean 0.5 and standard deviation 1.
    # The standard error of the mean is about 1 / sqrt(500) = 0.045, so a 95% interval is
    # roughly the sample mean +/- 0.09 and should cover 0.5 (seeded, so it is repeatable).
    values = np.random.default_rng(42).normal(0.5, 1.0, 500)
    result = robustness.circular_block_bootstrap_ci(values, np.mean, block_length=5)
    assert result["lower"] < 0.5 < result["upper"]
    assert result["lower"] < result["point"] < result["upper"]
    assert result["upper"] - result["lower"] < 0.4      # and it is not uselessly wide
    assert result["n_resamples"] == 2000                # the protocol's default


def test_bootstrap_resamples_wrap_around_and_keep_the_length():
    # Series [1, 2, 3] with block length 2: ceil(3 / 2) = 2 blocks = 4 items, cut to 3.
    # So a resample is [a, next(a), b] where next() is the circular successor:
    # next(1) = 2, next(2) = 3, next(3) = 1  (the wrap-around).
    seen = []

    def recording_mean(data):
        seen.append(tuple(data))
        return data.mean()

    robustness.circular_block_bootstrap_ci(np.array([1.0, 2.0, 3.0]), recording_mean,
                                           block_length=2, n_resamples=200, seed=3)
    resamples = seen[1:]                    # the first call is the original data (the point)
    assert seen[0] == (1.0, 2.0, 3.0)
    assert len(resamples) == 200
    successor = {1.0: 2.0, 2.0: 3.0, 3.0: 1.0}
    for resample in resamples:
        assert len(resample) == 3                           # same size as the original
        assert resample[1] == successor[resample[0]]        # a block is consecutive items
    # A block starting on the last item wrapped to the first one at least once.
    assert any(resample[:2] == (3.0, 1.0) for resample in resamples)
    # Every item can start a block.
    assert {resample[0] for resample in resamples} == {1.0, 2.0, 3.0}


def test_bootstrap_block_longer_than_the_series_gives_rotations():
    # Block length 5 on 3 items: one block, start s, positions s, s+1, s+2 modulo 3.
    # Every resample is a rotation of the series, so its sum is always 1 + 2 + 3 = 6.
    result = robustness.circular_block_bootstrap_ci(np.array([1.0, 2.0, 3.0]), np.sum,
                                                    block_length=5, n_resamples=100)
    assert result["lower"] == result["upper"] == result["point"] == 6.0


def test_bootstrap_grouped_mode_resamples_whole_groups():
    # Three groups (think: trades grouped by exit month). Group sizes 2, 1 and 3.
    groups = [np.array([1.0, 1.0]), np.array([10.0]), np.array([100.0, 100.0, 100.0])]
    seen = []

    def recording_sum(data):
        seen.append(np.array(data))
        return data.sum()

    result = robustness.circular_block_bootstrap_ci(groups, recording_sum, block_length=1,
                                                    n_resamples=300, seed=5)
    # Point: the groups joined as they are: 1 + 1 + 10 + 100 * 3 = 312.
    assert result["point"] == 312.0
    assert result["n"] == 3                 # three groups, not six trades
    # Each resample is 3 whole groups, so its sum is a + b + c with each of a, b, c one
    # of the group sums 2, 10, 300. The smallest is 2 * 3 = 6, the largest 300 * 3 = 900.
    group_sums = (2.0, 10.0, 300.0)
    allowed = {a + b + c for a in group_sums for b in group_sums for c in group_sums}
    assert all(data.sum() in allowed for data in seen[1:])
    # Groups are never split: the value 1.0 always comes in pairs, 100.0 in threes.
    assert all((data == 1.0).sum() % 2 == 0 for data in seen[1:])
    assert all((data == 100.0).sum() % 3 == 0 for data in seen[1:])
    # Resamples differ in length (3 groups of different sizes), unlike the flat case.
    assert len({data.size for data in seen[1:]}) > 1
    assert 6.0 <= result["lower"] <= result["upper"] <= 900.0


def test_bootstrap_grouped_blocks_wrap_around():
    # Groups [1], [2], [3] with block length 2: same shape as the flat wrap-around test,
    # so the second item is always the circular successor of the first.
    groups = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    seen = []

    def recording_mean(data):
        seen.append(tuple(data))
        return data.mean()

    robustness.circular_block_bootstrap_ci(groups, recording_mean, block_length=2,
                                           n_resamples=100, seed=11)
    successor = {1.0: 2.0, 2.0: 3.0, 3.0: 1.0}
    assert all(resample[1] == successor[resample[0]] for resample in seen[1:])
    assert any(resample[:2] == (3.0, 1.0) for resample in seen[1:])


def test_bootstrap_counts_and_drops_nan_resamples():
    # A statistic that fails (nan) whenever the resample starts with the value 3.0.
    # Block length 1 on [1, 2, 3]: the first item is 3.0 in about a third of resamples.
    def fails_on_three(data):
        return float("nan") if data[0] == 3.0 else data.mean()

    result = robustness.circular_block_bootstrap_ci(np.array([1.0, 2.0, 3.0]), fails_on_three,
                                                    block_length=1, n_resamples=300)
    assert 0 < result["n_failed"] < 300
    assert not math.isnan(result["lower"]) and not math.isnan(result["upper"])

    # If every resample fails the interval is nan and all of them are counted.
    always = robustness.circular_block_bootstrap_ci(np.array([1.0, 2.0]), lambda data: np.nan,
                                                    block_length=1, n_resamples=20)
    assert always["n_failed"] == 20
    assert math.isnan(always["lower"]) and math.isnan(always["upper"])


def test_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        robustness.circular_block_bootstrap_ci(np.array([]), np.mean, 2)
    with pytest.raises(ValueError):
        robustness.circular_block_bootstrap_ci(np.array([1.0, 2.0]), np.mean, 0)
    with pytest.raises(ValueError):
        robustness.circular_block_bootstrap_ci(np.array([1.0, 2.0]), np.mean, 1, level=1.0)


# ---------------------------------------------------------------------------
# 2. Probabilistic Sharpe ratio
# ---------------------------------------------------------------------------

def test_psr_formula_hand_worked_case():
    # SR = 0.1 per period, n = 101, skewness 0, kurtosis 3, benchmark 0.
    #   numerator   = (0.1 - 0) * sqrt(101 - 1) = 0.1 * 10 = 1.0
    #   denominator = sqrt(1 - 0 * 0.1 + (3 - 1) / 4 * 0.1**2) = sqrt(1 + 0.5 * 0.01)
    #               = sqrt(1.005) = 1.0024969
    #   z = 1.0 / 1.0024969 = 0.9975093,  PSR = Phi(0.9975093) = 0.8407 (to 4 places)
    z = 0.1 * 10 / math.sqrt(1 + 0.5 * 0.01)
    assert z == pytest.approx(0.9975093, abs=1e-7)
    result = robustness._psr_from_moments(0.1, 101, 0.0, 3.0, 0.0)
    assert result == pytest.approx(NormalDist().cdf(z), abs=1e-12)
    assert result == pytest.approx(0.8407, abs=1e-4)


def test_psr_formula_with_a_benchmark_and_non_normal_moments():
    # SR = 0.2, SR* = 0.1, n = 50, skewness -1, kurtosis 5.
    #   numerator   = (0.2 - 0.1) * sqrt(49) = 0.7
    #   denominator = sqrt(1 - (-1) * 0.2 + (5 - 1) / 4 * 0.04) = sqrt(1 + 0.2 + 0.04)
    #               = sqrt(1.24)
    expected = NormalDist().cdf(0.7 / math.sqrt(1.24))
    assert robustness._psr_from_moments(0.2, 50, -1.0, 5.0, 0.1) == pytest.approx(expected, abs=1e-12)


def test_psr_public_function_agrees_with_the_formula_on_a_random_series():
    returns = np.random.default_rng(3).normal(0.001, 0.01, 400)
    # The moments, computed here directly from their definitions.
    mean = returns.mean()
    sharpe = mean / returns.std(ddof=1)
    m2 = np.mean((returns - mean) ** 2)
    skew = np.mean((returns - mean) ** 3) / m2 ** 1.5
    kurt = np.mean((returns - mean) ** 4) / m2 ** 2
    for benchmark in (0.0, 0.05):
        expected = robustness._psr_from_moments(sharpe, 400, skew, kurt, benchmark)
        actual = robustness.probabilistic_sharpe_ratio(returns, benchmark)
        assert actual == pytest.approx(expected, abs=1e-12)
    # A pandas Series gives the same answer as the array.
    assert robustness.probabilistic_sharpe_ratio(pd.Series(returns)) == pytest.approx(
        robustness.probabilistic_sharpe_ratio(returns), abs=1e-15)


def test_psr_of_a_zero_mean_series_is_one_half():
    # Mean exactly 0 -> SR = 0 -> z = 0 -> Phi(0) = 0.5, whatever the other moments are.
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.03, -0.03])
    assert robustness.probabilistic_sharpe_ratio(returns) == pytest.approx(0.5, abs=1e-12)


def test_psr_negative_skew_lowers_psr_for_the_same_sharpe():
    # With SR > 0 the term -g3 * SR grows the denominator when g3 < 0, so z shrinks.
    #   skew  0: denominator^2 = 1 + 0.5 * 0.01           = 1.005
    #   skew -2: denominator^2 = 1 + 2 * 0.1 + 0.5 * 0.01 = 1.205
    symmetric = robustness._psr_from_moments(0.1, 101, 0.0, 3.0, 0.0)
    negative = robustness._psr_from_moments(0.1, 101, -2.0, 3.0, 0.0)
    assert negative < symmetric
    assert negative == pytest.approx(NormalDist().cdf(1.0 / math.sqrt(1.205)), abs=1e-12)


def test_psr_is_nan_when_the_sharpe_is_undefined():
    assert math.isnan(robustness.probabilistic_sharpe_ratio(np.array([0.01])))        # one point
    assert math.isnan(robustness.probabilistic_sharpe_ratio(np.full(10, 0.01)))       # no spread


# ---------------------------------------------------------------------------
# 3. Expected maximum Sharpe
# ---------------------------------------------------------------------------

def test_expected_max_sharpe_is_zero_for_one_trial():
    assert robustness.expected_max_sharpe(1, 0.5) == 0.0
    assert robustness.expected_max_sharpe(0, 0.5) == 0.0


def test_expected_max_sharpe_hand_value_for_ten_trials():
    # N = 10, std = 1, gamma = 0.5772156649:
    #   first  = Z^-1(1 - 1/10)       = Z^-1(0.9)       = 1.2815516
    #   second = Z^-1(1 - 1/(10 * e)) = Z^-1(0.9632121) = 1.7891...
    #   result = (1 - gamma) * first + gamma * second
    #          = 0.4227843 * 1.2815516 + 0.5772157 * 1.7891  = 1.5746 (to 4 places)
    gamma = 0.5772156649
    first = NormalDist().inv_cdf(1 - 1 / 10)
    second = NormalDist().inv_cdf(1 - 1 / (10 * math.e))
    assert first == pytest.approx(1.2815516, abs=1e-6)
    assert second == pytest.approx(1.7891, abs=1e-3)
    expected = (1 - gamma) * first + gamma * second
    assert robustness.expected_max_sharpe(10, 1.0) == pytest.approx(expected, abs=1e-12)
    assert robustness.expected_max_sharpe(10, 1.0) == pytest.approx(1.5746, abs=1e-3)
    # It scales with the spread of the trial Sharpes.
    assert robustness.expected_max_sharpe(10, 0.02) == pytest.approx(0.02 * expected, abs=1e-12)


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    values = [robustness.expected_max_sharpe(n, 1.0) for n in (2, 5, 14, 100, 1000)]
    assert all(later > earlier for earlier, later in zip(values, values[1:]))
    assert values[0] > 0.0


# ---------------------------------------------------------------------------
# 4. Deflated Sharpe ratio
# ---------------------------------------------------------------------------

def _good_returns():
    """A seeded series with a clearly positive daily Sharpe (about 0.1)."""
    return np.random.default_rng(11).normal(0.001, 0.01, 1000)


def test_dsr_with_one_trial_equals_psr_against_zero():
    returns = _good_returns()
    result = robustness.deflated_sharpe_ratio(returns, n_trials=1, trial_sharpe_std=0.05)
    # One trial -> expected maximum Sharpe 0 -> the benchmark is 0 -> DSR is PSR vs 0.
    assert result["expected_max_sharpe"] == 0.0
    assert result["dsr"] == result["psr_vs_zero"]
    assert result["psr_vs_zero"] == pytest.approx(robustness.probabilistic_sharpe_ratio(returns), abs=1e-15)
    assert result["n_trials"] == 1
    assert result["n_obs"] == 1000
    assert result["sharpe_per_period"] == pytest.approx(returns.mean() / returns.std(ddof=1))
    assert set(result) == {"dsr", "psr_vs_zero", "sharpe_per_period", "expected_max_sharpe",
                           "n_trials", "n_obs", "skew", "kurtosis"}
    # Normal draws: skewness near 0, kurtosis near 3 (not excess).
    assert abs(result["skew"]) < 0.3
    assert 2.5 < result["kurtosis"] < 3.5


def test_dsr_is_below_psr_with_several_trials_and_falls_as_trials_grow():
    returns = _good_returns()
    few = robustness.deflated_sharpe_ratio(returns, n_trials=5, trial_sharpe_std=0.03)
    many = robustness.deflated_sharpe_ratio(returns, n_trials=50, trial_sharpe_std=0.03)
    # The hurdle is positive, so DSR < PSR vs 0; a higher hurdle (more trials) lowers it more.
    assert 0.0 < few["expected_max_sharpe"] < many["expected_max_sharpe"]
    assert few["dsr"] < few["psr_vs_zero"]
    assert many["dsr"] < few["dsr"]
    # DSR is exactly PSR with the hurdle as benchmark.
    assert few["dsr"] == pytest.approx(
        robustness.probabilistic_sharpe_ratio(returns, few["expected_max_sharpe"]), abs=1e-15)


def test_dsr_with_zero_spread_of_trial_sharpes_equals_psr():
    # std = 0 -> hurdle = 0 * (...) = 0 -> no deflation, however many trials.
    result = robustness.deflated_sharpe_ratio(_good_returns(), n_trials=14, trial_sharpe_std=0.0)
    assert result["dsr"] == result["psr_vs_zero"]


# ---------------------------------------------------------------------------
# 5. PBO by CSCV
# ---------------------------------------------------------------------------

def _wiggle(n_rows):
    """+0.25, -0.25, +0.25, ... : a zero-mean pattern that gives every block some spread."""
    return np.where(np.arange(n_rows) % 2 == 0, 0.25, -0.25)


def _slow_pbo(frame, n_blocks):
    """The PBO done the slow, obvious way: re-slice the rows for every split."""
    n_rows, n_configs = frame.shape
    rows_per_block = n_rows // n_blocks
    trimmed = frame.iloc[n_rows - rows_per_block * n_blocks:]
    blocks = [trimmed.iloc[i * rows_per_block:(i + 1) * rows_per_block] for i in range(n_blocks)]
    logits, in_sample_all, out_of_sample_all = [], [], []
    for chosen in itertools.combinations(range(n_blocks), n_blocks // 2):
        in_sample = pd.concat([blocks[i] for i in range(n_blocks) if i in chosen])
        out_of_sample = pd.concat([blocks[i] for i in range(n_blocks) if i not in chosen])
        in_sharpe = in_sample.mean() / in_sample.std(ddof=1)
        out_sharpe = out_of_sample.mean() / out_of_sample.std(ddof=1)
        best = in_sharpe.idxmax()
        rank = out_sharpe.rank(method="average")[best]      # ascending, 1 = worst
        w = rank / (n_configs + 1)
        logits.append(math.log(w / (1 - w)))
        in_sample_all.append(in_sharpe.to_numpy())
        out_of_sample_all.append(out_sharpe.to_numpy())
    return np.array(logits), np.array(in_sample_all), np.array(out_of_sample_all)


def test_pbo_is_zero_when_one_column_is_best_in_every_block():
    # Column "best" = wiggle + 0.10 everywhere; the others = the same wiggle + a smaller
    # mean. All columns have the same spread in any set of blocks, so the column with the
    # highest mean has the highest Sharpe in-sample AND out-of-sample for every split.
    # N = 4, rank of the winner out-of-sample = 4, w = 4 / 5, logit = ln(0.8 / 0.2) = ln 4.
    wiggle = _wiggle(48)
    frame = pd.DataFrame({"a": wiggle + 0.01, "best": wiggle + 0.10,
                          "c": wiggle + 0.02, "d": wiggle - 0.03})
    result = robustness.pbo_cscv(frame, n_blocks=6)
    assert result["pbo"] == 0.0
    assert result["n_splits"] == math.comb(6, 3) == 20
    assert result["n_configs"] == 4
    assert np.allclose(result["logits"], math.log(4.0))
    assert result["median_logit"] == pytest.approx(math.log(4.0))
    assert result["oos_rank_of_is_best_mean"] == pytest.approx(4.0)


def test_pbo_is_one_when_the_in_sample_winner_always_loses_out_of_sample():
    # Two columns, 6 blocks of 4 rows. Block means:
    #     A: +0.5 +0.5 +0.5 -0.5 -0.5 -0.5        B = the opposite sign in every block.
    # Both carry the same +/-0.25 wiggle. For any set of blocks, B's block means are A's
    # with the sign flipped, so mean(B) = -mean(A) and std(B) = std(A): Sharpe(B) = -Sharpe(A).
    # In-sample is 3 blocks. If j of them are from the first three, A's in-sample mean is
    # 0.5 * (j - (3 - j)) / 3 = 0.5 * (2j - 3) / 3. 2j - 3 is odd, so it is never zero:
    #     j >= 2: A wins in-sample. Out-of-sample holds 3 - j <= 1 of the first three
    #             blocks, so A's mean there is negative and A is the WORSE column.
    #     j <= 1: B wins in-sample, and by the same argument B is worse out-of-sample.
    # So for all C(6, 3) = 20 splits the in-sample winner has rank 1 of 2 out-of-sample:
    #     w = 1 / (2 + 1) = 1/3, logit = ln((1/3) / (2/3)) = ln(1/2) = -ln 2, PBO = 1.0.
    wiggle = _wiggle(24)                                    # 4 rows per block
    block_means = np.repeat([0.5, 0.5, 0.5, -0.5, -0.5, -0.5], 4)
    frame = pd.DataFrame({"A": wiggle + block_means, "B": wiggle - block_means})
    result = robustness.pbo_cscv(frame, n_blocks=6)
    assert result["pbo"] == 1.0
    assert result["n_splits"] == 20
    assert np.allclose(result["logits"], -math.log(2.0))
    assert result["median_logit"] == pytest.approx(-math.log(2.0))
    assert result["oos_rank_of_is_best_mean"] == pytest.approx(1.0)


def test_pbo_ties_count_as_overfit_with_eight_blocks():
    # Same construction with 8 blocks: A has +0.5 in blocks 0-3 and -0.5 in blocks 4-7.
    # In-sample is 4 blocks, j of them from the first four; A's mean is 0.5 * (2j - 4) / 4.
    #   j = 2: both columns have mean exactly 0 in-sample and out-of-sample (the numbers
    #          are binary fractions, so the sums are exact). Tie: the leftmost column is
    #          taken, its out-of-sample rank is the average of 1 and 2 = 1.5,
    #          w = 1.5 / 3 = 0.5, logit = ln(1) = 0. There are C(4,2) * C(4,2) = 36 such splits.
    #   j != 2: the in-sample winner is the out-of-sample loser, logit = -ln 2,
    #          in the other C(8, 4) - 36 = 70 - 36 = 34 splits.
    # Every logit is <= 0, so PBO = 1.0 exactly.
    wiggle = _wiggle(32)
    block_means = np.repeat([0.5] * 4 + [-0.5] * 4, 4)
    frame = pd.DataFrame({"A": wiggle + block_means, "B": wiggle - block_means})
    result = robustness.pbo_cscv(frame, n_blocks=8)
    assert result["n_splits"] == math.comb(8, 4) == 70
    assert result["pbo"] == 1.0
    assert int(np.sum(result["logits"] == 0.0)) == 36
    assert int(np.sum(np.isclose(result["logits"], -math.log(2.0)))) == 34
    # Mean rank: (36 * 1.5 + 34 * 1.0) / 70 = 88 / 70.
    assert result["oos_rank_of_is_best_mean"] == pytest.approx(88.0 / 70.0)


def test_pbo_of_pure_noise_is_near_one_half():
    # 10 columns of independent noise: the in-sample winner is as likely to land in the
    # top half as the bottom half out-of-sample, so PBO should be around 0.5.
    # 12 blocks = C(12, 6) = 924 splits.
    #
    # Careful: the 924 splits all reuse the same rows, so they are far from independent.
    # The PBO of ONE noise data set scatters widely around 0.5 (standard deviation about
    # 0.2 here; seed 2026 gives 0.90). So the test checks one fixed seed that lands in the
    # band (seed 0 gives 0.57) and, more telling, the average over 30 seeded data sets.
    def noise_pbo(seed):
        noise = np.random.default_rng(seed).normal(0.0, 0.01, size=(1200, 10))
        return robustness.pbo_cscv(pd.DataFrame(noise), n_blocks=12)

    result = noise_pbo(0)
    assert result["n_splits"] == math.comb(12, 6) == 924
    assert result["n_configs"] == 10
    assert 0.3 < result["pbo"] < 0.7
    assert result["logits"].shape == (924,)
    assert 1.0 <= result["oos_rank_of_is_best_mean"] <= 10.0

    average = np.mean([noise_pbo(seed)["pbo"] for seed in range(30)])
    assert 0.4 < average < 0.6


def test_pbo_rows_dropped_come_from_the_start():
    # 50 rows, 6 blocks: 50 // 6 = 8 rows per block, 48 used, 2 dropped from the START.
    noise = np.random.default_rng(8).normal(0.0, 0.01, size=(50, 3))
    frame = pd.DataFrame(noise, columns=["a", "b", "c"])
    result = robustness.pbo_cscv(frame, n_blocks=6)
    assert result["rows_used"] == 48
    assert result["rows_dropped"] == 2
    # Changing the two dropped rows changes nothing; so the first two rows are not used.
    changed = frame.copy()
    changed.iloc[:2] = 99.0
    assert np.array_equal(robustness.pbo_cscv(changed, n_blocks=6)["logits"], result["logits"])
    # Dropping them by hand gives the same answer with nothing dropped.
    by_hand = robustness.pbo_cscv(frame.iloc[2:], n_blocks=6)
    assert by_hand["rows_dropped"] == 0
    assert np.array_equal(by_hand["logits"], result["logits"])


def test_pbo_fast_block_sums_match_a_slow_direct_computation():
    # 5 columns with different means, 43 rows (so 3 rows are trimmed), 8 blocks of 5 rows.
    rng = np.random.default_rng(77)
    data = rng.normal(0.0, 0.01, size=(43, 5)) + np.array([0.0, 0.001, -0.001, 0.002, 0.0005])
    frame = pd.DataFrame(data, columns=list("abcde"))
    slow_logits, slow_in, slow_out = _slow_pbo(frame, n_blocks=8)

    # (1) the whole function, split by split.
    result = robustness.pbo_cscv(frame, n_blocks=8)
    assert result["rows_dropped"] == 3
    assert np.allclose(result["logits"], slow_logits, atol=1e-12)
    assert result["pbo"] == pytest.approx(np.mean(slow_logits <= 0.0))
    assert result["median_logit"] == pytest.approx(np.median(slow_logits))

    # (2) the Sharpe ratios themselves, from block sums versus from the re-sliced rows.
    blocks = data[3:].reshape(8, 5, 5)                      # (blocks, rows per block, columns)
    combos = list(itertools.combinations(range(8), 4))
    mask = np.zeros((len(combos), 8))
    for row, chosen in enumerate(combos):
        mask[row, list(chosen)] = 1.0
    fast_in, fast_out = robustness._cscv_split_sharpes(
        blocks.sum(axis=1), (blocks ** 2).sum(axis=1), 5, mask)
    assert np.allclose(fast_in, slow_in, rtol=1e-9, atol=1e-12)
    assert np.allclose(fast_out, slow_out, rtol=1e-9, atol=1e-12)


def test_pbo_zero_variance_column_ranks_last():
    # "flat" is the same number every day: no spread, Sharpe -inf, so it can never be the
    # in-sample winner and it sits at rank 1 out-of-sample. "good" beats "poor" everywhere
    # (same wiggle, higher mean), so the winner "good" has rank 3 of 3 in every split:
    # w = 3 / 4, logit = ln(3), PBO = 0.
    wiggle = _wiggle(24)
    frame = pd.DataFrame({"flat": np.full(24, 0.3), "good": wiggle + 0.1, "poor": wiggle - 0.1})
    result = robustness.pbo_cscv(frame, n_blocks=6)
    assert result["pbo"] == 0.0
    assert np.allclose(result["logits"], math.log(3.0))
    # The helper reports -inf for the flat column (0.3 is not exact in binary, so this
    # also checks the rounding tolerance).
    blocks = frame.to_numpy().reshape(6, 4, 3)
    mask = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    in_sample, out_of_sample = robustness._cscv_split_sharpes(
        blocks.sum(axis=1), (blocks ** 2).sum(axis=1), 4, mask)
    assert in_sample[0, 0] == -np.inf and out_of_sample[0, 0] == -np.inf
    assert np.isfinite(in_sample[0, 1:]).all()


def test_pbo_sixteen_blocks_has_12870_splits_and_is_fast():
    # The protocol's setting: 16 blocks, 9 configurations, about 17 years of days.
    noise = np.random.default_rng(16).normal(0.0003, 0.01, size=(4280, 9))
    started = time.perf_counter()
    result = robustness.pbo_cscv(pd.DataFrame(noise))        # n_blocks defaults to 16
    elapsed = time.perf_counter() - started
    assert result["n_splits"] == 12870 == math.comb(16, 8)
    assert result["rows_used"] == 4272                       # 4280 // 16 = 267, 267 * 16
    assert result["rows_dropped"] == 8
    assert result["logits"].shape == (12870,)
    assert 0.0 <= result["pbo"] <= 1.0
    assert elapsed < 20.0


def test_pbo_rejects_bad_input():
    good = pd.DataFrame(np.random.default_rng(1).normal(size=(40, 3)))
    with pytest.raises(ValueError):
        robustness.pbo_cscv(good[[0]], n_blocks=4)           # one column
    with pytest.raises(ValueError):
        robustness.pbo_cscv(good.iloc[:7], n_blocks=4)       # 7 // 4 = 1 row per block
    with pytest.raises(ValueError):
        robustness.pbo_cscv(good, n_blocks=5)                # odd number of blocks
    with pytest.raises(ValueError):
        robustness.pbo_cscv(good.to_numpy(), n_blocks=4)     # not a DataFrame
    holed = good.copy()
    holed.iloc[10, 1] = np.nan
    with pytest.raises(ValueError):
        robustness.pbo_cscv(holed, n_blocks=4)               # nan in the matrix
    # Exactly 2 rows per block is the smallest size allowed.
    assert robustness.pbo_cscv(good.iloc[:8], n_blocks=4)["rows_used"] == 8


# ---------------------------------------------------------------------------
# 6. OLS with Newey-West errors
# ---------------------------------------------------------------------------

def _factors(n_rows, seed):
    """Two made-up factor series on business days."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n_rows)
    return pd.DataFrame({"x1": rng.normal(0.0, 0.01, n_rows),
                         "x2": rng.normal(0.0, 0.005, n_rows)}, index=index)


def test_ols_recovers_an_exact_linear_relation():
    # y = 0.001 + 2 * x1 - 0.5 * x2 with no noise: the fit is perfect.
    X = _factors(300, seed=1)
    y = 0.001 + 2.0 * X["x1"] - 0.5 * X["x2"]
    result = robustness.ols_newey_west(y, X)
    assert result["alpha"] == pytest.approx(0.001, abs=1e-12)
    assert result["alpha_annualized"] == pytest.approx(0.001 * 252, abs=1e-9)     # 0.252
    assert result["betas"]["x1"] == pytest.approx(2.0, abs=1e-10)
    assert result["betas"]["x2"] == pytest.approx(-0.5, abs=1e-10)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert result["n_obs"] == 300
    assert result["lags"] == 21                             # the protocol's default
    assert set(result["beta_t"]) == {"x1", "x2"}


def test_ols_with_noise_is_close_and_t_statistics_are_finite():
    X = _factors(2000, seed=2)
    noise = np.random.default_rng(3).normal(0.0, 0.002, 2000)
    y = 0.001 + 2.0 * X["x1"] - 0.5 * X["x2"] + noise
    result = robustness.ols_newey_west(y, X)
    # Standard error of alpha is about 0.002 / sqrt(2000) = 0.000045, so alpha is within
    # 0.0003 of 0.001 with a lot of room, and its t-statistic is around 0.001 / 0.000045 = 22.
    assert result["alpha"] == pytest.approx(0.001, abs=3e-4)
    assert result["betas"]["x1"] == pytest.approx(2.0, abs=0.05)
    assert result["betas"]["x2"] == pytest.approx(-0.5, abs=0.05)
    assert 0.9 < result["r_squared"] < 1.0
    assert math.isfinite(result["alpha_t"]) and result["alpha_t"] > 5.0
    assert math.isfinite(result["alpha_t_ols"]) and result["alpha_t_ols"] > 5.0
    assert all(math.isfinite(value) for value in result["beta_t"].values())
    assert result["beta_t"]["x1"] > 0.0 > result["beta_t"]["x2"]


def test_ols_series_without_alpha_has_a_small_alpha_t():
    # y = 1.5 * x1 + noise: the true intercept is 0, so its t-statistic should be noise
    # (|t| below 2 in this seeded example), while the beta is clearly there.
    X = _factors(1500, seed=4)[["x1"]]
    y = 1.5 * X["x1"] + np.random.default_rng(5).normal(0.0, 0.005, 1500)
    result = robustness.ols_newey_west(y, X)
    assert abs(result["alpha_t"]) < 2.0
    assert abs(result["alpha_t_ols"]) < 2.0
    assert result["beta_t"]["x1"] > 10.0
    assert list(result["betas"]) == ["x1"]


def test_ols_alignment_keeps_only_complete_overlapping_rows():
    X = _factors(100, seed=6)
    full_y = 0.002 + 1.0 * X["x1"] + 3.0 * X["x2"]
    # y covers rows 10..99 (90 rows); X covers rows 0..79. Overlap: rows 10..79 = 70 rows.
    y = full_y.iloc[10:].copy()
    X_short = X.iloc[:80].copy()
    # One nan in y and one nan in X inside the overlap: 70 - 2 = 68 rows are left.
    y.iloc[5] = np.nan
    X_short.iloc[30, 1] = np.nan
    result = robustness.ols_newey_west(y, X_short, lags=5)
    assert result["n_obs"] == 68
    assert result["lags"] == 5
    # The relation is exact, so the coefficients are still recovered.
    assert result["alpha"] == pytest.approx(0.002, abs=1e-12)
    assert result["betas"]["x1"] == pytest.approx(1.0, abs=1e-10)
    assert result["betas"]["x2"] == pytest.approx(3.0, abs=1e-10)
    # The order of the rows in the inputs does not matter: they are sorted by date.
    shuffled = robustness.ols_newey_west(y.sample(frac=1.0, random_state=1), X_short, lags=5)
    assert shuffled["alpha_t"] == pytest.approx(result["alpha_t"], rel=1e-9)


def test_ols_with_zero_lags_equals_white_standard_errors():
    # Heteroskedastic noise: its size grows with |x1|.
    X = _factors(400, seed=7)
    noise = np.random.default_rng(8).normal(0.0, 1.0, 400) * (0.001 + 0.5 * X["x1"].abs().to_numpy())
    y = 0.0005 + 1.2 * X["x1"] + 0.3 * X["x2"] + noise
    result = robustness.ols_newey_west(y, X, lags=0)

    # White (1980) errors computed directly, one row at a time:
    #   cov = n / (n - k) * (X'X)^-1 [ sum_t e_t^2 x_t x_t' ] (X'X)^-1
    design = np.column_stack([np.ones(400), X.to_numpy()])
    n, k = design.shape
    inverse = np.linalg.inv(design.T @ design)
    coefficients = inverse @ design.T @ y.to_numpy()
    residuals = y.to_numpy() - design @ coefficients
    meat = np.zeros((k, k))
    for t in range(n):
        meat += residuals[t] ** 2 * np.outer(design[t], design[t])
    white_errors = np.sqrt(np.diag(n / (n - k) * inverse @ meat @ inverse))
    white_t = coefficients / white_errors

    assert result["alpha_t"] == pytest.approx(white_t[0], rel=1e-9)
    assert result["beta_t"]["x1"] == pytest.approx(white_t[1], rel=1e-9)
    assert result["beta_t"]["x2"] == pytest.approx(white_t[2], rel=1e-9)

    # The plain OLS t-statistic, also directly: cov = s^2 (X'X)^-1, s^2 = SSR / (n - k).
    s_squared = residuals @ residuals / (n - k)
    ols_t = coefficients[0] / math.sqrt(s_squared * inverse[0, 0])
    assert result["alpha_t_ols"] == pytest.approx(ols_t, rel=1e-9)


def test_ols_newey_west_lag_terms_match_a_direct_loop():
    # Autocorrelated noise (each value carries 0.6 of the one before), 3 lags.
    X = _factors(250, seed=9)[["x1"]]
    shocks = np.random.default_rng(10).normal(0.0, 0.003, 250)
    noise = np.zeros(250)
    for t in range(1, 250):
        noise[t] = 0.6 * noise[t - 1] + shocks[t]
    y = 0.0002 + 0.8 * X["x1"] + noise
    lags = 3
    result = robustness.ols_newey_west(y, X, lags=lags)

    # S = sum_t e_t^2 x_t x_t' + sum_{l=1..3} w_l sum_t e_t e_{t-l} (x_t x_{t-l}' + x_{t-l} x_t')
    # with w_l = 1 - l / 4: w_1 = 0.75, w_2 = 0.5, w_3 = 0.25.
    design = np.column_stack([np.ones(250), X.to_numpy()])
    n, k = design.shape
    inverse = np.linalg.inv(design.T @ design)
    coefficients = inverse @ design.T @ y.to_numpy()
    e = y.to_numpy() - design @ coefficients
    S = np.zeros((k, k))
    for t in range(n):
        S += e[t] ** 2 * np.outer(design[t], design[t])
    for lag, weight in ((1, 0.75), (2, 0.5), (3, 0.25)):
        for t in range(lag, n):
            S += weight * e[t] * e[t - lag] * (np.outer(design[t], design[t - lag])
                                               + np.outer(design[t - lag], design[t]))
    errors = np.sqrt(np.diag(n / (n - k) * inverse @ S @ inverse))
    assert result["alpha_t"] == pytest.approx(coefficients[0] / errors[0], rel=1e-9)
    assert result["beta_t"]["x1"] == pytest.approx(coefficients[1] / errors[1], rel=1e-9)
    # Positive autocorrelation makes the honest error of alpha larger than the plain one.
    assert abs(result["alpha_t"]) < abs(result["alpha_t_ols"])


def test_ols_rejects_too_few_rows():
    X = _factors(3, seed=1)
    with pytest.raises(ValueError):
        robustness.ols_newey_west(X["x1"] * 2.0, X)          # 3 rows, 3 coefficients


# ---------------------------------------------------------------------------
# 7. Walk-forward selection
# ---------------------------------------------------------------------------

def _walk_forward_matrix():
    """Daily returns for 2015-2019 where A is the better column until 2016, then B.

    Both columns wiggle +/-0.01 around a yearly mean:
        year   2015    2016    2017    2018    2019
        A      0.002   0.002  -0.001  -0.001  -0.001
        B      0.000   0.000   0.006   0.006   0.006
    """
    index = pd.bdate_range("2015-01-01", "2019-12-31")
    wiggle = np.where(np.arange(len(index)) % 2 == 0, 0.01, -0.01)
    early = np.asarray(index.year) <= 2016
    return pd.DataFrame({"A": wiggle + np.where(early, 0.002, -0.001),
                         "B": wiggle + np.where(early, 0.000, 0.006)}, index=index)


def test_walk_forward_selections_year_by_year():
    frame = _walk_forward_matrix()
    result = robustness.walk_forward_selection(frame, baseline="A", first_test_year=2016)
    assert list(result.columns) == ["year", "selected", "selected_return", "baseline_return",
                                    "selected_is_baseline", "n_days"]
    assert list(result["year"]) == [2016, 2017, 2018, 2019]
    # By hand (the spread is about 0.01 for both columns, so the mean decides):
    #   2016: history 2015.       mean A 0.002, B 0.000                        -> A
    #   2017: history 2015-2016.  mean A 0.002, B 0.000                        -> A
    #   2018: history 2015-2017.  mean A (0.002 + 0.002 - 0.001) / 3 = 0.001,
    #                             mean B (0 + 0 + 0.006) / 3 = 0.002           -> B
    #   2019: history 2015-2018.  mean A 0.0005, B 0.003                       -> B
    # The switch to B shows up one year after B became better: selection only sees the past.
    assert list(result["selected"]) == ["A", "A", "B", "B"]
    assert list(result["selected_is_baseline"]) == [True, True, False, False]

    # Returns are compounded over the test year: (1 + r1)(1 + r2)... - 1.
    for row in result.itertuples():
        in_year = frame[frame.index.year == row.year]
        assert row.n_days == len(in_year)
        assert row.selected_return == pytest.approx((1.0 + in_year[row.selected]).prod() - 1.0)
        assert row.baseline_return == pytest.approx((1.0 + in_year["A"]).prod() - 1.0)
    # When the selection is the baseline the two returns are the same number.
    assert result.loc[0, "selected_return"] == result.loc[0, "baseline_return"]
    # In 2017 the walk-forward pick was A (mean -0.001 a day): it lost money that year.
    assert result.loc[1, "selected_return"] < 0.0
    # In 2018 it picked B (mean +0.006 a day) while the baseline A kept losing.
    assert result.loc[2, "selected_return"] > 0.0 > result.loc[2, "baseline_return"]


def test_walk_forward_compounds_a_tiny_hand_case():
    # 2020: three days, used only as history (min_history_days=3).
    # 2021: X returns +10% then -10%: (1.1)(0.9) - 1 = -0.01.
    #       Y returns +1% then +1%:   (1.01)(1.01) - 1 = 0.0201.
    # History Sharpe: X mean 0.02 / std 0.01 = 2; Y mean 0.0 -> X is selected.
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2021-01-04", "2021-01-05"])
    frame = pd.DataFrame({"X": [0.01, 0.02, 0.03, 0.10, -0.10],
                          "Y": [-0.01, 0.00, 0.01, 0.01, 0.01]}, index=index)
    result = robustness.walk_forward_selection(frame, baseline="Y", first_test_year=2021,
                                               min_history_days=3)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["year"] == 2021
    assert row["selected"] == "X"
    assert row["selected_return"] == pytest.approx(-0.01)
    assert row["baseline_return"] == pytest.approx(0.0201)
    assert row["selected_is_baseline"] is False or row["selected_is_baseline"] == False  # noqa: E712
    assert row["n_days"] == 2


def test_walk_forward_has_no_lookahead():
    frame = _walk_forward_matrix()
    original = robustness.walk_forward_selection(frame, baseline="A", first_test_year=2016)

    # Rewrite everything from 2018 onward so that A becomes hugely better than B.
    changed = frame.copy()
    late = changed.index.year >= 2018
    changed.loc[late, "A"] = changed.loc[late, "A"] + 0.05
    changed.loc[late, "B"] = changed.loc[late, "B"] - 0.05
    after = robustness.walk_forward_selection(changed, baseline="A", first_test_year=2016)

    # The selections for 2016, 2017 AND 2018 use only data before 2018: they cannot move.
    assert list(after["selected"])[:3] == list(original["selected"])[:3] == ["A", "A", "B"]
    # The rows for 2016 and 2017 are identical in every field (their test years are untouched).
    pd.testing.assert_frame_equal(after.iloc[:2], original.iloc[:2])
    # 2018 still selects B even though B is now terrible in 2018: that is the proof that
    # the 2018 rows were not seen when choosing. Its recorded return did change.
    assert after.loc[2, "selected_return"] < 0.0 < original.loc[2, "selected_return"]
    # 2019 may use the new 2018 data, and does: it now selects A.
    assert after.loc[3, "selected"] == "A"

    # Deleting the future entirely leaves the earlier rows unchanged as well.
    truncated = robustness.walk_forward_selection(frame[frame.index.year <= 2017],
                                                  baseline="A", first_test_year=2016)
    pd.testing.assert_frame_equal(truncated, original.iloc[:2])


def test_walk_forward_skips_years_without_enough_history():
    frame = _walk_forward_matrix()
    # 2015 has no earlier rows at all: it is skipped, 2016 onward is kept.
    from_start = robustness.walk_forward_selection(frame, baseline="A", first_test_year=2015)
    assert list(from_start["year"]) == [2016, 2017, 2018, 2019]
    # 2015 has 261 business days and 2015-2016 has 522. Asking for 300 days of history
    # skips 2016 (261 < 300) and keeps 2017 (522 >= 300).
    assert int((frame.index.year == 2015).sum()) == 261
    stricter = robustness.walk_forward_selection(frame, baseline="A", first_test_year=2016,
                                                 min_history_days=300)
    assert list(stricter["year"]) == [2017, 2018, 2019]
    # Asking for more history than exists gives an empty table with the right columns.
    empty = robustness.walk_forward_selection(frame, baseline="A", first_test_year=2016,
                                              min_history_days=10_000)
    assert len(empty) == 0
    assert "selected_return" in empty.columns


def test_walk_forward_rejects_an_unknown_baseline():
    with pytest.raises(ValueError):
        robustness.walk_forward_selection(_walk_forward_matrix(), baseline="Z", first_test_year=2016)
