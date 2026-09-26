"""Parity tests for the Mojo rolling-window kernels against the real adtk.

Each test compares `mojo_adtk.RollingAggregate` / `DoubleRollingAggregate`
against `adtk.transformer.RollingAggregate` / `DoubleRollingAggregate` on data
chosen to expose a specific plausible bug: a wrong window edge, a dropped
min_periods gate, a NaN that is not skipped, a quickselect that corrupts the
shared scratch buffer, and so on.
"""

import numpy as np
import pandas as pd
import pytest

import mojo_adtk
from adtk.transformer import DoubleRollingAggregate as RefDouble
from adtk.transformer import RollingAggregate as RefRolling

# FMA/two-pass vs pandas' incremental updates: these are reductions, so the
# agreement is close to machine epsilon rather than bit-exact.
TOL = dict(rtol=1e-11, atol=1e-11)
# skew/kurt amplify cancellation, so they need looser bounds
MOMENT_TOL = dict(rtol=1e-7, atol=1e-9)


def series(values, name="x", start="2021-03-01"):
    # adtk refuses a Series whose index is not a DatetimeIndex
    values = np.asarray(values, dtype=np.float64)
    return pd.Series(
        values, index=pd.date_range(start, periods=values.size, freq="h"),
        name=name,
    )


def compare(mine, theirs, **tol):
    theirs = np.asarray(theirs, dtype=np.float64)
    assert np.array_equal(np.isnan(mine), np.isnan(theirs)), "NaN mask differs"
    np.testing.assert_allclose(mine, theirs, equal_nan=True, **tol)


@pytest.mark.parametrize("agg", ["mean", "sum", "min", "max", "count", "nnz"])
def test_scalar_aggregates_match_adtk(agg):
    rng = np.random.default_rng(7)
    s = series(rng.standard_normal(300) * 10.0 + 5.0)
    got = mojo_adtk.RollingAggregate(17, agg=agg).transform(s)
    exp = RefRolling(17, agg=agg).transform(s)
    compare(got.to_numpy(), exp.to_numpy(), **TOL)


@pytest.mark.parametrize("agg", ["mean", "min", "max", "std", "var"])
def test_scalar_aggregates_match_adtk_small_windows(agg):
    rng = np.random.default_rng(8)
    s = series(rng.standard_normal(200))
    for w in (1, 2, 3, 8):
        got = mojo_adtk.RollingAggregate(w, agg=agg).transform(s)
        exp = RefRolling(w, agg=agg).transform(s)
        compare(got.to_numpy(), exp.to_numpy(), **TOL)


def test_std_var_match_adtk_ill_conditioned_window():
    # variance computed from raw power sums instead of a second pass would
    # lose several digits here; pandas is the reference.
    rng = np.random.default_rng(9)
    s = series(1e8 + rng.standard_normal(120) * 1.0)
    for agg in ("var", "std"):
        got = mojo_adtk.RollingAggregate(11, agg=agg).transform(s)
        exp = RefRolling(11, agg=agg).transform(s)
        compare(got.to_numpy(), exp.to_numpy(), rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("agg", ["skew", "kurt"])
def test_moment_aggregates_match_adtk(agg):
    rng = np.random.default_rng(10)
    s = series(rng.standard_normal(250) * 2.0 + 1.0)
    got = mojo_adtk.RollingAggregate(9, agg=agg).transform(s)
    exp = RefRolling(9, agg=agg).transform(s)
    compare(got.to_numpy(), exp.to_numpy(), **MOMENT_TOL)


def test_skew_kurt_observation_minimums():
    # skew needs 3 observations and kurt needs 4, independently of
    # min_periods: a kernel that only gated on min_periods would return a
    # number where pandas returns NaN.
    s = series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    skew = mojo_adtk.RollingAggregate(4, agg="skew", min_periods=1).transform(s)
    kurt = mojo_adtk.RollingAggregate(4, agg="kurt", min_periods=1).transform(s)
    # windows are [0], [0,1], [0,1,2] for the first three positions
    assert np.isnan(skew.to_numpy()[:2]).all()
    assert not np.isnan(skew.to_numpy()[2])
    assert np.isnan(kurt.to_numpy()[:3]).all()
    compare(skew.to_numpy(),
            RefRolling(4, agg="skew", min_periods=1).transform(s).to_numpy(),
            **MOMENT_TOL)
    compare(kurt.to_numpy(),
            RefRolling(4, agg="kurt", min_periods=1).transform(s).to_numpy(),
            **MOMENT_TOL)


def test_median_and_quantiles_match_adtk():
    rng = np.random.default_rng(11)
    s = series(rng.standard_normal(400) * 30.0)
    got = mojo_adtk.RollingAggregate(23, agg="median").transform(s)
    exp = RefRolling(23, agg="median").transform(s)
    compare(got.to_numpy(), exp.to_numpy(), **TOL)
    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        got = mojo_adtk.RollingAggregate(
            23, agg="quantile", agg_params={"q": q}
        ).transform(s)
        exp = RefRolling(23, agg="quantile", agg_params={"q": q}).transform(s)
        compare(got.to_numpy(), exp.to_numpy(), **TOL)


def test_quantile_list_returns_dataframe_with_adtk_names():
    rng = np.random.default_rng(12)
    s = series(rng.standard_normal(200))
    qs = [0.1, 0.5, 0.9]
    got = mojo_adtk.RollingAggregate(
        15, agg="quantile", agg_params={"q": qs}
    ).transform(s)
    exp = RefRolling(15, agg="quantile", agg_params={"q": qs}).transform(s)
    assert list(got.columns) == list(exp.columns)
    assert (got.to_numpy().shape) == (200, 3)
    compare(got.to_numpy(), exp.to_numpy(), **TOL)


@pytest.mark.parametrize("agg", ["iqr", "idr"])
def test_interquantile_ranges_match_adtk(agg):
    rng = np.random.default_rng(13)
    s = series(rng.standard_normal(250) * 4.0)
    got = mojo_adtk.RollingAggregate(31, agg=agg).transform(s)
    exp = RefRolling(31, agg=agg).transform(s)
    compare(got.to_numpy(), exp.to_numpy(), **TOL)


def test_nans_are_skipped_and_gate_on_min_periods():
    rng = np.random.default_rng(14)
    values = rng.standard_normal(180) * 6.0
    values[[3, 4, 40, 41, 42, 100, 179]] = np.nan
    s = series(values)
    for mp in (None, 1, 5, 9):
        for agg in ("mean", "sum", "median", "max", "std", "count", "nnz",
                    "quantile", "iqr", "skew"):
            params = {"q": 0.4} if agg == "quantile" else None
            got = mojo_adtk.RollingAggregate(
                9, agg=agg, agg_params=params, min_periods=mp
            ).transform(s)
            exp = RefRolling(9, agg=agg, agg_params=params, min_periods=mp)
            exp = exp.transform(s)
            tol = MOMENT_TOL if agg == "skew" else TOL
            compare(got.to_numpy(), exp.to_numpy(), **tol)


def test_all_nan_window_yields_nan():
    s = series([1.0, np.nan, np.nan, np.nan, np.nan, 2.0])
    for agg in ("mean", "median", "min", "max", "sum", "std", "count", "nnz",
                "quantile", "iqr", "skew", "kurt"):
        params = {"q": 0.5} if agg == "quantile" else None
        got = mojo_adtk.RollingAggregate(
            3, agg=agg, agg_params=params, min_periods=1
        ).transform(s)
        exp = RefRolling(3, agg=agg, agg_params=params, min_periods=1)
        compare(got.to_numpy(), exp.transform(s).to_numpy(), **TOL)


@pytest.mark.parametrize("center", [False, True])
def test_center_alignment_matches_adtk(center):
    rng = np.random.default_rng(15)
    s = series(rng.standard_normal(200) * 2.0)
    for w in (4, 5, 16):
        for agg in ("mean", "median", "std", "max", "count"):
            got = mojo_adtk.RollingAggregate(
                w, agg=agg, center=center
            ).transform(s)
            exp = RefRolling(w, agg=agg, center=center).transform(s)
            compare(got.to_numpy(), exp.to_numpy(), **TOL)


def test_window_alignment_is_exact_on_hand_checked_data():
    # a one-off window edge (w instead of w - 1, or a centred window anchored
    # on the right edge) shows up immediately on this series
    s = series([1.0, 2.0, 3.0, 4.0, 5.0])
    got = mojo_adtk.RollingAggregate(3, agg="sum").transform(s)
    assert np.isnan(got.to_numpy()[:2]).all()
    np.testing.assert_allclose(got.to_numpy()[2:], [6.0, 9.0, 12.0], **TOL)
    got = mojo_adtk.RollingAggregate(3, agg="mean", center=True).transform(s)
    np.testing.assert_allclose(got.to_numpy(), [np.nan, 2.0, 3.0, 4.0, np.nan],
                               equal_nan=True, **TOL)


def test_window_larger_than_series():
    s = series([1.0, 2.0, 3.0])
    got = mojo_adtk.RollingAggregate(10, agg="mean").transform(s)
    exp = RefRolling(10, agg="mean").transform(s)
    compare(got.to_numpy(), exp.to_numpy(), **TOL)
    got = mojo_adtk.RollingAggregate(10, agg="mean", min_periods=2).transform(s)
    exp = RefRolling(10, agg="mean", min_periods=2).transform(s)
    compare(got.to_numpy(), exp.to_numpy(), **TOL)


def test_repeated_calls_are_identical():
    # the quantile kernel reuses one scratch buffer across positions and
    rng = np.random.default_rng(16)
    s = series(rng.standard_normal(150) * 12.0)
    ra = mojo_adtk.RollingAggregate(19, agg="quantile", agg_params={"q": 0.3})
    first = ra.transform(s)
    for _ in range(3):
        np.testing.assert_array_equal(ra.transform(s).to_numpy(), first.to_numpy())
    # interleaving a different aggregate must not disturb the quantile state
    mojo_adtk.RollingAggregate(19, agg="iqr").transform(s)
    np.testing.assert_array_equal(ra.transform(s).to_numpy(), first.to_numpy())


def test_quantile_matches_bruteforce_order_statistic():
    rng = np.random.default_rng(17)
    values = rng.standard_normal(60) * 8.0
    w = 7
    got = mojo_adtk.RollingAggregate(
        w, agg="quantile", agg_params={"q": 0.3}
    ).transform(series(values))
    for i in range(w - 1, values.size):
        expect = np.quantile(values[i - w + 1:i + 1], 0.3)
        assert got.iloc[i] == pytest.approx(expect, rel=1e-12, abs=1e-12)


def test_index_and_name_are_preserved():
    idx = pd.date_range("2020-01-01", periods=50, freq="h")
    s = pd.Series(np.arange(50.0), index=idx, name="signal")
    got = mojo_adtk.RollingAggregate(5, agg="mean").transform(s)
    assert got.name == "signal"
    assert got.index.equals(idx)


def test_input_series_is_not_mutated():
    rng = np.random.default_rng(18)
    values = rng.standard_normal(80)
    s = series(values)
    before = s.to_numpy().copy()
    mojo_adtk.RollingAggregate(7, agg="median").transform(s)
    np.testing.assert_array_equal(s.to_numpy(), before)


def test_non_monotonic_index_is_rejected():
    idx = pd.date_range("2021-01-01", periods=4, freq="h")
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx[[1, 0, 2, 3]])
    with pytest.raises(ValueError):
        mojo_adtk.RollingAggregate(2, agg="mean").transform(s)


def test_zero_min_periods_is_rejected():
    s = series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        mojo_adtk.RollingAggregate(2, agg="sum", min_periods=0).transform(s)


@pytest.mark.parametrize("agg", ["nunique", "hist"])
def test_unported_aggregations_are_rejected_loudly(agg):
    with pytest.raises(ValueError):
        mojo_adtk.RollingAggregate(4, agg=agg)
    with pytest.raises(ValueError):
        mojo_adtk.RollingAggregate(4, agg=lambda w: 0.0)


def test_empty_series():
    s = series([])
    got = mojo_adtk.RollingAggregate(3, agg="mean").transform(s)
    assert got.to_numpy().size == 0
    exp = RefRolling(3, agg="mean").transform(s)
    assert exp.to_numpy().size == 0
