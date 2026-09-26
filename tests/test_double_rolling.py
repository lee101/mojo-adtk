"""Parity tests for the Mojo `DoubleRollingAggregate` against the real adtk.

`DoubleRollingAggregate` reduces two adjacent windows and differences them, so
these tests check the window placement (the left window is the series shifted
by the right window size; with `center=True` the right window is computed on
the reversed series), the difference convention, and NaN propagation.
"""

import numpy as np
import pandas as pd
import pytest

import mojo_adtk
from adtk.transformer import DoubleRollingAggregate as RefDouble

TOL = dict(rtol=1e-11, atol=1e-11)


def series(values, start="2021-05-01"):
    values = np.asarray(values, dtype=np.float64)
    return pd.Series(
        values, index=pd.date_range(start, periods=values.size, freq="h"),
        name="signal",
    )


def compare(mine, theirs):
    theirs = np.asarray(theirs, dtype=np.float64)
    assert np.array_equal(np.isnan(mine), np.isnan(theirs)), "NaN mask differs"
    np.testing.assert_allclose(mine, theirs, equal_nan=True, **TOL)


@pytest.mark.parametrize("diff", ["l1", "diff", "rel_diff", "abs_rel_diff"])
@pytest.mark.parametrize("center", [True, False])
def test_diff_conventions_match_adtk(diff, center):
    rng = np.random.default_rng(21)
    s = series(rng.standard_normal(240) * 3.0 + 10.0)
    got = mojo_adtk.DoubleRollingAggregate(
        12, agg="mean", center=center, diff=diff
    ).transform(s)
    exp = RefDouble(12, agg="mean", center=center, diff=diff).transform(s)
    compare(got.to_numpy(), exp.to_numpy())


@pytest.mark.parametrize("agg", ["mean", "median", "max", "min", "std", "count"])
@pytest.mark.parametrize("window", [5, (7, 13), (13, 7)])
def test_window_pairs_and_aggs_match_adtk(agg, window):
    rng = np.random.default_rng(22)
    s = series(rng.standard_normal(200) * 2.0 + 4.0)
    got = mojo_adtk.DoubleRollingAggregate(
        window, agg=agg, center=False, diff="l1"
    ).transform(s)
    exp = RefDouble(window, agg=agg, center=False, diff="l1").transform(s)
    compare(got.to_numpy(), exp.to_numpy())


def test_unequal_aggs_for_the_two_windows():
    rng = np.random.default_rng(23)
    s = series(rng.standard_normal(150) + 2.0)
    got = mojo_adtk.DoubleRollingAggregate(
        (6, 11), agg=("mean", "median"), center=False, diff="l1"
    ).transform(s)
    exp = RefDouble(
        (6, 11), agg=("mean", "median"), center=False, diff="l1"
    ).transform(s)
    compare(got.to_numpy(), exp.to_numpy())


def test_nan_values_propagate():
    rng = np.random.default_rng(24)
    values = rng.standard_normal(120) * 6.0
    values[[10, 11, 12, 60]] = np.nan
    s = series(values)
    got = mojo_adtk.DoubleRollingAggregate(
        9, agg="mean", center=True, diff="rel_diff"
    ).transform(s)
    exp = RefDouble(9, agg="mean", center=True, diff="rel_diff").transform(s)
    compare(got.to_numpy(), exp.to_numpy())


def test_hand_checked_shifted_windows():
    # window=(2, 3), center=False: the left window is the series shifted by
    # 3, so at t=4 the left window is [v1, v2] and the right is [v2, v3, v4].
    s = series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    got = mojo_adtk.DoubleRollingAggregate(
        (2, 3), agg="mean", center=False, diff="diff"
    ).transform(s)
    assert got.iloc[4] == pytest.approx(4.0 - 1.5)
    got = mojo_adtk.DoubleRollingAggregate(
        (2, 3), agg="mean", center=False, diff="l1"
    ).transform(s)
    assert got.iloc[4] == pytest.approx(2.5)


def test_index_and_name_preserved():
    idx = pd.date_range("2022-02-02", periods=40, freq="min")
    s = pd.Series(np.linspace(1.0, 5.0, 40), index=idx, name="v")
    got = mojo_adtk.DoubleRollingAggregate(5, agg="mean").transform(s)
    assert got.name == "v"
    assert got.index.equals(idx)


def test_unsupported_shapes_are_rejected():
    s = series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        mojo_adtk.DoubleRollingAggregate(3, diff=lambda a, b: 0.0).transform(s)
    with pytest.raises(ValueError):
        mojo_adtk.DoubleRollingAggregate(3, diff="l2").transform(s)
    with pytest.raises(ValueError):
        mojo_adtk.DoubleRollingAggregate("3s", agg="mean").transform(s)
    with pytest.raises(ValueError):
        mojo_adtk.DoubleRollingAggregate(
            3, agg="quantile", agg_params={"q": [0.1, 0.9]}
        ).transform(s)
