"""Correctness-gated benchmark for mojo-adtk.

Every case checks the Mojo result against the real `adtk` transformer before
timing, so a broken kernel shows up as a correctness failure rather than as a
suspiciously good number. The baseline is `adtk`'s own `RollingAggregate`,
which is `pandas.Series.rolling` (Cython) plus a thin wrapper: that is the
fastest reasonable formulation of the same reduction, not a Python loop.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_adtk  # noqa: E402
from adtk.transformer import DoubleRollingAggregate as RefDouble  # noqa: E402
from adtk.transformer import RollingAggregate as RefRolling  # noqa: E402


def _series(n, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n) * 10.0 + 50.0
    return pd.Series(
        values, index=pd.date_range("2021-01-01", periods=n, freq="s")
    )

def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_rolling(agg, n=1 << 21, w=32):
    values = _series(n)
    tol = dict(rtol=1e-6, atol=1e-9) if agg in ("skew", "kurt") else dict(
        rtol=1e-11, atol=1e-11
    )
    got = mojo_adtk.RollingAggregate(w, agg=agg).transform(values)
    exp = RefRolling(w, agg=agg).transform(values)
    np.testing.assert_array_equal(np.isnan(got.to_numpy()), np.isnan(exp.to_numpy()))
    np.testing.assert_allclose(got.to_numpy(), exp.to_numpy(), **tol)

    mine = mojo_adtk.RollingAggregate(w, agg=agg)
    theirs = RefRolling(w, agg=agg)
    numpy_time = _time(lambda: theirs.transform(values))
    mojo_time = _time(lambda: mine.transform(values))
    return f"rolling {agg} n={n} w={w}", numpy_time, mojo_time


def bench_rolling_quantile(q, n=1 << 21, w=32):
    values = _series(n, seed=1)
    params = {"q": q}
    got = mojo_adtk.RollingAggregate(
        w, agg="quantile", agg_params=params
    ).transform(values)
    exp = RefRolling(w, agg="quantile", agg_params=params).transform(values)
    np.testing.assert_array_equal(np.isnan(got.to_numpy()), np.isnan(exp.to_numpy()))
    np.testing.assert_allclose(got.to_numpy(), exp.to_numpy(), rtol=1e-11, atol=1e-11)

    mine = mojo_adtk.RollingAggregate(w, agg="quantile", agg_params=params)
    theirs = RefRolling(w, agg="quantile", agg_params=params)
    numpy_time = _time(lambda: theirs.transform(values))
    mojo_time = _time(lambda: mine.transform(values))
    return f"rolling q={q} n={n} w={w}", numpy_time, mojo_time


def bench_rolling_iqr(n=1 << 21, w=32):
    values = _series(n, seed=2)
    got = mojo_adtk.RollingAggregate(w, agg="iqr").transform(values)
    exp = RefRolling(w, agg="iqr").transform(values)
    np.testing.assert_array_equal(np.isnan(got.to_numpy()), np.isnan(exp.to_numpy()))
    np.testing.assert_allclose(got.to_numpy(), exp.to_numpy(), rtol=1e-11, atol=1e-11)

    mine = mojo_adtk.RollingAggregate(w, agg="iqr")
    theirs = RefRolling(w, agg="iqr")
    numpy_time = _time(lambda: theirs.transform(values))
    mojo_time = _time(lambda: mine.transform(values))
    return f"rolling iqr n={n} w={w}", numpy_time, mojo_time


def bench_double_rolling(n=1 << 21, w=32):
    values = _series(n, seed=3)
    mine = mojo_adtk.DoubleRollingAggregate(
        w, agg="mean", center=True, diff="rel_diff"
    )
    theirs = RefDouble(w, agg="mean", center=True, diff="rel_diff")
    np.testing.assert_allclose(
        mine.transform(values).to_numpy(),
        theirs.transform(values).to_numpy(),
        equal_nan=True, rtol=1e-11, atol=1e-11,
    )
    numpy_time = _time(lambda: theirs.transform(values))
    mojo_time = _time(lambda: mine.transform(values))
    return f"double rel_diff n={n} w={w}", numpy_time, mojo_time


def main():
    print(f"{'case':<30}{'adtk (pandas)':>16}{'mojo-adtk':>14}{'ratio':>9}")
    print("-" * 69)
    cases = [
        lambda: bench_rolling("mean"),
        lambda: bench_rolling("std"),
        lambda: bench_rolling("skew"),
        lambda: bench_rolling("count"),
        lambda: bench_rolling_quantile(0.5),
        lambda: bench_rolling_iqr(),
        lambda: bench_double_rolling(),
    ]
    for case in cases:
        label, ref, got = case()
        ratio = ref / got if got else float("nan")
        print(f"{label:<30}{ref*1e3:>14.2f}ms{got*1e3:>12.2f}ms{ratio:>8.2f}x")


if __name__ == "__main__":
    main()
