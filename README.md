# mojo-adtk

Mojo port of the compute core of [NVIDIA adtk](https://github.com/NVIDIA/adtk)
(NVIDIA's Anomaly Detection Toolkit, version 0.6.2). The Python package is
named `mojo_adtk`, so it installs alongside the real `adtk` and the parity
tests import both and compare them directly.

## What the compute core is

adtk is 8,000 lines of pandas time-series machinery: `_Model` plumbing, pipes,
detectors assembled out of transformers, `trtexec` launching, TensorRT config
files, visualisation helpers. There is exactly one place in it with a numeric
inner loop: `adtk.transformer.RollingAggregate`, which slides a window along a
series and reduces each window to a scalar with mean, median, sum, min, max,
std, var, skew, kurt, count, nnz, quantile, iqr or idr.
`DoubleRollingAggregate` is the same reduction applied to two adjacent windows
and then differenced, so these kernels are also the whole of its cost.

Everything else in adtk is either a composition of those reductions
(`PersistAD`, `LevelShiftAD`, `AutoregressionAD`, `SeasonalAD`, ...) or pure
control flow. Porting the reduction is therefore porting the numeric work, and
the detectors built on it can be reproduced on top of it.

## Covered subset

| area | implemented API |
| --- | --- |
| `RollingAggregate` | `mean`, `median`, `sum`, `min`, `max`, `std`, `var`, `skew`, `kurt`, `count`, `nnz`, `quantile` (scalar or list), `iqr`, `idr` |
| window alignment | integer windows with `center=False` (right edge) and `center=True` (left edge at `i - w // 2`), both clipped like `pandas.Series.rolling` |
| observation gate | `min_periods` (default: the window size), pandas NaN-skipping semantics |
| `DoubleRollingAggregate` | integer windows (scalar or a `(left, right)` pair), per-side `agg`/`agg_params`/`min_periods`, `diff` in `l1`, `diff`, `rel_diff`, `abs_rel_diff` |
| kernels | `adtk_rolling`, `adtk_rolling_quantile`, `adtk_rolling_qrange` |

Semantics reproduced from pandas, each covered by a parity test:

- windows are clipped at both ends *from the unclipped left edge*, so the first
  and last outputs of a windowed aggregate are NaN rather than computed from a
  short window;
- `std`/`var` are the sample (ddof=1) forms, `skew` and `kurt` are the
  Fisher-Pearson adjusted moment coefficients, and they need 3 and 4
  observations respectively regardless of `min_periods`;
- `count` is the one aggregate pandas gates on the number of *positions* in the
  window rather than on the number of non-NaN values;
- `nnz` counts non-zero entries including NaN, as `np.count_nonzero` does;
- quantiles use pandas' linear interpolation at index `q * (n - 1)`.

## Not implemented

- `RollingAggregate` with `agg='nunique'`, `agg='hist'`, or a callable `agg`,
  and the vector-output `getRollingVector` machinery. The shim raises
  `ValueError` for these rather than silently falling back.
- Timedelta (string) windows. Only positional integer windows are ported.
- `min_periods=0`, whose degenerate empty-window answers (sum 0, nnz over an
  empty slice) are not reproduced. The shim raises `ValueError`.
- Every detector, transformer, aggregator, pipe, profiler, `trtexec` launcher
  and TensorRT config helper. They are compositions of `RollingAggregate` or
  are pure control flow: use the real `adtk` for them.
- Multi-output `DoubleRollingAggregate` (`diff='l2'` on a vector aggregate) and
  callable `diff`.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build     # -> dist/libmojo-adtk.so
pixi run test
pixi run bench
```

`PYTHONPATH=python` is set by the Pixi activation; outside Pixi, set it
yourself. Never run `pixi install` on the shared toolchain box: the shared
environment is the environment.

## Tests

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

65 parity tests. They compare against the real `adtk` for every covered
aggregate, window size, alignment, `min_periods` value and NaN pattern, plus
hand-checked exact expectations for the window edges, a brute-force order
statistic check for the quickselect, a repeated-call check for scratch-buffer
corruption, and rejection tests for the surfaces that are deliberately absent.

Tolerances: `rtol=atol=1e-11` for the ordinary aggregates (the kernel uses
FMA and a two-pass variance against pandas' incremental update, so bit equality
is not available), `rtol=1e-7` for `skew`/`kurt`, which amplify cancellation.

## Performance

Best-of-five, same process, n = 2,097,152 and window = 32, against
`adtk.transformer.RollingAggregate` (which is `pandas.Series.rolling`, i.e.
Cython, plus a wrapper — the fastest reasonable formulation of the same
reduction). Every case verifies agreement with adtk before timing.

| case | adtk (pandas) | mojo-adtk | result |
| --- | ---: | ---: | ---: |
| rolling mean | 162.51 ms | 205.59 ms | 0.79x, slower |
| rolling std | 183.62 ms | 406.64 ms | 0.45x, slower |
| rolling skew | 194.04 ms | 609.62 ms | 0.32x, slower |
| rolling count | 151.70 ms | 130.08 ms | 1.17x faster |
| rolling q=0.5 | 1675.07 ms | 2869.86 ms | 0.58x, slower |
| rolling iqr | 5703.11 ms | 4607.71 ms | 1.24x faster |
| double rel_diff | 402.85 ms | 423.56 ms | 0.95x, near parity |

The honest reading: pandas' rolling kernels maintain incremental state and
cost O(n) per aggregate, while these kernels re-read each window and cost
O(n * w). That is a structural disadvantage that shows up on every aggregate,
and it is why the plain reductions lose. Where the reduction is genuinely
selection-based rather than incremental, the picture changes: `iqr` needs two
order statistics and wins, because pandas pays for two separate quantile
machinery passes there. `skew` and `kurt` lose because they need three central
moments, i.e. a second pass over the window. Making the additive aggregates
O(n) would mean carrying a running sum, which drifts over millions of updates
and would break the 1e-11 parity tolerance; the direct recomputation is the
right trade for a port whose contract is numerical agreement.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-adtk.so`.

`python/mojo_adtk/transformer.py` owns every array, the index plumbing and the
window policy, then makes one call into the kernel. Buffers cross the C ABI as
64-bit addresses and are rebuilt in Mojo as
`Pointer[Float64, AnyOrigin[mut=True]]`, which is what keeps the exported
symbols non-parametric.

The quantile kernels take a caller-owned scratch buffer rather than allocating
one: `Pointer.alloc` is gone in Mojo 1.2.0, and a non-raising `abi("C")`
function cannot allocate. The window is compacted once and each requested
quantile is read with a Hoare quickselect, which is O(window) per quantile
instead of a sort, and which leaves the buffer partitioned so that a second
selection for the neighbouring order statistic stays correct.

These kernels are memory-local and branch-heavy, so they are plain serial
loops: the first version of this file had a runtime predicate inside the
window scan and ran 5x slower than the branch-per-family version that replaced
it.

## License

MIT
