"""ctypes bridge to the compiled Mojo rolling-window kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below stay `c_int64` for addresses; `c_int` truncates
them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-adtk.so"

# codes shared with src/kernels.mojo
MEAN = 0
SUM = 1
MIN = 2
MAX = 3
STD = 4
VAR = 5
SKEW = 6
KURT = 7
COUNT = 8
NNZ = 9

_SCALAR_CODES = {
    "mean": MEAN,
    "sum": SUM,
    "min": MIN,
    "max": MAX,
    "std": STD,
    "var": VAR,
    "skew": SKEW,
    "kurt": KURT,
    "count": COUNT,
    "nnz": NNZ,
}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.adtk_rolling.restype = None
    lib.adtk_rolling.argtypes = [ctypes.c_int64] * 8
    lib.adtk_rolling_quantile.restype = None
    lib.adtk_rolling_quantile.argtypes = [ctypes.c_int64] * 9
    lib.adtk_rolling_qrange.restype = None
    lib.adtk_rolling_qrange.argtypes = (
        [ctypes.c_int64] * 4 + [ctypes.c_double] * 2 + [ctypes.c_int64] * 3
    )
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def rolling_scalar(x, window, agg, min_periods=None, center=False):
    """Rolling mean/sum/min/max/std/var/skew/kurt/count/nnz over a 1-D array."""
    values = np.ascontiguousarray(x, dtype=np.float64)
    n = values.size
    res = np.empty(n, dtype=np.float64)
    scratch = np.empty(max(int(window), 1), dtype=np.float64)
    code = _SCALAR_CODES[agg]
    mp = int(window) if min_periods is None else int(min_periods)
    lib.adtk_rolling(
        _addr(values), n, _addr(res), int(window), code, _addr(scratch), mp,
        1 if center else 0,
    )
    return res


def rolling_quantile(x, window, qs, min_periods=None, center=False):
    """Rolling quantiles. Returns an (n, len(qs)) array, one row per position."""
    values = np.ascontiguousarray(x, dtype=np.float64)
    n = values.size
    qarr = np.ascontiguousarray(qs, dtype=np.float64)
    res = np.empty((n, qarr.size), dtype=np.float64)
    scratch = np.empty(max(int(window), 1), dtype=np.float64)
    mp = int(window) if min_periods is None else int(min_periods)
    lib.adtk_rolling_quantile(
        _addr(values), n, _addr(res), int(window), _addr(qarr), qarr.size,
        _addr(scratch), mp, 1 if center else 0,
    )
    return res


def rolling_qrange(x, window, lo_q, hi_q, min_periods=None, center=False):
    """Rolling quantile(hi_q) - quantile(lo_q), i.e. adtk's iqr and idr."""
    values = np.ascontiguousarray(x, dtype=np.float64)
    n = values.size
    res = np.empty(n, dtype=np.float64)
    scratch = np.empty(max(int(window), 1), dtype=np.float64)
    mp = int(window) if min_periods is None else int(min_periods)
    lib.adtk_rolling_qrange(
        _addr(values), n, _addr(res), int(window), ctypes.c_double(lo_q),
        ctypes.c_double(hi_q), _addr(scratch), mp, 1 if center else 0,
    )
    return res
