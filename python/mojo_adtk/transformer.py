"""Rolling-window transformers with the same call shape as adtk's.

`adtk.transformer.RollingAggregate` is the one part of adtk with a numeric
inner loop, so it is the part ported here. The class signature, the `agg`
names, the `center`/`min_periods` window alignment and the return types match
upstream, which is what the parity tests compare against.
"""

import numpy as np
import pandas as pd

from . import _lib

__all__ = ["RollingAggregate", "DoubleRollingAggregate"]

_SCALAR_AGGS = frozenset(_lib._SCALAR_CODES)
_SUPPORTED = _SCALAR_AGGS | {"median", "quantile", "iqr", "idr"}


def _to_float(s):
    if isinstance(s, pd.Series):
        values = s.to_numpy()
    else:
        values = np.asarray(s)
    try:
        return np.ascontiguousarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "the input time series must be numeric float64-compatible"
        ) from exc


def _check_index(s):
    if isinstance(s, pd.Series) and not (
        s.index.is_monotonic_increasing or s.index.is_monotonic_decreasing
    ):
        raise ValueError("Time series must have a monotonic time index. ")


def _rolling(values, window, agg, agg_params, min_periods, center):
    """Dispatch one rolling aggregate. Returns a 1-D or 2-D float64 array."""
    if min_periods is not None and int(min_periods) < 1:
        # pandas allows min_periods=0, which gives degenerate answers for
        # empty windows (count 0, sum 0, nnz over an empty slice). Those are
        # not reproduced here; use the real adtk for that corner.
        raise ValueError("min_periods must be at least 1 in this port")
    if agg in _SCALAR_AGGS:
        return _lib.rolling_scalar(
            values, window, agg, min_periods=min_periods, center=center
        )
    if agg == "median":
        return _lib.rolling_quantile(
            values, window, [0.5], min_periods=min_periods, center=center
        )[:, 0]
    if agg == "quantile":
        q = (agg_params or {})["q"]
        qs = [float(q)] if not hasattr(q, "__iter__") else [float(v) for v in q]
        res = _lib.rolling_quantile(
            values, window, qs, min_periods=min_periods, center=center
        )
        return res[:, 0] if len(qs) == 1 else res
    if agg == "iqr":
        return _lib.rolling_qrange(
            values, window, 0.25, 0.75, min_periods=min_periods, center=center
        )
    if agg == "idr":
        return _lib.rolling_qrange(
            values, window, 0.1, 0.9, min_periods=min_periods, center=center
        )
    raise ValueError(f"Attribute agg must be one of {sorted(_SUPPORTED)}")


def _wrap(res, source, names=None):
    name = source.name if isinstance(source, pd.Series) else None
    if names is None:
        out = pd.Series(np.asarray(res, dtype=np.float64), name=name)
    else:
        out = pd.DataFrame(
            np.asarray(res, dtype=np.float64), columns=names, index=None
        )
    if isinstance(source, pd.Series):
        out.index = source.index
    return out


def _shift(values, periods):
    n = values.size
    if periods <= 0:
        return values.copy()
    if periods >= n:
        return np.full(n, np.nan)
    out = np.full(n, np.nan)
    out[periods:] = values[:-periods]
    return out


class RollingAggregate:
    """Mojo implementation of `adtk.transformer.RollingAggregate`.

    Supported `agg` values: mean, median, sum, min, max, std, var, skew,
    kurt, count, nnz, quantile, iqr, idr. `nunique`, `hist` and callable
    aggregations are not ported (see the README).
    """

    def __init__(self, window, agg="mean", agg_params=None, center=False,
                 min_periods=None):
        if not isinstance(window, int) or window <= 0:
            raise ValueError("this port only supports integer windows")
        if callable(agg):
            raise ValueError("callable aggregations are not ported")
        if agg not in _SUPPORTED:
            raise ValueError(f"Attribute agg must be one of {sorted(_SUPPORTED)}")
        self.window = window
        self.agg = agg
        self.agg_params = agg_params
        self.center = center
        self.min_periods = min_periods

    @property
    def _param_names(self):
        return ("window", "agg", "agg_params", "center", "min_periods")

    def get_params(self):
        return {key: getattr(self, key) for key in self._param_names}

    def set_params(self, **params):
        for key in params:
            if key not in self._param_names:
                raise KeyError(f"'{key}' is not a valid parameter name.")
        for key, value in params.items():
            setattr(self, key, value)

    def transform(self, s):
        _check_index(s)
        values = _to_float(s)
        res = _rolling(
            values, self.window, self.agg, self.agg_params, self.min_periods,
            self.center,
        )
        if self.agg == "quantile":
            q = (self.agg_params or {})["q"]
            if hasattr(q, "__iter__"):
                return _wrap(res, s, names=["q{}".format(v) for v in q])
        return _wrap(res, s)

    predict = transform

    def fit(self, s):
        return self

    fit_predict = transform


class DoubleRollingAggregate:
    """Mojo implementation of `adtk.transformer.DoubleRollingAggregate`.

    Only integer windows are ported, and only for scalar aggregated metrics
    (so `diff` cannot be `l2` on a vector result, and callable `diff` is not
    ported). Semantics follow upstream: the left window is the series shifted
    by the right window size, the right window is the current window, and the
    two are then differenced.
    """

    def __init__(self, window, agg="mean", agg_params=None, center=True,
                 min_periods=None, diff="l1"):
        windows = window if isinstance(window, tuple) else (window, window)
        for w in windows:
            if not isinstance(w, int) or w <= 0:
                raise ValueError("this port only supports integer windows")
        if callable(diff):
            raise ValueError("callable diff is not ported")
        if diff not in ("l1", "diff", "rel_diff", "abs_rel_diff"):
            raise ValueError("Invalid value of diff")
        aggs = agg if isinstance(agg, tuple) else (agg, agg)
        for a in aggs:
            if callable(a) or a not in _SUPPORTED:
                raise ValueError(f"Attribute agg must be one of {sorted(_SUPPORTED)}")
        if agg == "quantile":
            q = (agg_params or {})["q"]
            if hasattr(q, "__iter__"):
                raise ValueError("vector aggregated metrics are not ported")
        self.window = window
        self.agg = agg
        self.agg_params = agg_params
        self.center = center
        self.min_periods = min_periods
        self.diff = diff

    @property
    def _param_names(self):
        return ("window", "agg", "agg_params", "center", "min_periods")

    def get_params(self):
        return {key: getattr(self, key) for key in self._param_names}

    def set_params(self, **params):
        for key in params:
            if key not in self._param_names:
                raise KeyError(f"'{key}' is not a valid parameter name.")
        for key, value in params.items():
            setattr(self, key, value)

    def transform(self, s):
        _check_index(s)
        values = _to_float(s)
        window = self.window if isinstance(self.window, tuple) else (
            self.window, self.window
        )
        agg = self.agg if isinstance(self.agg, tuple) else (self.agg, self.agg)
        agg_params = self.agg_params if isinstance(self.agg_params, tuple) else (
            self.agg_params, self.agg_params
        )
        min_periods = (
            self.min_periods if isinstance(self.min_periods, tuple)
            else (self.min_periods, self.min_periods)
        )
        if self.center:
            left = _rolling(
                _shift(values, 1), window[0], agg[0], agg_params[0],
                min_periods[0], False,
            )
            right = _rolling(
                values[::-1].copy(), window[1], agg[1], agg_params[1],
                min_periods[1], False,
            )[::-1]
        else:
            left = _rolling(
                _shift(values, window[1]), window[0], agg[0], agg_params[0],
                min_periods[0], False,
            )
            right = _rolling(
                values, window[1], agg[1], agg_params[1], min_periods[1], False
            )
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if self.diff == "l1":
            res = np.abs(right - left)
        elif self.diff == "diff":
            res = right - left
        elif self.diff == "rel_diff":
            res = (right - left) / left
        else:
            res = np.abs(right - left) / left
        return _wrap(res, s)

    predict = transform

    def fit(self, s):
        return self

    fit_predict = transform
