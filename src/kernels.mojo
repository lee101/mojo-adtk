"""Rolling-window aggregations: the numeric core of adtk's `RollingAggregate`.

adtk is a pandas time-series toolkit. Almost all of it is plumbing (model
objects, pipes, `trtexec` launching, TensorRT config files). The one place with
a real numeric inner loop is `adtk.transformer.RollingAggregate`, which rolls a
sliding window along a series and reduces it to a scalar with mean / median /
sum / min / max / std / var / skew / kurt / count / nnz / quantile / iqr / idr.
`DoubleRollingAggregate` is the same reduction applied to two adjacent windows
and differenced, so the kernels here are the whole of its cost too.

Every window is contiguous and positionally indexed, so these kernels are
memory-local: a plain serial loop is the right shape (threading a
bandwidth-bound reduction makes it slower on this box).

Buffers cross the C ABI as 64-bit addresses, `@export` rejects parametric
functions, and `AnyOrigin[mut=True]` is the only usable mutable origin, so each
export takes `Int` addresses and rebuilds its own pointer.
"""

from std.math import fma, isnan, pow, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]

# aggregation codes shared with the Python shim
comptime A_MEAN: Int = 0
comptime A_SUM: Int = 1
comptime A_MIN: Int = 2
comptime A_MAX: Int = 3
comptime A_STD: Int = 4
comptime A_VAR: Int = 5
comptime A_SKEW: Int = 6
comptime A_KURT: Int = 7
comptime A_COUNT: Int = 8
comptime A_NNZ: Int = 9


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def nan() -> Float64:
    return Float64(0.0) / Float64(0.0)


def kth_smallest(s: FPtr, cnt: Int, k: Int) -> Float64:
    """Hoare quickselect: the k-th smallest (0-based) of the first `cnt` items.

    The buffer is left partitioned around `k`, so calling it again with a
    larger `k` still yields the true order statistic. That is what lets the
    quantile kernel evaluate several quantiles from one copy of the window
    without re-sorting.
    """
    var left = 0
    var right = cnt - 1
    while True:
        if right <= left:
            return s[unsafe_offset=left]
        var i = left
        var j = right
        var pivot = s[unsafe_offset=(left + right) // 2]
        while i <= j:
            while s[unsafe_offset=i] < pivot:
                i += 1
            while s[unsafe_offset=j] > pivot:
                j -= 1
            if i <= j:
                var t = s[unsafe_offset=i]
                s[unsafe_offset=i] = s[unsafe_offset=j]
                s[unsafe_offset=j] = t
                i += 1
                j -= 1
        if k <= j:
            right = j
        elif k >= i:
            left = i
        else:
            return s[unsafe_offset=k]


def quantile_of(s: FPtr, cnt: Int, q: Float64) -> Float64:
    """Linear-interpolation quantile, the way `pandas.Series.quantile` defines
    it: the fractional index is `q * (cnt - 1)`."""
    if cnt <= 0:
        return nan()
    if cnt == 1:
        return s[unsafe_offset=0]
    var pos = q * Float64(cnt - 1)
    if pos <= 0.0:
        return kth_smallest(s, cnt, 0)
    if pos >= Float64(cnt - 1):
        return kth_smallest(s, cnt, cnt - 1)
    var li = Int(pos)
    var frac = pos - Float64(li)
    # The two neighbours are read with two selections, not two sorts. After
    # the first one the buffer is partitioned around `li`, so the second
    # selection still sees the true order statistics.
    var a = kth_smallest(s, cnt, li)
    var b = kth_smallest(s, cnt, li + 1)
    return fma(frac, b - a, a)


def compact(x: FPtr, lo: Int, end: Int, scratch: FPtr) -> Int:
    """Copy the non-NaN values of `x[lo:end]` to the front of `scratch`."""
    var cnt = 0
    for k in range(lo, end):
        var v = x[unsafe_offset=k]
        if not isnan(v):
            scratch[unsafe_offset=cnt] = v
            cnt += 1
    return cnt


def moments(
    x: FPtr, lo: Int, end: Int, mean: Float64, cnt: Int
) -> Tuple[Float64, Float64, Float64]:
    """Central moments m2, m3, m4 (each divided by `cnt`) about `mean`."""
    var m2 = Float64(0.0)
    var m3 = Float64(0.0)
    var m4 = Float64(0.0)
    for k in range(lo, end):
        var v = x[unsafe_offset=k]
        if not isnan(v):
            var d = v - mean
            var d2 = d * d
            m2 += d2
            m3 = fma(d2, d, m3)
            m4 = fma(d2, d2, m4)
    var fcnt = Float64(cnt)
    return (m2 / fcnt, m3 / fcnt, m4 / fcnt)


@export("adtk_rolling")
def adtk_rolling(x_addr: Int, n: Int, res_addr: Int, w: Int, code: Int,
                 scratch_addr: Int, min_periods: Int, center: Int) abi("C"):
    """One rolling aggregate per output position.

    `x` is n values, `res` is n results, `scratch` is a caller-owned float64
    buffer of at least `w` elements (workspace; contents on return are
    undefined). NaN handling follows pandas: NaNs are skipped by every
    aggregate, and a window yields NaN unless the number of non-NaN
    observations reaches `min_periods`. `count` is the one aggregate pandas
    gates on the number of positions in the window instead, which is why it
    needs its own branch below.
    """
    var x = fp(x_addr)
    var res = fp(res_addr)
    for i in range(n):
        # pandas rolling(window, closed=None): the window hangs off the right
        # edge, or is centred with its left edge at i - w // 2. Both ends are
        # then clipped independently, from the *unclipped* left edge, so a
        # window that runs off the start of the series stays short instead of
        # sliding back in. That is what leaves the leading outputs NaN.
        var lo = i - w + 1
        if center == 1:
            lo = i - w // 2
        var end = lo + w
        if lo < 0:
            lo = 0
        if end > n:
            end = n

        var cnt = 0
        var nz = 0
        var s1 = Float64(0.0)
        var mn = Float64(0.0)
        var mx = Float64(0.0)
        # One loop per aggregate family, branched outside the window scan:
        # a runtime predicate inside the scan stops the compiler vectorising
        # it, and this loop is the whole cost of the kernel.
        if code == A_COUNT:
            for k in range(lo, end):
                if not isnan(x[unsafe_offset=k]):
                    cnt += 1
        elif code == A_NNZ:
            for k in range(lo, end):
                var v = x[unsafe_offset=k]
                if not isnan(v):
                    cnt += 1
                if v != Float64(0.0):
                    nz += 1
        elif code == A_MIN or code == A_MAX:
            for k in range(lo, end):
                var v = x[unsafe_offset=k]
                if not isnan(v):
                    cnt += 1
                    if cnt == 1:
                        mn = v
                        mx = v
                    else:
                        if v < mn:
                            mn = v
                        if v > mx:
                            mx = v
        else:
            for k in range(lo, end):
                var v = x[unsafe_offset=k]
                if not isnan(v):
                    cnt += 1
                    s1 += v

        var r = nan()
        if code == A_COUNT:
            # pandas' roll_count gates on the number of observations in the
            # window, not on the number of non-NaN ones: a four-slot window
            # holding one value reports 1.0 with min_periods=3.
            if end - lo >= min_periods:
                r = Float64(cnt)
        elif cnt >= min_periods:
            var fcnt = Float64(cnt)
            if code == A_NNZ:
                r = Float64(nz)
            elif code == A_MEAN:
                r = s1 / fcnt
            elif code == A_SUM:
                r = s1
            elif code == A_MIN:
                if cnt > 0:
                    r = mn
            elif code == A_MAX:
                if cnt > 0:
                    r = mx
            elif code == A_VAR or code == A_STD:
                if cnt > 1:
                    var mean = s1 / fcnt
                    var mm = moments(x, lo, end, mean, cnt)
                    r = mm[0] * Float64(cnt) / Float64(cnt - 1)
                    if code == A_STD:
                        r = sqrt(r)
            elif code == A_SKEW:
                if cnt > 2:
                    var mean = s1 / fcnt
                    var mm = moments(x, lo, end, mean, cnt)
                    var g1 = mm[1] / pow(mm[0], 1.5)
                    r = sqrt(Float64(cnt * (cnt - 1))) / Float64(cnt - 2) * g1
            elif code == A_KURT:
                if cnt > 3:
                    var mean = s1 / fcnt
                    var mm = moments(x, lo, end, mean, cnt)
                    var g2 = mm[2] / (mm[0] * mm[0]) - 3.0
                    r = Float64(cnt - 1) / Float64((cnt - 2) * (cnt - 3)) * (
                        Float64(cnt + 1) * g2 + 6.0
                    )
        res[unsafe_offset=i] = r


@export("adtk_rolling_quantile")
def adtk_rolling_quantile(x_addr: Int, n: Int, res_addr: Int, w: Int,
                          qs_addr: Int, nq: Int, scratch_addr: Int,
                          min_periods: Int, center: Int) abi("C"):
    """Rolling quantiles. `res` is n x nq, row major, one row per position.

    The window is compacted once and every requested quantile is read off the
    same quickselect buffer, so an `iqr` costs two selections over one copy
    rather than two sorts.
    """
    var x = fp(x_addr)
    var res = fp(res_addr)
    var qs = fp(qs_addr)
    var scratch = fp(scratch_addr)
    for i in range(n):
        var lo = i - w + 1
        if center == 1:
            lo = i - w // 2
        var end = lo + w
        if lo < 0:
            lo = 0
        if end > n:
            end = n
        var row = i * nq
        for j in range(nq):
            res[unsafe_offset=row + j] = nan()
        var cnt = compact(x, lo, end, scratch)
        if cnt >= min_periods and cnt > 0:
            for j in range(nq):
                res[unsafe_offset=row + j] = quantile_of(
                    scratch, cnt, qs[unsafe_offset=j]
                )


@export("adtk_rolling_qrange")
def adtk_rolling_qrange(x_addr: Int, n: Int, res_addr: Int, w: Int,
                        lo_q: Float64, hi_q: Float64, scratch_addr: Int,
                        min_periods: Int, center: Int) abi("C"):
    """Rolling `quantile(hi_q) - quantile(lo_q)`: adtk's `iqr` and `idr`."""
    var x = fp(x_addr)
    var res = fp(res_addr)
    var scratch = fp(scratch_addr)
    for i in range(n):
        var lo = i - w + 1
        if center == 1:
            lo = i - w // 2
        var end = lo + w
        if lo < 0:
            lo = 0
        if end > n:
            end = n
        var r = nan()
        var cnt = compact(x, lo, end, scratch)
        if cnt >= min_periods and cnt > 0:
            r = quantile_of(scratch, cnt, hi_q) - quantile_of(
                scratch, cnt, lo_q
            )
        res[unsafe_offset=i] = r
