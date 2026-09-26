"""Mojo port of the compute core of NVIDIA's adtk.

adtk is a pandas time-series toolkit for anomaly detection. Its numeric core is
the rolling-window aggregation in `adtk.transformer`; everything else (pipelines,
model plumbing, `trtexec` launching, TensorRT config files) is control flow and
is left to the real package.

```python
import pandas as pd
import mojo_adtk

s = pd.Series(range(10), dtype="float64")
mojo_adtk.RollingAggregate(3, agg="mean").transform(s)
```
"""

from .transformer import DoubleRollingAggregate, RollingAggregate

__all__ = ["RollingAggregate", "DoubleRollingAggregate"]
__version__ = "0.1.0"
