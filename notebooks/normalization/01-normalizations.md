---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Time Series Normalizations

Distance- and gradient-based regressors learn better when the sliding windows are
on a common scale. `pytspredit` offers five normalizers, each a `DalTransform` with
`fit` / `transform` / `inverse_transform` (so predictions can be mapped back to the
original scale). The `x=` argument (de)normalizes an aligned target/forecast with the
statistics of its windows.

**Reference:** Ogasawara, E. et al. (2009). *Neural network cartridges for data
mining on time series.* IJCNN. (Adaptive normalization: Ogasawara, E. et al. (2010),
*Adaptive Normalization*, IJCNN.)

```python
import pytspredit as ts

windows = ts.ts_data(ts.datasets.tsd()["y"], sw=10)
ts.ts_head(windows, 3)
```

## No normalization (`TsNormNone`)

Identity transform — keeps the pipeline interface without changing the data.

```python
ts.ts_head(ts.TsNormNone().fit(windows).transform(windows), 3)
```

## Global min–max (`TsNormGminmax`)

Maps the global training minimum to 0 and maximum to 1.

```python
norm = ts.TsNormGminmax().fit(windows)
tr = norm.transform(windows)
print("range:", float(tr.to_numpy().min()), "to", float(tr.to_numpy().max()))
ts.ts_head(tr, 3)
```

## Per-window min–max (`TsNormSwminmax`)

Normalizes each window by its own min/max — preserves local shape, removes level.

```python
ts.ts_head(ts.TsNormSwminmax().fit(windows).transform(windows), 3)
```

## First differences (`TsNormDiff`)

Differences consecutive lags (window width drops by one), then applies global
min–max. The target becomes the change from the most recent lag.

```python
tr = ts.TsNormDiff().fit(windows).transform(windows)
print("width:", windows.shape[1], "->", tr.shape[1])
ts.ts_head(tr, 3)
```

## Adaptive normalization (`TsNormAn`)

Normalizes each window relative to an adaptive reference (mean or EMA) and optional
local scale (sd/MAD), via one of four operators (`divide`, `subtract`, `softdivide`,
`asinh`), followed by global min–max.

```python
for op in ["divide", "subtract", "softdivide", "asinh"]:
    tr = ts.TsNormAn(operation=op).fit(windows).transform(windows)
    print(f"{op:10s} -> mean {tr.to_numpy().mean():+.4f}, std {tr.to_numpy().std():.4f}")
```

Every normalizer is invertible, so a forecast produced in the normalized space is
returned on the original scale:

```python
norm = ts.TsNormGminmax().fit(windows)
back = norm.inverse_transform(norm.transform(windows))
import numpy as np
print("max round-trip error:", float(np.abs(back.to_numpy() - windows.to_numpy()).max()))
```
