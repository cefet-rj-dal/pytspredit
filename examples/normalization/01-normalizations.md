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




             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073
    1  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472
    2  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661



## No normalization (`TsNormNone`)

Identity transform — keeps the pipeline interface without changing the data.


```python
ts.ts_head(ts.TsNormNone().fit(windows).transform(windows), 3)
```




             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073
    1  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472
    2  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661



## Global min–max (`TsNormGminmax`)

Maps the global training minimum to 0 and maximum to 1.


```python
norm = ts.TsNormGminmax().fit(windows)
tr = norm.transform(windows)
print("range:", float(tr.to_numpy().min()), "to", float(tr.to_numpy().max()))
ts.ts_head(tr, 3)
```

    range: 0.0 to 1.0





             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.500450  0.624351  0.740549  0.841818  0.921863  0.975706  1.000000  0.993235  0.955830  0.890113
    1  0.624351  0.740549  0.841818  0.921863  0.975706  1.000000  0.993235  0.955830  0.890113  0.800168
    2  0.740549  0.841818  0.921863  0.975706  1.000000  0.993235  0.955830  0.890113  0.800168  0.691588



## Per-window min–max (`TsNormSwminmax`)

Normalizes each window by its own min/max — preserves local shape, removes level.


```python
ts.ts_head(ts.TsNormSwminmax().fit(windows).transform(windows), 3)
```




             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.000000  0.248025  0.480630  0.683351  0.843584  0.951368  1.000000  0.986457  0.911581  0.780027
    1  0.000000  0.309325  0.578909  0.791993  0.935327  1.000000  0.981990  0.882418  0.707473  0.468034
    2  0.158751  0.487108  0.746646  0.921228  1.000000  0.978064  0.856784  0.643700  0.352061  0.000000



## First differences (`TsNormDiff`)

Differences consecutive lags (window width drops by one), then applies global
min–max. The target becomes the change from the most recent lag.


```python
tr = ts.TsNormDiff().fit(windows).transform(windows)
print("width:", windows.shape[1], "->", tr.shape[1])
ts.ts_head(tr, 3)
```

    width: 10 -> 9





             t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.998201  0.967289  0.907386  0.822218  0.717079  0.598507  0.473873  0.350928  0.237314
    1  0.967289  0.907386  0.822218  0.717079  0.598507  0.473873  0.350928  0.237314  0.140097
    2  0.907386  0.822218  0.717079  0.598507  0.473873  0.350928  0.237314  0.140097  0.065320



## Adaptive normalization (`TsNormAn`)

Normalizes each window relative to an adaptive reference (mean or EMA) and optional
local scale (sd/MAD), via one of four operators (`divide`, `subtract`, `softdivide`,
`asinh`), followed by global min–max.


```python
for op in ["divide", "subtract", "softdivide", "asinh"]:
    tr = ts.TsNormAn(operation=op).fit(windows).transform(windows)
    print(f"{op:10s} -> mean {tr.to_numpy().mean():+.4f}, std {tr.to_numpy().std():.4f}")
```

    divide     -> mean +0.5651, std 1.3326


    subtract   -> mean +0.5004, std 0.2286
    softdivide -> mean +0.4999, std 0.2175


    asinh      -> mean +0.4965, std 0.2064


Every normalizer is invertible, so a forecast produced in the normalized space is
returned on the original scale:


```python
norm = ts.TsNormGminmax().fit(windows)
back = norm.inverse_transform(norm.transform(windows))
import numpy as np
print("max round-trip error:", float(np.abs(back.to_numpy() - windows.to_numpy()).max()))
```

    max round-trip error:

     1.1102230246251565e-16

