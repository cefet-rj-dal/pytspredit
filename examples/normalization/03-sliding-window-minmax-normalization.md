# Sliding-window min-max normalization (`TsNormSwminmax`)

**About the technique**

* `TsNormSwminmax` rescales **each window by its own** min and max, so every row is
  mapped into `[0, 1]` independently. This removes slow level/trend drift between
  windows.

**Didactic goal:** normalize locally so the model sees the within-window shape rather
than the absolute level.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
```


```python
norm = ts.TsNormSwminmax().fit(w)
t = norm.transform(w)
ts.ts_head(t)
```




             t4        t3        t2        t1        t0
    0  0.000000  0.294014  0.569747  0.810056  1.000000
    1  0.000000  0.330713  0.618938  0.846755  1.000000
    2  0.000000  0.390321  0.698836  0.906363  1.000000
    3  0.000000  0.506028  0.846416  1.000000  0.957230
    4  0.000000  0.689084  1.000000  0.913417  0.434718
    5  0.778917  1.000000  0.938433  0.598046  0.000000




```python
ts.ts_head(norm.inverse_transform(t))
```




             t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471
    1  0.247404  0.479426  0.681639  0.841471  0.948985
    2  0.479426  0.681639  0.841471  0.948985  0.997495
    3  0.681639  0.841471  0.948985  0.997495  0.983986
    4  0.841471  0.948985  0.997495  0.983986  0.909297
    5  0.948985  0.997495  0.983986  0.909297  0.778073



> **What to observe.** Each row is scaled by its own range (per-window `[0, 1]`); the
> inverse restores the original windows using the stored per-row min/max.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
