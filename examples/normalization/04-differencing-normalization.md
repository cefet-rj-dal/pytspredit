# Differencing normalization (`TsNormDiff`)

**About the technique**

* `TsNormDiff` replaces each window by its successive differences, removing trend and
  helping stationarity before modeling. It is the normalization that makes linear
  window models (e.g. `TsDarima`) well-conditioned.

**Didactic goal:** stabilize the level/trend so the model learns the dynamics, not the
drift.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
```

Visualize the series used in this example:


```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```





![png](fig/04-differencing-normalization/output_3_0.png)






![png](fig/04-differencing-normalization/output_3_1.png)




```python
norm = ts.TsNormDiff().fit(w)
t = norm.transform(w)
ts.ts_head(t)
```




             t3        t2        t1        t0
    0  0.998201  0.967289  0.907386  0.822218
    1  0.967289  0.907386  0.822218  0.717079
    2  0.907386  0.822218  0.717079  0.598507
    3  0.822218  0.717079  0.598507  0.473873
    4  0.717079  0.598507  0.473873  0.350928
    5  0.598507  0.473873  0.350928  0.237314




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



> **What to observe.** The transformed windows hold differences; `inverse_transform`
> integrates them back to the original level.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis:
Forecasting and Control*, Wiley.
