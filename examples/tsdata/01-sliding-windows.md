# Sliding Windows (ts_data)

Time series forecasting with regression models works on *sliding windows*: each
window is a short slice of consecutive observations, turned into one training
example. `ts_data(y, sw)` builds this representation from a raw series, producing a
table with `sw` lag columns named `t{sw-1} ... t0`, from oldest to most recent.
`t0` is always the most recent value — the one-step-ahead target once the window is
projected.

**Reference:** Ogasawara, E., Murta, L., Zimbrão, G., Mattoso, M. (2009). *Neural
networks cartridges for data mining on time series.* IJCNN.


```python
import pytspredit as ts

tsd = ts.datasets.tsd()
tsd.head()
```




          x         y
    0  0.00  0.000000
    1  0.25  0.247404
    2  0.50  0.479426
    3  0.75  0.681639
    4  1.00  0.841471




```python
# a single-column representation (sw = 1): the raw series as column t0
ts.ts_data(tsd["y"]).head()
```




             t0
    0  0.000000
    1  0.247404
    2  0.479426
    3  0.681639
    4  0.841471




```python
# 10-lag sliding windows: columns t9 ... t0, one row per window
windows = ts.ts_data(tsd["y"], sw=10)
print("shape:", windows.shape)
ts.ts_head(windows, 3)
```

    shape: (32, 10)
    




             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073
    1  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472
    2  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661



Row `i` holds `y[i], y[i+1], ..., y[i+9]`. Consecutive rows overlap, shifted by one
step — exactly the structure a regressor needs to learn the mapping from recent
lags (`t9 ... t1`) to the next value (`t0`).
