# Build `ts_data`

**About the technique**

* `ts_data()` converts a raw time series into the tabular representation used by
  `pytspredit`.
* Each row stores one forecasting context; the lag columns (`t{k}`) define what the
  model can observe before predicting the next value (`t0`).

**Didactic goal:** understand the internal representation that projection,
preprocessing, and prediction examples rely on.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```

Load the example series.


```python
d = ts.datasets.tsd()
y = d["y"]
y.head()
```




    0    0.000000
    1    0.247404
    2    0.479426
    3    0.681639
    4    0.841471
    Name: y, dtype: float64



Build `ts_data` without a sliding window (only `t0`).


```python
base = ts.ts_data(y)
ts.ts_head(base)
```




             t0
    0  0.000000
    1  0.247404
    2  0.479426
    3  0.681639
    4  0.841471
    5  0.948985



Rebuild with a sliding window of size 10, so each row stores the recent lags
explicitly (`t9 ... t0`).


```python
w = ts.ts_data(y, 10)
ts.ts_head(w)
```




             t9        t8        t7        t6        t5        t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073
    1  0.247404  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472
    2  0.479426  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661
    3  0.681639  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661  0.141120
    4  0.841471  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661  0.141120 -0.108195
    5  0.948985  0.997495  0.983986  0.909297  0.778073  0.598472  0.381661  0.141120 -0.108195 -0.350783



The representation is a plain `pandas.DataFrame`, so ordinary indexing works.


```python
w.iloc[11]            # one row (a forecasting context)
```




    t9    0.381661
    t8    0.141120
    t7   -0.108195
    t6   -0.350783
    t5   -0.571561
    t4   -0.756802
    t3   -0.894989
    t2   -0.977530
    t1   -0.999293
    t0   -0.958924
    Name: 11, dtype: float64




```python
w.iloc[11:13, 0:2]    # a block of rows and columns
```




              t9        t8
    11  0.381661  0.141120
    12  0.141120 -0.108195



> **What to observe.** `ts_data(y, sw)` returns `len(y) - sw + 1` rows and `sw`
> columns `t{sw-1} ... t0`; `t0` is the most recent value (the one-step target).

**References:** Salles et al. (2023), *TSPredIT: Integrated Tuning of Data
Preprocessing and Time Series Prediction Models*, LNCS 14160.
