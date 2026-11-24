# Lag mapping -- strategy comparison

**About the technique**

* `TsLagmap` offers twelve strategies to choose which `input_size` lags a sliding-window
  regressor sees, grouped by the evidence they use: positional (`recent`, `even`,
  `geom`), correlation-driven (`acf`, `pacf`, `peaks`, `seasonal`, `acf_seasonal`,
  `pacf_seasonal`, `blocks`) and supervised (`mi`, `mrmr`).

**Didactic goal:** contrast the lags each strategy selects on the same data.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

io = ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))
methods = ["recent", "even", "geom", "acf", "pacf", "peaks",
           "seasonal", "acf_seasonal", "pacf_seasonal", "blocks", "mi", "mrmr"]
```


```python
for m in methods:
    lm = ts.TsLagmap(method=m, seasonality=4).fit(io.input, io.output, input_size=4)
    print(f"{m:14s} lags={list(lm.lags)}  columns={lm.columns}")
```

    recent         lags=[np.int64(4), np.int64(3), np.int64(2), np.int64(1)]  columns=['t4', 't3', 't2', 't1']
    even           lags=[np.int64(9), np.int64(6), np.int64(4), np.int64(1)]  columns=['t9', 't6', 't4', 't1']
    geom           lags=[np.int64(9), np.int64(4), np.int64(2), np.int64(1)]  columns=['t9', 't4', 't2', 't1']
    acf            lags=[np.int64(4), np.int64(3), np.int64(2), np.int64(1)]  columns=['t4', 't3', 't2', 't1']
    pacf           lags=[np.int64(4), np.int64(3), np.int64(2), np.int64(1)]  columns=['t4', 't3', 't2', 't1']
    peaks          lags=[np.int64(4), np.int64(3), np.int64(2), np.int64(1)]  columns=['t4', 't3', 't2', 't1']
    seasonal       lags=[np.int64(8), np.int64(4), np.int64(2), np.int64(1)]  columns=['t8', 't4', 't2', 't1']
    acf_seasonal   lags=[np.int64(8), np.int64(4), np.int64(2), np.int64(1)]  columns=['t8', 't4', 't2', 't1']
    pacf_seasonal  lags=[np.int64(8), np.int64(4), np.int64(2), np.int64(1)]  columns=['t8', 't4', 't2', 't1']
    blocks         lags=[np.int64(4), np.int64(3), np.int64(2), np.int64(1)]  columns=['t4', 't3', 't2', 't1']


    mi             lags=[np.int64(9), np.int64(5), np.int64(4), np.int64(2)]  columns=['t9', 't5', 't4', 't2']


    mrmr           lags=[np.int64(9), np.int64(6), np.int64(2), np.int64(1)]  columns=['t9', 't6', 't2', 't1']


> **What to observe.** All strategies return `input_size` lags but choose different
> positions; pass one as `input_map=ts.TsLagmap('pacf')` to a regressor to select lags
> by that evidence instead of simply keeping the most recent.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis*; Peng, Long &
Ding (2005), *mRMR feature selection*, IEEE TPAMI 27(8).
