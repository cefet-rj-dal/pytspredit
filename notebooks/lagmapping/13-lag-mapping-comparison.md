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

Visualize the series used in this example:

```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```


```python
for m in methods:
    lm = ts.TsLagmap(method=m, seasonality=4).fit(io.input, io.output, input_size=4)
    print(f"{m:14s} lags={list(lm.lags)}  columns={lm.columns}")
```

> **What to observe.** All strategies return `input_size` lags but choose different
> positions; pass one as `input_map=ts.TsLagmap('pacf')` to a regressor to select lags
> by that evidence instead of simply keeping the most recent.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis*; Peng, Long &
Ding (2005), *mRMR feature selection*, IEEE TPAMI 27(8).
