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

Visualize the series used in this example:

```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```


Load the example series.

```python
d = ts.datasets.tsd()
y = d["y"]
y.head()
```

Build `ts_data` without a sliding window (only `t0`).

```python
base = ts.ts_data(y)
ts.ts_head(base)
```

Rebuild with a sliding window of size 10, so each row stores the recent lags
explicitly (`t9 ... t0`).

```python
w = ts.ts_data(y, 10)
ts.ts_head(w)
```

The representation is a plain `pandas.DataFrame`, so ordinary indexing works.

```python
w.iloc[11]            # one row (a forecasting context)
```

```python
w.iloc[11:13, 0:2]    # a block of rows and columns
```

> **What to observe.** `ts_data(y, sw)` returns `len(y) - sw + 1` rows and `sw`
> columns `t{sw-1} ... t0`; `t0` is the most recent value (the one-step target).

**References:** Salles et al. (2023), *TSPredIT: Integrated Tuning of Data
Preprocessing and Time Series Prediction Models*, LNCS 14160.
