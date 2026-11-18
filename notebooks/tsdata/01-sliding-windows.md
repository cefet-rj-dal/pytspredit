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

```python
# a single-column representation (sw = 1): the raw series as column t0
ts.ts_data(tsd["y"]).head()
```

```python
# 10-lag sliding windows: columns t9 ... t0, one row per window
windows = ts.ts_data(tsd["y"], sw=10)
print("shape:", windows.shape)
ts.ts_head(windows, 3)
```

Row `i` holds `y[i], y[i+1], ..., y[i+9]`. Consecutive rows overlap, shifted by one
step — exactly the structure a regressor needs to learn the mapping from recent
lags (`t9 ... t1`) to the next value (`t0`).
