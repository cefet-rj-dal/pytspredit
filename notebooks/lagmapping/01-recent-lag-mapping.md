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

# Lag mapping -- Recent lags (`TsLagmap(method="recent")`)

**About the technique**

* `TsLagmap(method="recent")` keeps the `input_size` most recent lags (`t1 ... t{input_size}`) -- the default, no selection.

**Didactic goal:** pick the simplest baseline: the latest lags.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

io = ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))
```

```python
lm = ts.TsLagmap(method="recent", seasonality=4).fit(io.input, io.output, input_size=4)
print("selected lags:", list(lm.lags))
print("columns:", lm.columns)
```

> **What to observe.** The strategy returns `input_size` lag positions (`lag 1` = most recent); these are the columns actually fed to the backend.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis: Forecasting and Control*, Wiley.
