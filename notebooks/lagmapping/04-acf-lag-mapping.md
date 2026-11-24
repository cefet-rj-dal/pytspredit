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

# Lag mapping -- ACF lags (`TsLagmap(method="acf")`)

**About the technique**

* `TsLagmap(method="acf")` selects the lags with the strongest autocorrelation with the target.

**Didactic goal:** let the autocorrelation structure choose the lags.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

io = ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))
```

Visualize the series used in this example:

```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```


```python
lm = ts.TsLagmap(method="acf", seasonality=4).fit(io.input, io.output, input_size=4)
print("selected lags:", list(lm.lags))
print("columns:", lm.columns)
```

> **What to observe.** The strategy returns `input_size` lag positions (`lag 1` = most recent); these are the columns actually fed to the backend.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis: Forecasting and Control*, Wiley.
