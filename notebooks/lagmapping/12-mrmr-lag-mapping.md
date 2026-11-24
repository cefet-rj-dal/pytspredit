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

# Lag mapping -- mRMR lags (`TsLagmap(method="mrmr")`)

**About the technique**

* `TsLagmap(method="mrmr")` selects lags that are relevant to the target yet non-redundant with each other (min-redundancy max-relevance).

**Didactic goal:** avoid redundant lags while keeping relevance.

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
lm = ts.TsLagmap(method="mrmr", seasonality=4).fit(io.input, io.output, input_size=4)
print("selected lags:", list(lm.lags))
print("columns:", lm.columns)
```

> **What to observe.** The strategy returns `input_size` lag positions (`lag 1` = most recent); these are the columns actually fed to the backend.

**References:** Peng, Long & Ding (2005), *Feature selection based on mutual information: mRMR*, IEEE TPAMI 27(8).
