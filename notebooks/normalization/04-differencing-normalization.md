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

```python
norm = ts.TsNormDiff().fit(w)
t = norm.transform(w)
ts.ts_head(t)
```

```python
ts.ts_head(norm.inverse_transform(t))
```

> **What to observe.** The transformed windows hold differences; `inverse_transform`
> integrates them back to the original level.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis:
Forecasting and Control*, Wiley.
