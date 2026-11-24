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

# Lag mapping -- Mutual-information lags (`TsLagmap(method="mi")`)

**About the technique**

* `TsLagmap(method="mi")` selects the lags with the highest mutual information with the target (captures non-linear dependence).

**Didactic goal:** select lags by a non-linear relevance measure.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

io = ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))
```

```python
lm = ts.TsLagmap(method="mi", seasonality=4).fit(io.input, io.output, input_size=4)
print("selected lags:", list(lm.lags))
print("columns:", lm.columns)
```

> **What to observe.** The strategy returns `input_size` lag positions (`lag 1` = most recent); these are the columns actually fed to the backend.

**References:** Cover & Thomas (2006), *Elements of Information Theory*, Wiley (mutual information).
