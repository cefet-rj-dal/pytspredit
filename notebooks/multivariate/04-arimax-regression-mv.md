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

# ARIMA with external regressors (`TsArimax`)

**About the technique**

* `TsArimax` is the singular-multivariate counterpart of `TsArima`: ARIMA on the target
  with the auxiliaries as `xreg`.
* Every multivariate `predict` returns the target `y` forecast (the full system is available too).

**Didactic goal:** forecast the target with ARIMA plus exogenous regressors.

```python
import warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytspredit as ts

# canonical multivariate data from tsd: y = sine, x1 = y shifted one step,
# x2 = 3-point moving average of y (as in the R examples).
y = ts.datasets.tsd()["y"].to_numpy()
x1 = np.concatenate([y[1:], y[-1:]])
x2 = pd.Series(y).rolling(3).mean().to_numpy()
m = np.isnan(x2); x2[m] = y[m]
mv = ts.ts_data_mv(pd.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y")
samp = ts.ts_sample(mv, test_size=5)
```

```python
ax = ts.TsArimax(p=2, d=0, q=0).fit(samp.train)
np.asarray(ax.predict(samp.test, steps_ahead=5)).round(4)
```

> **What to observe.** Explicit `(p,d,q)` keeps the example fast; auxiliaries enter through `xreg`.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis*, Wiley; Salles et al. (2023), *TSPredIT*, LNCS 14160.
