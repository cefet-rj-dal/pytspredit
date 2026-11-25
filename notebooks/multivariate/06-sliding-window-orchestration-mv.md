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

# Sliding-window orchestration (`TsRegswMv`, `ts_mv_spec`)

**About the technique**

* `TsRegswMv` coordinates one univariate sliding-window learner per variable
  (`ts_mv_spec`) over a shared `window_size`, forecasting the target recursively.
* Every multivariate `predict` returns the target `y` forecast (the full system is available too).

**Didactic goal:** combine per-variable sliding-window models into a target-centered multivariate forecaster.

```python
import warnings
warnings.filterwarnings("ignore")

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
orch = ts.TsRegswMv(
    model_y=ts.ts_mv_spec(ts.TsKnn(ts.TsNormGminmax(), k=3), variables=["y", "x1", "x2"]),
    models_x={"x1": ts.ts_mv_spec(ts.TsPeriodic(7)),
              "x2": ts.ts_mv_spec(ts.TsPeriodic(7))},
    window_size=5).fit(samp.train)
orch.predict(steps_ahead=5).system
```

Plot the multivariate forecast for the target variable:

```python
pred = orch.predict(steps_ahead=5)
ts.plot_ts_pred_mv(samp.train, samp.test, pred, variable="y")
```

> **What to observe.** `ts_mv_spec` wires each variable to its own model; `predict().system` holds the full multivariate forecast.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
