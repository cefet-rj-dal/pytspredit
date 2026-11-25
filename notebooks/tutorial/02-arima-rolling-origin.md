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

# ARIMA rolling-origin evaluation

**Didactic goal:** Evaluate an ARIMA by rolling the forecast origin across the test set.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

rser = ts.ts_data(ts.datasets.tsd()["y"], 1)
samp = ts.ts_sample(rser, test_size=5)
```

```python
model = ts.TsArima(p=2, d=0, q=0).fit(samp.train)
pred = model.predict(samp.test, steps_ahead=1)
print("rolling-origin sMAPE:", round(float(ts.smape_ts(samp.test.to_numpy().ravel(), pred)), 4))
```

Plot the forecast against the observed series:

```python
import numpy as np
adjust = np.asarray(model.predict(samp.train)).ravel()          # in-sample fit
yvalues = np.concatenate([samp.train.to_numpy().ravel(), samp.test.to_numpy().ravel()])
ts.plot_ts_pred(y=yvalues, yadj=adjust, ypred=pred)
```

> **What to observe.** Rolling-origin is the honest way to assess forecast accuracy over time.

**References:** Tashman (2000), *Out-of-sample tests of forecasting accuracy*, Int. J. Forecasting 16(4).
