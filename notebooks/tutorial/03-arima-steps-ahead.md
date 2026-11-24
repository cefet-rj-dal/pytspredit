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

# ARIMA multi-step forecast

**Didactic goal:** Forecast several steps ahead directly from the fitted ARIMA.

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
pred = model.predict(samp.test.iloc[[0]], steps_ahead=5)
model.evaluate(samp.test.to_numpy().ravel(), pred)["metrics"]
```

> **What to observe.** A single multi-step forecast from the last training point.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis*, Wiley.
