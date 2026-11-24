# ARIMA one-step-ahead

**Didactic goal:** Fit an ARIMA and forecast one step ahead.


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
pred = model.predict(samp.test, steps_ahead=1)   # rolling 1-step
model.evaluate(samp.test.to_numpy().ravel(), pred)["metrics"]
```




                mse     smape   R2
    0  2.725253e-09  0.000137  1.0



> **What to observe.** One-step rolling forecast refits with the known order as actuals arrive.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis*, Wiley.
