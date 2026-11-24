# Lag mapping -- Seasonal lags (`TsLagmap(method="seasonal")`)

**About the technique**

* `TsLagmap(method="seasonal")` selects lags at multiples of the seasonal period.

**Didactic goal:** feed the model the seasonal lags explicitly.


```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

io = ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))
```


```python
lm = ts.TsLagmap(method="seasonal", seasonality=4).fit(io.input, io.output, input_size=4)
print("selected lags:", list(lm.lags))
print("columns:", lm.columns)
```

    selected lags: [np.int64(8), np.int64(4), np.int64(2), np.int64(1)]
    columns: ['t8', 't4', 't2', 't1']


> **What to observe.** The strategy returns `input_size` lag positions (`lag 1` = most recent); these are the columns actually fed to the backend.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis: Forecasting and Control*, Wiley.
