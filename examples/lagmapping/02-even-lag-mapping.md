# Lag mapping -- Even lags (`TsLagmap(method="even")`)

**About the technique**

* `TsLagmap(method="even")` spreads the selected lags evenly across the window, sampling the near and far past uniformly.

**Didactic goal:** cover the whole window with few, evenly spaced lags.


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





![png](fig/02-even-lag-mapping/output_3_0.png)






![png](fig/02-even-lag-mapping/output_3_1.png)




```python
lm = ts.TsLagmap(method="even", seasonality=4).fit(io.input, io.output, input_size=4)
print("selected lags:", list(lm.lags))
print("columns:", lm.columns)
```

    selected lags: [np.int64(9), np.int64(6), np.int64(4), np.int64(1)]
    columns: ['t9', 't6', 't4', 't1']


> **What to observe.** The strategy returns `input_size` lag positions (`lag 1` = most recent); these are the columns actually fed to the backend.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis: Forecasting and Control*, Wiley.
