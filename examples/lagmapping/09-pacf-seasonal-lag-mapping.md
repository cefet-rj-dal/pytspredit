# Lag mapping -- PACF seasonal lags (`TsLagmap(method="pacf_seasonal")`)

**About the technique**

* `TsLagmap(method="pacf_seasonal")` ranks the seasonal lags by partial autocorrelation and keeps the strongest.

**Didactic goal:** combine seasonality with PACF (direct-effect) evidence.


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





![png](fig/09-pacf-seasonal-lag-mapping/output_3_0.png)






![png](fig/09-pacf-seasonal-lag-mapping/output_3_1.png)




```python
lm = ts.TsLagmap(method="pacf_seasonal", seasonality=4).fit(io.input, io.output, input_size=4)
print("selected lags:", list(lm.lags))
print("columns:", lm.columns)
```

    selected lags: [np.int64(8), np.int64(4), np.int64(2), np.int64(1)]
    columns: ['t8', 't4', 't2', 't1']


> **What to observe.** The strategy returns `input_size` lag positions (`lag 1` = most recent); these are the columns actually fed to the backend.

**References:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis: Forecasting and Control*, Wiley.
