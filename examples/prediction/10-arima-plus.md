# ARIMA++ : smarter automatic order selection

`TsArima` offers two automatic order selectors (used when `p`/`d`/`q` are left `None`):

* `auto="aicc"` (default) — a compact AICc search, approximating `forecast::auto.arima`.
* `auto="cv"` — **ARIMA++**, which fixes two known weaknesses of `auto.arima`:
  1. **ACF/PACF-driven candidates** instead of a fixed cap of 5 — the significant
     autocorrelation lags enter the search, so a **seasonal spike (e.g. lag 12)** is
     tested when the data calls for it.
  2. **Selection by rolling-origin cross-validation matched to the forecast horizon**
     (out-of-sample sMAPE) instead of the in-sample AICc proxy — Hyndman himself notes
     "AIC is not a measure of forecast accuracy".

All native, no new dependency.

**Reference:** Box, Jenkins, Reinsel & Ljung (2015); Hyndman & Khandakar (2008); Tashman
(2000), *Out-of-sample tests of forecasting accuracy*, Int. J. Forecasting.


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts
```

## A series with long (seasonal) autoregressive structure


```python
rng = np.random.default_rng(1)
n = 260
e = rng.normal(0, 1, n)
y = np.zeros(n)
for t in range(12, n):
    y[t] = 0.5 * y[t - 1] + 0.5 * y[t - 12] + e[t]   # strong lag-12 dependence
y = y[40:]
train, test = y[:-12], y[-12:]
```

## The default AICc auto caps the order low


```python
m_aicc = ts.TsArima(auto="aicc").fit(train)
m_aicc._order()
```




    (0, 1, 2)



## ARIMA++ recovers the long lag and forecasts better


```python
m_cv = ts.TsArima(auto="cv", horizon=12).fit(train)
m_cv._order()
```




    (11, 1, 3)




```python
def smape(a, p):
    a, p = np.asarray(a), np.asarray(p)
    return float(np.mean(2 * np.abs(a - p) / (np.abs(a) + np.abs(p) + 1e-9)))

pa = m_aicc.predict(test, steps_ahead=12)
pc = m_cv.predict(test, steps_ahead=12)
print("12-step sMAPE  aicc = %.4f   arima++ = %.4f" % (smape(test, pa), smape(test, pc)))
```

    12-step sMAPE  aicc = 0.1912   arima++ = 0.0873


> **What to observe.** On autoregressive/seasonal structure the ACF/PACF candidates reach
> past the cap of 5 and the horizon-matched CV picks the order that forecasts best, so
> ARIMA++ beats the AICc auto. On real competition data (NN5) it is ~14% better on
> average — see `develop/parity/autoarima_benchmark.py`. On short or structureless series
> it falls back gracefully to the compact AICc search.
