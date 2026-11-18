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

# ARIMA

`TsArima` is a univariate AutoRegressive Integrated Moving Average model. Unlike the
sliding-window regressors, it works directly on the raw series (a `ts_data` with
`sw = 1`) — no normalization or lag map. Supply `(p, d, q)` for a fixed order, or leave
them out for automatic selection.

It is built on `statsmodels` (the R package uses `forecast`); forecasts are close but
not bit-identical across the two maximum-likelihood backends.

**Reference:** Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis:
Forecasting and Control*; Hyndman & Khandakar (2008), *Automatic time series
forecasting*, JSS.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

series = ts.ts_data(ts.datasets.tsd()["y"], sw=1)   # raw series
samp = ts.ts_sample(series, test_size=5)
```

## Fixed order ARIMA(5, 0, 0)

```python
model = ts.TsArima(p=5, d=0, q=0).fit(samp.train)
forecast = model.predict(samp.test.iloc[[0]], steps_ahead=5)
model.evaluate(samp.test.to_numpy().ravel(), forecast)["metrics"]
```

## Automatic order selection

With no orders, `TsArima` searches `(p, d, q)` by AICc:

```python
auto = ts.TsArima().fit(samp.train)
print("selected (p, d, q):", (auto.p, auto.d, auto.q))
auto.evaluate(samp.test.to_numpy().ravel(),
              auto.predict(samp.test.iloc[[0]], steps_ahead=5))["metrics"]
```
