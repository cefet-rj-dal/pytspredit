# The Regression Contract (TsReg / TsRegSW)

Every time series regressor in `pytspredit` follows one contract:

* **`TsReg`** — the ancestor: the forecasting `evaluate` (MSE / sMAPE / R2) and a
  baseline `predict` that returns the last lag (`t0`).
* **`TsRegSW`** — the sliding-window base: it combines a **preprocessor** (a
  `TsNorm*`), a **lag map** (`TsLagmap`) and a **backend** model. A concrete regressor
  only implements `_do_fit` and `_do_predict`; `fit`, `predict` (one-step and
  multi-step), normalization round-tripping and evaluation come for free.


```python
import numpy as np
import pytspredit as ts

io = ts.ts_projection(ts.ts_data(ts.datasets.tsd()["y"], 10))
```

## Baseline and evaluation


```python
base = ts.TsReg()
pred = base.predict(io.input)           # naive: repeat the most recent lag
base.evaluate(io.output, pred)["metrics"]
```




            mse     smape        R2
    0  0.034366  0.551218  0.921847



## Extending TsRegSW

Building a new sliding-window regressor is small: wrap a backend in `_do_fit` /
`_do_predict`. Here is a minimal mean-of-lags model to show the mechanics (the real
ML and statistical regressors arrive next).


```python
class MeanRegressor(ts.TsRegSW):
    def _do_fit(self, x, y):
        return self

    def _do_predict(self, x):
        return np.asarray(x, dtype=float).mean(axis=1)


model = MeanRegressor(ts.TsNormGminmax(), input_size=4, input_map=ts.TsLagmap("recent"))
model.fit(io.input, io.output)
pred = model.predict(io.input)
model.evaluate(io.output, pred)["metrics"]
```




            mse     smape       R2
    0  0.193562  0.957267  0.55981



## Multi-step forecasting

With `steps_ahead > 1`, `predict` forecasts iteratively, rolling each prediction back
into the window:


```python
samp = ts.ts_sample(ts.ts_data(ts.datasets.tsd()["y"], 10), test_size=5)
io_tr, io_te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)

model = MeanRegressor(ts.TsNormGminmax(), input_size=4, input_map=ts.TsLagmap("recent"))
model.fit(io_tr.input, io_tr.output)

forecast = model.predict(io_te.input.iloc[[0]], steps_ahead=5)
model.evaluate(io_te.output.to_numpy(), forecast)["metrics"]
```




            mse     smape       R2
    0  0.837743  1.591786 -6.23563


