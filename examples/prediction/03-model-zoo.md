# Model Zoo: Deterministic, DARIMA, WARMA

Three more univariate predictors complete the model family, all with no new
dependencies:

* **Deterministic** (`TsPeriodic`, `TsPersist`): forecast by a *rule* — repeat a
  learned cycle, or repeat the last value. Fully native; matches R exactly.
* **`TsDarima`**: a linear model over the lag windows with the integration
  (differencing) *delegated to the preprocessor* (e.g. `TsNormDiff`).
* **`TsWarma`**: a linear model with local per-window normalization in `steps`
  (0: none; 1: subtract the window mean; 2: also divide by its sd). `steps=None`
  auto-selects via a stationarity (KPSS) test.

Both `TsDarima` and `TsWarma` reproduce R's `lm` on rank-deficient designs (they
drop aliased columns rather than returning a min-norm fit).

**Reference:** Box, Jenkins, Reinsel & Ljung (2015); Salles et al. (2017), *A
framework for benchmarking ML methods using linear models for univariate time
series prediction*, IJCNN.


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts
```

## Deterministic predictors


```python
# periodic: learn a 7-step cycle and repeat it
ts.TsPeriodic(7).fit([4, 5, 6, 7, 1, 2, 3]).predict(steps_ahead=5)
```




    array([4., 5., 6., 7., 1.])




```python
# persistence: repeat the most recent value
ts.TsPersist().fit([10, 11, 11, 12]).predict(steps_ahead=3)
```




    array([12., 12., 12.])



## DARIMA and WARMA


```python
w = ts.ts_data(ts.datasets.tsd()["y"], sw=8)
samp = ts.ts_sample(w, test_size=5)
io_train, io_test = ts.ts_projection(samp.train), ts.ts_projection(samp.test)
output = io_test.output.to_numpy()

# DARIMA: differencing delegated to TsNormDiff
darima = ts.TsDarima(ts.TsNormDiff(), input_size=5).fit(io_train.input, io_train.output)
darima.evaluate(output, darima.predict(io_test.input.iloc[[0]], steps_ahead=5))["metrics"]
```




                mse         smape   R2
    0  9.195314e-30  8.601082e-15  1.0




```python
# WARMA: local per-window normalization; steps chosen automatically
warma = ts.TsWarma(input_size=5, steps=None).fit(io_train.input, io_train.output)
print("selected steps:", warma.steps)
warma.evaluate(output, warma.predict(io_test.input.iloc[[0]], steps_ahead=5))["metrics"]
```

    selected steps: 0
    




                mse         smape   R2
    0  1.822176e-29  7.787043e-15  1.0



> **Tip:** linear window models need a well-conditioned design. On a series with an
> exact low-order linear recurrence (e.g. a pure sine) and `input_size` above its
> effective rank, the lag columns are collinear and the fit is rank-deficient — use
> `TsNormDiff` or a smaller `input_size` there.
