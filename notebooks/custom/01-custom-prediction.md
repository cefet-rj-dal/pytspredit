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

# Custom prediction model

**About the technique**

* Extend `TsRegSW` to plug any backend into the sliding-window pipeline: implement
  `_do_fit(self, x, y)` and `_do_predict(self, x)`; preprocessing, lag mapping and
  multi-step forecasting are inherited.

**Didactic goal:** wrap your own learner with the uniform `fit`/`predict`/`evaluate` API.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LinearRegression
import pytspredit as ts

class MyLinearReg(ts.TsRegSW):
    def _do_fit(self, x, y):
        self.model = LinearRegression().fit(np.asarray(x, float), np.asarray(y, float).ravel())
        return self
    def _do_predict(self, x):
        return self.model.predict(np.asarray(x, float))
```

```python
w = ts.ts_data(ts.datasets.tsd()["y"], 6)
samp = ts.ts_sample(w, test_size=5)
tr, te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)

model = MyLinearReg(ts.TsNormGminmax(), input_size=5).fit(tr.input, tr.output)
pred = model.predict(te.input.iloc[[0]], steps_ahead=5)
model.evaluate(te.output.to_numpy().ravel(), pred)["metrics"]
```

> **What to observe.** Only the two hooks are custom; normalization, lag selection and
> rolling multi-step prediction come for free from `TsRegSW`.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160 (Experiment Line extensibility).
