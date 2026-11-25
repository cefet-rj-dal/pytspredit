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

# MLP baseline

**Didactic goal:** Train a plain MLP forecaster as a baseline.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

w = ts.ts_data(ts.datasets.tsd()["y"], 6)
samp = ts.ts_sample(w, test_size=5)
tr, te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)
out = te.output.to_numpy().ravel()
```

```python
model = ts.TsMlp(ts.TsNormGminmax(), input_size=5).fit(tr.input, tr.output)
model.evaluate(out, model.predict(te.input.iloc[[0]], steps_ahead=5))["metrics"]
```

Plot the 5-step forecast against the observed series:

```python
import numpy as np
pred = model.predict(te.input.iloc[[0]], steps_ahead=5)
adjust = np.asarray(model.predict(tr.input)).ravel()            # in-sample fit
yvalues = np.concatenate([tr.output.to_numpy().ravel(), te.output.to_numpy().ravel()])
ts.plot_ts_pred(y=yvalues, yadj=adjust, ypred=pred)
```

> **What to observe.** A normalized MLP is a strong non-linear baseline.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
