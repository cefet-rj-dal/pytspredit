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

# MLP with a filter

**Didactic goal:** Clean the series with a filter before windowing and modeling.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

y = ts.datasets.tsd()["y"].to_numpy()
y = ts.TsFilMa(3).fit(y).transform(y)
y = y[~np.isnan(y)]
w = ts.ts_data(y, 6)
samp = ts.ts_sample(w, test_size=5)
tr, te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)
out = te.output.to_numpy().ravel()
```

```python
model = ts.TsMlp(ts.TsNormGminmax(), input_size=5).fit(tr.input, tr.output)
model.evaluate(out, model.predict(te.input.iloc[[0]], steps_ahead=5))["metrics"]
```

> **What to observe.** Filtering upstream can denoise the signal the model learns from.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
