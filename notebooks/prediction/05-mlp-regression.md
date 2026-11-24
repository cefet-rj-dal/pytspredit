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

# MLP regression (`TsMlp`)

**About the technique**

* `TsMlp` wraps scikit-learn MLPRegressor (feed-forward neural network).
* A sliding-window regressor: preprocessing + lag map + backend, trained on the projected `(input, output)`.

**Didactic goal:** fit, forecast 5 steps ahead, and evaluate (MSE / sMAPE / R2).

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

w = ts.ts_data(ts.datasets.tsd()["y"], 6)
samp = ts.ts_sample(w, test_size=5)
tr, te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)
```

```python
model = ts.TsMlp(ts.TsNormGminmax(), input_size=5).fit(tr.input, tr.output)
pred = model.predict(te.input.iloc[[0]], steps_ahead=5)
model.evaluate(te.output.to_numpy().ravel(), pred)["metrics"]
```

Plot the 5-step forecast against the observed series (solid = observed, dashed = forecast):

```python
y_all = ts.datasets.tsd()["y"].to_numpy()
ts.plot_ts_pred(y=y_all, yadj=y_all[:-5], ypred=pred)
```

> **What to observe.** `fit` learns on the training windows; `predict(steps_ahead=5)` rolls forward from the first test window; `evaluate` reports MSE/sMAPE/R2.

**References:** Rumelhart, Hinton & Williams (1986). Learning representations by back-propagating errors. Nature, 323, 533-536.
