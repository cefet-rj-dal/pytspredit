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

# Model tuning (`TsTune`)

**About the technique**

* `TsTune` searches candidate `input_size` values and model hyperparameter ranges by
  cross-validated grid search (minimizing CV MSE), and returns the best model refitted on
  all the training data.

**Didactic goal:** pick the input size and hyperparameters without manual trial and error.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

w = ts.ts_data(ts.datasets.tsd()["y"], 10)
samp = ts.ts_sample(w, test_size=5)
tr, te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)
```

```python
tune = ts.TsTune(
    input_size=[3, 4, 5],
    base_model=ts.TsElm(ts.TsNormGminmax()),
    folds=5,
    ranges={"nhid": [1, 2, 3, 4, 5], "actfun": ["purelin"]},
    seed=0,
)
best = tune.fit(tr.input, tr.output)
print("selected:", tune.best_params)
best.evaluate(te.output.to_numpy().ravel(), best.predict(te.input.iloc[[0]], steps_ahead=5))["metrics"]
```

> **What to observe.** `best_params` reports the chosen `input_size` and hyperparameters;
> `fit` returns the best model already refitted on all training data.

**References:** Salles et al. (2023), *TSPredIT: Integrated Tuning of Data Preprocessing
and Time Series Prediction Models*, LNCS 14160.
