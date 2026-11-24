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

# Integrated tuning (`TsIntegtune`)

**About the technique**

* `TsIntegtune` extends `TsTune` by also searching over a list of **preprocessors** (and
  augmenters) -- co-optimizing preprocessing and modeling together. This joint search is
  the idea at the heart of TSPredIT.

**Didactic goal:** choose the normalization/augmentation *and* the model jointly, since the
best model depends on the preprocessing.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

w = ts.ts_data(ts.datasets.tsd()["y"], 10)
samp = ts.ts_sample(w, test_size=5)
tr, te = ts.ts_projection(samp.train), ts.ts_projection(samp.test)
```

```python
itune = ts.TsIntegtune(
    input_size=[3, 4, 5],
    base_model=ts.TsElm(),
    folds=5,
    ranges={"nhid": [1, 2, 3, 4, 5], "actfun": ["purelin"]},
    preprocess=[ts.TsNormGminmax(), ts.TsNormSwminmax()],
    augment=[ts.TsAugNone(), ts.TsAugFlip()],
    seed=0,
)
best = itune.fit(tr.input, tr.output)
print("selected:", itune.best_params)
best.evaluate(te.output.to_numpy().ravel(), best.predict(te.input.iloc[[0]], steps_ahead=5))["metrics"]
```

> **What to observe.** The `preprocess`/`augment` indices in `best_params` point into the
> lists you passed; the model and its preprocessing are selected together.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
