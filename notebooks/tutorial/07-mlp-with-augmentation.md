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

# MLP with augmentation (integrated)

**Didactic goal:** Use integrated tuning to search augmenters alongside the MLP.

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
itune = ts.TsIntegtune(input_size=[4, 5], base_model=ts.TsElm(), folds=5,
    ranges={"nhid": [2, 3, 4], "actfun": ["purelin"]},
    preprocess=[ts.TsNormGminmax()],
    augment=[ts.TsAugNone(), ts.TsAugFlip()], seed=0)
best = itune.fit(tr.input, tr.output)
print("selected:", itune.best_params)
best.evaluate(out, best.predict(te.input.iloc[[0]], steps_ahead=5))["metrics"]
```

> **What to observe.** Augmentation enters the search as another slot to co-optimize.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
