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

# Custom augmentation

**About the technique**

* An augmenter is a `DalTransform` over the window matrix that returns the original
  windows stacked with augmented ones. Implement `fit` and `transform`.

**Didactic goal:** add your own data-augmentation strategy to the training windows.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pydaltoolbox import DalTransform
import pytspredit as ts

class MyNegateAug(DalTransform):
    """Augment with mean-reflected copies of each window."""
    def fit(self, data=None, **kwargs):
        return self
    def transform(self, data, **kwargs):
        df = pd.DataFrame(data)
        reflected = 2 * df.mean(axis=1).to_numpy()[:, None] - df.to_numpy()
        aug = pd.DataFrame(reflected, columns=df.columns)
        return pd.concat([df, aug], ignore_index=True)
```

```python
windows = ts.ts_data(ts.datasets.tsd()["y"], 10)
out = MyNegateAug().fit(windows).transform(windows)
print("original:", windows.shape, "-> augmented:", out.shape)
```

> **What to observe.** The output stacks original + augmented windows, matching the
> built-in `TsAug*` contract and the `TsIntegtune` augmentation slot.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160 (Experiment Line extensibility).
