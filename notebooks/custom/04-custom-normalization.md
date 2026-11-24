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

# Custom normalization

**About the technique**

* A normalization is a `DalTransform` with `fit`, `transform` and `inverse_transform`
  (so forecasts can be mapped back to the original scale).

**Didactic goal:** write your own invertible normalization.

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pydaltoolbox import DalTransform
import pytspredit as ts

class MyCenterNorm(DalTransform):
    """Center the data by subtracting the global mean (invertible)."""
    def fit(self, data=None, **kwargs):
        self.mean_ = float(np.asarray(data, float).mean())
        return self
    def transform(self, data, **kwargs):
        return data - self.mean_
    def inverse_transform(self, data, **kwargs):
        return data + self.mean_
```

```python
w = ts.ts_data(ts.datasets.tsd()["y"], 5)
norm = MyCenterNorm().fit(w)
t = norm.transform(w)
ts.ts_head(norm.inverse_transform(t))   # recovers the original windows
```

> **What to observe.** Providing `inverse_transform` lets the pipeline return predictions
> on the original scale, exactly like the built-in `TsNorm*`.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160 (Experiment Line extensibility).
