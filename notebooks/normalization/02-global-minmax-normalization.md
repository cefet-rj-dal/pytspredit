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

# Global min-max normalization (`TsNormGminmax`)

**About the technique**

* `TsNormGminmax` rescales every window with a **single global** min and max learned
  from the training data, mapping values into `[0, 1]`.

**Didactic goal:** put all windows on one common scale (good for distance/gradient
models) while keeping the transform invertible.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
```

```python
norm = ts.TsNormGminmax().fit(w)
t = norm.transform(w)
ts.ts_head(t)
```

The transform is invertible (recovers the original windows).

```python
ts.ts_head(norm.inverse_transform(t))
```

> **What to observe.** All columns share one global `[0, 1]` scale; `inverse_transform`
> returns the original values.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
