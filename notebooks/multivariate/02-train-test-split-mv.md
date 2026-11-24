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

# Train/test split of multivariate data

**About the technique**

* `ts_sample()` splits the multivariate data in time order, preserving the MV metadata
  on both train and test parts.
* Every multivariate `predict` returns the target `y` forecast (the full system is available too).

**Didactic goal:** produce a temporally ordered multivariate holdout.

```python
import warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytspredit as ts

# canonical multivariate data from tsd: y = sine, x1 = y shifted one step,
# x2 = 3-point moving average of y (as in the R examples).
y = ts.datasets.tsd()["y"].to_numpy()
x1 = np.concatenate([y[1:], y[-1:]])
x2 = pd.Series(y).rolling(3).mean().to_numpy()
m = np.isnan(x2); x2[m] = y[m]
mv = ts.ts_data_mv(pd.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y")
samp = ts.ts_sample(mv, test_size=5)
```

```python
print("train:", samp.train.shape, "| test:", samp.test.shape)
samp.train.mv_representation
```

> **What to observe.** The split is chronological and the metadata propagates to the parts.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
