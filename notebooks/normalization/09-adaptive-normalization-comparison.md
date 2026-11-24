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

# Adaptive normalization -- operator comparison

**About the technique**

* The four `TsNormAn` operators (`divide`, `subtract`, `softdivide`, `asinh`) produce
  different normalized windows; this example contrasts them on the same data.

**Didactic goal:** build intuition for which operator to pick (scale-invariance vs.
robustness near zero vs. tail compression).

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
```

Visualize the series used in this example:

```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```


```python
# first transformed window (t4..t0) under each operator
row = []
for op in ["divide", "subtract", "softdivide", "asinh"]:
    t = ts.TsNormAn(operation=op).fit(w).transform(w)
    row.append((op, np.round(t.iloc[0].to_numpy(), 3)))
for op, vals in row:
    print(f"{op:10}", vals)
```

> **What to observe.** `divide`/`softdivide` are scale-relative, `subtract` is additive
> (safe near zero), and `asinh` compresses magnitude. All are invertible; the default is
> `divide`.

**References:** Ogasawara et al. (2010), *Adaptive Normalization*, IJCNN.
