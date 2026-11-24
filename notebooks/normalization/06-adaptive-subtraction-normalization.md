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

# Adaptive normalization -- subtraction (`TsNormAn`, `operation="subtract"`)

**About the technique**

* The *subtraction* operator of `TsNormAn` centers each window by subtracting its local
  average (optionally scaled), without dividing -- safe when values can be near zero.

**Didactic goal:** center per window additively, avoiding the instability of dividing by
a small average.

```python
import warnings
warnings.filterwarnings("ignore")

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
norm = ts.TsNormAn(operation="subtract").fit(w)
t = norm.transform(w)
ts.ts_head(t)
```

```python
ts.ts_head(norm.inverse_transform(t))
```

> **What to observe.** Windows are mean-centered (no division), so the operator is robust
> when the local average is close to zero.

**References:** Ogasawara et al. (2010), *Adaptive Normalization*, IJCNN.
