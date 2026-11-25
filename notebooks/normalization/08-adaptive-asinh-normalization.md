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

# Adaptive normalization -- asinh (`TsNormAn`, `operation="asinh"`)

**About the technique**

* The *asinh* operator of `TsNormAn` centers and scales each window and then applies the
  inverse hyperbolic sine, which compresses large magnitudes smoothly (like a signed log
  that is well defined at zero).

**Didactic goal:** damp heavy tails / spikes while keeping the transform invertible.

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
norm = ts.TsNormAn(operation="asinh").fit(w)
t = norm.transform(w)
ts.ts_head(t)
```

```python
ts.ts_head(norm.inverse_transform(t))
```

Compare the original and normalized target (`t0`) columns (solid = original, dashed = normalized):

```python
ts.plot_ts_pred(y=w.iloc[:, -1].to_numpy(), yadj=t.iloc[:, -1].to_numpy())
```

> **What to observe.** Large values are compressed by `asinh`; `inverse_transform`
> (via `sinh`) recovers the original windows.

**References:** Ogasawara et al. (2010), *Adaptive Normalization*, IJCNN.
