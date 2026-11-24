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

# Adaptive normalization -- soft divide (`TsNormAn`, `operation="softdivide"`)

**About the technique**

* The *soft divide* operator of `TsNormAn` divides by a stabilized average (a small
  epsilon / soft floor prevents blow-ups), combining the scale-invariance of division
  with robustness near zero.

**Didactic goal:** keep the division behavior while taming the explosions that a raw
divide causes when the local average is tiny.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts

y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
```

```python
norm = ts.TsNormAn(operation="softdivide").fit(w)
t = norm.transform(w)
ts.ts_head(t)
```

```python
ts.ts_head(norm.inverse_transform(t))
```

> **What to observe.** Values are divided by a stabilized average, so the transform stays
> finite even where a plain divide would explode.

**References:** Ogasawara et al. (2010), *Adaptive Normalization*, IJCNN.
