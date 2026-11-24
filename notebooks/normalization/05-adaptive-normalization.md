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

# Adaptive normalization (`TsNormAn`)

**About the technique**

* `TsNormAn` (Adaptive Normalization) centers each window by a local average and
  rescales it by a local scale, through one of four operators. The default,
  `operation="divide"`, divides by the (outlier-robust) window average.
* An outlier step (`OutliersBoxplot` by default) is applied before estimating the
  range, which is decisive for the division operator.

**Didactic goal:** normalize non-stationary series per window with a robust, invertible
transform.

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
norm = ts.TsNormAn().fit(w)          # operation="divide" (default)
t = norm.transform(w)
ts.ts_head(t)
```

```python
ts.ts_head(norm.inverse_transform(t))
```

> **What to observe.** Each window is normalized by its own robust average; the other
> operators (subtract / softdivide / asinh) are shown in the next examples.

**References:** Ogasawara et al. (2010), *Adaptive Normalization: A novel data
normalization approach for non-stationary time series*, IJCNN.
