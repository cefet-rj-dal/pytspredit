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

# Sliding-window min-max normalization (`TsNormSwminmax`)

**About the technique**

* `TsNormSwminmax` rescales **each window by its own** min and max, so every row is
  mapped into `[0, 1]` independently. This removes slow level/trend drift between
  windows.

**Didactic goal:** normalize locally so the model sees the within-window shape rather
than the absolute level.

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
norm = ts.TsNormSwminmax().fit(w)
t = norm.transform(w)
ts.ts_head(t)
```

```python
ts.ts_head(norm.inverse_transform(t))
```

> **What to observe.** Each row is scaled by its own range (per-window `[0, 1]`); the
> inverse restores the original windows using the stored per-row min/max.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
