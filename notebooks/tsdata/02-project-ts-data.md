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

# Project `ts_data` into input and output

**About the technique**

* `ts_projection()` splits the windowed `ts_data` into the model **input**
  (the lag columns `t{sw-1} ... t1`) and the **output** (the target `t0`).

**Didactic goal:** see the exact `(X, y)` a regressor is trained on.

```python
import warnings
warnings.filterwarnings("ignore")

import pytspredit as ts
```

Visualize the series used in this example:

```python
_d = ts.datasets.tsd()
ts.plot_ts(x=_d["x"], y=_d["y"])
```


```python
y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
proj = ts.ts_projection(w)
```

The input holds the past lags used as predictors.

```python
ts.ts_head(proj.input)
```

The output holds the value to predict for each row.

```python
ts.ts_head(proj.output)
```

> **What to observe.** `input` has `sw - 1` columns (`t4 ... t1`) and `output` is the
> single column `t0`; together they are the `(X, y)` handed to `fit`.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
