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


```python
y = ts.datasets.tsd()["y"]
w = ts.ts_data(y, 5)
proj = ts.ts_projection(w)
```

The input holds the past lags used as predictors.


```python
ts.ts_head(proj.input)
```




             t4        t3        t2        t1
    0  0.000000  0.247404  0.479426  0.681639
    1  0.247404  0.479426  0.681639  0.841471
    2  0.479426  0.681639  0.841471  0.948985
    3  0.681639  0.841471  0.948985  0.997495
    4  0.841471  0.948985  0.997495  0.983986
    5  0.948985  0.997495  0.983986  0.909297



The output holds the value to predict for each row.


```python
ts.ts_head(proj.output)
```




             t0
    0  0.841471
    1  0.948985
    2  0.997495
    3  0.983986
    4  0.909297
    5  0.778073



> **What to observe.** `input` has `sw - 1` columns (`t4 ... t1`) and `output` is the
> single column `t0`; together they are the `(X, y)` handed to `fit`.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
