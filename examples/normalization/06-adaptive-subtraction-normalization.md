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


```python
norm = ts.TsNormAn(operation="subtract").fit(w)
t = norm.transform(w)
ts.ts_head(t)
```




             t4        t3        t2        t1        t0
    0  0.034405  0.290397  0.530473  0.739706  0.905086
    1  0.094011  0.334087  0.543320  0.708701  0.819946
    2  0.178861  0.388094  0.553475  0.664721  0.714915
    3  0.283679  0.449060  0.560305  0.610500  0.596522
    4  0.401947  0.513193  0.563388  0.549410  0.472128
    5  0.526313  0.576507  0.562529  0.485248  0.349468




```python
ts.ts_head(norm.inverse_transform(t))
```




             t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471
    1  0.247404  0.479426  0.681639  0.841471  0.948985
    2  0.479426  0.681639  0.841471  0.948985  0.997495
    3  0.681639  0.841471  0.948985  0.997495  0.983986
    4  0.841471  0.948985  0.997495  0.983986  0.909297
    5  0.948985  0.997495  0.983986  0.909297  0.778073



> **What to observe.** Windows are mean-centered (no division), so the operator is robust
> when the local average is close to zero.

**References:** Ogasawara et al. (2010), *Adaptive Normalization*, IJCNN.
