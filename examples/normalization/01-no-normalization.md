# No normalization (`TsNormNone`)

**About the technique**

* `TsNormNone` is the identity preprocessor: it leaves the windows unchanged. It is the
  baseline against which the other normalizations are compared.

**Didactic goal:** see the untouched windows that the other normalizers transform.


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





![png](fig/01-no-normalization/output_3_0.png)






![png](fig/01-no-normalization/output_3_1.png)




```python
norm = ts.TsNormNone().fit(w)
ts.ts_head(norm.transform(w))
```




             t4        t3        t2        t1        t0
    0  0.000000  0.247404  0.479426  0.681639  0.841471
    1  0.247404  0.479426  0.681639  0.841471  0.948985
    2  0.479426  0.681639  0.841471  0.948985  0.997495
    3  0.681639  0.841471  0.948985  0.997495  0.983986
    4  0.841471  0.948985  0.997495  0.983986  0.909297
    5  0.948985  0.997495  0.983986  0.909297  0.778073



> **What to observe.** The transformed windows equal the input; `TsNormNone` is a
> pass-through used as the default/control.

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
