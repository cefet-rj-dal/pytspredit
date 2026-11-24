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





![png](fig/09-adaptive-normalization-comparison/output_3_0.png)






![png](fig/09-adaptive-normalization-comparison/output_3_1.png)




```python
# first transformed window (t4..t0) under each operator
row = []
for op in ["divide", "subtract", "softdivide", "asinh"]:
    t = ts.TsNormAn(operation=op).fit(w).transform(w)
    row.append((op, np.round(t.iloc[0].to_numpy(), 3)))
for op, vals in row:
    print(f"{op:10}", vals)
```

    divide     [0.234 0.39  0.537 0.665 0.766]
    subtract   [0.034 0.29  0.53  0.74  0.905]
    softdivide [0.178 0.354 0.519 0.663 0.776]
    asinh      [0.171 0.363 0.518 0.63  0.706]


> **What to observe.** `divide`/`softdivide` are scale-relative, `subtract` is additive
> (safe near zero), and `asinh` compresses magnitude. All are invertible; the default is
> `divide`.

**References:** Ogasawara et al. (2010), *Adaptive Normalization*, IJCNN.
