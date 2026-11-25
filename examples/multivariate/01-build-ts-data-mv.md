# Build multivariate `ts_data` (`ts_data_mv`)

**About the technique**

* `ts_data_mv()` builds the multivariate representation: a target `y` plus aligned
  auxiliary variables, carrying the metadata (target/auxiliaries/representation).
* Every multivariate `predict` returns the target `y` forecast (the full system is available too).

**Didactic goal:** understand the multivariate container the MV models consume.


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytspredit as ts

# canonical multivariate data from tsd: y = sine, x1 = y shifted one step,
# x2 = 3-point moving average of y (as in the R examples).
y = ts.datasets.tsd()["y"].to_numpy()
x1 = np.concatenate([y[1:], y[-1:]])
x2 = pd.Series(y).rolling(3).mean().to_numpy()
m = np.isnan(x2); x2[m] = y[m]
mv = ts.ts_data_mv(pd.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y")
samp = ts.ts_sample(mv, test_size=5)
```


```python
ts.ts_head(mv, 3)
print("target:", mv.mv_y, "| auxiliaries:", mv.mv_x)
```

    target: y | auxiliaries: ['x1', 'x2']


> **What to observe.** The container carries its metadata through sampling (`ts_sample`).

**References:** Salles et al. (2023), *TSPredIT*, LNCS 14160.
