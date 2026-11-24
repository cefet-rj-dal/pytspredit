# Vector autoregression (`TsVar`)

**About the technique**

* `TsVar` learns the whole multivariate system jointly by OLS over lagged observations,
  choosing the lag order by AICc (native implementation).
* Every multivariate `predict` returns the target `y` forecast (the full system is available too).

**Didactic goal:** model all variables as one system and forecast them together.


```python
import warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")

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
var = ts.TsVar(p_max=3).fit(samp.train)
np.asarray(var.predict(steps_ahead=5)).round(4)
```




    array([ 0.4121,  0.1739, -0.0752, -0.3195, -0.544 ])



> **What to observe.** `TsVar` forecasts the full system; by default `predict` returns the target `y`.

**References:** Lütkepohl, H. (2005), *New Introduction to Multiple Time Series Analysis*, Springer; Salles et al. (2023), *TSPredIT*, LNCS 14160.
