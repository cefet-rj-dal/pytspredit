# Filter -- Exponential moving average (`TsFilEma`)

**About the technique**

* `TsFilEma` smooths with an exponentially weighted moving average (recent points weigh more).
* A filter is a `DalTransform` taking a 1-D series and returning a filtered series of the same length.

**Didactic goal:** see how the filter cleans the raw series before modeling.


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

y = ts.datasets.tsd()["y"].to_numpy()
```


```python
out = ts.TsFilEma(3).fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [  nan   nan 0.345 0.562 0.744 0.88 ]
    length: 41


> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*.
