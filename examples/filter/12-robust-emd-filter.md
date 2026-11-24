# Filter -- Robust EMD (`TsFilRemd`)

**About the technique**

* `TsFilRemd` removes the high-frequency IMFs chosen by a roughness/curvature cutoff. Native.
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
out = ts.TsFilRemd().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [0.    0.247 0.479 0.682 0.841 0.949]
    length: 41


> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Wu, Z. & Huang, N. E. (2009). Ensemble EMD. Advances in Adaptive Data Analysis, 1(1), 1-41.
