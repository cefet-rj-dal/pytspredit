# Filter -- Seasonal adjustment (STL) (`TsFilSeasAdj`)

**About the technique**

* `TsFilSeasAdj` removes the seasonal component via STL decomposition.
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
out = ts.TsFilSeasAdj().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [0.    0.247 0.479 0.682 0.841 0.949]
    length: 41


> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Cleveland, R. B. et al. (1990). STL: a seasonal-trend decomposition based on loess. J. Official Statistics, 6(1), 3-73.
