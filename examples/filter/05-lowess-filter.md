# Filter -- LOWESS (`TsFilLowess`)

**About the technique**

* `TsFilLowess` locally weighted regression smoothing (robust local polynomial fit).
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
out = ts.TsFilLowess().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [0.042 0.244 0.441 0.634 0.782 0.881]
    length: 41


> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots. JASA, 74(368), 829-836.
