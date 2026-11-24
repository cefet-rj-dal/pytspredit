# Filter -- EMD (`TsFilEmd`)

**About the technique**

* `TsFilEmd` denoises via the native Empirical Mode Decomposition (removes the first IMF). Native numpy+scipy.
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
out = ts.TsFilEmd().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [0. 0. 0. 0. 0. 0.]
    length: 41


> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Huang, N. E. et al. (1998). The empirical mode decomposition... Proc. Royal Society A, 454, 903-995.
