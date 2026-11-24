# Filter -- Gabor (`TsFilGabor`)

**About the technique**

* `TsFilGabor` convolves with a Gabor kernel (Gaussian-modulated band-pass). Native kernel.
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
out = ts.TsFilGabor().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [-0.713 -0.495  0.053  0.696  1.222  1.541]
    length: 41


> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Gabor, D. (1946). Theory of communication. Journal of the IEE, 93(26), 429-457.
