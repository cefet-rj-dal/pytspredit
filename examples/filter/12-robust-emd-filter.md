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

_rng = np.random.default_rng(0)
_base = ts.datasets.tsd()["y"].to_numpy()
y = _base + _rng.normal(0, _base.std(ddof=1) / 10, len(_base))  # add noise
_spike = _rng.normal(0, _base.std(ddof=1))
y[[9, 19, 29]] += _spike                                        # inject a few spikes
```

Visualize the raw series:


```python
ts.plot_ts(y=y)
```





![png](fig/12-robust-emd-filter/output_3_0.png)






![png](fig/12-robust-emd-filter/output_3_1.png)




```python
out = ts.TsFilRemd().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [0.302 0.302 0.337 0.414 0.534 0.703]
    length: 41


Compare the original series with the filtered one:


```python
ts.plot_ts_pred(y=y, yadj=out)
```





![png](fig/12-robust-emd-filter/output_6_0.png)






![png](fig/12-robust-emd-filter/output_6_1.png)



> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Wu, Z. & Huang, N. E. (2009). Ensemble EMD. Advances in Adaptive Data Analysis, 1(1), 1-41.
