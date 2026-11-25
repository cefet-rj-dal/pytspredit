# Filter -- Moving average (`TsFilMa`)

**About the technique**

* `TsFilMa` smooths with a trailing moving average of the given window.
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





![png](fig/02-moving-average-filter/output_3_0.png)






![png](fig/02-moving-average-filter/output_3_1.png)




```python
out = ts.TsFilMa(3).fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [  nan   nan 0.256 0.483 0.672 0.822]
    length: 41


Compare the original series with the filtered one:


```python
ts.plot_ts_pred(y=y, yadj=out)
```





![png](fig/02-moving-average-filter/output_6_0.png)






![png](fig/02-moving-average-filter/output_6_1.png)



> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*, 3rd ed., OTexts.
