# Filter -- Quadratic exponential smoothing (`TsFilQes`)

**About the technique**

* `TsFilQes` Holt-Winters-style exponential smoothing with trend.
* A filter is a `DalTransform` taking a 1-D series and returning a filtered series of the same length.

**Didactic goal:** see how the filter cleans the raw series before modeling.


```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytspredit as ts

y = ts.datasets.tsd()["y"].to_numpy()
```

Visualize the raw series:


```python
ts.plot_ts(y=y)
```





![png](fig/08-qes-filter/output_3_0.png)






![png](fig/08-qes-filter/output_3_1.png)




```python
out = ts.TsFilQes().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [  nan   nan 0.495 0.711 0.884 1.001]
    length: 41


Compare the original series with the filtered one:


```python
ts.plot_ts_pred(y=y, yadj=out)
```





![png](fig/08-qes-filter/output_6_0.png)






![png](fig/08-qes-filter/output_6_1.png)



> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Holt, C. C. (2004 [1957]); Winters, P. R. (1960). Forecasting sales by exponentially weighted moving averages. Management Science.
