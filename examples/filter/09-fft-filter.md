# Filter -- FFT low-pass (`TsFilFft`)

**About the technique**

* `TsFilFft` keeps the low-frequency Fourier components (spectral denoising).
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





![png](fig/09-fft-filter/output_3_0.png)






![png](fig/09-fft-filter/output_3_1.png)




```python
out = ts.TsFilFft().fit(y).transform(y)
print('first values:', np.round(out[:6], 3))
print('length:', len(out))
```

    first values: [0.093 0.282 0.46  0.618 0.748 0.842]
    length: 41


Compare the original series with the filtered one:


```python
ts.plot_ts_pred(y=y, yadj=out)
```





![png](fig/09-fft-filter/output_6_0.png)






![png](fig/09-fft-filter/output_6_1.png)



> **What to observe.** The filtered series has the same length as the input; apply it as a preprocessing step before the sliding-window model.

**References:** Oppenheim, A. V. & Schafer, R. W. (2010). Discrete-Time Signal Processing, 3rd ed., Pearson.
